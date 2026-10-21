"""Whole-row normalization and tail masks under synthetic primitive semantics."""

import ctypes
import shutil
import struct
import subprocess

import numpy as np
import pytest

from merlin.compile.c_tiling import render_row_softmax_c
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.semantic_compiler.rules import InstructionDescriptor
from merlin.semantic_compiler.tiled_softmax import (
    assemble_row_softmax_outputs,
    decompose_row_softmax,
    materialize_row_softmax_inputs,
)


@pytest.mark.parametrize("shape", [(3, 5), (1, 7), (5, 6)])
@pytest.mark.parametrize("masked", [False, True, "empty"])
def test_normalizes_across_all_column_tiles_and_trims_tails(shape, masked, tmp_path):
    source_type = TensorType(shape, "f32", "raw-bit-pattern")
    source = KernelRequest(
        (
            SemanticNode("scores", "input", (), source_type, effect="input"),
            SemanticNode("result", "softmax", ("scores",), TensorType(shape, "f32", "chunked-policy"), (("axis", 1),)),
        ),
        ("result",),
        ("external",),
        (("scores", "external"),),
        "synthetic",
        lowering_policy="diagnostic",
    )
    arities = {
        "row_max": 1,
        "combine_max": 2,
        "subtract": 2,
        "exp": 1,
        "row_sum": 1,
        "combine_sum": 2,
        "reciprocal": 1,
        "multiply": 2,
    }
    row_maps = (IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)), IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)))
    descriptors = {
        role: InstructionDescriptor(
            role,
            role,
            ("register",) * arity,
            "register",
            "f32",
            "raw-bit-pattern",
            (2,),
            required_attrs=(("input_domain", "finite"),),
            input_dtypes=("f32",) * arity,
            input_numerical_policies=("raw-bit-pattern",) * arity,
            input_ranks=(2,) * arity,
            index_maps=row_maps if role.startswith("row_") else pointwise_identity_maps(2, arity),
        )
        for role, arity in arities.items()
    }
    if masked:
        from dataclasses import replace

        x, result = source.nodes
        keep = SemanticNode("keep", "input", (), source_type, effect="input")
        source = replace(
            source,
            nodes=(x, keep, replace(result, op="masked_softmax", inputs=(x.id, keep.id))),
            input_storages=((x.id, "external"), (keep.id, "external")),
        )
    plan = decompose_row_softmax(
        source,
        tile_type=TensorType((2, 3), "f32", "raw-bit-pattern"),
        operations=descriptors,
        policy="chunked-policy",
        pad_element=struct.pack("<f", 0),
        empty_rows_zero=masked == "empty",
    )
    values = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) % 7) / 8
    keep = (np.indices(shape)[1] % 3 != 0).astype(np.float32)
    if plan.empty_rows_zero:
        keep[0] = 0
        assert plan.record()["all_masked_rows"] == "positive-zero"
    if masked:
        values[keep == 0] = 0
        if not plan.empty_rows_zero:
            with pytest.raises(ValueError, match="no kept"):
                materialize_row_softmax_inputs(plan, values.tobytes(), np.zeros(shape, dtype=np.float32).tobytes())
        malformed = keep.copy()
        malformed[-1, -1] = 0.5
        with pytest.raises(ValueError, match="zero or one"):
            materialize_row_softmax_inputs(plan, values.tobytes(), malformed.tobytes())
    inputs = materialize_row_softmax_inputs(plan, values.tobytes(), keep.tobytes() if masked else None)
    compiler = shutil.which("cc")
    assert compiler, "C compiler required for row packing and assembly"
    (tmp_path / "rows.c").write_text(
        render_row_softmax_c(plan, "rows")
        + """
void pack(uint8_t *out, const uint8_t *in, size_t row, size_t col) {
  rows_pack_row_tile(out,in,row,col);
}
void scatter(uint8_t *out, const uint8_t *in, size_t row, size_t col) {
  rows_scatter_row_tile(out,in,row,col);
}
"""
    )
    library_name = f"rows-{shape[0]}-{shape[1]}.so"
    subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "rows.c", "-o", library_name],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(tmp_path / library_name))
    library.pack.argtypes = library.scatter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    source_bytes = ctypes.create_string_buffer(values.tobytes())
    assembled_c = ctypes.create_string_buffer(values.nbytes)

    def primitive(node, operands):
        arrays = [np.array(value.elements, dtype=np.uint32).view(np.float32).reshape(2, 3) for value in operands]
        a = arrays[0]
        if node.op == "row_max":
            result = np.broadcast_to(a.max(axis=1, keepdims=True), a.shape)
        elif node.op == "row_sum":
            result = np.broadcast_to(a.sum(axis=1, keepdims=True), a.shape)
        elif node.op == "combine_max":
            result = np.maximum(a, arrays[1])
        elif node.op == "subtract":
            result = a - arrays[1]
        elif node.op == "exp":
            result = np.exp(a)
        elif node.op == "combine_sum":
            result = a + arrays[1]
        elif node.op == "reciprocal":
            result = 1 / a
        else:
            result = a * arrays[1]
        return TensorValue(node.type, tuple(int(code) for code in np.ascontiguousarray(result).view(np.uint32).ravel()))

    outputs = []
    for region, data in zip(plan.regions, inputs, strict=True):
        for column in range(plan.column_tiles):
            packed_c = ctypes.create_string_buffer(24)
            library.pack(packed_c, source_bytes, region.row_start, column)
            assert packed_c.raw == data[f"column_{column}"]
        nodes = {node.id: node for node in region.request.nodes}
        decoded = {name: TensorValue(nodes[name].type, struct.unpack("<6I", raw)) for name, raw in data.items()}
        constants = {
            binding.node_id: TensorValue(
                nodes[binding.node_id].type, struct.unpack("<6I", bytes.fromhex(binding.data_hex))
            )
            for binding in region.request.constants
        }
        result = evaluate_graph(
            region.request,
            decoded,
            constants=constants,
            primitives={(role, "raw-bit-pattern"): primitive for role in arities},
        )
        outputs.append(tuple(struct.pack("<6I", *value.elements) for value in result))
        for column, tile in enumerate(outputs[-1]):
            library.scatter(assembled_c, ctypes.create_string_buffer(tile), region.row_start, column)
        reciprocals = [node.id for node in region.request.nodes if node.op == "reciprocal"]
        assert len(reciprocals) == 1
        assert all(reciprocals[0] in nodes[root].inputs for root in region.request.outputs)
    actual = np.frombuffer(assemble_row_softmax_outputs(plan, outputs), dtype=np.float32).reshape(shape)
    assert assembled_c.raw == actual.tobytes()
    exponential = np.exp(values - values.max(axis=1, keepdims=True))
    if masked:
        exponential *= keep
    denominator = exponential.sum(axis=1, keepdims=True)
    expected = np.divide(exponential, denominator, out=np.zeros_like(exponential), where=denominator != 0)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(actual.sum(axis=1), (denominator[:, 0] != 0).astype(float), atol=1e-6)
    with pytest.raises(ValueError, match="omit or add"):
        assemble_row_softmax_outputs(plan, [group[:-1] for group in outputs])
