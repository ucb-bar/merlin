"""Row batching retains all four products and ordered floating observers."""

import re
import struct
from unittest.mock import patch

import numpy as np
import pytest
import test_shared_rhs_views as rhs_fixture
from test_shared_rhs_head_route import fixture, stored
from xdsl.dialects import builtin, func
from xdsl.dialects.linalg.ops import GenericOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.head_source_route import project_heads
from merlin.llvmlower.sibling_source_route import _text


def _generic_slice_fixture(*shape, **kwargs):
    """Use the generic slice spelling shared by supported xDSL versions.

    Keep the inherited fixture unchanged: only this test's input text passes
    through the adapter before parsing. xDSL 0.68 lacks its custom slice parser.
    """
    parser = rhs_fixture.parse_mlir_text

    def parse(text):
        pattern = (
            r"tensor\.extract_slice (%\w+)\[(\d+),0,0\]"
            r"\[1,(\d+),(\d+)\]\[1,1,1\]\s*"
            r":(tensor<[^>]+>) to (tensor<[^>]+>)"
        )

        def generic(match):
            root, head, rows, cols, source_type, result_type = match.groups()
            return (
                f'"tensor.extract_slice"({root}) <{{'
                f"static_offsets = array<i64: {head}, 0, 0>, "
                f"static_sizes = array<i64: 1, {rows}, {cols}>, "
                "static_strides = array<i64: 1, 1, 1>, "
                "operandSegmentSizes = array<i32: 1, 0, 0, 0>"
                f"}}> : ({source_type}) -> {result_type}"
            )

        converted, count = re.subn(pattern, generic, text)
        assert count == 2 * shape[0] * shape[1] * kwargs["planes"]
        return parser(converted)

    with patch.object(rhs_fixture, "parse_mlir_text", parse):
        return fixture(*shape, **kwargs)


def _with_ordered_observers(shape, *, scale="literal"):
    """Observe every raw product and its four-term f32 reconstruction."""
    g, r, m, _k, n = shape
    module, _groups, _evidence = _generic_slice_fixture(*shape, planes=4, scale=scale)
    entry = module.body.block.first_op
    block = entry.body.block
    old_return = block.last_op
    original_results = tuple(old_return.arguments)
    typ = f"tensor<{m}x{n}xi32>"
    floating = f"tensor<{m}x{n}xf32>"
    observer = parse_mlir_text(f"""module {{
      func.func @observer(%a0:{typ},%b0:{typ},%c0:{typ},%d0:{typ}) -> {floating} {{
        %empty=tensor.empty():{floating}
        %result=linalg.generic {{
          indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>,
            affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
          iterator_types=["parallel","parallel"]
        }} ins(%a0,%b0,%c0,%d0:{typ},{typ},{typ},{typ}) outs(%empty:{floating}) {{
          ^bb0(%a:i32,%b:i32,%c:i32,%d:i32,%unused:f32):
          %af=arith.sitofp %a:i32 to f32
          %bf=arith.sitofp %b:i32 to f32
          %cf=arith.sitofp %c:i32 to f32
          %df=arith.sitofp %d:i32 to f32
          %s0=arith.constant 0.125:f32
          %s1=arith.constant -0.0625:f32
          %s2=arith.constant 0.000001:f32
          %s3=arith.constant 0.33333334:f32
          %den=arith.constant 3.25:f32
          %t0=arith.mulf %af,%s0:f32
          %t1=arith.mulf %bf,%s1:f32
          %t2=arith.mulf %cf,%s2:f32
          %t3=arith.mulf %df,%s3:f32
          %u0=arith.addf %t0,%t1:f32
          %u1=arith.addf %u0,%t2:f32
          %u2=arith.addf %u1,%t3:f32
          %out=arith.divf %u2,%den:f32
          linalg.yield %out:f32
        }} -> {floating}
        return %result:{floating}
      }} }}""")
    template = observer.body.block.first_op
    outputs = []
    for head in range(g * r):
        mapper = dict(
            zip(template.body.block.args, (original_results[p * g * r + head] for p in range(4)), strict=True)
        )
        for operation in tuple(template.body.block.ops)[:-1]:
            cloned = operation.clone(value_mapper=mapper)
            block.insert_op_before(cloned, old_return)
        outputs.append(mapper[template.body.block.last_op.arguments[0]])
    values = (*original_results, *outputs)
    block.insert_op_before(func.ReturnOp(*values), old_return)
    block.erase_op(old_return)
    entry.properties["function_type"] = builtin.FunctionType.from_lists(
        entry.function_type.inputs, [value.type for value in values]
    )
    module.verify()
    return module


@pytest.mark.parametrize("shape", [(2, 3, 1, 5, 7), (3, 2, 2, 7, 3)])
def test_four_plane_row_route_keeps_raw_outputs_and_original_float_programs(shape):
    source = _with_ordered_observers(shape)
    before = _text(source)
    observers = [op for op in source.walk() if type(op) is GenericOp and op.results[0].type.element_type == builtin.f32]
    projection = project_heads(source, shared_rhs_row_views=True)
    g, r, m, k, n = shape
    assert len(projection.members) == 4
    assert sum(len(row["members"]) for row in projection.members) == 4 * g * r
    assert all(row["batch_extent"] == g and row["geometry_m_k_n"] == [r * m, k, n] for row in projection.members)
    projected_observers = [
        op
        for op in projection.module.walk()
        if type(op) is GenericOp and op.results[0].type.element_type == builtin.f32
    ]
    # Additional smaller quantization generics may precede the observers.
    assert len(projected_observers) >= len(observers)
    for original, projected in zip(observers[-g * r :], projected_observers[-g * r :], strict=True):
        assert original.body.is_structurally_equivalent(projected.body)
    assert source.body.block.first_op.function_type == projection.module.body.block.first_op.function_type
    assert _text(source) == before
    projection.batching.revalidate()


def test_native_four_plane_row_route_preserves_float_bits_and_output_guards(tmp_path):
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    g, r, m, k, n = shape = (2, 3, 1, 5, 7)
    source = _with_ordered_observers(shape)
    projected = project_heads(source, shared_rhs_row_views=True)
    rng = np.random.default_rng(401)
    lhs = [rng.integers(-63, 64, (g * r, m, k), dtype=np.int8) for _ in range(4)]
    rhs = (np.arange(g * k * n, dtype=np.float32).reshape(g, 1, k, n) - 17) * np.float32(1.5)
    scales = np.repeat(np.arange(2, 2 + g, dtype=np.float32), r).reshape(g * r, 1, 1)
    quantized = np.clip(np.rint(np.repeat(rhs, r, axis=1).reshape(g * r, k, n) / scales), -127, 127).astype(np.int8)
    raw = [a.astype(np.int32) @ quantized.astype(np.int32) for a in lhs]
    expected = [*(row for plane in raw for row in plane), quantized]
    terms = [
        value.astype(np.float32) * np.float32(scale)
        for value, scale in zip(raw, (0.125, -0.0625, 0.000001, 0.33333334), strict=True)
    ]
    reconstructed = (((terms[0] + terms[1]) + terms[2]) + terms[3]) / np.float32(3.25)
    expected.extend(reconstructed)
    actual_runs = []
    for label, module in (("original", source), ("projected", projected.module)):
        result = lower_model(_text(module), tmp_path / label)
        invoke = HostModel.load(
            str(result.host_so), image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve())
        )
        guarded = [np.full(value.size + 32, 91, dtype=value.dtype) for value in expected]
        outputs = [buffer[16:-16].reshape(value.shape) for buffer, value in zip(guarded, expected, strict=True)]
        invoke([(value.ctypes.data, value.shape) for value in (*lhs, rhs, *outputs)])
        for actual, reference, buffer in zip(outputs, expected, guarded, strict=True):
            np.testing.assert_array_equal(actual.view(np.uint8), reference.view(np.uint8))
            assert (buffer[:16] == 91).all() and (buffer[-16:] == 91).all()
        actual_runs.append(outputs)
    for original, optimized in zip(*actual_runs, strict=True):
        np.testing.assert_array_equal(original.view(np.uint8), optimized.view(np.uint8))


def test_four_plane_route_requires_fresh_exact_generated_scale_binding(tmp_path):
    source = _with_ordered_observers((2, 3, 1, 5, 7), scale="runtime")
    before = _text(source)
    with pytest.raises(ValueError, match="no checked repeated RHS"):
        project_heads(source, shared_rhs_row_views=True)
    stored(tmp_path, source, planes=4)
    projected = project_heads(source, shared_rhs_row_views=True, capture=tmp_path)
    assert len(projected.members) == 4
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-4] + struct.pack("<f", 4.0))
    with pytest.raises(ValueError, match="stored argument"):
        projected.batching.revalidate()
    with pytest.raises(ValueError, match="no checked repeated RHS"):
        project_heads(source, shared_rhs_row_views=True, capture=tmp_path)
    assert _text(source) == before
