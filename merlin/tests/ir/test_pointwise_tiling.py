"""Independent affine packing checks, including broadcast and reversed views."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.c_pointwise import render_pointwise_c
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.pointwise_tiling import (
    PointwiseTiling,
    assemble_pointwise_output,
    materialize_pointwise_input,
)


def plan_for(kind):
    shape = {
        "vector": (5,),
        "channel_bias": (2, 3, 4, 5),
        "rank3_reverse": (2, 3, 5),
        "rank4_transpose": (2, 3, 4, 5),
    }.get(kind, (3, 5))
    other_shape, mapping = {
        "dense": (shape, IndexMap(2, ((1, 0), (0, 1)), (0, 0))),
        "nonzero_pad": (shape, IndexMap(2, ((1, 0), (0, 1)), (0, 0))),
        "broadcast": ((5,), IndexMap(2, ((0, 1),), (0,))),
        "column_broadcast": ((3,), IndexMap(2, ((1, 0),), (0,))),
        "scalar": ((1,), IndexMap(2, ((0, 0),), (0,))),
        "transpose": ((5, 3), IndexMap(2, ((0, 1), (1, 0)), (0, 0))),
        "reverse": (shape, IndexMap(2, ((1, 0), (0, -1)), (0, 4))),
        "vector": (shape, IndexMap(1, ((1,),), (0,))),
        "channel_bias": ((3,), IndexMap(4, ((0, 1, 0, 0),), (0,))),
        "rank3_reverse": (shape, IndexMap(3, ((1, 0, 0), (0, 1, 0), (0, 0, -1)), (0, 0, 4))),
        "rank4_transpose": (
            (4, 2, 5, 3),
            IndexMap(4, ((0, 0, 1, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0)), (0, 0, 0, 0)),
        ),
    }[kind]
    a = SemanticNode("a", "input", (), TensorType(shape, "bf16", "raw-bit-pattern"), effect="input")
    b = SemanticNode("b", "input", (), TensorType(other_shape, "bf16", "raw-bit-pattern"), effect="input")
    maps = (pointwise_identity_maps(len(shape), 1)[0], mapping, pointwise_identity_maps(len(shape), 1)[-1])
    r = SemanticNode("out", "pointwise", ("a", "b"), a.type, index_maps=maps)
    return PointwiseTiling(
        KernelRequest((a, b, r), (r.id,), ("external",), ((a.id, "external"), (b.id, "external")), "synthetic"),
        (2, 3),
        2,
        b"\0\0",
        (b"\0\0", b"\x80\x3f") if kind == "nonzero_pad" else (),
    )


@pytest.mark.parametrize(
    "kind",
    [
        "dense",
        "nonzero_pad",
        "broadcast",
        "column_broadcast",
        "scalar",
        "transpose",
        "reverse",
        "vector",
        "channel_bias",
        "rank3_reverse",
        "rank4_transpose",
    ],
)
@pytest.mark.parametrize("compact", [False, True])
def test_python_and_C_packers_follow_source_maps_and_initialize_tails(tmp_path, kind, compact):
    plan = replace(plan_for(kind), compact_broadcasts=compact)
    values = (
        np.arange(np.prod(plan.source.nodes[1].type.shape), dtype="<u2").reshape(plan.source.nodes[1].type.shape) + 1
    )
    expanded = {
        "dense": lambda: values,
        "nonzero_pad": lambda: values,
        "broadcast": lambda: np.broadcast_to(values, (3, 5)),
        "column_broadcast": lambda: np.broadcast_to(values[:, None], (3, 5)),
        "scalar": lambda: np.broadcast_to(values[0], (3, 5)),
        "transpose": lambda: values.T,
        "reverse": lambda: values[:, ::-1],
        "vector": lambda: values,
        "channel_bias": lambda: np.broadcast_to(values[None, :, None, None], plan.logical_shape),
        "rank3_reverse": lambda: values[:, :, ::-1],
        "rank4_transpose": lambda: values.transpose(1, 3, 0, 2),
    }[kind]()
    matrix = expanded.reshape(plan.shape)
    stored_rows = min(2, plan.shape[0]) if compact and kind in {"broadcast", "scalar"} else plan.shape[0]
    stored_cols = (min(3, plan.shape[1]) if compact and kind in {"column_broadcast", "scalar", "channel_bias"}
                   else plan.shape[1])
    tiles = []
    for row in range(0, stored_rows, 2):
        for col in range(0, stored_cols, 3):
            tile = np.full((2, 3), 0x3F80 if kind == "nonzero_pad" else 0, dtype="<u2")
            part = matrix[row : row + 2, col : col + 3]
            tile[: part.shape[0], : part.shape[1]] = part
            tiles.append(tile.tobytes())
    expected = b"".join(tiles)
    assert materialize_pointwise_input(plan, 1, values.tobytes()) == expected
    assert plan.input_packed_bytes[1] == len(expected)
    # Rebuild only for this test's output assembler: real consumers use tile
    # strides and never store the repeated copies. NumPy supplies the values.
    repeated = b"".join(tiles[
        (row // 2 if stored_rows == plan.shape[0] else 0) * ((stored_cols + 2) // 3)
        + (col // 3 if stored_cols == plan.shape[1] else 0)
    ] for row in range(0, plan.shape[0], 2) for col in range(0, plan.shape[1], 3))
    assert assemble_pointwise_output(plan, repeated) == expanded.astype("<u2").tobytes()
    compiler = shutil.which("cc")
    assert compiler
    (tmp_path / "pack.c").write_text(
        render_pointwise_c(plan, "tile") + "\nvoid run(uint8_t *out,const uint8_t *in) { tile_pack_1(out,in); }\n"
        "void unpack(uint8_t *out,const uint8_t *in) { tile_assemble(out,in); }\n"
        "void keep(uint8_t *out,const uint8_t *in) { tile_pack_0(out,in); }\n"
    )
    path = tmp_path / f"{kind}-compact-{int(compact)}.so"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(tmp_path / "pack.c"),
            "-o",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(path))
    library.run.argtypes = [ctypes.c_void_p] * 2
    library.unpack.argtypes = [ctypes.c_void_p] * 2
    output = ctypes.create_string_buffer(bytes([0xA5]) * (len(expected) + 32))
    library.run(ctypes.c_void_p(ctypes.addressof(output) + 16), values.ctypes.data)
    assert output.raw[16 : 16 + len(expected)] == expected
    assert output.raw[:16] == bytes([0xA5]) * 16 and output.raw[-17:-1] == bytes([0xA5]) * 16
    logical = ctypes.create_string_buffer(expanded.nbytes)
    library.unpack(logical, repeated)
    assert logical.raw == expanded.astype("<u2").tobytes()
    assert all(node.type.shape == (2, 3) for node in plan.body_request().nodes)
    assert plan.source.nodes[-1].type.shape == expanded.shape
    assert plan.record()["schema"] == ("merlin.pointwise_tiling.v3" if compact else
                                       "merlin.pointwise_tiling.v1" if expanded.ndim == 2 else
                                       "merlin.pointwise_tiling.v2")


@pytest.mark.parametrize(
    "fault", ["read_bounds", "output_map", "missing_input", "padding", "operand_pads", "compaction"]
)
def test_invalid_pointwise_maps_and_boundaries_are_rejected(fault):
    plan = plan_for("dense")
    source = plan.source
    output = source.nodes[-1]
    if fault == "read_bounds":
        output = replace(output, index_maps=(IndexMap(2, ((1, 0), (0, 1)), (0, 1)), *output.index_maps[1:]))
    elif fault == "output_map":
        output = replace(output, index_maps=(*output.index_maps[:2], IndexMap(2, ((0, 1), (1, 0)), (0, 0))))
    elif fault == "missing_input":
        output = replace(output, inputs=("a",), index_maps=(output.index_maps[0], output.index_maps[-1]))
    with pytest.raises(ValueError):
        if fault != "padding":
            source = replace(source, nodes=(*source.nodes[:-1], output))
        PointwiseTiling(
            source, (2, 3), 2, b"\0" if fault == "padding" else b"\0\0", (b"\0\0",) if fault == "operand_pads" else (),
            compact_broadcasts="yes" if fault == "compaction" else False,
        )


def test_shared_operand_keeps_one_input_and_rejects_distinct_views():
    original = plan_for("dense")
    a, _, output = original.source.nodes
    output = replace(output, inputs=(a.id, a.id))
    source = replace(original.source, nodes=(a, output), input_storages=((a.id, "external"),))
    shared = replace(original, source=source)
    assert len(shared.body_request().nodes) == 2
    assert shared.body_request().nodes[-1].inputs == (a.id, a.id)
    assert shared.body_request().nodes[-1].index_maps == pointwise_identity_maps(2, 2)
    values = bytes(range(30))
    assert materialize_pointwise_input(shared, 0, values) == materialize_pointwise_input(original, 0, values)
    with pytest.raises(ValueError, match="identical physical view"):
        replace(shared, source=replace(source, nodes=(a, replace(output, index_maps=(
            output.index_maps[0], IndexMap(2, ((1, 0), (0, -1)), (0, 4)), output.index_maps[-1])))))
