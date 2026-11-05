"""Independent axis permutation, seed placement and reduction-tail packing."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from math import prod

import numpy as np
import pytest

from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.reduction_tiling import (
    ReductionTiling,
    assemble_reduction_output,
    materialize_reduction_inputs,
)


def plan_for(shape, axes):
    kept = tuple(i for i in range(len(shape)) if i not in axes)
    out_shape = tuple(shape[i] for i in kept)
    dtype = TensorType(shape, "bf16", "raw-bit-pattern")
    seed_type = TensorType(out_shape, "bf16", "raw-bit-pattern")
    projection = IndexMap(
        len(shape), tuple(tuple(int(i == axis) for i in range(len(shape))) for axis in kept), (0,) * len(kept)
    )
    data = SemanticNode("data", "input", (), dtype, effect="input")
    seed = SemanticNode("seed", "input", (), seed_type, effect="input")
    out = SemanticNode(
        "sum",
        "reduce_sum",
        (data.id, seed.id),
        replace(seed_type, numerical_policy="explicit-sum-test-policy"),
        index_maps=(pointwise_identity_maps(len(shape), 1)[0], projection, projection),
    )
    source = KernelRequest(
        (data, seed, out), (out.id,), ("external",), ((data.id, "external"), (seed.id, "external")), "synthetic"
    )
    return ReductionTiling(source, axes, (2, 3), 2, bytes(2))


def pack_reference(matrix, rows=2, columns=3):
    result = []
    for row in range(0, matrix.shape[0], rows):
        for col in range(0, matrix.shape[1], columns):
            tile = np.zeros((rows, columns), dtype="<u2")
            part = matrix[row : row + rows, col : col + columns]
            tile[: part.shape[0], : part.shape[1]] = part
            result.append(tile.tobytes())
    return b"".join(result)


@pytest.mark.parametrize(
    "shape,axes", [((3, 5), (1,)), ((2, 3, 5), (0, 2)), ((2, 3, 4, 5), (1, 3)), ((1, 5, 65), (2,))]
)
def test_reduction_packing_matches_numpy_and_preserves_nonzero_initializers(shape, axes, tmp_path):
    plan = plan_for(shape, axes)
    data = (np.arange(prod(shape), dtype="<u2") + 17).reshape(shape)
    seed = (np.arange(prod(plan.output_shape), dtype="<u2") + 901).reshape(plan.output_shape)
    actual_data, actual_seed = materialize_reduction_inputs(plan, data.tobytes(), seed.tobytes())
    permuted = data.transpose(plan.kept_axes + axes).reshape(plan.matrix_shape)
    assert actual_data == pack_reference(permuted)
    assert actual_seed == pack_reference(seed.reshape(-1, 1))
    matrix = np.full((seed.size, 3), 0xDEAD, dtype="<u2")
    matrix[:, 0] = seed.reshape(-1)
    assert assemble_reduction_output(plan, pack_reference(matrix)) == seed.tobytes()
    assert plan.source.nodes[0].type.shape == shape
    assert plan.record()["output_shape"] == list(seed.shape)
    from merlin.compile.c_reduction import render_reduction_c

    rendered = (
        render_reduction_c(plan, "region")
        + "\n"
        + """
    void pack(unsigned char *a,unsigned char *b,const unsigned char *x,const unsigned char *seed) {
      region_pack_0(a,x); region_pack_1(b,seed);
    }
    void assemble(unsigned char *out,const unsigned char *tiles) { region_assemble(out,tiles); }
    """
    )
    (tmp_path / "packing.c").write_text(rendered)
    # Pytest may reuse a passed case's temporary path. dlopen keeps libraries
    # loaded, so distinguish the actual generated code, not just its directory.
    library_path = tmp_path / ("packing_" + plan.source.digest() + ".so")
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(tmp_path / "packing.c"),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.pack.argtypes = [ctypes.c_void_p] * 4
    library.assemble.argtypes = [ctypes.c_void_p] * 2
    a = ctypes.create_string_buffer(plan.packed_sizes[0])
    b = ctypes.create_string_buffer(plan.packed_sizes[1])
    library.pack(a, b, data.tobytes(), seed.tobytes())
    assert a.raw == actual_data
    assert b.raw == actual_seed
    out = ctypes.create_string_buffer(seed.nbytes)
    library.assemble(out, pack_reference(matrix))
    assert out.raw == seed.tobytes()
    with pytest.raises(ValueError):
        materialize_reduction_inputs(plan, data.tobytes()[:-1], seed.tobytes())


@pytest.mark.parametrize("fault", ["axes", "seed_shape", "result_map", "padding"])
def test_invalid_reduction_transport_is_rejected(fault):
    plan = plan_for((2, 3, 5), (1, 2))
    source = plan.source
    if fault == "seed_shape":
        seed = replace(source.nodes[1], type=TensorType((3,), "bf16", "raw-bit-pattern"))
        source = replace(source, nodes=(source.nodes[0], seed, source.nodes[2]))
    elif fault == "result_map":
        out = source.nodes[-1]
        out = replace(out, index_maps=(*out.index_maps[:2], IndexMap(3, ((0, 1, 0),), (0,))))
        source = replace(source, nodes=(*source.nodes[:2], out))
    with pytest.raises(ValueError):
        ReductionTiling(
            source,
            (1, 1) if fault == "axes" else plan.axes,
            plan.tile_shape,
            2,
            b"\0" if fault == "padding" else bytes(2),
        )
