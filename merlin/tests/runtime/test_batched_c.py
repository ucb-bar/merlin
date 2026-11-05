"""Dense batch proofs and generated C offsets, independent of a device backend."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from math import prod

import numpy as np
import pytest

from merlin.compile.batched_c import render_batched_contraction_c
from merlin.semantic_compiler.batched_contraction import plan_batched_contraction
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType


def source_graph(batch):
    rank = len(batch) + 3
    axes = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))
    prefix = axes[:-3]
    maps = tuple(
        IndexMap(rank, (*prefix, *rows), (0,) * (rank - 1))
        for rows in (
            (axes[-3], axes[-1]),
            (axes[-1], axes[-2]),
            (axes[-3], axes[-2]),
        )
    )
    types = [TensorType((*batch, *shape), "f32", "raw-bit-pattern") for shape in ((2, 3), (3, 4), (2, 4))]
    nodes = (
        SemanticNode("a", "input", (), types[0], effect="input"),
        SemanticNode("b", "input", (), types[1], effect="input"),
        SemanticNode("seed", "constant", (), types[2], effect="constant"),
        SemanticNode(
            "y",
            "source_f32_matmul_unqualified",
            ("a", "b", "seed"),
            types[2],
            (
                ("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                ("reduction_order", "unresolved"),
                ("scalar_add_order", "accumulator-product"),
            ),
            index_maps=(*maps, maps[-1]),
        ),
    )
    return KernelRequest(
        nodes,
        ("y",),
        ("external",),
        (("a", "external"), ("b", "external")),
        "synthetic",
        "diagnostic",
        constants=(ConstantBinding("seed", "f32-le-raw", "00000000" * (prod(batch) * 8)),),
    )


@pytest.mark.parametrize("batch", [(1,), (3,), (2, 3)])
def test_dense_batches_preserve_all_dimensions_and_independent_seeds(batch):
    graph = source_graph(batch)
    plan = plan_batched_contraction(KernelRequest.from_record(graph.record()))
    assert plan.batch_count == prod(batch)
    assert plan.batch_shape == batch and plan.matrix_elements == (6, 12, 8)
    assert [node.type.shape for node in plan.matrix_source.nodes] == [(2, 3), (3, 4), (2, 4), (2, 4)]
    assert plan.matrix_source.nodes[-1].attrs == graph.nodes[-1].attrs
    assert plan.record()["reduction_partition"] == "independent-complete-K-per-batch"


def test_changed_batch_maps_or_initialization_are_not_views():
    graph = source_graph((2, 3))
    result = graph.nodes[-1]
    wrong_map = replace(
        result.index_maps[0],
        coefficients=(
            result.index_maps[0].coefficients[1],
            result.index_maps[0].coefficients[0],
            *result.index_maps[0].coefficients[2:],
        ),
    )
    with pytest.raises(ValueError, match="maps mix batches"):
        plan_batched_contraction(
            replace(graph, nodes=(*graph.nodes[:-1], replace(result, index_maps=(wrong_map, *result.index_maps[1:]))))
        )
    with pytest.raises(ValueError, match="positive-zero"):
        plan_batched_contraction(replace(graph, constants=(replace(graph.constants[0], data_hex="00000080" * 48),)))
    plan = plan_batched_contraction(graph)
    with pytest.raises(ValueError, match="differs from its source"):
        render_batched_contraction_c(replace(plan, batch_shape=(6, 2)), "batch", "matrix")


def test_precision_records_complete_source_and_checks_every_constant_batch():
    from merlin.semantic_compiler.batched_contraction import lower_batched_contraction
    from merlin.tests.ir.test_contraction_precision import selected

    graph = source_graph((2, 3))
    _, profile = selected()
    graph = replace(graph, target_identity=profile.target_identity)
    precision = lower_batched_contraction(graph, profile=profile)
    values = np.arange(36, dtype=np.float32).reshape(2, 3, 2, 3) / 64
    converted = precision.inputs[0].convert_values(values)
    assert converted.shape == values.shape
    assert precision.record()["source_request_digest"] == graph.digest()
    assert precision.record()["batching"]["batch_count"] == 6
    invalid = values.copy()
    invalid[-1, -1, -1, -1] = np.nan
    with pytest.raises(ValueError):
        precision.inputs[0].convert_values(invalid)
    wrong = replace(graph.nodes[-1], attrs=((*graph.nodes[-1].attrs, ("reassociate", True))))
    with pytest.raises(ValueError, match="unsupported arithmetic"):
        lower_batched_contraction(replace(graph, nodes=(*graph.nodes[:-1], wrong)), profile=profile)


def test_generated_c_reuses_matrix_body_and_checks_future_batch_storage(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    plan = plan_batched_contraction(source_graph((2, 3)))
    code = (
        """#include <stddef.h>
#include <stdint.h>
#include <string.h>
static int calls,fail_on;
static int matrix(void *ctx,const void *const *inputs,void *const *outputs) {
  (void)ctx;
  ++calls; if (calls==fail_on) return 17;
  float a,b;
  memcpy(&a,inputs[0],4); memcpy(&b,inputs[1],4);
  for (unsigned i=0;i<8;++i) { float value=a+b+(float)i; memcpy((char *)outputs[0]+4*i,&value,4); }
  return 0;
}
"""
        + render_batched_contraction_c(plan, "batch", "matrix")
        + """
int run(void *a,void *b,void *out,void *reserved,size_t bytes,int failure) {
  calls=0; fail_on=failure; int context=1;
  const void *inputs[]={a,b},*ranges[]={reserved}; void *outputs[]={out};
  size_t sizes[]={bytes};
  return batch_batch_run(&context,inputs,outputs,ranges,sizes,reserved?1:0);
}
int count(void) { return calls; }
"""
    )
    (tmp_path / "batch.c").write_text(code)
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(tmp_path / "batch.c"),
            "-o",
            str(tmp_path / "batch.so"),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(tmp_path / "batch.so"))
    lib.run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t, ctypes.c_int]
    a = np.arange(36, dtype=np.float32).reshape(6, 6)
    b = (100 + np.arange(72, dtype=np.float32)).reshape(6, 12)
    output = ctypes.create_string_buffer(bytes([0x71]) * (6 * 8 * 4 + 32))
    ptr = ctypes.c_void_p(ctypes.addressof(output) + 16)
    assert lib.run(a.ctypes.data, b.ctypes.data, ptr, None, 0, 0) == 0 and lib.count() == 6
    actual = np.frombuffer(output.raw[16:-17], dtype=np.float32).reshape(6, 8)
    np.testing.assert_array_equal(actual, a[:, :1] + b[:, :1] + np.arange(8, dtype=np.float32))
    assert output.raw[:16] == output.raw[-17:-1] == bytes([0x71]) * 16
    before = output.raw
    # Reserved scratch intersects only the last batch: reject before batch zero.
    assert lib.run(a.ctypes.data, b.ctypes.data, ptr, a.ctypes.data + 5 * 6 * 4, 4, 0) == -1
    assert lib.count() == 0 and output.raw == before
    assert lib.run(a.ctypes.data, b.ctypes.data, ptr, None, 0, 2) == 17 and lib.count() == 2
    assert lib.run(a.ctypes.data, b.ctypes.data, a.ctypes.data, None, 0, 0) == -1 and lib.count() == 0
