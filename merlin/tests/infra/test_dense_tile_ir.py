"""Logical tile expansion keeps the existing symbolic dense proof boundary."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.compact_dense import (
    DenseLoopAdapter, DenseLoopStorage, DenseMemoryView, schedule_compact_dense,
)
from merlin.compile.scheduling.dense_tile_ir import encode_dense_tile, project_dense_tile
from merlin.sched.ir import Call, Const, instances
from merlin.sched.ir.kernel import concretize
from merlin.transforms.dense_planning import DenseProblem, DenseResources


def schedule(shape):
    problem = DenseProblem(*shape, 4, 4, 4, 1, 1, 2)
    storage = DenseLoopStorage(DenseMemoryView("lhs", "operands", 4, 0, 4),
                               DenseMemoryView("rhs", "operands", 4, 4, 4),
                               DenseMemoryView("acc", "sums", 8, 0, 4))
    return schedule_compact_dense(
        problem, DenseResources(32, 32, 1, 1, 1), storage,
        DenseLoopAdapter(encode_dense_tile, project_dense_tile, "test-config", "ordered-test", True, True, 64),
        name="floating", dtypes=("fp8_e4m3", "fp8_e4m3", "bf16"), static_budget=32,
        external_buffers_disjoint=True,
    )


@pytest.mark.parametrize("shape", [(4, 4, 4), (8, 12, 16)])
def test_actual_calls_retain_k_order_reset_and_all_outputs(shape):
    result = schedule(shape)
    assert result is not None
    result.revalidate()
    current, outputs = [], []
    for call, env in instances(result.kernel):
        assert encode_dense_tile(project_dense_tile(call)) == call
        args = concretize(call, env)
        if call.instr == "dense.preload":
            assert args["continue"] == bool(current)
        if call.instr == "dense.compute":
            current.append(args["k"])
        if call.instr == "dense.store":
            assert current == list(range(shape[1] // 4))
            outputs.append((args["row"], args["col"]))
            current = []
    assert outputs == [(m, n) for m in range(0, shape[0], 4) for n in range(0, shape[2], 4)]
    assert result.realization["machine_emission"] == "not_established"


def test_projection_and_checked_emission_reject_mutation():
    result = schedule((4, 4, 4))
    call = next(instances(result.kernel))[0]
    for bad in (replace(call, produces="async"), replace(call, args=call.args + (("extra", Const(0)),)),
                replace(call, instr="invented")):
        with pytest.raises(ValueError):
            project_dense_tile(bad)
    result = replace(result, kernel=replace(result.kernel, body=(Call("wrong", ()),)))
    with pytest.raises(ValueError, match="changed"):
        result.revalidate()


def test_checked_index_domain_and_report_cannot_reauthorize_changed_kernel():
    result = schedule((4, 4, 4))
    with pytest.raises(ValueError, match="changed"):
        replace(result, index_bits=32).revalidate()
    changed = replace(result, kernel=replace(result.kernel, body=(Call("wrong", ()),)))
    changed.realization["kernel_sha256"] = changed.kernel.digest()
    with pytest.raises(ValueError, match="changed"):
        changed.revalidate()


def test_tile_encoding_refuses_gathered_loads():
    result = schedule((4, 4, 4))
    load = project_dense_tile(next(instances(result.kernel))[0])
    with pytest.raises(ValueError):
        encode_dense_tile(replace(load, gather=(0, 4, 8, 12)))
