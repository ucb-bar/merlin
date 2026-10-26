"""Nested runtime compute epochs retain real source data and store completion."""
from dataclasses import replace

import pytest
from test_convolution_column_retention import law as retention_law
from test_convolution_exploration import issue_law, record
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionExplorationSource,
    ConvolutionScheduleChoice,
    convolution_pass_space,
    emit_convolution_schedule,
    explore_convolution_schedules,
    plan_convolution_schedule,
)
from merlin.compile.scheduling.window_transfer import (
    ConvolutionComputeEpochLaw,
    CountedConvolutionPlan,
    Window2D,
)
from merlin.transforms.pass_exploration import EmittedProgram
from merlin.transforms.pipeline_state import PipelineContext


def epoch_law():
    return ConvolutionComputeEpochLaw("neutral", "source-i32", "original-selected-queue",
        "separately-checked-cross-queue-dependencies", 4,
        local_read_after_write=True, local_write_after_read=True, local_write_after_write=True,
        accumulator_result_visibility=True, operand_consumption_retirement=True,
        ordered_configuration_capture=True, adjacent_descriptor_compute=True,
        finite_queue_pair_progress=True, complete_device_drain=True, exclusive_private_state=True)


def source(kernel=(3, 2)):
    # CI5/tile3 is deliberately not divisible; reduction_row//tile would give
    # wrong epoch boundaries after the first kernel position.
    p = CountedConvolutionPlan(Window2D((2, 3, 5, 5), kernel, padding=(1, 1, 1, 0)),
        9, 4, 4, 4, 3, 32, 12, 0, 8, 0, True)
    return ConvolutionExplorationSource(p, ((0, 4), (4, 4), (8, 1)), "original-tail-channel-window")


def plan(pairs, *, kernel=(3, 2), **changes):
    return plan_convolution_schedule(source(kernel), ConvolutionScheduleChoice(3, True, pairs),
        retention_law=retention_law(), issue_law=issue_law(),
        epoch_law=replace(epoch_law(), **changes))


@pytest.mark.parametrize("pairs", [2, 4])
@pytest.mark.parametrize("kernel", [(3, 2), (3, 1)])
def test_dynamic_nested_boundaries_preserve_exact_data_and_final_store_completion(pairs, kernel):
    original, actual, pending, drains = [], [], [], []
    emit_convolution_schedule(plan(1, kernel=kernel), Integers(), lambda k, a: original.append((k, dict(a))),
        complete=lambda *_: None)
    since_store = 0

    def issue(kind, fields):
        nonlocal since_store
        actual.append((kind, dict(fields)))
        # Finite queue acceptance makes progress, never an unbounded host batch.
        if len(pending) == 2:
            pending.pop(0)
        pending.append(kind)
        since_store += kind == "compute"

    def complete(kind, fields):
        nonlocal since_store
        assert pending[-1] == kind
        if kind == "compute":
            assert since_store % pairs == 0
        else:
            assert kind == "store"
            # 6 kernel positions * ceil(5/3) channel chunks * 3 columns.
            assert since_store in (kernel[0]*kernel[1]*2*3, 0)
            since_store = 0
        drains.append(kind)
        pending.clear()

    emit_convolution_schedule(plan(pairs, kernel=kernel), Integers(), issue, complete=complete)
    assert original == actual and not pending
    computes = sum(k == "compute" for k, _ in actual)
    stores = sum(k == "store" for k, _ in actual)
    block_computes = kernel[0]*kernel[1]*2*3
    assert drains.count("compute") == computes // block_computes * (block_computes // pairs)
    assert drains.count("store") == stores


@pytest.mark.parametrize("field", ["local_write_after_read", "accumulator_result_visibility",
    "operand_consumption_retirement", "adjacent_descriptor_compute", "finite_queue_pair_progress"])
def test_stronger_epoch_premises_are_independent_and_checked_before_emission(field):
    with pytest.raises(ValueError, match="dependency/progress premises"):
        plan(2, **{field: False})
    with pytest.raises(ValueError, match="dependency/progress premises"):
        plan(5)
    with pytest.raises(ValueError, match="stronger original source law"):
        plan_convolution_schedule(source(), ConvolutionScheduleChoice(3, True, 2),
            retention_law=retention_law(), issue_law=issue_law())


def test_epoch_registry_is_optional_and_realized_programs_are_bound_and_deduplicated():
    s = source()
    space = convolution_pass_space(s, verify_source=lambda _: None,
        realize=lambda p: EmittedProgram(record(p), "actual-effect-stream", p.identity),
        retention_law=retention_law(), issue_law=issue_law(), epoch_law=epoch_law(),
        live_tile_limits=(3,), issue_modes=(True,), compute_pair_limits=(1, 2, 4), max_candidates=3)
    results = list(explore_convolution_schedules(s, space, context=PipelineContext()))
    assert len(results) == 3 and all(r.evaluation.status == "compiled" for r in results)
    assert all(r.duplicate_of is None for r in results)
    assert all(len(r.evaluation.result.receipt["passes"]) == 5 for r in results)
    schedule = plan(2)
    object.__setattr__(schedule.epoch_law, "source_identity", "changed")
    with pytest.raises(ValueError, match="changed after planning"):
        schedule.verify_current()


def test_epoch_fence_is_inside_actual_counted_llvm_control():
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import BlockArgument

    from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
    from merlin.llvmlower.host_builder import FnBuilder

    fn = FnBuilder([])
    emit_convolution_schedule(plan(4), FnBuilderControlAdapter(fn), lambda *_: None,
        complete=lambda *_: fn.add(llvm.InlineAsmOp("fence", "~{memory}", [], has_side_effects=True)))
    module = ModuleOp([llvm.FuncOp("kernel", llvm.LLVMFunctionType([], llvm.LLVMVoidType()),
        linkage=llvm.LinkageAttr("external"), body=fn.finish())])
    module.verify()
    ops = list(module.walk())
    # The actual emitted program retains induction SSA, division/modulo and
    # conditional CFG edges; no Python template counter selects a constant arm.
    assert any(isinstance(op, llvm.MulOp) and any(isinstance(a, BlockArgument)
        for a in op.operands) for op in ops)
    assert any(isinstance(op, llvm.URemOp) for op in ops)
    assert any(isinstance(op, llvm.CondBrOp) for op in ops)
    assert any(isinstance(op, llvm.InlineAsmOp) for op in ops)
