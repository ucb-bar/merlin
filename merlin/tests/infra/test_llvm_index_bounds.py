"""Actual counted SSA indices supply conservative control-specialization bounds."""

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i64

from merlin.compile.scheduling.llvm_index_bounds import FnBuilderBoundedControlAdapter
from merlin.compile.scheduling.scf_commands import emit_scf_commands, materialize_scf_commands
from merlin.llvmlower.host_builder import FnBuilder
from merlin.sched.ir import Call, Kernel, Loop, Var, evaluate, instances


def finish(builder, inputs=()):
    module = ModuleOp([
        llvm.FuncOp("indices", llvm.LLVMFunctionType(list(inputs), llvm.LLVMVoidType()), body=builder.finish())
    ])
    module.verify()
    return module


def test_actual_scf_indices_bound_every_original_instance_and_emit_real_llvm():
    kernel = Kernel("columns", (), (
        Loop("i", 3, (Loop("j", 2, (Call("read", (("column", Var("i") * 64 + Var("j") * 16),)),)),)),
    ))
    builder = FnBuilder([])
    control = FnBuilderBoundedControlAdapter(builder)
    observed = []
    emit_scf_commands(materialize_scf_commands(kernel), control,
                      lambda instruction, args, metadata: observed.append(control.bounds(args["column"])))
    assert observed == [(0, 144)]
    concrete = [evaluate(call.arg("column"), env) for call, env in instances(kernel)]
    assert min(concrete) == observed[0][0] and max(concrete) == observed[0][1]
    assert all(observed[0][0] <= value <= observed[0][1] for value in concrete)
    module = finish(builder)
    assert any(type(op) is llvm.MulOp for op in module.walk())
    assert any(type(op) is llvm.CondBrOp for op in module.walk())


def test_unknown_payload_and_overflow_never_gain_a_control_range():
    builder = FnBuilder([i64])
    control = FnBuilderBoundedControlAdapter(builder)
    unknown = builder.entry.args[0]
    assert control.bounds(control.op("add", unknown, 16)) is None
    assert control.bounds(control.op("add", (1 << 63) - 1, 1)) is None
    assert control.bounds(control.op("mul", -(1 << 63), -1)) is None
    predicate = control.cmp("eq", unknown, 0)
    assert control.bounds(control.select(predicate, -3, 7)) == (-3, 7)
    assert control.bounds(control.select(predicate, unknown, 7)) is None
    assert control.bounds(control.op("and", 3, 1)) is None
    finish(builder, [i64])


def test_actual_step_negative_coefficients_and_empty_loops_remain_conservative():
    builder = FnBuilder([])
    control = FnBuilderBoundedControlAdapter(builder)
    ranges = []
    control.loop(2, 10, 3, lambda index: ranges.append(control.bounds(control.op("mul", index, -2))))
    control.loop(0, 0, 1, lambda index: ranges.append(control.bounds(index)))
    assert ranges == [(-16, -4)]  # The real builder never emits a zero-trip body.
    with pytest.raises(ValueError, match="bounded ascending"):
        control.loop(0, 4, 0, lambda index: None)
    finish(builder)
