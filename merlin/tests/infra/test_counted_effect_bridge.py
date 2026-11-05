"""The convenience bridge preserves raw annotation refusals and counted loops."""

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp

from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter, emit_index_effect_program
from merlin.llvmlower.host_builder import FnBuilder
from merlin.sched.ir import Call, Const, Kernel, Loop, Var


def emit(kernel):
    builder, calls = FnBuilder([]), []
    control = FnBuilderControlAdapter(builder)

    def lower(tag, fields):
        calls.append(tag)

    receipt = emit_index_effect_program(kernel, control, lower,
        complete_epoch=lambda: calls.append("completed"), completion_instruction="finish")
    module = ModuleOp([llvm.FuncOp("example", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())])
    module.verify()
    return calls, receipt, module


def test_billion_iterations_stay_counted_without_dynamic_expansion():
    kernel = Kernel("counted", (), (Loop("i", 10**9,
        (Call("issue", (("index", Var("i")),)), Call("finish", ()))),))
    calls, receipt, module = emit(kernel)
    assert calls == ["issue", "completed"]
    assert receipt["completion_mapping"]["emitted_static_drains"] == 1
    assert sum(1 for _ in module.walk()) < 50


@pytest.mark.parametrize("call", [Call("issue", (), unit="engine"),
                                 Call("issue", (), sets=(("mode", Const(1)),))])
def test_annotations_still_require_explicit_scf_lowering(call):
    with pytest.raises(ValueError, match="cannot erase execution or configuration annotations"):
        emit(Kernel("annotated", (), (call, Call("finish", ()))))


def test_full_completion_cannot_drop_operands_or_be_absent_callback():
    with pytest.raises(ValueError, match="cannot carry instruction operands"):
        emit(Kernel("bad", (), (Call("finish", (("unknown", Const(1)),)),)))
    with pytest.raises(ValueError, match="completion callback"):
        emit_index_effect_program(Kernel("bad", (), ()), None, lambda *_: None,
                                  complete_epoch=None, completion_instruction="finish")
