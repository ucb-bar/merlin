"""Printed block order does not establish completion order."""

import pytest

from merlin.targetgen import trace_check
from merlin.targetgen.rocc.control_flow import closing_fence_on_normal_returns
from merlin.targetgen.rocc.decode import _asm_template, _parse_module

FENCE = '"llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> ()'
WORK = (
    '"llvm.inline_asm"(%one, %one) <{asm_string = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1", '
    'constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()'
)
SOURCE = (
    """module {
  llvm.func @entry(%input: !llvm.ptr) {
    %one = llvm.mlir.constant(1 : i64) : i64
    %done = llvm.mlir.constant(true) : i1
    OPEN
    llvm.br ^loop
  ^exit:
    CLOSE
    llvm.return
  ^loop:
    WORK
    llvm.cond_br %done, ^exit, ^loop
  }
}""".replace("OPEN", FENCE)
    .replace("CLOSE", FENCE)
    .replace("WORK", WORK)
)


def _trace(module):
    return {
        "instructions": [
            {"class": "FENCE" if _asm_template(op) == "fence" else "MVIN"}
            for op in module.walk()
            if op.name == "llvm.inline_asm"
        ]
    }


def test_exit_before_loop_is_closed_on_every_returning_path():
    module = _parse_module(SOURCE)
    trace = _trace(module)
    assert trace["instructions"][-1]["class"] == "MVIN"
    assert trace_check.check(trace, {})["status"] == "fail"
    assert closing_fence_on_normal_returns(module) is True
    assert trace_check.check(trace, {}, lowered_module=module)["status"] == "pass"


@pytest.mark.parametrize("mutation", ["missing_exit", "work_after_exit", "bypass", "unfenced_loop_return"])
def test_unfenced_returning_paths_are_rejected(mutation):
    source = SOURCE
    if mutation == "missing_exit":
        source = source.replace("^exit:\n    " + FENCE, "^exit:")
    elif mutation == "work_after_exit":
        source = source.replace("^exit:\n    " + FENCE, "^exit:\n    " + FENCE + "\n" + WORK)
    elif mutation == "bypass":
        source = source.replace("llvm.br ^loop", "llvm.cond_br %done, ^loop, ^bypass")
        source = source.replace("  ^loop:", "  ^bypass:\n    " + WORK + "\n llvm.return\n  ^loop:")
    else:
        source = source.replace("llvm.cond_br %done, ^exit, ^loop", "llvm.return")
    module = _parse_module(source)
    assert closing_fence_on_normal_returns(module) is False
    assert trace_check.check(_trace(module), {}, lowered_module=module)["status"] == "fail"


def test_fence_with_no_side_effect_or_memory_clobber_is_not_proof():
    module = _parse_module(SOURCE.replace(FENCE, FENCE.replace(", has_side_effects", "")))
    assert closing_fence_on_normal_returns(module) is False


def test_opaque_call_remains_unknown():
    module = _parse_module(SOURCE.replace(WORK, "llvm.call @helper() : () -> ()\n" + WORK))
    assert closing_fence_on_normal_returns(module) is None
