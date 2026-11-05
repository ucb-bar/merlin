"""Actual counted-loop emission retains tails, addresses and execution barriers."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.llvm_program import emit_index_program
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, select


class ExecutingBuilder:
    def loop(self, start, end, step, callback):
        for index in range(start, end, step):
            callback(index)

    def op(self, opcode, lhs, rhs):
        return lhs + rhs if opcode == "add" else lhs * rhs

    def cmp(self, pred, lhs, rhs):
        assert pred == "eq"
        return lhs == rhs

    def select(self, cond, lhs, rhs):
        return lhs if cond else rhs


def test_actual_loop_indices_and_tail_expressions_reach_the_instruction_callback():
    inner = Call("write", (("offset", Var("m") * 9 + Var("n") * 4),
                           ("width", select("n", 2, 1, 4))))
    kernel = Kernel("index_test", (), (Loop("m", 2, (Loop("n", 3, (inner,)),)),))
    observed = []
    report = emit_index_program(kernel, ExecutingBuilder(), lambda instr, args: observed.append((instr, args)))
    assert observed == [("write", {"offset": m * 9 + n * 4, "width": 1 if n == 2 else 4})
                        for m in range(2) for n in range(3)]
    assert report["kernel_sha256"] == kernel.digest()


@pytest.mark.parametrize("change", [{"unit": "copy"}, {"produces": "ready"},
                                     {"sets": (("mode", Const(1)),)}])
def test_unsupported_annotations_refuse_before_any_emission(change):
    call = replace(Call("copy", ()), **change)
    kernel = Kernel("annotated", (), (Call("first", ()), call))
    observed = []
    with pytest.raises(ValueError, match="annotations"):
        emit_index_program(kernel, ExecutingBuilder(), lambda *args: observed.append(args))
    assert observed == []
