"""Ordered unrolling preserves actual state/effects and bounded compiler space."""

from dataclasses import replace

import pytest
from xdsl.dialects.builtin import UnregisteredOp
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.scf import ScfFunctions
from xdsl.irdl import IRDLOperation, irdl_op_definition, var_operand_def
from xdsl.rewriter import Rewriter

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.unrolling import unroll_ordered_loops


@irdl_op_definition
class _ObserveOp(IRDLOperation):
    name = "fresh.observe"
    arguments = var_operand_def()


def _source(lower, upper, step, *, observe=True):
    event = '"fresh.observe"(%acc, %i) : (i8, index) -> ()' if observe else ""
    module = parse_ir(f"""builtin.module {{
      func.func @kernel(%x: i8) -> i8 {{
        %l = arith.constant {lower} : index
        %u = arith.constant {upper} : index
        %s = arith.constant {step} : index
        %three = arith.constant 3 : i8
        %result = scf.for %i = %l to %u step %s iter_args(%acc = %x) -> (i8) {{
          {event}
          %iv = arith.index_cast %i : index to i8
          %p = arith.muli %acc, %three : i8
          %r = arith.addi %p, %iv : i8
          scf.yield %r : i8
        }}
        func.return %result : i8
      }}
    }}""")
    for operation in tuple(module.walk()):
        if isinstance(operation, UnregisteredOp):
            Rewriter.replace_op(operation, _ObserveOp.build(operands=[operation.operands]))
    return module


@register_impls
class _Events(InterpreterFunctions):
    def __init__(self):
        self.calls = []

    @impl(_ObserveOp)
    def observe(self, interpreter, operation, arguments):
        self.calls.append(arguments)
        return ()


def _execute(module, initial):
    interpreter = Interpreter(module)
    events = _Events()
    for functions in (ArithFunctions(), FuncFunctions(), ScfFunctions(), events):
        interpreter.register_implementations(functions)
    result = interpreter.call_op("kernel", (initial,))
    return result, events.calls


@pytest.mark.parametrize("lower,upper,step,factor", [(0, 0, 1, 4), (2, 5, 2, 7), (3, 31, 4, 3), (1, 26, 2, 4)])
def test_actual_results_and_unknown_body_effect_order(lower, upper, step, factor):
    source = _source(lower, upper, step)
    target = source.clone()
    witnesses = unroll_ordered_loops(target, (factor,), index_bits=64)
    assert witnesses and witnesses[0].iterations == len(range(lower, upper, step))
    for initial in (-128, -3, 0, 17, 127):
        actual, calls = _execute(source, initial)
        transformed, changed_calls = _execute(target, initial)
        assert transformed == actual and changed_calls == calls
        assert [entry[1] for entry in calls] == list(range(lower, upper, step))
        expected = initial
        for iv in range(lower, upper, step):
            expected = ((expected * 3 + iv + 128) % 256) - 128
        assert actual == (expected,)


def test_huge_loop_emission_is_bounded_by_factor():
    source = _source(0, 10**12 + 3, 1, observe=False)
    before = ir_text(source)
    witnesses = unroll_ordered_loops(source, (7,), index_bits=64)
    assert witnesses[0].iterations == 10**12 + 3
    assert len(ir_text(source)) < 10_000
    assert ir_text(source) != before


def test_explicit_overflow_and_growth_budgets_refuse():
    with pytest.raises(ValueError, match="index width"):
        unroll_ordered_loops(_source(0, 120, 1), (16,), index_bits=8)
    with pytest.raises(ValueError, match="growth budget"):
        unroll_ordered_loops(_source(0, 1000, 1), (100,), index_bits=64, max_added_operations=32)


def test_dynamic_bounds_and_unknown_loop_metadata_are_unchanged():
    source = _source(0, 20, 1)
    loop = next(operation for operation in source.walk() if operation.name == "scf.for")
    from xdsl.dialects.builtin import StringAttr

    loop.attributes["fresh.policy"] = StringAttr("unknown")
    before = ir_text(source)
    assert not unroll_ordered_loops(source, (4,), index_bits=64)
    assert ir_text(source) == before


def test_witness_geometry_tracks_every_actual_iteration():
    source = _source(3, 31, 4)
    (witness,) = unroll_ordered_loops(source, (3,), index_bits=64)
    assert witness.full_groups * witness.factor + witness.remainder == witness.iterations
    assert 0 <= witness.remainder < witness.factor
    wrong = replace(witness, remainder=witness.remainder + 1)
    assert wrong != witness
