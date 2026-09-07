"""Index folding must preserve integer semantics and refuse runtime input."""
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i8, i64
from xdsl.ir import Block

from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer


def const(value, ty=i64):
    return llvm.ConstantOp(IntegerAttr(value, ty), ty).results[0]


def test_iota_arithmetic_and_wrapped_integer_index():
    index = llvm.AddOp(llvm.MulOp(const(5), const(2)).results[0], const(1)).results[0]
    assert constant_integer(index) == 11
    narrowed = llvm.TruncOp(llvm.AddOp(const(127), const(1)).results[0], i8).results[0]
    assert constant_integer(llvm.SExtOp(narrowed, i64).results[0]) == -128
    assert constant_integer(llvm.ZExtOp(narrowed, i64).results[0]) == 128


def test_unknown_input_and_poison_shift_are_not_constant():
    arg = Block(arg_types=[i64]).args[0]
    assert constant_integer(llvm.AddOp(arg, const(1)).results[0]) is None
    assert constant_integer(llvm.ShlOp(const(1), const(64)).results[0]) is None
    ptr = Block(arg_types=[llvm.LLVMPointerType()]).args[0]
    assert constant_integer(llvm.LoadOp(ptr, i64).results[0]) is None
    flagged = llvm.AddOp(const(127, i8), const(1, i8), overflow=IntegerAttr(1, 32))
    assert constant_integer(flagged.results[0]) is None


def test_logical_and_arithmetic_right_shift_differ():
    assert constant_integer(llvm.LShrOp(const(-2), const(1)).results[0]) == 2**63 - 1
    assert constant_integer(llvm.AShrOp(const(-2), const(1)).results[0]) == -1
