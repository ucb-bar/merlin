"""Read actual final LLVM metadata/control while retaining symbolic ABI pointers."""

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import i64

from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.llvmlower.host_builder import PTR, FnBuilder
from merlin.perf.finite_pointer_trace import ArgumentAddress, project_finite_pointer_trace


def emitted():
    b = FnBuilder([PTR, PTR])
    index = FnBuilderControlAdapter(b)
    bases = [b.add(llvm.PtrToIntOp(arg, i64)).results[0] for arg in b.entry.args]

    def body(i):
        selected = index.select(index.cmp("slt", i, 2), bases[0], bases[1])
        offset = index.op("mul", i, 4)
        pointer = index.op("add", selected, offset)
        b.add(llvm.InlineAsmOp("actual_view $0", "r", [pointer], has_side_effects=True))

    index.loop(0, 4, 1, body)
    return llvm.FuncOp("actual", llvm.LLVMFunctionType([PTR, PTR], llvm.LLVMVoidType()), body=b.finish())


def observe(op, args):
    return args[0] if op.name == "llvm.inline_asm" else None


def test_final_pointer_select_cfg_and_byte_offsets_are_observed_without_guessed_bases():
    trace = project_finite_pointer_trace(emitted(), observe=observe, pointer_bits=64)
    assert trace == (ArgumentAddress(0, 0), ArgumentAddress(0, 4), ArgumentAddress(1, 8), ArgumentAddress(1, 12))


def test_bounded_failure_does_not_return_an_accepted_partial_trace():
    with pytest.raises(ValueError, match="budget"):
        project_finite_pointer_trace(emitted(), observe=observe, pointer_bits=64, max_operations=5)
    with pytest.raises(ValueError, match="pointer integer width"):
        project_finite_pointer_trace(emitted(), observe=observe, pointer_bits=32)


def test_unknown_tensor_control_refuses_instead_of_selecting_a_member():
    b = FnBuilder([PTR])
    index = FnBuilderControlAdapter(b)
    value = b.add(llvm.LoadOp(b.entry.args[0], i64)).results[0]
    index.branch(index.cmp("eq", value, 0), lambda: None, lambda: None)
    f = llvm.FuncOp("opaque", llvm.LLVMFunctionType([PTR], llvm.LLVMVoidType()), body=b.finish())
    with pytest.raises(ValueError, match="tensor data"):
        project_finite_pointer_trace(f, observe=observe, pointer_bits=64)
