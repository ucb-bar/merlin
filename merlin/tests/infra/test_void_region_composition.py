"""Mechanically compose original CFGs while retaining effects and custody."""
import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp,i32
from merlin.llvmlower.host_builder import FnBuilder,PTR,append_void_region


def function():
    b=FnBuilder([PTR])
    b.for_loop(0,3,1,lambda i:b.add(llvm.StoreOp(b.const(7,i32),b.gep(b.entry.args[0],i,i32))),retain_loop=True)
    b.add(llvm.InlineAsmOp('fence','~{memory}',[],has_side_effects=True))
    return llvm.FuncOp('member',llvm.LLVMFunctionType([PTR],llvm.LLVMVoidType()),body=b.finish())


def test_multiple_counted_members_preserve_source_and_effects():
    original=function();before=str(original)
    b=FnBuilder([PTR,PTR]);receipts=[append_void_region(b,original,[arg])for arg in b.entry.args]
    result=ModuleOp([llvm.FuncOp('combined',llvm.LLVMFunctionType([PTR,PTR],llvm.LLVMVoidType()),body=b.finish())])
    result.verify()
    assert str(original)==before
    assert len([op for op in result.walk()if isinstance(op,llvm.StoreOp)])==2
    assert len([op for op in result.walk()if isinstance(op,llvm.InlineAsmOp)])==2
    assert not any(isinstance(op,llvm.CallOp)for op in result.walk())
    assert len([op for op in result.walk()if isinstance(op,llvm.ReturnOp)])==1
    assert all(row['calls_added']==0 for row in receipts)


def test_argument_type_and_open_continuation_guards():
    original=function();b=FnBuilder([i32])
    with pytest.raises(ValueError,match='types'):append_void_region(b,original,b.entry.args)
    b=FnBuilder([PTR]);b.finish()
    with pytest.raises(ValueError,match='open'):append_void_region(b,original,b.entry.args)


def test_foreign_capture_refuses_before_modifying_builder():
    foreign=FnBuilder([]);constant=foreign.const(1,i32)
    b=FnBuilder([PTR]);b.add(llvm.StoreOp(constant,b.entry.args[0]));bad=llvm.FuncOp('bad',llvm.LLVMFunctionType([PTR],llvm.LLVMVoidType()),body=b.finish())
    dest=FnBuilder([PTR]);before=str(dest.region)
    with pytest.raises(Exception):append_void_region(dest,bad,dest.entry.args)
    assert str(dest.region)==before
