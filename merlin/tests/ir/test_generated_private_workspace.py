"""A staged-Q1 workspace permits only its original qualified target operands."""
from dataclasses import replace
import sys

import pytest
from merlin.common.paths import repo_root
sys.path.insert(0,str(repo_root()/'merlin/tests/infra'))
from test_dense_loaded_sum_composition import selected
from merlin.llvmlower.generated_private_workspace import PrivateWorkspaceOperand,recognize_quantized_workspace
from merlin.llvmlower.fresh_tensor_writer import PrivateWorkspaceContract
from merlin.llvmlower.host_builder import PTR,FnBuilder
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp,IntegerAttr,i8,i64


def fixture():
    source,_=selected()
    workspace=PrivateWorkspaceContract(32,64,source.identity,True,True,True)
    def generated():
        f=FnBuilder([PTR]);private=f.add(llvm.AllocaOp(f.const(32),i8,alignment=64)).res
        pointer=f.add(llvm.GEPOp.from_mixed_indices(private,[f.const(0)],i8)).result
        address=f.add(llvm.PtrToIntOp(pointer,i64)).output
        write=f.add(llvm.InlineAsmOp('original_qualified_write','r,~{memory}',[address],has_side_effects=True))
        f.add(llvm.InlineAsmOp('original_complete','~{memory}',[],has_side_effects=True))
        read=f.add(llvm.InlineAsmOp('original_qualified_read','r,~{memory}',[address],has_side_effects=True))
        f.add(llvm.InlineAsmOp('original_complete','~{memory}',[],has_side_effects=True))
        module=ModuleOp([llvm.FuncOp('entry',llvm.LLVMFunctionType([PTR],llvm.LLVMVoidType()),body=f.finish())])
        return module,(write,read),private
    module,ops,private=generated();reference,*_=generated()
    operands=tuple(PrivateWorkspaceOperand(op,0,direction,'TEST-original-primitive-source')for op,direction in zip(ops,('write','read')))
    return module,reference,source,workspace,operands,private


def test_exact_independent_emission_and_original_workspace_source_are_live():
    module,reference,source,workspace,operands,_=fixture()
    bound=recognize_quantized_workspace(module,reference,source,workspace,entry_symbol='entry',target_operands=operands)
    bound.revalidate()
    assert bound.record()['private_bytes']==32
    assert bound.record()['target_pointer_uses']==2
    with pytest.raises(ValueError,match='source lifetime'):
        recognize_quantized_workspace(module,reference,source,replace(workspace,bytes=64),entry_symbol='entry',target_operands=operands)
    with pytest.raises(ValueError,match='complete borrowed effects'):
        recognize_quantized_workspace(module,reference,source,replace(workspace,initializes_before_read=False),entry_symbol='entry',target_operands=operands)
    source.source.rewrite.entries['producer']['acc_scale']=0.25
    with pytest.raises(ValueError):bound.revalidate()


@pytest.mark.parametrize('mutation',['host_store','escape','changed_reference','incomplete_roster','foreign_operand'])
def test_host_writes_escapes_and_changed_source_do_not_gain_permission(mutation):
    module,reference,source,workspace,operands,private=fixture()
    if mutation in ('host_store','escape'):
        # Change actual AND reference so equality alone cannot hide an escape.
        for m in (module,reference):
            f=next(o for o in m.body.block.ops if isinstance(o,llvm.FuncOp));a=next(o for o in f.walk()if isinstance(o,llvm.AllocaOp)).res
            if mutation=='host_store':
                v=llvm.ConstantOp(IntegerAttr(1,8),i8);f.body.block.insert_ops_before([v,llvm.StoreOp(v,a)],f.body.block.last_op)
            else:
                f.body.block.insert_op_before(llvm.CallOp('escape',a),f.body.block.last_op)
                m.body.block.add_op(llvm.FuncOp('escape',llvm.LLVMFunctionType([PTR],llvm.LLVMVoidType())))
    elif mutation=='changed_reference':
        op=next(o for o in reference.walk()if isinstance(o,llvm.AllocaOp));op.properties['alignment']=IntegerAttr(16,64)
    elif mutation=='incomplete_roster':operands=operands[:1]
    elif mutation=='foreign_operand':operands=(*operands,replace(operands[0],operand=1))
    with pytest.raises(ValueError):
        recognize_quantized_workspace(module,reference,source,workspace,entry_symbol='entry',target_operands=operands)
