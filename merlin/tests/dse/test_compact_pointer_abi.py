"""Generic compiler ABI mapping: preserve pointer identities and body operations."""
import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i32
from xdsl.ir import Block, Region

from merlin.llvmlower.compact_abi import BaseBuffer, PointerBinding, compact_pointer_entry


def entry():
    block = Block(arg_types=[llvm.LLVMPointerType()] * 3)
    left, right = llvm.LoadOp(block.args[0], i32), llvm.LoadOp(block.args[1], i32)
    total = llvm.AddOp(left.results[0], right.results[0])
    block.add_ops([left, right, total, llvm.StoreOp(total.results[0], block.args[2]), llvm.ReturnOp()])
    return llvm.FuncOp('original', llvm.LLVMFunctionType([a.type for a in block.args]), body=Region(block))


def compact(fn, **overrides):
    options = dict(symbol='compact', bases=[BaseBuffer('constant', 16, 32), BaseBuffer('mutable', 32, 32)],
                   bindings=[PointerBinding(0, 0, 0, 4), PointerBinding(1, 0, 4, 4), PointerBinding(2, 1, 16, 4)],
                   binding_provenance='test compiler layout')
    options.update(overrides)
    return compact_pointer_entry(fn, **options)


def test_compact_entry_changes_only_address_binding_and_keeps_original():
    original = entry()
    result, contract = compact(original)
    ModuleOp([original, result]).verify()
    assert len(original.body.blocks.first.args) == 3
    assert len(result.body.blocks.first.args) == 2
    assert contract['storage_reused'] is False
    assert contract['performance_claim'] == 'UNMEASURED'
    ops = list(result.body.blocks.first.ops)
    assert [op.name for op in ops[4:]] == [op.name for op in original.body.blocks.first.ops]
    load_left, load_right, _, store, _ = ops[4:]
    assert load_left.operands[0] is result.body.blocks.first.args[0]
    assert load_right.operands[0] is ops[1].results[0]
    assert store.operands[1] is ops[3].results[0]
    assert 'inbounds' not in ops[1].properties
    assert ops[0].results[0].type.bitwidth == 32


@pytest.mark.parametrize('overrides', [
    {'symbol': 'original'},
    {'binding_provenance': ''},
    {'bindings': [PointerBinding(0, 0, 0, 4)]},
    {'bindings': [PointerBinding(0, 0, 0, 4)] * 3},
    {'bindings': [PointerBinding(0, 0, 0, 4), PointerBinding(1, 0, 14, 4), PointerBinding(2, 1, 0, 4)]},
    {'bases': [BaseBuffer('constant', 16, 4), BaseBuffer('mutable', 32, 32)]},
    {'bases': [BaseBuffer('constant', 16, 32), BaseBuffer('mutable', 32, True)]},
])
def test_invalid_contract_refuses_without_mutation(overrides):
    original = entry()
    with pytest.raises(ValueError):
        compact(original, **overrides)
    assert len(original.body.blocks.first.args) == 3
    original.verify()
