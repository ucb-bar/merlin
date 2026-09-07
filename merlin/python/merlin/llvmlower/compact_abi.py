"""Compiler-owned pointer ABI compaction with an explicit byte-offset contract.

This changes an entry signature, not its tensor storage plan. The caller must
provide the declared base buffers and bind old argument i to base[group]+offset.
No buffer reuse, non-aliasing, alignment, DMA completion or speedup is inferred.
The original function is never mutated, and a distinct symbol prevents an old
many-pointer harness from accidentally linking against the compact entry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, SymbolRefAttr, i8
from xdsl.ir import Attribute


@dataclass(frozen=True)
class BaseBuffer:
    name: str
    byte_extent: int
    pointer_index_bits: int


@dataclass(frozen=True)
class PointerBinding:
    argument_index: int
    base_index: int
    byte_offset: int
    byte_extent: int


def _natural(value: int, label: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} integer")


def _references_symbol(attribute: Attribute, symbol: str) -> bool:
    if isinstance(attribute, SymbolRefAttr):
        return attribute.root_reference.data == symbol
    parameters = getattr(attribute, 'parameters', ())
    data = getattr(attribute, 'data', ())
    children = data.values() if isinstance(data, dict) else data if isinstance(data, tuple) else ()
    return any(_references_symbol(child, symbol) for child in (*parameters, *children)
               if isinstance(child, Attribute))


def compact_pointer_entry(
    function: llvm.FuncOp,
    *,
    symbol: str,
    bases: Sequence[BaseBuffer],
    bindings: Sequence[PointerBinding],
    binding_provenance: str,
) -> tuple[llvm.FuncOp, dict]:
    """Clone an entry, replace every pointer use, return its new caller contract.

    Pointer index widths must come from the selected target's data layout. This
    routine deliberately requires them rather than assuming host pointer width.
    The binding provenance identifies the compiler/runtime layout agreement; it
    is not treated as proof that arbitrary old callers obey that agreement.
    """
    if not symbol or symbol == function.sym_name.data or not binding_provenance:
        raise ValueError('distinct entry symbol and binding provenance are required')
    if not function.body.blocks or function.function_type.is_variadic:
        raise ValueError('defined nonvariadic entry required')
    original_args = tuple(function.body.blocks.first.args)
    if not original_args or any(not isinstance(arg.type, llvm.LLVMPointerType) for arg in original_args):
        raise ValueError('only all-pointer entries are supported')
    for key in ('arg_attrs', 'res_attrs'):
        if key in function.properties or key in function.attributes:
            raise ValueError('argument/result ABI attributes need explicit remapping')
    for op in function.walk():
        if any(_references_symbol(attr, function.sym_name.data)
               for attr in (*op.properties.values(), *op.attributes.values())):
            raise ValueError('self-referencing entry cannot be renamed')
    if not bases or len({base.name for base in bases}) != len(bases) or any(not base.name for base in bases):
        raise ValueError('nonempty unique base names required')
    for base in bases:
        _natural(base.byte_extent, 'base byte extent', positive=True)
        _natural(base.pointer_index_bits, 'pointer index width', positive=True)
        if base.byte_extent >= 1 << (base.pointer_index_bits - 1):
            raise ValueError('base extent exceeds signed pointer index range')
    for binding in bindings:
        for label in ('argument_index', 'base_index', 'byte_offset', 'byte_extent'):
            _natural(getattr(binding, label), label)
    ordered = sorted(bindings, key=lambda binding: binding.argument_index)
    if [binding.argument_index for binding in ordered] != list(range(len(original_args))):
        raise ValueError('bindings must cover every argument exactly once')
    base_types = {}
    for binding in ordered:
        if binding.base_index >= len(bases):
            raise ValueError('binding refers to missing base')
        if binding.byte_offset + binding.byte_extent > bases[binding.base_index].byte_extent:
            raise ValueError('binding exceeds base extent')
        ty = original_args[binding.argument_index].type
        if base_types.setdefault(binding.base_index, ty) != ty:
            raise ValueError('one base cannot mix pointer address spaces')
    if len(base_types) != len(bases):
        raise ValueError('unused base buffer')

    result = function.clone()
    block = result.body.blocks.first
    old_args = tuple(block.args)
    new_args = [block.insert_arg(base_types[i], len(block.args)) for i in range(len(bases))]
    prefix = []
    addresses = []
    for binding in ordered:
        replacement = new_args[binding.base_index]
        if binding.byte_offset:
            index_type = IntegerType(bases[binding.base_index].pointer_index_bits)
            constant = llvm.ConstantOp(IntegerAttr(binding.byte_offset, index_type), index_type)
            address = llvm.GEPOp.from_mixed_indices(
                replacement, [constant.results[0]], i8, result_type=replacement.type)
            prefix.extend((constant, address))
            replacement = address.results[0]
        old_args[binding.argument_index].replace_all_uses_with(replacement)
        addresses.append(replacement)
    for argument in old_args:
        block.erase_arg(argument)
    if prefix:
        block.insert_ops_before(prefix, block.first_op)
    result.properties['sym_name'] = StringAttr(symbol)
    result.properties['function_type'] = llvm.LLVMFunctionType(
        [arg.type for arg in new_args], function.function_type.output)
    result.verify()
    # Check the whole CFG, not only an operation histogram: every original
    # operand/use and successor must survive under the declared address map.
    equivalence = dict(zip(function.body.blocks, result.body.blocks))
    equivalence.update(zip(original_args, addresses))
    for old_block, new_block in zip(function.body.blocks, result.body.blocks):
        if old_block is not function.body.blocks.first:
            equivalence.update(zip(old_block.args, new_block.args))
    for old_block, new_block in zip(function.body.blocks, result.body.blocks):
        new_ops = list(new_block.ops)
        if new_block is block:
            new_ops = new_ops[len(prefix):]
        old_ops = list(old_block.ops)
        if len(old_ops) != len(new_ops) or not all(
            old.is_structurally_equivalent(new, equivalence)
            for old, new in zip(old_ops, new_ops)
        ):
            raise ValueError('body changed beyond declared entry pointer substitution')
    return result, {
        'schema': 'compact_pointer_entry_v1',
        'original_symbol': function.sym_name.data,
        'compact_symbol': symbol,
        'original_argument_count': len(original_args),
        'compact_argument_count': len(bases),
        'bases': [asdict(base) for base in bases],
        'bindings': [asdict(binding) for binding in ordered],
        'binding_provenance': binding_provenance,
        'caller_obligation': 'old_argument[i] == base[base_index] + byte_offset; bases cover declared extents',
        'pointer_index_width_obligation': 'widths must match selected target data layout',
        'storage_reused': False,
        'whole_cfg_preserved_under_pointer_substitution': True,
        'alignment_or_noalias_added': False,
        'performance_claim': 'UNMEASURED',
    }
