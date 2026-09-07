"""Effects and poison cannot be discarded by integer address normalization."""
import pytest

from merlin.perf.integer_address_equivalence import compare_integer_address_identity
from merlin.targetgen.rocc.decode import _parse_module


def module(body, *, arguments="%i: i64, %j: i64, %p: !llvm.ptr", tail="llvm.return"):
    text = 'builtin.module { llvm.func @entry(' + arguments + ') {\n' + body + '\n' + tail + '\n} }'
    parsed = _parse_module(text)
    assert parsed is not None, text
    return parsed


PREFIX = '''%seven = llvm.mlir.constant(7 : i64) : i64
%eleven = llvm.mlir.constant(11 : i64) : i64
%seventy_seven = llvm.mlir.constant(77 : i64) : i64'''
HORNER = '''%a = llvm.mul %i, %seven : i64
%b = llvm.add %a, %j : i64
%result = llvm.mul %b, %eleven : i64'''
EXPANDED = '''%a = llvm.mul %i, %seventy_seven : i64
%b = llvm.mul %j, %eleven : i64
%result = llvm.add %a, %b : i64'''
STORE = 'llvm.store %result, %p : i64, !llvm.ptr'


def compare(before, after):
    return compare_integer_address_identity(module(PREFIX + '\n' + before), module(PREFIX + '\n' + after))


def test_horner_and_sum_products_preserve_store_address_value():
    result = compare(HORNER + '\n' + STORE, EXPANDED + '\n' + STORE)
    assert result["status"] == "conditional_integer_identity", result
    assert result["llvm_undef_semantics_qualified"] is False
    assert result["floating_arithmetic_normalized"] is False


def test_integer_modular_coefficients_not_unbounded_integers():
    a = module('%c = llvm.mlir.constant(-1 : i64) : i64\n%result = llvm.mul %i, %c : i64\n' + STORE)
    b = module('%z = llvm.mlir.constant(0 : i64) : i64\n%result = llvm.sub %z, %i : i64\n' + STORE)
    assert compare_integer_address_identity(a, b)["status"] == "conditional_integer_identity"


def test_unsigned_divide_by_one_identity():
    a = module('%one = llvm.mlir.constant(1 : i64) : i64\n%result = llvm.udiv %i, %one : i64\n' + STORE)
    b = module('llvm.store %i, %p : i64, !llvm.ptr')
    assert compare_integer_address_identity(a, b)["status"] == "conditional_integer_identity"


def test_remainder_one_keeps_poison_dependencies():
    a = module('%one = llvm.mlir.constant(1 : i64) : i64\n%result = llvm.urem %i, %one : i64\n' + STORE)
    b = module('%result = llvm.mlir.constant(0 : i64) : i64\n' + STORE)
    assert compare_integer_address_identity(a, b)["status"] == "UNKNOWN"


@pytest.mark.parametrize("change", [
    EXPANDED.replace('%seventy_seven', '%seven') + '\n' + STORE,
    EXPANDED + '\n' + STORE.replace('%p', '%q'),
    EXPANDED + '\n%loaded = llvm.load %p : !llvm.ptr -> i64\n' + STORE,
    EXPANDED + '\nllvm.inline_asm has_side_effects "fence", "" : () -> ()\n' + STORE,
])
def test_changed_coefficients_memory_and_sync_refuse(change):
    args = '%i: i64, %j: i64, %p: !llvm.ptr, %q: !llvm.ptr'
    result = compare_integer_address_identity(module(PREFIX+'\n'+HORNER+'\n'+STORE, arguments=args),
        module(PREFIX+'\n'+change, arguments=args))
    assert result["status"] == "UNKNOWN"


def test_flags_are_not_ignored():
    before = module(PREFIX+'\n'+HORNER+'\n'+STORE)
    after = module(PREFIX+'\n'+EXPANDED+'\n'+STORE)
    from xdsl.dialects.builtin import IntegerAttr, i32
    next(op for op in before.walk() if op.name == 'llvm.mul').properties['overflowFlags'] = IntegerAttr(1, i32)
    result = compare_integer_address_identity(before, after)
    assert result['status'] == 'UNKNOWN'
    assert 'overflowFlags' in result['reason']


def test_float_operations_are_exact_and_never_reassociated():
    args = '%x: f32, %y: f32, %p: !llvm.ptr'
    a = module('%r = llvm.fadd %x, %y : f32\nllvm.store %r, %p : f32, !llvm.ptr', arguments=args)
    b = module('%r = llvm.fadd %y, %x : f32\nllvm.store %r, %p : f32, !llvm.ptr', arguments=args)
    assert compare_integer_address_identity(a, b)['status'] == 'UNKNOWN'


def test_cfg_edge_change_is_not_integer_identity():
    a = module('llvm.br ^left\n^left:\nllvm.br ^right\n^right:')
    b = module('llvm.br ^right\n^left:\nllvm.br ^right\n^right:')
    assert compare_integer_address_identity(a, b)['status'] == 'UNKNOWN'


def test_analysis_bound_is_explicit_unknown():
    a = module('')
    assert compare_integer_address_identity(a, a, max_operations=1)['status'] == 'UNKNOWN'


def test_dead_nonzero_division_removed_only_when_semantically_dead():
    a = module('%c = llvm.mlir.constant(7 : i64) : i64\n%unused = llvm.udiv %i, %c : i64')
    b = module('')
    result = compare_integer_address_identity(a, b)
    assert result['status'] == 'conditional_integer_identity'
    assert result['before']['pruned_total_integer_operations'] == 1


def test_dead_division_feeding_remainder_one_needs_explicit_defined_domain():
    a = module('''%c = llvm.mlir.constant(7 : i64) : i64
%one = llvm.mlir.constant(1 : i64) : i64
%q = llvm.udiv %i, %c : i64
%result = llvm.urem %q, %one : i64
''' + STORE)
    b = module('%result = llvm.mlir.constant(0 : i64) : i64\n' + STORE)
    assert compare_integer_address_identity(a, b)['status'] == 'UNKNOWN'
    result = compare_integer_address_identity(a, b, defined_integer_domain=True)
    assert result['status'] == 'conditional_integer_identity', result
    assert result['defined_integer_domain_explicitly_assumed'] is True
    assert result['domain_assumptions_discharged'] is False
    assert result['llvm_undef_semantics_qualified'] is False


@pytest.mark.parametrize('divisor', ['%zero', '%j'])
def test_dead_zero_or_unknown_division_is_not_pruned(divisor):
    a = module('%zero = llvm.mlir.constant(0 : i64) : i64\n%unused = llvm.udiv %i, '+divisor+' : i64')
    b = module('')
    result = compare_integer_address_identity(a, b, defined_integer_domain=True)
    assert result['status'] == 'UNKNOWN'
    assert result['before']['pruned_total_integer_operations'] == 0


def test_dead_load_is_observable_and_never_pruned():
    a = module('%unused = llvm.load %p : !llvm.ptr -> i64')
    assert compare_integer_address_identity(a, module(''), defined_integer_domain=True)['status'] == 'UNKNOWN'


def test_explicit_undef_contradicts_defined_integer_domain():
    a = module('%u = llvm.mlir.undef : i64')
    result = compare_integer_address_identity(a, a, defined_integer_domain=True)
    assert result['status'] == 'UNKNOWN'
    assert 'undef/poison' in result['reason']
