"""Windows preserve the complete original typed AST and source ownership."""

import io
import re

import pytest
import test_connected_integer_bounds as B
import test_operator_structured_exclusion as RED
from merlin_experiments.phase0 import operator_structured_exclusion as R
from merlin_experiments.phase0 import operator_windowed_ir as W

from merlin.common import mlir_query as Q

LIMIT = 1 << 20


def parsed(raw, **kwargs):
    return W.parse(io.BytesIO(raw), max_window_bytes=kwargs.pop('max_window_bytes', LIMIT),
                   max_total_bytes=kwargs.pop('max_total_bytes', 4*LIMIT), **kwargs)


@pytest.mark.parametrize('source', [B.program().encode(), RED.reduction()])
def test_complete_original_ast_operation_types_and_canonical_hashes_are_equal(source):
    original = Q.parse(source.decode())
    rebuilt, metadata = parsed(source)
    assert original.is_structurally_equivalent(rebuilt)
    assert [Q.op_name(op) for op in original.walk()] == [Q.op_name(op) for op in rebuilt.walk()]
    assert metadata['compact_bytes'] == len(source)
    assert R.module_signatures(rebuilt) == R.signatures(source, max_bytes=LIMIT)


def test_cross_window_ssa_numeric_names_groups_aliases_and_nested_region_scope():
    raw = re.sub(rb'%x\b', b'%0', B.program().encode())
    raw = re.sub(rb'%w\b', b'%1', raw)
    raw = b'#kind = "public"\n' + raw.replace(b'module {', b'module attributes {prov.kind = #kind} {', 1)
    original = Q.parse(raw.decode())
    rebuilt, _ = parsed(raw)
    assert original.is_structurally_equivalent(rebuilt)
    assert R.module_signatures(rebuilt) == R.signatures(raw, max_bytes=LIMIT)


def test_long_complete_graph_is_read_under_the_same_small_per_window_cap():
    rows = ['module {', 'func.func @chain(%x: i32) -> i32 {']
    last = '%x'
    for i in range(300):
        rows.append(f'%v{i} = arith.addi {last}, %x : i32')
        last = f'%v{i}'
    rows += [f'return {last} : i32', '}', '}']
    raw = '\n'.join(rows).encode()
    assert len(raw) > 4096
    rebuilt, _ = parsed(raw, max_window_bytes=1024, max_total_bytes=32768)
    assert Q.parse(raw.decode()).is_structurally_equivalent(rebuilt)


@pytest.mark.parametrize('change', [
    lambda x: x.replace(b'%a32 = arith.extsi', b'%x = arith.extsi'),
    lambda x: x.replace(b'ins(%x, %w', b'ins(%unowned, %w'),
    lambda x: x.replace(b'%p = arith.muli', b'%p = arith.addi %unowned, %unowned : i32\n %p2 = arith.muli'),
    lambda x: x.replace(b'return %wide, %state', b'return %x, %state'),
    lambda x: x[:-6],
    lambda x: x + b'\nunknown.invalid',
])
def test_duplicates_cross_window_unowned_values_regions_footer_and_type_mutations_refuse(change):
    raw = B.program().encode()
    assert change(raw) != raw
    with pytest.raises(Exception):
        parsed(change(raw))


def test_all_allocation_and_operation_ceilings_are_explicit():
    raw = B.program().encode()
    for kw in [dict(max_window_bytes=1), dict(max_total_bytes=1), dict(max_operations=1)]:
        with pytest.raises(ValueError):
            parsed(raw, **kw)


def test_failed_partial_region_parse_cannot_pollute_prior_actual_ssa_uses():
    raw = B.program().encode()
    original = Q.parse(raw.decode())
    rebuilt, _ = parsed(raw)
    assert original.is_structurally_equivalent(rebuilt)
    # Every use owner belongs to the final reconstructed graph, including
    # values repeatedly encountered by unsuccessful multi-line candidates.
    owned = set(rebuilt.walk())
    for op in rebuilt.walk():
        for value in op.results:
            assert all(use.operation in owned for use in value.uses)


def test_grouped_results_and_cross_window_nested_regions_preserve_original_ownership():
    raw = b'''module { func.func @group(%a: i32) -> i32 {
%pair:2 = "public.pair"(%a) : (i32) -> (i32, i32)
%sum = arith.addi %pair#0, %pair#1 : i32
return %sum : i32
} }'''
    original = Q.parse(raw.decode())
    rebuilt, _ = parsed(raw)
    assert original.is_structurally_equivalent(rebuilt)


def test_original_fully_generic_module_function_and_block_header_are_preserved():
    import io

    from xdsl.printer import Printer

    original = Q.parse(B.program())
    out = io.StringIO()
    Printer(stream=out, print_generic_format=True).print_op(original)
    raw = out.getvalue().encode()
    rebuilt, _ = parsed(raw)
    assert original.is_structurally_equivalent(rebuilt)
    assert R.module_signatures(rebuilt) == R.module_signatures(original)


def test_dangling_or_cross_block_successors_never_gain_a_complete_graph():
    raw = b'''module { func.func @bad(%a: i32) -> i32 {
cf.br ^missing(%a : i32)
return %a : i32
} }'''
    with pytest.raises(Exception):
        parsed(raw)


def test_multiple_original_functions_calls_external_declarations_and_ssa_scopes():
    raw = b'''module attributes {prov.kind = "public"} {
func.func private @external(i32) -> i32
func.func @helper(%a: i32) -> i32 {
%next = arith.addi %a, %a : i32
return %next : i32
}
func.func @forward(%a: i32) -> i32 {
%next = func.call @helper(%a) : (i32) -> i32
%last = func.call @external(%next) : (i32) -> i32
return %last : i32
}
}'''
    from xdsl.printer import Printer

    original = Q.parse(raw.decode())
    original.verify()
    generic = io.StringIO()
    Printer(stream=generic, print_generic_format=True).print_op(original)
    for source in (raw, generic.getvalue().encode()):
        rebuilt, metadata = parsed(source)
        assert original.is_structurally_equivalent(rebuilt)
        assert metadata['functions'] == 3
        assert R.module_signatures(rebuilt) == R.module_signatures(original)


@pytest.mark.parametrize('mutation', [
    lambda x: x.replace(b'@helper(%a: i32)', b'@forward(%a: i32)'),
    lambda x: x.replace(b'func.call @helper(%a) : (i32) -> i32', b'func.call @missing(%a) : (i32) -> i32'),
    lambda x: x.replace(b'func.call @helper(%a) : (i32) -> i32', b'func.call @helper(%a) : (i32) -> i8'),
])
def test_cross_function_duplicate_symbols_missing_callee_and_typed_calls_refuse(mutation):
    raw = b'''module {
func.func @helper(%a: i32) -> i32 {
return %a : i32
}
func.func @forward(%a: i32) -> i32 {
%next = func.call @helper(%a) : (i32) -> i32
return %next : i32
}
}'''
    with pytest.raises(Exception):
        parsed(mutation(raw))


def test_cross_function_ssa_is_not_implicitly_imported_from_another_function():
    raw = b'''module {
func.func @first(%a: i32) -> i32 {
%v = arith.addi %a, %a : i32
return %v : i32
}
func.func @second(%a: i32) -> i32 {
return %v : i32
}
}'''
    with pytest.raises(Exception):
        parsed(raw)
