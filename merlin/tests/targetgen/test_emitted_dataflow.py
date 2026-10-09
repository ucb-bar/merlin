"""Actual emitted LLVM observations retain actions without target authority."""

from io import StringIO

import pytest

from merlin.targetgen.contract.emitted_dataflow import DataflowUnavailable, observe_emitted_dataflow


def body(operations, *, parameters="%src: !llvm.ptr, %dst: !llvm.ptr"):
    return "module { llvm.func @kernel(" + parameters + ") {\n" + operations + "\nllvm.return } }"


def observe(text, **kwargs):
    return observe_emitted_dataflow(text, entry_symbol="kernel", pointer_bits=64, **kwargs)


_ACTIONS = """
%address = llvm.ptrtoint %src : !llvm.ptr to i64
%seven = llvm.mlir.constant(7 : i64) : i64
llvm.inline_asm has_side_effects "op $0, $1", "r,r,~{memory}" %address, %seven : (i64, i64) -> ()
%value = llvm.load %src {alignment = 1 : i64} : !llvm.ptr -> i8
llvm.store volatile %value, %dst {alignment = 1 : i64} : i8, !llvm.ptr
"""


def test_actual_ssa_memory_and_assembly_actions_have_no_instruction_interpretation():
    result = observe(body(_ACTIONS))
    assert tuple(action.kind for action in result.actions) == ("assembly", "load", "store", "return")
    assembly, load, store, _ = result.actions
    assert assembly.assembly == "op $0, $1"
    assert assembly.constraints == "r,r,~{memory}"
    assert assembly.side_effects is True
    assert result.argument_origin(assembly.operands[0]) == 0
    assert result.constant(assembly.operands[1]) == 7
    assert result.value(load.results[0]).kind == "memory_read"
    assert store.operands == (load.results[0], result.arguments[1])
    assert load.volatile is False and store.volatile is True
    assert load.alignment == store.alignment == 1
    assert "no source/ISA/effect/runtime authority" in result.scope


def test_generic_operation_spelling_and_renamed_ssa_preserve_actual_actions():
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(llvm.LLVM)
    original = body(_ACTIONS)
    module = Parser(context, original).parse_module()
    generic = StringIO()
    Printer(stream=generic, print_generic_format=True).print_op(module)
    a, b = observe(original), observe(generic.getvalue())
    assert a.values == b.values and a.actions == b.actions
    assert a.source_sha256 != b.source_sha256


def test_independent_action_schedules_are_retained_without_template_matching():
    first = "%a = llvm.ptrtoint %src : !llvm.ptr to i64\n"
    second = "%b = llvm.ptrtoint %dst : !llvm.ptr to i64\n"
    action = 'llvm.inline_asm has_side_effects "op $0", "r" {arg} : (i64) -> ()\n'
    a = observe(body(first + second + action.format(arg="%a") + action.format(arg="%b")))
    b = observe(body(second + first + action.format(arg="%b") + action.format(arg="%a")))
    assert [a.argument_origin(row.operands[0]) for row in a.actions[:-1]] == [0, 1]
    assert [b.argument_origin(row.operands[0]) for row in b.actions[:-1]] == [1, 0]


def test_defined_bit_vector_constants_and_pointer_identity_are_separate():
    result = observe(
        body("""
%a = llvm.ptrtoint %src : !llvm.ptr to i64
%zero = llvm.mlir.constant(0 : i64) : i64
%offset = llvm.mlir.constant(1 : i64) : i64
%identity = llvm.add %a, %zero : i64
%shifted = llvm.add %identity, %offset : i64
%p = llvm.inttoptr %identity : i64 to !llvm.ptr
%max = llvm.mlir.constant(-1 : i8) : i8
%one = llvm.mlir.constant(1 : i8) : i8
%wrap = llvm.add %max, %one : i8
%factor = llvm.mlir.constant(3 : i8) : i8
%product = llvm.mul %factor, %factor : i8
llvm.store %wrap, %p : i8, !llvm.ptr
""")
    )
    values = {row.kind: row for row in result.values if row.kind == "llvm.inttoptr"}
    assert result.argument_origin(values["llvm.inttoptr"].ordinal) == 0
    adds = [row for row in result.values if row.kind == "llvm.add"]
    assert result.argument_origin(adds[1].ordinal) is None
    assert result.constant(adds[-1].ordinal) == 0
    assert result.constant(result.values[-1].ordinal) == 9
    assert result.constant(adds[0].ordinal) is None


@pytest.mark.parametrize(
    "operations",
    [
        "%a = llvm.ptrtoint %src : !llvm.ptr to i32",
        "%a = llvm.mlir.constant(1 : i32) : i32\n%p = llvm.inttoptr %a : i32 to !llvm.ptr",
        "%a = llvm.mlir.constant(1 : i8) : i8\n%b = llvm.add %a, %a overflow<nsw> : i8",
        "%a = llvm.mlir.constant(1 : i8) : i8\n%b = llvm.or disjoint %a, %a : i8",
        "%a = llvm.mlir.constant(8 : i8) : i8\n%b = llvm.shl %a, %a : i8",
        "%a = llvm.mlir.undef : i64",
        "%a = llvm.mlir.poison : i64",
        "%a = llvm.load %src {alignment = 3 : i64} : !llvm.ptr -> i8",
        "%a = llvm.load %src {ordering = 2 : i64} : !llvm.ptr -> i8",
        'llvm.inline_asm has_side_effects "op", "~{memory},r" %src : (!llvm.ptr) -> ()',
        'llvm.inline_asm has_side_effects "op", "~{}" : () -> ()',
        '%a = llvm.inline_asm "op", "=r" : () -> i64',
        'llvm.inline_asm "op", "m" %src : (!llvm.ptr) -> ()',
        "%a = llvm.mlir.constant(1 : i8) : i8\nllvm.store %a, %src {nontemporal} : i8, !llvm.ptr",
    ],
)
def test_undefined_or_unsupported_semantics_cannot_be_silently_dropped(operations):
    with pytest.raises(DataflowUnavailable):
        observe(body(operations))


@pytest.mark.parametrize(
    "text",
    [
        "module { llvm.func @kernel(!llvm.ptr) }",
        body("").replace("module {", "module { llvm.func @opaque(!llvm.ptr) "),
        body("").replace("{\n", "attributes {readonly} {\n"),
        body("").replace("llvm.func @kernel", "llvm.func fastcc @kernel"),
        body("").replace("%src: !llvm.ptr", "%src: i64"),
        body("").replace("llvm.return", "llvm.br ^next\n^next:\nllvm.return"),
        body("").replace("module {", 'module attributes {owner = "candidate"} {'),
        body("").replace("llvm.return", "llvm.return\nllvm.return"),
        "not an emitted module",
    ],
)
def test_extra_dispatch_control_flow_metadata_and_broken_rosters_refuse(text):
    with pytest.raises(DataflowUnavailable):
        observe(text)


def test_host_values_do_not_become_an_assembly_or_device_observation():
    result = observe(
        body("""
%v = llvm.load %src : !llvm.ptr -> i8
llvm.store %v, %dst : i8, !llvm.ptr
""")
    )
    assert tuple(row.kind for row in result.actions) == ("load", "store", "return")
    assert all(row.assembly is None for row in result.actions)


def test_long_definition_chain_is_bounded_and_does_not_recurse():
    operations = ["%zero = llvm.mlir.constant(0 : i64) : i64", "%v0 = llvm.ptrtoint %src : !llvm.ptr to i64"]
    operations.extend(f"%v{i} = llvm.add %v{i - 1}, %zero : i64" for i in range(1, 1600))
    text = body("\n".join(operations))
    result = observe(text, max_operations=1700)
    assert result.argument_origin(result.values[-1].ordinal) == 0
    with pytest.raises(DataflowUnavailable, match="operation roster"):
        observe(text, max_operations=100)


@pytest.mark.parametrize("selection", [{"pointer_bits": True}, {"pointer_bits": 0}, {"max_operations": True}])
def test_explicit_width_and_observation_budget_are_not_boolean_or_implicit(selection):
    kwargs = {"entry_symbol": "kernel", "pointer_bits": 64, **selection}
    with pytest.raises(DataflowUnavailable):
        observe_emitted_dataflow(body(""), **kwargs)
