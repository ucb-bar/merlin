"""Parsed LLVM address/control observations, without reading pointed storage."""

import pytest
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from merlin.llvmlower.static_llvm_cfg import (
    StaticAddress,
    StaticInt,
    StaticPointer,
    StaticTraceError,
    trace_static_function,
)


def parsed(body, args="%p: !llvm.ptr", declarations=""):
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(llvm.LLVM)
    module = Parser(ctx, f"builtin.module {{ {declarations} llvm.func @trace({args}) {{ {body} }} }}").parse_module()
    module.verify()
    return list(module.body.block.ops)[-1]


def run(fn, arguments=(), **kwargs):
    return list(
        trace_static_function(
            fn,
            arguments,
            observe=lambda op: isinstance(op, llvm.InlineAsmOp),
            pointer_index_bits=64,
            **kwargs,
        )
    )


def test_void_inline_asm_is_observed_at_each_actual_loop_and_tail_coordinate():
    fn = parsed("""
      %zero = llvm.mlir.constant(0 : i64) : i64
      %three = llvm.mlir.constant(3 : i64) : i64
      %end = llvm.mlir.constant(10 : i64) : i64
      llvm.br ^loop(%zero : i64)
    ^loop(%i: i64):
      %ptr = llvm.getelementptr %p[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i16
      %address = llvm.ptrtoint %ptr : !llvm.ptr to i64
      %unused = llvm.inline_asm has_side_effects "opaque $0", "r" %address : (i64) -> !llvm.void
      %next = llvm.add %i, %three : i64
      %more = llvm.icmp "ult" %next, %end : i64
      llvm.cond_br %more, ^loop(%next : i64), ^tail
    ^tail:
      %last = llvm.ptrtoint %p : !llvm.ptr to i64
      %ignored = llvm.inline_asm has_side_effects "tail $0", "r" %last : (i64) -> !llvm.void
      llvm.return
    """)
    root = object()
    steps = run(fn, [StaticPointer(root, StaticInt(8, 64))], pointer_bits=64)
    assert [step.inputs[0] for step in steps] == [StaticAddress(root, StaticInt(i, 64)) for i in [8, 14, 20, 26, 8]]


@pytest.mark.parametrize("result", ["i32", "!llvm.ptr", "!llvm.void"])
def test_only_unused_void_observed_result_is_admitted(result):
    use = 'llvm.inline_asm "use", "r" %v : (!llvm.void) -> ()' if result == "!llvm.void" else ""
    fn = parsed(f'%v = llvm.inline_asm "unknown", "" : () -> {result} {use} llvm.return', args="")
    with pytest.raises(StaticTraceError, match="unused void"):
        run(fn)


def test_unselected_void_asm_and_selected_control_still_refuse():
    fn = parsed('%v = llvm.inline_asm "unknown", "" : () -> !llvm.void llvm.return', args="")
    with pytest.raises(StaticTraceError, match="unsupported static operation"):
        list(trace_static_function(fn, [], observe=lambda op: False, pointer_index_bits=64))
    with pytest.raises(StaticTraceError, match="non-control"):
        list(trace_static_function(parsed("llvm.return", args=""), [], observe=lambda op: True, pointer_index_bits=64))


def test_symbolic_address_add_sub_roundtrip_preserves_root_and_modular_offset():
    fn = parsed("""
      %a = llvm.ptrtoint %p : !llvm.ptr to i64
      %n = llvm.mlir.constant(-3 : i64) : i64
      %b = llvm.add %n, %a : i64
      %c = llvm.sub %b, %n : i64
      %back = llvm.inttoptr %c : i64 to !llvm.ptr
      llvm.inline_asm "observe", "r,r,r" %a, %b, %back : (i64,i64,!llvm.ptr) -> ()
      llvm.return
    """)
    p = StaticPointer("root", StaticInt(1, 64))
    values = run(fn, [p], pointer_bits=64)[0].inputs
    assert values == (StaticAddress("root", StaticInt(1, 64)), StaticAddress("root", StaticInt(-2, 64)), p)


@pytest.mark.parametrize(
    "operation",
    [
        "%bad = llvm.and %a, %n : i64",
        "%bad = llvm.mul %a, %n : i64",
        "%bad = llvm.sub %n, %a : i64",
        "%bad = llvm.add %a, %a : i64",
        '%bad = llvm.icmp "eq" %a, %n : i64',
        "%bad = llvm.trunc %a : i64 to i32",
    ],
)
def test_symbolic_base_does_not_supply_absolute_integer_bits(operation):
    fn = parsed(
        f"%a = llvm.ptrtoint %p : !llvm.ptr to i64 %n = llvm.mlir.constant(1 : i64) : i64 {operation} llvm.return"
    )
    with pytest.raises(StaticTraceError, match="integer operand|required|symbolic address|poison-producing"):
        run(fn, [StaticPointer("root", StaticInt(0, 64))], pointer_bits=64)


@pytest.mark.parametrize("pointer_bits,result_bits", [(None, 64), (32, 64), (64, 32), (128, 128)])
def test_unknown_or_mixed_pointer_representation_width_refuses(pointer_bits, result_bits):
    fn = parsed(f"%a = llvm.ptrtoint %p : !llvm.ptr to i{result_bits} llvm.return")
    with pytest.raises(StaticTraceError, match="explicit equal pointer widths"):
        run(fn, [StaticPointer("root", StaticInt(0, 64))], pointer_bits=pointer_bits)


def test_symbolic_address_poison_and_step_budget_remain_checked():
    fn = parsed(
        """%a = llvm.ptrtoint %p : !llvm.ptr to i64
        %n = llvm.mlir.constant(1 : i64) : i64
        %b = llvm.add %a, %n overflow<nsw> : i64 llvm.return"""
    )
    with pytest.raises(StaticTraceError, match="poison-producing"):
        run(fn, [StaticPointer("root", StaticInt(0, 64))], pointer_bits=64)
    loop = parsed('llvm.br ^loop ^loop: %v = llvm.inline_asm "opaque", "" : () -> !llvm.void llvm.br ^loop', args="")
    with pytest.raises(StaticTraceError, match="step limit"):
        run(loop, max_steps=8)


@pytest.mark.parametrize("conversion", ["from", "to"])
def test_symbolic_conversion_does_not_infer_other_address_space_layout(conversion):
    if conversion == "from":
        fn = parsed("%a = llvm.ptrtoint %p : !llvm.ptr<1> to i64 llvm.return", args="%p: !llvm.ptr<1>")
    else:
        fn = parsed("%a = llvm.ptrtoint %p : !llvm.ptr to i64 %b = llvm.inttoptr %a : i64 to !llvm.ptr<1> llvm.return")
    with pytest.raises(StaticTraceError, match="address space"):
        run(fn, [StaticPointer("root", StaticInt(0, 64))], pointer_bits=64)


GLOBAL = "llvm.mlir.global external @buffer() : i8"


def test_declared_global_uses_only_explicit_caller_binding():
    fn = parsed(
        """%g = llvm.mlir.addressof @buffer : !llvm.ptr
      %a = llvm.ptrtoint %g : !llvm.ptr to i64
      llvm.inline_asm "observe", "r,r" %g, %a : (!llvm.ptr,i64) -> () llvm.return""",
        args="",
        declarations=GLOBAL,
    )
    pointer = StaticPointer(object(), StaticInt(24, 64))
    assert run(fn, global_addresses={"buffer": pointer}, pointer_bits=64)[0].inputs == (
        pointer,
        StaticAddress(pointer.base, pointer.offset),
    )
    with pytest.raises(StaticTraceError, match="no explicit binding"):
        run(fn)


@pytest.mark.parametrize(
    "declarations,mapping",
    [
        (GLOBAL, {"missing": StaticPointer("root", StaticInt(0, 64))}),
        ("llvm.func @buffer()", {"buffer": StaticPointer("root", StaticInt(0, 64))}),
        ("llvm.mlir.global external thread_local @buffer() : i8", {"buffer": StaticPointer("root", StaticInt(0, 64))}),
        (GLOBAL, {"buffer": StaticPointer("root", StaticInt(0, 32))}),
        (GLOBAL, {"buffer": StaticInt(0, 64)}),
        (GLOBAL, {1: StaticPointer("root", StaticInt(0, 64))}),
        (
            "llvm.mlir.global external @buffer() {addr_space = 1 : i32} : i8",
            {"buffer": StaticPointer("root", StaticInt(0, 64))},
        ),
    ],
)
def test_global_binding_cannot_invent_storage_or_layout(declarations, mapping):
    with pytest.raises(StaticTraceError, match="global"):
        run(parsed("llvm.return", args="", declarations=declarations), global_addresses=mapping)


def test_addressof_cannot_reinterpret_bound_global_in_other_address_space():
    fn = parsed("%g = llvm.mlir.addressof @buffer : !llvm.ptr<1> llvm.return", args="", declarations=GLOBAL)
    with pytest.raises(StaticTraceError, match="address space"):
        run(fn, global_addresses={"buffer": StaticPointer("root", StaticInt(0, 64))})


@pytest.mark.parametrize("mutation", ["replace", "add", "offset"])
def test_global_mapping_mutation_by_observer_or_iterator_client_refuses(mutation):
    fn = parsed('llvm.inline_asm "observe", "" : () -> () llvm.return', args="", declarations=GLOBAL)
    p = StaticPointer("root", StaticInt(0, 64))
    mapping = {"buffer": p}

    def change():
        if mutation == "replace":
            mapping["buffer"] = StaticPointer("other", StaticInt(0, 64))
        elif mutation == "add":
            mapping["extra"] = p
        else:
            object.__setattr__(p.offset, "value", 8)

    iterator = trace_static_function(
        fn, [], observe=lambda op: isinstance(op, llvm.InlineAsmOp), pointer_index_bits=64, global_addresses=mapping
    )
    next(iterator)
    change()
    with pytest.raises(StaticTraceError, match="global mapping changed"):
        list(iterator)


def test_observer_cannot_change_mapping_before_selected_step_is_yielded():
    fn = parsed('llvm.inline_asm "observe", "" : () -> () llvm.return', args="", declarations=GLOBAL)
    mapping = {"buffer": StaticPointer("root", StaticInt(0, 64))}

    def observe(op):
        mapping.clear()
        return True

    with pytest.raises(StaticTraceError, match="global mapping changed"):
        list(trace_static_function(fn, [], observe=observe, pointer_index_bits=64, global_addresses=mapping))
