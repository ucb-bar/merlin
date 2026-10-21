"""Whole-function affine loop projection and rejected control/arithmetic mutations."""

from dataclasses import replace

import pytest
from xdsl.dialects import llvm
from xdsl.parser import Parser

from merlin.llvmlower.counted_llvm_cfg import (
    CountedAddress,
    CountedPointer,
    CountedTraceUnavailable,
    canonical_counted_kernel,
    project_counted_function,
)
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, evaluate, instances
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context


def source(*, rows=512, columns=512):
    return f"""builtin.module {{
llvm.func @event(!llvm.ptr, i64, i64, i64)
llvm.func @kernel(%a: !llvm.ptr) {{
%zero = llvm.mlir.constant(0 : i64) : i64
%one = llvm.mlir.constant(1 : i64) : i64
%rows = llvm.mlir.constant({rows} : i64) : i64
%cols = llvm.mlir.constant({columns} : i64) : i64
llvm.br ^outer(%zero : i64)
^outer(%m: i64):
llvm.br ^inner(%zero : i64)
^inner(%n: i64):
%offset = llvm.mul %m, %cols : i64
%index = llvm.add %offset, %n : i64
llvm.call @event(%a, %m, %n, %index) : (!llvm.ptr, i64, i64, i64) -> ()
%nextn = llvm.add %n, %one : i64
%moren = llvm.icmp "ult" %nextn, %cols : i64
llvm.cond_br %moren, ^inner(%nextn : i64), ^nextouter
^nextouter:
%nextm = llvm.add %m, %one : i64
%morem = llvm.icmp "ult" %nextm, %rows : i64
llvm.cond_br %morem, ^outer(%nextm : i64), ^exit
^exit:
llvm.return
}}
}}"""


def parse(text):
    module = Parser(make_llvm_context(), text).parse_module()
    module.verify()
    return next(op for op in module.body.block.ops if type(op) is llvm.FuncOp and op.sym_name.data == "kernel")


def observe(operation, operands):
    if type(operation) is llvm.CallOp and operation.callee.root_reference.data == "event":
        if len(operands) != 4 or operands[0] != CountedPointer("input"):
            raise CountedTraceUnavailable("actual neutral event pointer or arity differs")
        return Call("event", tuple(zip(("m", "n", "index"), operands[1:], strict=True)))
    return None


def project(text):
    return project_counted_function(parse(text), (CountedPointer("input"),), observe=observe)


def test_large_nested_loops_preserve_actual_affine_operands_without_expansion():
    body = project(source())
    assert len(body) == 1 and type(body[0]) is Loop and body[0].extent == 512
    inner = body[0].body[0]
    assert type(inner) is Loop and inner.extent == 512
    assert len(inner.body) == 1 and type(inner.body[0]) is Call
    call = inner.body[0]
    for m, n in ((0, 0), (1, 511), (511, 511)):
        env = {body[0].var: m, inner.var: n}
        assert tuple(evaluate(value, env) for _, value in call.args) == (m, n, m * 512 + n)


def test_small_loop_execution_matches_independent_complete_order():
    actual = [
        (evaluate(call.arg("m"), env), evaluate(call.arg("n"), env), evaluate(call.arg("index"), env))
        for call, env in instances(Kernel("k", (), project(source(rows=3, columns=5))))
    ]
    assert actual == [(m, n, m * 5 + n) for m in range(3) for n in range(5)]


@pytest.mark.parametrize("predicate", ["eq", "ne"])
def test_induction_select_preserves_initialization_and_tail_operands(predicate):
    text = source(rows=2, columns=5).replace(
        "%offset = llvm.mul %m, %cols : i64",
        f'%first = llvm.icmp "{predicate}" %n, %zero : i64\n'
        "%selected = llvm.select %first, %one, %cols : i1, i64\n"
        "%offset = llvm.mul %m, %cols : i64",
    ).replace("%a, %m, %n, %index)", "%a, %m, %n, %selected)")
    actual = [evaluate(call.arg("index"), env) for call, env in instances(Kernel("k", (), project(text)))]
    assert actual == [1 if ((n == 0) == (predicate == "eq")) else 5 for _m in range(2) for n in range(5)]


@pytest.mark.parametrize("mutation", ["comparison", "unbound", "negative"])
def test_select_outside_piecewise_induction_domain_refuses(mutation):
    text = source(rows=2, columns=5).replace(
        "%offset = llvm.mul %m, %cols : i64",
        '%first = llvm.icmp "eq" %n, %zero : i64\n'
        "%selected = llvm.select %first, %one, %cols : i1, i64\n"
        "%offset = llvm.mul %m, %cols : i64",
    )
    if mutation == "comparison":
        text = text.replace('"eq" %n', '"ult" %n')
    elif mutation == "unbound":
        text = text.replace('"eq" %n', '"eq" %one')
    else:
        text = text.replace("constant(1 : i64)", "constant(-1 : i64)")
    with pytest.raises(CountedTraceUnavailable, match="select needs|negative"):
        project(text)


@pytest.mark.parametrize(
    "old,new,reason",
    [
        ("%nextn = llvm.add %n, %one", "%nextn = llvm.add %n, %cols", "stride"),
        ('llvm.icmp "ult" %nextn', 'llvm.icmp "ule" %nextn', "comparison"),
        ("^inner(%zero : i64)", "^inner(%one : i64)", "zero seed"),
        ("%offset = llvm.mul %m, %cols", "%offset = llvm.mul %m, %n", "affine"),
        ("%offset = llvm.mul %m, %cols", "%offset = llvm.sub %m, %cols", "negative"),
        ("%nextm = llvm.add %m, %one", "%nextm = llvm.add %m, %rows", "stride"),
    ],
)
def test_mutated_stride_seed_comparison_or_arithmetic_refuses(old, new, reason):
    with pytest.raises(CountedTraceUnavailable, match=reason):
        project(source(rows=3, columns=5).replace(old, new, 1))


def test_unreachable_effect_is_not_dropped():
    text = source(rows=2, columns=3).replace("^exit:\nllvm.return", "^exit:\nllvm.return\n^unreachable:\nllvm.return")
    with pytest.raises(CountedTraceUnavailable, match="complete function"):
        project(text)


def test_unknown_effect_inside_loop_is_a_barrier():
    text = source(rows=2, columns=3).replace("@event", "@unknown")
    with pytest.raises(CountedTraceUnavailable, match="unsupported counted LLVM operation"):
        project(text)


def test_i32_wrap_and_large_i64_intermediates_are_not_assumed_safe():
    with pytest.raises(CountedTraceUnavailable, match="i64"):
        project(source(rows=2, columns=3).replace("i64", "i32"))
    with pytest.raises(CountedTraceUnavailable, match="wrap"):
        project(source(rows=1 << 32, columns=1 << 32))


def test_explicit_static_and_depth_budgets():
    function = parse(source(rows=2, columns=3))
    with pytest.raises(CountedTraceUnavailable, match="static budget"):
        project_counted_function(function, (CountedPointer("input"),), observe=observe, max_operations=3)
    with pytest.raises(CountedTraceUnavailable, match="nesting"):
        project_counted_function(function, (CountedPointer("input"),), observe=observe, max_depth=1)


def test_argument_bindings_do_not_grant_a_pointer_identity():
    with pytest.raises(CountedTraceUnavailable, match="pointer identity"):
        project_counted_function(parse(source()), (Const(0),), observe=observe)


def test_bad_observer_cannot_consume_control_flow():
    with pytest.raises(CountedTraceUnavailable, match="non-control"):
        project_counted_function(parse(source()), (CountedPointer("input"),), observe=lambda *_: Call("fake", ()))


def test_runtime_unbound_scalar_does_not_become_a_loop_extent():
    text = source().replace("%a: !llvm.ptr)", "%a: !llvm.ptr, %runtime: i64)")
    text = text.replace('%moren = llvm.icmp "ult" %nextn, %cols', '%moren = llvm.icmp "ult" %nextn, %runtime')
    function = parse(text)
    with pytest.raises(CountedTraceUnavailable, match="extent must be constant"):
        project_counted_function(function, (CountedPointer("input"), Var("not_a_loop")), observe=observe)


def pretest(*, bound=512, step=16):
    return f"""builtin.module {{
llvm.func @event(!llvm.ptr, i64, i64, i64)
llvm.func @kernel(%a: !llvm.ptr) {{
%zero = llvm.mlir.constant(0 : i64) : i64
%end = llvm.mlir.constant({bound} : i64) : i64
%step = llvm.mlir.constant({step} : i64) : i64
llvm.br ^header(%zero : i64)
^header(%i: i64):
%more = llvm.icmp "slt" %i, %end : i64
llvm.cond_br %more, ^body, ^exit
^body:
llvm.call @event(%a, %i, %zero, %i) : (!llvm.ptr, i64, i64, i64) -> ()
llvm.br ^latch
^latch:
%next = llvm.add %i, %step : i64
llvm.br ^header(%next : i64)
^exit:
llvm.return
}}
}}"""


@pytest.mark.parametrize("bound,step", [(512, 16), (5, 2), (1, 16), (7, 1)])
def test_pretest_normalizes_positive_steps_and_preserves_partial_final_iterations(bound, step):
    body = project(pretest(bound=bound, step=step))
    assert type(body[0]) is Loop and body[0].extent == (bound + step - 1) // step
    actual = [evaluate(call.arg("m"), env) for call, env in instances(Kernel("k", (), body))]
    assert actual == list(range(0, bound, step))


@pytest.mark.parametrize(
    "old,new,reason",
    [
        ('icmp "slt"', 'icmp "sle"', "comparison"),
        ("constant(16 : i64)", "constant(0 : i64)", "positive"),
        ("llvm.add %i, %step", "llvm.add %step, %step", "induction"),
        ("^header(%zero : i64)", "^header(%step : i64)", "zero seed"),
    ],
)
def test_pretest_rejects_changed_bound_stride_and_seed(old, new, reason):
    with pytest.raises(CountedTraceUnavailable, match=reason):
        project(pretest().replace(old, new, 1))


def test_actual_pointer_offsets_are_preserved_with_an_explicit_representation():
    text = (
        pretest(bound=512, step=16)
        .replace("llvm.func @event(!llvm.ptr, i64, i64, i64)", "llvm.func @address(i64, i64)")
        .replace(
            "llvm.call @event(%a, %i, %zero, %i) : (!llvm.ptr, i64, i64, i64) -> ()",
            "%ptr = llvm.getelementptr %a[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32\n"
            "%address = llvm.ptrtoint %ptr : !llvm.ptr to i64\n"
            "llvm.call @address(%address, %i) : (i64, i64) -> ()",
        )
    )

    def address_observer(operation, inputs):
        if isinstance(operation, llvm.CallOp):
            assert type(inputs[0]) is CountedAddress and inputs[0].base == "input"
            return Call("offset", (("byte", inputs[0].offset), ("element", inputs[1])))

    actual = project_counted_function(
        parse(text), (CountedPointer("input"),), observe=address_observer, pointer_bits=64
    )
    assert [
        (evaluate(call.arg("byte"), env), evaluate(call.arg("element"), env))
        for call, env in instances(Kernel("k", (), actual))
    ] == [(i * 4, i) for i in range(0, 512, 16)]
    with pytest.raises(CountedTraceUnavailable, match="explicit width"):
        project_counted_function(parse(text), (CountedPointer("input"),), observe=address_observer)
    with pytest.raises(CountedTraceUnavailable, match="non-inbounds"):
        project_counted_function(
            parse(text.replace("llvm.getelementptr ", "llvm.getelementptr inbounds ")),
            (CountedPointer("input"),),
            observe=address_observer,
            pointer_bits=64,
        )


def test_global_addresses_require_actual_declarations_and_original_bindings():
    text = (
        pretest(bound=1, step=1)
        .replace("builtin.module {", "builtin.module {\nllvm.mlir.global internal @buf(0 : i64) : i64", 1)
        .replace(
            "llvm.call @event(%a, %i, %zero, %i)",
            "%buffer = llvm.mlir.addressof @buf : !llvm.ptr\nllvm.call @event(%buffer, %i, %zero, %i)",
        )
    )
    with pytest.raises(CountedTraceUnavailable, match="global address"):
        project(text)
    body = project_counted_function(
        parse(text), (CountedPointer("input"),), observe=observe, global_addresses={"buf": CountedPointer("input")}
    )
    assert body[0].extent == 1
    with pytest.raises(CountedTraceUnavailable, match="pointer identity"):
        project_counted_function(
            parse(text), (CountedPointer("input"),), observe=observe, global_addresses={"buf": Const(0)}
        )


def test_counted_alpha_normalization_preserves_source_operands_and_order():
    original = Kernel("k", (), project(source(rows=3, columns=5)))
    renamed = Kernel(
        "k",
        (),
        (
            Loop(
                "outer",
                3,
                (
                    Loop(
                        "inner",
                        5,
                        (
                            Call(
                                "event",
                                (("m", Var("outer")), ("n", Var("inner")), ("index", Var("outer") * 5 + Var("inner"))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    assert canonical_counted_kernel(original).digest() == canonical_counted_kernel(renamed).digest()
    outer = renamed.body[0]
    inner = outer.body[0]
    damaged_call = replace(inner.body[0], args=(("m", Var("outer")), ("n", Var("inner")), ("index", Const(0))))
    damaged = replace(renamed, body=(replace(outer, body=(replace(inner, body=(damaged_call,)),)),))
    assert canonical_counted_kernel(damaged).digest() != canonical_counted_kernel(original).digest()
