"""Block signatures: attribute host cost to the emitter SHAPE, as an operation sequence.

Measured 2026-09-09 on whole-model ResNet-50: three host levers worth 10%, 21% and 45% of the
host lane each hid under one `integer_arithmetic` family total. Reading the per-block operation
sequences is what separated them -- `fsub ashr and and xor or` (a float max built from bit
patterns), a ten-integer-op round-to-even, and an im2col row loop peeling coordinates with
`udiv urem`. The activity record now carries those sequences so an authoring loop is told the
primitive, not only the family. Every case below is built so the correct answer is unambiguous,
and one is a mutation that must FAIL if grouping or trip weighting breaks.
"""
from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects import builtin, func, llvm
from xdsl.parser import Parser

from merlin.perf.host_cfg_activity import analyze_host_cfg_activity


def _module(text: str):
    ctx = Context(allow_unregistered=True)
    for d in (builtin.Builtin, llvm.LLVM, func.Func):
        ctx.load_dialect(d)
    module = Parser(ctx, text).parse_module()
    return next(o for o in module.walk()
                if o.name in ("llvm.func", "func.func") and o.regions and o.regions[0].blocks)


# One counted loop of 10 trips whose body is `mul add` (an addressing shape), then a return.
_LOOP = '''
"builtin.module"() ({
  "llvm.func"() <{sym_name = "k", function_type = !llvm.func<void (!llvm.ptr)>}> ({
  ^entry(%p: !llvm.ptr):
    %c0 = "llvm.mlir.constant"() <{value = 0 : i64}> : () -> i64
    %c1 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
    %n  = "llvm.mlir.constant"() <{value = 10 : i64}> : () -> i64
    "llvm.br"(%c0)[^header] : (i64) -> ()
  ^header(%i: i64):
    %cond = "llvm.icmp"(%i, %n) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%cond)[^body, ^exit] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^body:
    %m = "llvm.mul"(%i, %c1) : (i64, i64) -> i64
    %a = "llvm.add"(%m, %c1) : (i64, i64) -> i64
    %next = "llvm.add"(%i, %c1) : (i64, i64) -> i64
    "llvm.br"(%next)[^header] : (i64) -> ()
  ^exit:
    "llvm.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
'''


def test_signatures_are_grouped_by_operation_sequence_and_weighted_by_trips() -> None:
    act = analyze_host_cfg_activity(_module(_LOOP))
    assert act["status"] == "derived", act["problems"]
    rows = act["block_signatures"]
    body = next(r for r in rows if r["signature"] == "mul add add br")
    assert body["blocks"] == 1
    assert body["trips"] == 10
    assert body["operations_per_trip"] == 4
    assert body["dynamic_operations"]["integer_arithmetic"] == 30   # mul + 2 add, x10 trips
    assert body["dynamic_operations"]["branch"] == 10
    assert body["dynamic_total"] == 40


def test_signatures_are_ranked_by_dynamic_total_descending() -> None:
    rows = analyze_host_cfg_activity(_module(_LOOP))["block_signatures"]
    totals = [r["dynamic_total"] for r in rows]
    assert totals == sorted(totals, reverse=True)
    assert rows[0]["signature"] == "mul add add br"          # the loop body dominates a 1-trip entry


def test_signature_totals_partition_the_family_totals() -> None:
    """Grouping must lose nothing: the sum over signatures equals the whole-lane total."""
    act = analyze_host_cfg_activity(_module(_LOOP))
    whole = sum(v for v in act["dynamic_operations"].values() if isinstance(v, int))
    grouped = sum(r["dynamic_total"] for r in act["block_signatures"])
    assert grouped == whole


def test_a_changed_body_produces_a_different_signature() -> None:
    """Mutation: swapping the body's `mul` for `sub` must move the cost to a new signature."""
    mutated = _LOOP.replace('"llvm.mul"(%i, %c1)', '"llvm.sub"(%i, %c1)')
    rows = analyze_host_cfg_activity(_module(mutated))["block_signatures"]
    sigs = {r["signature"] for r in rows}
    assert "sub add add br" in sigs
    assert "mul add add br" not in sigs
