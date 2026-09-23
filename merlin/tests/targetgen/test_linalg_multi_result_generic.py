"""Multi-result ``linalg.generic`` in the linalg-on-tensors reader
(:mod:`merlin.targetgen.contract.linalg_iface`).

An arg-reduce (``aten.min.dim`` / ``argmin`` / ``argmax``) lowers to ONE ``linalg.generic`` that
yields TWO results — the reduced value and the index that produced it. xDSL 0.68 prints that op with
a parenthesized result list (``-> (T0, T1)``) but parses the arrow with ``parse_attribute``, which
reads a leading ``(`` as the operand half of a function type and then demands an ``->``. So the op
did not round-trip, and the failure surfaced at the FOLLOWING op — which is what made this look like
a broken ``tensor.expand_shape`` rather than an unreadable generic.

Two properties are pinned here, because each failed independently:

1. the multi-result generic PARSES, and each result binds separately, so a consumer of result 1 (the
   index) is not silently told it reads result 0 (the value);
2. what still cannot be parsed FAILS CLOSED naming the op it stopped on — distinguishing an error
   raised inside an op from a desynchronisation that surfaces at the next one.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract.linalg_iface import (
    LinalgParseError,
    _line_depths,
    _op_mnemonic,
    is_linalg_on_tensors,
    parse_linalg_mlir,
)

# The shape smolvla's `aten.min.dim` actually emits: a two-result generic whose body materialises the
# index with `linalg.index`, followed by two `tensor.expand_shape` consumers -- one per result. The
# trailing ops are the point: alignment loss is invisible until something downstream must resolve.
_ARG_REDUCE = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<1x50xi64>) -> tensor<1x1xi64> {
    %c = arith.constant 0 : i64
    %e = tensor.empty() : tensor<1xi64>
    %v0 = linalg.fill ins(%c : i64) outs(%e : tensor<1xi64>) -> tensor<1xi64>
    %i0 = tensor.splat %c : tensor<1xi64>
    %val, %idx = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, \
affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], \
iterator_types = ["parallel", "reduction"]} ins(%a : tensor<1x50xi64>) \
outs(%v0, %i0 : tensor<1xi64>, tensor<1xi64>) \
attrs =  {prov.family = "arg_reduce", prov.op = "aten_min_dim"} {
    ^bb0(%x: i64, %y: i64, %z: i64):
      %k = linalg.index 1 : index
      %kc = arith.index_cast %k : index to i64
      %p = arith.cmpi slt, %x, %y : i64
      %nv = arith.select %p, %x, %y : i64
      %ni = arith.select %p, %kc, %z : i64
      linalg.yield %nv, %ni : i64, i64
    } -> (tensor<1xi64>, tensor<1xi64>)
    %ev = tensor.expand_shape %val [[0 : i64, 1 : i64]] output_shape [1, 1] : \
tensor<1xi64> into tensor<1x1xi64>
    %ei = tensor.expand_shape %idx [[0 : i64, 1 : i64]] output_shape [1, 1] : \
tensor<1xi64> into tensor<1x1xi64>
    return %ei : tensor<1x1xi64>
  }
}
"""


@pytest.fixture(scope="module")
def parsed() -> dict:
    return parse_linalg_mlir(_ARG_REDUCE)


def test_multi_result_generic_parses_at_all(parsed):
    """The op that blocked every argmin/argmax-bearing model, on every target and at every dtype."""
    generics = [o for o in parsed["ops"] if o["kind"] == "linalg.generic"]
    assert len(generics) == 1, [o["kind"] for o in parsed["ops"]]
    assert len(generics[0]["results"]) == 2, generics[0]["results"]


def test_both_results_are_bound_and_typed(parsed):
    """A tuple of results, not a single one silently truncated."""
    generic = next(o for o in parsed["ops"] if o["kind"] == "linalg.generic")
    assert [r["shape"] for r in generic["results"]] == [[1], [1]]
    assert [r["dtype"] for r in generic["results"]] == ["i64", "i64"]
    # the body's index materialisation survived, so the op is readable AS an arg-reduce
    assert "linalg.index" in generic["body_ops"], generic["body_ops"]
    assert generic["reduction_dims"] == [1]


def test_downstream_consumers_resolve_to_the_right_result(parsed):
    """THE ALIGNMENT PROPERTY. Both expand_shapes read the same producing op, so the op id alone
    cannot tell the reduced VALUE from the INDEX. ``result_index`` is what distinguishes them, and
    confusing the two is a silent wrong answer rather than a crash."""
    expands = [o for o in parsed["ops"] if o["kind"] == "tensor.expand_shape"]
    assert len(expands) == 2, [o["kind"] for o in parsed["ops"]]
    generic_id = next(o["id"] for o in parsed["ops"] if o["kind"] == "linalg.generic")
    assert [e["ins"][0]["source"] for e in expands] == [("op", generic_id), ("op", generic_id)]
    assert [e["ins"][0]["result_index"] for e in expands] == [0, 1]


def test_the_op_after_a_multi_result_generic_is_not_lost(parsed):
    """The symptom that hid the cause: a desynchronised parse made the FOLLOWING op look broken.
    Every op after the generic must be present, in order, with its shapes intact."""
    kinds = [o["kind"] for o in parsed["ops"]]
    assert kinds == ["linalg.generic", "tensor.expand_shape", "tensor.expand_shape"], kinds
    assert all(o["results"][0]["shape"] == [1, 1] for o in parsed["ops"][1:])
    assert parsed["results"] == [{"shape": [1, 1], "dtype": "i64"}]


def test_single_result_generic_still_parses():
    """The shimmed parse is a superset: the spelling that always worked must keep working."""
    one = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<4xf32>, %o: tensor<4xf32>) -> tensor<4xf32> {
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], \
iterator_types = ["parallel"]} ins(%a : tensor<4xf32>) outs(%o : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32):
      %m = arith.mulf %x, %x : f32
      linalg.yield %m : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
"""
    p = parse_linalg_mlir(one)
    generic = next(o for o in p["ops"] if o["kind"] == "linalg.generic")
    assert len(generic["results"]) == 1
    assert generic["body_ops"] == ["arith.mulf"]
    assert generic["ins"][0]["source"] == ("arg", 0)


def test_the_printers_own_spelling_round_trips():
    """The upstream defect stated exactly: xDSL PRINTS the multi-result arrow parenthesized and could
    not read its own output back. Print the parsed module and re-parse it -- the spelling under test
    is then xDSL's, not one this test invented."""
    import io

    from xdsl.parser import Parser
    from xdsl.printer import Printer

    from merlin.targetgen.contract.linalg_iface import make_linalg_context

    module = Parser(make_linalg_context(), _ARG_REDUCE).parse_module()
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    printed = buf.getvalue()
    assert "-> (tensor<1xi64>, tensor<1xi64>)" in printed, printed[:400]

    again = parse_linalg_mlir(printed)
    generic = next(o for o in again["ops"] if o["kind"] == "linalg.generic")
    assert len(generic["results"]) == 2
    expands = [o for o in again["ops"] if o["kind"] == "tensor.expand_shape"]
    assert [e["ins"][0]["result_index"] for e in expands] == [0, 1]


# ------------------------------------------------------------------ failing closed, at the right op


def _stock_context():
    """A context carrying xDSL's UNSHIMMED ``linalg.generic``, which reproduces the original
    desynchronisation exactly -- the only faithful way to test the attribution rule."""
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.tensor import Tensor as TensorDialect

    ctx = Context(allow_unregistered=True)
    for d in (Builtin, Func, Arith, Linalg, TensorDialect):
        ctx.load_dialect(d)
    return ctx


def test_a_desync_is_blamed_on_the_op_that_caused_it_not_the_next_one():
    """THE MISLEADING-TRACEBACK HALF OF THE DEFECT. Parsed with the stock op, the multi-result
    generic desynchronises and xDSL stops at the following ``return``. The diagnostic must name the
    generic as the culprit and the ``return`` only as where it surfaced."""
    with pytest.raises(LinalgParseError) as ei:
        parse_linalg_mlir(_ARG_REDUCE, ctx=_stock_context())
    err = ei.value
    assert err.op == "linalg.generic", err.op
    assert err.reported_op != "linalg.generic", err.reported_op
    assert "desynchronised" in str(err)


def test_a_failure_inside_an_op_is_blamed_on_that_op():
    """The opposite shape, which the same rule must get right: a generic missing its required
    ``indexing_maps`` fails INSIDE itself, past its own mnemonic. Blaming the enclosing
    ``func.func`` (or the previous op) would be exactly the mis-attribution this guards."""
    # The `linalg.fill` matters: it is a VALID op at the SAME nesting depth immediately before the
    # broken one. Without it there is nothing for a naive "always blame the previous op" rule to get
    # wrong, and this test would pass against that rule too.
    bad = """
module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%a: tensor<4xf32>, %o: tensor<4xf32>) -> tensor<4xf32> {
    %z = arith.constant 0.0 : f32
    %f = linalg.fill ins(%z : f32) outs(%o : tensor<4xf32>) -> tensor<4xf32>
    %r = linalg.generic {iterator_types = ["parallel"]} ins(%a : tensor<4xf32>) \
outs(%f : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32):
      linalg.yield %x : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
"""
    with pytest.raises(LinalgParseError) as ei:
        parse_linalg_mlir(bad)
    err = ei.value
    assert err.op == "linalg.generic", err.op
    assert err.reported_op == "linalg.generic", err.reported_op
    assert err.op != "linalg.fill", "blamed the preceding op instead of the one that failed"
    assert "indexing_maps" in str(err)


def test_the_diagnostic_carries_a_line_and_a_quoted_window():
    """An error that cannot be located costs as much as one that points at the wrong line."""
    with pytest.raises(LinalgParseError) as ei:
        parse_linalg_mlir(_ARG_REDUCE, ctx=_stock_context())
    err = ei.value
    assert err.line > 1 and err.column >= 1
    assert "|" in err.window and str(err.line) in err.window


@pytest.mark.slow
def test_the_whole_tracked_capsule_corpus_still_parses():
    """The shim replaces the op the reader uses for EVERY capsule, not just the multi-result ones, so
    the regression that matters is the corpus it already read. No tracked capsule contains a
    multi-result generic today -- which is exactly why this gap survived until a real model hit it --
    so this guards the other direction: that fixing the arrow broke nothing that worked."""
    root = repo_root() / "merlin" / "contract" / "capsules"
    parsed = 0
    for path in sorted(root.rglob("capsule.interface.mlir")):
        text = path.read_text(encoding="utf-8")
        if not is_linalg_on_tensors(text):
            continue
        try:
            inventory = parse_linalg_mlir(text)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{path.relative_to(root)} no longer parses: {type(exc).__name__}: {exc}")
        assert inventory["ops"], f"{path.relative_to(root)} parsed to an empty op list"
        parsed += 1
    assert parsed > 100, f"expected the tracked linalg corpus, found only {parsed} capsules"


# ------------------------------------------------------------------------ the structural primitives


def test_op_mnemonic_reads_both_spellings_structurally():
    """Result-bound and bare op lines, plus the non-op lines that must not be mistaken for ops."""
    assert _op_mnemonic("    %r = linalg.generic {a = 1} ins(")[0] == "linalg.generic"
    assert _op_mnemonic("    %a, %b = linalg.generic {x} ins(")[0] == "linalg.generic"
    assert _op_mnemonic("    return %r0 : tensor<1xi64>")[0] == "return"
    assert _op_mnemonic("    ^bb0(%x: i64):")[0] == ""
    assert _op_mnemonic("    } -> (tensor<1xi64>, tensor<1xi64>)")[0] == ""
    assert _op_mnemonic("")[0] == ""
    # the column is where the MNEMONIC starts, not where the line does -- that offset is what
    # separates "stopped inside this op" from "stopped at the start of the next one"
    assert _op_mnemonic("    %r = linalg.generic {a = 1}")[1] == 9
    assert _op_mnemonic("    return %r0 : tensor<1xi64>")[1] == 4


def test_line_depths_do_not_count_braces_inside_strings():
    """A brace in a string attribute is not nesting; miscounting it would send the walk-back to the
    wrong level and blame a region body."""
    # The brace inside the string is UNBALANCED on purpose: a balanced one ("a{b}c") miscounts by
    # +1 then -1 and cancels, so it cannot tell a string-aware scan from a naive one.
    lines = ["module {", "  func.func @f() {", '    %x = "op"() {s = "a{b"} : () -> ()', "  }", "}"]
    assert _line_depths(lines) == [0, 1, 2, 2, 1]
    # escaped quotes must not end the string early, or the scan resumes counting mid-literal
    assert _line_depths(['a "x\\"{" b', "next"]) == [0, 0]


def test_walk_back_skips_a_region_body():
    """The generic's own body sits one level deeper. A naive line-wise walk back blames
    ``linalg.yield``; the depth-aware one blames the generic."""
    with pytest.raises(LinalgParseError) as ei:
        parse_linalg_mlir(_ARG_REDUCE, ctx=_stock_context())
    assert ei.value.op != "linalg.yield"
