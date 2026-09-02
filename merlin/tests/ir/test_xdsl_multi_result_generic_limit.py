"""A pinned reproducer for the one xdsl limitation that costs a claim model its routing evidence.

`smolvla_fp32_consistent` is valid MLIR that `model_coverage.load_module` cannot parse, so smolvla
contributes 46 operator tags and ZERO routing evidence. The cause is narrow and worth stating exactly,
because three plausible "fixes" are all worse than the limitation:

  xdsl 0.68's `linalg.GenericOp.parse` reads its result list with `Delimiter.NONE`, so it accepts
  `-> tensor<i64>, tensor<i64>` and rejects `-> (tensor<i64>, tensor<i64>)`. MLIR emits the
  parenthesised form. **Multi-result generics are otherwise fully supported** -- the unparenthesised
  spelling parses -- so this is a syntax gap, not a semantic one.

Why not simply fix it:

* **Patch site-packages.** xdsl is a pinned pip dependency (`xdsl>=0.68`); an edit there is lost on the
  next install and reviewed by nobody.
* **Vendor `GenericOp.parse` into merlin to change one call.** Sixty lines of upstream logic that would
  drift silently against the real parser -- and drift in a PARSER is how a capture comes to be read
  differently by two tools.
* **Change m2m's decomposition to avoid two-output generics.** `_arg_reduce` emits one fused reduction
  on purpose: it threads a running (best_value, best_index) pair so the first extremum wins, matching
  torch. Splitting it into two passes to satisfy a downstream spelling makes the emitted IR worse and
  puts the tie-break at risk.

So the limitation stands, the failure is legible (`aten_coverage._parse_failure` names the construct and
the line), and this test is the ratchet: it asserts the limitation, so it FAILS the day xdsl gains the
parenthesised form -- at which point delete it and smolvla's routing evidence comes back for free.
"""
from __future__ import annotations

import pytest

_MODULE = """builtin.module {{
  func.func @f(%a: tensor<4xi64>) -> tensor<i64> {{
    %e = tensor.empty() : tensor<i64>
    %g = tensor.empty() : tensor<i64>
    %0, %1 = linalg.generic {{indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, \
affine_map<(d0) -> ()>], iterator_types = ["reduction"]}} ins(%a : tensor<4xi64>) \
outs(%e, %g : tensor<i64>, tensor<i64>) {{
    ^bb0(%x: i64, %y: i64, %z: i64):
      linalg.yield %x, %y : i64, i64
    }} -> {results}
    func.return %0 : tensor<i64>
  }}
}}"""


def _parse(results: str) -> None:
    from xdsl.context import Context
    from xdsl.parser import Parser
    from xdsl.universe import Universe

    ctx = Context(allow_unregistered=True)
    for name, factory in Universe.get_multiverse().all_dialects.items():
        ctx.register_dialect(name, factory)
    Parser(ctx, _MODULE.format(results=results), "reproducer").parse_module()


def test_multi_result_generics_are_semantically_supported():
    """The unparenthesised spelling parses, so nothing about two results is unsupported."""
    pytest.importorskip("xdsl")
    _parse("tensor<i64>, tensor<i64>")


def test_the_parenthesised_result_list_is_still_rejected():
    """⚠️ WHEN THIS FAILS, UPSTREAM IS FIXED — delete this file and re-measure smolvla.

    Asserting a limitation is only useful if it is checked, because the alternative is a comment that
    stays true-looking for a year after it stopped being true.
    """
    pytest.importorskip("xdsl")
    from xdsl.utils.exceptions import ParseError

    with pytest.raises(ParseError):
        _parse("(tensor<i64>, tensor<i64>)")


def test_the_capture_that_this_costs_is_named():
    """The consequence, kept next to the cause: one construct, one model's whole routing evidence."""
    from merlin.common.paths import artifacts_dir

    capture = artifacts_dir() / "recaptures" / "smolvla_fp32_consistent" / "model.mlir"
    if not capture.is_file():
        pytest.skip("smolvla is not captured on this host")
    from merlin.targetgen.model_coverage import load_module
    from xdsl.utils.exceptions import ParseError

    with pytest.raises(ParseError):
        load_module(capture)
