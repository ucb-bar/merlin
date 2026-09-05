"""A frozen contract must reach its analyzer, or be refused by name -- never skipped."""
from __future__ import annotations

import sys

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))

import perf_affine_claim as AF  # noqa: E402
import perf_claim_dispatch as D  # noqa: E402


def _descriptor(analyzer, family="PM"):
    return {"name": "PM00_m16n16", "inputs": [], "performance": {
        "family": family, "claim": "PREDICTS", "acceptance": {"analyzer": analyzer}}}


def test_the_affine_analyzer_is_registered_and_reachable():
    assert AF.ANALYZER in D._registry(), "PM/PV contracts would be inert without this"


def test_an_unknown_analyzer_is_refused_by_name_not_skipped():
    out = D.analyze([_descriptor("nobody.such/v9")], [{"capsule": "x"}])
    assert out["verdict"] == D.REFUSED
    assert out["declared_analyzer"] == "nobody.such/v9"
    assert "nobody.such/v9" in out["reason"]
    assert AF.ANALYZER in out["registered"]


def test_a_cohort_that_disagrees_about_its_analyzer_is_refused():
    out = D.analyze([_descriptor(AF.ANALYZER), _descriptor("other/v1")], [{"capsule": "x"}])
    assert out["verdict"] == D.REFUSED
    assert "one frozen analyzer" in out["reason"]


def test_a_descriptor_with_no_contract_is_refused():
    out = D.analyze([{"name": "x", "performance": {"family": "PM", "claim": "PREDICTS"}}], [{}])
    assert out["verdict"] == D.REFUSED


def test_dispatch_reaches_the_affine_analyzer_and_returns_its_verdict():
    # a deliberately incomplete cohort: the point is that the AFFINE analyzer answered, not dispatch
    out = D.analyze([_descriptor(AF.ANALYZER)], [{"capsule": "PM00_m16n16"}])
    assert out["declared_analyzer"] == AF.ANALYZER
    assert out["verdict"] == AF.REFUSED          # its own refusal, reached through dispatch
    assert "reason" in out
