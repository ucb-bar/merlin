"""An axis that requires nothing because it could not be derived is UNDETERMINABLE, not satisfied.

`required_regimes` already refuses honestly when it cannot resolve the operand store: "that is 'we do
not know', never 'nothing is required'". But `uncovered_regimes` dropped that string and computed
`n_required` from an empty mapping, so the summary printed `0 / 0 required regime(s)` — which reads
exactly like a closed axis. Measured 2026-09-01: mx_gemmini and radiance both printed 0/0 while their
operand-store capacity was absent, and any capsule authored against that axis would have proved nothing.

This is the same failure shape as every other one found that day: absence of evidence presenting itself
as evidence.
"""
from __future__ import annotations

from merlin.targetgen import memory_regime as MR


def test_an_underivable_axis_is_undeterminable_not_zero_of_zero():
    required = {"by_regime": {}, "region_counts": {}, "captures_unreadable": {},
                "why": "'t' declares no operand-store capacity we can derive, so no regime is required "
                       "of the corpus -- that is 'we do not know', never 'nothing is required'"}
    got = MR.uncovered_regimes(required, {"by_regime": {}, "capacity_rows": None})
    assert got["status"] == MR.UNDETERMINABLE_AXIS
    assert got["n_required"] is None and got["n_covered"] is None, (
        "an underivable axis must not report a 0/0 ratio, which reads as satisfied")
    assert "we do not know" in got["why"]


def test_a_derived_axis_still_reports_a_ratio():
    required = {"by_regime": {"fits_double": ["c0"], "spills": ["c1"]}}
    got = MR.uncovered_regimes(required, {"by_regime": {"fits_double": ["A0"]},
                                          "capacity_rows": 16384})
    assert got["status"] == "ok"
    assert got["n_required"] == 2 and got["n_covered"] == 1
    assert got["uncovered"] == ["spills"]


def test_an_empty_axis_with_no_reason_is_not_silently_undeterminable():
    """Only a deriver that EXPLAINED itself may claim undeterminable; otherwise report the ratio."""
    got = MR.uncovered_regimes({"by_regime": {}}, {"by_regime": {}, "capacity_rows": 16384})
    assert got["status"] == "ok", "without a recorded reason there is nothing to report as unknown"
    assert got["n_required"] == 0


def test_the_gate_treats_a_vacuous_axis_as_a_failure():
    """Wiring: --fail-on-uncovered must not let a target through on a vacuous axis."""
    from pathlib import Path

    from merlin.common.paths import repo_root
    src = (repo_root() / "build_tools/scripts/check_conformance_coverage.py").read_text(encoding="utf-8")
    assert 'status") == "undeterminable"' in src, (
        "the gate does not look for an undeterminable axis")
    assert "axis_undeterminable" in src, (
        "a vacuous axis is not ratchetable, so a known-missing fact cannot be carried deliberately")
    assert "bad or vacuous" in src, "a vacuous axis does not affect the exit status"


def test_the_conformance_consumer_forwards_the_reason():
    """The `why` existed and was thrown away between the spec and the report.

    The spec writer persists `memory_mapping.why` -- the deriver's own account of why it could not
    resolve the operand store -- but `uncovered()` forwarded only `by_regime` and then hardcoded
    `status = "ok"`, so the verdict was clobbered even when the reason was present. Measured
    2026-09-01: mx_gemmini's spec carried the full string while its report read
    `0 / 0 required regime(s) (operand store None rows)`.
    """
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin/python/merlin/targetgen/conformance.py").read_text(encoding="utf-8")
    call = src[src.index("mgap = MR.uncovered_regimes("):][:600]
    assert '"why"' in call, "uncovered_regimes is called without the spec's recorded reason"
    assert 'mgap.setdefault("status", "ok")' in call, (
        'the consumer still hardcodes status "ok", which clobbers an UNDETERMINABLE verdict')
