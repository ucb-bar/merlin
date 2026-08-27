"""A screen may eliminate; it may never certify — and it may never score its own blindness as failure.

A capsule declares the oracle tiers it requires. An oracle SELECTION supplies adapters for some subset of
them. A required tier with no adapter under the current selection did not run, and a tier that did not run
is absence of evidence about the backend, not evidence against it.

``NOT_RUN_IS_NOT_PASS`` is the correct rule for a CERTIFYING grade: it stops an unrun tier being read as a
pass. Applied as a FAILURE inside a cheap screen it scores the submission for the engine the caller
picked. Measured: a capsule grading ``{L0 pass, L1 pass, L2 pass, L3 unavailable}`` scored 0, and one run
spent 325 consecutive self-checks at zero while the certifying grade of the very same submission was
22/25. An agent told it fails everything rewrites working code.

Two properties are pinned here, and they pull in opposite directions on purpose:

  * a capsule blocked ONLY by a tier this selection cannot reach counts as passed-at-this-selection and is
    reported as ``screened_only`` -- not a failure;
  * ``all_pass`` keys on CERTIFICATION, so no screen can ever declare done, and a capsule that genuinely
    failed a tier that DID run is never reclassified.

The rule is derived from each capsule's own declared tiers and the adapters present. No engine and no
target is named in it.
"""
from __future__ import annotations

import ast

from merlin.common.paths import merlin_dir

_SC = merlin_dir() / "experiments/capsule_bench/harness/agent_selfcheck.py"


def _src() -> str:
    return _SC.read_text(encoding="utf-8")


def test_a_tier_the_selection_cannot_reach_is_not_scored_as_failure():
    s = _src()
    assert "_blocked_by_selection" in s
    assert 'NOT_RUN_IS_NOT_PASS' in s and 'tier_status") == "unavailable"' in s, (
        "the screened case must key on the tier being UNAVAILABLE, not on any engine name")


def test_only_a_capsule_whose_every_run_tier_passed_may_be_screened():
    """Anti-inflation: a capsule that failed a tier that DID run stays failed."""
    s = _src()
    assert "_ran_clean" in s
    assert 'all(v == "pass" for v in _ran.values())' in s, (
        "screened must require that every tier which actually ran passed")


def test_all_pass_requires_certification_not_merely_screening():
    s = _src()
    assert '"all_pass": ncert == n and n > 0' in s, (
        "a screen must never be able to declare done -- all_pass keys on the certified count")


def test_the_report_separates_measured_from_certified():
    s = _src()
    for key in ('"n_passed"', '"n_certified"', '"n_screened_only"'):
        assert key in s, f"{key} must be reported so a reader can tell the two apart"


def test_the_rule_names_no_engine_and_no_target():
    """Target-agnostic by construction: the classification reads declared tiers and adapter presence.

    Scoped to the classification STATEMENTS, not the enclosing function -- the CLI legitimately names the
    engines it can select; the rule that decides screened-vs-failed must not."""
    s = _src()
    start = s.index("_blocked_by_selection = (")
    end = s.index("nscreened += int(screened)", start)
    block = s[start:end].lower()
    for banned in ("spike", "verilator", "vcs", "cyclotron", "gsim"):
        assert banned not in block, (
            f"the screened/failed decision must not branch on the engine name {banned!r}; "
            f"it must read the capsule's declared tiers and whether an adapter was present")
    assert "tiers" in block and "unavailable" in block, "the rule must key on declared tiers + availability"
