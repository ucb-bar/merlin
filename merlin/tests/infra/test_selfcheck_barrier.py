"""The self-check's barrier tier must be one the target actually produces.

The barrier was `max(adapters)` — a LEXICOGRAPHIC max over tier names. Where that resolved to a tier
the target never runs (atlas: barrier L4, results carrying only {L0 skipped, L1 skipped, L2 pass}),
`tiers.get(barrier)` returned None and EVERY capsule scored failed regardless of its verdict. Measured
live: 110 consecutive self-checks reported 0/11 while the operator grade of the same submission was
10/11. An agent told it fails everything rewrites working code, so the run measures the harness rather
than the model — and a per-target 0/N from such a run is not a model result.

The fallback is deliberately narrow: it considers only tiers that produced a real verdict, never treats
`skipped` as a barrier, and the caller still requires `status == "pass"`, so it cannot promote a
genuine failure.
"""
from __future__ import annotations

import pytest


def score(tiers: dict, status: str, declared: str) -> tuple[bool, str]:
    """Mirror of agent_selfcheck's barrier resolution (kept in sync by the tests below)."""
    bar = tiers.get(declared)
    used = declared
    if bar is None:
        ran = [k for k, v in tiers.items() if v not in (None, "skipped")]
        if ran:
            used = max(ran)
            bar = tiers.get(used)
    return (status == "pass") and (bar == "pass"), used


ATLAS = {"L0": "skipped", "L1": "skipped", "L2": "pass"}
GEMMINI = {"L0": "pass", "L1": "pass", "L2": "pass", "L3": "pass"}


def test_a_barrier_the_target_never_runs_does_not_fail_a_passing_capsule():
    ok, used = score(ATLAS, "pass", "L4")
    assert ok and used == "L2", "atlas passes at L2; a declared L4 barrier must not veto it"


def test_the_declared_barrier_still_wins_when_it_ran():
    ok, used = score(GEMMINI, "pass", "L3")
    assert ok and used == "L3", "a barrier the grade produced must be used as-is"


def test_a_failing_barrier_is_still_a_failure():
    ok, _ = score({"L0": "pass", "L1": "pass", "L2": "pass", "L3": "fail"}, "fail", "L3")
    assert not ok


def test_the_fallback_cannot_promote_a_failed_capsule():
    """status must still be pass — the fallback only changes WHICH tier is read."""
    ok, _ = score(ATLAS, "fail", "L4")
    assert not ok


def test_a_skipped_tier_is_never_used_as_the_barrier():
    ok, used = score({"L0": "skipped", "L1": "skipped"}, "pass", "L4")
    assert not ok and used == "L4", "with no tier actually run there is nothing to fall back to"


@pytest.mark.parametrize("declared", ["L3", "L4", "L5"])
def test_atlas_is_gradeable_whatever_barrier_is_declared(declared):
    ok, used = score(ATLAS, "pass", declared)
    assert ok and used == "L2"


def test_the_helper_matches_the_shipped_implementation():
    """Guard against the mirror above drifting from agent_selfcheck."""
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin/experiments/capsule_bench/harness/agent_selfcheck.py").read_text()
    assert 'ran = [k for k, v in tiers.items() if v not in (None, "skipped")]' in src
    assert "bar_used = max(ran)" in src
