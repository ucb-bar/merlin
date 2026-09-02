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


def score(tiers: dict, status: str, declared: str, declared_ran: bool = False,
          required: set[str] | None = None,
          mandatory: bool = True, blocked_by_selection: bool = False) -> tuple[bool, str]:
    """Mirror of agent_selfcheck's barrier resolution (kept in sync by the tests below).

    `declared_ran` is the whole-corpus fact: did the declared tier produce a verdict for ANY capsule?
    It is what separates "this capsule fell short of a real bar" from "the bar does not exist here".
    """
    bar = tiers.get(declared)
    used = declared
    if bar is None:
        required = set(tiers) if required is None else required
        ran_required = [k for k, v in tiers.items()
                        if k in required and v not in (None, "skipped", "unavailable")]
        deeper_required = [k for k in ran_required if k > declared]
        if deeper_required:
            used = max(deeper_required)
            bar = tiers.get(used)
        elif not declared_ran:
            ran = [k for k, v in tiers.items() if v not in (None, "skipped")]
            if ran:
                used = max(ran)
                bar = tiers.get(used)
    # SCORING, as shipped. A capsule is CERTIFIED when it passed and the bar itself passed. It is
    # SCREENED when the bar could not run because this oracle selection supplies no adapter for it,
    # every tier that DID run was clean, and nothing failed -- absence of evidence about the backend,
    # not evidence against it. `passed` is either; only `certified` clears a mandatory tier, so a
    # screen can eliminate but can never certify.
    certified = (status == "pass") and (bar == "pass")
    screened = (not certified) and blocked_by_selection and bar in ("pass", None) \
        and all(v == "pass" for v in tiers.values() if v not in (None, "skipped", "unavailable"))
    return (certified or screened), used


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


def test_a_capsule_that_fell_short_of_a_REAL_barrier_still_fails():
    """The fallback must not become a leniency hole.

    Where the target genuinely produces the declared tier, a capsule missing it did not reach the bar.
    Scoring it against a weaker tier would report a pass on weaker evidence than the barrier demands --
    the same "plausible wrong number" failure as the original bug, only inverted. Measured on the live
    gemmini run: barrier_used came back ['L0', 'L2'], i.e. capsules were being passed at L0 on a target
    whose bar is L2.
    """
    short = {"L0": "pass", "L1": "pass"}          # L2 never reached for THIS capsule
    ok, used = score(short, "pass", "L2", declared_ran=True)
    assert not ok, "L2 runs on this target, so a capsule that never reached it has not met the bar"
    assert used == "L2", "the row must still report the bar it was held to"


def test_a_model_capsule_that_ran_a_deeper_required_tier_uses_its_own_barrier():
    """A per-op screen tier is not necessarily part of a whole-model result.

    Measured on gemmini's mixed suite: ordinary op capsules made the corpus-wide ``declared_ran``
    true for L2, while the M3 whole-model capsule legitimately reported
    ``{L0: skipped, L1: skipped, L3: pass}``.  Holding that capsule to the unrelated L2 row produced
    ``pass: false, barrier_status: null`` beside a top-level and numeric pass.  Reaching the deeper L3
    required tier is stronger evidence, not a fallback to a weaker screen.
    """
    model = {"L0": "skipped", "L1": "skipped", "L3": "pass"}
    ok, used = score(model, "pass", "L2", declared_ran=True)
    assert ok, "a deeper required tier that actually passed must certify the model capsule"
    assert used == "L3"


def test_a_deeper_advisory_tier_cannot_replace_the_real_barrier():
    tiers = {"L0": "pass", "L1": "pass", "L3": "pass"}
    ok, used = score(tiers, "pass", "L2", declared_ran=True, required={"L0", "L1"})
    assert not ok
    assert used == "L2"


def test_the_fallback_survives_when_the_tier_exists_nowhere():
    """The atlas case is unchanged: no capsule anywhere produced L4, so the bar is not real."""
    ok, used = score(ATLAS, "pass", "L4", declared_ran=False)
    assert ok and used == "L2"


def test_the_helper_matches_the_shipped_implementation():
    """Guard against the mirror above drifting from agent_selfcheck."""
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin/experiments/capsule_bench/harness/agent_selfcheck.py").read_text()
    # The BARRIER RESOLUTION lines (unchanged), then the SCORING lines as shipped. The scoring moved
    # from a mandatory/did-not-run test to certified-vs-screened during branch integration; the mirror
    # above tracks the shipped form, and this guard is what stops the two drifting apart again.
    assert "bar_used = max(ran)" in src
    assert 'certified = (d.get("status") == "pass") and (bar == "pass")' in src
    assert "screened = (not certified) and _blocked_by_selection" in src
    assert "passed = certified or screened" in src
    # the whole-corpus gate: without it the fallback is a leniency hole
    assert "if bar is None:" in src
    assert "_deeper_required" in src
    assert 'bool((v or {}).get("mandatory"))' in src
    assert "elif not _declared_ran:" in src
    assert "_declared_ran = True" in src
