"""The free screening action must not advertise an ordering verdict it does not have.

Two separate failures live here, and both shipped:

* the action's registry description promised "a differential verdict". The differential had been
  removed and replaced by ``not_attempted``, so the agent was told to expect a verdict that no
  longer existed. A stale description is not cosmetic -- it is what the agent plans against.
* every cheap signal readable from a command buffer was scored against the cycle oracle on the
  comparison the search actually makes, held out by workload. NONE beat chance: the dependence-graph
  makespan landed on exactly 0.500, a raw command count on 0.470, and a tile-pressure heuristic
  reading 0.728 overall points BACKWARDS at 0.273 on one workload with 33 decided pairs. Exposing
  any of them as a ranker would be worse than exposing nothing, because a signal below chance is
  followed. This pins the refusal so a later change has to face the numbers rather than a memory.
"""
from __future__ import annotations

import sys

from merlin.common.paths import merlin_dir
from merlin.perf import rank_validation

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import perf_agent_stage as PAS  # noqa: E402
import perf_prompt as PP  # noqa: E402


def test_no_recorded_signal_beats_chance():
    """Every number in the shipped record must actually be at or below the bar it claims to fail."""
    for signal, (agreed, decided, why) in PAS.ORDERING_EVIDENCE["agreement"].items():
        assert decided > 0, f"{signal} claims a rate on nothing decided"
        rate = agreed / decided
        beats_chance = rate > rank_validation.CHANCE
        assert (not beats_chance) or decided < 40 or "backwards" in why, (
            f"{signal} records {rate:.3f} over {decided} decided pair(s), which beats chance with "
            f"enough evidence -- it must then be exposed rather than refused, or the record is stale")


def test_the_action_refuses_to_order_and_says_so_with_numbers():
    assert PAS.ORDERING_REFUSED.startswith("refused")
    evidence = PAS.ORDERING_EVIDENCE
    assert evidence["held_out_pairs"] > 0 and evidence["held_out_workloads"] > 1, (
        "a refusal measured on one workload is not a refusal that generalises")
    assert evidence["artifact"].endswith(".json")


def test_the_registry_description_promises_only_what_is_returned():
    """It must not name a verdict the action stopped returning."""
    source = (_SCRIPTS / "perf_agent_stage.py").read_text(encoding="utf-8")
    start = source.index("ANALYSIS_ACTION, (_HOST_ANALYSIS_SENTINEL")
    description = source[start:start + 900]
    assert "differential verdict" not in description, (
        "the differential was removed and returns not_attempted; the description must not promise it")
    assert "cannot certify" in description or "never certify" in description, (
        "a screen that may only eliminate has to say so where the agent reads it")


def test_the_prompt_documents_the_free_action_and_its_limit():
    source = (_SCRIPTS / "perf_prompt.py").read_text(encoding="utf-8")
    assert PAS.ANALYSIS_ACTION in source, "the prompt never mentions the free screening action"
    assert "no oracle time" in source, "the agent must be told the screen is free, or it will not use it"
    assert "never certify" in source, (
        "the screen may eliminate and may never certify; the prompt has to carry the asymmetry")


def test_every_verdict_the_harness_can_emit_is_explained_to_the_agent():
    """A field the agent receives but is never told about is a field it cannot act on.

    `verdict` is the per-member "is this one finished" signal -- the only principled basis for giving
    up on a capsule -- and it reached the agent for a whole campaign with no mention anywhere in the
    prompt. Pin every value the module can emit, so adding a sixth cannot silently go undocumented.
    """
    import perf_capsule_verdict as CV  # noqa: PLC0415

    prompt_source = (_SCRIPTS / "perf_prompt.py").read_text(encoding="utf-8")
    emitted = {CV.NO_HEADROOM, CV.IMPROVED, CV.HEADROOM_OPEN, CV.REGRESSED, CV.REFUSED}
    missing = sorted(v for v in emitted if v not in prompt_source)
    assert not missing, f"verdict value(s) the agent is never told about: {missing}"
    assert "verdict_reason" in prompt_source
