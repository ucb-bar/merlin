"""Automatic continuation controls; synthetic broker state grants no authority."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import retained_revisions as R
from merlin_experiments.phase2 import search_session as S
from merlin_experiments.phase2.broker import Broker
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.search_coordinator import ComponentSearchCoordinator, SearchSettings
from test_search_coordinator import diagnostic_coordinator


def automatic(**values):
    return SearchSettings(continue_author_session=True, campaign_deadline_unix=12_200,
                          soft_seconds=14_400, hard_seconds=14_400,
                          final_reserve_seconds=1_200, **values)


@pytest.fixture
def clock(monkeypatch):
    value = dict(mono=1_000, wall=5_000)
    monkeypatch.setattr(S.time, "monotonic", lambda: value["mono"])
    monkeypatch.setattr(S.time, "time", lambda: value["wall"])
    return value


def test_absolute_campaign_deadline_includes_setup_and_never_resets_on_continuation(clock):
    settings = automatic()
    assert S.stage_deadline(settings, 14_400) == 8_200
    assert S.search_deadlines(settings, 1_000) == (8_200, 5_800)
    clock.update(mono=2_000, wall=6_000)
    assert S.stage_deadline(settings, 14_400) == 8_200
    assert S.search_deadlines(settings, 2_000) == (8_200, 5_800)


def test_finalization_at_announced_reserve_boundary_still_has_drain_budget(clock):
    search = diagnostic_coordinator([])
    search.settings = automatic()
    search._campaign_deadlines = S.search_deadlines(search.settings, search.started)
    drained = []
    search.pool.drain = lambda **kw: drained.append(kw["timeout_s"])
    clock.update(mono=5_800, wall=9_800)
    assert search.status()["stop_reason"] == "final drain and verification reserve reached"
    search.finalize(timeout_s=1_000)
    assert drained == [1_000]
    assert search.status()["finalized"]


def test_prompt_cutoffs_remain_fixed_when_setup_consumes_time(clock):
    before = S.timing_projection(automatic())
    clock.update(mono=2_000, wall=6_000)
    after = S.timing_projection(automatic())
    for name, expected in (("exploration", 9_800), ("author_return", 11_000), ("campaign", 12_200)):
        assert before["deadlines"][name]["unix"] == after["deadlines"][name]["unix"] == expected
        assert after["deadlines"][name]["remaining_seconds"] == before["deadlines"][name]["remaining_seconds"]-1000
    assert "FINAL turn message" in after["turn_policy"]
    assert S.timing_projection(SearchSettings()) is None


def test_automatic_search_refuses_early_finalization(clock):
    search = diagnostic_coordinator([])
    search.settings = automatic()
    with pytest.raises(StageGateError, match="continues until"):
        search.finalize(timeout_s=10)


@pytest.mark.parametrize("which", ["wall", "round"])
def test_automatic_stage_cannot_cut_off_its_finalization_turn(which):
    values = dict(wall_seconds=14_400, round_seconds=14_400)
    values[which + "_seconds"] = 10_800
    with pytest.raises(StageGateError, match="reserve"):
        S.require_stage_budget(automatic(), **values)
    S.require_stage_budget(automatic(), 14_400, 14_400)


def test_original_noncontinuing_stage_retains_its_soft_budget():
    S.require_stage_budget(SearchSettings(), 14_400, 14_400)


@pytest.mark.parametrize("value", [0, -1, True, "future", float("nan"), float("inf"), 10**1_000])
def test_invalid_shared_deadlines_refuse(value):
    with pytest.raises(StageGateError):
        SearchSettings(campaign_deadline_unix=value).verify()


def test_automatic_continuation_requires_an_explicit_shared_cap():
    with pytest.raises(StageGateError):
        SearchSettings(continue_author_session=True).verify()


def test_automatic_continuation_requires_two_usable_reserves():
    with pytest.raises(StageGateError):
        SearchSettings(continue_author_session=True, campaign_deadline_unix=12_200,
                       soft_seconds=2400, hard_seconds=2400, final_reserve_seconds=1200).verify()


def control(monkeypatch, tmp_path):
    state = dict(finalized=False, selected=None, ranked=[])
    coordinator = object.__new__(ComponentSearchCoordinator)
    coordinator.settings = automatic()
    coordinator.status = lambda: state
    evidence = dict(all_required_succeeded=False)
    inventory = dict(sha256="a" * 64)
    coordinator.pool = SimpleNamespace(revisions=SimpleNamespace(
        max_bytes=1024, max_files=10, get=lambda _: SimpleNamespace(inventory=dict(sha256="a" * 64))))
    broker = object.__new__(Broker)
    broker.workflow = SimpleNamespace(search_coordinator=coordinator, verify_receipts=lambda *a, **kw: evidence)
    broker.actions = {}
    broker.target_experiment, broker.candidate = object(), tmp_path / "candidate"
    broker.receipt_path = tmp_path / "receipts.jsonl"
    monkeypatch.setattr(S, "audit_codex_transcript", lambda *a: dict(clean=True))
    monkeypatch.setattr(S, "hash_tree", lambda *a: dict(sha256="a" * 64))
    monkeypatch.setattr(R, "source_inventory", lambda *a, **kw: inventory)
    options = S.driver_options(broker, stage_root=tmp_path, round_index=0)
    return options, state, evidence, broker, inventory


def test_completed_turn_resumes_until_in_lineage_selection_and_required_receipts(monkeypatch, tmp_path):
    options, state, evidence, _, _ = control(monkeypatch, tmp_path)
    stop = options["continuation_stop"]
    assert options["continue_session"] is True
    assert "search-status" in options["continuation_prompt"]
    assert "agent_selfcheck" not in options["continuation_prompt"]
    assert stop() is None
    state.update(finalized=True, ranked=[{"source_sha256": "a" * 64}])
    assert stop() is None  # Finalization still needs search-select in the same author.
    state["selected"] = {"source_sha256": "a" * 64, "revision_id": "selected"}
    assert stop() is None  # Selection still needs the original required feedback.
    evidence["all_required_succeeded"] = True
    assert stop() == "automatic search finalization complete"


def test_a_diagnostic_finalization_stops_as_incomplete(monkeypatch, tmp_path):
    options, state, _, _, _ = control(monkeypatch, tmp_path)
    state["finalized"] = True
    assert "incomplete" in options["continuation_stop"]()


def test_receipt_refusal_returns_to_the_same_author(monkeypatch, tmp_path):
    options, state, _, broker, _ = control(monkeypatch, tmp_path)
    state.update(finalized=True, selected={"source_sha256": "a" * 64, "revision_id": "selected"})
    def refusal(*args, **kwargs):
        raise StageGateError("required feedback not complete")
    broker.workflow.verify_receipts = refusal
    assert options["continuation_stop"]() is None


@pytest.mark.parametrize("during_verification", [False, True])
def test_selected_source_drift_cannot_complete_the_campaign(monkeypatch, tmp_path, during_verification):
    options, state, evidence, broker, inventory = control(monkeypatch, tmp_path)
    state.update(finalized=True, selected={"source_sha256": "a" * 64, "revision_id": "selected"})
    evidence["all_required_succeeded"] = True
    def drift(*args, **kwargs):
        inventory["sha256"] = "b" * 64
        return evidence
    if during_verification:
        broker.workflow.verify_receipts = drift
    else:
        drift()
    assert options["continuation_stop"]() is None
