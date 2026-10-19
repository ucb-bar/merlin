"""Neutral retained-queue transport; no target qualification or native job runs."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import test_checked_source_model_diagnostic as SOURCE
import test_ordinary_fast_feedback as NORMAL
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import public_program_measurement as PUBLIC
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen.compiler_pass_selection import scoped_compiler_passes


@pytest.fixture
def source_selection(tmp_path, monkeypatch):
    return SOURCE.selected.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def ordinary_inputs(tmp_path, monkeypatch, request):
    return NORMAL.ordinary_inputs.__wrapped__(tmp_path, monkeypatch, request)


@pytest.fixture
def queued(ordinary_inputs, source_selection, tmp_path, monkeypatch):
    # The normal fixture's original formal admission remains intact. A neutral
    # transport view exercises FAST directly; it is not a new policy issuer.
    passes, selected, issuer, _, capsule = source_selection
    calls = []
    state = {"capsule": capsule, "failure": None}
    monkeypatch.setattr(FAST.CR, "load_capsule", lambda *args, **kwargs: copy.deepcopy(state["capsule"]))
    monkeypatch.setattr(FAST.CR, "simulator_adapter", lambda *args: "original-functional-fixture")
    monkeypatch.setattr(FAST.PC, "package_sandbox_policy", lambda *args: "original-box-fixture")
    monkeypatch.setattr(FAST.PC, "boxed_entrypoints", lambda *args: nullcontext())
    monkeypatch.setattr(FAST, "compiled_observation", lambda *args, **kwargs: {"status": "UNKNOWN"})
    monkeypatch.setattr(FAST, "observe_passed_census", lambda *args, **kwargs: {"status": "UNKNOWN"})

    def public(*args, **kwargs):
        raise AssertionError("checked-source diagnostics must not fabricate a public placement")

    monkeypatch.setattr(PUBLIC, "selection", public)

    def run(actual, package, **kwargs):
        assert kwargs["checked_source_diagnostic"] is selected
        assert "model_execution_selection" not in kwargs
        selected.verify_for(actual, target="fixture")
        calls.append((actual, package, kwargs))
        if state["failure"] is not None:
            return {
                "status": "incomplete",
                "failure": state["failure"],
                "tiers": {"L0": {"status": "fail"}, "L1": {"status": "fail"}, "L2": {"status": "not_run"}},
            }
        return {"status": "incomplete", "tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")}}

    monkeypatch.setattr(FAST.CR, "run_capsule", run)
    normal = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = NORMAL.revisions(normal, tmp_path)
    policy = SimpleNamespace(
        validate_search_inputs=normal.validate_search_inputs,
        edit_authority=normal.edit_authority,
        search_corpus=normal.search_corpus,
        search_closure_sha256=normal.search_closure_sha256,
        target_experiment=normal.target_experiment,
        feedback_evaluator=normal.feedback_evaluator,
        _contract_root=normal._contract_root,
        _ordinary_screen=None,
        _checked_source_diagnostic=selected,
        _compiler_pass_selection=passes,
    )
    policy._test_state = state
    return policy, passes, selected, issuer, candidate, baseline, calls


def execute(queued, output, *, selected=True):
    policy, passes, _, _, candidate, baseline, _ = queued
    members = tuple(sorted(policy.validate_search_inputs())) if selected else None
    with scoped_compiler_passes(passes):
        return FAST.evaluate_retained(
            policy,
            candidate,
            baseline,
            work_root=output,
            call_index=0,
            timeout_s=30,
            workers=1,
            selected_members=members,
        )


def test_retained_selected_queue_carries_same_live_owner_and_never_scores(queued, tmp_path):
    policy, _, _, _, candidate, baseline, calls = queued
    # An archive owner must not be asked to score this explicit diagnostic.
    policy._ordinary_screen = object.__new__(FAST.OrdinaryArchiveScreen)
    result = execute(queued, tmp_path / "queued")
    assert {package for _, package, _ in calls} == {candidate.source, baseline.source}
    assert result["promotion"] == "DIAGNOSTIC_ONLY"
    assert result["screening"]["status"] == "UNKNOWN"
    for cell in result["cells"]:
        for arm in ("baseline", "candidate"):
            assert cell[arm]["functional_status"] == "PASS"
            assert cell[arm]["pending_tiers"] == ["L3"]
            assert cell[arm]["timing_cycles"] is None
            assert cell[arm]["archive_counts"]["status"] == "UNKNOWN"
    # Private evidence, not the public feedback transport, carries the record.
    evidence = json.loads(next((tmp_path / "queued").rglob("ordinary_observation.json")).read_bytes())
    assert evidence["checked_source_diagnostic"]["promotion_authority"] == "NONE"
    assert "checked_source_diagnostic" not in json.dumps(result)


def test_complete_scored_action_cannot_inherit_source_diagnostic(queued, tmp_path):
    with pytest.raises(StageGateError, match="selected diagnostic-only"):
        execute(queued, tmp_path / "not-selected", selected=False)
    assert not queued[-1]


@pytest.mark.parametrize("fault", ["saved_json", "foreign_passes", "stale_owner", "hidden"])
def test_source_owner_substitution_or_refusal_never_falls_back(queued, tmp_path, monkeypatch, fault):
    policy, passes, selected, issuer, *_ = queued
    if fault == "saved_json":
        policy._checked_source_diagnostic = selected.record()
    elif fault == "foreign_passes":
        policy._compiler_pass_selection = copy.copy(passes)
    elif fault == "stale_owner":
        issuer.refuse_on_check = issuer.checks + 1
    else:
        policy._test_state["capsule"] = dict(
            kind="model", label="hidden", required_oracle_tiers=["L0", "L1", "L2", "L3"]
        )
    with pytest.raises((ValueError, StageGateError)):
        execute(queued, tmp_path / fault)
    assert not queued[-1]


@pytest.mark.parametrize("failure", ["source bytes mismatch", "contraction shape mismatch"])
def test_original_verifier_refusal_propagates_without_timing_credit(queued, tmp_path, monkeypatch, failure):
    queued[0]._test_state["failure"] = failure
    result = execute(queued, tmp_path / failure.replace(" ", "-"))
    assert result["promotion"] == "DIAGNOSTIC_ONLY"
    assert all(
        row[arm]["functional_status"] == "UNKNOWN_OR_FAILURE"
        for row in result["cells"]
        for arm in ("baseline", "candidate")
    )


def test_absent_optin_preserves_default_placement_refusal(queued, tmp_path):
    queued[0]._checked_source_diagnostic = None
    with pytest.raises(AssertionError, match="public placement"):
        execute(queued, tmp_path / "default")


def test_optin_does_not_bypass_original_formal_phase1_gate(ordinary_inputs, source_selection, monkeypatch):
    def refuse(*args, **kwargs):
        raise StageGateError("original formal Phase 1 qualification is pending")

    monkeypatch.setattr(OS.FI, "require_formal_qualification", refuse)
    with pytest.raises(StageGateError, match="original formal Phase 1"):
        OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments, checked_source_diagnostic=source_selection[1])


@pytest.mark.parametrize("deadline,expected", [(None, 120), (15, 5)])
def test_original_metadata_admission_has_bounded_queue_wait(source_selection, monkeypatch, deadline, expected):
    from contextlib import contextmanager

    helper = FAST.CHECKED_SOURCE
    passes, selected, *_ = source_selection
    waits = []

    @contextmanager
    def lease(*, timeout_s):
        waits.append(timeout_s)
        yield

    monkeypatch.setattr(helper, "monotonic", lambda: 10)
    monkeypatch.setattr(helper, "shared_compute_pool", lease)
    assert helper.record(selected, passes, deadline=deadline)["promotion_authority"] == "NONE"
    assert waits == [expected]


def test_expired_or_overrun_metadata_cannot_expose_a_record(source_selection, monkeypatch):
    helper = FAST.CHECKED_SOURCE
    passes, selected, *_ = source_selection
    monkeypatch.setattr(helper, "monotonic", lambda: 10)
    with pytest.raises(StageGateError, match="aggregate deadline"):
        helper.record(selected, passes, deadline=9)
    ticks = iter((10, 10, 20))
    monkeypatch.setattr(helper, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(helper, "shared_compute_pool", lambda **_kwargs: nullcontext())
    with pytest.raises(StageGateError, match="aggregate deadline"):
        helper.record(selected, passes, deadline=15)


def test_member_selection_uses_remaining_feedback_budget(queued, monkeypatch):
    from contextlib import contextmanager

    helper = FAST.CHECKED_SOURCE
    policy, passes, selected, *_ = queued
    waits = []

    @contextmanager
    def lease(*, timeout_s):
        waits.append(timeout_s)
        yield

    monkeypatch.setattr(helper, "monotonic", lambda: 30)
    monkeypatch.setattr(helper, "shared_compute_pool", lease)
    with scoped_compiler_passes(passes):
        assert helper.for_capsule(policy, policy._test_state["capsule"], deadline=32) is selected
    assert waits == [2]
