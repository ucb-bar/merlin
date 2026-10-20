"""Original capsule receipts and neutral host routing, never target qualification."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_ordinary_search_policy as NORMAL
from merlin_experiments.phase2 import capsule_iteration as CI
from merlin_experiments.phase2 import capsule_ordinary_scope as C
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import search_inputs as SI
from merlin_experiments.phase2 import stage_prompt as SP
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from test_capsule_certification import selected as selected
from test_capsule_qualification_handoff import qualified as qualified
from test_qualified_capsule_iteration import configuration


@pytest.fixture
def ordinary_inputs(tmp_path, monkeypatch, request):
    root = tmp_path / "neutral"
    root.mkdir()
    return NORMAL.ordinary_inputs.__wrapped__(root, monkeypatch, request)


@pytest.fixture
def scoped(qualified, monkeypatch, tmp_path):
    s, _, record = qualified
    monkeypatch.setattr(FI, "_functional_input_snapshot", lambda base: base)
    base, owner = CI.selected_functional_run(s.run.parent, configuration(qualified), s.descriptor)
    cohort = tmp_path / "complete-pending-cohort.json"
    cohort.write_text(json.dumps({"planning_only": True, "members": ["pending-a", "pending-b"]}))
    pin = {"path": str(cohort), "sha256": sha256_file(cohort)}
    target = SimpleNamespace(path=s.descriptor, descriptor_sha256=sha256_file(s.descriptor))
    actual, scope = C.select(
        owner,
        functional_runs_root=s.run.parent,
        functional_run_id=s.run.name,
        functional_submission_sha256=record["submission_sha256"],
        target_experiment=target,
        required_cohort_declaration=pin,
    )
    assert actual.submission_dir == base.submission_dir
    return SimpleNamespace(
        scope=scope, owner=owner, base=base, target=target, cohort=cohort, pin=pin, original=qualified
    )


def test_original_qualification_and_complete_pending_declaration_remain_distinct(scoped):
    record = C.verify(scoped.scope)
    assert record["required_cohort"]["members"] == ["pending-a", "pending-b"]
    assert record["phase1"]["public_total"] == 2
    assert record["full_models_pending"] is True
    assert record["formal_grade_complete"] is record["final_results_qualified"] is False
    assert record["promotion_authority"] == "NONE"
    with pytest.raises(ValueError, match="external full-model"):
        CI.require_final_results(record["phase1"], descriptor=scoped.target.path)


@pytest.mark.parametrize("fault", ["receipt", "compiler", "cohort", "callback", "scope_callback", "source"])
def test_every_call_reopens_original_evidence_and_rejects_mutations(scoped, monkeypatch, tmp_path, fault):
    C.verify(scoped.scope)
    if fault == "callback":
        monkeypatch.setattr(CI.FrozenQualifiedCapsulePhase1, "verify", lambda self, base: {})
    elif fault == "scope_callback":
        scoped.scope.record = lambda: {}
    elif fault == "source":
        original = C.sha256_file
        selected_path = Path(C.__file__).resolve()
        monkeypatch.setattr(C, "sha256_file", lambda p: "0" * 64 if Path(p) == selected_path else original(p))
    else:
        path = {
            "receipt": scoped.owner.qualification_path,
            "compiler": scoped.base.submission_dir / "compiler.py",
            "cohort": scoped.cohort,
        }[fault]
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((ValueError, StageGateError)):
        C.verify(scoped.scope)


def test_saved_binding_or_subclass_cannot_issue_scope(scoped):
    with pytest.raises(StageGateError, match="exact original"):
        C.select(
            SimpleNamespace(**C.verify(scoped.scope)),
            functional_runs_root=scoped.base.run_dir.parent,
            functional_run_id=scoped.base.run_dir.name,
            functional_submission_sha256=scoped.owner.submission_sha256,
            target_experiment=scoped.target,
            required_cohort_declaration=scoped.pin,
        )

    class Foreign(C.CapsuleOrdinaryScope):
        pass

    with pytest.raises(StageGateError):
        C.verify(object.__new__(Foreign))
    with pytest.raises(StageGateError):
        C.verify(object.__new__(C.CapsuleOrdinaryScope))


@pytest.fixture
def neutral_policy(scoped, ordinary_inputs, monkeypatch):
    # External certificate/functional fixtures are synthetic. Only routing is tested here;
    # scoped above separately replays real original capsule receipt fixtures.
    f = ordinary_inputs
    monkeypatch.setattr(C, "select", lambda *args, **kwargs: (f.functional, scoped.scope))
    policy = OS.prepare_ordinary_search_policy(
        **f.arguments, capsule_phase1_owner=scoped.owner, required_cohort_declaration=scoped.pin
    )
    return policy, f, scoped


def test_capsule_policy_restricts_original_pool_kinds_and_direct_dispatch(neutral_policy):
    policy, f, _ = neutral_policy
    inputs = SI.SearchInputs(policy)
    assert set(inputs.kinds) == set(C.allowed_kinds())
    # Measured ranking retains its original unavailable RTL kind. Cheap jobs cannot
    # be mistaken for measurements by the unchanged coordinator.
    assert inputs.timing_kind == "corpus-rtl-feedback"
    assert "corpus-rtl-feedback" not in inputs.kinds
    assert not any("measures certified RTL" in a.purpose for a in policy.build_registry())
    for kind in ("corpus-rtl-feedback", "corpus-selected-rtl-feedback", "component-rtl-feedback"):
        with pytest.raises(StageGateError, match="RTL"):
            C.require_kind(policy, kind)
    with pytest.raises(StageGateError, match="cannot install"):
        policy.install_native_runtime(object())
    with pytest.raises(StageGateError, match="GSIM/RTL execution"):
        policy.feedback_evaluator._execute_once(
            arm="candidate",
            package=Path("unused"),
            package_sha256="a" * 64,
            member=None,
            decision=None,
            workspace=Path("unused"),
            timeout_s=1,
        )
    assert not f.executions


def test_scope_removal_or_feedback_replacement_refuses(neutral_policy):
    policy, _, _ = neutral_policy
    policy.feedback_evaluator._capsule_development_scope = None
    with pytest.raises(StageGateError, match="lost its original scope"):
        policy.validate_search_inputs()


def test_default_formal_admission_remains_mandatory(ordinary_inputs, monkeypatch):
    calls = []

    def refuse(*args):
        calls.append(args)
        raise StageGateError("formal test refusal")

    monkeypatch.setattr(FI, "require_formal_qualification", refuse)
    with pytest.raises(StageGateError, match="formal test refusal"):
        OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    assert len(calls) == 1


def test_receipts_need_original_pool_and_completed_final_source(neutral_policy, tmp_path):
    policy, _, _ = neutral_policy
    # Even a well-shaped saved pass cannot stand in for the original live pool.
    policy.evaluation_pool = SimpleNamespace(policy=policy, status=lambda _: {"state": "completed"})
    path = tmp_path / "receipts.jsonl"
    path.write_text("")
    with pytest.raises(StageGateError, match="original live pool"):
        C.live_pool(policy)
    with pytest.raises(StageGateError):
        C.verify_receipts(policy, path, (), {"events": []}, candidate_sha256="a" * 64)


def test_prompt_keeps_pending_cohort_and_has_no_full_model_sentinel(scoped):
    inputs = SimpleNamespace(
        capsule_development=C.verify(scoped.scope),
        functional_run_id="run",
        functional_submission_sha256="a" * 64,
        functional_public_capsules=2,
        functional_hidden_capsules=1,
        submission_path="/compiler",
        submission_initial_sha256="a" * 64,
        workload_manifest="/corpus/manifest.json",
        workload_capsules_sha256="b" * 64,
        tools=(),
        allowed_paths=("compiler.py",),
        max_tool_calls=10,
        wall_budget_seconds=60,
        tool_timeout_seconds=20,
    )
    prompt = SP.render_capsule_development_prompt(inputs)
    assert "pending-a" in prompt and "pending-b" in prompt
    assert "Full-model and formal/final qualification remain pending" in prompt
    assert "gSIM" in prompt and "unscored development checkpoint" in prompt


def test_late_cohort_mutation_is_refused_after_original_verifier(scoped, monkeypatch):
    original = C.read_cohort
    calls = []

    def change_after_first_read(pin):
        value = original(pin)
        calls.append(pin)
        if len(calls) == 1:
            scoped.cohort.write_bytes(scoped.cohort.read_bytes() + b" ")
        return value

    monkeypatch.setattr(C, "read_cohort", change_after_first_read)
    with pytest.raises((ValueError, StageGateError)):
        C.verify(scoped.scope)
    assert len(calls) == 1


def test_no_automatic_finalizer_and_standard_candidate_rejects_checkpoint(scoped, tmp_path):
    from merlin_experiments.phase2 import candidate_record as R

    called = []

    def target(*args, **kwargs):
        called.append(args)
        raise AssertionError("no RTL finalizer allowed")

    guard = C.final_guard(scoped.scope, target, None, None, None, None, tmp_path, None, None)
    assert guard["status"] == "pending" and not called
    record = {"admission": {}, "development_feedback": {}}
    C.development_record(scoped.scope, record, True)
    C.verify_checkpoint(record)
    assert record["admission"]["development_checkpoint_complete"] is True
    assert record["admission"]["consumable"] is False
    with pytest.raises(StageGateError, match="unsupported schema"):
        R.validate_candidate_record(record)
    record["final_results_qualified"] = True
    with pytest.raises(ValueError):
        C.verify_checkpoint(record)


def test_default_finalizer_calls_original_guard_and_preserves_arguments(tmp_path):
    seen = []

    def guard(*args, **kwargs):
        seen.append((args, kwargs))
        return {"status": "clean"}

    assert C.final_guard(None, guard, "base", "candidate", "target", None, tmp_path, "frozen", "inputs") == {
        "status": "clean"
    }
    assert seen[0][0] == ("base", "candidate", "target")
    assert seen[0][1]["grade_root"] == tmp_path / "functional_guard_grade"


def test_cheap_jobs_never_enter_measured_ranking(neutral_policy):
    from merlin_experiments.phase2.search_coordinator import rank_measured_contenders

    policy, _, _ = neutral_policy
    result = rank_measured_contenders(
        [{"state": "completed", "kind": C.allowed_kinds()[0]}],
        roster=policy.validate_search_inputs(),
        corpus_sha256=policy.search_corpus.capsules_sha256,
        configuration_sha256=policy.feedback_evaluator.certificate.sha256,
        timing_kind=SI.SearchInputs(policy).timing_kind,
    )
    assert result["ranked"] == []


@pytest.mark.parametrize("fault", ["function", "code"])
def test_original_pool_callback_is_pinned_even_before_submissions(scoped, monkeypatch, fault):
    from merlin_experiments.phase2.evaluation_pool import ComponentEvaluationPool

    if fault == "function":
        monkeypatch.setattr(ComponentEvaluationPool, "status", lambda self, job: {})
    else:

        def changed(self, job):
            return {}

        monkeypatch.setattr(ComponentEvaluationPool.status, "__code__", changed.__code__)
    with pytest.raises(StageGateError, match="pool verifier changed"):
        C.verify(scoped.scope)


def test_default_sentinel_record_preserves_exact_original_purpose():
    sentinel = SimpleNamespace(
        capsule="whole",
        capsule_path="/whole",
        frozen_source_path="/frozen",
        capsule_sha256="a" * 64,
        required_lanes=("host", "accelerator"),
        required_tiers=("L2", "L3"),
    )
    assert C.sentinel_record(None, sentinel) == {
        "capsule": "whole",
        "capsule_path": "/whole",
        "frozen_source_path": "/frozen",
        "capsule_sha256": "a" * 64,
        "required_lanes": ["host", "accelerator"],
        "required_tiers": ["L2", "L3"],
        "purpose": "functional_L2_L3_admission_not_performance_measurement",
    }
