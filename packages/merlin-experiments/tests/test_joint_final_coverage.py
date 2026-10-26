import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_completion as C
from merlin_experiments.phase1 import joint_final_coverage as F

from merlin.common.tree_hash import hash_tree
from merlin.targetgen import capsule_grade as CG


def _score(names):
    return {
        "gradeable": True, "n_capsules": len(names), "n_passed": len(names),
        "cohort_admission": {"n_admitted_capsules": len(names),
                             "admitted_name_set_sha256": CG._name_set_sha256(names)},
        "per_capsule": [{"capsule": name, "status": "pass", "numeric": "pass",
                         "tiers": {tier: "pass" for tier in ("L0", "L1", "L2")}} for name in names],
        "numeric_all_exact": True, "structural_evidence_all_pass": True,
        "functional_public_evidence": {"claim": "functional_corroboration"},
    }


def _prepared(tmp_path):
    ws, run = tmp_path / "workspace", tmp_path / "run"
    (ws / "submission").mkdir(parents=True)
    run.mkdir()
    (ws / "submission/compiler.py").write_text("# fresh compiler\n")
    return SimpleNamespace(workspace=ws, run_dir=run, verify_inputs=lambda: None)


@pytest.mark.parametrize("epoch,drain,passed,expected_calls,expected_rc", [
    (True, True, True, 0, 0), (False, False, True, 0, 1),
    (False, True, False, 1, 1), (False, True, True, 1, 0),
])
def test_only_true_final_drain_invokes_frozen_validation(tmp_path, monkeypatch, epoch, drain, passed,
                                                       expected_calls, expected_rc):
    prepared = _prepared(tmp_path)
    if epoch:
        (prepared.run_dir / "joint_epoch_stop.json").write_text("{}")
    monkeypatch.setattr(C.joint_control, "selected", lambda: {"started_at_epoch": 1, "policy_sha256": "policy"})
    monkeypatch.setattr(C.joint_control, "remaining", lambda **kw: 0 if drain else 5)
    calls = []

    def validate(frozen, digest):
        assert frozen == prepared.run_dir / "joint_frozen_submission"
        assert hash_tree(frozen)["sha256"] == digest
        calls.append(digest)
        return {"complete": passed, "passed": passed}

    rc = C.complete(prepared, verdict={"all_pass": True}, rounds=[], workflow_conformant=True,
                    final_coverage=validate)
    summary = json.loads((prepared.run_dir / "joint_campaign_summary.json").read_text())
    assert len(calls) == expected_calls and rc == expected_rc
    assert summary["final_validation"]["engine"] == "firesim"
    assert summary["final_validation"]["complete"] is False
    assert summary["converged"] is False
    if expected_calls:
        assert summary["numeric_all_pass"] == passed


def test_final_pass_cannot_drop_failed_or_unmeasured_obligations():
    caps = [{"name": "a"}, {"name": "b"}]
    for mutate in (
        lambda s: s["per_capsule"].pop(),
        lambda s: s.update(structural_evidence_all_pass=False),
        lambda s: s["per_capsule"][1].update(status="gated"),
        lambda s: s["per_capsule"][1].update(status="fail"),
        lambda s: s["per_capsule"].append(copy.deepcopy(s["per_capsule"][0])),
    ):
        score, group = _score(["a", "b"]), F._capsule_group(caps)
        mutate(score)
        F._apply_score(group, score)
        report = F._rollup({"groups": {"core": group}})
        assert set(group["obligations"]) == {"a", "b"}
        assert report["passed"] is False
    group = F._capsule_group(caps)
    F._apply_score(group, _score(["a", "b"]))
    assert F._rollup({"groups": {"core": group}})["passed"] is True


def test_withheld_native_passes_stay_counted_when_closure_fails():
    caps = [{"name": "a"}, {"name": "b"}]
    score, group = _score(["a", "b"]), F._capsule_group(caps)
    score["per_capsule"][1].update(status="fail")
    score.update(structural_evidence_all_pass=False)
    F._apply_score(group, score)
    report = F._rollup({"groups": {"core": group}})
    assert report["passed"] is False and report["n_passed"] == 0 and report["n_failed"] == 1
    assert group["obligations"]["a"]["status"] == "skipped"
    assert report["n_native_passed"] == 1 and report["n_closure_withheld"] == 1


@pytest.mark.parametrize("labels,engine,tier,no_oracle", [
    ({"hidden"}, "spike", "L2", False), ({"public"}, "gsim", "L2", False),
    ({"public"}, "spike", "L3", False), ({"public"}, "spike", "L2", True),
])
def test_functional_public_mode_cannot_admit_hidden_rtl_or_no_oracle(labels, engine, tier, no_oracle):
    adapter = SimpleNamespace(_merlin_simulator_engine=engine)
    with pytest.raises(ValueError, match="public/dev-only exact Spike"):
        CG.grade("unused", capsules_root="unused", runs_root="unused", target="fixture",
                 labels=labels, oracle_adapters={"L2": adapter}, functional_public_tier=tier,
                 no_oracle=no_oracle)


def _native_fixture(tmp_path, monkeypatch):
    from merlin_experiments.phase1.brokers import model_check as MC
    from merlin_experiments.phase1.feedback import loop_grading as LG
    from merlin_experiments.phase1.feedback import public_programs as PP
    from merlin_experiments.phase1.feedback import public_scalability as PS
    from merlin_experiments.phase1.feedback import selfcheck as SC

    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import target_experiment as TE

    prepared = _prepared(tmp_path)
    frozen = prepared.workspace / "submission"
    digest = hash_tree(frozen)["sha256"]
    core, scale = tmp_path / "core", tmp_path / "scale"
    names = {str(core): [f"core{i}" for i in range(173)],
             str(scale / "cases"): [f"scale{i}" for i in range(67)]}
    state = {"started_at_epoch": 1, "policy_sha256": "policy",
             "policy": {"duration_s": 43200, "final_only_capsules": names[str(core)][-6:]}}
    monkeypatch.setattr(F.joint_control, "selected", lambda: state)
    monkeypatch.setattr(F.joint_control, "remaining", lambda **kw: 0 if kw.get("drain") else 100)
    monkeypatch.setattr(F.joint_control, "require_full_model", lambda **kw: 100)
    monkeypatch.setattr(TE, "load_target_experiment", lambda p: SimpleNamespace(
        descriptor_sha256="descriptor", sim_via="fixture"))
    monkeypatch.setattr(CR, "discover_capsules", lambda root, **kw: [{"name": n} for n in names[str(root)]])
    monkeypatch.setattr(PS, "selected", lambda context: (scale, {"cases": names[str(scale / "cases")]}))
    monkeypatch.setattr(PP, 'selected', lambda context: None)
    monkeypatch.setattr(SC, "_adapters", lambda *a, **kw: (
        {"L2": SimpleNamespace(_merlin_simulator_engine="spike")}, "spike"))
    monkeypatch.setattr(LG, "compiler_execution", lambda *a, **kw: nullcontext())
    context = SimpleNamespace(repo=tmp_path, descriptor=tmp_path / "descriptor", target="fixture", readback_policy=None)
    inputs = SimpleNamespace(context=context, public_root=core, contract=None,
                             verify_grade_inputs=None, additional_forbidden=())
    calls = []

    def grade(candidate, **kw):
        assert candidate == frozen and kw["functional_public_tier"] == "L2"
        assert kw["defer_running"] is True and set(kw["oracle_adapters"]) == {"L2"}
        assert kw["timeout"] <= 100
        calls.append(("capsules", str(kw["capsules_root"])))
        return _score(names[str(kw["capsules_root"])])

    def model(ws, request, **kw):
        assert request["mode"] == "full" and kw["snapshot_source"] == frozen
        assert kw["timeout_s"] <= 100
        calls.append(("model", request["model"]))
        return {"status": "pass", "source_sha256": digest, "report": {
            "programs": [{"program": p, "census": {"denominator": 1, "device": 1}}
                         for p in MC.PROGRAMS[request["model"]]], "model_metrics": {"worst": {}}}}

    monkeypatch.setattr(CG, "grade", grade)
    monkeypatch.setattr(MC, "execute_visible", model)
    return prepared, inputs, frozen, digest, calls


def test_native_callback_requests_all_246_obligations_including_final_only(tmp_path, monkeypatch):
    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert report["n_required"] == 173 + 67 + 6
    assert report["n_passed"] == 246 and report["passed"] is True
    assert len(report["final_only"]) == 6 and len(calls) == 5
    assert report["deadline_epoch"] == 43201
    assert report["rtl_qualified"] is False


def test_expired_deadline_keeps_complete_denominator_without_native_launch(tmp_path, monkeypatch):
    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(F.joint_control, "remaining", lambda **kw: 0)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert calls == []
    assert report["n_required"] == 246 and report["n_unfinished"] == 246
    assert report["passed"] is False and report["complete"] is False


def test_budget_exhausted_during_core_cancels_following_work_without_new_clock(tmp_path, monkeypatch):
    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    original = CG.grade

    def exhaust(candidate, **kwargs):
        score = original(candidate, **kwargs)
        monkeypatch.setattr(F.joint_control, "remaining", lambda **kw: 0)
        assert kwargs["defer_pending"]() == "original joint campaign deadline exhausted"
        return score

    monkeypatch.setattr(CG, "grade", exhaust)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert len(calls) == 1 and report["n_passed"] == 173
    assert report["n_unfinished"] == 73 and report["n_required"] == 246
    assert report["deadline_epoch"] == 43201 and report["passed"] is False


@pytest.mark.parametrize('mutation', ['none', 'wrong_compiler', 'missing_output_grade', 'wrong_engine', 'suite_veto'])
def test_final_selected_programs_use_original_owner_and_cannot_disappear(tmp_path, monkeypatch, mutation):
    from merlin_experiments.phase1 import joint_model_activation
    from merlin_experiments.phase1.feedback import public_programs as PP

    prepared, inputs, frozen, digest, calls = _native_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(PP, 'selected', lambda context: ('selection', {}, {'connected': tmp_path}))

    def grade(candidate, **kwargs):
        assert candidate == frozen and kwargs['context'] is inputs.context and kwargs['timeout'] <= 100
        assert joint_model_activation._GRADE.get() is not None
        calls.append(('program', 'connected'))
        row = dict(status='pass', numeric={'status': 'pass', 'per_output': {'Y0': {'status': 'pass'}}}, tiers={
            'L0': 'pass', 'L1': 'pass', 'L2': dict(status='pass', engine='spike',
                                                 fidelity='functional_model', derived_from_rtl=False)})
        result = dict(candidate_sha256=digest, expected_cases=['connected'], requested_cases=['connected'],
                      passed=True, complete=True, observed={'connected': row})
        if mutation == 'wrong_compiler':
            result['candidate_sha256'] = 'foreign'
        elif mutation == 'missing_output_grade':
            row['numeric'] = 'unknown'
        elif mutation == 'wrong_engine':
            row['tiers']['L2']['engine'] = 'gsim'
        elif mutation == 'suite_veto':
            result['passed'] = False
        return result

    monkeypatch.setattr(PP, '_grade', grade)
    report = F.validate(prepared, inputs=inputs, frozen=frozen, compiler_sha256=digest)
    assert ('program', 'connected') in calls
    assert report['n_required'] == 247
    assert report['passed'] is (mutation == 'none')
    assert report['groups']['programs']['passed'] is (mutation == 'none')
    if mutation != 'none':
        assert report['groups']['programs']['obligations']['connected']['status'] != 'pass'
