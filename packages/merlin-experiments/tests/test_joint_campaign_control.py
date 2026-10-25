"""Joint policy owns routing and deadlines; no simulator or paid author starts."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from merlin_experiments.phase1 import joint_control as J
from merlin_experiments.phase1.options import parse_options


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setenv(J.ENV, "")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"version": 1, "final_only_capsules": ["whole"],
                               "gsim": {"witness_capsules": ["small"]}}))
    options = replace(parse_options(["--run-id", "test"]), joint_campaign=str(path),
                      fresh_backend=True, sandbox="bwrap", schedule="continuous")
    J.validate(options)
    J.configure(options, tmp_path / "run")
    monkeypatch.setattr(J.time, "time", lambda: 1000.0)
    return options


def test_clock_starts_at_author_launch_and_retains_resume(campaign, tmp_path):
    assert J.selected()["started_at_epoch"] is None
    with pytest.raises(RuntimeError, match="not started"):
        J.remaining()
    J.start()
    assert J.remaining() == 43200
    J.configure(campaign, tmp_path / "run", resume=True)
    J.start()
    assert J.selected()["started_at_epoch"] == 1000


@pytest.mark.parametrize("elapsed,allowed", [(0, False), (32399, False), (32400, True), (43199, True), (43200, False)])
def test_full_models_only_in_final_window(campaign, elapsed, allowed):
    J.start()
    if allowed:
        assert J.require_full_model(now=1000 + elapsed) == 43200 - elapsed
    else:
        with pytest.raises(RuntimeError):
            J.require_full_model(now=1000 + elapsed)
    with pytest.raises(RuntimeError):
        J.require_full_model(engine="gsim", now=1000 + 33000)


def test_explicit_empty_backend_required(campaign):
    with pytest.raises(ValueError, match="fresh-backend"):
        J.validate(replace(campaign, fresh_backend=False))


def test_gsim_only_single_selected_witness_and_budget(campaign, monkeypatch):
    J.start()
    with pytest.raises(ValueError, match="roster"):
        with J.physical_witness("all"):
            pytest.fail("unselected witness admitted")
    with J.physical_witness("small") as budget:
        assert budget == 1200
        ledger = Path(J.os.environ[J.ENV]).with_name("joint_gsim_budget.json")
        assert json.loads(ledger.read_text())["reserved_s"] == 1200
    ledger.write_text(json.dumps({"reserved_s": 14400, "active": []}))
    with pytest.raises(ValueError, match="budget"):
        with J.physical_witness("small"):
            pytest.fail("exhausted budget admitted")


def test_default_selfcheck_is_spike_and_physical_tiers_refuse(campaign):
    J.start()
    observed = []
    @J.selfcheck_policy
    def run(argv, **kwargs):
        observed.append(argv)
        return 0
    assert run(["--capsules", "large"]) == 0
    assert observed[0][observed[0].index("--sim") + 1] == "spike"
    assert run(["--sim", "spike", "--tiers", "L3"]) == 2
    assert run(["--sim", "gsim", "--capsules", "large"]) == 2
    assert len(observed) == 1


def test_final_capsules_remain_pending_without_mutating_inputs(campaign, tmp_path):
    root = tmp_path / "capsules"
    for name in ("small", "whole"):
        (root / name).mkdir(parents=True)
        (root / name / "capsule.yaml").write_text("name: " + name)
    assert J.development_roots(root) == [root / "small"]
    assert (root / "whole" / "capsule.yaml").is_file()
    report = J.stage_evidence({"all_pass": True})
    assert report["joint_campaign"]["final_only_pending"] == ["whole"]
    assert report["joint_campaign"]["physical_certificate"] is False


def test_changed_policy_refuses_resume(campaign, tmp_path):
    path = Path(campaign.joint_campaign)
    value = json.loads(path.read_text()); value["final_model_s"] = 14400
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        J.configure(campaign, tmp_path / "run", resume=True)


def test_physical_autopromotion_disabled(campaign, tmp_path):
    from merlin_experiments.phase1.feedback.promotion import promote
    assert promote(tmp_path, tmp_path, {}, None, None, None, None, context=None) == []


def test_legacy_selection_unchanged(monkeypatch):
    monkeypatch.delenv(J.ENV, raising=False)
    assert J.selected() is None
    assert J.require_full_model(engine="gsim") is None
    observed = []
    @J.selfcheck_policy
    def run(argv, **kwargs):
        observed.append(argv)
        return 7
    assert run(["--sim", "gsim"]) == 7
    assert observed == [["--sim", "gsim"]]


def test_physical_batches_require_operator_question_and_enforce_counts(campaign):
    from merlin_experiments.phase1 import joint_measurements as M
    J.start()
    for _ in range(2):
        with J.physical_witness("small"):
            pass
    with pytest.raises(ValueError, match="operator-authorized"):
        with J.physical_witness("small"):
            pytest.fail("third unapproved diagnostic admitted")
    with pytest.raises(ValueError, match="at most"):
        M.authorize(["small"] * 5, question="missing transfer price")
    M.authorize(["small"] * 4, question="identify rate and fixed cost")
    M.authorize(["small"] * 2, question="independent bootstrap validation")
    with pytest.raises(ValueError, match="bootstrap"):
        M.authorize(["small"], question="would be ninth bootstrap observation")
    for _ in range(6):
        with J.physical_witness("small"):
            pass
    with pytest.raises(ValueError, match="operator-authorized"):
        with J.physical_witness("small"):
            pytest.fail("unapproved refinement admitted")
    M.authorize(["small"], question="new buffering regime", stage="refinement")
    with J.physical_witness("small"):
        pass


def test_joint_periodic_qa_never_constructs_physical_ladder(campaign, tmp_path, monkeypatch):
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.feedback import qa
    J.start()
    root = tmp_path / "capsules"
    for name in ("small", "whole"):
        (root / name).mkdir(parents=True)
        (root / name / "capsule.yaml").write_text(f"name: {name}\nrequired_oracle_tiers: [L2, L3]\n")
    context = InvocationContext(tmp_path, tmp_path / "descriptor", tmp_path, "fixture",
                                tmp_path, tmp_path, tmp_path, ())
    monkeypatch.setattr(qa, "_loop_target_sim_via", lambda _: ("fixture", ""))
    monkeypatch.setattr(qa.CR, "qa_checkpoint_adapters", lambda *a, **k: pytest.fail("physical ladder selected"))
    monkeypatch.setattr(qa.CR, "qa_loop_adapters", lambda *a, **k: {"L2": "functional"})
    seen = []
    def grade(*args, **kwargs):
        seen.append(kwargs)
        assert kwargs["oracle_adapters"] == {"L2": "functional"}
        assert kwargs["capsules_root"] == [root / "small"]
        return {"n_capsules": 1, "n_passed": 0, "per_capsule": []}
    monkeypatch.setattr(qa.CG, "grade", grade)
    monkeypatch.setattr(qa, "_per_capsule_from_results", lambda *a, **k: {})
    result = qa.run("submission", str(root), tmp_path / "runs", {"public"}, False, 1, context=context)
    assert len(seen) == 1 and result["all_pass"] is False


def test_external_policy_uses_dynamic_pool_without_changing_strict_limits(tmp_path):
    path = tmp_path / "policy.json"
    value = {"version": 1, "duration_s": 18000, "final_model_s": 3600,
             "compiler_execution_mode": "snapshot_external_v1",
             "gsim": {"max_jobs": 20, "worker_budget_s": 360000}}
    path.write_text(json.dumps(value))
    assert J.policy(path)["gsim"]["max_jobs"] == 20
    value.pop("compiler_execution_mode")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="gSIM"):
        J.policy(path)
    value["compiler_execution_mode"] = "snapshot_external_v1"
    value["gsim"]["worker_budget_s"] = 360001
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="worker_budget"):
        J.policy(path)


def test_external_model_request_admission_has_no_final_window_curriculum(monkeypatch):
    monkeypatch.setattr(J, "selected", lambda: {"policy": {
        "compiler_execution_mode": "snapshot_external_v1", "final_model_s": 100}})
    monkeypatch.setattr(J, "remaining", lambda **kw: 5000)
    assert J.require_full_model() == 5000
    with pytest.raises(RuntimeError):
        J.require_full_model(engine="gsim")
    monkeypatch.setattr(J, "remaining", lambda **kw: 0)
    with pytest.raises(RuntimeError):
        J.require_full_model()
