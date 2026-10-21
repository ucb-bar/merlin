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
