"""An upgraded library retains only this campaign's compiler and original budget."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import pytest
import yaml

from merlin.common.tree_hash import hash_tree
from merlin_experiments.phase1 import fresh_backend, joint_control as J, joint_epochs as E
from merlin_experiments.phase1.options import parse_options


@pytest.fixture
def epoch(tmp_path, monkeypatch):
    monkeypatch.delenv(J.ENV, raising=False)
    monkeypatch.delenv(E.PERFORMANCE_ENV, raising=False)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "gsim": {"witness_capsules": ["small"]}}))
    options = replace(parse_options(["--run-id", "first"]), joint_campaign=str(policy),
                      fresh_backend=True, sandbox="bwrap", schedule="continuous")
    prior = tmp_path / "prior"
    J.configure(options, prior)
    J.start()
    source = prior / "joint_frozen_submission"
    source.mkdir()
    (source / "manifest.yaml").write_text("target: fixture\n")
    (source / "compiler.py").write_text("# newly authored\n")
    environment = {"run_id": "first", "lineage_id": "first", "backend_origin": {
        "schema": "merlin.empty_backend_origin.v1", "initial_files": 0},
        "additional_compiler_library": {"contract_sha256": "a" * 64}}
    (prior / "environment.yaml").write_text(yaml.safe_dump(environment))
    (prior / "joint_campaign_summary.json").write_text(json.dumps({
        "schema": "merlin.joint_campaign_completion.v1", "frozen_compiler": str(source),
        "compiler_sha256": hash_tree(source)["sha256"], "completion_reason": "interrupted"}))
    J.gsim_ledger().write_text(json.dumps({"reserved_s": 14400, "active": []}))
    review = tmp_path / "review.json"
    E.prepare(prior, contract_sha256="b" * 64, review_id="reviewed-fix", output=review)
    return replace(options, run_id="second", joint_epoch=str(review)), prior


def test_epoch_uses_same_clock_and_physical_budget(epoch, tmp_path):
    options, prior = epoch
    started = J.selected()["started_at_epoch"]
    J.configure(options, tmp_path / "next")
    J.start()
    assert J.selected()["started_at_epoch"] == started
    assert J.gsim_ledger() == prior / "joint_gsim_budget.json"
    with pytest.raises(ValueError, match="budget exhausted"):
        with J.physical_witness("small"):
            pytest.fail("epoch reset physical budget")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    record = E.seed(options, workspace, tmp_path / "next")
    assert record["lineage"]["lineage_id"] == "first"
    assert (workspace / "submission/compiler.py").read_text() == "# newly authored\n"
    assert fresh_backend.origin(options, workspace)["initial_files"] == 0
    (workspace / "submission/compiler.py").write_text("# author may continue after origin\n")
    assert fresh_backend.origin(options, workspace, prior={"backend_origin": {
        "schema": "merlin.empty_backend_origin.v1", "initial_files": 0}})["initial_files"] == 0


def test_epoch_rejects_mutated_source_policy_and_expired_time(epoch, tmp_path, monkeypatch):
    options, prior = epoch
    old = E.selected(options)
    with pytest.raises(ValueError, match="policy"):
        E.inherited_state(options, {"changed": True}, "different")
    monkeypatch.setattr(E.time, "time", lambda: time.monotonic() + 10**12)
    with pytest.raises(ValueError, match="time remaining"):
        E.inherited_state(options, J.selected()["policy"], J.selected()["policy_sha256"])
    (Path(old["source"]) / "compiler.py").write_text("# foreign seed\n")
    with pytest.raises(ValueError, match="compiler changed"):
        E.selected(options)


def test_epoch_rejects_ordinary_seed_and_requires_new_library(epoch, tmp_path):
    options, prior = epoch
    with pytest.raises(ValueError, match="ordinary seeds"):
        E.selected(replace(options, seed_submission="foreign"))
    with pytest.raises(ValueError, match="changed reviewed contract"):
        E.prepare(prior, contract_sha256="a" * 64, review_id="no-change", output=tmp_path / "bad.json")


def test_operator_stop_is_private_and_does_not_reset_clock(epoch):
    _options, prior = epoch
    started = J.selected()["started_at_epoch"]
    E.request_stop(prior, reason="reviewed scheduling fix ready")
    assert "reviewed scheduling fix" in J.stop_reason()
    assert J.selected()["started_at_epoch"] == started


def test_timing_only_epoch_requires_explicit_selected_model_pin(epoch, tmp_path, monkeypatch):
    options, prior = epoch
    path = tmp_path / "timing-review.json"
    timing = tmp_path / "timing.json"
    timing.write_text('{"model": "new"}')
    pin = {"path": str(timing), "sha256": hashlib.sha256(timing.read_bytes()).hexdigest()}
    E.prepare(prior, contract_sha256="a" * 64, review_id="new-timing-model", output=path,
              performance_selection_sha256=pin["sha256"])
    options = replace(options, joint_epoch=str(path))
    with pytest.raises(ValueError, match="performance selection"):
        E.selected(options)
    monkeypatch.setenv(E.PERFORMANCE_ENV, json.dumps(pin))
    assert E.selected(options)["new_contract_sha256"] == "a" * 64


def test_model_service_drains_then_prepares_pinned_transition(epoch, tmp_path, monkeypatch):
    options, prior = epoch
    summary = prior / "joint_campaign_summary.json"
    completed = summary.read_bytes()
    summary.unlink()
    timing = tmp_path / "model.json"
    timing.write_text('{"model": "reviewed-calibration"}')
    digest = hashlib.sha256(timing.read_bytes()).hexdigest()
    kwargs = dict(selection_path=timing, expected_sha256=digest, review_id="heldout-accepted-v2",
                  output=tmp_path / "model-epoch.json")
    assert E.request_model_epoch(prior, **kwargs)["status"] == "waiting_for_native_drain"
    assert E.stop_reason(prior)
    summary.write_bytes(completed)
    ready = E.request_model_epoch(prior, **kwargs)
    assert ready["status"] == "ready_for_native_epoch"
    assert E.request_model_epoch(prior, **kwargs) == ready
    monkeypatch.setenv(E.PERFORMANCE_ENV, json.dumps(ready["performance_selection"]))
    next_options = replace(options, joint_epoch=ready["joint_epoch"])
    J.configure(next_options, tmp_path / "next-model")
    assert J.selected()["started_at_epoch"] == json.loads((prior / "joint_campaign.json").read_text())["started_at_epoch"]
    assert J.gsim_ledger() == prior / "joint_gsim_budget.json"
    with pytest.raises(ValueError, match="changed performance"):
        E.request_model_epoch(tmp_path / "next-model", **kwargs)
    monkeypatch.delenv(E.PERFORMANCE_ENV)
    with pytest.raises(ValueError, match="changed on resume"):
        J.configure(next_options, tmp_path / "next-model", resume=True)
    timing.write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="content differs"):
        E.request_model_epoch(prior, **kwargs)


def test_stop_notice_preserves_grade_and_survives_background_update(epoch, tmp_path):
    from merlin_experiments.phase1.feedback.visible_model_gate import publish

    _options, prior = epoch
    workspace = tmp_path / 'author'
    (workspace / 'qa').mkdir(parents=True)
    path = workspace / 'qa/verdict.json'
    grade = {'all_pass': False, 'n_capsules': 13, 'n_passed': 0,
             'graded_source_sha256': 'source', 'package_failure': {'detail': 'original error'},
             'graded_at': 'original-time'}
    path.write_text(json.dumps(grade))
    env_path = prior / 'environment.yaml'
    environment = yaml.safe_load(env_path.read_text())
    environment['workspace_path'] = str(workspace)
    env_path.write_text(yaml.safe_dump(environment))
    request = E.request_stop(prior, reason='Repair library routing; keep valid generic edits.')
    current = json.loads(path.read_text())
    notice = current.pop('operator_epoch_notice')
    assert current == grade
    assert notice['action'] == 'checkpoint_and_return'
    assert notice['stop_request_sha256'] == hashlib.sha256(request.read_bytes()).hexdigest()
    later = publish(path, {**grade, 'n_passed': 1})
    assert later['operator_epoch_notice'] == notice
    assert later['n_passed'] == 1
    assert later['graded_source_sha256'] == 'source'
