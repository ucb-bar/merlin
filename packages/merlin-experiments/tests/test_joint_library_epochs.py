"""An upgraded library retains only this campaign's compiler and original budget."""
from dataclasses import replace
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
    E.prepare(prior, contract_sha256="a" * 64, review_id="new-timing-model", output=path,
              performance_selection_sha256="c" * 64)
    options = replace(options, joint_epoch=str(path))
    with pytest.raises(ValueError, match="performance selection"):
        E.selected(options)
    monkeypatch.setenv("MERLIN_JOINT_PERFORMANCE_SELECTION", json.dumps({"sha256": "c" * 64}))
    assert E.selected(options)["new_contract_sha256"] == "a" * 64
