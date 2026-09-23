"""Real formal receipts reach Phase 2; synthetic observations are NOT OS/RTL proof.

Only external engine results, source revision observation, and prior authoring /
sandbox observations are fixture data. Grading, snapshot verification, freeze,
formal manifest publication and both Phase-2 admission layers remain real.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.context import load_context
from merlin_experiments.phase1.feedback import formal, freeze
from merlin_experiments.phase2 import campaign
from merlin_experiments.phase2.global_inputs import FrozenPhase1

from merlin.benchharness import hash_tree
from merlin.common import artifacts, provenance
from merlin.common.paths import data_path
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.sandbox import bwrap as BW


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    return build_handoff(tmp_path, monkeypatch)


def build_handoff(tmp_path, monkeypatch, *, reviewed=None, authored_submission=None):
    """Join real frozen inputs and formal receipts; external oracle observations stay synthetic."""
    # The explicit legacy context updates environment during formal.main.
    monkeypatch.setattr(os, "environ", os.environ.copy())

    def forbidden(*args, **kwargs):
        pytest.fail("formal handoff must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    spec = importlib.util.spec_from_file_location(
        "formal_handoff_inputs", Path(__file__).with_name("phase1_feedback_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    seed_root = tmp_path if reviewed is None else tmp_path / "synthetic-submission-seed"
    resources = (
        {}
        if reviewed is None
        else {
            "contract_root": Path(reviewed.fixture["environment"]["MERLIN_CONTRACT_DIR"]),
            "schemas_root": Path(reviewed.fixture["environment"]["MERLIN_SCHEMAS_DIR"]),
        }
    )
    workspace, public, descriptor, _, env, _ = helper.inputs(seed_root, **resources)
    for name in ("MERLIN_OUT_ROOT", "TMPDIR"):
        monkeypatch.setenv(name, env[name])
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "public-cas"))
    monkeypatch.setenv("MERLIN_MESH_SIM", "synthetic")
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT", raising=False)
    repo = tmp_path
    contract = data_path("contract")
    if reviewed is None:
        hidden = tmp_path / "hidden/H"
        hidden.mkdir(parents=True)
        capsule = json.loads((public / "A/capsule.yaml").read_text())
        capsule.update(name="H", label="hidden")
        (hidden / "capsule.yaml").write_text(json.dumps(capsule))
        bundle = {
            "bundle_id": "merlin_assisted_rtlchecks_synthetic_handoff",
            "allowed": [{"path": str(public)}],
            "denied": [],
            "host_inputs": [{"path": str(hidden.parent)}],
        }
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
        frozen_public, frozen_hidden = BW.snapshot_input_paths(workspace, bundle, [public, hidden.parent], repo=repo)
        public_count = 1
    else:
        seed_submission = workspace / "submission"
        seed_manifest = seed_submission / "manifest.yaml"
        if authored_submission is None:
            seed_document = json.loads(seed_manifest.read_text())
            seed_document["target"] = reviewed.target.target
            seed_manifest.write_text(json.dumps(seed_document))
        else:
            assert (authored_submission / "manifest.yaml").is_file()
            assert (authored_submission / "compiler.py").is_file()
            shutil.copytree(authored_submission, seed_submission, dirs_exist_ok=True)
            assert hash_tree(seed_submission) == hash_tree(authored_submission)
        workspace = reviewed.workspace
        shutil.copytree(seed_submission, workspace / "submission")
        descriptor = reviewed.descriptor
        repo = reviewed.fixture["workspace"]
        contract = Path(reviewed.fixture["environment"]["MERLIN_CONTRACT_DIR"])
        bundle = reviewed.prepared.bundle
        frozen_public = reviewed.view.public
        hidden_roots = reviewed.target.hidden_roots()
        assert len(hidden_roots) == 1
        (frozen_hidden,) = BW.snapshot_input_paths(workspace, bundle, hidden_roots, repo=repo)
        public_count = 2
    snapshot = BW.snapshot_record(workspace)
    assert snapshot["version"] == 4
    runs = tmp_path / "runs"
    run = runs / "merlin_assisted" / "formal-handoff"
    run.mkdir(parents=True)
    shutil.copytree(workspace / "submission", run / "submission")
    manifest = run / "input_bundle_manifest.yaml"
    if reviewed is None:
        manifest.write_text(yaml.safe_dump(bundle))
    else:
        shutil.copyfile(reviewed.run / "input_bundle_manifest.yaml", manifest)
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == reviewed.prepared.effective_sha256
    environment = {
        "run_id": run.name,
        "bundle_id": bundle["bundle_id"],
        "bundle_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "workspace_path": str(workspace),
        "bundle_input_snapshot": snapshot,
        "sandbox": "bwrap",
        "isolation_violations": [],
        "golden_mask_selftest": {"leaked_answer_files": [], "n_answer_files_masked": 1},
        "task_scope": {"required_public_dev_capsules": public_count, "held_out_capsules": 1},
        "fixture_observations": "synthetic external sandbox/authoring; no OS isolation claim",
    }
    if reviewed is not None:
        environment.update(
            authored_bundle_manifest_sha256=reviewed.prepared.authored_sha256,
            bundle_manifest_sha256=reviewed.prepared.effective_sha256,
            public_corpus_input=reviewed.prepared.corpus_record,
        )
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    (run / "qa_loop_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "converged": True,
                "rounds": [{"answer_access_clean": True, "audit_hits": []}],
                "finalize": {"answer_access_clean": True, "audit_hits": [], "regrade_all_pass": True},
                "fixture_observations": "synthetic authoring observations, not an executed agent",
            }
        )
    )
    monkeypatch.setattr(CR, "_config_for_target", lambda *args: SimpleNamespace(rtl_tiers={"L3"}))
    monkeypatch.setattr(CR, "_rtl_tiers_of", lambda target: {"L3"})
    monkeypatch.setattr(
        CR,
        "describe_l3_engine",
        lambda target: {"available": True, "engine": "fixture", "fidelity": "elaborated_rtl"},
    )
    monkeypatch.setattr(CR, "oracle_adapters", lambda *args: {"L0": object(), "L2": object(), "L3": object()})
    monkeypatch.setattr(CR, "suite_for", lambda target: target + "-capsule-bench")
    monkeypatch.setattr(provenance, "load_pins", lambda: {})
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    monkeypatch.setattr(artifacts, "git_sha7", lambda: "fixture")
    monkeypatch.setattr(freeze, "repo_sha", lambda **kwargs: "synthetic-source-observation")
    events = []

    def external_execution(caps, package_dir, *, runs_root, oracle_adapters, target, **kwargs):
        is_hidden = {cap["label"] for cap in caps} == {"hidden"}
        assert (run / "freeze.json").exists() == is_hidden
        events.append("hidden" if is_hidden else "public")
        rows = []
        for cap in caps:
            row = {
                "capsule": cap["name"],
                "label": cap["label"],
                "kind": "isa",
                "status": "pass",
                "tiers": {name: {"status": "pass", "derived_from_rtl": name == "L3"} for name in oracle_adapters},
                "highest_tier": "L3",
                "numeric": {"status": "pass", "mismatch_count": 0},
                "trace_check": {"status": "pass", "violations": []},
            }
            result = Path(runs_root) / "runs" / CR.suite_for(target) / cap["name"]
            result.mkdir(parents=True)
            (result / "capsule_result.json").write_text(json.dumps(row))
            rows.append(row)
        return rows

    monkeypatch.setattr(CR, "run_suite", external_execution)
    assert (
        formal.main(
            [
                "--run-dir",
                str(run),
                "--arm",
                "merlin_assisted",
                "--capsules",
                str(frozen_public),
                "--hidden-capsules",
                str(frozen_hidden),
                "--contract",
                str(contract),
            ],
            context=load_context(descriptor, repo=repo),
        )
        == 0
    )
    assert events == ["public", "hidden"]
    digest = hash_tree(run / "submission")["sha256"]
    return SimpleNamespace(run=run, runs=runs, digest=digest, environment=environment, events=events)


def test_formal_receipts_admit_without_mocking_phase2(handoff):
    frozen = campaign.inspect_functional_run(handoff.runs, handoff.run.name, handoff.digest)
    assert not frozen.deviations
    binding = FrozenPhase1(handoff.runs, handoff.run.name, handoff.digest, (), 1, 1, ()).verify(
        handoff.run / "submission"
    )
    assert binding["public_passed"] == binding["public_total"] == 1
    assert len(binding["evidence_sha256"]) == 6


def test_actual_diagnostic_observations_still_refuse(handoff):
    environment = dict(handoff.environment, sandbox="none", golden_mask_selftest={})
    (handoff.run / "environment.yaml").write_text(yaml.safe_dump(environment))
    with pytest.raises(campaign.CampaignGateError, match="sandbox_not_bwrap.*answer_mask_vacuous"):
        campaign.inspect_functional_run(handoff.runs, handoff.run.name, handoff.digest)


@pytest.mark.parametrize("mutation", ["submission", "score", "snapshot"])
def test_formal_handoff_refuses_changed_evidence(handoff, mutation):
    if mutation == "submission":
        path = handoff.run / "submission/compiler.py"
    elif mutation == "score":
        path = handoff.run / "grading_public/score_capsule.json"
    else:
        path = Path(handoff.environment["bundle_input_snapshot"]["path"]) / "snapshot.json"
    path.chmod(0o600)
    if mutation == "score":
        document = json.loads(path.read_text())
        document["n_passed"] = 0
        path.write_text(json.dumps(document))
    else:
        path.write_bytes(path.read_bytes() + b"\n# altered evidence\n")
    with pytest.raises((campaign.CampaignGateError, ValueError)):
        campaign.inspect_functional_run(handoff.runs, handoff.run.name, handoff.digest)
