"""Focused tests for the holdout-safe paper capture materializer."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from merlin.baselines.bundle import CaptureBundle
from merlin.common.artifacts import ProductDir
from merlin.common.paths import bench_dir, repo_root
from merlin.compare import capture_workflow as workflow
from merlin.compare.capture_workflow import CaptureWorkflowNotReady
from merlin.compare.paper import PaperStudySpec
from merlin.compare.host_experiment import HostExperimentSpec


STUDY = bench_dir() / "rvv_paper" / "study_v2.yaml"
HOST = repo_root() / "merlin" / "experiments" / "cpu_host_compiler_v0" / "experiment.yaml"
MODEL2MLIR = Path("/scratch/agustin/projects/model2MLIR")


def _product(path: Path) -> ProductDir:
    path.mkdir(parents=True)
    return ProductDir(
        path=path, manifest_path=path / "manifest.yaml", run_id=path.name,
        topic="paper-captures", version=1, git_sha="abcdef0",
        timestamp="20260831T000000Z", target="test", sources=[], _artifacts=[])


def _input_record(spec: PaperStudySpec) -> tuple[Path, dict]:
    bundle = workflow._resolve_paper_inputs(spec)
    return bundle, json.loads((bundle / "paper_inputs.json").read_text(encoding="utf-8"))


def test_curated_execute_is_blocked_before_any_holdout_loader_runs(tmp_path):
    called = False

    def forbidden_runner(*_args):
        nonlocal called
        called = True
        return 0

    with pytest.raises(CaptureWorkflowNotReady) as raised:
        workflow.materialize(
            STUDY, HOST, MODEL2MLIR, execute=True, runner=forbidden_runner,
            product=_product(tmp_path / "capture-run"))

    assert called is False
    plan = json.loads((raised.value.output_dir / "capture-plan.json").read_text())
    assert plan["status"] == "blocked"
    assert len(plan["tasks"]) == 10
    assert any("campaign_complete" in reason for reason in plan["errors"])
    assert not (raised.value.output_dir / "staged-study.yaml").exists()


def test_unpromoted_completed_campaign_cannot_execute_a_holdout_loader(
        tmp_path, monkeypatch):
    host = dataclasses.replace(
        HostExperimentSpec.from_yaml(HOST), status="campaign_complete_unpromoted")
    monkeypatch.setattr(workflow.HostExperimentSpec, "from_yaml", lambda _path: host)
    called = False

    def forbidden_runner(*_args):
        nonlocal called
        called = True
        return 0

    with pytest.raises(CaptureWorkflowNotReady) as raised:
        workflow.materialize(
            STUDY, HOST, MODEL2MLIR, execute=True, runner=forbidden_runner,
            product=_product(tmp_path / "unpromoted"))
    assert called is False
    plan = json.loads((raised.value.output_dir / "capture-plan.json").read_text())
    assert any("not campaign_complete" in reason for reason in plan["errors"])


def test_plan_uses_exact_bundle_environment_and_removes_inherited_smoke_knobs(
        tmp_path, monkeypatch):
    spec = PaperStudySpec.from_yaml(STUDY)
    inputs, record = _input_record(spec)
    tasks, errors = workflow._tasks(
        spec, record, inputs, MODEL2MLIR, _product(tmp_path / "capture-run"))

    assert errors == []
    assert len(tasks) == 10
    assert all(task.python.is_file() for task in tasks)
    gemma = next(task for task in tasks if task.model.name == "gemma2_2b")
    assert gemma.environment["M2M_GEMMA_LAYERS"] == ""
    assert gemma.environment["M2M_GEMMA_SLICE_LAYERS"] == ""
    assert gemma.environment["M2M_GEMMA_PAPER_READY"] == "1"
    assert gemma.environment["M2M_GEMMA_TOKEN_IDS"].startswith(str(inputs) + "/")
    smol = next(task for task in tasks if task.model.name == "smolvla")
    assert "M2M_SMOLVLA_VLM_LAYERS" not in smol.environment
    monkeypatch.setenv("M2M_SMOLVLA_VLM_LAYERS", "1")
    child = workflow._sanitized_environment(smol.environment)
    assert "M2M_SMOLVLA_VLM_LAYERS" not in child
    assert child["M2M_SMOLVLA_PAPER_READY"] == "1"


def test_complete_fake_capture_set_emits_one_freeze_ready_staged_study(tmp_path, monkeypatch):
    spec = PaperStudySpec.from_yaml(STUDY)
    inputs, record = _input_record(spec)
    evidence = {"model2mlir": {"path": str(MODEL2MLIR), "source_sha256": "a" * 64}}
    monkeypatch.setattr(
        workflow, "_preflight", lambda *_args: ([], evidence, record))

    def fake_runner(task, _environment, _stdout, _stderr):
        task.output.mkdir(parents=True)
        return 0

    def fake_validate(task, _source, _elapsed):
        return {"path": str(task.output.resolve()), "sha256": "b" * 64,
                "session_kind": task.model.session.kind, "paper_ready": True}

    monkeypatch.setattr(workflow, "_validate_output", fake_validate)
    output = workflow.materialize(
        STUDY, HOST, MODEL2MLIR, execute=True, runner=fake_runner,
        product=_product(tmp_path / "capture-run"))

    staged = PaperStudySpec.from_yaml(output / "staged-study.yaml")
    artifacts = [artifact for model in staged.models for artifact in model.artifacts.values()]
    assert len(artifacts) == 10
    assert all(artifact["sha256"] == "b" * 64 for artifact in artifacts)
    assert all(Path(artifact["path"]).is_relative_to(output / "bundles")
               for artifact in artifacts)
    assert json.loads((output / "capture-registration.json").read_text())["complete"] is True
    assert json.loads((output / "capture-plan.json").read_text())["status"] == "complete"


def test_rejected_capture_never_emits_partial_registration(tmp_path, monkeypatch):
    spec = PaperStudySpec.from_yaml(STUDY)
    _inputs, record = _input_record(spec)
    evidence = {"model2mlir": {"path": str(MODEL2MLIR), "source_sha256": "a" * 64}}
    monkeypatch.setattr(
        workflow, "_preflight", lambda *_args: ([], evidence, record))

    def fake_runner(task, _environment, _stdout, _stderr):
        task.output.mkdir(parents=True)
        return 0

    def reject(_task, _source, _elapsed):
        raise ValueError("session contract is not marked paper_ready=true")

    monkeypatch.setattr(workflow, "_validate_output", reject)
    product = _product(tmp_path / "capture-run")
    with pytest.raises(CaptureWorkflowNotReady, match="paper_ready=true"):
        workflow.materialize(
            STUDY, HOST, MODEL2MLIR, execute=True, runner=fake_runner, product=product)
    assert not (product.path / "staged-study.yaml").exists()
    assert not (product.path / "capture-registration.json").exists()
    assert json.loads((product.path / "capture-plan.json").read_text())["status"] == "failed"


def test_capture_bundle_require_accepts_complete_multi_program_root(tmp_path):
    import yaml

    root = tmp_path / "capture"
    child = root / "stages" / "decode"
    child.mkdir(parents=True)
    (child / "model.mlir").write_text("module {}\n", encoding="utf-8")
    (child / "golden.npy").write_bytes(b"golden")
    (root / "session_contract.yaml").write_text(yaml.safe_dump({
        "version": 2, "programs": [{"name": "decode", "bundle": "stages/decode"}],
    }), encoding="utf-8")
    capture = CaptureBundle("model", "fp32", root)
    assert capture.require() is capture
    (child / "golden.npy").unlink()
    with pytest.raises(FileNotFoundError, match="decode"):
        capture.require()
