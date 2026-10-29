"""The seed rehearsal grades the planned run's sealed release and can write the starting grade."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def R():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("seed_rehearsal_test", HERE / "seed_rehearsal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corpus_is_the_planned_release_staged_as_the_session_does(R, tmp_path, monkeypatch):
    from merlin_experiments.phase1 import corpus_inputs

    planned, seed_run = tmp_path / "planned/target_experiment.yaml", tmp_path / "seed-run"
    bundle = tmp_path / "planned/bundle.yaml"
    bundle.parent.mkdir()
    bundle.write_text("bundle_id: planned\n")
    calls = []

    def stage(run_dir, te, selected, *, contract):
        calls.append((te, selected, contract))
        root = run_dir / "private_corpus_input"
        for relative in ("public/A/capsule.yaml", "scalability/cases/S/capsule.yaml", "contract/schemas/x"):
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text("x")
        for path in [*root.rglob("*"), root]:  # the session seals its copy read-only
            path.chmod(path.stat().st_mode & ~0o222)

    monkeypatch.setattr(corpus_inputs, "stage", stage)
    monkeypatch.setattr("merlin.targetgen.target_experiment.load_target_experiment", lambda path: ("te", path))
    plan = {"cwd": str(tmp_path / "export"), "seed": {"run": {"run_dir": str(seed_run)}},
            "argv": ["python", "-m", "merlin_experiments.phase1", "--descriptor", str(planned),
                     "--repo", str(tmp_path / "export"), "--bundle-manifest", str(bundle)]}
    run = tmp_path / "run"
    run.mkdir()
    corpus = R.planned_corpus(plan, run, tmp_path / "corpus")
    assert calls == [(("te", planned), {"bundle_id": "planned"}, tmp_path / "export/merlin/contract")]
    assert not (run / "private_corpus_input").exists()
    assert R._cases("all", corpus) == {"core": ["A"], "scalability": ["S"]}
    (corpus / "public/A/capsule.yaml").write_text("writable")  # the rehearsal's copy, not a sealed one


def test_planning_inventory_comes_from_the_planned_corpus(R, tmp_path, monkeypatch):
    from merlin.targetgen import capsule_common
    from merlin_experiments.phase1 import capability_inventory, joint_control

    (tmp_path / "corpus/scalability/cases").mkdir(parents=True)
    seen = []
    monkeypatch.setattr(capsule_common, "discover_capsules",
                        lambda root, labels=None, contract=None: seen.append(Path(root).name) or [{"name": Path(root).name}])
    monkeypatch.setattr(joint_control, "selected", lambda: {"policy": {"p": 1}})
    monkeypatch.setattr(capability_inventory, "build",
                        lambda native, *, scalability, policy: {"native": native, "scale": scalability, "policy": policy})
    workspace, run = tmp_path / "ws", tmp_path / "run"
    workspace.mkdir()
    run.mkdir()
    R.planning_inventory(tmp_path / "corpus", workspace, run)
    document = json.loads((workspace / "PUBLIC_CAPABILITIES.json").read_text())
    assert seen == ["public", "cases"] and document["native"] == [{"name": "public"}]
    assert (run / "PUBLIC_CAPABILITIES.json").read_text() == (workspace / "PUBLIC_CAPABILITIES.json").read_text()


def test_start_grade_lists_pass_carried_and_cycles_per_case(R, tmp_path):
    grade = {"wall_s": 12.5, "cases": {
        "conv": {"status": "pass", "carried_from": "seed-run", "cycles": 100, "cycles_evidence": "carried from seed",
                 "suite": "core"},
        "big": {"status": "pass", "carried_from": None, "cycles": 9000, "cycles_evidence": "timing model",
                "suite": "scalability"},
        "attn": {"status": "fail", "carried_from": None, "cycles": None, "suite": "core"}}}
    summary = R.start_grade(grade, tmp_path / "evidence")
    assert summary == {"cases": 3, "passed": 2, "carried": 1, "wall_s": 12.5}
    document = json.loads((tmp_path / "evidence/SEED_START_GRADE.json").read_text())
    assert document["schema"] == "merlin.seed_start_grade.v1" and document["per_case"]["conv"]["cycles"] == 100
    text = (tmp_path / "evidence/SEED_START_GRADE.md").read_text()
    assert "**2 of 3 test cases pass**, 1 of them carried" in text
    assert "| conv | core | pass (carried) | 100 |" in text and "| attn | core | fail |  |" in text


def test_start_grade_option_is_parsed(R, monkeypatch, tmp_path):
    captured = {}

    def stop(args):
        captured.update(vars(args))
        raise SystemExit(0)

    monkeypatch.setattr(R, "_context", stop)
    with pytest.raises(SystemExit):
        R.main(["--spec", "s.yaml", "--start-grade", str(tmp_path)])
    assert captured["start_grade"] == tmp_path
