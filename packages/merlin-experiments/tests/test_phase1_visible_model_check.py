"""The visible broker grades copied bytes and never accepts sealed requests."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from merlin_experiments.phase1.brokers import model_check as broker
from merlin_experiments.phase1.feedback import visible_model_gate as gate


def test_visible_model_check_uses_immutable_copy_and_no_private_argv(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    source = ws / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("commands: {}\n")
    (source / "compiler.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    calls = []

    class Completed:
        returncode = 0

        def __init__(self, argv, *, cwd, stdout, stderr, start_new_session):
            calls.append(argv)
            snapshot = Path(argv[argv.index("--submission") + 1])
            assert (snapshot / "compiler.py").read_text() == "VALUE = 1\n"
            (source / "compiler.py").write_text("VALUE = 2\n")
            assert start_new_session and cwd == ws
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({
                "programs": {"model": {"status": "pass"}}, "split": "visible",
            }))

        def poll(self):
            return 0

    monkeypatch.setattr(broker.subprocess, "Popen", Completed)
    result = broker.execute_visible(ws, {"action": "check", "model": "resnet50", "split": "visible"})
    assert result["status"] == "pass"
    assert result["report"]["programs"]["model"]["status"] == "pass"
    assert all("sealed" not in " ".join(argv) and "/private/" not in " ".join(argv) for argv in calls)
    assert (source / "compiler.py").read_text() == "VALUE = 2\n"


def test_sealed_or_private_report_never_reaches_agent(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("commands: {}\n")
    assert broker.execute_visible(ws, {"action": "check", "model": "resnet50", "split": "sealed"})[
        "status"] == "rejected"
    assert broker._safe_report({"split": "sealed", "status": "pass"}) is False
    assert broker._safe_report({"path": "/work/merlin/private/gemmini/numerics/v1"}) is False


def test_model_gate_preserves_capsule_verdict_and_requires_same_source(tmp_path):
    path = tmp_path / "qa" / "verdict.json"
    rows = gate._rows("a" * 64)
    current = gate.publish(path, {"model_gate": {"source_sha256": "a" * 64,
                                                   "split": "visible", "programs": rows}})
    assert current["all_pass"] is None
    current = gate.publish(path, {"all_pass": True, "graded_source_sha256": "a" * 64,
                                  "per_capsule": [{"status": "pass"}]})
    assert current["all_pass"] is None and current["model_gate"]["programs"] == rows
    passed = {key: {**row, "status": "pass"} for key, row in rows.items()}
    current = gate.publish(path, {"model_gate": {"source_sha256": "a" * 64,
                                                   "split": "visible", "programs": passed}})
    assert current["all_pass"] is True and current["model_gate"]["complete"]
    current = gate.publish(path, {"all_pass": True, "graded_source_sha256": "b" * 64})
    assert current["all_pass"] is None


def test_visible_gate_publishes_all_six_program_rows(tmp_path, monkeypatch):
    from merlin.common.tree_hash import hash_tree

    ws, run = tmp_path / "workspace", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.yaml").write_text("commands: {}\n")
    digest = hash_tree(snapshot)["sha256"]

    def check(_ws, request, **kwargs):
        model = request["model"]
        assert kwargs["snapshot_source"] == snapshot
        return {"status": "pass", "source_sha256": digest, "report": {"programs": {
            name: {"status": "pass", "census": {"eligible": 1, "offloaded": 1},
                   "metrics": {"rel_rmse": 0.0}, "cycles": 12}
            for name in broker.PROGRAMS[model]
        }}}

    monkeypatch.setattr(gate, "execute_visible", check)
    result = gate.check_snapshot(ws, run, snapshot, digest, stop=threading.Event())
    visible = json.loads((ws / "qa/verdict.json").read_text())
    assert result["programs"] == visible["model_gate"]["programs"]
    assert len(result["programs"]) == 6 and visible["model_gate"]["passed"]
    assert (run / "qa_history" / f"model_gate_{digest[:16]}" / "gate.json").is_file()
