"""The visible broker grades copied bytes and never accepts sealed requests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from merlin_experiments.phase1 import authoring
from merlin_experiments.phase1.brokers import model_check as broker
from merlin_experiments.phase1.feedback import visible_model_gate as gate
from merlin_experiments.phase1.tools import model_check as client


@pytest.fixture(autouse=True)
def selected_target_contract(monkeypatch):
    contract = Path(__file__).resolve().parents[3] / "examples/gemmini/target/contracts/target_contract.yaml"
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(contract))


def test_visible_model_check_uses_immutable_copy_and_no_private_argv(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    source = ws / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("commands: {}\n")
    (source / "compiler.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("MERLIN_MODEL_CHECK_BUNDLE", str(tmp_path / "selected-bundle"))
    calls = []

    class Completed:
        returncode = 0

        def __init__(self, argv, *, cwd, stdout, stderr, start_new_session):
            calls.append(argv)
            snapshot = Path(argv[argv.index("--submission") + 1])
            assert (snapshot / "compiler.py").read_text() == "VALUE = 1\n"
            (source / "compiler.py").write_text("VALUE = 2\n")
            assert start_new_session and cwd == ws
            assert argv[argv.index("--mode") + 1] == "fast"
            assert argv[argv.index("--bundle") + 1] == str(tmp_path / "selected-bundle")
            assert Path(argv[argv.index("--models") + 1]).is_file()
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({
                "schema": broker.REPORT_SCHEMA, "command": "check", "model": "resnet50",
                "status": "pass", "exit_code": 0, "agent_view": True,
                "programs": [{"program": "model", "build": {"status": "linked"}}],
                "split": "visible",
            }))

        def poll(self):
            return 0

    monkeypatch.setattr(broker.subprocess, "Popen", Completed)
    result = broker.execute_visible(ws, {"action": "check", "model": "resnet50", "split": "visible"})
    assert result["status"] == "pass"
    assert result["report"]["programs"][0]["program"] == "model"
    assert all("--agent-view" in argv and argv[argv.index("--split") + 1] == "visible" for argv in calls)
    assert (source / "compiler.py").read_text() == "VALUE = 2\n"


def test_replay_request_uses_contract_program_and_report_file(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("commands: {}\n")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    class Completed:
        returncode = 0

        def __init__(self, argv, **kwargs):
            assert argv[argv.index("--program") + 1] == "decode"
            assert argv[argv.index("--group") + 1] == "matmul_0"
            assert argv[argv.index("--engine") + 1] == "spike"
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({
                "schema": broker.REPORT_SCHEMA, "command": "replay-kernel", "model": "tiny_llama",
                "split": "visible", "agent_view": True, "status": "pass", "exit_code": 0,
                "programs": [{"program": "decode"}],
            }))

        def poll(self):
            return 0

    monkeypatch.setattr(broker.subprocess, "Popen", Completed)
    result = broker.execute_visible(ws, {"action": "replay-kernel", "model": "tiny_llama",
                                         "program": "decode", "group": "matmul_0", "engine": "spike",
                                         "split": "visible"})
    assert result["status"] == "pass"
    missing_program = broker.execute_visible(ws, {"action": "replay-kernel", "model": "tiny_llama",
                                                  "group": "matmul_0", "engine": "spike", "split": "visible"})
    assert missing_program["status"] == "rejected"


def test_visible_model_replay_rejects_gsim_before_spawning(tmp_path, monkeypatch):
    def forbidden_spawn(*_args, **_kwargs):
        pytest.fail("Phase 1 must not spawn a model-kernel gSIM replay")

    monkeypatch.setattr(broker.subprocess, "Popen", forbidden_spawn)
    result = broker.execute_visible(tmp_path, {"action": "replay-kernel", "model": "tiny_llama",
                                               "program": "decode", "group": "matmul_0",
                                               "engine": "gsim", "split": "visible"})
    assert result["status"] == "rejected"
    assert "spike" in result["error"]


def test_visible_check_accepts_stable_source_with_generated_build_state(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    source = ws / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("commands: {}\n")
    (source / "CMakeCache.txt").write_text("generated cache\n")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    class Completed:
        returncode = 0

        def __init__(self, argv, **kwargs):
            snapshot = Path(argv[argv.index("--submission") + 1])
            assert not (snapshot / "CMakeCache.txt").exists()
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({
                "schema": broker.REPORT_SCHEMA, "command": "check", "model": "resnet50",
                "split": "visible", "agent_view": True, "status": "pass", "exit_code": 0,
            }))

        def poll(self):
            return 0

    monkeypatch.setattr(broker.subprocess, "Popen", Completed)
    result = broker.execute_visible(ws, {"action": "check", "model": "resnet50", "split": "visible"})
    assert result["status"] == "pass"


def test_sealed_or_private_report_never_reaches_agent(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("commands: {}\n")
    assert broker.execute_visible(ws, {"action": "check", "model": "resnet50", "split": "sealed"})[
        "status"] == "rejected"
    assert broker._safe_report({"split": "sealed", "status": "pass"}) is False
    assert broker._safe_report({"path": "/work/merlin/private/gemmini/numerics/v1"}) is False


def test_visible_report_from_private_bundle_preserves_results_and_passes_export_guard(tmp_path):
    from merlin.model_check import report as R

    root = str(tmp_path / "private" / "numerics" / "bundle")
    report = R.new_report("check", model="resnet50", split="visible",
                          submission={"tree_sha256": "a" * 64})
    report["bundle"] = {"path": root, "manifest_sha256": "b" * 64}
    report["gates"] = {"kernels": {"status": "fail"}}
    report["programs"] = [{"program": "model", "runs": [{
        "split": "visible", "status": "fail", "artifact": root + "/resnet50/visible.json",
    }]}]
    report["model_metrics"] = {"per_input": [{"split": "visible", "error": 0.125}]}
    R.settle(report)
    view = R.agent_view(report)
    out = tmp_path / "visible-report.json"
    R.write(view, out)
    published = json.loads(out.read_text())
    assert broker._safe_report(published)
    assert "path" not in published["bundle"]
    assert published["status"] == "fail" and published["exit_code"] == 1
    assert published["submission"] == report["submission"]
    assert published["gates"] == report["gates"]
    assert published["model_metrics"] == report["model_metrics"]
    assert published["programs"][0]["runs"][0]["artifact"] == "<numerics_bundle>/resnet50/visible.json"
    assert report["bundle"]["path"] == root  # Original host report is retained intact.
    assert report["programs"][0]["runs"][0]["artifact"].startswith(root)


@pytest.mark.parametrize("unexpected", [
    "/unrelated/private/record", "/original/private/bundle-other/data", "sealed_inputs", "sealed",
])
def test_known_bundle_redaction_does_not_admit_other_private_data(unexpected):
    from merlin.model_check import report as R

    report = R.new_report("check", model="resnet50", split="visible", submission={})
    report["bundle"] = {"path": "/original/private/bundle"}
    report["unexpected"] = unexpected
    assert broker._safe_report(R.agent_view(report)) is False


def test_missing_model_check_report_is_unavailable_not_numeric_failure(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("commands: {}\n")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    class Missing:
        returncode = 1

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 1

    monkeypatch.setattr(broker.subprocess, "Popen", Missing)
    result = broker.execute_visible(ws, {"action": "check", "model": "resnet50", "split": "visible"})
    assert result["status"] == "unavailable"
    assert gate._program_row("resnet50", "model", result)["status"] == "unavailable"


def test_model_pass_without_census_or_metrics_does_not_pass_visible_verdict():
    result = {"status": "pass", "report": {"programs": [{
        "program": "model", "build": {"built": 25}, "kernels": [{"symbol": "conv_0", "status": "pass"}],
    }], "model_metrics": None}}
    assert gate._program_row("resnet50", "model", result)["status"] == "unavailable"


def test_model_check_client_refuses_output_through_workspace_symlink(tmp_path, monkeypatch):
    ws, outside = tmp_path / "workspace", tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / ".model_check_channel").mkdir()
    (ws / "escape").symlink_to(outside, target_is_directory=True)
    (ws / ".model_check_channel/resp_job.json").write_text('{"status":"pass"}')
    monkeypatch.chdir(ws)
    with pytest.raises(SystemExit):
        client.main(["wait", "--job-id", "job", "--out", "escape/result.json"])
    assert not (outside / "result.json").exists()


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
        return {"status": "pass", "source_sha256": digest, "report": {
            "model_metrics": {"worst": {"rel_rmse": 0.0}},
            "programs": [{"program": name, "census": {"denominator": 1, "device": 1},
                          "runs": [{"cycles_functional": 12}]}
                         for name in broker.PROGRAMS[model]]}}

    monkeypatch.setattr(gate, "execute_visible", check)
    result = gate.check_snapshot(ws, run, snapshot, digest, stop=threading.Event())
    visible = json.loads((ws / "qa/verdict.json").read_text())
    assert result["programs"] == visible["model_gate"]["programs"]
    assert len(result["programs"]) == 6 and visible["model_gate"]["passed"]
    assert (run / "qa_history" / f"model_gate_{digest[:16]}" / "gate.json").is_file()


def test_background_model_gate_reaches_agent_verdict_before_first_interval(tmp_path, monkeypatch):
    ws, run = tmp_path / "workspace", tmp_path / "run"
    source = ws / "submission"
    source.mkdir(parents=True)
    run.mkdir()
    (source / "manifest.yaml").write_text("commands: {}\n")
    (source / "CMakeCache.txt").write_text("generated cache\n")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    def check(_ws, request, **kwargs):
        from merlin.common.tree_hash import hash_tree

        model = request["model"]
        return {"status": "pass", "source_sha256": hash_tree(kwargs["snapshot_source"])["sha256"],
                "report": {"model_metrics": {"worst": {}},
                           "programs": [{"program": program, "census": {"denominator": 1, "device": 1}}
                                        for program in broker.PROGRAMS[model]]}}

    monkeypatch.setattr(gate, "execute_visible", check)
    handle = gate.start(ws, run, interval_s=30)
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            verdict = json.loads((ws / "qa/verdict.json").read_text())
            if verdict.get("model_gate", {}).get("complete"):
                break
            time.sleep(0.02)
        assert verdict["model_gate"]["complete"] is True
        assert len(verdict["model_gate"]["programs"]) == 6
    finally:
        gate.stop_gate(handle)


def test_background_gate_restarts_superseded_snapshot_without_waiting_interval(tmp_path, monkeypatch):
    from merlin.common.tree_hash import hash_tree

    ws, run = tmp_path / "workspace", tmp_path / "run"
    source = ws / "submission"
    source.mkdir(parents=True)
    run.mkdir()
    (source / "manifest.yaml").write_text("commands: {}\n")
    (source / "compiler.py").write_text("VERSION = 1\n")
    checked = []

    def check(_ws, request, **kwargs):
        digest = hash_tree(kwargs["snapshot_source"])["sha256"]
        checked.append((request["model"], digest))
        if len(checked) == 1:
            (source / "compiler.py").write_text("VERSION = 2\n")
        return {"status": "pass", "source_sha256": digest, "report": {
            "model_metrics": {"worst": {}},
            "programs": [{"program": program, "census": {"denominator": 1, "device": 1}}
                         for program in broker.PROGRAMS[request["model"]]]}}

    monkeypatch.setattr(gate, "execute_visible", check)
    handle = gate.start(ws, run, interval_s=30)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            verdict = json.loads((ws / "qa/verdict.json").read_text())
            if verdict.get("model_gate", {}).get("complete"):
                break
            time.sleep(0.02)
        assert verdict["model_gate"]["complete"] is True
        assert len(checked) == 4  # one obsolete check, then all three models on the new source
        assert checked[0][1] != checked[1][1]
        assert verdict["model_gate"]["source_sha256"] == hash_tree(source)["sha256"]
    finally:
        gate.stop_gate(handle)


def test_verified_status_line_leads_with_all_six_visible_programs():
    rows = {key: {"status": "pass"} for key in gate._rows(None)}
    line = authoring._verified_status_line({"model_gate": {"programs": rows}})
    assert line.startswith("Visible models: resnet50/model=pass")
    assert line.count("=pass") == 6
    assert line.endswith("Sealed official grade pending.")
