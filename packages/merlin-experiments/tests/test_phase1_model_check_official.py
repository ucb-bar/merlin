"""The sealed grade requires full model checks and bounded unique RTL evidence."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from merlin_experiments.phase1.feedback import model_check_official as official


def test_official_gate_dedupes_explicit_same_shape_kernels(tmp_path, monkeypatch):
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/manifest.yaml").write_text("commands: {}\n")
    calls = []

    def invoke(command, **kwargs):
        calls.append((command, kwargs))
        if command == "replay-kernel":
            return {"status": "pass", "body": {"status": "pass"}, "report": str(kwargs["output"])}
        return {"status": "pass", "report": str(kwargs["output"]), "body": {"programs": [
            {"program": program, "build": {"status": "linked"},
             "census": {"device": 1, "denominator": 1},
             "kernels": [{"source_region_id": f"group_{program}",
                          "interface_sha256": "a" * 64, "shape_sha256": "b" * 64}]}
            for program in ("prefill", "decode")
        ]}}

    monkeypatch.setattr(official, "_invoke", invoke)
    summary = official.run_gate(tmp_path / "submission", tmp_path / "run",
                                {"tiny_llama": ("prefill", "decode")})
    assert summary["passed"] is True
    assert summary["unique_kernel_count"] == 1
    assert [command for command, _ in calls] == ["check", "replay-kernel"]
    assert calls[0][1]["model"] == "tiny_llama" and calls[0][1]["timeout_s"] == 7200
    assert calls[1][1]["program"] == "prefill" and calls[1][1]["timeout_s"] <= 300
    assert json.loads((tmp_path / "run/grading_model_check/summary.json").read_text()) == summary


def test_official_gate_refuses_missing_kernel_evidence(tmp_path, monkeypatch):
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/manifest.yaml").write_text("commands: {}\n")
    def invoke(command, **kwargs):
        assert command == "check"
        return {"status": "pass", "report": str(kwargs["output"]), "body": {"programs": [
            {"program": "model", "census": {"device": 1}, "kernels": []}
        ]}}

    monkeypatch.setattr(official, "_invoke", invoke)
    summary = official.run_gate(tmp_path / "submission", tmp_path / "run", {"resnet50": ("model",)})
    assert summary["passed"] is False
    assert summary["models"]["resnet50"]["status"] == "unavailable"
    assert summary["rtl_complete"] is True and summary["unique_kernel_count"] == 0


def test_official_gate_runs_distinct_rtl_kernels_in_bounded_parallel(tmp_path, monkeypatch):
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/manifest.yaml").write_text("commands: {}\n")
    rendezvous = threading.Barrier(2, timeout=2)
    replayed = []

    def invoke(command, **kwargs):
        if command == "check":
            return {"status": "pass", "body": {"programs": [{
                "program": "model", "census": {"device": 2},
                "kernels": [{"source_region_id": "conv_0"}, {"source_region_id": "conv_1"}],
            }]}}
        replayed.append(kwargs["group"])
        rendezvous.wait()
        return {"status": "pass", "body": {}, "report": str(kwargs["output"])}

    monkeypatch.setattr(official, "_invoke", invoke)
    summary = official.run_gate(tmp_path / "submission", tmp_path / "run", {"resnet50": ("model",)},
                                rtl_workers=2)
    assert summary["passed"] and summary["rtl_complete"]
    assert summary["rtl_workers"] == 2
    assert replayed == ["conv_0", "conv_1"]
    assert [row["group"] for row in summary["rtl"]] == ["conv_0", "conv_1"]


def test_official_gate_marks_unstarted_rtl_pending_when_budget_expires(tmp_path, monkeypatch):
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/manifest.yaml").write_text("commands: {}\n")

    def invoke(command, **kwargs):
        assert command == "check"
        return {"status": "pass", "body": {"programs": [{
            "program": "model", "census": {"device": 1},
            "kernels": [{"source_region_id": "conv_0"}],
        }]}}

    monkeypatch.setattr(official, "_invoke", invoke)
    summary = official.run_gate(tmp_path / "submission", tmp_path / "run", {"resnet50": ("model",)},
                                rtl_budget_s=1)
    assert summary["rtl"] == [{"model": "resnet50", "program": "model", "group": "conv_0",
                               "status": "pending", "reason": "RTL budget exhausted"}]
    assert summary["passed"] is False and summary["rtl_complete"] is False


def test_official_invocation_uses_sealed_full_mode_and_bounded_rtl(tmp_path, monkeypatch):
    calls = []

    class Completed:
        returncode = 0

        def __init__(self, argv, **kwargs):
            calls.append(argv)
            assert kwargs["start_new_session"] is True
            command = argv[3]
            Path(argv[argv.index("--out") + 1]).write_text(json.dumps({
                "schema": official.REPORT_SCHEMA, "command": command, "model": "resnet50",
                "split": "sealed", "status": "pass", "exit_code": 0,
            }))

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(official.subprocess, "Popen", Completed)
    submission = tmp_path / "submission"
    submission.mkdir()
    checked = official._invoke("check", submission=submission, model="resnet50",
                               output=tmp_path / "check.json", bundle=tmp_path / "bundle", timeout_s=17)
    replayed = official._invoke("replay-kernel", submission=submission, model="resnet50",
                                program="model", group="conv_0", output=tmp_path / "replay.json",
                                bundle=tmp_path / "bundle", timeout_s=13)
    assert checked["status"] == replayed["status"] == "pass"
    assert all(argv[argv.index("--split") + 1] == "sealed" for argv in calls)
    assert all(Path(argv[argv.index("--models") + 1]).is_file() for argv in calls)
    assert calls[0][calls[0].index("--mode") + 1] == "full"
    assert calls[1][calls[1].index("--engine") + 1] == "gsim"
    assert calls[1][calls[1].index("--timeout-s") + 1] == "13"
