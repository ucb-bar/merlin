"""Focused process-boundary checks; these do not execute a submitted compiler."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
import subprocess

import pytest

from merlin.perf.analysis_worker import IsolatedAnalysisWorker, _read_output


@pytest.mark.parametrize("failure", ["first", "second", "second_timeout"])
def test_worker_emission_pair_preserves_first_failure_without_running_second(tmp_path, monkeypatch, failure):
    from merlin.perf import analysis_worker as worker
    source = tmp_path / "interface.mlir"
    source.write_text("module {}\n")
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    compiler_scratch = tmp_path / "compiler_scratch"
    compiler_scratch.mkdir()
    stage = tmp_path / "trusted_pair_stage.py"
    stage.write_text(
        "from pathlib import Path\n"
        "class StageE2ESentinel:\n"
        " def __init__(self, **kwargs): pass\n"
        "def analyze_whole_model_emission(*args, emit_pair_runner, **kwargs):\n"
        " root=Path(__file__).parent\n"
        " rc, lowered, cb=emit_pair_runner(object(), root/'interface.mlir', root/'diagnostics', 'candidate', 10)\n"
        " if rc:\n"
        "  raise ValueError('emission failed: '+(root/'diagnostics/emission_candidate.json').read_text())\n"
        " return {}\n")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"stage_path": str(stage), "baseline": str(tmp_path),
        "candidate": str(tmp_path), "sentinel": {}, "timeout_s": 10,
        "kwargs": {"target": "not-a-real-target"}, "sandboxes": {"candidate": {}},
        "scratch": str(compiler_scratch), "result": str(tmp_path / "result.json")}))
    calls = []
    def entrypoint(package, name, *args, **kwargs):
        calls.append(name)
        if name == "emit_command_buffer":
            if failure == "first":
                return subprocess.CompletedProcess([name], 1, "", "original printer failure")
            Path(args[1]).write_text("{}")
            return subprocess.CompletedProcess([name], 0, "", "buffer diagnostic")
        if failure == "second":
            return subprocess.CompletedProcess([name], 2, "partial LLVM", "LLVM emitter failure")
        raise subprocess.TimeoutExpired("later LLVM emission", 10, stderr=b"LLVM timeout diagnostic")
    monkeypatch.setattr(worker, "run_sandboxed_entrypoint", entrypoint)
    assert worker._worker(request) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    retained = json.loads((tmp_path / "candidate_emission.json").read_text())
    assert result["failure"]["emission_diagnostics"]["candidate"] == retained
    if failure == "first":
        assert calls == ["emit_command_buffer"]
        assert result["failure"]["type"] == "ValueError"
        assert "original printer failure" in result["failure"]["reason"]
        assert retained["entrypoints"] == [{"command": "emit_command_buffer", "returncode": 1,
                                             "stderr_tail": "original printer failure"}]
        assert not (tmp_path / "candidate_lower_target_to_llvm.stderr").exists()
    else:
        assert calls == ["emit_command_buffer", "lower_target_to_llvm"]
        assert retained["entrypoints"][0]["returncode"] == 0
        last = retained["entrypoints"][1]
        assert last["command"] == "lower_target_to_llvm"
        if failure == "second_timeout":
            assert result["failure"]["type"] == last["exception"] == "TimeoutExpired"
            assert last["returncode"] is None
            assert last["stderr_tail"] == "LLVM timeout diagnostic"
        else:
            assert last["returncode"] == 2
            assert "LLVM emitter failure" in result["failure"]["reason"]


@pytest.mark.parametrize("failure", ["first", "second", "timeout", "none"])
def test_direct_emission_pair_stops_and_retains_diagnostics(tmp_path, monkeypatch, failure):
    import importlib
    import sys
    from merlin.common.paths import merlin_dir
    from merlin.targetgen import oot_runner
    sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
    stage = importlib.import_module("perf_agent_stage")
    calls = []
    def entrypoint(package, name, interface, destination, **kwargs):
        calls.append(name)
        assert kwargs["timeout"] == 7
        if failure == "first":
            return subprocess.CompletedProcess([name], 1, "", "first failure")
        if name == "emit_command_buffer":
            destination.write_text("{}")
            return subprocess.CompletedProcess([name], 0, "", "buffer diagnostic")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(name, 7, stderr=b"partial timeout stderr")
        return subprocess.CompletedProcess([name], 2 if failure == "second" else 0,
                                           "module {}", "LLVM diagnostic")
    monkeypatch.setattr(oot_runner, "run_entrypoint", entrypoint)
    if failure == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            stage._emit_pair(object(), tmp_path / "interface.mlir", tmp_path, "candidate", 7)
    else:
        result = stage._emit_pair(object(), tmp_path / "interface.mlir", tmp_path, "candidate", 7)
        assert result == {"first": (1, "", ""), "second": (2, "", ""),
                          "none": (0, "module {}", "{}")}[failure]
    rows = json.loads((tmp_path / "emission_candidate.json").read_text())["entrypoints"]
    assert calls == (["emit_command_buffer"] if failure == "first" else
                     ["emit_command_buffer", "lower_target_to_llvm"])
    assert len(rows) == len(calls)
    if failure == "timeout":
        assert rows[-1]["exception"] == "TimeoutExpired"
        assert rows[-1]["stderr_tail"] == "partial timeout stderr"


def test_whole_model_analysis_stops_after_failed_baseline_pair(tmp_path, monkeypatch):
    import importlib
    import sys
    from merlin.common.paths import merlin_dir
    from merlin.targetgen import oot_runner
    sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
    stage = importlib.import_module("perf_agent_stage")
    baseline, candidate, source = (tmp_path / name for name in ("baseline", "candidate", "source"))
    for path in (baseline, candidate, source):
        path.mkdir()
    (source / "capsule.yaml").write_text(json.dumps({"id": "test-model"}))
    (source / "capsule.interface.mlir").write_text("module {}")
    sentinel = stage.StageE2ESentinel("test-model", str(source), str(source),
                                    stage._exact_tree_record(source)["sha256"], (), ())
    calls = []
    def emit(package, interface, scratch, tag, timeout_s):
        calls.append(tag)
        if tag != "baseline":
            raise subprocess.TimeoutExpired("later candidate must not run", 10)
        (scratch / "emission_baseline.json").write_text(json.dumps({
            "schema": "compiler_emission_diagnostics_v1", "arm": "baseline", "entrypoints": [
                {"command": "emit_command_buffer", "returncode": 1, "stderr_tail": "baseline printer failed"}]}))
        return 1, "", ""
    monkeypatch.setattr(oot_runner, "load_package", lambda path: path)
    with pytest.raises(stage.StageGateError, match="baseline printer failed") as caught:
        stage.analyze_whole_model_emission(baseline, candidate, sentinel, timeout_s=10,
            peak_macs_per_cycle=None, achievable_macs_per_cycle=None,
            target="fixture-target", emit_pair_runner=emit)
    assert calls == ["baseline"]
    assert "emit_command_buffer" in str(caught.value)


@dataclass
class Sentinel:
    capsule: str = "test"


def test_worker_deadline_kills_analysis_and_child(tmp_path):
    pid_file = tmp_path / "child.pid"
    stage = tmp_path / "trusted_stage.py"
    stage.write_text(
        "import subprocess, sys, time\n"
        "class StageE2ESentinel:\n"
        " def __init__(self, **kwargs): pass\n"
        "def analyze_whole_model_emission(*args, **kwargs):\n"
        " child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f" open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        " time.sleep(30)\n")
    worker = IsolatedAnalysisWorker(stage_path=stage,
                                    sandbox_factory=lambda *args: {}, output=tmp_path / "worker")
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        worker(tmp_path, tmp_path, Sentinel(), timeout_s=1)
    assert time.monotonic() - started < 4
    assert pid_file.is_file()
    child = int(pid_file.read_text())
    # A killed child can briefly remain as a zombie until its new parent reaps it.
    status = Path(f"/proc/{child}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"
    receipt = json.loads(next((tmp_path / "worker").glob("*/receipt.json")).read_text())
    assert receipt["status"] == "timeout"
    assert receipt["process_group_cleanup"] is True


def test_compiler_output_does_not_follow_link(tmp_path):
    secret = tmp_path / "host_control.json"
    secret.write_text("host-only")
    output = tmp_path / "compiler_output.json"
    output.symlink_to(secret)
    with pytest.raises(OSError):
        _read_output(output)
    output.unlink()
    os.mkfifo(output)
    with pytest.raises(ValueError, match="regular file"):
        _read_output(output)
