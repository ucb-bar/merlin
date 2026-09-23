"""Focused process-boundary checks; these do not execute a submitted compiler."""

import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import contracts as CONTRACTS
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import emission_diagnostics as ED
from merlin_experiments.phase2 import stage_inputs as INPUTS

from merlin.common.digest import sha256_bytes
from merlin.common.paths import merlin_dir
from merlin.perf.analysis_worker import IsolatedAnalysisWorker, _read_output


@pytest.fixture(autouse=True)
def synthetic_analysis_owner(tmp_path, monkeypatch):
    """A test-only canonical owner for fixture modules, shared with harmless children."""
    fullname = "merlin_experiments.phase2.emission_analysis"
    original = sys.modules.get(fullname)
    # Each test owns at most one trusted_* source; no production fallback exists.
    finder_source = (
        "import importlib.abc, importlib.util, pathlib, sys\n"
        "class FixtureFinder(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname != 'merlin_experiments.phase2.emission_analysis': return None\n"
        f"  candidates=list(pathlib.Path({str(tmp_path)!r}).glob('trusted_*stage.py'))\n"
        "  if len(candidates) != 1: return None\n"
        "  return importlib.util.spec_from_file_location(fullname, candidates[0])\n"
        "sys.meta_path.insert(0, FixtureFinder())\n"
    )
    namespace = {}
    exec(finder_source, namespace)
    finder = sys.meta_path[0]
    overlay = tmp_path / "test_import_policy"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text(finder_source)
    monkeypatch.setenv("PYTHONPATH", str(overlay) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    yield
    sys.meta_path.remove(finder)
    if original is None:
        sys.modules.pop(fullname, None)
    else:
        sys.modules[fullname] = original


def test_worker_ignores_timestamp_valid_stage_bytecode(tmp_path):
    import py_compile

    from merlin.perf import analysis_worker as worker

    stage = tmp_path / "trusted_source_stage.py"
    source = "def analyze_whole_model_emission(*args, **kwargs):\n return {'owner': 'source'}\n"
    stage.write_text(source.replace("'source'", "'cached'"))
    timestamp = stage.stat().st_mtime_ns
    py_compile.compile(str(stage), doraise=True)
    assert Path(importlib.util.cache_from_source(str(stage))).is_file()
    stage.write_text(source)
    os.utime(stage, ns=(timestamp, timestamp))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "analysis_source": str(stage),
                "contract_root": str(tmp_path),
                "baseline": str(tmp_path),
                "candidate": str(tmp_path),
                "sentinel": {
                    "capsule": "model",
                    "capsule_path": "/model",
                    "frozen_source_path": "/frozen/model",
                    "capsule_sha256": "a" * 64,
                    "required_lanes": [],
                    "required_tiers": [],
                },
                "timeout_s": 10,
                "kwargs": {},
                "sandboxes": {},
                "scratch": str(tmp_path),
                "result": str(tmp_path / "result.json"),
            }
        )
    )
    assert worker._worker(request) == 0
    assert json.loads((tmp_path / "result.json").read_text())["analysis"] == {"owner": "source"}


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
        "def analyze_whole_model_emission(*args, emit_pair_runner, **kwargs):\n"
        " root=Path(__file__).parent\n"
        " rc, lowered, cb=emit_pair_runner(object(), root/'interface.mlir', root/'diagnostics', 'candidate', 10)\n"
        " if rc:\n"
        "  raise ValueError('emission failed: '+(root/'diagnostics/emission_candidate.json').read_text())\n"
        " return {}\n"
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "analysis_source": str(stage),
                "contract_root": str(tmp_path),
                "baseline": str(tmp_path),
                "candidate": str(tmp_path),
                "sentinel": {
                    "capsule": "model",
                    "capsule_path": "/model",
                    "frozen_source_path": "/frozen/model",
                    "capsule_sha256": "a" * 64,
                    "required_lanes": [],
                    "required_tiers": [],
                },
                "timeout_s": 10,
                "kwargs": {"target": "not-a-real-target"},
                "sandboxes": {"candidate": {}},
                "scratch": str(compiler_scratch),
                "result": str(tmp_path / "result.json"),
            }
        )
    )
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
        assert retained["entrypoints"] == [
            {"command": "emit_command_buffer", "returncode": 1, "stderr_tail": "original printer failure"}
        ]
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


def test_worker_uses_optional_one_pass_analysis_bundle(tmp_path, monkeypatch):
    from merlin.perf import analysis_worker as worker

    source = tmp_path / "interface.mlir"
    source.write_text("module {}\n")
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    compiler_scratch = tmp_path / "compiler_scratch"
    compiler_scratch.mkdir()
    stage = tmp_path / "trusted_bundle_stage.py"
    stage.write_text(
        "from pathlib import Path\n"
        "class Package:\n"
        " manifest={'commands': {'emit_analysis_bundle': {'argv': []}}}\n"
        "def analyze_whole_model_emission(*args, emit_pair_runner, **kwargs):\n"
        " root=Path(__file__).parent\n"
        " rc, lowered, cb=emit_pair_runner(Package(), root/'interface.mlir', "
        "root/'diagnostics', 'candidate', 10)\n"
        " return {'rc': rc, 'lowered': lowered, 'command_buffer': cb}\n"
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "analysis_source": str(stage),
                "contract_root": str(tmp_path),
                "baseline": str(tmp_path),
                "candidate": str(tmp_path),
                "sentinel": {
                    "capsule": "model",
                    "capsule_path": "/model",
                    "frozen_source_path": "/frozen/model",
                    "capsule_sha256": "a" * 64,
                    "required_lanes": [],
                    "required_tiers": [],
                },
                "timeout_s": 10,
                "kwargs": {"target": "not-a-real-target"},
                "sandboxes": {"candidate": {}},
                "scratch": str(compiler_scratch),
                "result": str(tmp_path / "result.json"),
            }
        )
    )
    calls = []

    def entrypoint(package, name, source_path, output_path, **kwargs):
        calls.append(name)
        assert output_path.name == "command_buffer.json"
        output_path.write_text('{"commands": []}')
        return subprocess.CompletedProcess([name], 0, "module {}", "one-pass diagnostic")

    monkeypatch.setattr(worker, "run_sandboxed_entrypoint", entrypoint)
    assert worker._worker(request) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert calls == ["emit_analysis_bundle"]
    assert result["analysis"] == {"rc": 0, "lowered": "module {}", "command_buffer": '{"commands": []}'}
    retained = json.loads((tmp_path / "candidate_emission.json").read_text())
    assert retained["entrypoints"] == [
        {"command": "emit_analysis_bundle", "returncode": 0, "stderr_tail": "one-pass diagnostic"}
    ]


@pytest.mark.parametrize("failure", ["first", "second", "timeout", "none"])
def test_direct_emission_pair_stops_and_retains_diagnostics(tmp_path, monkeypatch, failure):
    from merlin.targetgen import oot_runner

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
        return subprocess.CompletedProcess([name], 2 if failure == "second" else 0, "module {}", "LLVM diagnostic")

    monkeypatch.setattr(oot_runner, "run_entrypoint", entrypoint)
    if failure == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            EA.emit_pair(object(), tmp_path / "interface.mlir", tmp_path, "candidate", 7)
    else:
        result = EA.emit_pair(object(), tmp_path / "interface.mlir", tmp_path, "candidate", 7)
        assert result == {"first": (1, "", ""), "second": (2, "", ""), "none": (0, "module {}", "{}")}[failure]
    rows = json.loads((tmp_path / "emission_candidate.json").read_text())["entrypoints"]
    assert calls == (["emit_command_buffer"] if failure == "first" else ["emit_command_buffer", "lower_target_to_llvm"])
    assert len(rows) == len(calls)
    if failure == "timeout":
        assert rows[-1]["exception"] == "TimeoutExpired"
        assert rows[-1]["stderr_tail"] == "partial timeout stderr"


@pytest.mark.parametrize("returncode", [0, 3])
def test_direct_emission_pair_uses_optional_bundle_without_fallback(tmp_path, monkeypatch, returncode):
    from merlin.targetgen import oot_runner

    package = SimpleNamespace(manifest={"commands": {"emit_analysis_bundle": {"argv": []}}})
    calls = []

    def entrypoint(package, name, interface, destination, **kwargs):
        calls.append(name)
        destination.write_text('{"commands": []}')
        return subprocess.CompletedProcess([name], returncode, "module {}", "bundle failure" if returncode else "")

    monkeypatch.setattr(oot_runner, "run_entrypoint", entrypoint)
    result = EA.emit_pair(package, tmp_path / "interface.mlir", tmp_path, "candidate", 7)
    assert calls == ["emit_analysis_bundle"]
    assert result == ((returncode, "", "") if returncode else (0, "module {}", '{"commands": []}'))


def test_whole_model_analysis_stops_after_failed_baseline_pair(tmp_path, monkeypatch):
    import sys

    from merlin.targetgen import oot_runner

    sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
    baseline, candidate, source = (tmp_path / name for name in ("baseline", "candidate", "source"))
    for path in (baseline, candidate, source):
        path.mkdir()
    (source / "capsule.yaml").write_text(json.dumps({"id": "test-model"}))
    (source / "capsule.interface.mlir").write_text("module {}")
    sentinel = INPUTS.StageE2ESentinel(
        "test-model", str(source), str(source), P2_CONTRACTS.exact_tree_record(source)["sha256"], (), ()
    )
    calls = []

    def emit(package, interface, scratch, tag, timeout_s):
        calls.append(tag)
        if tag != "baseline":
            raise subprocess.TimeoutExpired("later candidate must not run", 10)
        (scratch / "emission_baseline.json").write_text(
            json.dumps(
                {
                    "schema": "compiler_emission_diagnostics_v1",
                    "arm": "baseline",
                    "entrypoints": [
                        {"command": "emit_command_buffer", "returncode": 1, "stderr_tail": "baseline printer failed"}
                    ],
                }
            )
        )
        return 1, "", ""

    monkeypatch.setattr(oot_runner, "load_package", lambda path: path)
    with pytest.raises(CONTRACTS.StageGateError, match="baseline printer failed") as caught:
        EA.analyze_whole_model_emission(
            baseline,
            candidate,
            sentinel,
            timeout_s=10,
            peak_macs_per_cycle=None,
            achievable_macs_per_cycle=None,
            target="fixture-target",
            emit_pair_runner=emit,
            contract_root=merlin_dir() / "contract",
        )
    assert calls == ["baseline"]
    assert "emit_command_buffer" in str(caught.value)


def test_whole_model_analysis_reuses_exact_seed_and_cached_baseline(tmp_path, monkeypatch):
    import sys

    from merlin.perf import artifact_activity, model_placement
    from merlin.targetgen import oot_runner
    from merlin.targetgen.rocc import decode

    sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
    baseline, candidate, source = (tmp_path / name for name in ("baseline", "candidate", "source"))
    for path in (baseline, candidate, source):
        path.mkdir()
    for path in (baseline, candidate):
        (path / "manifest.yaml").write_text("same bytes\n")
    (source / "capsule.yaml").write_text(json.dumps({"id": "test-model"}))
    (source / "capsule.interface.mlir").write_text("module {}")
    sentinel = INPUTS.StageE2ESentinel(
        "test-model", str(source), str(source), P2_CONTRACTS.exact_tree_record(source)["sha256"], (), ()
    )
    package = SimpleNamespace(manifest={"commands": {"emit_analysis_bundle": {"argv": []}}})
    monkeypatch.setattr(oot_runner, "load_package", lambda path: package)
    monkeypatch.setattr(
        ED,
        "analyze_command_buffers",
        lambda *args, **kwargs: {"arms": {"baseline": {"status": "emitted"}, "candidate": {"status": "emitted"}}},
    )
    monkeypatch.setattr(EA, "guidance_for_emission_analysis", lambda *args: {})
    monkeypatch.setattr(EA, "inspect_compiler_package", lambda *args: {})
    monkeypatch.setattr(
        model_placement,
        "captured_global_graph",
        lambda *args, **kwargs: {"status": "verified", "logical_dispatch_digest": "b" * 64},
    )
    monkeypatch.setattr(model_placement, "contraction_placement", lambda *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(
        artifact_activity, "analyze_artifact_activity", lambda *args, **kwargs: {"status": "verified", "issued": {}}
    )
    parse_calls = []
    decode_calls = []
    monkeypatch.setattr(decode, "_parse_module", lambda text: parse_calls.append(text) or object())
    monkeypatch.setattr(
        decode,
        "decode_module",
        lambda module, **kwargs: decode_calls.append(kwargs.get("source")) or {"instructions": []},
    )

    emitted = []
    verifier_calls = []
    machine_calls = []
    lowered = "module {}"
    raw_buffer = '{"commands": [], "params": {}}'
    policy = {"schema": "fixture_machine_policy_v1"}

    def emit(package, interface, scratch, tag, timeout_s):
        emitted.append((tag, timeout_s))
        return 0, lowered, raw_buffer

    def verify(**kwargs):
        verifier_calls.append(kwargs["candidate_sha256"])
        return {
            "status": "verified",
            "plan_digest": "c" * 64,
            "source_sha256": sha256_bytes(b"module {}"),
            "candidate_sha256": kwargs["candidate_sha256"],
            "logical_dispatch_digest": "b" * 64,
            "candidate_lowered_sha256": sha256_bytes(lowered.encode()),
            "candidate_command_buffer_sha256": sha256_bytes(raw_buffer.encode()),
        }

    def audit(text, *, arm, timeout_s):
        machine_calls.append(arm)
        return {"status": "verified", "source_sha256": sha256_bytes(text.encode()), "build_policy_identity": policy}

    retained = {}
    first = EA.analyze_whole_model_emission(
        baseline,
        candidate,
        sentinel,
        timeout_s=1200,
        peak_macs_per_cycle=None,
        achievable_macs_per_cycle=None,
        target="fixture-target",
        global_plan_verifier=verify,
        artifact_sink=retained.update,
        emit_pair_runner=emit,
        machine_artifact_auditor=audit,
        machine_build_policy_identity=policy,
        host_verifier_policy_sha256="d" * 64,
        contract_root=merlin_dir() / "contract",
    )
    assert emitted == [("baseline", 1200)]
    assert len(parse_calls) == len(decode_calls) == len(verifier_calls) == len(machine_calls) == 1
    emission_execution = first["diagnostics"]["emission_execution"]
    measured_emission_wall = emission_execution.pop("baseline_emission_measured_wall_seconds")
    assert measured_emission_wall >= 0
    assert emission_execution == {
        "schema": "whole_model_emission_execution_v1",
        "identical_compiler_trees": True,
        "retained_baseline_reused": False,
        "candidate_reused_baseline_artifacts": True,
        "launched_entrypoint_count": 1,
        "baseline_entrypoints": 1,
        "candidate_entrypoints": 0,
        "per_entrypoint_timeout_seconds": 1200,
        "analysis_budget_seconds": 1200,
        "baseline_emission_source": "compiler_executed",
        "baseline_emission_cache_key": None,
    }

    # A changed candidate with the retained baseline must launch/analyse only the candidate.
    (candidate / "candidate_change.py").write_text("# changed\n")
    emitted.clear()
    parse_calls.clear()
    decode_calls.clear()
    verifier_calls.clear()
    machine_calls.clear()
    second = EA.analyze_whole_model_emission(
        baseline,
        candidate,
        sentinel,
        timeout_s=100,
        peak_macs_per_cycle=None,
        achievable_macs_per_cycle=None,
        target="fixture-target",
        global_plan_verifier=verify,
        baseline_artifacts=retained["baseline_artifacts"],
        emit_pair_runner=emit,
        machine_artifact_auditor=audit,
        machine_build_policy_identity=policy,
        host_verifier_policy_sha256="d" * 64,
        contract_root=merlin_dir() / "contract",
    )
    assert emitted == [("candidate", 100)]
    assert len(parse_calls) == len(decode_calls) == len(verifier_calls) == len(machine_calls) == 1
    execution = second["diagnostics"]["emission_execution"]
    assert execution["retained_baseline_reused"] is True
    assert execution["baseline_entrypoints"] == 0
    assert execution["candidate_entrypoints"] == execution["launched_entrypoint_count"] == 1


@dataclass
class Sentinel:
    capsule: str = "test"
    capsule_path: str = "/model"
    frozen_source_path: str = "/frozen/model"
    capsule_sha256: str = "a" * 64
    required_lanes: tuple[str, ...] = ()
    required_tiers: tuple[str, ...] = ()


def test_worker_deadline_kills_analysis_and_child(tmp_path):
    pid_file = tmp_path / "child.pid"
    stage = tmp_path / "trusted_stage.py"
    stage.write_text(
        "import subprocess, sys, time\n"
        "def analyze_whole_model_emission(*args, **kwargs):\n"
        " child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f" open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        " time.sleep(30)\n"
    )
    worker = IsolatedAnalysisWorker(
        analysis_source=stage, contract_root=tmp_path, sandbox_factory=lambda *args: {}, output=tmp_path / "worker"
    )
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


def test_worker_reserves_result_transport_grace_inside_outer_budget(tmp_path):
    stage = tmp_path / "trusted_stage.py"
    stage.write_text(
        "def analyze_whole_model_emission(*args, timeout_s, **kwargs):\n"
        " return {'analysis_budget_seconds': timeout_s}\n"
    )
    worker = IsolatedAnalysisWorker(
        analysis_source=stage, contract_root=tmp_path, sandbox_factory=lambda *args: {}, output=tmp_path / "worker"
    )
    result = worker(tmp_path, tmp_path, Sentinel(), timeout_s=10)
    assert 0 < result["analysis_budget_seconds"] < 10
    receipt = json.loads(next((tmp_path / "worker").glob("*/receipt.json")).read_text())
    assert receipt["analysis_budget_seconds"] == result["analysis_budget_seconds"]
    assert receipt["analysis_budget_seconds"] < receipt["budget_seconds"] == 10


def test_host_worker_uses_injected_transport_and_records_both_commands(tmp_path, monkeypatch):
    import merlin.perf.analysis_worker as module

    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    stage = tmp_path / "trusted_stage.py"
    stage.write_text(
        "def analyze_whole_model_emission(*args, timeout_s, **kwargs):\n return {'ran': True, 'budget': timeout_s}\n"
    )
    observed = []

    def command(native):
        observed.append(list(native))
        now[0] = 2.0
        # A real second command, without introducing a frozen-execution claim.
        return [native[0], "-B", *native[1:]]

    worker = IsolatedAnalysisWorker(
        analysis_source=stage,
        contract_root=tmp_path,
        sandbox_factory=lambda *args: {},
        output=tmp_path / "worker",
        python_command=command,
    )
    assert worker(tmp_path, tmp_path, Sentinel(), timeout_s=10) == {"ran": True, "budget": 7.2}
    receipt = json.loads(next((tmp_path / "worker").glob("*/receipt.json")).read_text())
    assert len(observed) == 1
    assert observed[0][1:3] == ["-m", "merlin.perf.analysis_worker"]
    assert receipt["native_argv"] == observed[0]
    assert receipt["transport_argv"] == [observed[0][0], "-B", *observed[0][1:]]
    assert receipt["status"] == "completed" and receipt["process_group_cleanup"] is True
    assert receipt["analysis_budget_seconds"] == 7.2


def test_host_worker_transport_refusal_records_failure_without_spawning(tmp_path, monkeypatch):
    import merlin.perf.analysis_worker as module

    def refuse(native):
        raise ValueError("frozen source identity changed")

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not spawn after refusal"))
    worker = IsolatedAnalysisWorker(
        analysis_source=tmp_path / "stage.py",
        contract_root=tmp_path,
        sandbox_factory=lambda *args: {},
        output=tmp_path / "worker",
        python_command=refuse,
    )
    with pytest.raises(ValueError, match="frozen source identity changed"):
        worker(tmp_path, tmp_path, Sentinel(), timeout_s=10)
    receipt = json.loads(next((tmp_path / "worker").glob("*/receipt.json")).read_text())
    assert receipt["status"] == "failed" and receipt["worker_pid"] is None
    assert receipt["transport_argv"] is None
    assert receipt["process_group_cleanup"] is False


def test_host_worker_transport_preparation_spends_the_existing_deadline(tmp_path, monkeypatch):
    import merlin.perf.analysis_worker as module

    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: pytest.fail("deadline already exhausted"))

    def command(native):
        now[0] = 11.0
        return native

    worker = IsolatedAnalysisWorker(
        analysis_source=tmp_path / "stage.py",
        contract_root=tmp_path,
        sandbox_factory=lambda *args: {},
        output=tmp_path / "worker",
        python_command=command,
    )
    with pytest.raises(TimeoutError, match="preparation exhausted"):
        worker(tmp_path, tmp_path, Sentinel(), timeout_s=10)
    receipt = json.loads(next((tmp_path / "worker").glob("*/receipt.json")).read_text())
    assert receipt["worker_pid"] is None
    assert receipt["wall_seconds"] == 11.0 and receipt["budget_seconds"] == 10


def test_worker_accepts_host_only_static_budget_above_simulation_ceiling(tmp_path):
    from merlin.perf.execution_policy import FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS, ITERATION_MAX_SECONDS

    stage = tmp_path / "trusted_stage.py"
    stage.write_text(
        "def analyze_whole_model_emission(*args, timeout_s, **kwargs):\n"
        " return {'analysis_budget_seconds': timeout_s}\n"
    )
    worker = IsolatedAnalysisWorker(
        analysis_source=stage, contract_root=tmp_path, sandbox_factory=lambda *args: {}, output=tmp_path / "worker"
    )

    maximum = FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS
    assert maximum > ITERATION_MAX_SECONDS, "host-only analysis must retain its separate, larger budget"
    result = worker(tmp_path, tmp_path, Sentinel(), timeout_s=maximum)

    assert maximum - 10 <= result["analysis_budget_seconds"] < maximum
    receipts = list((tmp_path / "worker").glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["budget_seconds"] == maximum
    assert receipt["analysis_budget_seconds"] == result["analysis_budget_seconds"]
    with pytest.raises(ValueError, match=f"at most {maximum:g}s"):
        worker(tmp_path, tmp_path, Sentinel(), timeout_s=maximum + 1)
    assert list((tmp_path / "worker").glob("*/receipt.json")) == receipts


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
