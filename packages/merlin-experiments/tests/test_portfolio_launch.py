"""File-lock and launch lifecycle regressions with no real worker execution."""

import fcntl
import json
import os
import socket
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import portfolio_launch as L
from merlin_experiments.phase2 import portfolio_options as O

from merlin.perf.host_resources import HostMemorySample


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("launch tests cannot start or signal real processes or bind listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)
    monkeypatch.setattr(os, "kill", refuse)
    monkeypatch.setattr(os, "killpg", refuse)


def sample(*, available=64 * 1024**3):
    return HostMemorySample(1.0, 128 * 1024**3, available, 0, 0)


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    invocation = O.parse_invocation(
        [
            "--campaign-config",
            str(tmp_path / "campaign.json"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "stage"),
        ]
    )
    deployment = L.PortfolioDeployment(
        source_root=source,
        output_root=tmp_path / "storage",
        lease_path=tmp_path / "leases/compiler.lock",
        worker_entrypoint=(sys.executable, "-m", "synthetic_worker"),
        snapshot_options={"source_roots": ("python",), "python_roots": ("python",)},
        target_name="synthetic",
        selected_provider=None,
        declared_inputs={"descriptor": tmp_path / "descriptor.yaml"},
        inherited_environment={"SYNTHETIC_INHERITED": "kept", "PYTHONPATH": "/unselected"},
        worker_python_roots=("python", "helpers"),
    )
    return SimpleNamespace(root=tmp_path, invocation=invocation, deployment=deployment)


def assert_lease_released(case):
    with case.deployment.lease_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_worker_invocation_refuses_before_acquiring_lease(case):
    case.invocation.args.source_worker = True
    with pytest.raises(ValueError, match="parent invocation"):
        L.launch(case.invocation, deployment=case.deployment)
    assert not case.deployment.lease_path.exists()


def test_real_lease_contention_records_refusal_and_release_allows_next_owner(case):
    first = L.acquire_host_resource_lease(case.root / "first", lease_path=case.deployment.lease_path)
    assert first is not None
    try:
        assert L.acquire_host_resource_lease(case.root / "second", lease_path=case.deployment.lease_path) is None
        receipt = json.loads((case.root / "second/host_resource_telemetry.json").read_text())
        assert receipt["reasons"] == ["host_resource_lease_busy"]
        assert receipt["worker_returncode"] is None
    finally:
        first.close()
    assert_lease_released(case)


@pytest.mark.parametrize("refusal", ["lease", "pressure"])
def test_admission_refusal_never_copies_or_executes(case, monkeypatch, refusal):
    first = None
    if refusal == "lease":
        first = L.acquire_host_resource_lease(case.root / "first", lease_path=case.deployment.lease_path)
    monkeypatch.setattr(L, "sample_host_memory", lambda: sample(available=0))
    monkeypatch.setattr(L.perf_snapshot, "create", lambda *a, **k: pytest.fail("refusal copied sources"))
    monkeypatch.setattr(L, "run_resource_guarded_worker", lambda *a, **k: pytest.fail("refusal ran worker"))
    try:
        assert L.launch(case.invocation, deployment=case.deployment) == 75
        receipt = json.loads((case.root / "stage/host_resource_telemetry.json").read_text())
        assert receipt["status"] == "launch_refused"
        assert receipt["reasons"]
    finally:
        if first is not None:
            first.close()
    assert_lease_released(case)


def test_failed_snapshot_releases_lease_without_running_worker(case, monkeypatch):
    monkeypatch.setattr(L, "sample_host_memory", lambda: sample())

    def fail(*args, **kwargs):
        raise ValueError("synthetic snapshot failure")

    monkeypatch.setattr(L.perf_snapshot, "create", fail)
    monkeypatch.setattr(L, "run_resource_guarded_worker", lambda *a, **k: pytest.fail("failed copy ran worker"))
    with pytest.raises(ValueError, match="synthetic snapshot failure"):
        L.launch(case.invocation, deployment=case.deployment)
    assert_lease_released(case)


@pytest.mark.parametrize("captured", [None, "b" * 64])
def test_changed_admitted_input_refuses_before_transport_and_releases_lease(case, monkeypatch, captured):
    deployment = replace(case.deployment, declared_input_sha256={"descriptor": "a" * 64})
    monkeypatch.setattr(L, "sample_host_memory", lambda: sample())
    monkeypatch.setattr(L.perf_snapshot, "create", lambda *a, **k: None)
    receipt = {"declared_inputs": {"descriptor": {"snapshot": "inputs/descriptor"}}, "files": {}}
    if captured is not None:
        receipt["files"]["inputs/descriptor"] = captured
    monkeypatch.setattr(L.perf_snapshot, "verify", lambda *a: receipt)
    monkeypatch.setattr(
        L.perf_snapshot, "provider_environment", lambda *a: pytest.fail("changed input reached transport")
    )
    with pytest.raises(ValueError, match="changed after admission: descriptor"):
        L.launch(case.invocation, deployment=deployment)
    assert_lease_released(case)


@pytest.mark.parametrize("pinned", [False, True])
def test_launch_orders_capture_transport_receipt_and_exact_execution(case, monkeypatch, pinned):
    events = []
    snapshot = case.root / "stage.source"
    receipt = {"schema": "synthetic snapshot receipt"}
    if pinned:
        case.deployment = replace(case.deployment, declared_input_sha256={"descriptor": "a" * 64})
        receipt.update(
            declared_inputs={"descriptor": {"snapshot": "inputs/descriptor"}}, files={"inputs/descriptor": "a" * 64}
        )

    def sampled():
        events.append("sample")
        return sample()

    def create(source, destination, **kwargs):
        events.append("capture")
        assert source == case.deployment.source_root
        assert destination == snapshot
        assert kwargs == {
            "output_root": case.deployment.output_root,
            "target_name": "synthetic",
            "provider": None,
            **case.deployment.snapshot_options,
            "declared_inputs": case.deployment.declared_inputs,
        }
        # The launch owns the real lease throughout capture.
        with case.deployment.lease_path.open("a+") as rival:
            with pytest.raises(BlockingIOError):
                fcntl.flock(rival.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def verify(root):
        events.append("verify")
        assert root == snapshot
        return receipt

    def provider_environment(root, observed):
        events.append("provider_environment")
        assert root == snapshot and observed is receipt
        return {"MERLIN_TARGET_PATH": "/sealed/provider"}

    native = [*case.deployment.worker_entrypoint, *O.worker_arguments(case.invocation.args)]
    transport = ["synthetic-frozen-launch", *native]

    def frozen_command(root, command, *, verifier_source):
        events.append("transport")
        assert root == snapshot and command == native
        assert str(verifier_source) == L.perf_snapshot.__file__
        return transport

    def run(command, environment, **kwargs):
        events.append("execute")
        assert command == transport
        assert environment == {
            "SYNTHETIC_INHERITED": "kept",
            "MERLIN_TARGET_PATH": "/sealed/provider",
            "MERLIN_REPO_ROOT": str(snapshot),
            "MERLIN_OUT_ROOT": str(case.deployment.output_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(snapshot / root) for root in ("python", "helpers")),
        }
        assert kwargs == {
            "stage_root": case.root / "stage",
            "policy": case.invocation.resource_policy,
            "sample_seconds": case.invocation.args.resource_sample_seconds,
        }
        paths = list((case.root / "stage.transport").glob("attempt-*/*.json"))
        assert len(paths) == 1
        recorded = json.loads(paths[0].read_text())
        assert recorded["native_argv"] == native
        assert recorded["transport_argv"] == transport
        assert recorded["source_snapshot"] == str(snapshot)
        assert recorded["stage_output"] == str(case.root / "stage")
        assert not paths[0].stat().st_mode & 0o222
        return 7

    monkeypatch.setattr(L, "sample_host_memory", sampled)
    monkeypatch.setattr(L.perf_snapshot, "create", create)
    monkeypatch.setattr(L.perf_snapshot, "verify", verify)
    monkeypatch.setattr(L.perf_snapshot, "provider_environment", provider_environment)
    monkeypatch.setattr(L.frozen_python, "python_command", frozen_command)
    monkeypatch.setattr(L, "run_resource_guarded_worker", run)
    assert L.launch(case.invocation, deployment=case.deployment) == 7
    assert events == ["sample", "capture", "verify", "provider_environment", "transport", "execute"]
    assert_lease_released(case)


@pytest.mark.parametrize("pressure", [False, True])
def test_supervisor_preserves_completed_or_resource_limit_receipt(case, monkeypatch, pressure):
    samples = (
        iter([sample(), sample(available=0), sample(available=0), sample()])
        if pressure
        else iter(
            [
                sample(),
                sample(),
            ]
        )
    )
    monkeypatch.setattr(L, "sample_host_memory", lambda: next(samples))
    stops = []

    class Process:
        pid = 2**30
        returncode = None

        def __init__(self, command, **kwargs):
            assert command == ["synthetic-worker"]
            assert kwargs == {"env": {"ONLY": "selected"}, "start_new_session": True}

        def wait(self, timeout):
            assert timeout == 0.25
            if pressure:
                raise subprocess.TimeoutExpired("synthetic-worker", timeout)
            self.returncode = 3
            return 3

    def stop(process):
        stops.append(process.pid)
        process.returncode = -15

    monkeypatch.setattr(L.subprocess, "Popen", Process)
    monkeypatch.setattr(L, "_stop_worker_tree", stop)
    result = L.run_resource_guarded_worker(
        ["synthetic-worker"],
        {"ONLY": "selected"},
        stage_root=case.root / "stage",
        policy=case.invocation.resource_policy,
        sample_seconds=0.25,
    )
    assert result == (75 if pressure else 3)
    receipt = json.loads((case.root / "stage/host_resource_telemetry.json").read_text())
    assert receipt["status"] == ("resource_limit" if pressure else "completed")
    assert receipt["worker_returncode"] == (-15 if pressure else 3)
    assert receipt["summary"]["sample_count"] == (4 if pressure else 2)
    assert stops == ([2**30] if pressure else [])


def test_failed_lease_metadata_releases_real_lock(case, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic metadata failure")

    monkeypatch.setattr(L.json, "dumps", fail)
    with pytest.raises(ValueError, match="synthetic metadata failure"):
        L.acquire_host_resource_lease(case.root / "stage", lease_path=case.deployment.lease_path)
    assert_lease_released(case)


def test_lock_error_closes_opened_handle(case, monkeypatch):
    from pathlib import Path

    opened = []
    original = Path.open

    def tracked(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        opened.append(handle)
        return handle

    def fail(*args, **kwargs):
        raise OSError("synthetic lock failure")

    monkeypatch.setattr(Path, "open", tracked)
    monkeypatch.setattr(L.fcntl, "flock", fail)
    with pytest.raises(OSError, match="synthetic lock failure"):
        L.acquire_host_resource_lease(case.root / "stage", lease_path=case.deployment.lease_path)
    assert len(opened) == 1 and opened[0].closed


def test_unlock_error_still_closes_real_lease(case, monkeypatch):
    original = fcntl.flock

    def fail_unlock(fd, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError("synthetic unlock failure")
        return original(fd, operation)

    monkeypatch.setattr(L, "sample_host_memory", lambda: sample(available=0))
    with monkeypatch.context() as scoped:
        scoped.setattr(L.fcntl, "flock", fail_unlock)
        with pytest.raises(OSError, match="synthetic unlock failure"):
            L.launch(case.invocation, deployment=case.deployment)
    assert_lease_released(case)
