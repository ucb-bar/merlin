"""Explicit portfolio source freezing and resource-bounded worker supervision."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.perf.host_resources import (
    HostResourcePolicy,
    HostResourceTripwire,
    sample_host_memory,
    summarize_samples,
    violations,
)
from merlin_experiments import frozen_python
from merlin_experiments import source_snapshot as perf_snapshot
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2.portfolio_options import PortfolioInvocation, worker_arguments


@dataclass(frozen=True)
class PortfolioDeployment:
    """Caller-selected deployment; native layout discovery remains at its launch edge."""

    source_root: Path
    output_root: Path
    lease_path: Path
    worker_entrypoint: tuple[str, ...]
    snapshot_options: dict[str, Any]
    target_name: str
    selected_provider: dict[str, Any] | None
    declared_inputs: dict[str, Path]
    inherited_environment: dict[str, str]
    worker_python_roots: tuple[str, ...]
    declared_input_sha256: dict[str, str] | None = None


def acquire_host_resource_lease(stage_root: Path, *, lease_path: Path):
    """Exclude another heavy compiler experiment without claiming machine-wide ownership."""
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lease_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        stage_root.mkdir(parents=True, exist_ok=True)
        P2_CONTRACTS.write_json(
            stage_root / "host_resource_telemetry.json",
            {
                "schema": "host_resource_telemetry_v1",
                "status": "launch_refused",
                "reasons": ["host_resource_lease_busy"],
                "lease": str(lease_path),
                "worker_returncode": None,
            },
        )
        return None
    except BaseException:
        handle.close()
        raise
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "stage_root": str(stage_root)}) + "\n")
        handle.flush()
    except BaseException:
        handle.close()
        raise
    return handle


def _descendant_pids(root_pid: int, proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    """Snapshot descendants without depending on a process-management package."""
    children: dict[int, list[int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            line = next(row for row in (entry / "status").read_text().splitlines() if row.startswith("PPid:"))
            parent = int(line.partition(":")[2].strip())
        except (OSError, StopIteration, ValueError):
            continue
        children.setdefault(parent, []).append(int(entry.name))
    result, pending = [], list(children.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        result.append(pid)
        pending.extend(children.get(pid, ()))
    return tuple(result)


def _signal_worker_tree(process: subprocess.Popen, sig: signal.Signals) -> None:
    # Some nested tools create their own sessions. Signal descendants directly before the source
    # worker's group so those sessions cannot outlive a resource-triggered controller shutdown.
    for pid in reversed(_descendant_pids(process.pid)):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def _stop_worker_tree(process: subprocess.Popen, *, grace_s: float = 5.0) -> None:
    known_descendants = set(_descendant_pids(process.pid))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        known_descendants.update(_descendant_pids(process.pid))
        for pid in reversed(tuple(known_descendants)):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        _signal_worker_tree(process, sig)
        try:
            process.wait(timeout=grace_s)
            # A nested new-session child is reparented when the leader exits; retain its original PID.
            for pid in reversed(tuple(known_descendants)):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            return
        except subprocess.TimeoutExpired:
            continue


def run_resource_guarded_worker(
    command: list[str],
    environment: dict[str, str],
    *,
    stage_root: Path,
    policy: HostResourcePolicy,
    sample_seconds: float,
) -> int:
    """Supervise the complete experiment tree and retain only memory extrema/endpoints."""
    samples = [sample_host_memory()]
    initial_reasons = violations(samples[-1], policy)
    if initial_reasons:
        stage_root.mkdir(parents=True, exist_ok=True)
        P2_CONTRACTS.write_json(
            stage_root / "host_resource_telemetry.json",
            {
                "schema": "host_resource_telemetry_v1",
                "status": "launch_refused",
                "reasons": list(initial_reasons),
                "policy": policy.record(),
                "summary": summarize_samples(samples),
                "worker_returncode": None,
            },
        )
        return 75
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    guard = HostResourceTripwire(policy)
    stop_decision = None
    returncode = None
    try:
        while returncode is None:
            try:
                returncode = process.wait(timeout=sample_seconds)
            except subprocess.TimeoutExpired:
                samples.append(sample_host_memory())
                decision = guard.observe(samples[-1])
                if decision["status"] == "stop":
                    stop_decision = dict(decision)
                    _stop_worker_tree(process)
                    returncode = process.returncode
    except BaseException:
        _stop_worker_tree(process)
        raise
    finally:
        samples.append(sample_host_memory())
        stage_root.mkdir(parents=True, exist_ok=True)
        P2_CONTRACTS.write_json(
            stage_root / "host_resource_telemetry.json",
            {
                "schema": "host_resource_telemetry_v1",
                "status": "resource_limit" if stop_decision is not None else "completed",
                "policy": policy.record(),
                "sample_period_seconds": sample_seconds,
                "summary": summarize_samples(samples),
                "trip": stop_decision,
                "worker_returncode": returncode,
            },
        )
    return 75 if stop_decision is not None else int(returncode)


def launch(invocation: PortfolioInvocation, *, deployment: PortfolioDeployment) -> int:
    """Freeze once, publish transport provenance and supervise the selected worker."""
    args = invocation.args
    if args.source_worker:
        raise ValueError("portfolio launch requires a parent invocation, not a source worker")
    resource_policy = invocation.resource_policy
    source = deployment.source_root
    snapshot = args.output.resolve().with_name(args.output.name + ".source")
    lease = acquire_host_resource_lease(args.output.resolve(), lease_path=deployment.lease_path)
    if lease is None:
        return 75
    try:
        before_snapshot = sample_host_memory()
        pressure = violations(before_snapshot, resource_policy)
        if pressure:
            args.output.resolve().mkdir(parents=True, exist_ok=True)
            P2_CONTRACTS.write_json(
                args.output.resolve() / "host_resource_telemetry.json",
                {
                    "schema": "host_resource_telemetry_v1",
                    "status": "launch_refused",
                    "reasons": list(pressure),
                    "policy": resource_policy.record(),
                    "summary": summarize_samples([before_snapshot]),
                    "worker_returncode": None,
                },
            )
            return 75
        provider = deployment.selected_provider
        perf_snapshot.create(
            source,
            snapshot,
            output_root=deployment.output_root,
            target_name=deployment.target_name,
            provider=provider,
            **deployment.snapshot_options,
            declared_inputs=deployment.declared_inputs,
        )
        receipt = perf_snapshot.verify(snapshot)
        for name, digest in (deployment.declared_input_sha256 or {}).items():
            declared = receipt.get("declared_inputs", {}).get(name, {})
            if (
                name not in deployment.declared_inputs
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or receipt.get("files", {}).get(declared.get("snapshot")) != digest
            ):
                raise ValueError(f"portfolio declared input changed after admission: {name}")
        environment = {
            **deployment.inherited_environment,
            **perf_snapshot.provider_environment(snapshot, receipt),
            "MERLIN_REPO_ROOT": str(snapshot),
            "MERLIN_OUT_ROOT": str(deployment.output_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(snapshot / item) for item in deployment.worker_python_roots),
        }
        command = [*deployment.worker_entrypoint, *worker_arguments(args)]
        transport = frozen_python.python_command(snapshot, command, verifier_source=Path(perf_snapshot.__file__))
        transport_root = args.output.resolve().with_name(args.output.name + ".transport")
        if transport_root.is_symlink():
            raise ValueError("transport receipt directory cannot be a symlink")
        transport_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=transport_root))
        perf_snapshot.seal(
            attempt,
            "transport_launch",
            {
                "schema": "merlin.frozen-python-transport-launch.v1",
                "role": "current trusted launch instrumentation; not historical grading authority",
                "stage_output": str(args.output.resolve()),
                "source_snapshot": str(snapshot),
                "native_argv": command,
                "transport_argv": transport,
                "limits": (
                    "Only this process and explicitly migrated host Python descendants are guarded; "
                    "external engines and candidate tools are separate dependencies."
                ),
            },
        )

        return run_resource_guarded_worker(
            transport,
            environment,
            stage_root=args.output.resolve(),
            policy=resource_policy,
            sample_seconds=args.resource_sample_seconds,
        )
    finally:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        finally:
            lease.close()
