"""Shared host ownership of one optional calibration watcher per campaign.

This does not create pools, start/reset campaign clocks, authorize observations
or select a model. A stop request is cooperative: the currently owned native
operation finishes before its next boundary observes the request.
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from merlin.runtime import _native_files as F

_STOP = ContextVar("calibration_watcher_stop_file", default=None)


class StopRequested(RuntimeError):
    pass


@contextlib.contextmanager
def stop_scope(path):
    token = _STOP.set(Path(path) if path else None)
    try:
        yield
    finally:
        _STOP.reset(token)


def raise_if_stopping():
    path = _STOP.get()
    if path is not None and path.exists():
        raise StopRequested("calibration owner requested cooperative shutdown")


def _read(pin):
    path = F.canonical(Path(pin["path"]))
    before = F.pin(path, max_bytes=16 << 20)
    if before["sha256"] != pin["sha256"]:
        raise ValueError("calibration watcher profile changed")
    value = json.loads(path.read_bytes())
    if F.pin(path, max_bytes=16 << 20) != before:
        raise ValueError("calibration watcher profile changed during read")
    return value


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".next.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@dataclass
class Watcher:
    process: object | None
    report: dict
    directory: Path | None = None
    stop_file: Path | None = None
    lease: object | None = None
    shutdown_s: float = 1235

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    def _release(self):
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    def wait(self, *, timeout=None):
        if self.process is None:
            return 0
        result = self.process.wait(timeout=timeout)
        self._release()
        return result

    def poll(self):
        result = self.process.poll() if self.process is not None else 0
        if result is not None:
            self._release()
        return result

    def stop(self):
        """Drain the actual current operation; never kill its simulator child."""
        if self.process is None:
            return self.report
        if self.poll() is None:
            _write(self.stop_file, {"reason": "watcher host lifecycle finished", "at_epoch": time.time()})
            try:
                self.process.wait(timeout=self.shutdown_s)
            except subprocess.TimeoutExpired as error:
                # Keep the original lease/owner. The enclosing native guardian
                # must resolve this ownership failure before tearing down brokers.
                self.report.update(status="shutdown_unknown", reason="current native operation did not drain")
                _write(self.directory / "owner.json", self.report)
                raise RuntimeError("calibration watcher shutdown UNKNOWN; retain broker/native ownership") from error
        self._release()
        self.report.update(status="stopped", returncode=self.process.returncode, stopped_at=time.time())
        _write(self.directory / "owner.json", self.report)
        return self.report


def start(*, state_path, descriptor, repo, python, environment, output, directory,
          cpus=None, profile=None, request_epoch=True, poll_seconds=30):
    """Spawn against the exact admitted profile and existing shared resources."""
    if state_path is None:
        return Watcher(None, dict(status="skipped_no_campaign", reason="no selected joint campaign"))
    state_path = F.canonical(Path(state_path))
    state = json.loads(state_path.read_bytes())
    selected = state.get("policy", {}).get("native_calibration")
    if not selected:
        return Watcher(None, dict(status="skipped_no_profile", reason="native_calibration is not selected"))
    if profile is not None and any(profile.get(k) != selected.get(k) for k in ("path", "sha256")):
        raise ValueError("watcher profile differs from the admitted native_calibration selection")
    _read(selected)
    if state.get("started_at_epoch") is None:
        raise ValueError("calibration watcher requires the already-started admitted campaign clock")
    if not 5 <= poll_seconds <= 60:
        raise ValueError("calibration watcher polling must be five to sixty seconds")
    repo = F.canonical(Path(repo))
    descriptor = F.canonical(Path(descriptor))
    env = dict(environment)
    env["MERLIN_JOINT_CAMPAIGN_STATE"] = str(state_path)
    pools = {k: env.get(k) for k in ("MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GSIM_SLOT_ROOT")}
    if any(not value or not Path(value).is_dir() for value in pools.values()):
        raise ValueError("calibration watcher requires the existing shared compute/gSIM pools")
    cpus = sorted(os.sched_getaffinity(0) if cpus is None else cpus)
    if not cpus or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("calibration watcher CPU selection is outside its admitted owner")
    lease = state_path.with_name(".joint_calibration_watcher.lock").open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return Watcher(None, dict(status="skipped_existing_owner", reason="campaign watcher already has an owner"))
    directory, output = Path(directory), Path(output)
    try:
        directory.mkdir(parents=True, exist_ok=False)
        stop_file = directory / "stop.json"
        command = ["/usr/bin/taskset", "-c", ",".join(map(str, cpus)), str(python), "-B", "-m",
            "merlin_experiments.phase1.joint_calibration_service", "--profile", selected["path"],
            "--profile-sha256", selected["sha256"], "--state", str(state_path), "--descriptor", str(descriptor),
            "--repo", str(repo), "--output", str(output), "--poll-seconds", str(poll_seconds),
            "--stop-file", str(stop_file)]
        if not request_epoch:
            command.append("--evidence-only")
        with (directory / "watcher.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lease.fileno(),))
        report = dict(status="watcher_started", pid=child.pid, argv=command, started_at=time.time(),
            profile=selected, state=str(state_path), cwd=str(repo), shared_pools=pools,
            request_epoch=request_epoch, owner=str(directory / "owner.json"),
            status_file=str(output / "status.json"))
        _write(directory / "owner.json", report)
        return Watcher(child, report, directory, stop_file, lease,
            max(120, state["policy"].get("gsim", {}).get("per_job_s", 1200)) + poll_seconds + 5)
    except BaseException:
        lease.close()
        raise
