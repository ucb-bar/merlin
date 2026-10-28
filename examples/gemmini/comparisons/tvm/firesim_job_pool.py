"""Bounded jobs on an explicitly supplied, already provisioned FireSim slot list.

An installation-specific adapter consumes request.json and writes result.json.
It owns the actual manager API, boot evidence, guest cap, and remote cancellation.
This module never allocates instances, flashes images, or chooses schedules.
"""

import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from firesim_timeout import positive_seconds


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def checked(item):
    if identity(item["path"])["sha256"] != item["sha256"]:
        raise ValueError("Changed FireSim evidence: " + item["path"])
    return Path(item["path"])


class FireSimPool:
    def __init__(self, config, *, deadline_epoch, finish_active_runs=False, active_run_timeout_seconds=None):
        self.config = json.loads(Path(config).read_text())
        self.config_identity = identity(config)
        self.hardware = (
            identity(self.config["hardware_receipt"])
            if self.config.get("hardware_receipt")
            else dict(
                deployment_id=self.config.get("deployment_id"),
                expected_memory_model="DDR3",
                source="user-provided bitstream",
            )
        )
        if not self.config.get("adapter_sources"):
            raise ValueError("Pin the installed manager wrapper's source bytes")
        for source in self.config["adapter_sources"]:
            checked(source)
        self.deadline = float(deadline_epoch)
        self.finish_active_runs = finish_active_runs
        if finish_active_runs and (active_run_timeout_seconds is None
                                  or not math.isfinite(active_run_timeout_seconds)
                                  or active_run_timeout_seconds <= 0):
            raise ValueError("Finish-active mode requires a separate positive per-run watchdog")
        self.active_run_timeout_seconds = active_run_timeout_seconds
        self.slots = self.config["slots"]
        if not self.slots or len({s["lock"] for s in self.slots}) != len(self.slots):
            raise ValueError("Supply distinct locks for already assigned physical slots")
        if not self.config.get("persistent_manager") or not self.config.get("adapter_cancels_remote_on_timeout"):
            raise ValueError("Require a persistent manager and qualified remote cancellation")
        for slot in self.slots:
            if not Path(slot["lock"]).is_absolute() or not slot["command"]:
                raise ValueError("Use a shared absolute slot lock and an explicit adapter argv")

    def run(self, elf, folder, *, wall_timeout=None):
        """Return a verified warm interval, or raise; failures never become costs."""
        elf, folder = Path(elf).resolve(), Path(folder).resolve()
        folder.mkdir(parents=True, exist_ok=False)
        if "path" in self.hardware:
            checked(self.hardware)
        for source in self.config["adapter_sources"]:
            checked(source)
        lease, admitted_at = None, None
        launched, admitted_monotonic = False, None
        try:
            while lease is None:
                if time.time() >= self.deadline:
                    raise TimeoutError("Study deadline reached while waiting for a slot")
                for slot in self.slots:
                    path = Path(slot["lock"])
                    if path.with_suffix(path.suffix + ".quarantine").exists():
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handle = path.open("a+")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        continue
                    # Another owner may quarantine the slot between our first
                    # check and acquiring its lease. Never reuse that slot.
                    if path.with_suffix(path.suffix + ".quarantine").exists():
                        fcntl.flock(handle, fcntl.LOCK_UN)
                        handle.close()
                        continue
                    lease = handle
                    break
                if lease is None:
                    if all(Path(s["lock"]).with_suffix(Path(s["lock"]).suffix + ".quarantine").exists() for s in self.slots):
                        raise RuntimeError("All slots are quarantined pending confirmed remote cancellation")
                    time.sleep(0.1)
            admitted_at = time.time()
            admitted_monotonic = time.monotonic()
            if admitted_at >= self.deadline:
                admitted_at = None
                raise TimeoutError("Study admission deadline reached")
            wall = (self.active_run_timeout_seconds if self.finish_active_runs
                    else self.deadline - admitted_at)
            if wall_timeout is not None:
                wall = positive_seconds(wall_timeout() if callable(wall_timeout) else wall_timeout)
            execution_deadline = admitted_at + wall
            request = dict(
                schema="gemmini_firesim_probe_request_v4",
                elf=identity(elf),
                hardware=self.hardware,
                slot=slot["id"],
                timing_policy="fixed-warm",
                timing_scope="complete second model_run including completion fence",
                expected_memory_model="DDR3",
                configure_memory=False,
                startup_included=False,
                timeout_policy="median_wall_v1",
                incumbent_cycle_pruning=False,
                wall_seconds=wall,
                deadline_epoch=execution_deadline,
                admission_deadline_epoch=self.deadline,
                finish_active_runs=self.finish_active_runs,
                require_all_outputs=True,
                keep_manager_alive=True,
                scope="small workload experiment; not EL4 complete-chain qualification",
            )
            request_path, result_path = folder / "request.json", folder / "result.json"
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            argv = [
                arg.format_map(dict(request=str(request_path), result=str(result_path), slot=slot["id"]))
                for arg in slot["command"]
            ]
            with (folder / "adapter.log").open("w") as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                launched = True
                try:
                    code = process.wait(timeout=max(0.001, execution_deadline - time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise TimeoutError("FireSim wall watchdog expired; unconfirmed remote cancellation quarantines the slot")
            if code:
                raise RuntimeError("FireSim adapter failed; see " + str(folder / "adapter.log"))
            result = json.loads(result_path.read_text())
            if (
                result.get("status") != "passed"
                or result.get("guest_exit") != 0
                or result.get("elf_sha256") != request["elf"]["sha256"]
                or not result.get("wall_limit_enforced")
                or result.get("remote_stopped") is not True
            ):
                raise ValueError("FireSim result lacks matching ELF, deadline enforcement or successful guest exit")
            if "sha256" in self.hardware and result.get("hardware_sha256") != self.hardware["sha256"]:
                raise ValueError("FireSim result differs from the supplied deployment receipt")
            log = checked(result["transcript"]).read_text()
            boot = json.loads(checked(result["boot_evidence"]).read_text())
            if boot.get("elf_sha256") != request["elf"]["sha256"]:
                raise ValueError("Actual boot-file evidence differs from submitted ELF")
            samples = re.findall(r"^DSE_CYCLES first=(\d+) second=(\d+)$", log, re.M)
            if len(samples) != 1 or "DSE_PASS" not in log.splitlines():
                raise ValueError("Require one fixed-warm pair and every-output pass marker")
            first, warm = map(int, samples[0])
            if first <= 0 or warm <= 0:
                raise ValueError("Require positive measured active cycles")
            if result.get("active_cycles", warm) != warm:
                raise ValueError("Result cost differs from the guest active-cycle window")
            return warm
        finally:
            if launched:
                stopped = False
                try:
                    final = json.loads(result_path.read_text())
                    stopped = (final.get("remote_stopped") is True
                               and final.get("elf_sha256") == request["elf"]["sha256"])
                except (ValueError, OSError):
                    pass
                if not stopped:
                    marker = Path(slot["lock"]).with_suffix(Path(slot["lock"]).suffix + ".quarantine")
                    marker.write_text(json.dumps(dict(reason="Remote termination unconfirmed", request=identity(request_path))) + "\n")
            if lease is not None:
                fcntl.flock(lease, fcntl.LOCK_UN)
                lease.close()
            if admitted_at is not None:
                finished_at = time.time()
                (folder / "occupancy.json").write_text(json.dumps(dict(
                    admitted_at=admitted_at, finished_at=finished_at,
                    admission_deadline_epoch=self.deadline,
                    occupied_seconds=max(0, time.monotonic() - admitted_monotonic),
                    post_budget_seconds=max(0, finished_at - self.deadline),
                    finish_active_runs=self.finish_active_runs,
                    scope="local slot lease/adapter duration, including failed jobs; remote ledger owns campaign totals",
                ), indent=2) + "\n")
