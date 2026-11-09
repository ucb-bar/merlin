#!/usr/bin/env python3
"""Trusted K1 process monitor: drop the capsule to nobody and record independent OS evidence."""
from __future__ import annotations

import ctypes
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _drop_privileges(harts: int) -> None:
    os.setsid()
    os.sched_setaffinity(0, set(range(harts)))
    resource.setrlimit(resource.RLIMIT_AS, (1536 * 1024**2, 1536 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ctypes.CDLL(None).prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)


def _task_sample(
        pid: int) -> tuple[dict[int, tuple[int, int, int, str, str | None]], int, str | None]:
    """Sample every task with high-resolution scheduled time and its task-local affinity.

    Linux process status reports only the calling thread's mask. Reading each task's status avoids
    mistaking an eight-CPU process mask for eight pinned workers; ``schedstat`` also avoids the
    coarse clock-tick granularity of fields 14/15 in ``stat`` for short-lived kernels.
    """
    values: dict[int, tuple[int, int, int, str, str | None]] = {}
    pending, observed_pids = [pid], set()
    rss_kb, affinities = 0, set()
    while pending:
        process_id = pending.pop()
        if process_id in observed_pids:
            continue
        observed_pids.add(process_id)
        task_root = Path(f"/proc/{process_id}/task")
        try:
            tids = list(task_root.iterdir())
        except OSError:
            continue
        for entry in tids:
            try:
                text = (entry / "stat").read_text(encoding="utf-8")
                rest = text[text.rfind(")") + 2:].split()
                ticks = int(rest[11]) + int(rest[12])
                processor = int(rest[36])
                state = rest[0]
                runtime_ns = int((entry / "schedstat").read_text(
                    encoding="utf-8").split()[0])
                task_affinity = None
                for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                    if line.startswith("Cpus_allowed_list:"):
                        task_affinity = line.split(":", 1)[1].strip()
                        break
                values[int(entry.name)] = (
                    ticks, processor, runtime_ns, state, task_affinity)
            except (OSError, ValueError, IndexError):
                continue
        try:
            children = (task_root / str(process_id) / "children").read_text(
                encoding="utf-8").split()
            pending.extend(int(value) for value in children)
        except (OSError, ValueError):
            pass
        try:
            for line in Path(f"/proc/{process_id}/status").read_text(
                    encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_kb += int(line.split()[1])
                elif line.startswith("Cpus_allowed_list:"):
                    affinities.add(line.split(":", 1)[1].strip())
        except (OSError, ValueError):
            pass
    affinity = next(iter(affinities)) if len(affinities) == 1 else None
    return values, rss_kb, affinity


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: k1_monitor.py BINARY SEED HARTS")
    binary, seed, harts = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3])
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C", "LC_ALL": "C",
           "OMP_NUM_THREADS": str(harts), "OMP_PROC_BIND": "true", "OMP_DYNAMIC": "false"}
    started = time.monotonic_ns()
    # Temporary files avoid the classic monitor deadlock where an untrusted child fills a PIPE while
    # the parent is busy sampling /proc. RLIMIT_FSIZE bounds each child stream to 16 MiB.
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen([str(binary), seed], stdout=stdout_file, stderr=stderr_file,
                                   env=env, start_new_session=False,
                                   preexec_fn=lambda: _drop_privileges(harts))
        previous: dict[int, tuple[int, int, int, str, str | None]] = {}
        active_tids: set[int] = set()
        cpus_observed: set[int] = set()
        active_cpus: set[int] = set()
        pinned_affinities_observed: set[int] = set()
        pinned_runtime_cpus: set[int] = set()
        running_cpus_observed: set[int] = set()
        tids_observed: set[int] = set()
        max_tasks = peak_rss_kb = max_simultaneous_running_cpus = 0
        affinity_samples: set[str] = set()
        timed_out = False
        while process.poll() is None:
            sample, rss_kb, affinity = _task_sample(process.pid)
            max_tasks = max(max_tasks, len(sample)); peak_rss_kb = max(peak_rss_kb, rss_kb)
            tids_observed.update(sample)
            cpus_observed.update(value[1] for value in sample.values())
            if affinity:
                affinity_samples.add(affinity)
            running_cpus = {value[1] for value in sample.values() if value[3] == "R"}
            running_cpus_observed.update(running_cpus)
            max_simultaneous_running_cpus = max(
                max_simultaneous_running_cpus, len(running_cpus))
            for tid, (_, cpu, runtime_ns, _, task_affinity) in sample.items():
                if task_affinity is not None and task_affinity.isdigit():
                    pinned_cpu = int(task_affinity)
                    pinned_affinities_observed.add(pinned_cpu)
                    if runtime_ns > 0 and cpu == pinned_cpu:
                        pinned_runtime_cpus.add(cpu)
                if tid in previous and runtime_ns > previous[tid][2]:
                    active_tids.add(tid); active_cpus.add(cpu)
            previous = sample
            if time.monotonic_ns() - started > 75_000_000_000:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.001)
        process.wait(timeout=5)

        def tail(stream, limit: int = 8000) -> str:
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode(errors="replace")

        stdout, stderr = tail(stdout_file), tail(stderr_file)
    report = {
        "version": 1, "returncode": process.returncode, "timed_out": timed_out,
        "wall_ns": time.monotonic_ns() - started, "requested_harts": harts,
        "max_tasks": max_tasks, "tids_observed": len(tids_observed),
        "active_tids": len(active_tids), "cpus_observed": sorted(cpus_observed),
        "active_cpus": sorted(active_cpus), "affinity_samples": sorted(affinity_samples),
        "pinned_affinities_observed": sorted(pinned_affinities_observed),
        "pinned_runtime_cpus": sorted(pinned_runtime_cpus),
        "running_cpus_observed": sorted(running_cpus_observed),
        "max_simultaneous_running_cpus": max_simultaneous_running_cpus,
        "peak_rss_kb": peak_rss_kb, "child_stdout": stdout, "child_stderr": stderr,
    }
    print("MERLIN_K1_MONITOR " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
