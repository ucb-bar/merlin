"""Run one measured process group and observe its CPU and memory interval."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .paper_measurement_protocol import _sha_bytes


def _proc_state(
    pid: int, requested: set[int], previous_ticks: Mapping[int, int]
) -> tuple[int, set[int], int, set[int], dict[int, int], dict[int, int]]:
    rss = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                rss = max(rss, int(line.split()[1]) * 1024)
        affinity = set(os.sched_getaffinity(pid))
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
        active_cores: set[int] = set()
        current_ticks: dict[int, int] = {}
        delta_by_core = {core: 0 for core in requested}
        for task in tasks:
            children = (task / "children").read_text(encoding="utf-8").split()
            if children:
                raise ValueError(
                    "measured executable spawned an unmonitored child process; paper runners "
                    "must keep all work in the observed thread group"
                )
            task_affinity = set(os.sched_getaffinity(int(task.name)))
            if not task_affinity or not task_affinity <= requested:
                raise ValueError("a measured task escaped controller-owned CPU affinity")
            stat = (task / "stat").read_text(encoding="utf-8")
            fields_after_name = stat[stat.rfind(")") + 2 :].split()
            if len(fields_after_name) <= 36:
                raise ValueError("measured task stat omits scheduler/CPU-time fields")
            state = fields_after_name[0]
            ticks = int(fields_after_name[11]) + int(fields_after_name[12])
            tid = int(task.name)
            running_core = int(fields_after_name[36])
            if running_core not in requested:
                raise ValueError("a measured task ran outside controller-owned CPU affinity")
            current_ticks[tid] = ticks
            delta = max(0, ticks - int(previous_ticks.get(tid, ticks)))
            delta_by_core[running_core] += delta
            # A sleeping thread's last CPU is not concurrent work.  Count a core only when the
            # scheduler reports the task runnable/running and externally observed CPU time moved.
            if state == "R" and delta > 0:
                active_cores.add(running_core)
    except (FileNotFoundError, ProcessLookupError):
        return rss, set(), 0, set(), {}, {core: 0 for core in requested}
    return rss, affinity, len(tasks), active_cores, current_ticks, delta_by_core


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_iteration(
    argv: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    core_ids: list[int],
    observation: Path,
    timeout: float,
    phase: str,
    index: int,
    request: bytes | None = None,
    allow_stdout: bool = False,
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    if observation.exists():
        observation.unlink()
    requested = set(core_ids)
    with (
        tempfile.TemporaryFile() as stdin_file,
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        if request is not None:
            stdin_file.write(request)
            stdin_file.seek(0)
        started = time.monotonic_ns()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdin=stdin_file if request is not None else subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            preexec_fn=lambda: os.sched_setaffinity(0, requested),
            start_new_session=True,
        )
        peak_rss, observed_affinity, max_tasks, best_running_cores = 0, set(), 0, set()
        previous_ticks: dict[int, int] = {}
        cpu_ticks_by_core = {core: 0 for core in requested}
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("paper measurement exceeded its whole-cell deadline")
                rss, affinity, tasks, running_cores, ticks, deltas = _proc_state(process.pid, requested, previous_ticks)
                previous_ticks = ticks or previous_ticks
                peak_rss = max(peak_rss, rss)
                observed_affinity = affinity or observed_affinity
                max_tasks = max(max_tasks, tasks)
                for core, delta in deltas.items():
                    cpu_ticks_by_core[core] += delta
                if len(running_cores) > len(best_running_cores):
                    best_running_cores = running_cores
                time.sleep(0.0005)
        except BaseException:
            _terminate_process_group(process)
            raise
        ended = time.monotonic_ns()
        rss, affinity, tasks, running_cores, _ticks, deltas = _proc_state(process.pid, requested, previous_ticks)
        peak_rss = max(peak_rss, rss)
        observed_affinity = affinity or observed_affinity
        max_tasks = max(max_tasks, tasks)
        if len(running_cores) > len(best_running_cores):
            best_running_cores = running_cores
        for core, delta in deltas.items():
            cpu_ticks_by_core[core] += delta
        return_code = process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout, stderr = stdout_file.read(), stderr_file.read()
    if return_code != 0:
        raise ValueError(f"cell {phase} command failed rc={return_code}: {stderr.decode(errors='replace')[-500:]}")
    if request is None:
        if stdout and not allow_stdout:
            raise ValueError("measured executable may write only observation bytes, not evidence JSON")
        if not observation.is_file():
            raise ValueError("measured executable did not materialize controller-owned observation")
        output = observation.read_bytes()
    else:
        if observation.exists():
            raise ValueError("MRLNSES2 executable wrote an undeclared observation file")
        output = stdout
    if not output:
        raise ValueError("measured executable produced an empty observation")
    if observed_affinity != requested:
        raise ValueError("controller could not observe exact requested process affinity")
    if peak_rss <= 0:
        raise ValueError("controller could not observe positive RSS for measured process")
    if best_running_cores != requested:
        raise ValueError("controller did not observe simultaneous work on every requested core")
    row = {
        "phase": phase,
        "index": index,
        "started_monotonic_ns": started,
        "ended_monotonic_ns": ended,
        "elapsed_ns": ended - started,
        "exit_code": return_code,
        "stdout_sha256": _sha_bytes(stdout),
        "stderr_sha256": _sha_bytes(stderr),
        "observation_sha256": _sha_bytes(output),
        "peak_rss_bytes": peak_rss,
        "affinity_core_ids": sorted(observed_affinity),
        "max_task_count": max_tasks,
        "simultaneous_running_core_ids": sorted(best_running_cores),
        "cpu_time_ticks_by_core": {str(core): ticks for core, ticks in sorted(cpu_ticks_by_core.items())},
    }
    return row, output, stdout, stderr
