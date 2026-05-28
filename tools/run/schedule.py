#!/usr/bin/env python3
"""Execute a multi-model schedule on any aarch64-linux board (typically
QRB5165) by launching one `iree-run-module` per model instance, taskset-
pinned to the schedule's `hardware_target` cluster, with start-time gating
to honor the scheduled launch times.

Model-agnostic. Pass `--model <name>:<vmfb>:<shape>` once per model
instance; nothing in this script is tied to a specific model.

This is a pragmatic counterpart to the C++ on-device scheduler runner at
`samples/common/xpu-rt/scheduler_runner.cc` — that runner needs an IREE
API port for newer iree_bar versions (post `iree_hal_device_create_params_t`
refactor), while this Python driver works with the static
iree-run-module already cross-built and deployed on the board. Per-process
startup overhead is significant (~50–100 ms per launch), so this is best
used as a **correctness demonstration** of the schedule on real hardware
rather than a high-precision timing harness. For tight timing use
iree-benchmark-module against the same VMFBs.

The schedule input is the combined_schedule.json produced by
`XPU-RT/scripts/merlin_adapter.py multi`. For each model INSTANCE
(distinguished by `job_name` + `instance` fields), this script:

    1. Picks the assigned cluster from the start-of-instance dispatch's
       `hardware_target` (assumes all dispatches in an instance share the
       same target, which the greedy scheduler typically produces; mixed
       targets per-instance need the C runner).
    2. Maps cluster to a CPU pin via --cluster-cpu (board-dependent default
       like 'CPU_P:7,CPU_E:0' for QRB5165; override per board).
    3. At the instance's earliest start time, spawns
       `taskset -c <cpu> iree-run-module --module=<model.vmfb>
       --device=local-task --input=<dummy>...`.
    4. Reports per-instance wall-clock and aggregate makespan.

Usage:

    python tools/run/schedule.py \
        --schedule <merlin_dir>/breakdowns/combined_schedule.json \
        --model "<job_name>:<vmfb_path>:<input_shape>" \
        --model "<job_name>:<vmfb_path>:<input_shape>" \
        --remote-dir /root/iree_run/<run_name>
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shlex
import subprocess
import sys
import threading
import time
from typing import Any

_LOG = logging.getLogger("run_schedule_on_board")

_DEFAULT_SSH_HOST = "root@10.44.120.201"
_DEFAULT_SSH_KEY = pathlib.Path("~/.ssh/DIMA_SLICE").expanduser()


def _ssh(host: str, key: pathlib.Path, cmd: str, timeout: float = 60.0) -> tuple[int, str, str]:
    args = ["ssh", "-i", str(key), "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", host, cmd]
    res = subprocess.run(args, capture_output=True, text=True, check=False, timeout=timeout)
    return res.returncode, res.stdout, res.stderr


def _scp(key: pathlib.Path, src: pathlib.Path, dst: str, timeout: float = 120.0) -> None:
    args = ["scp", "-O", "-i", str(key), "-q", str(src), dst]
    subprocess.run(args, check=True, timeout=timeout)


def push_models(
    models: dict[str, dict],
    *,
    ssh_host: str,
    ssh_key: pathlib.Path,
    remote_dir: pathlib.Path,
    runtime_bin: pathlib.Path,
) -> None:
    """One-time push of the runtime binary + each model's full VMFB."""
    _ssh(ssh_host, ssh_key, f"mkdir -p {shlex.quote(str(remote_dir))}/models")
    _LOG.info("pushing runtime: %s", runtime_bin)
    _scp(ssh_key, runtime_bin, f"{ssh_host}:{remote_dir}/iree-run-module")
    _ssh(ssh_host, ssh_key, f"chmod +x {shlex.quote(str(remote_dir))}/iree-run-module")
    for name, m in models.items():
        local = pathlib.Path(m["vmfb"])
        if not local.exists():
            raise FileNotFoundError(f"vmfb not found: {local}")
        remote = f"{remote_dir}/models/{name}.vmfb"
        _LOG.info("  push %s -> %s", local, remote)
        _scp(ssh_key, local, f"{ssh_host}:{remote}")


def collect_instances(schedule: dict) -> dict[tuple[str, int], dict[str, Any]]:
    """Group dispatches by (job_name, instance) and pick the earliest
    start_time + the dominant hardware_target. Mixed hardware_target
    within an instance means the schedule wants the C runner — we fall
    back to the majority cluster and log a warning so the user knows the
    comparison is approximate."""
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for name, e in schedule["dispatches"].items():
        job = e.get("job_name", "unknown")
        inst = e.get("instance", 0)
        # The combined_schedule.json sometimes drops `instance` when there
        # was only one — try to recover it from the per-dispatch name
        # prefix shape "<job><inst>_<rest>".
        if "instance" not in e and name.startswith(job):
            tail = name[len(job) :]
            try:
                inst = int(tail.split("_", 1)[0])
            except ValueError:
                inst = 0
        key = (job, inst)
        slot = grouped.setdefault(
            key,
            {
                "start_time_us": float("inf"),
                "targets": [],
                "dispatch_count": 0,
            },
        )
        slot["start_time_us"] = min(slot["start_time_us"], e.get("start_time_us", e.get("start_time", 0) * 1000.0))
        slot["targets"].append(e.get("hardware_target", "CPU_P"))
        slot["dispatch_count"] += 1
    # Reduce targets to the dominant one per instance.
    for slot in grouped.values():
        targets = slot.pop("targets")
        # Counter-style mode without importing collections (keeps deps tight).
        counts: dict[str, int] = {}
        for t in targets:
            counts[t] = counts.get(t, 0) + 1
        dominant = max(counts.items(), key=lambda kv: kv[1])[0]
        slot["hardware_target"] = dominant
        slot["target_distribution"] = counts
    return grouped


def _launch_instance(
    job: str,
    inst: int,
    start_us: float,
    target: str,
    *,
    models: dict[str, dict],
    cluster_cpu: dict[str, str],
    ssh_host: str,
    ssh_key: pathlib.Path,
    remote_dir: pathlib.Path,
    t0: float,
    results: dict[tuple[str, int], dict[str, Any]],
) -> None:
    delay_s = (start_us / 1e6) - (time.perf_counter() - t0)
    if delay_s > 0:
        time.sleep(delay_s)

    if job not in models:
        results[(job, inst)] = {"error": f"no model for job '{job}'"}
        return
    m = models[job]
    cpu_mask = cluster_cpu.get(target, "0")
    # Construct dummy --input flags from the model's declared input shape.
    inputs = " ".join(f"--input={s}=0" for s in m["inputs"])
    cmd = (
        f"cd {shlex.quote(str(remote_dir))} && "
        f"taskset -c {shlex.quote(cpu_mask)} ./iree-run-module "
        f"--module=models/{shlex.quote(job)}.vmfb "
        f"--device=local-task --task_topology_max_group_count=1 "
        f"{inputs} 2>&1"
    )
    t_launch = time.perf_counter()
    rc, stdout, stderr = _ssh(ssh_host, ssh_key, cmd, timeout=120.0)
    t_done = time.perf_counter()
    results[(job, inst)] = {
        "rc": rc,
        "wall_us": (t_done - t_launch) * 1e6,
        "actual_start_us": (t_launch - t0) * 1e6,
        "scheduled_start_us": start_us,
        "target": target,
        "cpu_mask": cpu_mask,
        "stderr_tail": stderr[-200:] if stderr else "",
        "stdout_tail": stdout[-200:] if stdout else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=pathlib.Path)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="`<job_name>:<vmfb_path>:<input_shape>[,...]`. "
        "Repeatable. Input shapes are MLIR-style "
        "without 'tensor<>' (e.g. 1x3x112x112xf32).",
    )
    parser.add_argument("--remote-dir", default=pathlib.Path("/root/iree_run/multi_demo"), type=pathlib.Path)
    parser.add_argument("--ssh-host", default=_DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-key", default=_DEFAULT_SSH_KEY, type=pathlib.Path)
    parser.add_argument(
        "--runtime-bin", required=True, type=pathlib.Path, help="Cross-built iree-run-module to push to the board."
    )
    parser.add_argument(
        "--cluster-cpu",
        default="CPU_P:7,CPU_E:0",
        help="Comma-separated cluster:cpu pairs. The CPU is " "passed verbatim to taskset -c.",
    )
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    models: dict[str, dict] = {}
    for spec in args.model:
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(f"bad --model '{spec}'; expected " f"<job>:<vmfb>:<shape>[,<shape>]")
        job, vmfb = parts[0], parts[1]
        shapes = ":".join(parts[2:]).split(",")
        models[job] = {"vmfb": vmfb, "inputs": shapes}

    cluster_cpu = dict(p.split(":", 1) for p in args.cluster_cpu.split(","))

    schedule = json.loads(args.schedule.read_text())
    instances = collect_instances(schedule)
    print(f"plan: {len(instances)} instances across " f"{len({j for j, _ in instances})} jobs")
    for (job, inst), slot in sorted(instances.items()):
        print(
            f"  {job}[{inst}] start={slot['start_time_us']/1000:.1f}ms "
            f"target={slot['hardware_target']} "
            f"({slot['dispatch_count']} dispatches; "
            f"distrib={slot['target_distribution']})"
        )

    if not args.skip_push:
        push_models(
            models,
            ssh_host=args.ssh_host,
            ssh_key=args.ssh_key,
            remote_dir=args.remote_dir,
            runtime_bin=args.runtime_bin,
        )

    # Spawn one thread per instance — each blocks until its scheduled
    # start time, then ssh-launches iree-run-module. Threads make
    # concurrent ssh sessions, which the board's sshd handles fine.
    results: dict[tuple[str, int], dict[str, Any]] = {}
    threads: list[threading.Thread] = []
    t0 = time.perf_counter()
    for (job, inst), slot in instances.items():
        thr = threading.Thread(
            target=_launch_instance,
            args=(
                job,
                inst,
                slot["start_time_us"],
                slot["hardware_target"],
            ),
            kwargs={
                "models": models,
                "cluster_cpu": cluster_cpu,
                "ssh_host": args.ssh_host,
                "ssh_key": args.ssh_key,
                "remote_dir": args.remote_dir,
                "t0": t0,
                "results": results,
            },
            daemon=True,
        )
        threads.append(thr)
        thr.start()
    for thr in threads:
        thr.join(timeout=180.0)

    print()
    print(f"{'instance':22} {'sched ms':>9} {'actual ms':>10} " f"{'wall ms':>9} {'rc':>3} {'cpu':>4}")
    print("-" * 70)
    max_actual_finish = 0.0
    for (job, inst), r in sorted(results.items()):
        line = f"{job}[{inst}]"
        sched = r.get("scheduled_start_us", 0) / 1000
        actual = r.get("actual_start_us", 0) / 1000
        wall = r.get("wall_us", 0) / 1000
        rc = r.get("rc", "?")
        cpu = r.get("cpu_mask", "?")
        print(f"{line:22} {sched:>9.1f} {actual:>10.1f} {wall:>9.1f} " f"{rc:>3} {cpu:>4}")
        if isinstance(rc, int) and rc == 0:
            max_actual_finish = max(max_actual_finish, actual + wall)
    if max_actual_finish:
        print(f"actual board makespan: {max_actual_finish:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
