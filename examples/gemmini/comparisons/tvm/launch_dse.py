#!/usr/bin/env python3
"""Supervise stock TVM campaigns and host resources; never choose schedules."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from merlin.common.artifacts import new_product


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def memory():
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"))
    }


def process_tree(process):
    pending, result, seen = [process.pid], [], set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        root = Path("/proc") / str(pid)
        try:
            # Workers are launched from Python threads, so include every task's
            # children rather than only the main thread's child list.
            for task in (root / "task").iterdir():
                pending.extend(int(value) for value in (task / "children").read_text().split())
            status = dict(line.split(":", 1) for line in (root / "status").read_text().splitlines())
            result.append(
                dict(
                    pid=pid,
                    name=status["Name"].strip(),
                    rss_bytes=int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
                    state=status["State"].strip(),
                )
            )
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def read_progress(path, progress):
    """Read only complete new records; never retain large schedule traces."""
    if path.exists():
        with path.open() as stream:
            stream.seek(progress["offset"])
            while True:
                line = stream.readline()
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                progress["trials"] += 1
                progress["valid_timings"] += row.get("cycles") is not None
                progress["offset"] = stream.tell()
    return {key: progress[key] for key in ("trials", "valid_timings")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvm", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--toolchain", required=True, type=Path)
    parser.add_argument("--resnet", required=True, type=Path)
    parser.add_argument("--resnet-fixture", type=Path)
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        help="Optional JSON list of {name, source_args, scope}; passed to stock task tuning",
    )
    parser.add_argument(
        "--execution-pool", type=Path, help="Join an existing shared pool without replacing its resource configuration"
    )
    parser.add_argument("--work-dir", type=Path, help="Resume this supervisor product, preserving campaign roots")
    parser.add_argument("--cpus", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    parser.add_argument("--build-slots", type=int, default=2)
    parser.add_argument("--reserve-gib", type=float, default=4)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--trials-per-task", type=int, default=32)
    parser.add_argument("--rtl-seconds", type=int, default=7200)
    parser.add_argument(
        "--deadline-epoch", type=float, help="Absolute wall deadline; a product policy may impose an earlier deadline"
    )
    parser.add_argument(
        "--bounded-dse",
        action="store_true",
        help="Use moving target-cycle cutoffs and the compact native search profile",
    )
    args = parser.parse_args(argv)
    cpus = sorted(os.sched_getaffinity(0))
    if not 1 <= args.cpus <= len(cpus) or args.build_slots < 1 or args.reserve_gib < 2:
        parser.error("Choose available CPUs, positive builder slots and at least 2 GiB reserve")
    root = args.work_dir
    if root is None:
        root = new_product(
            "perf-studies",
            target="gemmini-stock-tvm-dse",
            version=1,
            notes="Stock MetaSchedule campaigns; shared host admission; latency-pipe memory",
        ).path
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    deadline_epoch = args.deadline_epoch
    policy = root / "policy.json"
    if policy.exists():
        pinned = json.loads(policy.read_text())["deadline_epoch"]
        if deadline_epoch is not None and deadline_epoch > pinned:
            parser.error("Cannot extend the frozen product deadline")
        deadline_epoch = min(deadline_epoch, pinned) if deadline_epoch is not None else pinned
    if args.bounded_dse and deadline_epoch is None:
        parser.error("--bounded-dse requires an absolute deadline or frozen policy.json")
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        parser.error("DSE wall deadline already reached")
    pool = args.execution_pool.resolve() if args.execution_pool else root / "host-pool"
    pool.mkdir(exist_ok=True)
    if (root / "STOP").exists() or (pool / "STOP").exists():
        parser.error("This product has an explicit STOP marker; use a fresh product")
    if args.execution_pool:
        shared_config = json.loads((pool / "config.json").read_text())
        if shared_config.get("deadline_epoch") is not None:
            deadline_epoch = (
                min(deadline_epoch, shared_config["deadline_epoch"])
                if deadline_epoch
                else shared_config["deadline_epoch"]
            )
    else:
        write_json(
            pool / "config.json",
            dict(
                cpus=cpus[-args.cpus :],
                build_slots=args.build_slots,
                min_available_bytes=int(args.reserve_gib * 2**30),
                deadline_epoch=deadline_epoch,
            ),
        )
    chain = json.loads(args.toolchain.read_text())
    common = [
        str(args.python),
        "-u",
        str(args.tvm / "apps/gemmini/tune_models.py"),
        "--execution-pool",
        str(pool),
        "--baselines-after-search",
        "--strategy",
        "evolutionary",
        "--timing-policy",
        "fixed-warm",
        "--parallel",
        str(args.parallel),
        "--builders",
        str(args.build_slots),
        "--baseline-workers",
        str(args.parallel),
        "--trials-per-iteration",
        str(args.parallel),
        "--trials-per-task",
        str(args.trials_per_task),
        "--seed",
        "20261025",
        "--max-cycles",
        "40000000",
        "--cpu-max-cycles",
        "2000000",
        "--validation-overhead-cycles",
        "1000000",
        "--build-seconds",
        "1200",
        "--rtl-seconds",
        str(args.rtl_seconds),
    ]
    if args.bounded_dse:
        common.remove("--baselines-after-search")
        common += [
            "--moving-cutoff",
            "--compact-search",
            "--incumbent-baseline-only",
            "--no-cpu",
            "--build-seconds",
            "180",
            "--max-cycles",
            "10000000",
            "--cap-factor",
            "1",
        ]
    if deadline_epoch is not None:
        common += ["--deadline-epoch", str(deadline_epoch)]
    for key in (
        "gcc",
        "adapter_receipt",
        "spike_receipt",
        "hardware_receipt",
        "memory_variant_receipt",
        "firesim_memory_artifact",
        "memory_pin_registry",
    ):
        common += ["--" + key.replace("_", "-"), str(chain[key])]
    resnet = ["--program", str(args.resnet)]
    if args.resnet_fixture:
        resnet += ["--input-fixture", str(args.resnet_fixture)]
    campaigns = [
        ("resnet-real", resnet, "fresh actual full ResNet export; all eligible tasks"),
        (
            "tinyllama-decode",
            ["--model", "tinyllama-decode"],
            "synthetic full-dimension decode operators; not the selected EL4 export",
        ),
        (
            "smolvla-expert",
            ["--model", "smolvla-expert"],
            "synthetic full-dimension expert operators; not the selected EL4 export",
        ),
    ]
    if args.campaign_manifest:
        campaigns = [
            (row["name"], row["source_args"], row["scope"]) for row in json.loads(args.campaign_manifest.read_text())
        ]
        if (
            not campaigns
            or len({row[0] for row in campaigns}) != len(campaigns)
            or any(
                not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name)
                for name, _, _ in campaigns
            )
        ):
            parser.error("Campaign names must be distinct nonempty lowercase names")
    commands = {name: common + source + ["--work-dir", str(root / name)] for name, source, _ in campaigns}
    write_json(
        root / "launch.json",
        dict(
            schema="gemmini_stock_tvm_supervisor_v1",
            commands=commands,
            deadline_epoch=deadline_epoch,
            scopes={name: scope for name, _, scope in campaigns},
            host=dict(cpu_count=len(cpus), selected_cpus=cpus[-args.cpus :], memory=memory()),
            configuration={
                key: str(value) if isinstance(value, Path) else value
                for key, value in (vars(args) | {"work_dir": root}).items()
            },
        ),
    )
    print(str(root), flush=True)
    stopped = False
    deadline_reached = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    children, logs = {}, []
    pending = [name for name, _, _ in campaigns]
    progress = {name: dict(offset=0, trials=0, valid_timings=0) for name in pending}
    try:
        while True:
            if deadline_epoch is not None and time.time() >= deadline_epoch - 2:
                deadline_reached = True
                (root / "STOP").write_text("Four-hour wall deadline reached; remaining DSE deferred to handoff\n")
            stopped = stopped or (root / "STOP").exists()
            if pending and not stopped and memory()["MemAvailable"] >= (args.reserve_gib + 1) * 2**30:
                name = pending.pop(0)
                log = (root / f"{name}.log").open("a")
                logs.append(log)
                children[name] = subprocess.Popen(
                    commands[name], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            rows = {}
            for name, child in children.items():
                processes = process_tree(child)
                rows[name] = dict(
                    pid=child.pid,
                    returncode=child.poll(),
                    processes=processes,
                    deferred=(root / name / "deferred.json").exists(),
                    rss_bytes=sum(row["rss_bytes"] for row in processes),
                    **read_progress(root / name / "trials.jsonl", progress[name]),
                )
            host = memory()
            if host["MemAvailable"] < 2**30:
                (root / "STOP").write_text("Emergency stop: available host memory below 1 GiB\n")
            stopped = stopped or (root / "STOP").exists()
            finished = not pending and all(child.poll() is not None for child in children.values())
            status = "stopping" if stopped else "running"
            if finished:
                status = "failed" if any(child.returncode for child in children.values()) else "complete"
                if status == "complete" and any(row["deferred"] for row in rows.values()):
                    status = "complete_with_deferrals"
            state = dict(
                time=datetime.now(timezone.utc).isoformat(),
                memory=host,
                status="deadline_reached" if deadline_reached else status,
                deadline_epoch=deadline_epoch,
                pending=pending,
                campaigns=rows,
            )
            write_json(root / "status.json", state)
            with (root / "host-history.jsonl").open("a") as stream:
                stream.write(json.dumps(state) + "\n")
            if stopped or finished:
                break
            time.sleep(1 if deadline_epoch is not None else 10)
    finally:
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + (1 if deadline_reached else 10)
        for child in children.values():
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()
        write_json(
            root / "exit.json",
            dict(
                stopped=stopped,
                deadline_reached=deadline_reached,
                deadline_epoch=deadline_epoch,
                returncodes={name: child.returncode for name, child in children.items()},
            ),
        )
    return 1 if stopped or any(child.returncode for child in children.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
