"""Plan or run independent stock-TVM campaigns against one shared FireSim pool.

Dry-run is the default. The installation owns active-cycle observation and
remote cancellation; this launcher never provisions hardware or changes memory.
"""

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from firesim_job_pool import checked, identity


def qualification(path):
    pool = json.loads(Path(path).read_text())
    if not pool.get("observer_qualification"):
        raise ValueError("Missing live observer qualification; total-cycle caps do not qualify")
    proof = json.loads(checked(pool["observer_qualification"]).read_text())
    if proof.get("cycle_limit_scope") != "active_measured" or proof.get("limit_refresh_scope") != "in_flight":
        raise ValueError("Installed observer does not qualify live active-cycle pruning")
    sources = {identity(checked(s))["sha256"] for s in pool["adapter_sources"]}
    if sources != set(proof.get("adapter_source_sha256", [])):
        raise ValueError("Observer tests belong to different installed adapter sources")
    if pool.get("deployment_id") != proof.get("deployment_id"):
        raise ValueError("Observer qualification belongs to another deployment")
    for name in (
        "startup_exclusion",
        "same_elf_uncapped_agreement",
        "inflight_tightening_abort",
        "remote_cancellation",
    ):
        result = json.loads(checked(proof["tests"][name]).read_text())
        if result.get("test") != name or result.get("status") != "pass":
            raise ValueError("Missing actual observer control: " + name)
    return pool


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux MemAvailable is required for build admission")


def reserve_campaigns(path, rows, root, *, deadline, clock_hz, pool, cap,
                      finish_active_runs=False, active_run_timeout_seconds=None):
    """Persist one deadline/proposal budget across launcher invocations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        binding = dict(
            deadline_epoch=deadline,
            target_clock_hz=clock_hz,
            pool_sha256=identity(pool)["sha256"],
            proposal_cap=cap,
            finish_active_runs=finish_active_runs,
            active_run_timeout_seconds=active_run_timeout_seconds,
        )
        state = json.loads(path.read_text()) if path.exists() else dict(binding=binding, campaigns={})
        if state["binding"] != binding:
            raise ValueError("Shared study deadline, target clock, pool, cap or completion policy changed; do not reset a continuation")
        fresh = []
        for row in rows:
            record = dict(
                program_sha256=identity(root / row["program"])["sha256"],
                fixture_sha256=identity(root / row["fixture"])["sha256"],
                trials=row["trials"],
                workload_keys=row.get("workload_keys", []),
                graph_options={key: row.get(key, default) for key, default in
                               (("graph_fusion_level", 2), ("prepack_weights", True),
                                ("bias_preload", True), ("graph_cse", False))},
            )
            old = state["campaigns"].get(row["id"])
            if old is not None:
                if old != record:
                    raise ValueError("An existing campaign identity changed")
                continue
            state["campaigns"][row["id"]] = record
            fresh.append(row)
        if sum(r["trials"] for r in state["campaigns"].values()) > cap:
            raise ValueError("Cumulative proposal reservations exceed the shared study cap")
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)
        return fresh


def command(row, root, args, output, deadline):
    argv = [
        sys.executable,
        str(Path(__file__).with_name("tune_firesim_chunks.py")),
        "--program",
        str(root / row["program"]),
        "--fixture",
        str(root / row["fixture"]),
        "--pool",
        str(args.pool.resolve()),
        "--toolchain",
        str(args.toolchain.resolve()),
        "--output-dir",
        str(output),
        "--deadline-epoch",
        str(deadline),
        "--target-clock-hz",
        str(args.target_clock_hz),
        "--trials",
        str(row["trials"]),
        "--batch",
        str(row.get("batch", 4)),
        "--builders",
        "1",
        "--graph-fusion-level",
        str(row.get("graph_fusion_level", 2)),
    ]
    if getattr(args, "finish_active_runs", False):
        argv.extend(["--finish-active-runs", "--active-run-timeout-seconds",
                     str(args.active_run_timeout_seconds)])
    if row.get("graph_cse", False):
        argv.append("--graph-cse")
    if not row.get("prepack_weights", True):
        argv.append("--no-prepack-weights")
    if not row.get("bias_preload", True):
        argv.append("--no-bias-preload")
    for key in row.get("workload_keys", []):
        argv.extend(["--workload-key", key])
    return argv


def stop_owned(process):
    """Terminate this campaign and descendant adapter groups, including setsid children.

    Remote cancellation on these signals must be covered by the installed
    observer's qualified cancellation test; killing SSH alone is insufficient.
    """
    if process.poll() is not None:
        return
    pending, owned = [process.pid], set()
    while pending:
        pid = pending.pop()
        if pid in owned:
            continue
        owned.add(pid)
        try:
            pending.extend(map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split()))
        except FileNotFoundError:
            pass
    groups = set()
    for pid in owned:
        try:
            group = os.getpgid(pid)
            if group in owned and group != os.getpgrp():
                groups.add(group)
        except ProcessLookupError:
            pass
    for group in sorted(groups, key=lambda g: g == process.pid):
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "pool", "toolchain", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-clock-hz", type=float, required=True)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--memory-reserve-gib", type=float, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finish-active-runs", action="store_true")
    parser.add_argument("--active-run-timeout-seconds", type=float,
                        help="Per-run hang watchdog supplied by the remote worker; separate from tuning budget")
    parser.add_argument("--phase", choices=("initial", "continuation", "all"), default="initial")
    parser.add_argument("--ledger", type=Path, help="Same ledger path for all study phases and restarts")
    args = parser.parse_args()
    if not math.isfinite(args.target_clock_hz) or args.target_clock_hz <= 0:
        parser.error("Provide the documented target cycle-counter frequency, not FPGA/emulation frequency")
    if args.finish_active_runs and (args.active_run_timeout_seconds is None
                                   or not math.isfinite(args.active_run_timeout_seconds)
                                   or args.active_run_timeout_seconds <= 0):
        parser.error("Finish-active mode requires a separate positive per-run watchdog")
    manifest = json.loads(args.study.read_text())
    root = args.study.resolve().parent
    rows = [
        r
        for r in manifest["campaigns"]
        if r["status"] == "ready_for_functional_build"
        and (args.phase == "all" or r.get("phase", "initial") == args.phase)
    ]
    if sum(r["trials"] for r in rows) > manifest["global_proposal_cap"]:
        parser.error("Campaign proposals exceed the shared study cap")
    keys = [key for row in rows for key in row.get("workload_keys", [])]
    if len(set(keys)) != len(keys):
        parser.error("A prepared workload has duplicate tuning owners")
    for row in rows:
        for field in ("program", "fixture"):
            identity(root / row[field])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = [dict(id=r["id"], argv=command(r, root, args, args.output_dir / r["id"], args.deadline_epoch)) for r in rows]
    (args.output_dir / "commands.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute:
        print(
            json.dumps(
                dict(
                    status="planned",
                    campaigns=len(plan),
                    proposals=sum(r["trials"] for r in rows),
                    skipped=[r["id"] for r in manifest["campaigns"] if r not in rows],
                ),
                indent=2,
            )
        )
        return
    if args.deadline_epoch is None or not time.time() < args.deadline_epoch <= time.time() + 14400:
        parser.error("Use the same future absolute deadline, at most four hours away, for the entire handoff study")
    pool = qualification(args.pool)
    if not pool.get("slots"):
        parser.error("No assigned physical slots")
    if args.ledger is None:
        parser.error("Supply one shared --ledger for the full study")
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    supervisor_lock = args.ledger.with_suffix(args.ledger.suffix + ".supervisor.lock").open("a+")
    try:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("One supervisor owns this ledger; combine parallel campaigns in that study")
    fresh = reserve_campaigns(
        args.ledger,
        rows,
        root,
        deadline=args.deadline_epoch,
        clock_hz=args.target_clock_hz,
        pool=args.pool,
        cap=manifest["global_proposal_cap"],
        finish_active_runs=args.finish_active_runs,
        active_run_timeout_seconds=args.active_run_timeout_seconds,
    )
    fresh_ids = {r["id"] for r in fresh}
    plan = [p for p in plan if p["id"] in fresh_ids]
    cpu_count = len(os.sched_getaffinity(0))
    reserve = args.memory_reserve_gib * 1024**3
    capacity = max(
        1,
        min(
            2 * len(pool["slots"]),
            max(1, cpu_count // 2),
            int(max(0, memory_available() - reserve) // (1536 * 1024**2)),
        ),
    )
    limit = min(capacity, args.jobs) if args.jobs else capacity
    if limit < 1:
        parser.error("Use at least one job")
    pending, active, completed = list(plan), [], []
    progress = {r["id"]: dict(offset=0, proposals=0, valid_timings=0, buffer="") for r in plan}
    stopped, admission_closed = None, False
    try:
        while pending or active:
            if time.time() >= args.deadline_epoch:
                admission_closed = True
                if not args.finish_active_runs:
                    stopped = "shared deadline"
                    break
                if not active:
                    break
            if memory_available() < 1024**3:
                stopped = "critical memory pressure"
                break
            while (pending and not admission_closed and time.time() < args.deadline_epoch
                   and len(active) < limit and memory_available() > reserve + 1536 * 1024**2):
                item = pending.pop(0)
                log = (args.output_dir / (item["id"] + ".log")).open("w")
                process = subprocess.Popen(item["argv"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((item, process, log))
            for item, process, log in list(active):
                trial_log = args.output_dir / item["id"] / "trials.jsonl"
                stats = progress[item["id"]]
                if trial_log.exists():
                    with trial_log.open() as stream:
                        stream.seek(stats["offset"])
                        data = stats["buffer"] + stream.read()
                        stats["offset"] = stream.tell()
                    lines = data.split("\n")
                    stats["buffer"] = lines.pop()
                    for line in lines:
                        if line:
                            trial = json.loads(line)
                            stats["proposals"] += 1
                            stats["valid_timings"] += int(
                                trial.get("cycles") is not None and not trial.get("run_error")
                            )
                if process.poll() is not None:
                    completed.append(dict(id=item["id"], returncode=process.returncode))
                    log.close()
                    active.remove((item, process, log))
            state = dict(
                deadline_epoch=args.deadline_epoch,
                admission_closed=admission_closed,
                finish_active_runs=args.finish_active_runs,
                active=[dict(id=i["id"], pid=p.pid) for i, p, _ in active],
                completed=completed,
                pending=[i["id"] for i in pending],
                memory_available=memory_available(),
                concurrency=limit,
                progress={
                    name: {key: value for key, value in values.items() if key in ("proposals", "valid_timings")}
                    for name, values in progress.items()
                },
            )
            (args.output_dir / "status.json").write_text(json.dumps(state, indent=2) + "\n")
            time.sleep(1)
    finally:
        for _, process, log in active:
            stop_owned(process)
            log.close()
    (args.output_dir / "exit.json").write_text(
        json.dumps(
            dict(
                reason=stopped or ("admission closed; active runs drained" if admission_closed else "campaigns finished"),
                post_deadline_seconds=max(0, time.time() - args.deadline_epoch),
                completed=completed,
                unfinished=[i["id"] for i in pending] + [i["id"] for i, _, _ in active],
            ),
            indent=2,
        )
        + "\n"
    )
    if stopped or any(r["returncode"] for r in completed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
