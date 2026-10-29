"""Operator-side health watchdog and quiet-host marker for reserved Phase 1 runs.

    run_watchdog.py watch [--unit UNIT ...] [--once]   # sample every 60 s, report, never remediate
    run_watchdog.py check                              # quiet host: exit 3 while a reserved run is live

``watch`` samples cheap signals at low priority: the unit's busy share of its reserved CPUs, how old
its latest completed grade is, processes waiting on file locks or the mount lock, host io pressure and
the squashfs/loop mount count. A condition held for HOLD_SAMPLES consecutive samples prints a plain
warning; the current picture is written to ``watchdog-status.json`` beside the reservation state
(``<task>/.tmp/<run-id>/operator/``), never inside the agent workspace. It exits once its units end.

``check`` is the quiet-host protocol: while a run is live, defer disk-wide scans, new squashfs/loop
mounts and anything else heavy on shared I/O. The launcher writes RUN_IN_PROGRESS.json (MARKER) for
its reservation and removes it at the end; active reservation cgroups count even without a marker.

Never reads /proc/locks: with a large lease table every reader stalls all lease operations host-wide.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

INTERVAL_S = 60
HOLD_SAMPLES = 5  # ~5 minutes at the default interval
BUSY_BELOW = 0.25  # reserved CPUs busy share ...
GRADE_STALE_S = 1200  # ... while the latest grade is older than this
LOCK_WAITERS = 10
MOUNT_WAITERS = 10
IO_FULL_AVG60 = 30.0
MOUNT_COUNT = 150
START_GRACE_S = 900
LOCK_WCHANS = ("percpu_rwsem_wait", "locks_lock_inode_wait", "flock_lock_inode_wait", "posix_lock_inode_wait")
MOUNT_WCHANS = ("m_start", "do_lock_mount", "lock_mount", "namespace_lock")
MARKER = Path(os.environ.get("MERLIN_RUN_MARKER", "/work/merlin/RUN_IN_PROGRESS.json"))
PROC, CGROUP, SLICE = Path("/proc"), Path("/sys/fs/cgroup"), "merlinphase1.slice"


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)  # noqa: UP017 -- also runs under the system Python 3.10


def _read(path: Path) -> str:
    if path == PROC / "locks":
        raise PermissionError("the watchdog never reads /proc/locks")
    return path.read_text()


def cpus(text: str) -> set[int]:
    result = set()
    for part in text.replace(" ", ",").strip().split(","):
        if part:
            low, _, high = part.partition("-")
            result.update(range(int(low), int(high or low) + 1))
    return result


def pressure(name: str) -> dict:
    """``{"some": {"avg10": ...}, "full": {...}}`` from /proc/pressure/<name>."""
    result = {}
    for line in _read(PROC / "pressure" / name).splitlines():
        kind, *fields = line.split()
        result[kind] = {key: float(value) for key, _, value in (f.partition("=") for f in fields)}
    return result


def mount_count() -> int:
    """squashfs or loop-device mounts visible to this process."""
    count = 0
    for line in _read(PROC / "self/mountinfo").splitlines():
        fstype, source = (line.partition(" - ")[2].split() + ["", ""])[:2]
        count += fstype == "squashfs" or source.startswith("/dev/loop")
    return count


def systemctl_show(unit: str, prop: str) -> str:
    done = subprocess.run(["systemctl", "show", unit, "-p", prop, "--value"], capture_output=True, text=True)
    return done.stdout.strip()


def unit_state(unit: str) -> tuple[Path | None, dict]:
    """The reservation state the unit's ExecStart names (``--state``), and its contents."""
    tokens = systemctl_show(unit + ".service", "ExecStart").split()
    if "--state" not in tokens:
        return None, {}
    path = Path(tokens[tokens.index("--state") + 1])
    try:
        return path, json.loads(path.read_text())
    except (OSError, ValueError):
        return path, {}


def runs_of(state_path: Path | None, state: dict) -> list[dict]:
    """``[{run_id, out_root}]`` from the campaign argv, or from both plans of a paired reservation."""
    plans = []
    argv = state.get("argv") or []
    if "--run-id" in argv:
        env = dict(item.partition("=")[::2] for item in argv if "=" in item and item.split("=")[0].isupper())
        plans.append({"run_id": argv[argv.index("--run-id") + 1], "environment": env})
    elif state_path is not None and (state_path.parent / "pair.json").is_file():
        pair = json.loads((state_path.parent / "pair.json").read_text())
        plans += [json.loads(Path(pin["path"]).read_text()) for pin in pair.get("plans", ())]
    return [
        {"run_id": plan["run_id"], "out_root": plan["environment"].get("MERLIN_OUT_ROOT")}
        for plan in plans
        if plan.get("run_id")
    ]


def grade_progress(run: dict, at: float) -> dict:
    """Latest completed grade time and case results written in the last 10 minutes."""
    out = Path(run["out_root"] or "/nonexistent")
    result = {"run_id": run["run_id"], "graded_at": None, "grade_file_age_s": None, "cases_last_10min": 0}
    for workspace in out.glob(f"build/agent-workspaces/*/*/*/{run['run_id']}/workspace"):
        grade = workspace / "qa/latest_completed_grade.json"
        if grade.is_file():
            result["grade_file_age_s"] = round(at - grade.stat().st_mtime)
            result["graded_at"] = json.loads(grade.read_text()).get("graded_at")
    for run_dir in (out / "runs").glob(f"*/*/{run['run_id']}"):
        for item in run_dir.glob("_qa_work/fruns_*/runs/*/*/capsule_result.json"):
            with contextlib.suppress(OSError):
                result["cases_last_10min"] += at - item.stat().st_mtime < 600
    return result


def sample_unit(unit: str, previous: dict | None, at: float) -> dict | None:
    group = CGROUP / SLICE / f"{unit}.service"
    if not group.is_dir():
        return None
    usage = int(
        next(line.split()[1] for line in _read(group / "cpu.stat").splitlines() if line.startswith("usage_usec"))
    )
    reserved = cpus(_read(group / "cpuset.cpus.effective"))
    states, wchans, sandbox_ages = {}, {}, []
    uptime = float(_read(PROC / "uptime").split()[0])
    tick = os.sysconf("SC_CLK_TCK")
    for procs in (group / "cgroup.procs", *group.rglob("*/cgroup.procs")):
        for pid in _read(procs).split():
            try:
                head, _, rest = _read(PROC / pid / "stat").rpartition(")")
                wchan = _read(PROC / pid / "wchan") or "0"
            except OSError:
                continue
            fields = rest.split()
            states[fields[0]] = states.get(fields[0], 0) + 1
            wchans[wchan] = wchans.get(wchan, 0) + 1
            if fields[0] == "D" and head.partition("(")[2] == "bwrap":
                sandbox_ages.append(uptime - int(fields[19]) / tick)
    sample = {
        "unit": unit,
        "cpus": len(reserved),
        "usage_usec": usage,
        "at": at,
        "busy_share": None,
        "processes": sum(states.values()),
        "states": states,
        "top_wchans": sorted(wchans.items(), key=lambda item: -item[1])[:8],
        "lock_waiters": sum(wchans.get(name, 0) for name in LOCK_WCHANS),
        "mount_waiters": sum(wchans.get(name, 0) for name in MOUNT_WCHANS),
        "sandbox_d_state": len(sandbox_ages),
        "sandbox_d_median_age_s": round(statistics.median(sandbox_ages), 1) if sandbox_ages else None,
    }
    if previous and at > previous["at"] and reserved:
        sample["busy_share"] = round((usage - previous["usage_usec"]) / 1e6 / (at - previous["at"]) / len(reserved), 3)
    return sample


def _age(stamp: str | None, at: float) -> float | None:
    return at - dt.datetime.fromisoformat(stamp).timestamp() if stamp else None


def conditions(unit: dict | None, runs: list[dict], host: dict, started: float, at: float) -> dict[str, str]:
    """Plain-language text for every warning condition true in this sample."""
    found = {}
    if host["io_full_avg60"] > IO_FULL_AVG60:
        found["io_pressure"] = (
            f"host I/O is saturated (full pressure {host['io_full_avg60']:.0f}% over 60 s): another process is "
            "likely scanning or copying on shared disk; find it (iotop/pidstat -d) and ask its owner to defer"
        )
    if host["mount_count"] > MOUNT_COUNT:
        found["mount_count"] = (
            f"{host['mount_count']} squashfs/loop mounts are present: sandbox starts serialize on the mount "
            "lock; unmount stale images (snap/loop) when the run allows"
        )
    if unit is None:
        return found
    name = unit["unit"]
    if unit["lock_waiters"] >= LOCK_WAITERS:
        found["lock_waiters"] = (
            f"{name}: {unit['lock_waiters']} processes wait on file locks/leases: something may be reading "
            "/proc/locks or holding a lease table lock; stop that reader (see sysctl fs.leases-enable)"
        )
    if unit["mount_waiters"] >= MOUNT_WAITERS or (unit["sandbox_d_median_age_s"] or 0) > 5:
        found["mount_waiters"] = (
            f"{name}: {unit['mount_waiters']} processes wait on the mount lock, {unit['sandbox_d_state']} sandbox "
            f"starts stuck (median {unit['sandbox_d_median_age_s']} s) with {host['mount_count']} squashfs/loop "
            "mounts on the host: sandbox starts are serializing on the mount namespace; avoid new mounts and "
            "processes that read mount tables in a loop"
        )
    stale = [min(value for value in (_age(run["graded_at"], at), at - started) if value is not None) for run in runs]
    if unit["busy_share"] is not None and unit["busy_share"] < BUSY_BELOW and stale and min(stale) > GRADE_STALE_S:
        found["idle_stale"] = (
            f"{name}: reserved CPUs only {unit['busy_share']:.0%} busy and the newest grade is "
            f"{min(stale) / 60:.0f} min old: the run is likely stalled on a host problem; check the other warnings"
        )
    return found


def host_sample() -> dict:
    return {
        "io_full_avg60": pressure("io")["full"]["avg60"],
        "cpu_some_avg60": pressure("cpu")["some"]["avg60"],
        "mount_count": mount_count(),
    }


def host_warnings() -> list[str]:
    """One-shot host checks for a launch preflight (warnings only)."""
    found = conditions(None, [], host_sample(), 0, 0)
    found.update({f"live:{entry['unit']}": f"reserved run live: {entry['unit']}" for entry in live_runs()})
    return list(found.values())


def watch(units: list[str], *, interval: float, once: bool, out: Path | None) -> int:
    known, previous, held, first = {}, {}, {}, time.monotonic()
    while True:
        names = units or sorted(p.name.removesuffix(".service") for p in (CGROUP / SLICE).glob("merlinphase1*.service"))
        at, host = time.time(), host_sample()
        samples = {}
        for name in names:
            if name not in known and (CGROUP / SLICE / f"{name}.service").is_dir():
                state_path, state = unit_state(name)
                started = state.get("acquired_at")
                known[name] = (state_path, runs_of(state_path, state), at - (_age(started, at) or 0))
            if name in known and (unit := sample_unit(name, previous.get(name), at)) is not None:
                samples[name], previous[name] = unit, unit
        if once and not samples:
            print(json.dumps({"host": host, "warnings": list(conditions(None, [], host, at, at).values())}, indent=1))
            return 0
        if once and not any(s["busy_share"] is not None for s in samples.values()):
            time.sleep(5)
            continue
        if not samples and (known or time.monotonic() - first > START_GRACE_S):
            print(f"{now():%H:%M:%SZ} watchdog: no watched unit is active; exiting", flush=True)
            return 0
        for name, unit in samples.items():
            state_path, runs, started = known[name]
            progress = [grade_progress(run, at) for run in runs]
            found = conditions(unit, progress, host, started, at)
            for key, text in found.items():
                held[(name, key)] = held.get((name, key), 0) + 1
                if held[(name, key)] == HOLD_SAMPLES or once:
                    print(f"{now():%H:%M:%SZ} WARNING {text}", flush=True)
            for key in [k for k in held if k[0] == name and k[1] not in found]:
                if held.pop(key) >= HOLD_SAMPLES:
                    print(f"{now():%H:%M:%SZ} cleared {name}: {key[1]}", flush=True)
            status = {
                "schema": "merlin.operator.run_watchdog.v1",
                "updated_at": now().isoformat(),
                "host": host,
                "unit": unit,
                "runs": progress,
                "warnings": [found[k] for k in found if held[(name, k)] >= HOLD_SAMPLES or once],
                "pending": [k for k in found if held[(name, k)] < HOLD_SAMPLES and not once],
            }
            target = out or (state_path.parent / "operator" if state_path else None)
            if once or target is None:
                print(json.dumps(status, indent=1))
            else:
                target.mkdir(parents=True, exist_ok=True)
                temporary = target / "watchdog-status.json.next"
                temporary.write_text(json.dumps(status, indent=1) + "\n")
                temporary.replace(target / "watchdog-status.json")
        if once:
            return 0
        time.sleep(interval)


def start_detached(unit: str, reserved: set[int], log: Path) -> int | None:
    """Start ``watch --unit`` at nice 19 on CPUs outside every reservation; it exits with the unit."""
    online = cpus(Path("/sys/devices/system/cpu/online").read_text())
    taken = set(reserved)
    for group in (CGROUP / SLICE).glob("*.service"):
        with contextlib.suppress(OSError):
            taken |= cpus(_read(group / "cpuset.cpus.effective"))
    free = sorted(online - taken)
    argv = ["nice", "-n", "19", *(["taskset", "-c", ",".join(map(str, free))] if free else [])]
    argv += [sys.executable, str(Path(__file__).resolve()), "watch", "--unit", unit]
    try:
        with open(log, "ab") as stream:
            return subprocess.Popen(
                argv, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
            ).pid
    except OSError as error:
        print(f"watchdog not started: {error}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- quiet-host marker


def _update_marker(change) -> None:
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    with open(MARKER.with_name(MARKER.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = json.loads(MARKER.read_text()).get("runs", {}) if MARKER.is_file() else {}
        change(runs)
        if not runs:
            MARKER.unlink(missing_ok=True)
            return
        document = {
            "schema": "merlin.operator.run_in_progress.v1",
            "updated_at": now().isoformat(),
            "runs": runs,
            "please": "defer disk-wide scans, new squashfs/loop mounts and /proc/locks reads; "
            "check with run_watchdog.py check",
        }
        temporary = MARKER.with_name(MARKER.name + ".next")
        temporary.write_text(json.dumps(document, indent=1) + "\n")
        temporary.replace(MARKER)


@contextlib.contextmanager
def announced(unit: str, reserved: set[int], max_seconds: int, run_ids: list[str]):
    """Hold a RUN_IN_PROGRESS entry for the duration of the block; marker errors never stop a launch."""
    entry = {
        "unit": unit,
        "run_ids": run_ids,
        "cpus": ",".join(map(str, sorted(reserved))),
        "started_at": now().isoformat(),
        "pid": os.getpid(),
        "expected_end_at": (now() + dt.timedelta(seconds=max_seconds)).isoformat(),
        "contact": os.environ.get("MERLIN_RUN_CONTACT") or f"{os.environ.get('USER')}@{socket.gethostname()}",
    }
    try:
        _update_marker(lambda runs: runs.__setitem__(unit, entry))
    except OSError as error:
        print(f"quiet-host marker not written: {error}", file=sys.stderr)
    try:
        yield entry
    finally:
        with contextlib.suppress(OSError):
            _update_marker(lambda runs: runs.pop(unit, None))


def live_runs() -> list[dict]:
    """Marker entries whose launcher or unit is alive, plus any active reservation cgroup."""
    entries = json.loads(MARKER.read_text()).get("runs", {}) if MARKER.is_file() else {}
    live = {}
    for unit, entry in entries.items():
        alive = (PROC / str(entry.get("pid"))).exists() or (CGROUP / SLICE / f"{unit}.service").is_dir()
        overdue = _age(entry.get("expected_end_at"), time.time())
        if alive and (overdue is None or overdue < 3600):
            live[unit] = entry
    for group in (CGROUP / SLICE).glob("merlinphase1*.service"):
        unit = group.name.removesuffix(".service")
        with contextlib.suppress(OSError):
            live.setdefault(unit, {"unit": unit, "cpus": _read(group / "cpuset.cpus.effective").strip()})
    return list(live.values())


def check() -> int:
    live = live_runs()
    if not live:
        print("quiet host: no reserved run is live")
        return 0
    for entry in live:
        print(
            f"RUN IN PROGRESS: {entry['unit']} {entry.get('run_ids', '')} on CPUs {entry['cpus']}, "
            f"ends ~{entry.get('expected_end_at', '?')}, contact {entry.get('contact', '?')}"
        )
    print(
        "defer disk-wide scans (du/find/rg over shared trees, storage prune), new squashfs/loop mounts and "
        "any /proc/locks read until it ends",
        file=sys.stderr,
    )
    return 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    watching = sub.add_parser("watch", help="sample the selected (default: every active) reserved unit")
    watching.add_argument("--unit", action="append", default=[], help="merlinphase1... unit name (repeatable)")
    watching.add_argument("--interval", type=float, default=INTERVAL_S)
    watching.add_argument("--once", action="store_true", help="print one sample and its warnings; write nothing")
    watching.add_argument("--out", type=Path, help="status directory (default: <state dir>/operator)")
    sub.add_parser("check", help="quiet-host check: exit 3 while a reserved run is live")
    args = parser.parse_args(argv)
    if args.command == "check":
        return check()
    with contextlib.suppress(OSError):
        os.nice(19)
    return watch(args.unit, interval=args.interval, once=args.once, out=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
