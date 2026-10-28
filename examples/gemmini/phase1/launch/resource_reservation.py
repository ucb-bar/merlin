"""Run one finite experiment with a temporary CPU reservation and automatic release.

Uses a transient system service outside user.slice/system.slice. Existing user
and system workloads retain their processes and run on the complementary CPUs.
Their original AllowedCPUs properties are restored by ExecStopPost, including
startup failure, signals and RuntimeMaxSec expiry. No persistent unit is written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import resource
import signal
import subprocess
import sys
from pathlib import Path

HOST_UNITS = ("system.slice", "user.slice")
# Strict tool receipts need thousands of live read-lease descriptors; systemd's
# default 1024 otherwise forces full hashing of overflow pins at every boundary.
FILE_DESCRIPTOR_LIMIT = 65536


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()  # noqa: UP017 -- root helper uses Python 3.10


def save(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cpus(value: str) -> set[int]:
    result = set()
    for part in value.replace(" ", ",").split(","):
        if not part:
            continue
        endpoints = part.split("-")
        if len(endpoints) == 1:
            result.add(int(part))
        elif len(endpoints) == 2 and int(endpoints[0]) <= int(endpoints[1]):
            result.update(range(int(endpoints[0]), int(endpoints[1]) + 1))
        else:
            raise ValueError("invalid CPU range")
    if any(cpu < 0 for cpu in result):
        raise ValueError("negative CPU")
    return result


def cpu_text(value):
    return ",".join(str(cpu) for cpu in sorted(value))


def command(argv):
    return subprocess.check_output(argv, text=True).strip()


def allowed(unit):
    return command(["systemctl", "show", unit, "--property=AllowedCPUs", "--value"])


def effective_cpus(unit):
    group = command(["systemctl", "show", unit, "--property=ControlGroup", "--value"])
    return cpus((Path("/sys/fs/cgroup") / group.lstrip("/") / "cpuset.cpus.effective").read_text())


def set_allowed(unit, value):
    subprocess.run(["systemctl", "set-property", "--runtime", unit, "AllowedCPUs=" + value], check=True)


def root_state(path):
    if os.geteuid() != 0:
        raise ValueError("reservation lifecycle requires the system service root helper")
    if path.is_symlink() or not path.is_file():
        raise ValueError("reservation state must be an ordinary selected file")
    state = json.loads(path.read_text())
    if (
        state.get("schema") != "merlin.operator.cpu_reservation.v1"
        or set(state["host_original_allowed_cpus"]) != set(HOST_UNITS)
        or not re.fullmatch(r"merlinphase1[a-z0-9]+", state["unit"])
    ):
        raise ValueError("reservation state is invalid")
    return state


def acquire(path):
    state = root_state(path)
    selected = set(state["assigned_cpus"])
    host = set(state["online_cpus"]) - selected
    if not selected or not host:
        raise ValueError("reservation must leave CPUs for other machine workloads")
    for unit, original in state["host_original_allowed_cpus"].items():
        if allowed(unit) != original:
            raise ValueError(f"{unit} changed before reservation")
    state.update(status="acquiring", acquire_started_at=now())
    save(path, state)
    try:
        for unit, original in state["host_original_allowed_cpus"].items():
            original_set = cpus(original) if original else set(state["online_cpus"])
            complement = original_set & host
            if not complement:
                raise ValueError(f"reservation leaves no previously allowed CPU for {unit}")
            set_allowed(unit, cpu_text(complement))
        checks = {}
        for unit in HOST_UNITS:
            group = command(["systemctl", "show", unit, "--property=ControlGroup", "--value"])
            effective = cpus((Path("/sys/fs/cgroup") / group.lstrip("/") / "cpuset.cpus.effective").read_text())
            checks[unit] = sorted(effective)
            if not effective or effective & selected:
                raise ValueError(f"{unit} still overlaps the experiment reservation")
        state.update(status="reserved", acquired_at=now(), host_effective_cpus=checks)
        save(path, state)
    except BaseException:
        release(path)
        raise


def release(path):
    state = root_state(path)
    if "acquire_started_at" not in state:
        state.update(
            status="released_without_acquisition",
            released_at=now(),
            release_complete=True,
            release_errors=[],
        )
        save(path, state)
        return
    errors = []
    for unit, original in state["host_original_allowed_cpus"].items():
        try:
            original_effective = set(state["host_original_effective_cpus"][unit])
            # systemd 249 clears its property without expanding the populated kernel cpuset.
            # Restore the real mask first; an empty property alone is not release evidence.
            if not original:
                set_allowed(unit, cpu_text(original_effective))
            set_allowed(unit, original)
            if allowed(unit) != original or effective_cpus(unit) != original_effective:
                raise ValueError("CPU property or effective kernel mask did not restore")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"{unit}: {exc}")
    state.update(
        status="released" if not errors else "release_failed",
        released_at=now(),
        release_complete=not errors,
        release_errors=errors,
        service_result=os.environ.get("SERVICE_RESULT"),
        exit_code=os.environ.get("EXIT_CODE"),
        exit_status=os.environ.get("EXIT_STATUS"),
    )
    save(path, state)
    if errors:
        raise RuntimeError("; ".join(errors))


def child(path):
    state = json.loads(path.read_text())
    actual = os.sched_getaffinity(0)
    if actual != set(state["assigned_cpus"]) or state["status"] != "reserved":
        raise ValueError("experiment service does not have the exact reserved CPUs")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < FILE_DESCRIPTOR_LIMIT:
        raise ValueError("experiment service lacks the selected file-descriptor capacity")
    env = json.loads(Path(state["environment_path"]).read_text())
    argv = state["argv"]
    log = Path(state["log_path"])
    with log.open("ab", buffering=0) as stream:
        stream.write(
            (
                json.dumps(
                    {
                        "at": now(),
                        "event": "cpu_reservation_verified",
                        "cpus": sorted(actual),
                        "file_descriptor_limit": {"soft": soft, "hard": hard},
                        "open_descriptors": len(os.listdir("/proc/self/fd")),
                    }
                )
                + "\n"
            ).encode()
        )
        os.chdir(state["cwd"])
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        os.execvpe(argv[0], argv, env)


def launch(args):
    if os.geteuid() == 0:
        raise ValueError("invoke the launcher as the experiment user; helpers use sudo")
    if not re.fullmatch(r"merlinphase1[a-z0-9]+", args.unit):
        raise ValueError("unit must be a unique merlinphase1 name with lower-case alphanumerics")
    selected = cpus(args.cpus)
    online = cpus(Path("/sys/devices/system/cpu/online").read_text())
    if not selected or not selected < online:
        raise ValueError("assigned CPUs must be a nonempty proper subset of online CPUs")
    if args.max_seconds <= 0 or not args.command:
        raise ValueError("a finite experiment duration and command are required")
    for entry in Path("/sys/fs/cgroup").iterdir():
        if entry.is_dir() and entry.name not in (*HOST_UNITS, "init.scope"):
            # Judge each occupied cgroup by its own CPUs: a slice that holds a reservation reports every
            # CPU while its occupied unit is pinned to a subset, so two disjoint reservations can coexist.
            for procs in (entry / "cgroup.procs", *entry.rglob("*/cgroup.procs")):
                effective = procs.parent / "cpuset.cpus.effective"
                if (
                    procs.exists()
                    and procs.read_text().strip()
                    and effective.exists()
                    and cpus(effective.read_text()) & selected
                ):
                    raise ValueError(f"another top-level cgroup can use the selected CPUs: {entry.name}")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    if args.state.exists() or args.state.is_symlink():
        raise ValueError("reservation needs a new state file")
    environment_path = args.state.with_suffix(".environment.json")
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "HOME",
            "USER",
            "LOGNAME",
            "PATH",
            "PYTHONPATH",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONUNBUFFERED",
        }
    }
    with environment_path.open("x") as stream:
        os.chmod(environment_path, 0o600)
        json.dump(env, stream)
    state = {
        "schema": "merlin.operator.cpu_reservation.v1",
        "status": "prepared",
        "prepared_at": now(),
        "unit": args.unit,
        "assigned_cpus": sorted(selected),
        "online_cpus": sorted(online),
        "host_original_allowed_cpus": {unit: allowed(unit) for unit in HOST_UNITS},
        "host_original_effective_cpus": {unit: sorted(effective_cpus(unit)) for unit in HOST_UNITS},
        "argv": args.command,
        "cwd": str(Path.cwd()),
        "log_path": str(args.log.resolve()),
        "environment_path": str(environment_path.resolve()),
        "max_seconds": args.max_seconds,
        "file_descriptor_limit": FILE_DESCRIPTOR_LIMIT,
        "release_complete": False,
        "scope": (
            "CPU isolation of ordinary user/system workloads; shared memory, disk and "
            "OS interrupts remain observed resources"
        ),
    }
    save(args.state, state)
    source = str(Path(__file__).resolve())
    state_path = str(args.state.resolve())
    # '+' runs just the lifecycle helpers as root while the experiment remains its invoking user.
    argv = [
        "sudo",
        "-n",
        "systemd-run",
        "--wait",
        "--collect",
        "--unit=" + args.unit,
        "--slice=merlinphase1.slice",
        "--property=Type=exec",
        "--property=User=" + command(["id", "-un"]),
        "--property=AllowedCPUs=" + cpu_text(selected),
        "--property=LimitNOFILE=" + str(FILE_DESCRIPTOR_LIMIT),
        "--property=RuntimeMaxSec=" + str(args.max_seconds),
        "--property=TimeoutStopSec=30",
        "--property=KillMode=control-group",
        f"--property=ExecStartPre=+/usr/bin/python3 {source} _acquire --state {state_path}",
        f"--property=ExecStopPost=+/usr/bin/python3 {source} _release --state {state_path}",
        "/usr/bin/python3",
        source,
        "_child",
        "--state",
        state_path,
    ]
    if any(any(char.isspace() for char in value) for value in (source, state_path)):
        raise ValueError("systemd helper paths must not contain whitespace")
    prior = {}

    def stop(signum, _frame):
        subprocess.run(["sudo", "-n", "systemctl", "stop", args.unit + ".service"], check=False)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        prior[signum] = signal.signal(signum, stop)
    try:
        with announced(args, selected):
            result = subprocess.run(argv, check=False)
        final = json.loads(args.state.read_text())
        if final.get("release_complete") is not True:
            subprocess.run(
                ["sudo", "-n", "/usr/bin/python3", source, "_release", "--state", state_path],
                check=True,
            )
        return result.returncode
    finally:
        for signum, handler in prior.items():
            signal.signal(signum, handler)


def announced(args, selected):
    """The quiet-host marker entry (run_watchdog.py beside this file) for the reservation's lifetime."""
    import contextlib
    import importlib.util

    source = Path(__file__).with_name("run_watchdog.py")
    try:
        spec = importlib.util.spec_from_file_location("reservation_run_watchdog", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (OSError, ImportError) as error:
        print(f"quiet-host marker unavailable: {error}", file=sys.stderr)
        return contextlib.nullcontext()
    command = args.command
    run_ids = [command[command.index("--run-id") + 1]] if "--run-id" in command[:-1] else []
    return module.announced(args.unit, selected, args.max_seconds, run_ids)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"_acquire", "_release", "_child"}:
        parser = argparse.ArgumentParser()
        parser.add_argument("action")
        parser.add_argument("--state", type=Path, required=True)
        args = parser.parse_args()
        return {"_acquire": acquire, "_release": release, "_child": child}[args.action](args.state) or 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    try:
        return launch(args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "reservation_refused", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
