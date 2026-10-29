"""Launch a matched study through the existing finite CPU reservation owner.

``plan`` only prints commands. ``launch`` acquires the selected CPU union once;
independent study services share each selected pool under equal CPU quotas. Their
lifetimes are bound to the reservation service, including abnormal termination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_text(cpus, *, compact=False):
    if compact and cpus == list(range(cpus[0], cpus[-1] + 1)):
        return f"{cpus[0]}-{cpus[-1]}"
    return ",".join(map(str, cpus))


def plan(path, unit, *, user=None):
    if not unit.startswith("merlinphase1") or not unit.isalnum() or not unit.islower():
        raise ValueError("select a unique lower-case alphanumeric merlinphase1 unit")
    path = Path(path).resolve(strict=True)
    document = json.loads(path.read_bytes())
    runs = document["runs"]
    if not 1 <= len(runs) <= 12:
        raise ValueError("select one to twelve prepared runs")
    records, pools = [], {}
    for row in runs:
        config = Path(row["config"])
        if config.is_symlink() or digest(config) != row["config_sha256"]:
            raise ValueError("prepared run config changed")
        settings = json.loads(config.read_bytes())
        pool = settings["cpus"]
        if (not isinstance(pool, list) or len(pool) != 4
                or any(type(cpu) is not int or cpu < 0 for cpu in pool)
                or pool != sorted(set(pool))):
            raise ValueError("each study requires four distinct sorted CPU IDs")
        key = tuple(pool)
        pools[key] = pools.get(key, 0) + 1
        records.append((row, config, settings))
    for left in pools:
        if any(left != right and set(left) & set(right) for right in pools):
            raise ValueError("distinct study CPU pools must be disjoint")
    cpus = sorted({cpu for pool in pools for cpu in pool})
    quota = 100 * len(cpus) // len(runs)
    if any(quota * count > 100 * len(pool) for pool, count in pools.items()):
        raise ValueError("equal quotas exceed a selected study pool's capacity")
    user = user or subprocess.check_output(["id", "-un"], text=True).strip()
    slice_name = "merlinkernelstudy" + unit.removeprefix("merlinphase1") + ".slice"
    children, timeouts, frameworks = [], [], set()
    for index, (row, config, settings) in enumerate(records):
        if settings["cpu_quota_percent"] != quota:
            raise ValueError("all services require equal bounded CPU quota")
        if settings["workers"] != 1 or digest(settings["runner"]["path"]) != settings["runner"]["sha256"]:
            raise ValueError("prepared runner changed or evaluator concurrency differs")
        expected = [
            "taskset",
            "-c",
            cpu_text(settings["cpus"]),
            settings["python"],
            "-B",
            settings["runner"]["path"],
            "launch",
            "--config",
            str(config),
        ]
        if row["launch_argv"] != expected:
            raise ValueError("launch argv differs from its pinned original runner/config")
        frameworks.add(settings["framework"])
        maximum = settings["wall_s"] + 120
        timeouts.append(maximum)
        name = unit + "r" + str(index)
        argv = [
            "sudo",
            "-n",
            "systemd-run",
            "--wait",
            "--collect",
            "--quiet",
            "--unit=" + name,
            "--slice=" + slice_name,
            "--property=Type=exec",
            "--property=User=" + user,
            "--property=AllowedCPUs=" + cpu_text(settings["cpus"], compact=True),
            f"--property=CPUQuota={quota}%",
            "--property=CPUQuotaPeriodSec=100ms",
            "--property=LimitNOFILE=65536",
            f"--property=RuntimeMaxSec={maximum}",
            "--property=TimeoutStopSec=30",
            "--property=KillMode=control-group",
            "--property=BindsTo=" + unit + ".service",
            "--property=After=" + unit + ".service",
            "--property=PartOf=" + unit + ".service",
            *row["launch_argv"],
        ]
        children.append(
            dict(
                unit=name + ".service",
                argv=argv,
                config=str(config),
                log=str(Path(settings["private"]) / "service-launch.log"),
            )
        )
    if len(frameworks) != 1:
        raise ValueError("all study runs must select the same framework")
    root = path.parent
    wrapper = Path(next(iter(frameworks))) / "examples/gemmini/phase1/launch/resource_reservation.py"
    maximum = max(timeouts) + 60
    command = [sys.executable, str(Path(__file__).resolve()), "_run", "--runs", str(path), "--unit", unit]
    reserved = [
        sys.executable,
        str(wrapper),
        "--cpus",
        cpu_text(cpus, compact=True),
        "--max-seconds",
        str(maximum),
        "--unit",
        unit,
        "--state",
        str(root / "cpu-reservation.json"),
        "--log",
        str(root / "reservation.log"),
        "--",
        *command,
    ]
    return dict(
        schema="private.kernel_abstractions.launch.v1",
        runs_sha256=digest(path),
        unit=unit,
        cpus=cpus,
        quota_percent=quota,
        slice=slice_name,
        runtime_max_seconds=maximum,
        reservation_argv=reserved,
        children=children,
        author_launched=False,
    )


def coordinate(path, unit):
    """Executed by the existing reservation service, never by an author."""
    selected = plan(path, unit)
    group = Path("/proc/self/cgroup").read_text()
    if unit + ".service" not in group or set(os.sched_getaffinity(0)) != set(selected["cpus"]):
        raise ValueError("coordinator requires its original active selected CPU reservation")
    processes, logs, previous = [], [], {}

    def interrupted(number, _frame):
        raise SystemExit(128 + number)

    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, interrupted)
        for child in selected["children"]:
            log = Path(child["log"]).open("wb")
            logs.append(log)
            processes.append(subprocess.Popen(child["argv"], stdout=log, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in processes):
            time.sleep(1)
        codes = [process.returncode for process in processes]
        (Path(path).parent / "launch-result.json").write_text(
            json.dumps(
                dict(
                    schema="private.kernel_abstractions.launch_result.v1",
                    unit=unit,
                    returncodes=codes,
                    completed_unix=time.time(),
                    reservation_release="owned by original ExecStopPost",
                ),
                indent=2,
            )
            + "\n"
        )
        return int(any(code != 0 for code in codes))
    finally:
        # Only this unique study's units; never the shared phase slice or other experiments.
        subprocess.run(
            ["sudo", "-n", "systemctl", "stop", *[row["unit"] for row in selected["children"]]], check=False, timeout=45
        )
        for stream in logs:
            stream.close()
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "launch", "_run"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    if args.action == "_run":
        return coordinate(args.runs, args.unit)
    selected = plan(args.runs, args.unit)
    if args.action == "plan":
        print(json.dumps(selected, indent=2))
        return 0
    return subprocess.call(selected["reservation_argv"])


if __name__ == "__main__":
    raise SystemExit(main())
