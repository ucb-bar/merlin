"""Run one local preparation child with explicit memory headroom and retained logs."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine memory headroom")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--min-headroom-gib", type=float, default=1.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0 or args.min_headroom_gib <= 0:
        parser.error("Supply a command and positive execution bounds")
    if args.log.exists() or args.status.exists():
        parser.error("Use new log/status paths; preserve previous attempts")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failure = None
    with args.log.open("w") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - started
                headroom = available_memory()
                record = dict(status="running", pid=child.pid, elapsed_s=elapsed, available_memory_bytes=headroom)
                args.status.write_text(json.dumps(record, indent=2) + "\n")
                if headroom < args.min_headroom_gib * 1024**3:
                    failure = "memory_headroom_limit"
                elif elapsed > args.timeout:
                    failure = "preparation_timeout"
                if failure:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(3)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    record = dict(
        status="complete" if child.returncode == 0 and failure is None else "failed",
        returncode=child.returncode,
        failure=failure,
        elapsed_s=time.monotonic() - started,
        command=command,
    )
    args.status.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    raise SystemExit(0 if record["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
