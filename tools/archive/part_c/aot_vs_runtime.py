#!/usr/bin/env python3
"""AOT vs runtime side-by-side comparison driver (Part C / C8).

Runs the same model + schedule through both execution modes:

  AOT mode      -- one full-model VMFB compiled via
                   ./merlin compile --with-schedule schedule.json
                   The stamped stream.affinity per dispatch routes through
                   IREE's Stream/HAL layer; the multi-device runner from
                   C1 picks it up and dispatches to pinned devices.

  Runtime mode  -- per-dispatch micro-benchmark VMFBs from
                   tools/breakdown_vmfb.py + the xpu-rt scheduler runner
                   (samples/common/xpu-rt/scheduler_runner.cc). The
                   scheduler reads schedule.json and dispatches per-node.

Emits a comparison report JSON capturing wall-clock + correctness:

  {
    "schedule": "<path>",
    "host_md5": "<reference md5 from un-scheduled host run>",
    "aot": {
      "vmfb": "<path>",
      "wall_ms": <float>,
      "output_md5": "<...>",
      "matches_host": true|false
    },
    "runtime": {
      "scheduler_runner": "<path>",
      "wall_ms": <float>,
      "output_md5": "<...>",
      "matches_host": true|false
    }
  }

Usage:
    tools/aot_vs_runtime.py \\
        --schedule schedule.json \\
        --aot-vmfb dronet.scheduled.vmfb \\
        --baseline-vmfb dronet.unscheduled.vmfb \\
        --runtime-runner build/.../merlin_multi_device_runner \\
        --breakdowns-dir build/.../breakdowns \\
        --scheduler-runner build/.../scheduler_runner_main \\
        --input '1x200x200x1xi8=@input.bin' \\
        --host qdev \\
        --report comparison_report.json

The --host flag drives invocation over SSH (matches the existing
tools/run_schedule_on_board.py + tools/run_multi_device_on_board.py
conventions). When omitted the comparison runs locally.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


@dataclasses.dataclass
class RunResult:
    wall_ms: float
    output_md5: str
    matches_host: bool


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_local(cmd: list[str], stdout_path: Path | None = None) -> float:
    print(f"[run] {' '.join(shlex.quote(a) for a in cmd)}", file=sys.stderr)
    t0 = time.monotonic()
    if stdout_path is not None:
        with stdout_path.open("w") as f:
            res = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=False)
    else:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    if res.returncode != 0:
        raise RuntimeError(f"command failed (exit={res.returncode}): " f"{res.stderr[-2000:]}")
    return elapsed_ms


def ssh(host: str, cmd: str) -> tuple[float, str]:
    """Run cmd on host via SSH; return (wall_ms, stdout)."""
    full = ["ssh", host, "bash", "-lc", cmd]
    t0 = time.monotonic()
    res = subprocess.run(full, capture_output=True, text=True, check=False)
    elapsed = (time.monotonic() - t0) * 1000.0
    if res.returncode != 0:
        raise RuntimeError(f"ssh failed: {res.stderr[-2000:]}")
    return elapsed, res.stdout


def push(host: str, src: Path, dst: str) -> None:
    print(f"[scp] {src} -> {host}:{dst}", file=sys.stderr)
    subprocess.run(["scp", "-q", str(src), f"{host}:{dst}"], check=True)


def pull(host: str, src: str, dst: Path) -> None:
    subprocess.run(["scp", "-q", f"{host}:{src}", str(dst)], check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--schedule",
        required=True,
        type=Path,
        help="schedule.json (canonical source of truth for " "machines + per-dispatch directives).",
    )
    p.add_argument("--aot-vmfb", required=True, type=Path, help="Schedule-applied full-model VMFB.")
    p.add_argument(
        "--baseline-vmfb",
        required=True,
        type=Path,
        help="Un-scheduled full-model VMFB. Used to compute the " "reference md5 against which both modes are checked.",
    )
    p.add_argument("--runtime-runner", type=Path, default=None, help="Path to the C1 merlin_multi_device_runner.")
    p.add_argument(
        "--breakdowns-dir",
        type=Path,
        default=None,
        help="Directory of per-dispatch breakdown VMFBs (output " "of tools/breakdown_vmfb.py).",
    )
    p.add_argument(
        "--scheduler-runner", type=Path, default=None, help="Path to the xpu-rt scheduler_runner_main binary."
    )
    p.add_argument(
        "--input",
        action="append",
        default=[],
        help="Repeatable iree-run-module input spec. Use " "@<absolute-path> for binary inputs.",
    )
    p.add_argument(
        "--host",
        default=None,
        help="SSH host (e.g. qdev). When set, all binaries + "
        "VMFBs + inputs are pushed to <remote-dir> and run "
        "there. When omitted, runs locally.",
    )
    p.add_argument("--remote-dir", default="/data/local/tmp/aot_vs_runtime", help="Remote staging directory.")
    p.add_argument(
        "--cluster",
        action="append",
        default=[],
        help="Per-cluster cpu pin (name:cpu_ids), repeatable. "
        "Forwarded to the multi-device runner. Required "
        "for AOT mode.",
    )
    p.add_argument("--report", type=Path, default=Path("comparison_report.json"), help="Output report path.")
    return p.parse_args()


def md5_pair(local_path: Path) -> str:
    return md5_of(local_path)


def run_aot_local(args: argparse.Namespace) -> RunResult:
    """Local AOT mode: invoke iree-run-module against --aot-vmfb."""
    out_path = args.report.parent / ".aot_out.bin"
    cmd = [
        "build/host-vanilla-release/tools/iree-run-module",
        f"--module={args.aot_vmfb}",
        "--device=local-task",
        f"--output=@{out_path}",
    ]
    for s in args.input:
        cmd.append(f"--input={s}")
    wall = run_local(cmd)
    return RunResult(wall_ms=wall, output_md5=md5_of(out_path), matches_host=False)  # filled after we have the host md5


def run_aot_ssh(args: argparse.Namespace) -> RunResult:
    remote = args.remote_dir
    ssh(args.host, f"mkdir -p {shlex.quote(remote)} && rm -f " f"{shlex.quote(remote)}/aot_out.*.bin")
    push(args.host, args.aot_vmfb, f"{remote}/{args.aot_vmfb.name}")
    if args.runtime_runner is not None:
        push(args.host, args.runtime_runner, f"{remote}/{args.runtime_runner.name}")
        ssh(args.host, f"chmod +x {shlex.quote(remote)}/" f"{args.runtime_runner.name}")
    for s in args.input:
        if "=@" in s:
            _, _, p = s.partition("=@")
            push(args.host, Path(p), f"{remote}/{Path(p).name}")
    rargs = []
    runner_name = args.runtime_runner.name if args.runtime_runner else None
    if runner_name:
        rargs.append(f"./{runner_name}")
        rargs += [f"--module={args.aot_vmfb.name}", "--function=main", "--output_dump=aot_out"]
        for c in args.cluster:
            rargs.append(f"--cluster={c}")
        for s in args.input:
            if "=@" in s:
                shape, _, p = s.partition("=@")
                rargs.append(f"--input={shape}=@{Path(p).name}")
            else:
                rargs.append(f"--input={s}")
    else:
        rargs += [
            "taskset",
            "-c",
            "4-7",
            "iree-run-module",
            f"--module={args.aot_vmfb.name}",
            "--device=local-task",
            "--output=@aot_out.0.bin",
        ]
        for s in args.input:
            if "=@" in s:
                shape, _, p = s.partition("=@")
                rargs.append(f"--input={shape}=@{Path(p).name}")
            else:
                rargs.append(f"--input={s}")
    cmd = " ".join(shlex.quote(a) for a in rargs)
    wall, _ = ssh(args.host, f"cd {shlex.quote(remote)} && {cmd}")
    pull(args.host, f"{remote}/aot_out.0.bin", Path("./aot_out.0.bin"))
    return RunResult(wall_ms=wall, output_md5=md5_of(Path("./aot_out.0.bin")), matches_host=False)


def run_runtime_local(args: argparse.Namespace) -> RunResult | None:
    if args.scheduler_runner is None or args.breakdowns_dir is None:
        print("[runtime] skipping: --scheduler-runner / --breakdowns-dir not set", file=sys.stderr)
        return None
    out_path = args.report.parent / ".runtime_out.bin"
    cmd = [
        str(args.scheduler_runner),
        f"--schedule={args.schedule}",
        f"--vmfb_dir={args.breakdowns_dir}",
        f"--output={out_path}",
    ]
    for s in args.input:
        cmd.append(f"--input={s}")
    wall = run_local(cmd)
    return RunResult(
        wall_ms=wall, output_md5=(md5_of(out_path) if out_path.exists() else "<no output>"), matches_host=False
    )


def main() -> int:
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)

    # Reference: run the un-scheduled baseline locally for a host md5.
    host_out = args.report.parent / ".host_out.bin"
    cmd = [
        "build/host-vanilla-release/tools/iree-run-module",
        f"--module={args.baseline_vmfb}",
        "--device=local-task",
        f"--output=@{host_out}",
    ]
    for s in args.input:
        cmd.append(f"--input={s}")
    try:
        run_local(cmd)
        host_md5 = md5_of(host_out)
    except Exception as e:
        print(f"[host] WARNING: baseline run failed: {e}", file=sys.stderr)
        host_md5 = ""

    aot = run_aot_ssh(args) if args.host else run_aot_local(args)
    aot.matches_host = host_md5 != "" and aot.output_md5 == host_md5

    runtime = run_runtime_local(args)
    if runtime is not None:
        runtime.matches_host = host_md5 != "" and runtime.output_md5 == host_md5

    report = {
        "schedule": str(args.schedule),
        "host_md5": host_md5,
        "aot": dataclasses.asdict(aot),
        "runtime": (dataclasses.asdict(runtime) if runtime else None),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[report] {args.report}")
    print(f"  AOT     wall={aot.wall_ms:.1f}ms  md5={aot.output_md5} " f"matches_host={aot.matches_host}")
    if runtime is not None:
        print(
            f"  RUNTIME wall={runtime.wall_ms:.1f}ms  " f"md5={runtime.output_md5} matches_host={runtime.matches_host}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
