#!/usr/bin/env python3
"""End-to-end XPU-RT loop driver: profile -> schedule -> run -> fold -> repeat.

Model-agnostic. Given one or more models with per-dispatch breakdown VMFBs
already on an aarch64 board (typically QRB5165), this:

  1. Profiles each dispatch on each cluster with the SAME pinned topology
     the scheduler will use at runtime (board_roundtrip with
     --task-topology-cpu-ids).
  2. Optionally runs forced-target schedules to refine the cost matrix
     under in-scheduler conditions (recommended for the first iteration
     since isolated bench is hot-cache amortised and ~3x optimistic).
  3. Calls XPU-RT's merlin_adapter.py schedule (or multi) with the
     supplied solver + transfer-cost + critical-path-bias knobs.
  4. Pushes the schedule + runs scheduler_bin on the board.
  5. Folds the resulting trace's run_us back into the cost matrix.
  6. Iterates until planned ≈ observed converges (or max_iters reached).
  7. Renders the planned-vs-observed Gantt for each iteration.

Driver wraps the loop so a typical experiment is one command:

    tools/run/full_loop.py --model <model_name> \\
        --merlin-dir <captured_eval_dir>/<model_name> \\
        --remote-vmfb-dir /root/iree_run/<model_name>/breakdowns \\
        --solver greedy --transfer-time-us 50 --critical-path-bias-us 300 \\
        --iters 3

For multi-model:

    tools/run/full_loop.py --multi \\
        --workload <captured_eval_dir>/<model_a>:1:<job_a> \\
        --workload <captured_eval_dir>/<model_b>:1:<job_b> \\
        --solver mosek --iters 2 --output <captured_eval_dir>/multi
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import os

XPU_RT_ROOT = Path(os.environ.get("MERLIN_XPU_RT_ROOT", str(REPO_ROOT.parent / "XPU-RT")))
ADAPTER = XPU_RT_ROOT / "scripts" / "merlin_adapter.py"
BOARD_ROUNDTRIP = REPO_ROOT / "tools" / "run" / "roundtrip.py"
FORCE_TARGET = REPO_ROOT / "tools" / "run" / "sched_force_target.py"
TRACE_TO_PROFILE = REPO_ROOT / "tools" / "perf" / "trace_to_profile.py"
PLOT = REPO_ROOT / "tools" / "perf" / "plot_planned_vs_observed.py"
STREAMING_FB = XPU_RT_ROOT / "xpu-rt" / "streaming_feedback.py"


def ingest_feedback(merlin_dir: Path) -> dict:
    """Persist xpurt_feedback.json (just emitted by the adapter) into
    breakdowns/feedback.json via the targetgen_mcp ingest tool. Returns
    the tool's summary dict (hint counts etc.) or {} if there was nothing
    to ingest. No-op when no xpurt_feedback.json was emitted — preserves
    the additive-only invariant.
    """
    payload_path = merlin_dir / "breakdowns" / "xpurt_feedback.json"
    if not payload_path.exists():
        return {}
    # Call the tool function directly via a tiny in-process script so we
    # can capture the return dict for the convergence check.
    src = (
        "import json, sys\n"
        "sys.path.insert(0, %r)\n"
        "from targetgen_mcp.tools import dispatch_tool\n"
        "result = dispatch_tool('ingest_xpurt_feedback', {\n"
        "    'merlin_dir': %r, 'payload_path': %r})\n"
        "sys.stdout.write(json.dumps(result))\n"
    ) % (str(REPO_ROOT / "tools"), str(merlin_dir), str(payload_path))
    cp = subprocess.run([sys.executable, "-c", src], check=True, capture_output=True, text=True)
    try:
        return json.loads(cp.stdout) if cp.stdout else {}
    except json.JSONDecodeError:
        return {}


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(shlex.quote(a) for a in cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=cwd, check=True)


def ssh(host: str, cmd: str, capture: bool = False) -> str:
    full = ["ssh", host, "bash", "-lc", cmd]
    print(f"[ssh:{host}] {cmd}", file=sys.stderr)
    res = subprocess.run(full, check=True, text=True, capture_output=capture)
    return res.stdout if capture else ""


def scp(src: Path, host: str, dst: str) -> None:
    subprocess.run(["scp", "-q", str(src), f"{host}:{dst}"], check=True)


def scp_pull(host: str, src: str, dst: Path) -> None:
    subprocess.run(["scp", "-q", f"{host}:{src}", str(dst)], check=True)


def initial_profile(args: argparse.Namespace, merlin_dir: Path) -> None:
    """Run board_roundtrip twice (CPU_P, CPU_E) to seed profiled_manifest."""
    breakdowns = merlin_dir / "breakdowns"
    if not (breakdowns / "manifest.json").exists():
        print(f"[err] missing {breakdowns}/manifest.json — pull from board first", file=sys.stderr)
        sys.exit(2)

    for cluster, ids in [("CPU_P", args.cpu_p_cpu_ids), ("CPU_E", args.cpu_e_cpu_ids)]:
        run(
            [
                "uv",
                "run",
                "python",
                str(BOARD_ROUNDTRIP),
                "--output-dir",
                str(merlin_dir),
                "--ssh-host",
                args.ssh_host,
                "--remote-dir",
                args.remote_root,
                "--skip-push",
                f"--task-topology-cpu-ids={ids}",
                f"--profile-key={cluster}",
                f"--repetitions={args.repetitions}",
            ]
        )


def force_target_pass(args: argparse.Namespace, merlin_dir: Path, schedule_for_paths: Path) -> None:
    """Run force-target schedules on the board so the manifest has
    in-scheduler costs for both clusters."""
    for cluster, ids in [("CPU_P", args.cpu_p_cpu_ids), ("CPU_E", args.cpu_e_cpu_ids)]:
        forced_local = merlin_dir / f"forced_{cluster}.json"
        run(
            [
                "uv",
                "run",
                "python",
                str(FORCE_TARGET),
                "--base",
                str(schedule_for_paths),
                "--target",
                cluster,
                "--out",
                str(forced_local),
            ]
        )
        scp(forced_local, args.ssh_host, f"/root/forced_{cluster}.json")
        ssh(
            args.ssh_host,
            (
                f"cd {shlex.quote(args.remote_root)} && "
                f"./scheduler_bin /root/forced_{cluster}.json "
                f"local-task 1 1 0 "
                f"--vmfb_dir={shlex.quote(args.remote_vmfb_dir)} "
                f"--cpu_p_cpu_ids={args.cpu_p_cpu_ids} "
                f"--cpu_e_cpu_ids={args.cpu_e_cpu_ids} "
                f"--visible_cores={args.visible_cores} "
                f"--trace_csv=/root/forced_{cluster}_trace.csv"
            ),
        )
        trace_local = merlin_dir / f"forced_{cluster}_trace.csv"
        scp_pull(args.ssh_host, f"/root/forced_{cluster}_trace.csv", trace_local)
        run(
            [
                "uv",
                "run",
                "python",
                str(TRACE_TO_PROFILE),
                "--trace-csv",
                str(trace_local),
                "--manifest",
                str(merlin_dir / "breakdowns" / "profiled_manifest.json"),
                "--write",
            ]
        )


def schedule_once(args: argparse.Namespace, merlin_dir: Path, iter_idx: int) -> Path:
    # Stable run_id across iterations of the same loop so the targetgen_mcp
    # merge semantics accumulate hints rather than overwrite. The streaming
    # daemon (when enabled) reuses the same id.
    run_id = getattr(args, "run_id", None) or f"runfullloop_{merlin_dir.name}"
    cmd = [
        "python3",
        str(ADAPTER),
        "schedule",
        str(merlin_dir),
        "--machines",
        *args.machines,
        f"--transfer-time-us={args.transfer_time_us}",
        f"--solver={args.solver}",
        f"--critical-path-bias-us={args.critical_path_bias_us}",
        f"--time-limit-s={args.time_limit_s}",
        "--emit-feedback",
        f"--feedback-run-id={run_id}",
    ]
    run(cmd, cwd=XPU_RT_ROOT)
    snap = merlin_dir / f"schedule_iter{iter_idx}.json"
    snap.write_bytes((merlin_dir / "breakdowns" / "schedule.json").read_bytes())
    return snap


def run_on_board(args: argparse.Namespace, schedule: Path, iter_idx: int, merlin_dir: Path) -> Path:
    remote = f"/root/{merlin_dir.name}_iter{iter_idx}.json"
    scp(schedule, args.ssh_host, remote)
    trace_remote = f"/root/{merlin_dir.name}_iter{iter_idx}_trace.csv"

    # Hardware-in-the-loop streaming. When --stream-epochs is set, ask
    # the on-board runner to emit JSON-Lines telemetry, run a host-side
    # streaming_feedback.py daemon in parallel that posts incremental
    # feedback to the MCP ingest tool while the workload runs. When
    # --stream-epochs is unset, the run is byte-identical to the prior
    # batch-only loop (additive-only invariant).
    stream_epochs = int(getattr(args, "stream_epochs", 0) or 0)
    extra_runner_flags = ""
    telemetry_remote = f"/root/{merlin_dir.name}_iter{iter_idx}_telemetry.jsonl"
    telemetry_local = merlin_dir / f"telemetry_iter{iter_idx}.jsonl"
    daemon: subprocess.Popen | None = None
    if stream_epochs > 0:
        extra_runner_flags = f" --telemetry_jsonl={telemetry_remote}"
        # Best-effort: pipe `ssh tail -f <remote>` into the local jsonl,
        # then point streaming_feedback.py at that local file. We start
        # both *before* the actual run so they catch the first events.
        # Touch the local file first so the daemon's path-exists check
        # passes even if the board is slow to flush.
        telemetry_local.write_text("")
        run_id = getattr(args, "run_id", None) or f"runfullloop_{merlin_dir.name}"
        # tail-on-board background job — we kill it via SIGINT later via
        # the bound subprocess handle.
        tail_cmd = [
            "ssh",
            args.ssh_host,
            "bash",
            "-lc",
            f"touch {shlex.quote(telemetry_remote)} && " f"tail -F {shlex.quote(telemetry_remote)}",
        ]
        tail_proc = subprocess.Popen(tail_cmd, stdout=telemetry_local.open("ab"), stderr=subprocess.DEVNULL)
        # Streaming daemon (foreground will exit on EOF when --follow not
        # set; here we want --follow and kill it after run completes).
        daemon_cmd = [
            sys.executable,
            str(STREAMING_FB),
            "--telemetry-stream",
            str(telemetry_local),
            "--merlin-dir",
            str(merlin_dir),
            "--run-id",
            run_id,
            f"--epoch-window={stream_epochs}",
            f"--post-every-n-epochs={max(1, stream_epochs // 4)}",
            "--follow",
        ]
        daemon = subprocess.Popen(daemon_cmd)
        daemon._tail_proc = tail_proc  # type: ignore[attr-defined]
        print(f"[stream-fb] daemon pid={daemon.pid}, " f"tail pid={tail_proc.pid}")

    try:
        ssh(
            args.ssh_host,
            (
                f"cd {shlex.quote(args.remote_root)} && "
                f"./scheduler_bin {shlex.quote(remote)} local-task 1 1 0 "
                f"--vmfb_dir={shlex.quote(args.remote_vmfb_dir)} "
                f"--cpu_p_cpu_ids={args.cpu_p_cpu_ids} "
                f"--cpu_e_cpu_ids={args.cpu_e_cpu_ids} "
                f"--visible_cores={args.visible_cores} "
                f"--trace_csv={trace_remote}"
                f"{extra_runner_flags}"
            ),
        )
    finally:
        if daemon is not None:
            try:
                daemon.terminate()
                daemon.wait(timeout=5)
            except Exception:
                daemon.kill()
            tail_proc = getattr(daemon, "_tail_proc", None)
            if tail_proc is not None:
                try:
                    tail_proc.terminate()
                    tail_proc.wait(timeout=2)
                except Exception:
                    tail_proc.kill()
    trace_local = merlin_dir / f"trace_iter{iter_idx}.csv"
    scp_pull(args.ssh_host, trace_remote, trace_local)
    return trace_local


def fold(args: argparse.Namespace, merlin_dir: Path, trace: Path) -> None:
    run(
        [
            "uv",
            "run",
            "python",
            str(TRACE_TO_PROFILE),
            "--trace-csv",
            str(trace),
            "--manifest",
            str(merlin_dir / "breakdowns" / "profiled_manifest.json"),
            "--write",
        ]
    )


def plot(trace: Path, out_path: Path, title: str) -> None:
    run(
        [
            "uv",
            "run",
            "python",
            str(PLOT),
            "--trace-csv",
            str(trace),
            "--out",
            str(out_path),
            "--title",
            title,
        ]
    )


def measure_gap(trace: Path) -> tuple[float, float, float]:
    """Returns (planned_makespan_ms, observed_makespan_ms, mean_abs_pct)."""
    import pandas as pd

    df = pd.read_csv(trace)
    plan_end = (df.planned_start_us + df.planned_duration_us).max()
    obs_end = df.end_us.max()
    df["delta_pct"] = 100.0 * (df.run_us - df.planned_duration_us) / df.planned_duration_us.replace(0, 1)
    return plan_end / 1000.0, obs_end / 1000.0, df.delta_pct.abs().mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--merlin-dir", required=True, type=Path, help="Per-model output dir (must contain breakdowns/manifest.json)."
    )
    p.add_argument("--remote-vmfb-dir", required=True, help="Absolute path on the board where dispatch_*.vmfb live.")
    p.add_argument("--remote-root", default="/root/iree_run/dronet", help="Board cwd where scheduler_bin lives.")
    p.add_argument("--ssh-host", default="qdev")
    p.add_argument("--machines", nargs="+", default=["CPU_P", "CPU_E"])
    p.add_argument("--cpu-p-cpu-ids", default="4,5,6,7")
    p.add_argument("--cpu-e-cpu-ids", default="0,1")
    p.add_argument("--visible-cores", type=int, default=8)
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--solver", choices=["greedy", "mosek"], default="greedy")
    p.add_argument(
        "--transfer-time-us",
        type=float,
        default=50.0,
        help="Inter-cluster transfer cost; bumped from the old "
        "default of 10 to better match observed cross-cluster "
        "blocking behaviour.",
    )
    p.add_argument(
        "--critical-path-bias-us",
        type=float,
        default=300.0,
        help="Greedy bias to keep critical-path predecessors on " "the same cluster. 0 disables.",
    )
    p.add_argument("--time-limit-s", type=float, default=30.0)
    p.add_argument(
        "--iters",
        type=int,
        default=3,
        help="Profile/schedule/run/fold cycles after the initial " "force-target seeding pass.",
    )
    p.add_argument(
        "--seed-with-bench",
        action="store_true",
        help="Also run the isolated iree-benchmark-module pass " "before the force-target seeding. Default off (slow).",
    )
    p.add_argument(
        "--seed-from-existing-schedule",
        type=Path,
        help="Existing schedule.json on disk to use for the "
        "force-target seeding pass (so we don't have to "
        "build one from bench costs first). Recommended.",
    )
    p.add_argument("--out-prefix", default="loop", help="Prefix for per-iteration plots written to merlin-dir.")
    p.add_argument(
        "--run-id",
        default=None,
        help="Stable feedback run id (forwarded to merlin_adapter "
        "--feedback-run-id and to streaming_feedback.py). "
        "Defaults to 'runfullloop_<merlin_dir.name>'.",
    )
    p.add_argument(
        "--stream-epochs",
        type=int,
        default=0,
        help="If > 0, enable hardware-in-the-loop streaming: "
        "the on-board runner emits per-dispatch JSON-Lines "
        "telemetry, a host-side daemon (xpu-rt/"
        "streaming_feedback.py) tails the stream and posts "
        "incremental feedback to the targetgen_mcp ingest "
        "tool. The value sets the rolling epoch window. "
        "Default 0 = batch-only loop (no overhead).",
    )
    p.add_argument(
        "--converge-on-stable-hints",
        action="store_true",
        help="Stop the loop early once the feedback hint set " "stops changing (offline convergence signal).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    merlin_dir = args.merlin_dir.resolve()
    profiled = merlin_dir / "breakdowns" / "profiled_manifest.json"

    if args.seed_with_bench or not profiled.exists():
        print("[loop] initial isolated bench profiling …")
        initial_profile(args, merlin_dir)

    if args.seed_from_existing_schedule:
        print("[loop] force-target pass to refine costs under scheduler " "conditions …")
        force_target_pass(args, merlin_dir, args.seed_from_existing_schedule)

    history: list[dict] = []
    last_hint_counts: dict | None = None
    for it in range(1, args.iters + 1):
        snap = schedule_once(args, merlin_dir, it)
        # Persist offline-derived feedback to <merlin_dir>/breakdowns/
        # feedback.json BEFORE running on the board, so any compile that
        # the runner's downstream tooling triggers (or that the operator
        # kicks off out-of-band) sees the latest hints.
        ingest_summary = ingest_feedback(merlin_dir)
        if ingest_summary:
            print(f"[ingest iter{it}] {ingest_summary.get('hint_counts', {})}")
        trace = run_on_board(args, snap, it, merlin_dir)
        plan_ms, obs_ms, mean_abs_pct = measure_gap(trace)
        plot_path = merlin_dir / f"{args.out_prefix}_iter{it}.png"
        title = (
            f"{merlin_dir.name} iter{it} "
            f"({args.solver}, transfer={args.transfer_time_us:.0f}µs, "
            f"bias={args.critical_path_bias_us:.0f}µs) — "
            f"planned {plan_ms:.1f}ms vs observed {obs_ms:.1f}ms"
        )
        plot(trace, plot_path, title)
        print(f"[loop iter{it}] planned={plan_ms:.2f}ms " f"observed={obs_ms:.2f}ms mean|delta|={mean_abs_pct:.1f}%")
        history.append(
            {
                "iter": it,
                "planned_ms": plan_ms,
                "observed_ms": obs_ms,
                "mean_abs_pct": mean_abs_pct,
                "schedule": str(snap),
                "trace": str(trace),
                "plot": str(plot_path),
                "hint_counts": (ingest_summary.get("hint_counts", {}) if ingest_summary else {}),
                "n_dispatches_with_hints": (ingest_summary.get("n_dispatches_with_hints", 0) if ingest_summary else 0),
            }
        )
        # Two-tier convergence: stop early if the offline hint set has
        # stabilised across iterations and the user opted into it.
        new_hint_counts = ingest_summary.get("hint_counts") if ingest_summary else None
        if args.converge_on_stable_hints and last_hint_counts is not None and new_hint_counts == last_hint_counts:
            print(f"[loop] hint set stable across iter{it - 1}→iter{it}; " f"stopping early")
            break
        last_hint_counts = new_hint_counts
        # fold for the NEXT iter
        if it < args.iters:
            fold(args, merlin_dir, trace)

    summary_path = merlin_dir / f"{args.out_prefix}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "args": vars(args) | {"merlin_dir": str(merlin_dir)},
                "history": history,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"[loop] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
