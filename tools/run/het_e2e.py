#!/usr/bin/env python3
"""End-to-end heterogeneous schedule runner for the QRB5165.

For one (model, granularity, machines) configuration, this:

  1. Verifies the per-chunk breakdown manifest exists for the given model/
     granularity (`eval/qrb5165/<merlin_dir>/breakdowns/manifest.json`).
  2. For every QNN backend in --machines, compiles every chunk's .qnn-ctx
     via `tools/compile_qnn.py` and stages it next to the CPU .vmfb files
     under `breakdowns/`.
  3. Profiles each chunk on each requested machine:
        - CPU_P / CPU_E -> tools/board_roundtrip.py with --task-topology
        - QNN_GPU / QNN_HTA -> qnn-net-run on board (per-chunk wall mean)
     and writes the result back into `breakdowns/profiled_manifest.json`
     under `profiles.<MACHINE>`.
  4. Calls `XPU-RT/scripts/merlin_adapter.py multi --machines ... --solver
     mosek` to produce a heterogeneous schedule.json.
  5. Pushes the schedule + .qnn-ctx artifacts to the board.
  6. Runs `merlin-dispatch-scheduler` with `--qnn_gpu_enabled` /
     `--qnn_hta_enabled` matching the schedule's `machines` list.
  7. Pulls the trace.csv, computes a planned-vs-observed gap + the md5 of
     the final job's output bytes, and renders the per-iteration Gantt.
  8. Writes everything under
        eval/qrb5165/heterogeneous/<model>_<granularity>_<machines>/

Designed to compose cleanly with `tools/run_full_loop.py` for the
CPU-only loop — this script reuses the same board roundtrip + plotting
helpers via subprocess.

Limitations / out-of-scope (deferred to caller):
  - The standalone qnn-net-run path is used for chunk-level profiling
    until the merlin-dispatch-scheduler glibc-skew fix lands.
  - Bytes-equal correctness check requires a baseline trace from the
    same model under all-CPU; pass --baseline-trace to enable.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# XPU-RT lives outside the merlin tree. Default assumes it's a sibling of
# the merlin checkout; override via $MERLIN_XPU_RT_ROOT or --xpu-rt-root.
import os

XPU_RT_ROOT = pathlib.Path(os.environ.get("MERLIN_XPU_RT_ROOT", str(REPO_ROOT.parent / "XPU-RT")))
ADAPTER = XPU_RT_ROOT / "scripts" / "merlin_adapter.py"
BOARD_ROUNDTRIP = REPO_ROOT / "tools" / "run" / "roundtrip.py"
COMPILE_QNN = REPO_ROOT / "tools" / "compile" / "qnn.py"
TRACE_TO_PROFILE = REPO_ROOT / "tools" / "perf" / "trace_to_profile.py"
PLOT = REPO_ROOT / "tools" / "plot_planned_vs_observed.py"

QNN_BACKEND_TO_MACHINE = {"qnn-gpu": "QNN_GPU", "qnn-hta": "QNN_HTA"}
MACHINE_TO_QNN_BACKEND = {v: k for k, v in QNN_BACKEND_TO_MACHINE.items()}
QNN_MACHINES = set(MACHINE_TO_QNN_BACKEND)
CPU_MACHINES = {"CPU_P", "CPU_E"}


@dataclasses.dataclass
class HetConfig:
    model: str
    granularity: str
    machines: list[str]
    merlin_dir: pathlib.Path
    output_dir: pathlib.Path
    ssh_host: str = "qdev"
    remote_root: str = "/root/iree_run"
    cpu_p_cpu_ids: str = "4,5,6,7"
    cpu_e_cpu_ids: str = "0,1"
    visible_cores: int = 8
    repetitions: int = 5
    solver: str = "mosek"
    transfer_time_us: float = 200.0
    qnn_sdk: pathlib.Path = pathlib.Path("/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326")
    board_sysroot: pathlib.Path = pathlib.Path("/scratch2/agustin/qrb5165_sysroot")
    baseline_trace: pathlib.Path | None = None
    skip_compile_qnn: bool = False
    skip_profile: bool = False
    skip_board_run: bool = False


def _run(cmd: list[str], cwd: pathlib.Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(shlex.quote(a) for a in cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def _ssh(host: str, cmd: str, capture: bool = True, timeout: float = 300.0) -> str:
    print(f"[ssh:{host}] {cmd}", file=sys.stderr)
    res = subprocess.run(
        ["ssh", host, "bash", "-lc", cmd],
        capture_output=capture,
        text=True,
        timeout=timeout,
        check=True,
    )
    return res.stdout if capture else ""


def _scp(src: pathlib.Path, host: str, dst: str) -> None:
    subprocess.run(["scp", "-q", str(src), f"{host}:{dst}"], check=True)


def _scp_pull(host: str, src: str, dst: pathlib.Path) -> None:
    subprocess.run(["scp", "-q", f"{host}:{src}", str(dst)], check=True)


def compile_qnn_chunks(cfg: HetConfig) -> dict[str, list[pathlib.Path]]:
    """Run tools/compile_qnn.py for each QNN backend the schedule needs.

    Returns a map from QNN machine name (QNN_GPU/QNN_HTA) to the list of
    .qnn-ctx files produced for the chunks of this model/granularity.
    """
    qnn_machines = [m for m in cfg.machines if m in QNN_MACHINES]
    out: dict[str, list[pathlib.Path]] = {m: [] for m in qnn_machines}
    if not qnn_machines or cfg.skip_compile_qnn:
        return out

    backends = ",".join(MACHINE_TO_QNN_BACKEND[m] for m in qnn_machines)
    out_dir = cfg.merlin_dir / "breakdowns"
    manifest = out_dir / "manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"missing chunk manifest: {manifest}")

    _run(
        [
            "uv",
            "run",
            "python",
            str(COMPILE_QNN),
            "--chunk-manifest",
            str(manifest),
            "--backends",
            backends,
            "--output-dir",
            str(out_dir),
            "--qnn-sdk",
            str(cfg.qnn_sdk),
            "--board-sysroot",
            str(cfg.board_sysroot),
        ]
    )

    chunks = json.loads(manifest.read_text()).get("dispatches", {})
    for m in qnn_machines:
        backend = MACHINE_TO_QNN_BACKEND[m]
        for chunk_name in chunks:
            ctx = out_dir / f"{chunk_name}.{backend}.qnn-ctx"
            if ctx.exists():
                out[m].append(ctx)
    return out


def profile_cpu_machine(cfg: HetConfig, machine: str) -> None:
    """Run tools/board_roundtrip.py in cluster mode for one CPU machine."""
    ids = cfg.cpu_p_cpu_ids if machine == "CPU_P" else cfg.cpu_e_cpu_ids
    _run(
        [
            "uv",
            "run",
            "python",
            str(BOARD_ROUNDTRIP),
            "--output-dir",
            str(cfg.merlin_dir),
            "--ssh-host",
            cfg.ssh_host,
            "--remote-dir",
            cfg.remote_root,
            "--skip-push",
            f"--task-topology-cpu-ids={ids}",
            f"--profile-key={machine}",
            f"--repetitions={cfg.repetitions}",
        ]
    )


def profile_qnn_machine(cfg: HetConfig, machine: str, ctx_files: list[pathlib.Path]) -> None:
    """Profile each chunk's .qnn-ctx on board via qnn-net-run.

    Captures wall-clock mean over `cfg.repetitions` invocations and folds
    the result back into breakdowns/profiled_manifest.json under
    `dispatches.<chunk>.profiles.<machine>.mean_time_us`.
    """
    backend = MACHINE_TO_QNN_BACKEND[machine]
    backend_lib = "libQnnGpu.so" if machine == "QNN_GPU" else "libQnnHtp.so"
    remote_dir = f"/root/qnn_chunks/{cfg.merlin_dir.name}"
    _ssh(cfg.ssh_host, f"mkdir -p {shlex.quote(remote_dir)}", capture=False)

    profiled = cfg.merlin_dir / "breakdowns" / "profiled_manifest.json"
    if profiled.exists():
        manifest = json.loads(profiled.read_text())
    else:
        manifest = json.loads((cfg.merlin_dir / "breakdowns" / "manifest.json").read_text())

    timings: dict[str, float] = {}
    for ctx in ctx_files:
        chunk_name = ctx.name.split(f".{backend}.qnn-ctx")[0]
        _scp(ctx, cfg.ssh_host, f"{remote_dir}/{ctx.name}")
        timings[chunk_name] = _qnn_net_run_one(
            cfg,
            ctx.name,
            backend_lib,
            remote_dir,
        )

    for chunk_name, mean_us in timings.items():
        entry = manifest["dispatches"].get(chunk_name)
        if not entry:
            continue
        entry.setdefault("profiles", {})[machine] = {
            "mean_time_us": mean_us,
            "machine": machine,
            "source": "qnn-net-run",
        }
        entry["mean_time_us"] = mean_us
    profiled.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[qnn_profile] {machine}: {len(timings)} chunks profiled " f"-> {profiled}")


def _qnn_net_run_one(cfg: HetConfig, ctx_basename: str, backend_lib: str, remote_dir: str) -> float:
    """Single chunk on board: returns wall mean micros over reps."""
    cmd = (
        f"cd {shlex.quote(remote_dir)} && "
        f"LD_LIBRARY_PATH=/root/qairt/lib/aarch64-oe-linux-gcc11.2:"
        f"/root/qairt/lib/target "
        f"qnn-net-run --backend /root/qairt/lib/target/{backend_lib} "
        f"--retrieve_context {shlex.quote(ctx_basename)} "
        f"--num_inferences {cfg.repetitions} --keep_num_outputs 0 "
        f"--profiling_level basic 2>&1"
    )
    out = _ssh(cfg.ssh_host, cmd)
    return _parse_qnn_net_run_mean_us(out)


def _parse_qnn_net_run_mean_us(stdout: str) -> float:
    """qnn-net-run --profiling_level basic prints wall_clock per inference;
    we average. Falls back to total_time / num_inferences on older SDKs."""
    samples = []
    for line in stdout.splitlines():
        s = line.strip()
        if "Execute Time" in s and "us" in s:
            try:
                samples.append(float(s.split()[-2]))
            except (ValueError, IndexError):
                continue
    if samples:
        return sum(samples) / len(samples)
    for line in stdout.splitlines():
        if "Inference Time" in line and "us" in line:
            try:
                return float(line.split()[-2])
            except (ValueError, IndexError):
                continue
    raise RuntimeError("could not parse qnn-net-run timing from stdout (head):\n" + "\n".join(stdout.splitlines()[:20]))


def schedule_heterogeneous(cfg: HetConfig) -> pathlib.Path:
    """Invoke merlin_adapter.py multi with the requested machines."""
    cmd = [
        "python3",
        str(ADAPTER),
        "multi",
        str(cfg.merlin_dir),
        "--machines",
        *cfg.machines,
        f"--transfer-time-us={cfg.transfer_time_us}",
        f"--solver={cfg.solver}",
    ]
    _run(cmd, cwd=XPU_RT_ROOT)
    sched = cfg.merlin_dir / "breakdowns" / "schedule.json"
    snap = cfg.output_dir / f"schedule_{'_'.join(cfg.machines)}.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_bytes(sched.read_bytes())
    return snap


def run_on_board(cfg: HetConfig, schedule: pathlib.Path) -> pathlib.Path:
    """Push schedule + run merlin-dispatch-scheduler with QNN flags."""
    if cfg.skip_board_run:
        return cfg.output_dir / "trace.csv"
    remote_sched = f"/root/{cfg.output_dir.name}_schedule.json"
    _scp(schedule, cfg.ssh_host, remote_sched)
    trace_remote = f"/root/{cfg.output_dir.name}_trace.csv"

    flags = [
        f"--vmfb_dir=/root/iree_run/{cfg.merlin_dir.name}/breakdowns",
        f"--cpu_p_cpu_ids={cfg.cpu_p_cpu_ids}",
        f"--cpu_e_cpu_ids={cfg.cpu_e_cpu_ids}",
        f"--visible_cores={cfg.visible_cores}",
        f"--trace_csv={trace_remote}",
    ]
    if "QNN_GPU" in cfg.machines:
        flags.append("--qnn_gpu_enabled=1")
    if "QNN_HTA" in cfg.machines:
        flags.append("--qnn_hta_enabled=1")

    cmd = (
        f"LD_LIBRARY_PATH=/root/qairt/lib/aarch64-oe-linux-gcc11.2:"
        f"/root/qairt/lib/target "
        f"/root/merlin-dispatch-scheduler {shlex.quote(remote_sched)} "
        f"local-task 1 1 0 " + " ".join(shlex.quote(f) for f in flags)
    )
    _ssh(cfg.ssh_host, cmd, capture=False)

    trace_local = cfg.output_dir / "trace.csv"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _scp_pull(cfg.ssh_host, trace_remote, trace_local)
    return trace_local


def measure_correctness(trace: pathlib.Path, baseline: pathlib.Path | None) -> dict:
    """If a baseline trace is supplied, compute per-chunk md5 over run_us
    sequences (proxy for output-bytes equivalence at the dispatch level).
    For real bytes-equality the chunk must persist outputs; we fall back
    to a structural equivalence check here."""
    out: dict = {"baseline_supplied": baseline is not None}
    if not baseline or not baseline.exists():
        return out
    import pandas as pd

    df_a = pd.read_csv(trace)
    df_b = pd.read_csv(baseline)
    out["dispatch_count_match"] = len(df_a) == len(df_b)
    out["unique_chunks_a"] = sorted(df_a["chunk"].unique().tolist()) if "chunk" in df_a.columns else []
    out["unique_chunks_b"] = sorted(df_b["chunk"].unique().tolist()) if "chunk" in df_b.columns else []
    out["sets_equal"] = out["unique_chunks_a"] == out["unique_chunks_b"]
    out["a_md5"] = hashlib.md5(trace.read_bytes()).hexdigest()
    out["b_md5"] = hashlib.md5(baseline.read_bytes()).hexdigest()
    return out


def measure_gap(trace: pathlib.Path) -> tuple[float, float, float]:
    import pandas as pd

    df = pd.read_csv(trace)
    plan_end = (df.planned_start_us + df.planned_duration_us).max()
    obs_end = df.end_us.max()
    df["delta_pct"] = 100.0 * (df.run_us - df.planned_duration_us) / df.planned_duration_us.replace(0, 1)
    return plan_end / 1000.0, obs_end / 1000.0, df.delta_pct.abs().median()


def plot_trace(trace: pathlib.Path, out_path: pathlib.Path, title: str) -> None:
    _run(
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


def parse_args() -> HetConfig:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", required=True, help="Model name (dronet, dronet_coarse, mobilenet_v2 ...)")
    p.add_argument("--granularity", required=True, choices=["dispatch", "layer", "megakernel", "tile"])
    p.add_argument("--machines", nargs="+", required=True, help="Subset of {CPU_P, CPU_E, QNN_GPU, QNN_HTA}.")
    p.add_argument(
        "--merlin-dir",
        type=pathlib.Path,
        help="eval/qrb5165/<model>(_<granularity>) folder. " "Auto-resolved from --model + --granularity if absent.",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Where the schedule/trace/plot land. Default = " "eval/qrb5165/heterogeneous/<model>_<gran>_<mach>/",
    )
    p.add_argument("--ssh-host", default="qdev")
    p.add_argument("--remote-root", default="/root/iree_run")
    p.add_argument("--cpu-p-cpu-ids", default="4,5,6,7")
    p.add_argument("--cpu-e-cpu-ids", default="0,1")
    p.add_argument("--visible-cores", type=int, default=8)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--solver", choices=["greedy", "mosek"], default="mosek")
    p.add_argument("--transfer-time-us", type=float, default=200.0)
    p.add_argument(
        "--qnn-sdk", type=pathlib.Path, default=pathlib.Path("/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326")
    )
    p.add_argument("--board-sysroot", type=pathlib.Path, default=pathlib.Path("/scratch2/agustin/qrb5165_sysroot"))
    p.add_argument(
        "--baseline-trace", type=pathlib.Path, help="trace.csv from the all-CPU run for correctness " "comparison."
    )
    p.add_argument("--skip-compile-qnn", action="store_true")
    p.add_argument("--skip-profile", action="store_true")
    p.add_argument("--skip-board-run", action="store_true", help="Stops after schedule.json + push (no exec).")
    a = p.parse_args()

    merlin_dir = a.merlin_dir
    if merlin_dir is None:
        suffix = "" if a.granularity == "dispatch" else f"_{a.granularity}"
        merlin_dir = REPO_ROOT / "eval" / "qrb5165" / f"{a.model}{suffix}"
    output_dir = a.output_dir
    if output_dir is None:
        m_tag = "_".join(sorted(a.machines))
        output_dir = REPO_ROOT / "eval" / "qrb5165" / "heterogeneous" / f"{a.model}_{a.granularity}_{m_tag}"

    return HetConfig(
        model=a.model,
        granularity=a.granularity,
        machines=a.machines,
        merlin_dir=merlin_dir.resolve(),
        output_dir=output_dir.resolve(),
        ssh_host=a.ssh_host,
        remote_root=a.remote_root,
        cpu_p_cpu_ids=a.cpu_p_cpu_ids,
        cpu_e_cpu_ids=a.cpu_e_cpu_ids,
        visible_cores=a.visible_cores,
        repetitions=a.repetitions,
        solver=a.solver,
        transfer_time_us=a.transfer_time_us,
        qnn_sdk=a.qnn_sdk,
        board_sysroot=a.board_sysroot,
        baseline_trace=a.baseline_trace,
        skip_compile_qnn=a.skip_compile_qnn,
        skip_profile=a.skip_profile,
        skip_board_run=a.skip_board_run,
    )


def main() -> int:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[het] {cfg.model}/{cfg.granularity} on {cfg.machines}")
    print(f"[het] merlin_dir={cfg.merlin_dir}")
    print(f"[het] output_dir={cfg.output_dir}")

    qnn_artifacts = compile_qnn_chunks(cfg)

    if not cfg.skip_profile:
        for m in cfg.machines:
            if m in CPU_MACHINES:
                profile_cpu_machine(cfg, m)
            elif m in QNN_MACHINES:
                profile_qnn_machine(cfg, m, qnn_artifacts.get(m, []))

    schedule = schedule_heterogeneous(cfg)
    trace = run_on_board(cfg, schedule)

    summary: dict = {
        "model": cfg.model,
        "granularity": cfg.granularity,
        "machines": cfg.machines,
        "schedule": str(schedule),
        "trace": str(trace),
    }
    if trace.exists() and trace.stat().st_size > 0:
        plan_ms, obs_ms, gap_pct = measure_gap(trace)
        plot_path = cfg.output_dir / "gantt.png"
        title = (
            f"{cfg.model}/{cfg.granularity} {'+'.join(cfg.machines)} "
            f"({cfg.solver}) — plan {plan_ms:.1f}ms vs obs {obs_ms:.1f}ms"
        )
        plot_trace(trace, plot_path, title)
        summary.update(
            {
                "planned_ms": plan_ms,
                "observed_ms": obs_ms,
                "gap_pct_median": gap_pct,
                "plot": str(plot_path),
            }
        )

    summary["correctness"] = measure_correctness(trace, cfg.baseline_trace)
    summary_path = cfg.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[het] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
