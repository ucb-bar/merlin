#!/usr/bin/env python3
"""Repeatability sweep as a CHIA workflow — the same N-repeat baseline as run_repeatability.py,
fanned out under Ray instead of run serially.

``run_repeatability.py`` runs its repeats sequentially on purpose: "the oracle/verilator is heavy;
parallel runs would contend". That is exactly the constraint a logical Ray resource expresses. Each
repeat declares ``resources={"verilator": 1}``, and ``--verilator-slots`` caps how many may hold it
at once — so repeats overlap on everything else (agent thinking, LLM latency, compile) while never
oversubscribing Verilator. ``--verilator-slots 1`` reproduces the sequential behaviour exactly.

Nothing about isolation changes. Each repeat shells out to the **unchanged**
``run_baseline_qa_loop.py`` under the main ``.venv`` interpreter, so bwrap, the redaction brokers,
the transcript audit, rate-limit backoff and ``--resume`` all keep working exactly as they do today,
and no ray/mcp/pydantic-2.12 ever enters the agent's process tree.

Runs under the CHIA venv:

  build/chia-venv/bin/python chia_repeatability.py --n 3 --stub-seconds 2   # token-free layer test
  build/chia-venv/bin/python chia_repeatability.py --n 3 --verilator-slots 2

Output lands in an aet-managed run dir (``runs/gemmini/capsule-bench/<run-id>/``) alongside the
CHIA profiler JSONL, which ``chia viz-profile`` renders.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402
import run_repeatability as RR  # noqa: E402  — reuse _load()/_agg()/DRIVER, don't reimplement

sys.path.insert(0, str(C.REPO / "merlin" / "python"))

from merlin.benchharness.chia_bridge import chia_get, chia_run, driver_python, require_chia  # noqa: E402

try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:
    require_chia()  # raises with the venv fix instructions
    raise


# max_retries=0: the driver already owns rate-limit backoff and --resume. A Ray-level retry would
# relaunch a live `claude` session on top of the one that is still checkpointing.
@ChiaFunction(resources={"verilator": 1}, num_cpus=1, max_retries=0)
def run_repeat(cmd: list[str], cwd: str, run_id: str) -> dict:
    """One repeat, on a Ray worker: shell out to the unchanged driver, return small telemetry.

    Deliberately returns a few scalars rather than the transcript — the driver already wrote
    everything durable to its own run dir, and a fat return value would just be serialized twice.
    """
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd)
    return {"run_id": run_id, "returncode": proc.returncode,
            "wall_s": round(time.monotonic() - t0, 1)}


def _repeat_cmd(a, run_id: str, run_dir: Path) -> list[str]:
    """Build the driver command for one repeat. Mirrors run_repeatability.main()."""
    if a.stub_seconds:
        # Exercises the CHIA layer (fan-out, resource gating, get(), run-dir adapter) with zero
        # tokens and no driver involvement. NOT a test of the QA loop itself.
        return [driver_python(), "-c", f"import time; time.sleep({a.stub_seconds})"]

    cmd = [driver_python(), str(RR.DRIVER), "--run-id", run_id, "--arm", a.arm,
           "--model", a.model, "--effort", a.effort, "--max-rounds", str(a.max_rounds),
           "--round-timeout", str(a.round_timeout), "--qa-timeout", str(a.qa_timeout),
           "--sandbox", a.sandbox, "--max-rate-limit-waits", str(a.max_rate_limit_waits)]
    if a.account_config_dir:
        cmd += ["--account-config-dir", a.account_config_dir]
    if a.resume or run_dir.exists():
        cmd.append("--resume")  # an existing-but-unfinished run_dir must resume, not be refused
    return cmd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--arm", default="raw_baseline")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--round-timeout", type=int, default=2700)
    ap.add_argument("--qa-timeout", type=int, default=1200)
    ap.add_argument("--sandbox", default="none")
    ap.add_argument("--start-index", type=int, default=1)
    ap.add_argument("--prefix", default="chia_rep")
    ap.add_argument("--max-rate-limit-waits", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--account-config-dir", default="")
    ap.add_argument("--verilator-slots", type=int, default=1,
                    help="how many repeats may hold the logical 'verilator' resource at once. "
                         "1 == the sequential behaviour of run_repeatability.py")
    ap.add_argument("--stub-seconds", type=float, default=0.0,
                    help="replace the driver with a sleep of N seconds: exercises the CHIA "
                         "orchestration layer token-free. Does NOT run the QA loop.")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, launch nothing")
    a = ap.parse_args(argv)

    if a.verilator_slots < 1:
        print("--verilator-slots must be >= 1", file=sys.stderr)
        return 2

    scripts_cwd = str(C.EXP / "scripts")
    planned = []  # (run_id, run_dir, cmd)
    for k in range(a.start_index, a.start_index + a.n):
        run_id = f"{a.prefix}_{k:02d}"
        run_dir = C.RUNS / a.arm / run_id
        if not a.stub_seconds and (run_dir / "run_manifest.yaml").exists():
            print(f"  {run_id}: already complete (run_manifest.yaml present) — skipping")
            continue
        planned.append((run_id, run_dir, _repeat_cmd(a, run_id, run_dir)))

    print(f"=== chia repeatability: arm={a.arm} n={len(planned)} "
          f"verilator_slots={a.verilator_slots}{' [STUB]' if a.stub_seconds else ''} ===")
    for run_id, run_dir, cmd in planned:
        print(f"  {run_id} -> {run_dir}\n     $ {' '.join(cmd)}")
    if a.dry_run:
        print("\n[dry-run] nothing launched.")
        return 0
    if not planned:
        print("nothing to run.")
        return 0

    t0 = time.monotonic()
    with chia_run(suite="capsule-bench", method=f"chia_repeat_{a.arm}", target="gemmini",
                  extra={"arm": a.arm, "model": a.model, "n": len(planned),
                         "verilator_slots": a.verilator_slots, "stub": bool(a.stub_seconds)},
                  ray_resources={"verilator": a.verilator_slots}) as run:
        refs = [run_repeat.chia_remote(cmd, scripts_cwd, rid) for rid, _, cmd in planned]
        results = chia_get(refs)  # not chia's get(): it does not unwrap a *list* while profiling

        for i, r in enumerate(results):
            run.metrics.log_scalar("repeat/wall_s", r["wall_s"], i)
            run.metrics.log_scalar("repeat/returncode", r["returncode"], i)
        wall_total = round(time.monotonic() - t0, 1)
        run.metrics.log_scalar("sweep/wall_s", wall_total, 0)

        # Real telemetry comes from each driver's own run dir, exactly as run_repeatability reads it.
        loaded = [] if a.stub_seconds else [
            m for m in (RR._load(rd) for _, rd, _ in planned) if m
        ]
        n_full = sum(1 for m in loaded if m["public"] == "4/4" and m["hidden"] == "3/3")
        run.summary = {
            "arm": a.arm, "model": a.model, "n_launched": len(planned),
            "n_graded": len(loaded), "stub": bool(a.stub_seconds),
            "verilator_slots": a.verilator_slots,
            "sweep_wall_s": wall_total,
            "fanout_wall_s": RR._agg([r["wall_s"] for r in results]),
            "nonzero_returncodes": [r["run_id"] for r in results if r["returncode"] != 0],
            "public_4of4_AND_hidden_3of3": f"{n_full}/{len(loaded)}" if loaded else None,
            "rounds_to_converge": RR._agg([m["n_rounds"] for m in loaded]) if loaded else {},
            "cost_usd": RR._agg([m["cost_usd"] for m in loaded]) if loaded else {},
        }
        (run.run_dir / "chia" / "repeats.json").write_text(
            json.dumps({"results": results, "graded": loaded}, indent=2))

        print(f"\nsweep wall {wall_total}s over {len(planned)} repeats "
              f"({a.verilator_slots} verilator slot(s))")
        print(f"run dir: {run.run_dir}")
        print(f"profile: chia viz-profile {run.profile_path} --format table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
