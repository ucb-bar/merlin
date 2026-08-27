#!/usr/bin/env python3
"""chia-driven RVV beam (BB3) — fan each generation's fork certifications out as Ray tasks gated on a
single-slot ``k1`` resource, so K1 forks serialize on the one board while spike-only forks overlap.

This is the CHIA orchestration layer over :func:`merlin.mining.beam_cli.run_instrumented_beam`. It runs
under the CHIA venv (chia hard-pins ray/pydantic — see ``merlin.benchharness.chia_bridge``):

  build/chia-venv/bin/python build_tools/scripts/rvv_chia_beam.py \
      --model-dir <workload> --expert-objdump merlin/tests/data/cca_asm/xnnpack_f32_gemm_rvv.objdump \
      --k1-slots 1

The beam engine (propose -> rank -> audit -> escalate, gen by gen) stays on the Ray driver holding the
aet parent handle; only ``run_sweep`` (the per-generation certify fan-out) is offloaded to Ray via a
``sweep_fn`` that submits one ``@ChiaFunction(resources={"k1": k1_slots})`` task per fork. ``--k1-slots 1``
is the board gate; the host-wide ``k1.board_lock`` flock is the stronger cross-process serialization
that also protects against OTHER sessions on this machine. Each fork's certify still fails-closed
inside the task (one fork's error never aborts the sweep) exactly as the ThreadPool sweep does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.benchharness.chia_bridge import chia_get, chia_run, require_chia  # noqa: E402
from merlin.common.paths import repo_root  # noqa: E402


def _make_chia_sweep(k1_slots: int):
    """Build a ``sweep_fn(jobs, certify_fn, max_workers)`` that fans forks out as Ray tasks."""
    from chia.base.ChiaFunction import ChiaFunction

    @ChiaFunction(resources={"k1": k1_slots}, num_cpus=1, max_retries=0)
    def _certify_task(job: dict) -> dict:
        # runs on a Ray worker under the chia venv; import inside so the closure stays serializable.
        import sys as _sys
        _sys.path.insert(0, str(REPO / "merlin" / "python"))
        from merlin.mining.runner import certify_rvv
        try:
            return certify_rvv(**job)
        except Exception as e:  # fork error must not abort the sweep (same contract as run_sweep)
            return {"status": "error", "error": f"{type(e).__name__}: {e}", "run_id": job.get("run_id")}

    def chia_sweep(jobs, *, certify_fn=None, max_workers=None):
        # certify_fn/max_workers are ignored: certification runs in the Ray task (certify_rvv), and
        # concurrency is governed by the k1 resource, not a ThreadPool.
        if not jobs:
            return []
        refs = [_certify_task.chia_remote(j) for j in jobs]
        return chia_get(refs)

    return chia_sweep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rvv-chia-beam")
    ap.add_argument("--seed-pkg", default=None)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--expert-objdump", required=True)
    ap.add_argument("--op", default="matmul")
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--shape-regime", default="square")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--k1-slots", type=int, default=1,
                    help="how many forks may hold the logical 'k1' board resource at once (1 = serial)")
    a = ap.parse_args(argv)

    require_chia()
    from merlin.mining.beam_cli import run_instrumented_beam

    seed_pkg = a.seed_pkg or str(repo_root() / "out/artifacts/targets/rvv/hand_v0")
    sweep_fn = _make_chia_sweep(a.k1_slots)
    # chia_run owns the aet parent + Ray init (declares the k1 resource); run_instrumented_beam opens
    # its own aet parent for the beam tree, and the sweep tasks run under the Ray cluster chia_run set up.
    with chia_run(suite=f"beam/{a.op}", method="chia_cca_beam", target="rvv",
                  extra={"op": a.op, "k1_slots": a.k1_slots, "workload": Path(a.model_dir).name},
                  ray_resources={"k1": a.k1_slots}) as run:
        res = run_instrumented_beam(
            seed_pkg=seed_pkg, model_dir=a.model_dir, expert_objdump=a.expert_objdump,
            op=a.op, dtype=a.dtype, shape_regime=a.shape_regime, targets=("k1",),
            width=a.width, depth=a.depth, top_k=a.top_k, sweep_fn=sweep_fn)
        best = res.get("best") or {}
        run.summary = {"best_run_id": best.get("run_id"), "best_speedup": best.get("speedup"),
                       "best_lever": best.get("lever"), "n_forks": len(res.get("nodes", [])) - 1}
        print(f"parent_run={res.get('parent_run_dir')}")
        print(f"best: run_id={best.get('run_id')} lever={best.get('lever')} "
              f"speedup={best.get('speedup')} gate_ok={best.get('gate_ok')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
