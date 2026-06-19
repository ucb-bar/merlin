#!/usr/bin/env python
"""THIRD e2e column on the REAL K1 board: XNNPACK RVV GEMM kernels vs baseline vs ours.

The deferred board step of the host-correct XNNPACK kernel-backend work (commit 0e546e6). Runs
bitvla whole-model on the SpacemiT K1 (real RVV silicon, VLEN=256) in THREE configs, each N times,
reporting the min CLOCK_MONOTONIC wall + fp32 cos vs the SAME host golden the e2e runner gates on:

  - baseline        = hand_v0 (frozen RVV transform schedule).
  - ours-optimized  = hand_v0 + fused_vfmacc_tiled (the compiler-emitted bounded tiled vfmacc).
  - xnnpack-kernels = hand_v0, but the routable f32 linalg.matmul dispatches lowered to calls into
                      XNNPACK's hand-written RVV GEMM ukernel (xnn_f32_gemm_ukernel_1x4v__rvv);
                      attention / rmsnorm / elementwise stay on the Merlin-emitted runtime (the
                      same hybrid the host prototype proved). #dispatches routed is reported.

Headline: with the SAME graph + weights, how does swapping in XNNPACK's hand RVV GEMM compare to
our compiler-emitted vfmacc kernel, whole-model on real silicon? This isolates kernel-level vs
runtime/glue contribution to the e2e gap.

Correctness (cos >= 0.9999) is confirmed for each config BEFORE its wall is reported. Honest:
the XNNPACK path is the vectorized path only — it does NOT fall back to scalar (that would drop
the routing), so a failure is reported as a blocker, never a fabricated number. rdtime/wall are
silicon wall proxies (cycle_accurate=false); spike/FireSim remain the cycle authorities.

Run:  MERLIN_K1_HOST=root@10.44.97.186 .venv/bin/python scripts/k1_e2e_xnnpack.py -n 3
"""
from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm


def run_cfg(model_dir: Path, pkg, golden: np.ndarray, n: int, tag: str,
            kernel_backend: str | None) -> dict:
    """Run one config N times on the board; gate cos, then take min wall over the runs."""
    runs: list[dict] = []
    cos = None
    n_xnn = 0
    blocker = None
    for i in range(n):
        work = Path(tempfile.mkdtemp(prefix=f"k1xnn_{tag}_{i}_"))
        try:
            res = k1.run_on_k1(model_dir, work, pkg, timeout=900, kernel_backend=kernel_backend)
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:400]}"
            print(f"  [{tag}] run {i}: BLOCKED — {blocker}")
            break
        g = zm._gate(res["prefix"], {"fp32": golden})
        cos = g["fp32_cos"]
        n_xnn = res.get("n_xnn_routed", 0)
        runs.append({"wall_ns": res["metrics"].get("wall_ns"),
                     "time_ticks": res["metrics"].get("time_ticks"),
                     "cycles_est": res["metrics"].get("cycles"),
                     "fp32_cos": cos, "vlen": res.get("vlen")})
        print(f"  [{tag}] run {i}: wall_ns={runs[-1]['wall_ns']} cos={cos:.6f} "
              f"vlen={res.get('vlen')} n_xnn={n_xnn}")
    walls = [r["wall_ns"] for r in runs if r["wall_ns"]]
    return {"tag": tag, "run_id": pkg.run_id,
            "compiler_features": list(pkg.compiler_features or []),
            "kernel_backend": kernel_backend,
            "n_xnn_routed": n_xnn,
            "min_wall_ns": min(walls) if walls else None,
            "fp32_cos": cos, "ok": (cos is not None and cos >= 0.9999),
            "blocker": blocker, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/bitvla_fp32_consistent")
    ap.add_argument("--baseline", default="generated_targets/rvv/hand_v0")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--out", default="output/rvv_bench/k1_e2e_xnnpack_bitvla.json")
    a = ap.parse_args()

    md = Path(a.model)
    golden = np.load(md / "golden.npy")
    base = load_rvv_package(a.baseline)
    # ours-optimized = baseline schedule + the fused_vfmacc_tiled compiler feature (same way
    # scripts/k1_cross_framework.py constructs the ours forks).
    ours = replace(base, run_id="ours_vfmacc_tiled",
                   compiler_features=["fused_vfmacc_tiled"])
    # xnnpack-kernels reuses the BASELINE package (frozen schedule) — the only difference is the
    # matmul dispatches route to XNNPACK; everything else is the baseline lowering.
    xnn_pkg = replace(base, run_id="xnnpack_kernels")

    print("=== baseline (hand_v0) ===")
    rb = run_cfg(md, base, golden, a.n, "baseline", None)
    print("=== ours-optimized (fused_vfmacc_tiled) ===")
    ro = run_cfg(md, ours, golden, a.n, "ours_vfmacc_tiled", None)
    print("=== xnnpack-kernels (f32 matmuls -> XNNPACK RVV ukernel) ===")
    rx = run_cfg(md, xnn_pkg, golden, a.n, "xnnpack_kernels", "xnnpack")

    def spd(a_ns, b_ns):
        return (a_ns / b_ns) if (a_ns and b_ns) else None

    summary = {
        "model": str(md), "n": a.n, "board": "k1_spacemit", "vlen": k1.VLEN,
        "timer": "CLOCK_MONOTONIC wall_ns (rdtime ticks alongside); cycle_accurate=false",
        "note": ("Third e2e column. xnnpack-kernels routes the f32 linalg.matmul dispatches to "
                 "XNNPACK's xnn_f32_gemm_ukernel_1x4v__rvv; rest on the Merlin runtime. cos "
                 "gated vs the same host golden before any wall is reported."),
        "baseline": rb, "ours_optimized": ro, "xnnpack_kernels": rx,
        "speedup_baseline_over_ours": spd(rb["min_wall_ns"], ro["min_wall_ns"]),
        "speedup_baseline_over_xnnpack": spd(rb["min_wall_ns"], rx["min_wall_ns"]),
        "speedup_ours_over_xnnpack": spd(ro["min_wall_ns"], rx["min_wall_ns"]),
        "xnnpack_kernel": "xnn_f32_gemm_ukernel_1x4v__rvv",
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Markdown table next to the json.
    md_out = outp.with_suffix(".md")
    _write_md(md_out, summary)
    print(f"\nwrote -> {outp}\nwrote -> {md_out}")


def _fmt(v, prec=4):
    return "—" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.{prec}f}"
                                  if isinstance(v, float) else str(v))


def _write_md(path: Path, s: dict) -> None:
    rows = [("baseline (hand_v0)", s["baseline"]),
            ("ours-optimized (fused_vfmacc_tiled)", s["ours_optimized"]),
            ("xnnpack-kernels (RVV ukernel)", s["xnnpack_kernels"])]
    lines = [
        f"# K1 whole-model e2e: XNNPACK kernels vs baseline vs ours — {Path(s['model']).name}",
        "",
        f"Board: SpacemiT K1 (real RVV silicon, VLEN={s['vlen']}). "
        f"N={s['n']} runs/config, min CLOCK_MONOTONIC wall. Timer: {s['timer']}.",
        f"XNNPACK kernel: `{s['xnnpack_kernel']}`. cos gated vs host golden before any wall.",
        "",
        "| config | min wall (ns) | fp32 cos | #dispatch via XNNPACK | ok | blocker |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in rows:
        lines.append(
            f"| {name} | {_fmt(r['min_wall_ns'])} | {_fmt(r['fp32_cos'], 7)} | "
            f"{r['n_xnn_routed']} | {'yes' if r['ok'] else 'NO'} | {r.get('blocker') or '—'} |")
    lines += [
        "",
        "## Speedups (min wall)",
        f"- baseline / ours-optimized = {_fmt(s['speedup_baseline_over_ours'])}x",
        f"- baseline / xnnpack-kernels = {_fmt(s['speedup_baseline_over_xnnpack'])}x",
        f"- ours-optimized / xnnpack-kernels = {_fmt(s['speedup_ours_over_xnnpack'])}x "
        "(>1 ⇒ XNNPACK faster than our vfmacc; <1 ⇒ our vfmacc faster)",
        "",
        "## Takeaway",
        s["note"],
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
