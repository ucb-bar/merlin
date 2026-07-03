#!/usr/bin/env python
"""K1 KERNEL-SPEEDUP sweep: can OUR GEMM kernel close the measured matmul-bucket gap to XNNPACK 7x4v
by matching its REGISTER BLOCKING (MR)? — the "first try the kernel speedup" experiment.

The fair four-way found our matmul bucket 3-14x slower than XNNPACK 7x4v (MR=7), root-caused to a
register-block (MR) blind spot: our v3 ceiling kernel is MR=4, the small-M-safe whole-model path is
MR=1, while XNNPACK reuses each loaded B-row across MR=7 vfmacc (1+1/MR loads/useful-FMA, MR
independent accumulator chains to hide vfmacc latency). The `ours_board` shim is now MR-configurable
(-DOURS_MR), so this sweeps MR in {1,4,7} and MEASURES the matmul bucket (rdtime bracket, same as the
XNNPACK arm) to see how far higher MR closes the COMPUTE gap — reported SEPARATELY from the dispatch/
runtime overhead (wall - matmul = dispatch bucket), per the "compare compute speedup besides the
overhead" directive.

Both arms are the SAME baseline non-matmul lowering with only the matmul kernel swapped + BOTH timed,
so this is apples-to-apples at the kernel level. cos-gated (>=0.9999) before any wall is recorded.

Run: MERLIN_K1_HOST=root@<ip> .venv/bin/python scripts/k1_kernel_speedup.py \
        --model output/bitvla_fp32_consistent --mrs 1,4,7 -n 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm

TIMEBASE_HZ = k1.K1_TIMEBASE_HZ


def _ns(ticks):
    return None if ticks is None else float(ticks) * (1e9 / TIMEBASE_HZ)


def _run_cfg(model_dir, pkg, golden, n, tag, kernel_backend, ours_mr):
    """Run one config N times; gate cos each run; collect wall + matmul ticks (dispatch_timing on)."""
    runs, cos, n_routed, blocker = [], 0.0, 0, None
    for i in range(n):
        work = Path(tempfile.mkdtemp(prefix=f"k1ksp_{tag}_{i}_"))
        try:
            res = k1.run_on_k1(model_dir, work, pkg, timeout=1800, kernel_backend=kernel_backend,
                               dispatch_timing=True, ours_mr=ours_mr)
            g = zm._gate(res["prefix"], {"fp32": golden})
            cos = g["fp32_cos"]
            n_routed = res.get("n_ours_routed", res.get("n_xnn_routed", 0))
            m = res["metrics"]
            runs.append({"wall_ns": m.get("wall_ns"), "matmul_ticks": m.get("matmul_ticks"),
                         "matmul_calls": m.get("matmul_calls"), "fp32_cos": cos})
            mb = _ns(m.get("matmul_ticks"))
            print(f"  [{tag}] run {i}: wall_ns={m.get('wall_ns')} cos={cos:.7f} "
                  f"n_routed={n_routed} matmul_ns={None if mb is None else round(mb)} "
                  f"calls={m.get('matmul_calls')}")
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [{tag}] run {i}: BLOCKED — {blocker}")
            break
        finally:
            shutil.rmtree(work, ignore_errors=True)

    walls = [r["wall_ns"] for r in runs if r["wall_ns"]]
    min_wall = min(walls) if walls else None
    matmul_ns = dispatch_ns = matmul_frac = matmul_calls = None
    if min_wall is not None:
        best = min(runs, key=lambda r: r["wall_ns"] if r["wall_ns"] else 1 << 62)
        matmul_ns = _ns(best.get("matmul_ticks"))
        matmul_calls = best.get("matmul_calls")
        if matmul_ns is not None:
            dispatch_ns = float(min_wall) - matmul_ns
            matmul_frac = matmul_ns / float(min_wall)
    return {"tag": tag, "kernel_backend": kernel_backend, "ours_mr": ours_mr,
            "n_routed": n_routed, "min_wall_ns": min_wall, "matmul_bucket_ns": matmul_ns,
            "dispatch_bucket_ns": dispatch_ns, "matmul_frac": matmul_frac, "matmul_calls": matmul_calls,
            "fp32_cos": cos, "ok": (cos is not None and cos >= 0.9999), "blocker": blocker, "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/recaptures/bitvla_fp32_consistent")
    ap.add_argument("--baseline", default="artifacts/targets/rvv/hand_v0")
    ap.add_argument("--mrs", default="1,4,7", help="ours register-block MR values to sweep")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--out", default="artifacts/measurements/k1_spacemit/k1_kernel_speedup.json")
    a = ap.parse_args()

    md = Path(a.model)
    golden = np.load(md / "golden.npy")
    base = load_rvv_package(a.baseline)
    mrs = [int(x) for x in a.mrs.split(",") if x]

    results = {}
    # XNNPACK 7x4v reference (its matmul bucket = the target our kernel must close to).
    print("=== xnnpack_kernels (7x4v, MR=7 reference) ===")
    results["xnnpack_kernels"] = _run_cfg(md, replace(base, run_id="xnnpack_kernels"), golden, a.n,
                                          "xnnpack_kernels", "xnnpack", 4)
    # ours kernel swept over MR (same baseline non-matmul lowering; only the matmul register block changes).
    for mr in mrs:
        tag = f"ours_kernels_mr{mr}"
        print(f"=== {tag} (ours v3 vfmacc.vf, MR={mr}) ===")
        results[tag] = _run_cfg(md, replace(base, run_id=tag), golden, a.n, tag, "ours", mr)

    xn = results["xnnpack_kernels"]
    xm = xn.get("matmul_bucket_ns")
    sweep = []
    for mr in mrs:
        r = results[f"ours_kernels_mr{mr}"]
        om = r.get("matmul_bucket_ns")
        sweep.append({
            "ours_mr": mr,
            "ours_matmul_bucket_ns": om, "xnnpack_matmul_bucket_ns": xm,
            "ours_over_xnnpack_matmul": (om / xm) if (om and xm) else None,
            "ours_wall_ns": r.get("min_wall_ns"), "xnnpack_wall_ns": xn.get("min_wall_ns"),
            "ours_dispatch_bucket_ns": r.get("dispatch_bucket_ns"),
            "xnnpack_dispatch_bucket_ns": xn.get("dispatch_bucket_ns"),
            "cos_ok": r.get("ok"),
        })

    summary = {
        "model": str(md), "n": a.n, "board": "k1_spacemit", "vlen": k1.VLEN,
        "timebase_hz": TIMEBASE_HZ,
        "method": ("ours_board shim MR-configurable (-DOURS_MR); matmul bucket measured via rdtime in "
                   "BOTH arms (same baseline non-matmul lowering, only the matmul register block "
                   "swapped). Tests whether matching XNNPACK 7x4v register blocking (MR=7) closes the "
                   "COMPUTE gap; dispatch bucket = wall - matmul reported separately (the runtime "
                   "overhead)."),
        "xnnpack_kernel": "xnn_f32_gemm_ukernel_7x4v__rvv (MR=7)",
        "mr_sweep": sweep, "results": results,
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    print("\n=== MR SWEEP (matmul bucket = COMPUTE; dispatch = OVERHEAD) ===")
    for s in sweep:
        r = s["ours_over_xnnpack_matmul"]
        print(f"  MR={s['ours_mr']}: ours_matmul={None if s['ours_matmul_bucket_ns'] is None else round(s['ours_matmul_bucket_ns']/1e6,1)}ms "
              f"xnn_matmul={None if xm is None else round(xm/1e6,1)}ms "
              f"ours/xnn={None if r is None else round(r,2)}  cos_ok={s['cos_ok']}")
    print(f"\nwrote -> {outp}")


if __name__ == "__main__":
    main()
