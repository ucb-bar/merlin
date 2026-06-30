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
import shutil
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
            g = zm._gate(res["prefix"], {"fp32": golden})
            cos = g["fp32_cos"]
            n_xnn = res.get("n_xnn_routed", 0)
            runs.append({"wall_ns": res["metrics"].get("wall_ns"),
                         "time_ticks": res["metrics"].get("time_ticks"),
                         "cycles_est": res["metrics"].get("cycles"),
                         "fp32_cos": cos, "vlen": res.get("vlen")})
            print(f"  [{tag}] run {i}: wall_ns={runs[-1]['wall_ns']} cos={cos:.6f} "
                  f"vlen={res.get('vlen')} n_xnn={n_xnn}")
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:400]}"
            print(f"  [{tag}] run {i}: BLOCKED — {blocker}")
            break
        finally:
            # Each build tempdir holds an ~897 MB copy of the model weights; clean it after the
            # run so a multi-config/N-rep campaign doesn't fill the root fs (the ENOSPC we just hit).
            shutil.rmtree(work, ignore_errors=True)
    walls = sorted(r["wall_ns"] for r in runs if r["wall_ns"])
    spread = None
    if walls:
        mn = walls[0]; mx = walls[-1]; med = walls[len(walls)//2]
        mean = sum(walls)/len(walls)
        std = (sum((w-mean)**2 for w in walls)/len(walls))**0.5
        spread = {"min_ns": mn, "max_ns": mx, "median_ns": med,
                  "stdev_ns": round(std), "range_pct": round(100.0*(mx-mn)/mn, 2), "n": len(walls)}
    return {"tag": tag, "run_id": pkg.run_id,
            "compiler_features": list(pkg.compiler_features or []),
            "kernel_backend": kernel_backend,
            "n_xnn_routed": n_xnn,
            "min_wall_ns": walls[0] if walls else None,
            "spread": spread,
            "fp32_cos": cos, "ok": (cos is not None and cos >= 0.9999),
            "blocker": blocker, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/recaptures/bitvla_fp32_consistent")
    ap.add_argument("--baseline", default="generated_targets/rvv/hand_v0")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--configs", default="baseline,ours_tiled,ours_v3,xnnpack_kernels",
                    help="comma list of: baseline,ours_tiled,ours_v3,ours_wholemodel,xnnpack_kernels")
    ap.add_argument("--out", default="artifacts/measurements/k1_spacemit/k1_e2e_xnnpack_bitvla.json")
    a = ap.parse_args()

    md = Path(a.model)
    golden = np.load(md / "golden.npy")
    base = load_rvv_package(a.baseline)
    want = set(a.configs.split(","))
    ours = replace(base, run_id="ours_vfmacc_tiled", compiler_features=["fused_vfmacc_tiled"])
    ours_v3 = replace(base, run_id="ours_v3", compiler_features=["accumulator_resident_microkernel_v3"])
    # ours-wholemodel = the beam's best kernel on openvla/rdt2 (whole-model-safe tail clamps).
    ours_wm = replace(base, run_id="ours_wholemodel",
                      compiler_features=["accumulator_resident_wholemodel"])
    # ours-wholemodel-vf = wholemodel tail clamps + v3's .vf scalarize (kills the .vv broadcast ladder).
    ours_wm_vf = replace(base, run_id="ours_wholemodel_vf",
                         compiler_features=["accumulator_resident_wholemodel_vf"])
    xnn_pkg = replace(base, run_id="xnnpack_kernels")
    ob_pkg = replace(base, run_id="openblas_kernels")

    def maybe(tag, pkg, backend):
        if tag not in want:
            return {"tag": tag, "skipped": True, "min_wall_ns": None, "fp32_cos": None,
                    "ok": False, "n_xnn_routed": 0, "spread": None, "blocker": "not in --configs",
                    "compiler_features": list(pkg.compiler_features or []), "runs": []}
        print(f"=== {tag} ===")
        return run_cfg(md, pkg, golden, a.n, tag, backend)

    rb = maybe("baseline", base, None)
    ro = maybe("ours_tiled", ours, None)
    rv = maybe("ours_v3", ours_v3, None)
    rw = maybe("ours_wholemodel", ours_wm, None)
    rwv = maybe("ours_wholemodel_vf", ours_wm_vf, None)
    rx = maybe("xnnpack_kernels", xnn_pkg, "xnnpack")
    rob = maybe("openblas_kernels", ob_pkg, "openblas")

    def spd(a_ns, b_ns):
        return (a_ns / b_ns) if (a_ns and b_ns) else None

    # best ours config that actually ran (highest speedup vs baseline)
    ours_cands = {"ours_tiled": ro, "ours_v3": rv, "ours_wholemodel": rw, "ours_wholemodel_vf": rwv}
    ours_best_tag = max((t for t, r in ours_cands.items() if r.get("min_wall_ns")),
                        key=lambda t: spd(rb["min_wall_ns"], ours_cands[t]["min_wall_ns"]) or 0,
                        default=None)
    ours_best = ours_cands.get(ours_best_tag) if ours_best_tag else None
    summary = {
        "model": str(md), "n": a.n, "board": "k1_spacemit", "vlen": k1.VLEN, "same_pass": True,
        "timer": "CLOCK_MONOTONIC wall_ns; cycle_accurate=false",
        "note": ("Same-pass head-to-head vs the SAME baseline in ONE pass. xnnpack-kernels routes f32 "
                 "linalg.matmul to xnn_f32_gemm_ukernel_1x4v__rvv with RESIDENT-WEIGHT pack (excluded "
                 "from the timed path, matching ours' pack-free scope). cos gated before any wall."),
        "configs_run": sorted(want),
        "baseline": rb, "ours_tiled": ro, "ours_v3": rv, "ours_wholemodel": rw,
        "ours_wholemodel_vf": rwv, "xnnpack_kernels": rx, "openblas_kernels": rob,
        "speedup_ours_tiled": spd(rb["min_wall_ns"], ro["min_wall_ns"]),
        "speedup_ours_v3": spd(rb["min_wall_ns"], rv["min_wall_ns"]),
        "speedup_ours_wholemodel": spd(rb["min_wall_ns"], rw["min_wall_ns"]),
        "speedup_ours_wholemodel_vf": spd(rb["min_wall_ns"], rwv["min_wall_ns"]),
        "speedup_xnnpack": spd(rb["min_wall_ns"], rx["min_wall_ns"]),
        "speedup_openblas": spd(rb["min_wall_ns"], rob["min_wall_ns"]),
        "v3_over_xnnpack": spd(rx["min_wall_ns"], rv["min_wall_ns"]),
        "ours_best_tag": ours_best_tag,
        "ours_best_over_xnnpack": spd(rx["min_wall_ns"], ours_best["min_wall_ns"]) if ours_best else None,
        "ours_best_over_openblas": spd(rob["min_wall_ns"], ours_best["min_wall_ns"]) if ours_best else None,
        "xnnpack_kernel": "xnn_f32_gemm_ukernel_1x4v__rvv",
        "openblas_kernel": "sgemm_kernel_8x8_zvl128b",
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
    rows = [("baseline (hand_v0)", s["baseline"], None),
            ("ours-tiled (fused_vfmacc_tiled)", s["ours_tiled"], s["speedup_ours_tiled"]),
            ("ours-v3 (accum-resident microkernel)", s["ours_v3"], s["speedup_ours_v3"]),
            ("ours-wholemodel (accum-resident, tail-safe)", s["ours_wholemodel"], s["speedup_ours_wholemodel"]),
            ("ours-wholemodel-vf (.vf, no broadcast ladder)", s.get("ours_wholemodel_vf", {}), s.get("speedup_ours_wholemodel_vf")),
            ("xnnpack-kernels (RVV ukernel, resident pack)", s["xnnpack_kernels"], s["speedup_xnnpack"]),
            ("openblas-kernels (sgemm 8x8, resident pack)", s.get("openblas_kernels", {}), s.get("speedup_openblas"))]
    rows = [(n, r, sp) for (n, r, sp) in rows if r and not r.get("skipped")]
    lines = [
        f"# K1 whole-model SAME-PASS head-to-head — {Path(s['model']).name}",
        "",
        f"Board: SpacemiT K1 (real RVV silicon, VLEN={s['vlen']}). N={s['n']} runs/config, ONE pass, "
        f"min CLOCK_MONOTONIC wall + spread. Timer: {s['timer']}.",
        f"XNNPACK kernel: `{s['xnnpack_kernel']}` (resident-weight pack, excluded from timed path). "
        "cos gated vs host golden before any wall.",
        "",
        "| config | min wall (ns) | range % (N) | fp32 cos | speedup | #xnn | ok | blocker |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r, sp in rows:
        spr = r.get("spread") or {}
        rng = f"{spr.get('range_pct','—')}% ({spr.get('n','—')})" if spr else "—"
        lines.append(
            f"| {name} | {_fmt(r['min_wall_ns'])} | {rng} | {_fmt(r['fp32_cos'], 7)} | "
            f"{_fmt(sp)}x | {r['n_xnn_routed']} | {'yes' if r['ok'] else 'NO'} | {r.get('blocker') or '—'} |")
    bx = s.get("ours_best_over_xnnpack")
    bt = s.get("ours_best_tag")
    verdict = ("ours FASTER than XNNPACK" if (bx and bx > 1) else
               "XNNPACK faster than ours" if bx else "—")
    lines += [
        "",
        "## Headline (same-pass, fair resident-weight pack)",
        f"- **best-ours ({bt}) / xnnpack = {_fmt(bx)}x** — {verdict} (>1 ⇒ our compiler kernel faster).",
        f"- speedups vs baseline: tiled {_fmt(s['speedup_ours_tiled'])}x · v3 {_fmt(s['speedup_ours_v3'])}x · "
        f"wholemodel {_fmt(s['speedup_ours_wholemodel'])}x · xnnpack {_fmt(s['speedup_xnnpack'])}x.",
        "",
        "## Takeaway",
        s["note"],
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
