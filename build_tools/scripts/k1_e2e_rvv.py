#!/usr/bin/env python
"""PART 1 — e2e whole-model before/after RVV-optimization on the real K1 board.

Builds tiny_llama (fp32 vectorized-matmul path; the fp8_consistent capture is all-f32 in the
graph) twice and runs each N times on the K1 over SSH:
  (a) BASELINE   = hand_v0 (frozen RVV transform schedule).
  (b) OPTIMIZED  = an impr fork enabling `fused_vfmacc_tiled` (bounded tiled vfmacc).

For each it records: which MLIR lowering path actually compiled (vectorized vs scalar fallback —
this exposes whole-model safety of the feature), the min whole-model wall_ns over N runs, and the
fp32 cosine vs golden.npy. Honest: if the feature's vectorized lowering raises (whole-model
unsafe), build_k1_binary silently falls back to scalar; we DETECT and REPORT that instead of
pretending the optimized binary ran the tiled vfmacc.
"""
from __future__ import annotations

import argparse, json, tempfile, traceback
from pathlib import Path

import numpy as np

from merlin.rvvgen import k1
from merlin.rvvgen.registry import load_rvv_package
from merlin.llvmlower.lower import lower_model_file
from merlin.llvmlower.pipeline import PipelineError
from merlin.runtime.backends import zephyr_model as zm


def lowering_path(model_dir: Path, pkg) -> str:
    """Return 'vectorized' if the package's vectorized lowering compiles for this whole model,
    else 'scalar_fallback' (the feature's recipe raised PipelineError -> build_k1_binary falls
    back to scalar). This is the whole-model-safety probe."""
    w = Path(tempfile.mkdtemp(prefix="lpath_"))
    prepared = zm._prepare_model_mlir(model_dir / "model.mlir", w, int8_compute=pkg.is_int8)
    feats = frozenset(pkg.compiler_features or []) or None
    try:
        lower_model_file(prepared, w / "lower", targets=(), textual=True, vectorize=True,
                         transform_schedule=pkg.schedule_text, hoist_static_allocs=False,
                         features=feats)
        return "vectorized"
    except PipelineError:
        return "scalar_fallback"


def fmuladd_count(model_dir: Path, pkg) -> int:
    w = Path(tempfile.mkdtemp(prefix="fmc_"))
    prepared = zm._prepare_model_mlir(model_dir / "model.mlir", w, int8_compute=pkg.is_int8)
    feats = frozenset(pkg.compiler_features or []) or None
    try:
        res = lower_model_file(prepared, w / "lower", targets=(), textual=True, vectorize=True,
                               transform_schedule=pkg.schedule_text, hoist_static_allocs=False,
                               features=feats)
    except PipelineError:
        res = lower_model_file(prepared, w / "lower_s", targets=(), textual=True,
                               vectorize=False, hoist_static_allocs=False)
    return Path(res.ll_path).read_text().count("fmuladd")


def run_pkg(model_dir: Path, pkg, golden: np.ndarray, n: int, tag: str) -> dict:
    lp = lowering_path(model_dir, pkg)
    fmc = fmuladd_count(model_dir, pkg)
    runs = []
    cos = None
    for i in range(n):
        work = Path(tempfile.mkdtemp(prefix=f"k1_{tag}_{i}_"))
        res = k1.run_on_k1(model_dir, work, pkg, timeout=900)
        g = zm._gate(res["prefix"], {"fp32": golden})
        runs.append({"wall_ns": res["metrics"].get("wall_ns"),
                     "time_ticks": res["metrics"].get("time_ticks"),
                     "cycles_est": res["metrics"].get("cycles"),
                     "fp32_cos": g["fp32_cos"], "vlen": res.get("vlen")})
        cos = g["fp32_cos"]
        print(f"  [{tag}] run {i}: wall_ns={runs[-1]['wall_ns']} cos={cos:.6f} vlen={res.get('vlen')}")
    walls = [r["wall_ns"] for r in runs if r["wall_ns"]]
    return {"tag": tag, "run_id": pkg.run_id, "compiler_features": list(pkg.compiler_features),
            "lowering_path": lp, "fmuladd_in_ll": fmc,
            "min_wall_ns": min(walls) if walls else None,
            "fp32_cos": cos, "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/recaptures/tiny_llama_fp8_consistent")
    ap.add_argument("--baseline", default="artifacts/targets/rvv/hand_v0")
    ap.add_argument("--optimized", required=True)
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--out", default="artifacts/measurements/k1_spacemit/k1_e2e_tiny_llama.json")
    a = ap.parse_args()

    md = Path(a.model)
    golden = np.load(md / "golden.npy")
    base = load_rvv_package(a.baseline)
    opt = load_rvv_package(a.optimized)

    print("=== BASELINE (hand_v0) ===")
    rb = run_pkg(md, base, golden, a.n, "baseline")
    print("=== OPTIMIZED (fused_vfmacc_tiled) ===")
    ro = run_pkg(md, opt, golden, a.n, "optimized")

    speedup = (rb["min_wall_ns"] / ro["min_wall_ns"]
               if rb["min_wall_ns"] and ro["min_wall_ns"] else None)
    summary = {"model": str(md), "n": a.n, "baseline": rb, "optimized": ro,
               "e2e_speedup_baseline_over_optimized": speedup}
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
