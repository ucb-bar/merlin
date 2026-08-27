#!/usr/bin/env python
"""Validate the GENERAL compiler features (commit e5dd143) on REAL models on the K1 board.

This is EVIDENCE that the general transforms help (or honestly do not) on real whole models — NOT
a tuning loop. It reuses the frozen harness (`merlin.mining.k1.run_on_k1`, the bitvla 9.35x
precedent path) and the multi-tier `_gate` cos-vs-host-golden. Baseline is FROZEN hand_v0.

For each (model, feature-config) it records, all measurement-only:
  * `lowering_path`     — 'vectorized' if the config's vfmacc schedule lowers the WHOLE model, else
                          'scalar_fallback' with the exact PipelineError op (whole-model-safety probe;
                          build_k1_binary silently falls back to scalar on PipelineError, so this
                          tells us whether the optimized binary actually ran vfmacc or collapsed).
  * `ll_fmuladd`        — count of llvm.intr.fmuladd (the vfmacc proxy) in the emitted .ll.
  * `ll_fixedvec`       — count of fixed-width `<N x float>` SIMD vectors (baseline RVV vectorizes
                          to vfmul.vv+vfadd.vv WITHOUT fusion; this shows it is NOT scalar).
  * `attn_vectorized`   — did the model's batch_matmul (attention) ops vectorize? (the N-tail payoff)
  * min wall_ns over N runs (CLOCK_MONOTONIC ground truth), rdtime ticks, fp32 cos vs host golden.

Honest: numbers come only from runs that actually completed and verified; not_run with the exact
blocker otherwise. Never fabricated.
"""
from __future__ import annotations

import argparse, json, tempfile, traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.mining import k1
from merlin.mining.registry import load_rvv_package
from merlin.llvmlower.lower import lower_model_file
from merlin.llvmlower.pipeline import PipelineError
from merlin.runtime.backends import zephyr_model as zm


def lower_and_characterize(model_dir: Path, pkg) -> dict:
    """Lower the whole model with this pkg's (schedule, features). Return whether it vectorized
    whole-model or fell back, the PipelineError op if any, and .ll fma/fixed-vector counts.

    Mirrors build_k1_binary's vectorized-or-fall-back decision so the report reflects what the
    K1 binary actually compiled."""
    w = Path(tempfile.mkdtemp(prefix="lc_"))
    prepared = zm._prepare_model_mlir(model_dir / "model.mlir", w, int8_compute=pkg.is_int8)
    feats = frozenset(pkg.compiler_features or []) or None
    out = {"lowering_path": None, "pipeline_error_op": None,
           "ll_fmuladd": None, "ll_fixedvec": None, "attn_vectorized": None}
    try:
        res = lower_model_file(prepared, w / "lower", targets=(), textual=True, vectorize=True,
                               transform_schedule=pkg.schedule_text, hoist_static_allocs=False,
                               features=feats)
        out["lowering_path"] = "vectorized"
    except PipelineError as e:
        out["lowering_path"] = "scalar_fallback"
        msg = str(e)
        # MLIR verifier messages read `'<dialect.op>' op ...` — pull the quoted op name before "' op"
        # structurally (no regex): find the marker, then the opening quote just before it.
        end = msg.find("' op")
        start = msg.rfind("'", 0, end) if end != -1 else -1
        op = msg[start + 1:end] if (end != -1 and start != -1) else ""
        out["pipeline_error_op"] = op or msg[-160:].replace("\n", " ")
        # build_k1_binary falls back to the SCALAR (vectorize=False) lowering for this config.
        res = lower_model_file(prepared, w / "lower_s", targets=(), textual=True,
                               vectorize=False, hoist_static_allocs=False)
    ll = Path(res.ll_path).read_text()
    out["ll_fmuladd"] = ll.count("fmuladd")
    out["ll_fixedvec"] = ll.count("x float>")
    # Attention (batch_matmul) vectorized iff the whole-model vectorized path compiled AND it forms
    # fma; if it fell back to scalar, attention is scalar. (fixed-vec without fma = vfmul+vfadd, no
    # fused vfmacc, but still vectorized SIMD.)
    if out["lowering_path"] == "vectorized":
        out["attn_vectorized"] = "vfmacc" if out["ll_fmuladd"] > 0 else "fixedwidth_simd_no_fma"
    else:
        out["attn_vectorized"] = "scalar (vfmacc path fell back whole-model)"
    return out


def run_pkg(model_dir: Path, pkg, golden: np.ndarray, n: int, tag: str, timeout: int) -> dict:
    info = lower_and_characterize(model_dir, pkg)
    print(f"  [{tag}] lowering={info['lowering_path']} fmuladd={info['ll_fmuladd']} "
          f"fixedvec={info['ll_fixedvec']} attn={info['attn_vectorized']} "
          f"err={info['pipeline_error_op']}")
    runs, cos = [], None
    for i in range(n):
        work = Path(tempfile.mkdtemp(prefix=f"k1_{tag}_{i}_"))
        try:
            res = k1.run_on_k1(model_dir, work, pkg, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            return {"tag": tag, "run_id": pkg.run_id,
                    "compiler_features": list(pkg.compiler_features), **info,
                    "status": "not_run", "blocker": f"{type(e).__name__}: {str(e)[:300]}",
                    "min_wall_ns": None, "fp32_cos": None, "runs": runs}
        g = zm._gate(res["prefix"], {"fp32": golden})
        cos = g["fp32_cos"]
        runs.append({"wall_ns": res["metrics"].get("wall_ns"),
                     "time_ticks": res["metrics"].get("time_ticks"),
                     "cycles_est": res["metrics"].get("cycles"),
                     "fp32_cos": cos, "vlen": res.get("vlen")})
        print(f"  [{tag}] run {i}: wall_ns={runs[-1]['wall_ns']} ticks={runs[-1]['time_ticks']} "
              f"cos={cos:.6f} vlen={res.get('vlen')}")
    walls = [r["wall_ns"] for r in runs if r["wall_ns"]]
    ticks = [r["time_ticks"] for r in runs if r["time_ticks"]]
    return {"tag": tag, "run_id": pkg.run_id, "compiler_features": list(pkg.compiler_features),
            **info, "status": "pass", "blocker": None,
            "min_wall_ns": min(walls) if walls else None,
            "min_time_ticks": min(ticks) if ticks else None,
            "fp32_cos": cos, "runs": runs}


# OPTIMIZED config rationale (documented in the .md): both fused_vfmacc_tiled and
# accumulator_resident_ntail are FULL schedule replacements (edit_schedule ignores its input). They
# do NOT compose — apply_schedule iterates sorted(features), so enabling BOTH lets fused_vfmacc_tiled
# (sorts last) CLOBBER the ntail N=8 batch_matmul clamp => the attention N-tail fix is LOST. The
# config that delivers the stated goal (attention vectorizes) is accumulator_resident_ntail ALONE.
# We measure all three to make the composition behaviour explicit and honest.
CONFIGS = [
    ("optimized_ntail", ["accumulator_resident_ntail"]),
    ("opt_tiled_only", ["fused_vfmacc_tiled"]),
    ("opt_combined_clobbered", ["fused_vfmacc_tiled", "accumulator_resident_ntail"]),
    # WHOLE-MODEL-SAFE composed config (the fix): a SINGLE feature carrying BOTH tail clamps inherent
    # (matmul MR_mm=1 M-tail + batch_matmul NR_bmm=8 N-tail) on the accumulator-resident tiled-vfmacc
    # recipe. Unlike opt_combined_clobbered (two full-schedule replacements that clobber), this one
    # feature composes by construction, so the M=1 token-decode matmuls (rdt2/smolVLA leading-M=1)
    # and small-N attention vectorize to vfmacc whole-model. This is the config the payoff measures.
    ("wholemodel_composed", ["accumulator_resident_wholemodel"]),
    # M-tail alone (no batch_matmul in rdt2, so this is equivalent for rdt2 — isolates the M-tail fix).
    ("mtail_only", ["accumulator_resident_mtail"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/artifacts/recaptures/smolvla_fp32_consistent")
    ap.add_argument("--baseline", default="out/artifacts/targets/rvv/hand_v0")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--configs", default="optimized_ntail,opt_combined_clobbered",
                    help="comma list of config tags from CONFIGS to run")
    ap.add_argument("--out", default="out/artifacts/measurements/k1_spacemit/k1_e2e_general_validate.json")
    a = ap.parse_args()

    md = Path(a.model)
    golden = np.load(md / "golden.npy")
    want = set(a.configs.split(","))
    base = load_rvv_package(a.baseline)

    print(f"=== MODEL {md} (golden {golden.shape}, host-captured) ===")
    print("=== BASELINE (hand_v0, FROZEN) ===")
    rb = run_pkg(md, base, golden, a.n, "baseline", a.timeout)

    opts = []
    hb = load_rvv_package(a.baseline)
    for tag, feats in CONFIGS:
        if tag not in want:
            continue
        print(f"=== {tag} (features={feats}) ===")
        pkg = replace(hb, run_id=f"e2e_{tag}", compiler_features=list(feats))
        opts.append(run_pkg(md, pkg, golden, a.n, tag, a.timeout))

    summary = {"model": str(md), "n": a.n, "golden_shape": list(golden.shape),
               "baseline": rb, "optimized": opts}
    for ro in opts:
        if rb.get("min_wall_ns") and ro.get("min_wall_ns"):
            ro["speedup_vs_baseline"] = rb["min_wall_ns"] / ro["min_wall_ns"]
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
