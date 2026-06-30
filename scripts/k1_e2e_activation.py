#!/usr/bin/env python
"""VALIDATION 1 — does `vectorized_transcendental_activation` LIFT whole-model e2e on K1 silicon?

Reuses the FROZEN e2e harness (`scripts/k1_e2e_general_validate.run_pkg` ->
`merlin.rvvgen.k1.run_on_k1` -> `build_k1_binary`, the same bitvla 9.35x precedent path and the
multi-tier cos-vs-host-golden gate). Baseline is FROZEN hand_v0. Measurement only.

Per model (bitvla, openvla) it runs these configs, N reps each, min wall, cos vs the SAME host
golden the e2e runner uses (golden.npy):
  (a) baseline            — hand_v0, features=[]                              (matmul vectorizes)
  (b) act_alone           — ["vectorized_transcendental_activation"]          (the activation feature)
  (c) matmul_only         — ["fused_vfmacc_tiled"]                            (matmul vfmacc, comparison)
  (d) act_plus_matmul     — ["vectorized_transcendental_activation","fused_vfmacc_tiled"]
                            COMPOSITION PROBE: both are schedule_replace=True, so apply_schedule
                            raises CompositionError. We attempt the lowering and RECORD the
                            composition verdict honestly (it does NOT compose).

CRITICAL HONESTY NOTE (recorded per row via lower_and_characterize): on a WHOLE model the
activation feature's `_ACT_POLY_SCHEDULE` tiles+vectorizes EVERY linalg.generic [16], which fails on
the model's non-activation generics -> PipelineError -> `build_k1_binary` silently falls back to the
SCALAR (vectorize=False, no-feature) lowering. So `act_alone` whole-model does NOT vectorize the
activation here; it is a KERNEL-level feature (validated isolated in V2). The `lowering_path` field
records 'vectorized' vs 'scalar_fallback' + the exact PipelineError op so the e2e number is honest.
"""
from __future__ import annotations

import argparse, json
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.rvvgen.registry import load_rvv_package
from merlin.llvmlower.impr_features import apply_schedule, normalize, CompositionError
from merlin.llvmlower.pipeline import RVV_TRANSFORM_SCHEDULE

# reuse the frozen e2e measurement helpers
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "k1_e2e_gv", str(Path(__file__).resolve().parent / "k1_e2e_general_validate.py"))
_gv = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_gv)
run_pkg = _gv.run_pkg


CONFIGS = [
    ("baseline", []),
    ("act_alone", ["vectorized_transcendental_activation"]),
    ("matmul_only", ["fused_vfmacc_tiled"]),
    ("act_plus_matmul", ["vectorized_transcendental_activation", "fused_vfmacc_tiled"]),
    ("microkernel_v3", ["accumulator_resident_microkernel_v3"]),
    # --- faithful whole-model beam candidate set (distinct mined ideas + composable combos) ---
    ("vfmacc_contraction", ["fused_vfmacc_contraction"]),
    ("accum_wholemodel", ["accumulator_resident_wholemodel"]),
    ("accum_ntail", ["accumulator_resident_ntail"]),
    ("lmul_widen", ["lmul_widen_n"]),
    ("v3_plus_act", ["accumulator_resident_microkernel_v3", "vectorized_transcendental_activation"]),
    ("v3_plus_lmul", ["accumulator_resident_microkernel_v3", "lmul_widen_n"]),
    ("tiled_plus_lmul", ["fused_vfmacc_tiled", "lmul_widen_n"]),
]


def composition_probe(feats: list[str]) -> dict:
    """Static composition verdict: does apply_schedule accept these features together?"""
    try:
        normalize(feats)
        apply_schedule(RVV_TRANSFORM_SCHEDULE, frozenset(feats))
        return {"composes": True, "composition_error": None}
    except CompositionError as e:
        return {"composes": False, "composition_error": str(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="artifacts/recaptures/bitvla_fp32_consistent,artifacts/recaptures/openvla_fp32_consistent")
    ap.add_argument("--baseline", default="generated_targets/rvv/hand_v0")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--configs", default="baseline,act_alone,matmul_only,act_plus_matmul")
    ap.add_argument("--out", default="artifacts/measurements/k1_spacemit/k1_e2e_activation.json")
    a = ap.parse_args()

    want = set(a.configs.split(","))
    hb = load_rvv_package(a.baseline)
    results = {}

    for model in a.models.split(","):
        md = Path(model)
        golden = np.load(md / "golden.npy")
        print(f"\n=== MODEL {md} (golden {golden.shape}) ===")
        model_rows = []
        for tag, feats in CONFIGS:
            if tag not in want:
                continue
            print(f"--- {tag} (features={feats}) ---")
            probe = composition_probe(feats)
            if not probe["composes"]:
                # honest composition limitation: do NOT run the board, record the verdict.
                print(f"    COMPOSITION BLOCKED: {probe['composition_error'][:140]}")
                model_rows.append({
                    "tag": tag, "compiler_features": feats, "status": "not_run",
                    "blocker": "composition_error: " + probe["composition_error"],
                    "composes": False, "min_wall_ns": None, "fp32_cos": None})
                continue
            pkg = replace(hb, run_id=f"act_e2e_{md.name}_{tag}", compiler_features=list(feats))
            row = run_pkg(md, pkg, golden, a.n, tag, a.timeout)
            row["composes"] = True
            model_rows.append(row)
        # speedups vs baseline
        base = next((r for r in model_rows if r["tag"] == "baseline" and r.get("min_wall_ns")), None)
        for r in model_rows:
            if base and r.get("min_wall_ns"):
                r["speedup_vs_baseline"] = base["min_wall_ns"] / r["min_wall_ns"]
        results[md.name] = {"model": str(md), "golden_shape": list(golden.shape),
                            "n": a.n, "rows": model_rows}

    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote -> {outp}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
