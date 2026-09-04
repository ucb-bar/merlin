#!/usr/bin/env python3
"""Ours vs ExecuTorch on one int8 model, measured so the ratio actually means something.

Four things have to hold at once for an ours-vs-ExecuTorch int8 number to be citable, and every
published ratio in this repo so far has broken at least one of them:

1. **The same ARITHMETIC.** ``variant="int8"`` defaults to ExecuTorch's weight-only module swap,
   whose dequant const-folds into an fp32 const weight that XNNPACK partitions as a normal fp32
   GEMM — so that cell measures fp32 compute with int8 *storage* and never reaches an int8 ukernel.
   Merlin's int8 dynamically quantizes each activation to i8 (symmetric, per output row) against
   per-channel weight scales, i.e. qd8. This script drives ExecuTorch with ``qd8=True`` so both
   sides run the same scheme, and records ``quant_recipe`` on the result either way.

2. **The same PROTOCOL.** ExecuTorch's runner has no warmup and averages its cold first execution
   into ``--num_executions``; merlin's certify path is min-of-n after warmup. On small_llama int8
   ET's cold inference is 1.62x its warm one. So ET is measured at TWO Ns and the warm cost is the
   slope, ``total(N) = cold + (N-1)*warm`` — not a single cold shot.

3. **The same BUNDLE.** ``bundle.resolve()`` prefers ``<model>_<variant>_full`` over the truncated
   ``_consistent``, so two runs of the "same" cell can be two different models.

4. **The same CONDITIONS.** Two beam runs of a byte-identical frozen seed measured 1.9915x apart,
   and nothing in either artifact could show why. So the arms are INTERLEAVED in one session and the
   board conditions are read around every one of them: a drifting board then moves both arms
   together instead of silently landing on one.

Output is a JSON record carrying both walls, both recipes, the bundle, the protocol, and every
conditions probe. It never computes a ratio it cannot justify — a missing field yields a refusal
string, not a number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.baselines import executorch as et
from merlin.common.paths import repo_root
from merlin.mining import k1
from merlin.mining.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm


def _conditions() -> dict | None:
    """A board-conditions probe that never aborts the campaign. Unreadable is None (UNKNOWN), which
    downstream refuses to compare against — it is never silently treated as 'same as before'."""
    try:
        return k1.board_conditions()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def ours_arm(model_dir: Path, pkg, golden_refs: dict, work_root: Path, *,
             n: int, warmup: int, iters: int) -> dict:
    """Our side: n gated launches, each min-of-`iters` after `warmup` untimed.

    The verdict is ``zephyr_model._gate``'s OWN ``ok``, not a threshold re-implemented here. That
    distinction is the whole of P0.5: `_gate` is two-tier and W8A8-aware (T1 vs a W8A8 reference at
    cos>0.999 / rel<1e-2 / per-element; T2 vs the fp32 golden at cos>0.99 + argmax), while the flat
    `cos>0.9999 AND rel<1e-2` that `k1_op_profile` applies is an fp32-strict bar. On the very config
    this script exists to measure they disagree on identical numbers -- cos=0.9999078512191772,
    rel=0.014796391526079668 -- and the flat bar rejects a run the principled gate accepts. Copying
    the flat bar here is exactly the bug this measured on its first launch.

    ``golden_refs`` is a ``{tier: array}`` dict and the TIER KEYS MATTER: passing a W8A8 golden under
    the "fp32" key grades int8 output by fp32 rules and, worse, hides that the w8a8 tier was never in
    play. `_gate` reports `tiers` / `tier_ok` for exactly that reason.
    """
    walls, gated, blocker = [], 0, None
    conds = []
    last_tiers = last_tier_ok = last_cos = last_rel = None
    for i in range(n):
        conds.append(_conditions())
        try:
            res = k1.run_on_k1(model_dir, work_root / f"ours_{i}", pkg, timeout=1800,
                               op_profile=False, iters=iters, warmup=warmup)
            g = zm._gate(res["prefix"], golden_refs)
            cos, rel, ok = g.get("cos"), g.get("rel"), g.get("ok")
            last_tiers, last_tier_ok = g.get("tiers"), g.get("tier_ok")
            last_cos, last_rel = cos, rel
            if not ok:
                blocker = (f"gate failed: cos={cos} rel={rel} tiers={g.get('tiers')} "
                           f"tier_ok={g.get('tier_ok')}")
                print(f"  [ours {i}] NOT_GATED {blocker}", flush=True)
                continue
            walls.append(res["metrics"]["wall_ns"])
            gated += 1
            print(f"  [ours {i}] wall_ns={res['metrics']['wall_ns']} cos={cos:.7f} rel={rel} "
                  f"tier_ok={g.get('tier_ok')} tiers={g.get('tiers')}", flush=True)
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [ours {i}] BLOCKED — {blocker}", flush=True)
            break
    conds.append(_conditions())
    return {"ok": gated > 0, "n_gated": gated, "walls": walls,
            "min_wall_ns": min(walls) if walls else None,
            "gate": {"tiers": last_tiers, "tier_ok": last_tier_ok, "cos": last_cos, "rel": last_rel},
            "protocol": {"warmup": warmup, "iters": iters, "launches": n, "pick": "min-of-n"},
            "board_conditions": conds, "blocker": blocker}


def et_arm(model: str, *, qd8: bool, n_lo: int, n_hi: int) -> dict:
    """ExecuTorch at two Ns so the WARM cost is a slope, not a cold shot.

    ``BaselineResult.e2e_wall_ns`` is already normalised per-inference by the producer, so the totals
    are recovered by multiplying back by N before differencing.
    """
    recipe = "pt2e_qd8" if qd8 else "weight_only"
    out: dict = {"recipe_requested": recipe, "n_lo": n_lo, "n_hi": n_hi, "runs": []}
    totals: dict[int, float] = {}
    for n in (n_lo, n_hi):
        cond_before = _conditions()
        try:
            r = et.run_model(model, "int8", qd8=qd8, int8_whole_model=(not qd8) or None,
                             num_executions=n, run_board=True, write=True)
            per_inf = getattr(r, "e2e_wall_ns", None)
            rec = {"n": n, "status": r.status(), "per_inference_ns": per_inf,
                   "total_ns": (per_inf * n) if per_inf else None,
                   "cos": r.cos, "rel": r.rel,
                   "quant_recipe": getattr(r, "quant_recipe", ""),
                   "bundle_id": getattr(r, "bundle_id", ""),
                   "gap_reason": r.gap_reason,
                   "board_conditions": {"before": cond_before, "after": _conditions()}}
            if per_inf:
                totals[n] = per_inf * n
        except Exception as e:  # noqa: BLE001
            rec = {"n": n, "status": "error", "error": f"{type(e).__name__}: {str(e)[:400]}",
                   "board_conditions": {"before": cond_before, "after": _conditions()}}
        out["runs"].append(rec)
        print(f"  [et {recipe} N={n}] {json.dumps({k: v for k, v in rec.items() if k != 'board_conditions'})}",
              flush=True)
    if n_lo in totals and n_hi in totals and n_hi > n_lo:
        warm = (totals[n_hi] - totals[n_lo]) / (n_hi - n_lo)
        out["warm_ns"] = warm
        out["cold_ns"] = totals[n_lo] - (n_lo - 1) * warm
        out["cold_over_warm"] = (out["cold_ns"] / warm) if warm else None
    else:
        out["warm_ns"] = None
        out["refusal"] = ("cannot extract a warm slope: need a passing wall at BOTH N values; a "
                          "single-N number is mostly ExecuTorch's cold execution and is not a "
                          "comparand for a warm loop")
    return out


def verdict(ours: dict, arm: dict, ours_bundle: str) -> dict:
    """The ratio, or a concrete refusal. Never a number whose basis cannot be shown."""
    from merlin.compare.executorch_column import (bundle_mismatch_reason,
                                                  quant_recipe_mismatch_reason)
    ours_w = ours.get("min_wall_ns")
    et_w = arm.get("warm_ns")
    if not ours_w or not et_w:
        return {"status": "not_measured",
                "reason": arm.get("refusal") or ours.get("blocker") or "one arm produced no wall"}
    ref_bundle = next((r.get("bundle_id", "") for r in arm["runs"] if r.get("bundle_id")), "")
    ref_recipe = next((r.get("quant_recipe", "") for r in arm["runs"] if r.get("quant_recipe")), "")
    # Ours is the merlin int8 datapath by construction (the package is int8); ET's is whatever it
    # recorded. Both guards refuse UNKNOWN as firmly as a mismatch.
    for why in (bundle_mismatch_reason(ours_bundle, ref_bundle),
                quant_recipe_mismatch_reason("pt2e_qd8" if arm["recipe_requested"] == "pt2e_qd8"
                                             else "merlin_int8_w8a8", ref_recipe)):
        if why:
            return {"status": "not_comparable", "reason": why}
    return {"status": "measured", "ours_ns": ours_w, "executorch_warm_ns": et_w,
            "ours_over_executorch": ours_w / et_w,
            "speedup_vs_executorch": et_w / ours_w,
            "beats_executorch": et_w > ours_w}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name for the ExecuTorch arm (e.g. small_llama)")
    ap.add_argument("--model-dir", required=True, help="our capture bundle (out/artifacts/recaptures/...)")
    ap.add_argument("--baseline", required=True, help="our rvv package (out/artifacts/targets/rvv/...)")
    ap.add_argument("--features", default=None,
                    help="comma-separated compiler features for OUR arm; omitted = the package's own")
    ap.add_argument("--n", type=int, default=3, help="gated launches for our arm")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--et-n-lo", type=int, default=1)
    ap.add_argument("--et-n-hi", type=int, default=6)
    ap.add_argument("--also-weight-only", action="store_true",
                    help="additionally measure ExecuTorch's weight-only recipe as a LABELLED second "
                         "column (it is not int8 compute; kept because it is the historical cell)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    md = Path(a.model_dir)
    base = load_rvv_package(a.baseline)
    # A package's own compiler_features ARE part of what it is, so an OMITTED --features means "what
    # this package was certified with" and an explicitly empty one means the frozen baseline. Same
    # rule as k1_op_profile; `replace` is how the features actually reach the build (there is no
    # with_features on the package, and silently dropping them would measure the wrong lowering).
    feats = ([f.strip() for f in a.features.split(",") if f.strip()] if a.features is not None
             else list(base.compiler_features or []))
    ftag = "base" if not feats else "_".join(sorted(feats))[:40]
    pkg = replace(base, run_id=f"tr_fair_{ftag}", compiler_features=feats)
    print(f"[features] {feats or '<baseline>'} "
          f"({'from the package' if a.features is None else 'from --features'})", flush=True)
    # Same golden rule as k1_op_profile: a W8A8 build graded against the weight-only golden fails
    # cos for a CORRECT build, which reads as a regression rather than as the wrong reference.
    # TIER-KEYED references. `_gate` decides which tier carries the verdict from these KEYS, so a
    # W8A8 golden handed over under "fp32" both grades int8 output by fp32 rules and hides that the
    # w8a8 tier was never in play -- the failure that cost a multi-hour hunt for a TinyLlama int8
    # "board bug" that did not exist. Pass both when both exist so T1 (w8a8) and T2 (fp32+argmax)
    # are live and the derived quantization-excess veto is measurable.
    refs: dict = {}
    if (md / "golden.npy").is_file():
        refs["fp32"] = np.load(md / "golden.npy")
    if pkg.is_int8:
        w = md / "golden_w8a8.npy"
        if not w.is_file():
            raise SystemExit(f"{a.baseline} is int8 but {md.name} ships no golden_w8a8.npy — "
                             "recapture it rather than grading W8A8 output against the weight-only "
                             "golden (a missing w8a8 tier reads as a codegen defect)")
        refs["w8a8"] = np.load(w)
    if not refs:
        raise SystemExit(f"{md} ships no golden to gate against")
    print(f"[golden] tiers={sorted(refs)}  [bundle] {md.name}", flush=True)

    work = Path(repo_root()) / "out" / "build" / "fair_compare" / md.name
    work.mkdir(parents=True, exist_ok=True)

    rec: dict = {"model": a.model, "ours_bundle": md.name, "package": Path(a.baseline).name,
                 "golden_tiers": sorted(refs), "started": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}

    # INTERLEAVED, ours first, so a board that drifts during the session moves both arms rather than
    # landing the drift entirely on one of them.
    print("== ours ==", flush=True)
    rec["ours"] = ours_arm(md, pkg, refs, work, n=a.n, warmup=a.warmup, iters=a.iters)
    print("== executorch qd8 (the matching arithmetic) ==", flush=True)
    rec["executorch_qd8"] = et_arm(a.model, qd8=True, n_lo=a.et_n_lo, n_hi=a.et_n_hi)
    if a.also_weight_only:
        print("== executorch weight-only (LABELLED: not int8 compute) ==", flush=True)
        rec["executorch_weight_only"] = et_arm(a.model, qd8=False, n_lo=a.et_n_lo, n_hi=a.et_n_hi)
    print("== ours (second pass, brackets the ET arms) ==", flush=True)
    rec["ours_after"] = ours_arm(md, pkg, refs, work, n=a.n, warmup=a.warmup, iters=a.iters,
      )

    rec["verdict_qd8"] = verdict(rec["ours"], rec["executorch_qd8"], md.name)
    if a.also_weight_only:
        rec["verdict_weight_only"] = verdict(rec["ours"], rec["executorch_weight_only"], md.name)
    # Session drift, measured rather than assumed: if the two ours passes disagree by more than the
    # board's own noise band, the ET arm between them sat on a moving board and the ratio is suspect.
    w0, w1 = rec["ours"].get("min_wall_ns"), rec["ours_after"].get("min_wall_ns")
    if w0 and w1:
        drift = max(w0, w1) / min(w0, w1)
        rec["session_drift"] = {"ours_first_ns": w0, "ours_second_ns": w1, "ratio": drift,
                                "within_noise_band": drift <= 1.026,
                                "note": "K1 noise floor >=1.9%, band 2.6%; a larger drift means the "
                                        "arms were not measured under one condition"}
    out = Path(a.out) if a.out else (work / "fair_compare.json")
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items()
                      if k in ("verdict_qd8", "verdict_weight_only", "session_drift")}, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    sys.exit(main())
