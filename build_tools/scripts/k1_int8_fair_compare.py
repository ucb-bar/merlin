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
import hashlib
import json
import os
import subprocess
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


def _host_conditions() -> dict:
    """Load on the BUILD host, which decides whether a compile-ceiling refusal is reproducible.

    Our arm's wall comes from the board and is insulated from this. The BUILD does not: tiny_llama's
    whole-model clang was killed at ~880s while this shared 48-core host sat at load 46.6, so the
    same command on an idle host may well finish inside the same ceiling. Without this number a
    compile-timeout refusal reads as a property of the model, and a reader cannot tell whether to
    retry or to fix something. Unreadable is UNKNOWN, never a comforting default.
    """
    out: dict = {}
    try:
        out["loadavg_1_5_15"] = list(os.getloadavg())
    except OSError as e:  # noqa: BLE001
        out["loadavg_1_5_15"] = None
        out["loadavg_error"] = f"{type(e).__name__}: {e}"
    try:
        out["cpu_count"] = os.cpu_count()
    except Exception as e:  # noqa: BLE001
        out["cpu_count"] = None
        out["cpu_count_error"] = f"{type(e).__name__}: {e}"
    return out


_MERLIN_PY = repo_root() / "merlin" / "python"


def ours_arm(model_dir: Path, pkg, golden_refs: dict, work_root: Path, *,
             n: int, warmup: int, iters: int, parallel_harts: int | None = None) -> dict:
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
    from merlin.baselines import rvv_audit as _rvv_audit
    walls, gated, blocker = [], 0, None
    rvv_cov = None
    conds = []
    last_tiers = last_tier_ok = last_cos = last_rel = None
    last_gate = None
    for i in range(n):
        conds.append(_conditions())
        try:
            res = k1.run_on_k1(model_dir, work_root / f"ours_{i}", pkg, timeout=1800,
                               op_profile=False, iters=iters, warmup=warmup,
                               parallel_harts=parallel_harts)
            g = zm._gate(res["prefix"], golden_refs)
            # WHAT FRACTION OF THE OUTPUT THIS SCORE COVERS. The board harness prints at most
            # MERLIN_DUMP_CAP elements, and _gate truncates the reference to match, so a prefix
            # score reads exactly like a whole-output one. On tiny_llama that meant grading 4096 of
            # 256000 logits -- 1.6%, and specifically token 0, the one position W8A8 quantization
            # destroys for every implementation including the independent reference (torchao itself
            # scores -0.32 there). It reported cos 0.8925 and looked like a codegen defect; the full
            # tensor is cos 0.9532, slightly BETTER than the reference's own 0.9487.
            if not g.get("comparison_complete", True):
                print(f"  [ours {i}] PARTIAL OUTPUT: gate compared {g.get('n_compared')} of "
                      f"{g.get('n_reference')} elements ({g.get('compared_fraction', 0):.1%}) — "
                      "the cos/rel below are a PREFIX score, NOT the model's accuracy", flush=True)
            cos, rel, ok = g.get("cos"), g.get("rel"), g.get("ok")
            last_gate = {k: v for k, v in g.items() if isinstance(v, (int, float, bool))}
            last_tiers, last_tier_ok = g.get("tiers"), g.get("tier_ok")
            last_cos, last_rel = cos, rel
            if not ok:
                blocker = (f"gate failed: cos={cos} rel={rel} tiers={g.get('tiers')} "
                           f"tier_ok={g.get('tier_ok')}")
                print(f"  [ours {i}] NOT_GATED {blocker}", flush=True)
                continue
            walls.append(res["metrics"]["wall_ns"])
            gated += 1
            # IS THIS EVEN VECTORIZED? A transform failure is caught as a whole-model scalar
            # fallback, which is CORRECT and much slower -- bit-identical numerics with a 5x wall.
            # Without this the two are indistinguishable from the outside, and the regression gets
            # attributed to the lever's tiling choice instead of to the lever not surviving lowering.
            if rvv_cov is None:
                obj = work_root / f"ours_{i}" / "rvv" / "model.o"
                if obj.is_file():
                    try:
                        rep = _rvv_audit.audit_binary(obj)
                        # The MODEL-level verdict belongs to the compute-bearing symbol, not to a
                        # list of zero-vector symbols: `_mlir_ciface_forward` only unpacks memref
                        # descriptors, so it is always vector==0 and is flagged in a build that
                        # vectorized perfectly as well as in one that did not. Reading a fallback
                        # off that list misattributed a devectorization of `forward` (0.399 ->
                        # 0.256) as a whole-model scalar fallback earlier today.
                        _cs = rep.compute_symbol()
                        rvv_cov = {"coverage_overall": rep.coverage_overall,
                                   "vector": rep.vector, "scalar_compute": rep.scalar_compute,
                                   "compute_symbol": (_cs[0] if _cs else None),
                                   "compute_symbol_coverage": (_cs[1].coverage if _cs else None),
                                   "zero_vector_symbols": sorted(
                                       n for n, sym in rep.by_symbol.items()
                                       if sym.is_scalar_fallback)[:12]}
                    except Exception as _e:      # never let a diagnostic break a measurement
                        rvv_cov = {"error": f"{type(_e).__name__}: {_e}"}
            print(f"  [ours {i}] wall_ns={res['metrics']['wall_ns']} cos={cos:.7f} rel={rel} "
                  f"tier_ok={g.get('tier_ok')} tiers={g.get('tiers')}", flush=True)
        except Exception as e:  # noqa: BLE001
            blocker = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"  [ours {i}] BLOCKED — {blocker}", flush=True)
            break
    conds.append(_conditions())
    return {"ok": gated > 0, "n_gated": gated, "walls": walls,
            "min_wall_ns": min(walls) if walls else None,
            # The FULL gate record, not the collapsed pair. `_gate` sets out["rel"] to the W8A8
            # tier's score when that tier is present, so keeping only ("cos","rel") silently drops
            # fp32_rel -- the one number comparable with a reference that scored against fp32. That
            # omission produced a false "ExecuTorch is more accurate" from this very artifact.
            "gate": dict(last_gate or {},
                         tiers=last_tiers, tier_ok=last_tier_ok, cos=last_cos, rel=last_rel),
            # WHICH reference each of our tiers was scored against, so a consumer can tell whether
            # it may be compared with the other side's number at all.
            "accuracy_reference_by_tier": {"fp32": "capture_golden_fp32",
                                           "w8a8": "capture_golden_w8a8"},
            "protocol": {"warmup": warmup, "iters": iters, "launches": n, "pick": "min-of-n"},
            "board_conditions": conds, "blocker": blocker,
            # Vector coverage of the code that produced these walls. A wall without it cannot
            # distinguish "this lever is slow" from "this lever did not survive lowering".
            "rvv": rvv_cov}


def et_arm(model: str, *, qd8: bool, n_lo: int, n_hi: int,
           cpu_threads: int | None = None) -> dict:
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
                             cpu_threads=cpu_threads,
                             num_executions=n, run_board=True, write=True)
            per_inf = getattr(r, "e2e_wall_ns", None)
            rec = {"n": n, "status": r.status(), "per_inference_ns": per_inf,
                   "total_ns": (per_inf * n) if per_inf else None,
                   "cos": r.cos, "rel": r.rel,
                   # ExecuTorch's ahead-of-time work lands HERE, not in the execute line every ratio
                   # in this repo is taken against: the XNNPACK delegate prepacks weights into its
                   # blocked layout at delegate init. Whether our offline weight-transpose hoisting
                   # is the honest mirror of that, or an advantage we granted ourselves, is decided
                   # by this number -- which is why it is no longer dropped.
                   "load_ns": getattr(r, "load_ns", None),
                   # int8 forces compute_golden (executorch.py:221-224), so the score is against an
                   # fp32 reference RECOMPUTED from the loaded model -- not our captured golden and
                   # not a host int8 reference.
                   "accuracy_reference": "recomputed_fp32",
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


#: What OUR arm computes, always. Not a function of the reference.
OURS_QUANT_RECIPE = "merlin_int8_w8a8"

#: The tier of ours that is comparable IN KIND with a reference scored against fp32. Our W8A8 tier
#: answers a different (tighter) question and must not be ranked against an fp32-scored number.
OURS_ACCURACY_REFERENCE = "capture_golden_fp32"


def verdict(ours: dict, arm: dict, ours_bundle: str) -> dict:
    """The ratio, or a concrete refusal. Never a number whose basis cannot be shown."""
    from merlin.compare.executorch_column import (accuracy_reference_mismatch_reason,
                                                  bundle_mismatch_reason, layout_equivalence,
                                                  quant_recipe_mismatch_reason)
    ours_w = ours.get("min_wall_ns")
    et_w = arm.get("warm_ns")
    if not ours_w or not et_w:
        return {"status": "not_measured",
                "reason": arm.get("refusal") or ours.get("blocker") or "one arm produced no wall"}
    ref_bundle = next((r.get("bundle_id", "") for r in arm["runs"] if r.get("bundle_id")), "")
    ref_recipe = next((r.get("quant_recipe", "") for r in arm["runs"] if r.get("quant_recipe")), "")
    # Ours is the merlin int8 datapath by construction (the package is int8) and its recipe name is
    # a CONSTANT -- never derived from what the reference ran. Deriving it from the comparand is how
    # a guard ends up comparing the reference against itself: the qd8 arm passed that way once, and
    # the mismatch check could not have fired on any input. Whether merlin_int8_w8a8 is comparable
    # to pt2e_qd8 is a declared equivalence in QUANT_RECIPE_EQUIVALENT, justified there.
    # Both guards refuse UNKNOWN as firmly as a mismatch.
    for why in (bundle_mismatch_reason(ours_bundle, ref_bundle),
                quant_recipe_mismatch_reason(OURS_QUANT_RECIPE, ref_recipe)):
        if why:
            return {"status": "not_comparable", "reason": why}
    # A speed number without its accuracy is not a result on a quantized datapath -- but an
    # accuracy number scored against a different reference is not a comparison either. Report the
    # pair, or refuse it, on the same fail-closed terms as the ratio itself.
    ref_acc = next((r.get("accuracy_reference", "") for r in arm["runs"]
                    if r.get("accuracy_reference")), "")
    why_acc = accuracy_reference_mismatch_reason(OURS_ACCURACY_REFERENCE, ref_acc)
    if why_acc:
        accuracy = {"status": "not_comparable", "reason": why_acc,
                    "ours": {"reference": OURS_ACCURACY_REFERENCE,
                             "cos": ours.get("gate", {}).get("fp32_cos"),
                             "rel": ours.get("gate", {}).get("fp32_rel")},
                    "executorch": {"reference": ref_acc,
                                   "cos": next((r.get("cos") for r in arm["runs"] if r.get("cos")), None),
                                   "rel": next((r.get("rel") for r in arm["runs"] if r.get("rel")), None)}}
    else:
        accuracy = {"status": "measured", "reference": ref_acc,
                    "ours_cos": ours.get("gate", {}).get("fp32_cos"),
                    "ours_rel": ours.get("gate", {}).get("fp32_rel"),
                    "executorch_cos": next((r.get("cos") for r in arm["runs"] if r.get("cos")), None),
                    "executorch_rel": next((r.get("rel") for r in arm["runs"] if r.get("rel")), None)}
    # ExecuTorch's weight prepacking happens at delegate init, OUTSIDE the execute line this ratio
    # divides. Surfaced beside the ratio so the reader can see what each side did not pay for.
    load = next((r.get("load_ns") for r in arm["runs"] if r.get("load_ns")), None)
    # If the bundle check passed only because the two are a DECLARED layout-only pair, say so in the
    # artifact -- the ratio is legitimate only while both sides do their weight layout once outside
    # the timed window (theirs at delegate init, ours at build time), and a reader must be able to
    # see that assumption rather than infer it.
    _layout_eq = layout_equivalence(ours_bundle, ref_bundle) if ours_bundle != ref_bundle else None
    return {"status": "measured", "ours_ns": ours_w, "executorch_warm_ns": et_w,
            "bundle_layout_equivalence": _layout_eq,
            "ours_over_executorch": ours_w / et_w,
            "speedup_vs_executorch": et_w / ours_w,
            "beats_executorch": et_w > ours_w,
            "executorch_load_ns": load,
            "accuracy": accuracy}


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
    ap.add_argument("--ref-cpu-threads", type=int, default=None,
                    help="cores the REFERENCE arm may use (its --cpu_threads). Left unset its "
                         "threadpool takes every online CPU -- 8 on this board -- and the count is "
                         "NOT readable from the build flags, so an unpinned ratio silently compares "
                         "our core count against theirs. Pass 1 alongside a 1-core ours arm for a "
                         "per-core comparison, or match it to --parallel-harts.")
    ap.add_argument("--parallel-harts", type=int, default=None,
                    help="cores OUR arm may use (default: 1, single-threaded). The board has 8, and "
                         "the reference is built with pthreadpool ON and a shared XNNPACK "
                         "workspace, so a default run compares our ONE core against however many "
                         "ExecuTorch takes. Recorded on every cell either way, so no ratio is "
                         "readable without knowing the core counts it was taken at.")
    ap.add_argument("--compile-timeout-s", type=int, default=3600,
                    help="ceiling on any single build command for OUR arm. The module default is "
                         "900s, which is a KERNEL budget: a whole-model int8 clang invocation "
                         "exceeds it (tiny_llama's was killed at ~880s), and the cell then reports "
                         "BLOCKED, indistinguishable from a real codegen defect. Sized to the cell "
                         "here so the ceiling is a stated parameter of the run, not an ambient "
                         "constant nobody passed.")
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

    # Key the build tree by the FEATURE SET as well as the bundle. One directory per bundle means
    # the next run overwrites the emitted object of the last one, and the emitted object is the only
    # thing that can explain a result: a 5.1x regression whose numerics were bit-identical -- the
    # signature of a silent scalar fallback -- became un-diagnosable because the control run
    # launched to attribute it had already replaced the binary that produced it.
    _feat_key = hashlib.sha256(",".join(sorted(feats or ())).encode()).hexdigest()[:10] if feats \
        else "nofeatures"
    work = Path(repo_root()) / "out" / "build" / "fair_compare" / md.name / _feat_key
    work.mkdir(parents=True, exist_ok=True)
    (work / "FEATURES.txt").write_text("\n".join(sorted(feats or ())) + "\n", encoding="utf-8")
    print(f"[work] {work}", flush=True)

    # WHAT SOURCE PRODUCED THIS. The commit says which revision is checked out; on a tree several
    # sessions are editing at once that is not what the compiler READ. A measurement here built from
    # another session's uncommitted edit and the result was attributed to the feature under test:
    # a run started at 15:42:18 picked up a quantization fix committed at 15:43:24, so a wall and an
    # accuracy improvement produced by that fix were both credited to the flag being measured.
    # Feature-keyed build dirs key on features and codegen_env keys on the environment; neither sees
    # source state. `source_digest` is over the bytes actually read, so a dirty tree is identifiable
    # instead of looking pinned.
    # Bound at import from MERLIN_COMPILE_TIMEOUT_S, so it has to be re-bound after parsing.
    k1._K1_CMD_TIMEOUT_S = int(a.compile_timeout_s)
    print(f"[budget] compile ceiling {a.compile_timeout_s}s per build command", flush=True)

    from merlin.common import provenance as _prov
    _src = [_MERLIN_PY / "merlin" / "llvmlower" / "passes_quant_int.py",
            _MERLIN_PY / "merlin" / "llvmlower" / "impr_features.py",
            _MERLIN_PY / "merlin" / "llvmlower" / "quant_passes.py",
            _MERLIN_PY / "merlin" / "runtime" / "backends" / "zephyr_model.py",
            _MERLIN_PY / "merlin" / "mining" / "k1.py"]
    _src = [q for q in _src if q.is_file()]
    try:
        _src_digest = _prov.source_digest(_src)
    except Exception as _e:                      # a provenance stamp must not break a measurement
        _src_digest = f"UNKNOWN:{type(_e).__name__}"
    _dirty = sorted(q.name for q in _src
                    if subprocess.run(["git", "status", "--porcelain", "--", str(q)],
                                      cwd=repo_root(), capture_output=True,
                                      text=True).stdout.strip())
    print(f"[source] digest={_src_digest} dirty={_dirty or 'none'}", flush=True)
    _host = _host_conditions()
    print(f"[host] loadavg={_host.get('loadavg_1_5_15')} cpus={_host.get('cpu_count')}", flush=True)
    rec: dict = {"model": a.model, "ours_bundle": md.name, "package": Path(a.baseline).name,
                 "golden_tiers": sorted(refs), "started": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                 # The compiler sources as READ, plus which of them were uncommitted at the time.
                 # A non-empty `dirty` means this number cannot be reproduced from the commit alone.
                 "source_digest": _src_digest, "source_dirty": _dirty,
                 # Load on the BUILD host at the start of the cell. The board wall is insulated from
                 # this; the compile ceiling is not, so a compile-timeout refusal has to be readable
                 # as "the host was saturated" vs "this model cannot be built in that budget".
                 "host": _host_conditions(),
                 "compile_timeout_s": int(a.compile_timeout_s),
                 # Cores each side was allowed. Ours defaults to 1; the reference's runner links
                 # pthreadpool and is built with a shared XNNPACK workspace, so its count is
                 # whatever its threadpool chooses and is NOT controlled here. A ratio taken across
                 # different core counts is a system comparison, not a compiler one, and the row has
                 # to carry enough for a reader to tell which it is looking at.
                 "cores": {"ours": int(a.parallel_harts) if a.parallel_harts else 1,
                           "reference": (int(a.ref_cpu_threads) if a.ref_cpu_threads
                                         else "UNKNOWN (pthreadpool default = every online CPU, "
                                              "8 on this board; not pinned by this run)")}}

    # INTERLEAVED, ours first, so a board that drifts during the session moves both arms rather than
    # landing the drift entirely on one of them.
    print("== ours ==", flush=True)
    rec["ours"] = ours_arm(md, pkg, refs, work, n=a.n, warmup=a.warmup, iters=a.iters,
                           parallel_harts=a.parallel_harts)
    print("== executorch qd8 (the matching arithmetic) ==", flush=True)
    rec["executorch_qd8"] = et_arm(a.model, cpu_threads=a.ref_cpu_threads, qd8=True, n_lo=a.et_n_lo, n_hi=a.et_n_hi)
    if a.also_weight_only:
        print("== executorch weight-only (LABELLED: not int8 compute) ==", flush=True)
        rec["executorch_weight_only"] = et_arm(a.model, cpu_threads=a.ref_cpu_threads, qd8=False, n_lo=a.et_n_lo, n_hi=a.et_n_hi)
    print("== ours (second pass, brackets the ET arms) ==", flush=True)
    rec["ours_after"] = ours_arm(md, pkg, refs, work, n=a.n, warmup=a.warmup, iters=a.iters,
                                 parallel_harts=a.parallel_harts,
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
