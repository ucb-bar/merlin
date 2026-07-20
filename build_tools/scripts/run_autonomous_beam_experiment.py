#!/usr/bin/env python
"""Fully-autonomous beam experiment: rediscover the whole-model optimizations WITHOUT hand-feeding.

For each (model, dtype) cell it runs the CCA beam from the FROZEN hand_v0 seed with the whole-model
proposer (byte-traffic-ranked levers) and the whole-model objective (rank on the model's own K1
wall), depth>=2 so it can discover a STACK of levers, not just the best single one. It then writes an
autonomous comparison: what the beam DISCOVERED (which levers, in which order, final speedup +
attainment vs the dtype-appropriate reference) against the MANUAL ours_best (the clean four-way), the
XNNPACK/OpenBLAS GEMM microkernels routed INSIDE OUR runtime, and the TRUE external ExecuTorch(+XNNPACK)
whole-model system. Everything is measured on the board and correctness-gated; a cell that fails is
recorded as such, never faked.

Two distinct reference arms are reported, never conflated (see merlin.compare.executorch_column):
  * ``xnnpack_kernels`` — XNNPACK's GEMM microkernel routed inside OUR runtime (four-way; fp32 only).
  * ``executorch``      — the TRUE external ExecuTorch+XNNPACK system (its own whole-model runtime),
                          ingested board-free from the latest PASSING BaselineResult (never re-run here).
The int8/fp16 reference bug is fixed: a non-fp32 fork wall is NEVER divided by the fp32 XNNPACK wall —
the reference for a non-fp32 cell is the ExecuTorch wall of the SAME variant, else ``None``.

Run (board-serialized, long): MERLIN_COMPILE_TIMEOUT_S=3600 .venv/bin/python
  build_tools/scripts/run_autonomous_beam_experiment.py --cells fp32:bitvla,fp32:openvla,fp32:rdt2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "merlin/python")
from merlin.common.paths import artifacts_dir, repo_root  # noqa: E402
from merlin.compare.executorch_column import (  # noqa: E402
    EXECUTORCH_LABEL, XNNPACK_KERNELS_LABEL, executorch_cell)

# expert objdump fixture per dtype (the CCA target); f16 carries the native-accumulate caveat.
_EXPERT_OBJDUMP = {
    "fp32": "merlin/tests/data/cca_asm/xnnpack_f32_gemm_rvv.objdump",
    "int8": "merlin/tests/data/cca_asm/xnnpack_qd8_gemm_rvv.objdump",
    "fp16": "merlin/tests/data/cca_asm/xnnpack_f16_gemm_rvv.objdump",
}
_BUNDLE = {  # (dtype, model) -> recapture bundle
    ("fp32", "bitvla"): "out/artifacts/recaptures/bitvla_fp32_consistent",
    ("fp32", "openvla"): "out/artifacts/recaptures/openvla_fp32_consistent",
    ("fp32", "rdt2"): "out/artifacts/recaptures/rdt2_fp32_consistent",
    ("int8", "bitvla"): "out/artifacts/recaptures/bitvla_int8_consistent",
    ("int8", "openvla"): "out/artifacts/recaptures/openvla_int8_consistent",
    ("int8", "rdt2"): "out/artifacts/recaptures/rdt2_int8_consistent",
}


def _xnnpack_kernels_wall_ns(model: str) -> float | None:
    """XNNPACK-GEMM-in-OUR-runtime whole-model wall (ns) from the clean four-way cache, else the older
    cache. This is the ``xnnpack_kernels`` arm — XNNPACK's matmul microkernel routed inside our own
    runtime, NOT a standalone external system (see ``XNNPACK_KERNELS_LABEL``). The four-way is fp32,
    so this number is ONLY comparable to an fp32 fork wall."""
    for name in (f"k1_4way_clean_{model}.json", f"k1_4way_{model}.json"):
        p = Path("out/artifacts/kernel-mining/rvv/bench") / name
        if p.is_file():
            d = json.loads(p.read_text())
            w = (d.get("xnnpack_kernels", {}) or {}).get("min_wall_ns")
            if w:
                return float(w)
    return None


def _reference(model: str, dtype: str, root: Path) -> dict:
    """The dtype-APPROPRIATE expert reference wall (ns) + provenance for attainment.

    fp32: the four-way XNNPACK-GEMM-in-our-runtime wall (the historical reference).
    non-fp32: there is NO fp32-comparable four-way number, and dividing an int8/fp16 fork wall by the
      fp32 XNNPACK wall is meaningless — so the reference is the TRUE external ExecuTorch wall of the
      SAME variant if a passing one exists, else ``None`` (NEVER the fp32 XNNPACK number).
    """
    if dtype == "fp32":
        return {"wall_ns": _xnnpack_kernels_wall_ns(model), "kind": "xnnpack_kernels_in_runtime",
                "dtype": "fp32", "label": XNNPACK_KERNELS_LABEL}
    et = executorch_cell(model, dtype, root=root)
    if et.get("executorch_status") == "measured":
        return {"wall_ns": et["executorch_wall_ns"], "kind": "executorch_external", "dtype": dtype,
                "label": EXECUTORCH_LABEL,
                "note": (f"non-fp32 cell: reference = TRUE external ExecuTorch {dtype} wall "
                         "(int8-vs-int8 / fp16-vs-fp16); the four-way XNNPACK wall is fp32 and MUST "
                         "NOT be used here")}
    return {"wall_ns": None, "kind": None, "dtype": dtype,
            "note": (f"non-fp32 cell: no dtype-appropriate reference (attain_vs_xnn=None) — NEVER "
                     f"compared to the fp32 XNNPACK wall; {et.get('reason')}")}


def _manual_best(model: str) -> dict:
    """The MANUAL ours_best (clean four-way) for comparison against the beam's autonomous result."""
    p = Path("out/artifacts/kernel-mining/rvv/bench") / f"k1_4way_clean_{model}.json"
    if not p.is_file():
        return {}
    d = json.loads(p.read_text())
    ob = d.get("ours_best", {}) or {}
    xnn = (d.get("xnnpack_kernels", {}) or {}).get("min_wall_ns")
    return {"features": d.get("ours_best_features"), "wall_ns": ob.get("min_wall_ns"),
            "vs_xnn": (xnn / ob["min_wall_ns"]) if (xnn and ob.get("min_wall_ns")) else None}


def run_cell(dtype: str, model: str, *, width: int, depth: int, top_k: int) -> dict:
    from merlin.rvvgen.beam_cli import run_instrumented_beam
    from merlin.rvvgen.wholemodel_proposer import propose_wholemodel_levers
    bundle = _BUNDLE.get((dtype, model))
    obj = _EXPERT_OBJDUMP.get(dtype)
    root = repo_root()
    # The TRUE external ExecuTorch(+XNNPACK) whole-model column (board-free ingest; never re-run ET).
    et = executorch_cell(model, dtype, root=root)
    # dtype-appropriate expert reference (fp32 XNNPACK-in-runtime, else ExecuTorch same-variant, else None).
    ref = _reference(model, dtype, root)
    comparability = {"xnnpack_kernels": XNNPACK_KERNELS_LABEL, "executorch": EXECUTORCH_LABEL,
                     "reference_used": ref.get("kind")}
    if not bundle or not Path(bundle).is_dir():
        return {"cell": f"{dtype}:{model}", "status": "not_run", "blocker": f"no bundle {bundle}",
                "executorch": et, "reference": ref, "comparability": comparability}
    xnn = ref["wall_ns"]
    t0 = time.time()
    try:
        res = run_instrumented_beam(
            seed_pkg="out/artifacts/targets/rvv/hand_v0", model_dir=bundle,
            expert_objdump=obj, op="matmul", dtype=dtype, targets=("k1",),
            width=width, depth=depth, top_k=top_k, expert_wall_ns=xnn,
            proposer=propose_wholemodel_levers)
    except Exception as e:  # noqa: BLE001
        return {"cell": f"{dtype}:{model}", "status": "error", "blocker": f"{type(e).__name__}: {e}",
                "executorch": et, "reference": ref, "comparability": comparability}
    best = res.get("best") or {}
    # the stack the beam discovered = the winner's compiler_features (read from its package).
    feats = None
    pdir = best.get("package_dir")
    if pdir and (Path(pdir) / "knobs.yaml").is_file():
        import yaml
        feats = (yaml.safe_load((Path(pdir) / "knobs.yaml").read_text()) or {}).get("compiler_features")
    attain = best.get("attainment_vs_expert")
    return {
        "cell": f"{dtype}:{model}", "status": "ok", "seconds": round(time.time() - t0),
        "beam_best_run_id": best.get("run_id"),
        "beam_discovered_features": feats,
        "beam_speedup_vs_baseline": best.get("speedup"),
        # attainment vs the dtype-appropriate reference (fp32: xnnpack-in-runtime; non-fp32: ET same-variant).
        "beam_attainment_vs_reference": attain,
        # back-compat: only meaningful for fp32 (xnnpack-in-runtime). None for non-fp32 — NEVER an
        # int8/fp16 fork wall over an fp32 XNNPACK wall (that bug is fixed via _reference).
        "beam_attainment_vs_xnnpack": attain if dtype == "fp32" else None,
        "beam_gate_ok": best.get("gate_ok"),
        "n_forks": len(res.get("nodes", [])),
        "manual_best": _manual_best(model),
        # fp32 four-way xnnpack-in-runtime wall (the reference for fp32 cells only); None otherwise.
        "xnnpack_wall_ns": _xnnpack_kernels_wall_ns(model) if dtype == "fp32" else None,
        "reference": ref,
        "executorch": et,
        "comparability": comparability,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="fp32:bitvla,fp32:openvla,fp32:rdt2",
                    help="comma list of <dtype>:<model> (dtype in fp32|int8|fp16)")
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    out_path = Path(a.out) if a.out else (artifacts_dir() / "kernel-mining" / "rvv" / "bench"
                                          / "autonomous_beam_experiment.json")
    results = []
    for c in cells:
        dtype, model = c.split(":")
        print(f"\n######## BEAM CELL {c} (width={a.width} depth={a.depth}) ########", flush=True)
        r = run_cell(dtype, model, width=a.width, depth=a.depth, top_k=a.top_k)
        results.append(r)
        et = r.get("executorch", {}) or {}
        et_str = (f"{et.get('executorch_wall_ns')/1e6:.0f}ms" if et.get("executorch_wall_ns")
                  else f"not_measured({et.get('reason','')[:60]})")
        print(f"[{c}] {r.get('status')} :: beam={r.get('beam_discovered_features')} "
              f"spd={r.get('beam_speedup_vs_baseline')} "
              f"attain_vs_ref={r.get('beam_attainment_vs_reference')} "
              f"(ref={r.get('reference',{}).get('kind')}) "
              f"| manual={r.get('manual_best',{}).get('vs_xnn')} "
              f"| executorch={et_str}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"cells": results}, indent=2))
    print(f"\n==== autonomous beam experiment done -> {out_path} ====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
