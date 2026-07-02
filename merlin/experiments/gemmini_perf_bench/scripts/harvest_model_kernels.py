#!/usr/bin/env python3
"""Harvest distinctive int8 matmul kernels from real models (model2MLIR) for the perf benchmark.

We do NOT lower whole models. We parse each model's linalg MLIR, extract the matmul ops' SHAPES +
provenance (`prov.module`/`prov.region_id`), pick a small set of *distinctive* shapes that the golden
bareMetalC tests do not already cover (wide-K, narrow-M attention, large dense-M, extreme aspect), pad
to DIM=16, and emit Gemmini-native **int8** capsules (acc_scale -> i8) with deterministic data. Each
capsule is the SAME kernel every approach will run.

The matmul dtype in the captured MLIR is the model's (often f32 with QDQ around it); we standardize to
i8 (Gemmini-native) and keep only the shape + provenance — the int8 path is the meaningful comparison.

Usage:
  harvest_model_kernels.py [--models tiny_llama,smolvla,openvla] [--max-per-model 4] [--total 8]
                           [--golden-shapes <csv of MxKxN already covered>] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

import _pbcommon as PB
from merlin.targetgen import model_slice_export as MSE  # noqa: E402

# linalg.matmul {... prov.module = "X" ... prov.region_id = "Y" ...}
#   ins(%a, %b : tensor<MxKxT>, tensor<KxNxT>) outs(...) -> ...
_MATMUL = re.compile(
    r'linalg\.matmul\s*\{([^}]*)\}\s*ins\([^:]+:\s*'
    r'tensor<(\d+)x(\d+)x[\w.]+>,\s*tensor<(\d+)x(\d+)x[\w.]+>\)')
_PROV = lambda body, key: (re.search(rf'prov\.{key}\s*=\s*"([^"]*)"', body) or [None, None])[1]

ACC_SCALE = 0.0625  # standard Gemmini-native int8 requant (matches the capsule_bench convention)
# Verilator L3 is cycle-accurate but slow: kernels above ~this many MACs (≈ ideal cycles × 256) are
# impractical on verilator, so they run spike-L2 (functional cycles) only with L3 deferred. Small ones
# get the full L2+L3 ladder. Tunable; the runner reads the per-kernel `sim_hint`.
SIM_MAX_MACS_L3 = 2_000_000  # ~8K ideal cycles -> tens of seconds on verilator


def _sim_hint(macs: int) -> str:
    return "L2+L3" if macs <= SIM_MAX_MACS_L3 else "L2_only"


def parse_matmuls(mlir_text: str) -> list[dict]:
    """Return [{M,K,N, module, region}] for every linalg.matmul (K from both operands must agree)."""
    out = []
    for m in _MATMUL.finditer(mlir_text):
        body, M, K1, K2, N = m.group(1), *(int(m.group(i)) for i in range(2, 6))
        if K1 != K2:
            continue  # not a well-formed (M,K)x(K,N) contraction
        out.append({"M": M, "K": K1, "N": N,
                    "module": _PROV(body, "module") or "?",
                    "region": _PROV(body, "region_id") or "?"})
    return out


def _distinctiveness(k: dict) -> tuple:
    """Sort key favoring shapes that stress Gemmini differently: extreme aspect ratios + size."""
    M, K, N = k["M"], k["K"], k["N"]
    aspect = max(M, K, N) / max(1, min(M, K, N))   # how non-square
    return (round(aspect, 1), K, M * K * N)


def select_distinctive(cands: list[dict], golden_shapes: set[tuple], max_per_model: int,
                       total: int) -> list[dict]:
    """Dedup by padded (M,K,N), drop golden-covered shapes, prefer distinctive ones, cap per-model +
    overall. Ensure a MIX of verilator-feasible (<=SIM_MAX_MACS_L3) and large-representative kernels so
    the comparison is actually runnable: reserve ~60% of slots for feasible kernels."""
    seen_shape: set[tuple] = set()
    by_model: dict[str, int] = {}
    feasible: list[dict] = []
    large: list[dict] = []
    for k in sorted(cands, key=_distinctiveness, reverse=True):
        shp = (PB.align(k["M"]), PB.align(k["K"]), PB.align(k["N"]))
        if shp in seen_shape or shp in golden_shapes or min(shp) == 0:
            continue
        seen_shape.add(shp)
        macs = PB.matmul_macs(*shp)
        rec = {**k, "Mp": shp[0], "Kp": shp[1], "Np": shp[2], "macs": macs,
               "sim_hint": _sim_hint(macs)}
        (feasible if macs <= SIM_MAX_MACS_L3 else large).append(rec)

    n_feasible = max(1, int(round(total * 0.6)))
    picked: list[dict] = []

    def take(pool: list[dict], limit: int) -> None:
        for k in pool:
            if len(picked) >= limit:
                break
            if by_model.get(k["model"], 0) >= max_per_model:
                continue
            by_model[k["model"]] = by_model.get(k["model"], 0) + 1
            picked.append(k)

    take(feasible, n_feasible)          # ~60% verilator-feasible kernels
    take(large, total)                  # fill the rest with large-representative kernels
    if len(picked) < total:             # backfill, relaxing the per-model cap
        for k in feasible + large:
            if k not in picked:
                picked.append(k)
            if len(picked) >= total:
                break
    return picked[:total]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny_llama,smolvla,openvla")
    ap.add_argument("--max-per-model", type=int, default=4)
    ap.add_argument("--total", type=int, default=8)
    ap.add_argument("--golden-shapes", default="",
                    help="csv of MxKxN already covered by the golden set (skip these)")
    ap.add_argument("--dry-run", action="store_true", help="print selection, do not emit capsules")
    a = ap.parse_args(argv)

    golden = set()
    for s in filter(None, a.golden_shapes.split(",")):
        try:
            M, K, N = (int(x) for x in s.lower().split("x"))
            golden.add((PB.align(M), PB.align(K), PB.align(N)))
        except ValueError:
            pass

    cands: list[dict] = []
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        f = PB.MODEL2MLIR / model / f"{model}_int8.mlir"
        if not f.is_file():
            print(f"  [skip] no int8 mlir for {model}: {f}", file=sys.stderr)
            continue
        ms = parse_matmuls(f.read_text(errors="ignore"))
        for k in ms:
            k["model"] = model
        cands += ms
        print(f"  {model}: {len(ms)} matmuls "
              f"(shapes e.g. {sorted({(k['M'],k['K'],k['N']) for k in ms})[:4]})")

    picked = select_distinctive(cands, golden, a.max_per_model, a.total)
    print(f"\nselected {len(picked)} distinctive kernels:")
    corpus = []
    for i, k in enumerate(picked):
        kid = f"M{i:02d}_{k['model']}_{k['module']}_{k['Mp']}x{k['Kp']}x{k['Np']}_i8"
        kid = re.sub(r"[^A-Za-z0-9_]", "_", kid)
        print(f"  {kid:54s} (orig {k['M']}x{k['K']}x{k['N']} {k['model']}/{k['region']}) "
              f"macs={k['macs']:,} sim={k['sim_hint']}")
        corpus.append({
            "id": kid, "op": "matmul", "dtype": "i8",
            "M": k["Mp"], "K": k["Kp"], "N": k["Np"],
            "epilogue": ["acc_scale"], "output_dtype": "i8", "acc_scale": ACC_SCALE,
            "source": f"model:{k['model']}/{k['module']}/{k['region']}",
            "orig_shape": f"{k['M']}x{k['K']}x{k['N']}",
            "macs": k["macs"], "sim_hint": k["sim_hint"],
        })

    if a.dry_run:
        print("\n[dry-run] no capsules emitted")
        return 0

    PB.KERNELS.mkdir(parents=True, exist_ok=True)
    for entry in corpus:
        cap = MSE.make_matmul_capsule(
            name=entry["id"], semantic=f"model_matmul_{entry['source']}",
            M=entry["M"], K=entry["K"], N=entry["N"],
            lhs="X", weight="W", out="Y0",
            epilogue=entry["epilogue"], output_dtype=entry["output_dtype"],
            acc_scale=entry["acc_scale"], label="dev",
            source_reference=entry["source"])
        MSE.export_capsule_dir(PB.KERNELS, cap,
                               comment=f"{entry['id']} (harvested {entry['orig_shape']})")
    # merge into kernel_corpus.yaml (model section)
    corpus_path = PB.KERNELS / "kernel_corpus.yaml"
    doc = yaml.safe_load(corpus_path.read_text()) if corpus_path.exists() else {}
    doc = doc or {}
    doc["model_kernels"] = corpus
    corpus_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"\nemitted {len(corpus)} model kernels -> {PB.KERNELS} (+ kernel_corpus.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
