#!/usr/bin/env python3
"""Generate the golden-set kernels: canonical Gemmini matmul shapes mirroring the bareMetalC test
coverage (single-tile, multi-tile square, rectangular, K-accumulation, wide-N, tall-M, + the requant
and relu epilogue paths). Each is emitted as a merlin_iface capsule (the SAME kernel every approach
runs) with deterministic data + shared golden, plus a matching bareMetalC C reference name for the
golden approach (a).

These are the verilator-feasible reference shapes; the model harvester fills the gaps with distinctive
real-model shapes. Emitting both writes a unified kernel_corpus.yaml.

Usage: gen_golden_kernels.py [--dry-run]
"""
from __future__ import annotations

import argparse

import yaml

import _pbcommon as PB
from merlin.targetgen import model_slice_export as MSE  # noqa: E402

ACC_SCALE = 0.0625

# (id, M, K, N, epilogue, output_dtype, baremetalc_reference, note)
# All DIM-16 multiples; sizes chosen so verilator L3 is feasible. The bareMetalC golden kernel that
# exercises the same character is named so approach (a) can build the canonical C reference.
GOLDEN = [
    ("G00_single_tile_16x16x16",     16, 16, 16,  [],            "i32", "matmul.c",            "one DIM tile"),
    ("G01_multitile_sq_64x64x64",    64, 64, 64,  [],            "i32", "tiled_matmul_ws.c",   "multi-tile square (classic WS)"),
    ("G02_rect_32x64x16",            32, 64, 16,  [],            "i32", "tiled_matmul_ws.c",   "rectangular tiles"),
    ("G03_kaccum_16x128x16",         16, 128, 16, [],            "i32", "tiled_matmul_ws.c",   "deep-K accumulation (8 K-tiles)"),
    ("G04_wideN_16x16x128",          16, 16, 128, [],            "i32", "tiled_matmul_ws.c",   "wide N"),
    ("G05_tallM_128x16x16",          128, 16, 16, [],            "i32", "tiled_matmul_ws.c",   "tall M"),
    ("G06_acc_scale_i8_64x64x64",    64, 64, 64,  ["acc_scale"], "i8",  "tiled_matmul_ws.c",   "requant acc->i8"),
    ("G07_relu_i8_64x64x64",         64, 64, 64,  ["acc_scale", "relu"], "i8", "tiled_matmul_ws.c", "requant+relu->i8"),
    ("G08_large_sq_128x128x128",     128, 128, 128, [],          "i32", "tiled_matmul_ws.c",   "larger square (util headroom)"),
]


def build_corpus() -> list[dict]:
    out = []
    for kid, M, K, N, epi, odt, ref, note in GOLDEN:
        macs = PB.matmul_macs(M, K, N)
        out.append({
            "id": kid, "op": "matmul", "dtype": "i8", "M": M, "K": K, "N": N,
            "epilogue": epi, "output_dtype": odt,
            **({"acc_scale": ACC_SCALE} if "acc_scale" in epi else {}),
            "source": f"baremetalc:{ref}", "note": note,
            "macs": macs, "sim_hint": "L2+L3" if macs <= 2_000_000 else "L2_only",
        })
    return out


def golden_shapes() -> set[tuple]:
    return {(PB.align(e["M"]), PB.align(e["K"]), PB.align(e["N"])) for e in build_corpus()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    corpus = build_corpus()
    for e in corpus:
        print(f"  {e['id']:30s} {e['M']}x{e['K']}x{e['N']} {e['output_dtype']:3s} "
              f"epi={e['epilogue']} macs={e['macs']:,} sim={e['sim_hint']}  [{e['source']}]")
    if a.dry_run:
        print("\n[dry-run] no capsules emitted")
        return 0
    PB.KERNELS.mkdir(parents=True, exist_ok=True)
    for e in corpus:
        cap = MSE.make_matmul_capsule(
            name=e["id"], semantic=f"golden_matmul_{e['source']}",
            M=e["M"], K=e["K"], N=e["N"], lhs="X", weight="W", out="Y0",
            epilogue=e["epilogue"], output_dtype=e["output_dtype"],
            acc_scale=e.get("acc_scale"), label="dev", source_reference=e["source"])
        MSE.export_capsule_dir(PB.KERNELS, cap, comment=f"{e['id']} ({e['note']})")
    corpus_path = PB.KERNELS / "kernel_corpus.yaml"
    doc = yaml.safe_load(corpus_path.read_text()) if corpus_path.exists() else {}
    doc = doc or {}
    doc["golden_kernels"] = corpus
    corpus_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"\nemitted {len(corpus)} golden kernels -> {PB.KERNELS} (+ kernel_corpus.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
