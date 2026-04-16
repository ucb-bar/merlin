#!/usr/bin/env python3
"""Classify IREE-emitted dispatch function bodies from a linked RISC-V
assembly dump into execution-path segments.

Rationale — see METHODS_dispatch_classification.md for the full writeup.
The script is the single source of truth for how the decomposition
figures turn raw ELF assembly into the segment breakdowns shown in the
paper. It is deliberately simple (name-based regex + OPU-opcode count)
so the output can be audited.

Two things the script takes pains to get right:

1. It **only counts real IREE dispatches** — not library helpers like
   `fma`, `__truncsfhf2`, `iree_hal_executable_library_query`, or
   `.L...$local` labels that a linked dispatch ELF happens to pull in.
   A dispatch is a function whose mangled name matches one of the IREE
   patterns below.

2. It **subdivides elementwise into readable categories** driven by the
   I/O dtype signature IREE embeds in the dispatch name. A reader
   looking at the figure can tell `quantize (f32→i8)` from
   `dequantize (i8→f32)` apart from the fused `BN+scale` class.

Usage:
    python3 classify_dispatches.py \
        --dumps-root /tmp/verify_all \
        --output benchmarks/SaturnOPU/model_dispatch_decomposition.csv \
        --summary benchmarks/SaturnOPU/per_model_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

# ------------------------------------------------------------------------
# What counts as a dispatch
# ------------------------------------------------------------------------
# IREE emits async-dispatch bodies with these name shapes (captured in
# the first group). Everything else — math library thunks, runtime
# support functions, local labels — is excluded from the count.
DISPATCH_NAME_RE = re.compile(r"^(?:\w+\$async_dispatch_\d+_|_encoding_\d+_encode_|_initializer_\d+_dispatch_\d+_)")

# OPU outer-product VOPACC: .insn r 87, 2, 81, ... (16×16 outer product).
VOPACC_RE = re.compile(r"\.insn\s+r\s+87\s*,\s*2\s*,\s*81\b")

# Function boundary marker in the linked assembly dump.
FN_TYPE_RE = re.compile(r"^\s*\.type\s+([^,\s]+),@function", re.M)


# ------------------------------------------------------------------------
# Segment classifier
# ------------------------------------------------------------------------
def classify(name: str, vopacc_count: int) -> str:
    """Map a dispatch function name + its OPU opcode count to a segment.

    Rules, in order:
      1. Compile-time pack / encode / initializer bodies  → data_movement.
      2. Any matmul/matvec body with ≥1 VOPACC  → one of the OPU
         tile categories, chosen by the name pattern and VOPACC count.
      3. Non-OPU matmul bodies (VOPACC==0, matmul in name) → rvv_matmul.
      4. Convolution bodies without VOPACC  → direct_conv.
      5. Reduction / softmax / norm bodies → rvv_reduction_softmax_norm.
      6. Element-wise bodies with a dtype signature — classify by the
         dtypes appearing in the name (see patterns below).
      7. Fallback → elementwise_other (should be rare with the refined
         classifier; anything landing here is either an unexpected op
         or a missed pattern we should extend).
    """
    n = name.lower()

    # 1. Data-movement / pack / init
    if n.startswith("_encoding_") or n.startswith("_initializer_"):
        return "data_movement"
    # 2-3. Matmul / matvec bodies
    if "matmul" in n or "matvec" in n or "bmm" in n:
        if vopacc_count > 0:
            # Tile size is determined by the VOPACC count in the inner
            # loop body (4 = 32×32 sub-tile, 1 = single 16×16 tile),
            # NOT by the _like_ suffix. That suffix only tells us which
            # compile path emitted the tile (AOT encoding-resolver vs
            # LLVM vector-contract pattern); the hardware work is the
            # same either way.
            if vopacc_count >= 4:
                return "encoding_32x32_tile"
            return "encoding_16x16_tile"
        return "rvv_matmul"
    # 4. Convolution bodies without VOPACC
    if "conv" in n:
        return "direct_conv"
    # 5. Reductions / softmax / layernorm / batchnorm
    if "reduction" in n or "softmax" in n or "_norm" in n or "layernorm" in n:
        return "rvv_reduction_softmax_norm"
    # 6. Element-wise dtype-signature fan-out.
    #    IREE encodes the I/O dtypes in the dispatch name, which is how
    #    we tell quantize / dequantize / requantize / activation apart.
    if "elementwise" in n or "broadcast" in n or "generic_" in n:
        # Longer, more-specific patterns first (≥4 dtypes = fused BN/scale).
        multi = re.search(r"_(?:[if]\d+x){3,}[if]\d+\b", n)
        if multi:
            return "elementwise_multi_dtype"  # fused BN / scale / bias-add
        if "transpose" in n:
            return "transpose_reshape"
        # Binary quant conversions
        if re.search(r"_f32xi8\b", n):
            return "quantize_f32_to_i8"
        if re.search(r"_i32xi8\b", n):
            return "requantize_i32_to_i8"
        if re.search(r"_i8xf32\b", n) or re.search(r"_i32xf32\b", n):
            return "dequantize_i8_to_f32"
        if re.search(r"_(?:i8xi32|i32xi32)\b", n):
            return "requantize_i32_to_i8"  # i8 lanes extended to i32 accumulator
        # Pure-i8 elementwise (broadcast, add, relu, etc.)
        if re.search(r"_i8\b", n):
            return "activation"
        # Pure-f32 elementwise (GELU on f32, etc.)
        if re.search(r"_f32\b", n):
            return "activation"
    # 7. Fallback
    return "elementwise_other"


# ------------------------------------------------------------------------
# Asm walking
# ------------------------------------------------------------------------
def parse_asm(asm_path: Path) -> list[tuple[str, int]]:
    """Yield (function_name, vopacc_count) for every function in the asm
    that matches DISPATCH_NAME_RE. Non-dispatch helpers are excluded."""
    text = asm_path.read_text()
    bounds = [(m.start(), m.group(1)) for m in FN_TYPE_RE.finditer(text)]
    bounds.append((len(text), None))
    out = []
    for i, (start, name) in enumerate(bounds[:-1]):
        if not name:
            continue
        # Filter: must match a dispatch name pattern
        if not DISPATCH_NAME_RE.match(name) and not DISPATCH_NAME_RE.search(name):
            continue
        body = text[start : bounds[i + 1][0]]
        v = len(VOPACC_RE.findall(body))
        out.append((name, v))
    return out


# ------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------
OPU_SEGMENTS = {
    "encoding_32x32_tile",
    "encoding_16x16_tile",
    "encoding_narrow_tile",
    "encoding_other_tile",
    "runtime_32x32_tile",
    "runtime_16x16_tile",
    "runtime_8x8_tile",
    "runtime_narrow_tile",
    "runtime_other_tile",
    "fused_qdq",
    "inline_vopacc",
}

DEFAULT_DUMPS = {
    "opu_bench_convnet": ("ConvNet", "convnet_OPU_IM2COL"),
    "dronet": ("DroNet", "dronet_OPU_IM2COL"),
    "yolov8_nano": ("YOLOv8-n", "yolov8_OPU_IM2COL"),
    "opu_bench_vit": ("ViT", "vit_v3_OPU_LLM"),
    "mlp_wide": ("MLP-Wide", "mlp_wide_OPU"),
    "mlp_fast": ("MLP-Fast", "mlp_fast_OPU"),
    "tinyllama": ("TinyLlama", "tinyllama_OPU_LLM"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps-root", type=Path, default=Path("/tmp/verify_all"))
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "model_dispatch_decomposition.csv")
    p.add_argument("--summary", type=Path, default=Path(__file__).parent / "per_model_summary.csv")
    args = p.parse_args()

    # Load existing CSVs to preserve rows for models we're not refreshing.
    existing = list(csv.DictReader(open(args.output)))
    fieldnames = list(existing[0].keys())
    # Drop rows for the models we're about to rebuild.
    keep = [r for r in existing if r["model_key"] not in DEFAULT_DUMPS]

    new_rows = []
    per_model_stats = []
    for mk, (display, subdir) in DEFAULT_DUMPS.items():
        asm_files = list((args.dumps_root / subdir).glob("*.s"))
        if not asm_files:
            print(f"[skip] {mk}: no .s in {args.dumps_root / subdir}")
            continue
        # raw function count before filter
        raw_count = len(FN_TYPE_RE.findall(asm_files[0].read_text()))
        parsed = parse_asm(asm_files[0])
        counts = Counter()
        for name, vopacc in parsed:
            seg = classify(name, vopacc)
            counts[seg] += 1
            row = {fn: "" for fn in fieldnames}
            row.update(
                {
                    "model_key": mk,
                    "model": display,
                    "idx": str(len(new_rows)),
                    "symbol": name,
                    "source_file": "",
                    "include_in_model": "1",
                    "layer_id": "",
                    "op_kind": seg,
                    "segment": seg,
                    "opu_path": "",
                    "ops": "0",
                    "opu_ops": "0",
                    "compute_pct": "0",
                    "shape": "",
                    "vopacc": str(vopacc),
                    "opmvinbcast": "0",
                    "opu_fetch": "0",
                    "rvv_reduction": "0",
                    "rvv_sqrt": "0",
                    "rvv_gather": "0",
                }
            )
            new_rows.append(row)
        total = sum(counts.values())
        opu = sum(counts[s] for s in OPU_SEGMENTS)
        per_model_stats.append(
            {
                "model_key": mk,
                "display": display,
                "raw_functions": raw_count,
                "dispatches": total,
                "opu_dispatches": opu,
                "counts": counts,
            }
        )

    # Write the dispatch CSV
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(keep + new_rows)

    # Update per_model_summary.csv (dispatches / opu_dispatches / per-segment).
    summ_rows = list(csv.DictReader(open(args.summary)))
    seg_cols = [s for s in new_rows[0].keys() if s in []]  # placeholder
    extra_seg_cols = [
        "quantize_f32_to_i8",
        "dequantize_i8_to_f32",
        "requantize_i32_to_i8",
        "activation",
        "elementwise_multi_dtype",
        "transpose_reshape",
    ]
    summ_fields = list(summ_rows[0].keys())
    for c in extra_seg_cols:
        if c not in summ_fields:
            summ_fields.append(c)
    # Ensure every DEFAULT_DUMPS entry has a row so the classifier can
    # populate its counts. New rows are created with empty values; only
    # model_key and model are filled so the match succeeds below.
    existing_keys = {r["model_key"] for r in summ_rows}
    for mk, (display, _subdir) in DEFAULT_DUMPS.items():
        if mk not in existing_keys:
            new = {fn: "" for fn in summ_fields}
            new["model_key"] = mk
            new["model"] = display
            summ_rows.append(new)
    for r in summ_rows:
        r.setdefault("quantize_f32_to_i8", "")
        r.setdefault("dequantize_i8_to_f32", "")
        r.setdefault("requantize_i32_to_i8", "")
        r.setdefault("activation", "")
        r.setdefault("elementwise_multi_dtype", "")
        r.setdefault("transpose_reshape", "")
        for stat in per_model_stats:
            if stat["model_key"] == r["model_key"]:
                c = stat["counts"]
                total = stat["dispatches"]
                opu = stat["opu_dispatches"]
                # Non-segment stat columns must be set directly — don't
                # let the per-segment loop below clobber them to 0.
                STAT_COLS = {
                    "model_key",
                    "model",
                    "dispatches",
                    "opu_dispatches",
                    "opu_dispatch_pct",
                    "artifact_dir",
                    "opu_params",
                    "total_params",
                    "opu_param_pct",
                    "opu_compute_ops",
                    "total_compute_ops",
                    "opu_compute_pct",
                    "total_ops",
                    "opu_ops",
                    "opu_pct",
                    "runtime_support",
                }
                # Only iterate over known segment columns — these are the
                # ones the counter can populate. Anything else (stats,
                # runtime_support) is left alone.
                all_segs = set(c.keys()) | (set(r.keys()) - STAT_COLS)
                for col in all_segs:
                    if col in r:
                        r[col] = str(c.get(col, 0))
                r["dispatches"] = str(total)
                r["opu_dispatches"] = str(opu)
                r["opu_dispatch_pct"] = f"{100*opu/total:.6f}" if total else "0"
                r["runtime_support"] = str(stat["raw_functions"] - stat["dispatches"])
                break
    with open(args.summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ_fields)
        w.writeheader()
        w.writerows(summ_rows)

    # Print transparency table
    print(f"\n{'model':20s} {'raw':>6s} {'dispatches':>11s} {'excluded':>9s} " f"{'OPU':>5s} {'OPU%':>6s}")
    for s in per_model_stats:
        excluded = s["raw_functions"] - s["dispatches"]
        pct = 100 * s["opu_dispatches"] / s["dispatches"] if s["dispatches"] else 0
        print(
            f"  {s['model_key']:18s} {s['raw_functions']:>6d} {s['dispatches']:>11d} "
            f"{excluded:>9d} {s['opu_dispatches']:>5d} {pct:>5.1f}%"
        )
        for seg, n in sorted(s["counts"].items(), key=lambda kv: -kv[1]):
            tag = "OPU" if seg in OPU_SEGMENTS else "   "
            print(f"    [{tag}] {seg:30s} n={n}")


if __name__ == "__main__":
    main()
