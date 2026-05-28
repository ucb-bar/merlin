#!/usr/bin/env python3
"""Summarize quantization-relevant IR signals for IREE phase dumps.

Usage:
  python tools/analyze_quant_ir.py tmp/smolvla_global_opt_phases_quantfix5_attn_decomp_dtype
  python tools/analyze_quant_ir.py tmp/... --enforce-low-precision
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = [
    ("linalg.matmul", r"linalg\.matmul"),
    ("linalg.batch_matmul", r"linalg\.batch_matmul"),
    ("linalg.generic", r"linalg\.generic"),
    ("iree_linalg_ext.attention", r"iree_linalg_ext\.attention"),
    ("linalg.softmax", r"linalg\.softmax"),
    ("linalg.softmax f32", r"linalg\.softmax[^\n]*tensor<[^>]*xf32>"),
    ("linalg.softmax bf16", r"linalg\.softmax[^\n]*tensor<[^>]*xbf16>"),
    ("attention region f32", r"^\s*\^bb0\(%arg\d+: f32\):"),
    ("attention region bf16", r"^\s*\^bb0\(%arg\d+: bf16\):"),
    ("matmul i8*i8->i32", r"linalg\.matmul .*tensor<[^>]*xi8>.*, tensor<[^>]*xi8>.*tensor<[^>]*xi32>"),
    ("matmul f8*f8->f16", r"linalg\.matmul .*tensor<[^>]*xf8E4M3FN>.*, tensor<[^>]*xf8E4M3FN>.*tensor<[^>]*xf16>"),
    ("batch_matmul f32*f32->f32", r"linalg\.batch_matmul .*tensor<[^>]*xf32>.*, tensor<[^>]*xf32>.*tensor<[^>]*xf32>"),
    (
        "batch_matmul f32*bf16->f32",
        r"linalg\.batch_matmul .*tensor<[^>]*xf32>.*, tensor<[^>]*xbf16>.*tensor<[^>]*xf32>",
    ),
    (
        "batch_matmul bf16*bf16->f32",
        r"linalg\.batch_matmul .*tensor<[^>]*xbf16>.*, tensor<[^>]*xbf16>.*tensor<[^>]*xf32>",
    ),
    (
        "batch_matmul bf16*bf16->bf16",
        r"linalg\.batch_matmul .*tensor<[^>]*xbf16>.*, tensor<[^>]*xbf16>.*tensor<[^>]*xbf16>",
    ),
    ("f8->bf16 extf", r"arith\.extf .*f8E4M3FN to bf16"),
    ("i32->f32 bitcast", r"iree_tensor_ext\.bitcast .*xi32.*xf32"),
    ("f32->bf16 truncf", r"arith\.truncf .*f32 to bf16"),
    ("tensor<...xf32>", r"tensor<[^>]*xf32>"),
    ("tensor<...xbf16>", r"tensor<[^>]*xbf16>"),
    ("tensor<...xf8E4M3FN>", r"tensor<[^>]*xf8E4M3FN>"),
]


def count_regex(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def count_initializer_scoped(lines: list[str]) -> dict[str, int]:
    in_initializer = False
    counts = {
        "bitcast xi32->f32 in initializer": 0,
        "bitcast xi32->f32 outside initializer": 0,
        "truncf f32->bf16 in initializer": 0,
        "truncf f32->bf16 outside initializer": 0,
    }
    for line in lines:
        if re.match(r"^\s*util\.initializer\b", line):
            in_initializer = True
        elif in_initializer and re.match(r"^  }$", line):
            in_initializer = False

        if re.search(r"iree_tensor_ext\.bitcast .*xi32.*xf32", line):
            key = "bitcast xi32->f32 in initializer" if in_initializer else "bitcast xi32->f32 outside initializer"
            counts[key] += 1
        if re.search(r"arith\.truncf .*f32 to bf16", line):
            key = "truncf f32->bf16 in initializer" if in_initializer else "truncf f32->bf16 outside initializer"
            counts[key] += 1
    return counts


def summarize_file(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    counts: dict[str, int] = {"lines": len(lines)}
    for label, pattern in PATTERNS:
        counts[label] = count_regex(text, pattern)
    counts.update(count_initializer_scoped(lines))

    print(f"== {path}")
    print(f"lines: {counts['lines']}")
    for label, _ in PATTERNS:
        print(f"{label:36} {counts[label]}")
    for label in [
        "bitcast xi32->f32 in initializer",
        "bitcast xi32->f32 outside initializer",
        "truncf f32->bf16 in initializer",
        "truncf f32->bf16 outside initializer",
    ]:
        print(f"{label:36} {counts[label]}")
    print()
    return counts


def enforce_low_precision(input_counts: dict[str, int], global_counts: dict[str, int], args: argparse.Namespace) -> int:
    checks: list[tuple[str, int, int]] = [
        ("input attention region f32", input_counts["attention region f32"], args.max_input_attention_region_f32),
        ("global attention region f32", global_counts["attention region f32"], args.max_global_attention_region_f32),
        ("global linalg.softmax f32", global_counts["linalg.softmax f32"], args.max_global_softmax_f32),
        (
            "global batch_matmul f32*f32->f32",
            global_counts["batch_matmul f32*f32->f32"],
            args.max_global_batch_matmul_f32_f32,
        ),
        (
            "global batch_matmul f32*bf16->f32",
            global_counts["batch_matmul f32*bf16->f32"],
            args.max_global_batch_matmul_f32_bf16,
        ),
        (
            "global batch_matmul bf16*bf16->f32",
            global_counts["batch_matmul bf16*bf16->f32"],
            args.max_global_batch_matmul_bf16_bf16_f32,
        ),
        (
            "global tensor<...xf32>",
            global_counts["tensor<...xf32>"],
            args.max_global_tensor_f32,
        ),
        (
            "global bitcast xi32->f32 outside initializer",
            global_counts["bitcast xi32->f32 outside initializer"],
            args.max_global_bitcast_outside_initializer,
        ),
        (
            "global truncf f32->bf16 outside initializer",
            global_counts["truncf f32->bf16 outside initializer"],
            args.max_global_truncf_outside_initializer,
        ),
    ]

    failures = [(name, value, limit) for name, value, limit in checks if value > limit]
    if not failures:
        print("LOW_PRECISION_ENFORCEMENT: PASS")
        return 0

    print("LOW_PRECISION_ENFORCEMENT: FAIL")
    for name, value, limit in failures:
        print(f"  - {name}: observed={value}, limit={limit}")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path, help="Path containing module.*.mlir phase dumps")
    parser.add_argument(
        "--enforce-low-precision",
        action="store_true",
        help="Fail with non-zero exit code if configured fp32 thresholds are exceeded.",
    )
    parser.add_argument("--max-input-attention-region-f32", type=int, default=0)
    parser.add_argument("--max-global-attention-region-f32", type=int, default=0)
    parser.add_argument("--max-global-softmax-f32", type=int, default=0)
    parser.add_argument("--max-global-batch-matmul-f32-f32", type=int, default=0)
    parser.add_argument("--max-global-batch-matmul-f32-bf16", type=int, default=0)
    parser.add_argument("--max-global-batch-matmul-bf16-bf16-f32", type=int, default=0)
    parser.add_argument("--max-global-tensor-f32", type=int, default=0)
    parser.add_argument("--max-global-bitcast-outside-initializer", type=int, default=0)
    parser.add_argument("--max-global-truncf-outside-initializer", type=int, default=0)
    args = parser.parse_args()

    files = [
        args.dump_dir / "module.1.input.mlir",
        args.dump_dir / "module.4.global-optimization.mlir",
    ]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        print("Missing expected files:")
        for p in missing:
            print(f"  - {p}")
        return 1

    input_counts = summarize_file(files[0])
    global_counts = summarize_file(files[1])

    if args.enforce_low_precision:
        return enforce_low_precision(input_counts, global_counts, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
