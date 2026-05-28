#!/usr/bin/env python3
"""Simulate the IREE-emitted requantize loop on a given i32 tensor.

The dispatch ELF's per-element requantize for an int8-quantized matmul
is (matching the LLVM IR in our dumps):

    f = sitofp(i32_acc)
    f = fmul(f, input_weight_scale)     // round-to-nearest-even
    f = fdiv(f, output_scale)
    f = round(f)                        // round-to-nearest-even
    f = max(f, -128.0)
    f = min(f, 127.0)
    out_i8 = fptosi(f, i8)

Useful for validating that backend i8 outputs match expectation given
a known i32 accumulator. e.g.:

    # bias-only case (zero input matmul → output = bias only)
    ./merlin simulate-requantize --bias-i32 -1,2,3,4 \\
        --input-weight-scale 6.25015527e-5 \\
        --output-scale 0.0250296649

This script lets you sanity-check whether the i32 → i8 conversion
the backend is doing matches what onnxruntime computes.
"""

from __future__ import annotations

import argparse
import sys


def parse_i32_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def requantize(i32_values: list[int], in_w_scale: float, out_scale: float, bias_add: float = 0.0) -> list[int]:
    """Match the bit-precise IR pipeline."""
    out = []
    for v in i32_values:
        # sitofp
        f = float(v)
        # fmul
        f = f * in_w_scale
        # fdiv
        f = f / out_scale
        # roundeven (Python round() uses banker's rounding for halves)
        f = round(f)
        f = float(f)
        # addf
        f = f + bias_add
        # clamp -128..127
        if f < -128.0:
            f = -128.0
        if f > 127.0:
            f = 127.0
        # fptosi
        out.append(int(f))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bias-i32", required=True, help="Comma-separated i32 bias accumulator values")
    p.add_argument(
        "--input-weight-scale",
        type=float,
        required=True,
        help="input_scale × weight_scale, the fmul constant",
    )
    p.add_argument("--output-scale", type=float, required=True, help="The fdiv divisor")
    p.add_argument(
        "--bias-add",
        type=float,
        default=0.0,
        help="Optional additive bias constant (default 0.0)",
    )
    args = p.parse_args()

    i32 = parse_i32_list(args.bias_i32)
    i8 = requantize(i32, args.input_weight_scale, args.output_scale, args.bias_add)
    for v32, v8 in zip(i32, i8):
        print(f"  i32={v32:>12}  →  i8={v8:>4}  (hex={v8 & 0xFF:02x})")


if __name__ == "__main__":
    sys.exit(main())
