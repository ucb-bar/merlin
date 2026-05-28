#!/usr/bin/env python3
"""Export QNN-friendly conv islands from an XPU-RT dispatch manifest.

The emitted MLIR contains standalone functions for conv-shaped yolov8
dispatches. Each island keeps input, weight, and bias as function arguments
so profiling does not depend on embedded synthetic weights. Boundary layout
transposes, pads, and Q/DQ are intentionally outside the island; those costs
belong to the scheduler as CPU-side boundary work.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections.abc import Iterable

CONV_SUMMARY_RE = re.compile(
    r"^conv_(?P<oc>\d+)x(?P<oh>\d+)x(?P<ow>\d+)x(?P<ic>\d+)x" r"(?P<kh>\d+)x(?P<kw>\d+)_i8xi8xi32$"
)
TENSOR_RE = re.compile(r"tensor<(?P<shape>(?:\d+x)+)(?P<elem>i8|i32|f32)")


def _parse_tensor(text: str) -> tuple[tuple[int, ...], str] | None:
    match = TENSOR_RE.search(text)
    if not match:
        return None
    dims = tuple(int(x) for x in match.group("shape").rstrip("x").split("x"))
    return dims, match.group("elem")


def _infer_stride(in_size: int, kernel: int, out_size: int) -> int | None:
    for stride in (1, 2, 4):
        if ((in_size - kernel) // stride) + 1 == out_size:
            return stride
    return None


def _select_input_shape(inputs: list[str], ic: int, oh: int, ow: int, kh: int, kw: int) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int]] = []
    for text in inputs:
        parsed = _parse_tensor(text)
        if not parsed:
            continue
        dims, elem = parsed
        if elem != "i8" or len(dims) != 3 or dims[0] != ic:
            continue
        if _infer_stride(dims[1], kh, oh) and _infer_stride(dims[2], kw, ow):
            candidates.append(dims)
    if not candidates:
        return None
    # Prefer the smallest valid input. Larger tensors are usually unrelated
    # scratch/readwrite outs carried in the same dispatch wrapper.
    return min(candidates, key=lambda x: x[1] * x[2])


def _has_bias(inputs: list[str], oc: int) -> bool:
    for text in inputs:
        parsed = _parse_tensor(text)
        if not parsed:
            continue
        dims, elem = parsed
        if elem == "i32" and dims == (oc,):
            return True
    return False


def _emit_func(
    name: str,
    ic: int,
    ih: int,
    iw: int,
    oc: int,
    oh: int,
    ow: int,
    kh: int,
    kw: int,
    stride_h: int,
    stride_w: int,
    *,
    static_weights: bool,
) -> str:
    weight_ty = f"tensor<{kh}x{kw}x{ic}x{oc}xi8>"
    input_ty = f"tensor<1x{ih}x{iw}x{ic}xi8>"
    bias_ty = f"tensor<{oc}xi32>"
    acc_ty = f"tensor<1x{oh}x{ow}x{oc}xi32>"
    out_ty = f"tensor<1x{oh}x{ow}x{oc}xi8>"
    if static_weights:
        signature = f"func.func @{name}(%input: {input_ty}) -> {out_ty} {{"
        weight_defs = f"""
    %bias = arith.constant dense<0> : {bias_ty}
    %weight = arith.constant dense<1> : {weight_ty}"""
    else:
        signature = f"func.func @{name}(%input: {input_ty}, %weight: {weight_ty}, " f"%bias: {bias_ty}) -> {out_ty} {{"
        weight_defs = ""
    return f"""
  {signature}
    %c0_i32 = arith.constant 0 : i32
    %cst_min_i8 = arith.constant -1.280000e+02 : f32
    %cst_max_i8 = arith.constant 1.270000e+02 : f32
    %cst_zp = arith.constant 0.000000e+00 : f32
    %cst_acc_scale = arith.constant 5.957487e-03 : f32
    %cst_out_scale = arith.constant 1.322218e+00 : f32
{weight_defs}
    %acc_init = tensor.empty() : {acc_ty}
    %broadcasted = linalg.broadcast ins(%bias : {bias_ty}) outs(%acc_init : {acc_ty}) dimensions = [0, 1, 2]
    %conv_i32 = linalg.conv_2d_nhwc_hwcf_q {{dilations = dense<1> : vector<2xi64>, strides = dense<[{stride_h}, {stride_w}]> : vector<2xi64>}}
        ins(%input, %weight, %c0_i32, %c0_i32 : {input_ty}, {weight_ty}, i32, i32)
        outs(%broadcasted : {acc_ty}) -> {acc_ty}
    %out_init = tensor.empty() : {out_ty}
    %out = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
        ins(%conv_i32 : {acc_ty}) outs(%out_init : {out_ty}) {{
      ^bb0(%in: i32, %y: i8):
        %f = arith.sitofp %in : i32 to f32
        %scaled = arith.mulf %f, %cst_acc_scale : f32
        %requant = arith.divf %scaled, %cst_out_scale : f32
        %rounded = math.roundeven %requant : f32
        %with_zp = arith.addf %rounded, %cst_zp : f32
        %lo = arith.maximumf %with_zp, %cst_min_i8 : f32
        %hi = arith.minimumf %lo, %cst_max_i8 : f32
        %q = arith.fptosi %hi : f32 to i8
        linalg.yield %q : i8
    }} -> {out_ty}
    return %out : {out_ty}
  }}
"""


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--out-mlir", type=pathlib.Path, required=True)
    parser.add_argument("--out-map", type=pathlib.Path, required=True)
    parser.add_argument(
        "--static-weights",
        action="store_true",
        help="Embed dense zero/one bias/weight constants. This is a "
        "static-weight shape-latency probe for HTA, not full model-constant "
        "extraction.",
    )
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    funcs: list[str] = ["module {\n"]
    mapping: dict[str, dict] = {}
    for dispatch, row in sorted(
        manifest.get("dispatches", {}).items(),
        key=lambda kv: int(kv[1].get("id", 0)),
    ):
        match = CONV_SUMMARY_RE.match(row.get("op_summary", ""))
        if not match:
            continue
        vals = {k: int(v) for k, v in match.groupdict().items()}
        input_shape = _select_input_shape(
            row.get("inputs", []),
            vals["ic"],
            vals["oh"],
            vals["ow"],
            vals["kh"],
            vals["kw"],
        )
        if input_shape is None or not _has_bias(row.get("inputs", []), vals["oc"]):
            continue
        ic, ih, iw = input_shape
        stride_h = _infer_stride(ih, vals["kh"], vals["oh"])
        stride_w = _infer_stride(iw, vals["kw"], vals["ow"])
        if stride_h is None or stride_w is None:
            continue
        func_name = f"qnn_conv_island_{dispatch}"
        funcs.append(
            _emit_func(
                func_name,
                ic,
                ih,
                iw,
                vals["oc"],
                vals["oh"],
                vals["ow"],
                vals["kh"],
                vals["kw"],
                stride_h,
                stride_w,
                static_weights=args.static_weights,
            )
        )
        mapping[func_name] = {
            "dispatch": dispatch,
            "op_summary": row.get("op_summary", ""),
            "weights": "static_probe" if args.static_weights else "runtime_args",
            "input_nhwc": [1, ih, iw, ic],
            "weight_hwcf": [vals["kh"], vals["kw"], ic, vals["oc"]],
            "bias": [vals["oc"]],
            "output_nhwc": [1, vals["oh"], vals["ow"], vals["oc"]],
            "strides": [stride_h, stride_w],
            "dependencies": row.get("dependencies", []),
        }
    funcs.append("}\n")
    args.out_mlir.parent.mkdir(parents=True, exist_ok=True)
    args.out_mlir.write_text("\n".join(funcs))
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    args.out_map.write_text(json.dumps(mapping, indent=2))
    print(f"wrote {args.out_mlir} ({len(mapping)} islands)")
    print(f"wrote {args.out_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
