#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Regression harness for cuda_tile convolution support.
#
# This is a manual GPU-backed correctness suite. It compiles each case with both
# the cuda_tile and cuda backends, runs both on GPU, and checks results against
# a NumPy oracle. Cases are annotated with the current expected cuda_tile
# support boundary so the harness can detect both regressions and support
# changes.
#
# Example:
#   python3 compiler/plugins/target/CudaTile/test/conv_regression.py \
#     --build-dir /scratch/ashvin/merlin/build/host-merlin-release

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
BUFFER_VIEW_BLOCK_RE = re.compile(
    r"result\[\d+\]:\s*hal\.buffer_view\s*"
    r"(?P<shape_dtype>(?:\d+x)*[A-Za-z0-9_<>]+)=(?P<data>.*?)(?=\nresult\[\d+\]:|\Z)",
    re.DOTALL,
)

Expected = Literal["pass", "xfail"]
ConvKind = Literal["conv1d", "conv2d", "conv3d"]


def format_scalar(value: np.generic | int | float) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int):
        return str(value)
    return np.format_float_positional(float(value), trim="-")


def tensor_to_cli_arg(array: np.ndarray) -> str:
    array = np.asarray(array, dtype=np.float32)
    shape_prefix = ""
    if array.ndim > 0:
        shape_prefix = "x".join(str(dim) for dim in array.shape) + "x"
    payload = ",".join(format_scalar(v) for v in array.reshape(-1))
    return f"{shape_prefix}f32={payload}"


def parse_shape_dtype(shape_dtype: str) -> tuple[tuple[int, ...], str]:
    pieces = shape_dtype.split("x")
    return tuple(int(piece) for piece in pieces[:-1] if piece), pieces[-1]


def parse_buffer_view(text: str) -> np.ndarray:
    matches = list(BUFFER_VIEW_BLOCK_RE.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"expected exactly 1 result buffer view, got {len(matches)}")
    match = matches[0]
    shape, dtype = parse_shape_dtype(match.group("shape_dtype"))
    if dtype != "f32":
        raise ValueError(f"unsupported dtype: {dtype}")
    values = np.array(
        [float(v) for v in NUMBER_RE.findall(match.group("data"))], dtype=np.float32
    )
    expected_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if len(values) != expected_size:
        raise ValueError(
            f"parsed {len(values)} values for shape {shape}, expected {expected_size}"
        )
    return values.reshape(shape if shape else ())


def dense_i64_attr(values: tuple[int, ...]) -> str:
    if len(set(values)) == 1:
        return f"dense<{values[0]}> : tensor<{len(values)}xi64>"
    inner = ", ".join(str(v) for v in values)
    return f"dense<[{inner}]> : tensor<{len(values)}xi64>"


def tensor_type(shape: tuple[int, ...]) -> str:
    return "tensor<" + "x".join(str(dim) for dim in shape) + "xf32>"


def valid_conv_extent(input_size: int, kernel_size: int, stride: int, dilation: int) -> int:
    return (input_size - dilation * (kernel_size - 1) - 1) // stride + 1


def conv1d_nwc_wcf(
    inputs: np.ndarray,
    filters: np.ndarray,
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
) -> np.ndarray:
    n, w, cin = inputs.shape
    kw, cin2, cout = filters.shape
    if cin != cin2:
        raise ValueError(f"channel mismatch: {cin} vs {cin2}")
    sw = strides[0]
    dw = dilations[0]
    ow = valid_conv_extent(w, kw, sw, dw)
    out = np.zeros((n, ow, cout), dtype=np.float32)
    for batch in range(n):
        for ox in range(ow):
            acc = np.zeros((cout,), dtype=np.float32)
            base_x = ox * sw
            for kx in range(kw):
                x = base_x + kx * dw
                acc += inputs[batch, x, :] @ filters[kx, :, :]
            out[batch, ox, :] = acc
    return out


def conv2d_nhwc_hwcf(
    inputs: np.ndarray,
    filters: np.ndarray,
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
) -> np.ndarray:
    n, h, w, cin = inputs.shape
    kh, kw, cin2, cout = filters.shape
    if cin != cin2:
        raise ValueError(f"channel mismatch: {cin} vs {cin2}")
    sh, sw = strides
    dh, dw = dilations
    oh = valid_conv_extent(h, kh, sh, dh)
    ow = valid_conv_extent(w, kw, sw, dw)
    out = np.zeros((n, oh, ow, cout), dtype=np.float32)
    for batch in range(n):
        for oy in range(oh):
            base_y = oy * sh
            for ox in range(ow):
                base_x = ox * sw
                acc = np.zeros((cout,), dtype=np.float32)
                for ky in range(kh):
                    y = base_y + ky * dh
                    for kx in range(kw):
                        x = base_x + kx * dw
                        acc += inputs[batch, y, x, :] @ filters[ky, kx, :, :]
                out[batch, oy, ox, :] = acc
    return out


def conv3d_ndhwc_dhwcf(
    inputs: np.ndarray,
    filters: np.ndarray,
    strides: tuple[int, ...],
    dilations: tuple[int, ...],
) -> np.ndarray:
    n, d, h, w, cin = inputs.shape
    kd, kh, kw, cin2, cout = filters.shape
    if cin != cin2:
        raise ValueError(f"channel mismatch: {cin} vs {cin2}")
    sd, sh, sw = strides
    dd, dh, dw = dilations
    od = valid_conv_extent(d, kd, sd, dd)
    oh = valid_conv_extent(h, kh, sh, dh)
    ow = valid_conv_extent(w, kw, sw, dw)
    out = np.zeros((n, od, oh, ow, cout), dtype=np.float32)
    for batch in range(n):
        for oz in range(od):
            base_z = oz * sd
            for oy in range(oh):
                base_y = oy * sh
                for ox in range(ow):
                    base_x = ox * sw
                    acc = np.zeros((cout,), dtype=np.float32)
                    for kz in range(kd):
                        z = base_z + kz * dd
                        for ky in range(kh):
                            y = base_y + ky * dh
                            for kx in range(kw):
                                x = base_x + kx * dw
                                acc += inputs[batch, z, y, x, :] @ filters[kz, ky, kx, :, :]
                    out[batch, oz, oy, ox, :] = acc
    return out


@dataclass(frozen=True)
class ConvCase:
    name: str
    kind: ConvKind
    input_shape: tuple[int, ...]
    filter_shape: tuple[int, ...]
    strides: tuple[int, ...]
    dilations: tuple[int, ...]
    seed: int
    expected_tile: Expected
    atol: float = 1e-4
    rtol: float = 1e-4

    def output_shape(self) -> tuple[int, ...]:
        if self.kind == "conv1d":
            n, w, _ = self.input_shape
            kw, _, cout = self.filter_shape
            ow = valid_conv_extent(w, kw, self.strides[0], self.dilations[0])
            return (n, ow, cout)
        if self.kind == "conv2d":
            n, h, w, _ = self.input_shape
            kh, kw, _, cout = self.filter_shape
            oh = valid_conv_extent(h, kh, self.strides[0], self.dilations[0])
            ow = valid_conv_extent(w, kw, self.strides[1], self.dilations[1])
            return (n, oh, ow, cout)
        if self.kind == "conv3d":
            n, d, h, w, _ = self.input_shape
            kd, kh, kw, _, cout = self.filter_shape
            od = valid_conv_extent(d, kd, self.strides[0], self.dilations[0])
            oh = valid_conv_extent(h, kh, self.strides[1], self.dilations[1])
            ow = valid_conv_extent(w, kw, self.strides[2], self.dilations[2])
            return (n, od, oh, ow, cout)
        raise ValueError(f"unsupported kind: {self.kind}")

    def make_inputs(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        return [
            rng.standard_normal(self.input_shape, dtype=np.float32),
            rng.standard_normal(self.filter_shape, dtype=np.float32),
        ]

    def oracle(self, inputs: list[np.ndarray]) -> np.ndarray:
        if self.kind == "conv1d":
            return conv1d_nwc_wcf(inputs[0], inputs[1], self.strides, self.dilations)
        if self.kind == "conv2d":
            return conv2d_nhwc_hwcf(inputs[0], inputs[1], self.strides, self.dilations)
        if self.kind == "conv3d":
            return conv3d_ndhwc_dhwcf(inputs[0], inputs[1], self.strides, self.dilations)
        raise ValueError(f"unsupported kind: {self.kind}")

    def mlir(self) -> str:
        out_shape = self.output_shape()
        input_ty = tensor_type(self.input_shape)
        filter_ty = tensor_type(self.filter_shape)
        out_ty = tensor_type(out_shape)
        stride_attr = dense_i64_attr(self.strides)
        dilation_attr = dense_i64_attr(self.dilations)
        if self.kind == "conv1d":
            op_name = "linalg.conv_1d_nwc_wcf"
        elif self.kind == "conv2d":
            op_name = "linalg.conv_2d_nhwc_hwcf"
        elif self.kind == "conv3d":
            op_name = "linalg.conv_3d_ndhwc_dhwcf"
        else:
            raise ValueError(f"unsupported kind: {self.kind}")
        return f"""\
func.func @{self.name}(%input: {input_ty}, %filter: {filter_ty}) -> {out_ty} {{
  %empty = tensor.empty() : {out_ty}
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero: f32) outs(%empty: {out_ty}) -> {out_ty}
  %result = {op_name} {{dilations = {dilation_attr}, strides = {stride_attr}}}
    ins(%input, %filter: {input_ty}, {filter_ty})
    outs(%fill: {out_ty}) -> {out_ty}
  return %result : {out_ty}
}}"""


@dataclass
class CaseResult:
    case: ConvCase
    cuda_ok: bool
    tile_outcome: str
    detail: str


CASES = [
    ConvCase("conv2d_1x1_odd", "conv2d", (1, 7, 11, 5), (1, 1, 5, 7), (1, 1), (1, 1), 101, "pass"),
    ConvCase("conv2d_1x1_stride2", "conv2d", (1, 9, 11, 3), (1, 1, 3, 5), (2, 2), (1, 1), 106, "pass"),
    ConvCase("conv2d_1x1_stride3", "conv2d", (1, 13, 16, 2), (1, 1, 2, 4), (3, 3), (1, 1), 108, "pass"),
    ConvCase("conv2d_4x4_odd", "conv2d", (1, 9, 10, 3), (4, 4, 3, 5), (1, 1), (1, 1), 102, "pass"),
    ConvCase("conv2d_3x3_stride2", "conv2d", (1, 11, 13, 4), (3, 3, 4, 6), (2, 2), (1, 1), 103, "pass"),
    ConvCase("conv2d_2x5_stride3", "conv2d", (1, 14, 17, 2), (2, 5, 2, 3), (3, 3), (1, 1), 104, "pass"),
    ConvCase("conv2d_batch2_1x1", "conv2d", (2, 7, 9, 3), (1, 1, 3, 4), (1, 1), (1, 1), 107, "pass"),
    ConvCase("conv2d_batch2_3x3", "conv2d", (2, 7, 9, 3), (3, 3, 3, 4), (1, 1), (1, 1), 105, "pass"),
    ConvCase("conv2d_batch2_1x1_stride2", "conv2d", (2, 9, 11, 3), (1, 1, 3, 5), (2, 2), (1, 1), 109, "pass"),
    ConvCase("conv2d_batch2_1x1_stride3", "conv2d", (2, 13, 16, 2), (1, 1, 2, 4), (3, 3), (1, 1), 110, "pass"),
    ConvCase("conv2d_batch2_3x3_stride2", "conv2d", (2, 11, 13, 3), (3, 3, 3, 4), (2, 2), (1, 1), 111, "pass"),
    ConvCase("conv1d_k1_stride1_batch2", "conv1d", (2, 15, 3), (1, 3, 4), (1,), (1,), 204, "pass"),
    ConvCase("conv1d_k1_stride2", "conv1d", (1, 17, 2), (1, 2, 4), (2,), (1,), 205, "xfail"),
    ConvCase("conv1d_k3_stride1_odd", "conv1d", (1, 13, 3), (3, 3, 5), (1,), (1,), 201, "xfail"),
    ConvCase("conv1d_k4_stride2_odd", "conv1d", (1, 17, 2), (4, 2, 4), (2,), (1,), 202, "xfail"),
    ConvCase("conv1d_batch2_k2", "conv1d", (2, 15, 3), (2, 3, 4), (1,), (1,), 203, "xfail"),
    ConvCase("conv3d_1x1x1_odd", "conv3d", (1, 5, 7, 6, 2), (1, 1, 1, 2, 3), (1, 1, 1), (1, 1, 1), 301, "pass"),
    ConvCase("conv3d_1x1x1_stride2", "conv3d", (1, 7, 9, 8, 2), (1, 1, 1, 2, 3), (2, 2, 2), (1, 1, 1), 304, "pass"),
    ConvCase("conv3d_batch2_1x1x1", "conv3d", (2, 4, 5, 6, 2), (1, 1, 1, 2, 3), (1, 1, 1), (1, 1, 1), 305, "pass"),
    ConvCase("conv3d_2x3x2_odd", "conv3d", (1, 6, 8, 7, 2), (2, 3, 2, 2, 4), (1, 1, 1), (1, 1, 1), 302, "xfail"),
    ConvCase("conv3d_2x2x3_stride2", "conv3d", (1, 7, 9, 8, 2), (2, 2, 3, 2, 3), (2, 2, 2), (1, 1, 1), 303, "xfail"),
]


def compile_module(
    iree_compile: Path, mlir_path: Path, backend: str, vmfb_path: Path
) -> tuple[bool, str]:
    cmd = [str(iree_compile), str(mlir_path), f"--iree-hal-target-backends={backend}"]
    if backend == "cuda_tile":
        cmd.append("--iree-cuda-tile-enable-codegen=true")
    cmd.extend(["-o", str(vmfb_path)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip()
    return True, ""


def run_module(
    iree_run: Path, vmfb_path: Path, device: str, function: str, inputs: list[np.ndarray]
) -> tuple[np.ndarray | None, str]:
    cmd = [str(iree_run), f"--device={device}", f"--module={vmfb_path}", f"--function={function}"]
    for array in inputs:
        cmd.append(f"--input={tensor_to_cli_arg(array)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, result.stderr.strip() or result.stdout.strip()
    try:
        return parse_buffer_view(result.stdout), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"parse error: {exc}\nraw stdout:\n{result.stdout}"


def max_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))


def evaluate_case(
    case: ConvCase,
    iree_compile: Path,
    iree_run: Path,
    artifact_dir: Path,
) -> CaseResult:
    mlir_path = artifact_dir / f"{case.name}.mlir"
    tile_vmfb = artifact_dir / f"{case.name}_tile.vmfb"
    cuda_vmfb = artifact_dir / f"{case.name}_cuda.vmfb"
    mlir_path.write_text(case.mlir())

    inputs = case.make_inputs()
    expected = case.oracle(inputs)

    ok, msg = compile_module(iree_compile, mlir_path, "cuda_tile", tile_vmfb)
    if not ok:
        if case.expected_tile == "xfail":
            return CaseResult(case, True, "XFAIL", f"cuda_tile compile failed: {msg}")
        return CaseResult(case, False, "FAIL", f"cuda_tile compile failed: {msg}")

    ok, msg = compile_module(iree_compile, mlir_path, "cuda", cuda_vmfb)
    if not ok:
        return CaseResult(case, False, "FAIL", f"cuda compile failed: {msg}")

    cuda_out, cuda_err = run_module(iree_run, cuda_vmfb, "cuda", case.name, inputs)
    if cuda_out is None:
        return CaseResult(case, False, "FAIL", f"cuda run failed: {cuda_err}")
    cuda_ok = np.allclose(cuda_out, expected, atol=case.atol, rtol=case.rtol)
    if not cuda_ok:
        detail = f"cuda mismatch vs oracle (max_abs_diff={max_abs_diff(cuda_out, expected):.6g})"
        return CaseResult(case, False, "FAIL", detail)

    tile_out, tile_err = run_module(iree_run, tile_vmfb, "cuda_tile", case.name, inputs)
    if tile_out is None:
        if case.expected_tile == "xfail":
            return CaseResult(case, True, "XFAIL", f"cuda_tile run failed: {tile_err}")
        return CaseResult(case, True, "FAIL", f"cuda_tile run failed: {tile_err}")

    tile_ok = np.allclose(tile_out, expected, atol=case.atol, rtol=case.rtol)
    if tile_ok and case.expected_tile == "pass":
        return CaseResult(case, True, "PASS", "matched oracle")
    if not tile_ok and case.expected_tile == "xfail":
        detail = f"expected mismatch (max_abs_diff={max_abs_diff(tile_out, expected):.6g})"
        return CaseResult(case, True, "XFAIL", detail)
    if tile_ok and case.expected_tile == "xfail":
        return CaseResult(case, True, "XPASS", "matched oracle")
    detail = f"unexpected mismatch (max_abs_diff={max_abs_diff(tile_out, expected):.6g})"
    return CaseResult(case, True, "FAIL", detail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cuda_tile convolution regression cases")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/scratch/ashvin/merlin/build/host-merlin-release"),
        help="Directory containing iree-compile and iree-run-module under tools/",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/tmp/cuda_tile_conv_regression"),
        help="Temporary directory for generated MLIR and VMFB artifacts",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only run cases whose name contains this substring",
    )
    parser.add_argument(
        "--allow-xpass",
        action="store_true",
        help="Do not fail the run if an expected-fail case starts passing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    iree_compile = args.build_dir / "tools" / "iree-compile"
    iree_run = args.build_dir / "tools" / "iree-run-module"

    missing_tools = [str(path) for path in (iree_compile, iree_run) if not path.is_file()]
    if missing_tools:
        print("Missing required tools:")
        for path in missing_tools:
            print(f"  {path}")
        return 2

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = [case for case in CASES if args.filter in case.name]
    if not selected_cases:
        print(f"No cases matched filter: {args.filter!r}")
        return 2

    summary = {"PASS": 0, "XFAIL": 0, "XPASS": 0, "FAIL": 0}
    failed = False

    for case in selected_cases:
        result = evaluate_case(case, iree_compile, iree_run, args.artifact_dir)
        summary[result.tile_outcome] += 1
        print(f"{result.tile_outcome:5} {case.name}: {result.detail}")
        if result.tile_outcome == "FAIL":
            failed = True
        if result.tile_outcome == "XPASS" and not args.allow_xpass:
            failed = True

    print("\nSummary:")
    for key in ("PASS", "XFAIL", "XPASS", "FAIL"):
        print(f"  {key}: {summary[key]}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
