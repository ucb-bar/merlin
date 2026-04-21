#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Manual GPU-backed regression harness for the cuda_tile codegen breadth path.
#
# Example:
#   python3 compiler/plugins/target/CudaTile/test/codegen_regression.py \
#     --build-dir /scratch/ashvin/merlin/build/host-merlin-release

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
BUFFER_VIEW_BLOCK_RE = re.compile(
    r"result\[\d+\]:\s*hal\.buffer_view\s*"
    r"(?P<shape_dtype>(?:\d+x)*[A-Za-z0-9_<>]+)=(?P<data>.*?)(?=\nresult\[\d+\]:|\Z)",
    re.DOTALL,
)


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
        [float(value) for value in NUMBER_RE.findall(match.group("data"))],
        dtype=np.float32,
    )
    expected_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if len(values) != expected_size:
        raise ValueError(
            f"parsed {len(values)} values for shape {shape}, expected {expected_size}"
        )
    return values.reshape(shape if shape else ())


@dataclass(frozen=True)
class CodegenCase:
    name: str
    mlir: str
    inputs: tuple[np.ndarray, ...]
    expected: np.ndarray
    atol: float = 1e-4
    rtol: float = 1e-4


@dataclass(frozen=True)
class CaseResult:
    case: CodegenCase
    ok: bool
    detail: str


def make_add_case() -> CodegenCase:
    lhs = np.arange(1, 9, dtype=np.float32)
    rhs = np.arange(10, 90, 10, dtype=np.float32)
    mlir = """\
#map = affine_map<(d0) -> (d0)>

func.func @add(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %result = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel"]}
      ins(%lhs, %rhs: tensor<8xf32>, tensor<8xf32>)
      outs(%empty: tensor<8xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}
"""
    return CodegenCase("add", mlir, (lhs, rhs), lhs + rhs)


def make_reduce_sum_case() -> CodegenCase:
    data = np.arange(1, 33, dtype=np.float32).reshape(4, 8)
    mlir = """\
#in_map = affine_map<(d0, d1) -> (d0, d1)>
#out_map = affine_map<(d0, d1) -> (d0)>

func.func @reduce_sum(%input: tensor<4x8xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero: f32) outs(%empty: tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [#in_map, #out_map],
      iterator_types = ["parallel", "reduction"]}
      ins(%input: tensor<4x8xf32>)
      outs(%fill: tensor<4xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %sum = arith.addf %x, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}
"""
    return CodegenCase("reduce_sum", mlir, (data,), data.sum(axis=1))


def make_matmul_case(name: str, lowering_config: bool) -> CodegenCase:
    lhs = np.arange(1, 33, dtype=np.float32).reshape(4, 8) / 7.0
    rhs = np.arange(1, 33, dtype=np.float32).reshape(8, 4) / 11.0
    attr = ""
    if lowering_config:
        attr = """
      {lowering_config = #iree_gpu.lowering_config<{workgroup = [4, 4, 8]}>}"""
    mlir = f"""\
func.func @{name}(%lhs: tensor<4x8xf32>, %rhs: tensor<8x4xf32>) -> tensor<4x4xf32> {{
  %empty = tensor.empty() : tensor<4x4xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero: f32) outs(%empty: tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.matmul
      ins(%lhs, %rhs: tensor<4x8xf32>, tensor<8x4xf32>)
      outs(%fill: tensor<4x4xf32>){attr}
      -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}}
"""
    return CodegenCase(name, mlir, (lhs, rhs), lhs @ rhs, atol=1e-3, rtol=1e-4)


def compile_module(iree_compile: Path, mlir_path: Path, vmfb_path: Path) -> tuple[bool, str]:
    cmd = [
        str(iree_compile),
        str(mlir_path),
        "--iree-hal-target-backends=cuda_tile",
        "--iree-cuda-tile-enable-codegen=true",
        "-o",
        str(vmfb_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip()
    return True, ""


def run_module(
    iree_run: Path,
    vmfb_path: Path,
    function: str,
    inputs: tuple[np.ndarray, ...],
) -> tuple[np.ndarray | None, str]:
    cmd = [
        str(iree_run),
        "--device=cuda_tile",
        f"--module={vmfb_path}",
        f"--function={function}",
    ]
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
    case: CodegenCase,
    iree_compile: Path,
    iree_run: Path,
    artifact_dir: Path,
) -> CaseResult:
    mlir_path = artifact_dir / f"{case.name}.mlir"
    vmfb_path = artifact_dir / f"{case.name}.vmfb"
    mlir_path.write_text(case.mlir)

    ok, message = compile_module(iree_compile, mlir_path, vmfb_path)
    if not ok:
        return CaseResult(case, False, f"compile failed: {message}")

    actual, message = run_module(iree_run, vmfb_path, case.name, case.inputs)
    if actual is None:
        return CaseResult(case, False, f"run failed: {message}")

    if not np.allclose(actual, case.expected, atol=case.atol, rtol=case.rtol):
        return CaseResult(
            case,
            False,
            f"mismatch (max_abs_diff={max_abs_diff(actual, case.expected):.6g})",
        )
    return CaseResult(case, True, "matched oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cuda_tile codegen regression cases")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("/scratch/ashvin/merlin/build/host-merlin-release"),
        help="Directory containing iree-compile and iree-run-module under tools/",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/tmp/cuda_tile_codegen_regression"),
        help="Temporary directory for generated MLIR and VMFB artifacts",
    )
    parser.add_argument("--filter", default="", help="Only run matching case names")
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
    cases = [
        make_add_case(),
        make_reduce_sum_case(),
        make_matmul_case("matmul", lowering_config=False),
        make_matmul_case("matmul_ireegpu_config", lowering_config=True),
    ]
    cases = [case for case in cases if args.filter in case.name]
    if not cases:
        print(f"No cases matched filter: {args.filter!r}")
        return 2

    failed = False
    for case in cases:
        result = evaluate_case(case, iree_compile, iree_run, args.artifact_dir)
        status = "PASS" if result.ok else "FAIL"
        print(f"{status:4} {case.name}: {result.detail}")
        failed |= not result.ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
