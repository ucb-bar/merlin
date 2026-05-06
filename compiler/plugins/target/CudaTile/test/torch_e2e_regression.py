#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Torch → CudaTile → cuda_new end-to-end regression harness.

Exports small PyTorch modules, compiles with cuda_tile, runs on cuda_new
and cuda_tile, compares against PyTorch reference.

Usage:
  uv run python3 compiler/plugins/target/CudaTile/test/torch_e2e_regression.py \
    --build-dir build/host-merlin-release \
    --artifact-dir /scratch/ashvin/merlin/compile/cudatile_cmp_step1
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
BUFFER_RE = re.compile(
    r"result\[\d+\]:\s*hal\.buffer_view\s*"
    r"(?P<shape_dtype>(?:\d+x)*[A-Za-z0-9_<>]+)=(?P<data>.*?)(?=\nresult\[\d+\]:|\Z)",
    re.DOTALL,
)


def tensor_to_cli(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    shape = "x".join(str(d) for d in arr.shape) + "x" if arr.ndim > 0 else ""
    vals = ",".join(f"{v}" for v in arr.reshape(-1))
    return f"{shape}f32={vals}"


def parse_output(text: str) -> np.ndarray | None:
    matches = list(BUFFER_RE.finditer(text))
    if not matches:
        return None
    m = matches[0]
    nums = [float(x) for x in NUMBER_RE.findall(m.group("data"))]
    pieces = m.group("shape_dtype").split("x")
    shape = tuple(int(p) for p in pieces[:-1] if p)
    return np.array(nums, dtype=np.float32).reshape(shape) if shape else np.array(nums[0], dtype=np.float32)


@dataclass
class Case:
    name: str
    export_fn: object  # callable() -> (torch.nn.Module, tuple[torch.Tensor, ...])
    atol: float = 1e-3
    xfail: str = ""


@dataclass
class Result:
    name: str
    status: str  # PASS, FAIL, XFAIL, SKIP
    detail: str = ""


def _define_cases() -> list[Case]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def tensor_add():
        class M(nn.Module):
            def forward(self, a, b):
                return a + b
        return M().eval(), (torch.randn(2, 4), torch.randn(2, 4))

    def linear():
        m = nn.Linear(8, 4, bias=False)
        return m.eval(), (torch.randn(2, 8),)

    def linear_bias():
        m = nn.Linear(8, 4, bias=True)
        return m.eval(), (torch.randn(2, 8),)

    def linear_bias_relu():
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
            def forward(self, x):
                return F.relu(self.fc(x))
        return M().eval(), (torch.randn(2, 8),)

    def softmax():
        class M(nn.Module):
            def forward(self, x):
                return F.softmax(x, dim=-1)
        return M().eval(), (torch.randn(2, 8),)

    def reduce_sum():
        class M(nn.Module):
            def forward(self, x):
                return x.sum(dim=-1)
        return M().eval(), (torch.randn(4, 8),)

    def tiny_mlp():
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(16, 4)
            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))
        return M().eval(), (torch.randn(2, 8),)

    def tiny_attention():
        class M(nn.Module):
            def forward(self, q, k, v):
                scores = q @ k.transpose(-2, -1)
                weights = F.softmax(scores, dim=-1)
                return weights @ v
        q = torch.randn(1, 4, 8)
        k = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        return M().eval(), (q, k, v)

    return [
        Case("tensor_add", tensor_add),
        Case("linear", linear),
        Case("linear_bias", linear_bias),
        Case("linear_bias_relu", linear_bias_relu),
        Case("softmax", softmax),
        Case("reduce_sum", reduce_sum),
        Case("tiny_mlp", tiny_mlp),
        Case("tiny_attention", tiny_attention, atol=1e-2),
    ]


def _define_extended_cases() -> list[Case]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def reshape_flatten():
        class M(nn.Module):
            def forward(self, x):
                return x.reshape(2, 12)
        return M().eval(), (torch.randn(2, 3, 4),)

    def permute_3d():
        class M(nn.Module):
            def forward(self, x):
                return x.permute(0, 2, 1)
        return M().eval(), (torch.randn(2, 3, 4),)

    def strided_slice():
        class M(nn.Module):
            def forward(self, x):
                return x[:, 1:7:2]
        return M().eval(), (torch.randn(3, 8),)

    def concat_dim1():
        class M(nn.Module):
            def forward(self, a, b):
                return torch.cat([a, b], dim=1)
        return M().eval(), (torch.randn(2, 3), torch.randn(2, 5))

    def gelu():
        class M(nn.Module):
            def forward(self, x):
                return F.gelu(x)
        return M().eval(), (torch.randn(2, 8),)

    def layer_norm():
        return nn.LayerNorm(8).eval(), (torch.randn(2, 8),)

    def conv2d_1x1():
        return nn.Conv2d(3, 4, kernel_size=1, stride=1, bias=True).eval(), (
            torch.randn(1, 3, 5, 5),
        )

    def conv2d_3x3_stride2():
        return nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=True).eval(), (
            torch.randn(1, 3, 7, 7),
        )

    def maxpool2d():
        return nn.MaxPool2d(kernel_size=2, stride=2).eval(), (torch.randn(1, 3, 6, 6),)

    def transformer_encoder_layer():
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.TransformerEncoderLayer(
                    d_model=8,
                    nhead=2,
                    dim_feedforward=16,
                    dropout=0.0,
                    batch_first=True,
                    activation="relu",
                )

            def forward(self, x):
                return self.layer(x)

        return M().eval(), (torch.randn(1, 4, 8),)

    return [
        Case("reshape_flatten", reshape_flatten),
        Case("permute_3d", permute_3d),
        Case("strided_slice", strided_slice),
        Case("concat_dim1", concat_dim1),
        Case("gelu", gelu, atol=1e-3, xfail="codegen produces wrong results for erf/tanh gelu approximation"),
        Case("layer_norm", layer_norm, atol=1e-3),
        Case("conv2d_1x1", conv2d_1x1, atol=1e-3, xfail="pointwise conv1x1 codegen produces incorrect results"),
        Case("conv2d_3x3_stride2", conv2d_3x3_stride2, atol=1e-3),
        Case("maxpool2d", maxpool2d, atol=1e-3),
        Case("transformer_encoder_layer", transformer_encoder_layer, atol=1e-2, xfail="translation fails on multi-dispatch transformer pattern"),
    ]


def _export(case: Case, artifact_dir: Path, iree_compile: Path) -> tuple[Path | None, np.ndarray | None, list[str]]:
    """Export and compile. Returns (vmfb_path, expected_output, input_cli_args)."""
    import torch
    import iree.turbine.aot as aot

    case_dir = artifact_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    model, args = case.export_fn()

    with torch.no_grad():
        expected = model(*args)
    if isinstance(expected, torch.Tensor):
        expected_np = expected.numpy()
    else:
        expected_np = expected[0].numpy() if isinstance(expected, tuple) else expected.numpy()

    np.save(case_dir / "expected.npy", expected_np)

    input_cli = [f"--input={tensor_to_cli(a.numpy())}" for a in args]

    # Export to MLIR
    try:
        export_output = aot.export(model, args=args)
        export_output.save_mlir(str(case_dir / "torch_input.mlir"))
    except Exception as e:
        (case_dir / "export_error.txt").write_text(str(e))
        return None, expected_np, input_cli

    # Compile
    compile_cmd = [
        str(iree_compile),
        str(case_dir / "torch_input.mlir"),
        "--iree-hal-target-backends=cuda_tile",
        "--iree-cuda-tile-enable-codegen=true",
        f"--iree-cuda-tile-dump-kernel-plan-to={case_dir / 'kernel_plan.txt'}",
        f"--dump-compilation-phases-to={case_dir / 'phases'}",
        f"--iree-hal-dump-executable-files-to={case_dir / 'executables'}",
        f"-o={case_dir / 'module.vmfb'}",
    ]
    result = subprocess.run(compile_cmd, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (case_dir / "compile_log.txt").write_text(result.stdout)
    if result.returncode != 0:
        return None, expected_np, input_cli

    return case_dir / "module.vmfb", expected_np, input_cli


def _run(vmfb: Path, device: str, input_cli: list[str],
         iree_run_module: Path) -> np.ndarray | None:
    cmd = [str(iree_run_module), f"--device={device}",
           f"--module={vmfb}", "--function=main"] + input_cli
    result = subprocess.run(cmd, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=60)
    if result.returncode != 0:
        return None
    return parse_output(result.stdout)


def _run_case(case: Case, artifact_dir: Path, iree_compile: Path,
              iree_run_module: Path, compile_only: bool) -> Result:
    vmfb, expected_np, input_cli = _export(case, artifact_dir, iree_compile)

    if vmfb is None:
        if case.xfail:
            return Result(case.name, "XFAIL", f"compile: {case.xfail}")
        return Result(case.name, "FAIL", "compile failed")

    if compile_only:
        return Result(case.name, "PASS", "compile-only")

    # Run on cuda_tile
    tile_out = _run(vmfb, "cuda_tile", input_cli, iree_run_module)
    if tile_out is None:
        if case.xfail:
            return Result(case.name, "XFAIL", f"cuda_tile run: {case.xfail}")
        return Result(case.name, "FAIL", "cuda_tile run failed")

    # Run on cuda_new
    new_out = _run(vmfb, "cuda_new", input_cli, iree_run_module)
    if new_out is None:
        return Result(case.name, "FAIL", "cuda_new run failed")

    # Compare against expected
    if expected_np.shape != tile_out.shape:
        return Result(case.name, "FAIL",
                      f"shape mismatch: expected {expected_np.shape}, got {tile_out.shape}")

    max_diff_tile = float(np.max(np.abs(expected_np - tile_out)))
    max_diff_new = float(np.max(np.abs(expected_np - new_out)))
    tile_new_diff = float(np.max(np.abs(tile_out - new_out)))

    if max_diff_tile > case.atol:
        if case.xfail:
            return Result(case.name, "XFAIL",
                          f"correctness: {case.xfail} (diff={max_diff_tile:.2e})")
        return Result(case.name, "FAIL",
                      f"cuda_tile vs expected: max_diff={max_diff_tile:.6e} > atol={case.atol}")
    if tile_new_diff > 1e-6:
        return Result(case.name, "FAIL",
                      f"cuda_new vs cuda_tile: max_diff={tile_new_diff:.6e}")

    return Result(case.name, "PASS",
                  f"max_diff_tile={max_diff_tile:.2e} max_diff_new={max_diff_new:.2e}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path,
                        default=Path("/scratch/ashvin/merlin/build/host-merlin-release"))
    parser.add_argument("--artifact-dir", type=Path,
                        default=Path("/scratch/ashvin/merlin/compile/cudatile_cmp_step1"))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument("--extended", action="store_true",
                        help="Also run broader model/pattern discovery cases.")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)

    iree_compile = args.build_dir / "tools/iree-compile"
    iree_run_module = args.build_dir / "tools/iree-run-module"

    if not iree_compile.exists():
        print(f"missing iree-compile: {iree_compile}", file=sys.stderr)
        return 2

    all_cases = _define_cases()
    if args.extended:
        all_cases += _define_extended_cases()
    if args.cases:
        all_cases = [c for c in all_cases if c.name in args.cases]

    results: list[Result] = []
    for case in all_cases:
        try:
            r = _run_case(case, args.artifact_dir, iree_compile,
                          iree_run_module, args.compile_only)
        except Exception as e:
            r = Result(case.name, "FAIL", str(e))
        results.append(r)
        status_str = f"[{r.status}]"
        print(f"{status_str:8s} {r.name}  {r.detail}")
        if r.status == "FAIL" and not args.keep_going:
            break

    passes = sum(1 for r in results if r.status in ("PASS", "XFAIL"))
    fails = sum(1 for r in results if r.status == "FAIL")
    print(f"\n{passes} passed, {fails} failed out of {len(results)} cases")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
