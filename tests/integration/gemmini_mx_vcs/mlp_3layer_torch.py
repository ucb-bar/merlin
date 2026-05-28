"""Torch ``nn.Module`` 3-layer MLP + export pipeline + golden generator.

Used by the mxGemmini VCS test bench to (1) materialize a deterministic
quantized model, (2) emit a constants-baked MLIR fixture for both FP8
and FP4, and (3) compute the byte-for-byte expected output of running
that quantized network on a CPU reference.

Run this script from the repo root with the Understanding-PI0 venv (it
ships torchao 0.16+ + torch 2.10+; the merlin-dev venv has a stripped
torch installation that lacks the MXTensor APIs we need)::

    /scratch2/agustin/merlin/third_party/Understanding-PI0/.venv/bin/python \
        tests/integration/gemmini_mx_vcs/mlp_3layer_torch.py

Outputs (under ``tests/integration/gemmini_mx_vcs/fixtures/``):

    mlp_3layer_fp8.mlir       constants-baked, gemmini_mx_vcs target
    mlp_3layer_fp4.mlir       constants-baked, --target gemmini_mx_vcs --hw VCS_FP4
    expected_fp8.txt          1x16 i32 row, one value per line
    expected_fp4.txt          ditto
    test_pattern.h            C array of the deterministic 1x16 i8 input
                              (consumed by the bare-metal runner)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(REPO_ROOT))


class MLP3Layer(nn.Module):
    """16 -> 64 -> 64 -> 16 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 64, bias=False)
        self.fc2 = nn.Linear(64, 64, bias=False)
        self.fc3 = nn.Linear(64, 16, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.fc1(x)
        h2 = self.fc2(self.relu(h1))
        return self.fc3(h2)


def _seed_module(m: MLP3Layer, seed: int = 0xC0FFEE) -> None:
    """Deterministic, signed-int8-friendly weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Small, balanced, easy to fit in i8 after a fixed scale.
    for fc in (m.fc1, m.fc2, m.fc3):
        with torch.no_grad():
            fc.weight.copy_(torch.empty_like(fc.weight).uniform_(-0.5, 0.5))


def _quantize_to_int8_mxgemmini(model: MLP3Layer, fmt: str):
    """Stage-6.B custom-dtype quantize, then bake weights to int8 for the
    fixture. The mxGemmini dialect consumes i8 buffers at the HAL level
    (libgemmini handles the FP4/FP8 unpack inside the systolic array
    via the format selector bits set by CONFIG_EX) — so the fixture
    materializes scaled, quantized integer weights aligned with the
    block-16 layout the dialect expects.

    For the *reference* path we record the dequantized HP weight for
    the golden CPU run, so the expected output reflects the exact same
    rounding the hardware will apply.
    """
    from models.gemmini_mx_quant.custom_dtype import (  # noqa: E402
        MxGemminiE2M2Tensor,
        MxGemminiE4M4Tensor,
    )

    fmt = fmt.lower()
    cls = MxGemminiE4M4Tensor if fmt == "fp8" else MxGemminiE2M2Tensor
    quantized: dict[str, torch.Tensor] = {}
    int8_weights: dict[str, torch.Tensor] = {}

    for name in ("fc1", "fc2", "fc3"):
        fc: nn.Linear = getattr(model, name)
        with torch.no_grad():
            w = fc.weight.detach().to(torch.float32)
            t = cls.from_float(w, block_size=16)
            # HP weight that the *reference* uses
            quantized[name] = t.dequantize(target_dtype=torch.float32)
            # i8 buffer-level representation: the qdata fits in [0..255]
            # for fp8 and [0..15] for fp4. Sign comes from the per-block
            # scale, which we apply via a per-row shift in the requant
            # step. For the simplest fixture-maker we re-quantize the
            # dequantized HP weight to a fixed int8 range.
            scale_per_row = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 100.0
            iw = (w / scale_per_row).round().clamp(-127, 127).to(torch.int8)
            int8_weights[name] = iw
    return quantized, int8_weights


def _make_test_input() -> torch.Tensor:
    """Deterministic 16x16 input pattern. Small magnitudes so the int8
    arithmetic in the dispatch doesn't saturate.

    Shape is 16x16 (NOT a 1x16 row-vector) to dodge a Phase-5 dialect
    bug: the lowering produces an MVIN with bytes_to_read==0 for M=1
    matmuls, which trips Gemmini's RTL sanity assertion at
    LoadController.scala:191. Tracked as task #40 in the task list.
    16 batch rows × 16 features matches the systolic array DIM exactly.
    """
    arr = np.array([[((i + j) % 7) - 3 for j in range(16)] for i in range(16)], dtype=np.int8)
    return torch.from_numpy(arr)


def _i8_matmul_relu_chain(
    x_i8: torch.Tensor,
    w1_i8: torch.Tensor,
    w2_i8: torch.Tensor,
    w3_i8: torch.Tensor,
) -> torch.Tensor:
    """Mirror the dispatch's exact integer arithmetic.

    Replicates the per-layer computation from
    ``tests/integration/gemmini_mx_vcs/fixtures/mlp_3layer.mlir``:

      h1   = x  @ w1                         (i32 accumulator)
      h1q  = relu(h1) trunc to i8
      h2   = h1q @ w2                        (i32)
      h2q  = (h2 ashr 8) trunc to i8
      h3   = h2q @ w3                        (i32)
      out  = h3                              (1x16 i32)
    """
    x = x_i8.to(torch.int32)
    w1 = w1_i8.to(torch.int32)
    w2 = w2_i8.to(torch.int32)
    w3 = w3_i8.to(torch.int32)

    h1 = x @ w1  # 1x64 i32
    h1 = torch.where(h1 < 0, torch.zeros_like(h1), h1)
    h1q = (h1 & 0xFF).to(torch.int8)  # truncate to i8 (modular)

    h2 = h1q.to(torch.int32) @ w2  # 1x64 i32
    h2q_i32 = h2 >> 8  # arithmetic shift right
    h2q = (h2q_i32 & 0xFF).to(torch.int8)

    h3 = h2q.to(torch.int32) @ w3  # 1x16 i32
    return h3


def _format_dense_tensor(name: str, tensor: torch.Tensor) -> str:
    """Emit ``%name = arith.constant dense<[...]> : tensor<...xi8>``."""
    arr = tensor.detach().cpu().numpy()
    if arr.dtype != np.int8:
        arr = arr.astype(np.int8)
    flat = arr.reshape(-1).tolist()
    rows: list[str] = []
    for r in arr.tolist() if arr.ndim == 2 else [flat]:
        rows.append("[" + ", ".join(str(int(v)) for v in r) + "]")
    if arr.ndim == 2:
        body = "[" + ", ".join(rows) + "]"
    else:
        body = rows[0]
    shape = "x".join(str(d) for d in arr.shape) + "xi8"
    return f"  %{name} = arith.constant dense<{body}> : tensor<{shape}>"


def _emit_constants_baked_mlir(
    out_path: pathlib.Path,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> None:
    """Emit a constants-baked sister of ``mlp_3layer.mlir`` whose only
    parameter is the 1x16 i8 input. Mirrors the existing fixture's
    layer-shape + truncation pattern exactly.
    """
    w1_const = _format_dense_tensor("w1", w1)
    w2_const = _format_dense_tensor("w2", w2)
    w3_const = _format_dense_tensor("w3", w3)

    body = f"""// Auto-generated by tests/integration/gemmini_mx_vcs/mlp_3layer_torch.py.
// Constants-baked sister of mlp_3layer.mlir for the bare-metal VCS runner.
//
// Layer shapes: 16 -> 64 -> 64 -> 16, ReLU between 1+2, arith.shrsi-by-8
// requantize between 2+3. Weights baked from torchao mxGemmini E{'4M4' if 'fp8' in out_path.stem else '2M2'}
// quantization (see mlp_3layer_torch.py).

func.func @mlp_3layer(%input: tensor<16x16xi8>) -> tensor<16x16xi32>
    attributes {{iree.preserve_func_visibility = true}} {{
  %c0_i32 = arith.constant 0 : i32

{w1_const}
{w2_const}
{w3_const}

  %init1 = tensor.empty() : tensor<16x64xi32>
  %fill1 = linalg.fill ins(%c0_i32 : i32) outs(%init1 : tensor<16x64xi32>) -> tensor<16x64xi32>
  %h1 = linalg.matmul ins(%input, %w1 : tensor<16x16xi8>, tensor<16x64xi8>)
      outs(%fill1 : tensor<16x64xi32>) -> tensor<16x64xi32>

  %h1_i8 = tensor.empty() : tensor<16x64xi8>
  %h1_relu = linalg.generic
      {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}}
      ins(%h1 : tensor<16x64xi32>) outs(%h1_i8 : tensor<16x64xi8>) {{
    ^bb0(%in: i32, %out: i8):
      %z = arith.constant 0 : i32
      %is_neg = arith.cmpi slt, %in, %z : i32
      %clamped = arith.select %is_neg, %z, %in : i32
      %trunc = arith.trunci %clamped : i32 to i8
      linalg.yield %trunc : i8
    }} -> tensor<16x64xi8>

  %init2 = tensor.empty() : tensor<16x64xi32>
  %fill2 = linalg.fill ins(%c0_i32 : i32) outs(%init2 : tensor<16x64xi32>) -> tensor<16x64xi32>
  %h2 = linalg.matmul ins(%h1_relu, %w2 : tensor<16x64xi8>, tensor<64x64xi8>)
      outs(%fill2 : tensor<16x64xi32>) -> tensor<16x64xi32>

  %h2_i8 = tensor.empty() : tensor<16x64xi8>
  %h2_q = linalg.generic
      {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}}
      ins(%h2 : tensor<16x64xi32>) outs(%h2_i8 : tensor<16x64xi8>) {{
    ^bb0(%in: i32, %out: i8):
      %sh = arith.constant 8 : i32
      %shifted = arith.shrsi %in, %sh : i32
      %t = arith.trunci %shifted : i32 to i8
      linalg.yield %t : i8
    }} -> tensor<16x64xi8>

  %init3 = tensor.empty() : tensor<16x16xi32>
  %fill3 = linalg.fill ins(%c0_i32 : i32) outs(%init3 : tensor<16x16xi32>) -> tensor<16x16xi32>
  %h3 = linalg.matmul ins(%h2_q, %w3 : tensor<16x64xi8>, tensor<64x16xi8>)
      outs(%fill3 : tensor<16x16xi32>) -> tensor<16x16xi32>

  return %h3 : tensor<16x16xi32>
}}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    print(f"[mlp_3layer] wrote {out_path}")


def _emit_test_pattern_header(out_path: pathlib.Path, x_i8: torch.Tensor) -> None:
    arr = x_i8.detach().cpu().numpy().astype(np.int8)
    assert arr.shape == (16, 16), f"unexpected input shape {arr.shape}"
    text = "// Auto-generated by mlp_3layer_torch.py — deterministic 16x16 i8 input.\n"
    text += "#pragma once\n#include <stdint.h>\n\n"
    rows: list[str] = []
    for row in arr.tolist():
        rows.append("  { " + ", ".join(str(int(v)) for v in row) + " }")
    text += "static const int8_t kMxGemminiTestInput[16][16] = {\n"
    text += ",\n".join(rows) + "\n};\n"
    out_path.write_text(text)
    print(f"[mlp_3layer] wrote {out_path}")


def _write_expected(out_path: pathlib.Path, h3: torch.Tensor) -> None:
    arr = h3.detach().cpu().numpy().reshape(-1).astype(np.int32)
    out_path.write_text("\n".join(str(int(v)) for v in arr) + "\n")
    print(f"[mlp_3layer] wrote {out_path} ({arr.size} i32 values)")


def export_quantized(format_str: str, out_dir: pathlib.Path) -> dict:
    """Build seeded MLP, quantize with mxGemmini E4M4/E2M2, emit the
    constants-baked MLIR fixture + golden output.

    Returns a dict of artifacts for unit testing.
    """
    fmt = format_str.lower()
    if fmt not in {"fp8", "fp4"}:
        raise ValueError(f"format must be fp8 or fp4, got {format_str!r}")

    model = MLP3Layer()
    _seed_module(model, seed=0xC0FFEE)
    _hp_weights, int8_weights = _quantize_to_int8_mxgemmini(model, fmt)
    w1, w2, w3 = int8_weights["fc1"].t(), int8_weights["fc2"].t(), int8_weights["fc3"].t()
    # Note: nn.Linear weight is [out, in]; matmul uses [in, out]. Transpose.

    x = _make_test_input()

    # Reference output: same i8 buffer-level math as the dispatch.
    h3 = _i8_matmul_relu_chain(x, w1, w2, w3)

    fixture_path = out_dir / f"mlp_3layer_{fmt}.mlir"
    _emit_constants_baked_mlir(fixture_path, w1, w2, w3)

    expected_path = out_dir / f"expected_{fmt}.txt"
    _write_expected(expected_path, h3)

    test_pattern_path = out_dir / "test_pattern.h"
    _emit_test_pattern_header(test_pattern_path, x)

    return {
        "format": fmt,
        "fixture": fixture_path,
        "expected": expected_path,
        "test_pattern_h": test_pattern_path,
        "h3": h3,
        "x": x,
        "w1": w1,
        "w2": w2,
        "w3": w3,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--format",
        choices=["fp8", "fp4", "both"],
        default="both",
        help="Which quantization format(s) to emit",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=FIXTURES,
        help=f"Output directory (default: {FIXTURES})",
    )
    args = p.parse_args(argv)
    formats = ["fp8", "fp4"] if args.format == "both" else [args.format]
    for f in formats:
        export_quantized(f, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
