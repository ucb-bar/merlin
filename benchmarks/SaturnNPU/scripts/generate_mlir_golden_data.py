#!/usr/bin/env python3
"""Generate golden test data by compiling and executing MLIR ops via IREE.

For each key op type found in the global-optimization MLIR, this script:
  1. Creates a standalone MLIR function wrapping the operation
  2. Compiles it with iree.compiler (target: llvm-cpu)
  3. Executes it with iree.runtime using random inputs
  4. Saves input/output as .npy golden data

This gives numerically correct golden data from the *actual MLIR compiler*,
not just PyTorch reference implementations.

Usage:
    python tools/generate_mlir_golden_data.py \
        --output-dir benchmarks/SaturnNPU/golden_data/mlir_level/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import iree.compiler as compiler
import iree.runtime as runtime
import numpy as np

SEED = 42


# ---------------------------------------------------------------------------
# MLIR op templates — standalone functions for each kernel type
# ---------------------------------------------------------------------------

MLIR_OPS = {
    "matmul_f32": {
        "description": "Matrix multiplication (f32)",
        "mlir": """
func.func @matmul(%arg0: tensor<{M}x{K}xf32>, %arg1: tensor<{K}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<{M}x{N}xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %result = linalg.matmul ins(%arg0, %arg1 : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%fill : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32> # noqa: E501
  return %result : tensor<{M}x{N}xf32>
}}
""",
        "shapes": [
            {"M": 50, "K": 720, "N": 720, "name": "gemma_self_proj"},
            {"M": 50, "K": 720, "N": 960, "name": "gemma_q_proj"},
            {"M": 50, "K": 720, "N": 2048, "name": "gemma_gate_up_proj"},
        ],
        "input_gen": lambda s: [
            np.random.randn(s["M"], s["K"]).astype(np.float32),
            np.random.randn(s["K"], s["N"]).astype(np.float32),
        ],
        "ref_fn": lambda inputs: inputs[0] @ inputs[1],
    },
    "batch_matmul_f32": {
        "description": "Batched matrix multiplication (f32, attention score path)",
        "mlir": """
func.func @batch_matmul(%arg0: tensor<{B}x{M}x{K}xf32>, %arg1: tensor<{B}x{K}x{N}xf32>) -> tensor<{B}x{M}x{N}xf32> {{
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<{B}x{M}x{N}xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<{B}x{M}x{N}xf32>) -> tensor<{B}x{M}x{N}xf32>
  %result = linalg.batch_matmul ins(%arg0, %arg1 : tensor<{B}x{M}x{K}xf32>, tensor<{B}x{K}x{N}xf32>) outs(%fill : tensor<{B}x{M}x{N}xf32>) -> tensor<{B}x{M}x{N}xf32> # noqa: E501
  return %result : tensor<{B}x{M}x{N}xf32>
}}
""",
        "shapes": [
            {"B": 15, "M": 291, "K": 64, "N": 291, "name": "gemma_self_attn_score"},
            {"B": 15, "M": 50, "K": 64, "N": 241, "name": "gemma_cross_attn_score"},
            {"B": 12, "M": 1024, "K": 64, "N": 1024, "name": "siglip_attn_score"},
        ],
        "input_gen": lambda s: [
            np.random.randn(s["B"], s["M"], s["K"]).astype(np.float32),
            np.random.randn(s["B"], s["K"], s["N"]).astype(np.float32),
        ],
        "ref_fn": lambda inputs: inputs[0] @ inputs[1],
    },
    "elementwise_add_f32": {
        "description": "Element-wise addition (f32, residual connections)",
        "mlir": """
func.func @add(%arg0: tensor<{S0}x{S1}xf32>, %arg1: tensor<{S0}x{S1}xf32>) -> tensor<{S0}x{S1}xf32> {{
  %empty = tensor.empty() : tensor<{S0}x{S1}xf32>
  %result = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%arg0, %arg1 : tensor<{S0}x{S1}xf32>, tensor<{S0}x{S1}xf32>) outs(%empty : tensor<{S0}x{S1}xf32>) {{ # noqa: E501
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %0 = arith.addf %in0, %in1 : f32
    linalg.yield %0 : f32
  }} -> tensor<{S0}x{S1}xf32>
  return %result : tensor<{S0}x{S1}xf32>
}}
""",
        "shapes": [
            {"S0": 50, "S1": 720, "name": "gemma_residual"},
            {"S0": 1024, "S1": 768, "name": "siglip_residual"},
        ],
        "input_gen": lambda s: [
            np.random.randn(s["S0"], s["S1"]).astype(np.float32),
            np.random.randn(s["S0"], s["S1"]).astype(np.float32),
        ],
        "ref_fn": lambda inputs: inputs[0] + inputs[1],
    },
    "elementwise_mul_f32": {
        "description": "Element-wise multiplication (f32, gating/scaling)",
        "mlir": """
func.func @mul(%arg0: tensor<{S0}x{S1}xf32>, %arg1: tensor<{S0}x{S1}xf32>) -> tensor<{S0}x{S1}xf32> {{
  %empty = tensor.empty() : tensor<{S0}x{S1}xf32>
  %result = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%arg0, %arg1 : tensor<{S0}x{S1}xf32>, tensor<{S0}x{S1}xf32>) outs(%empty : tensor<{S0}x{S1}xf32>) {{ # noqa: E501
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %0 = arith.mulf %in0, %in1 : f32
    linalg.yield %0 : f32
  }} -> tensor<{S0}x{S1}xf32>
  return %result : tensor<{S0}x{S1}xf32>
}}
""",
        "shapes": [
            {"S0": 50, "S1": 2048, "name": "gemma_gate_mul_up"},
        ],
        "input_gen": lambda s: [
            np.random.randn(s["S0"], s["S1"]).astype(np.float32),
            np.random.randn(s["S0"], s["S1"]).astype(np.float32),
        ],
        "ref_fn": lambda inputs: inputs[0] * inputs[1],
    },
    "gelu_tanh_f32": {
        "description": "GELU activation with tanh approximation (f32)",
        "mlir": """
func.func @gelu(%arg0: tensor<{S0}x{S1}xf32>) -> tensor<{S0}x{S1}xf32> {{
  %empty = tensor.empty() : tensor<{S0}x{S1}xf32>
  %result = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%arg0 : tensor<{S0}x{S1}xf32>) outs(%empty : tensor<{S0}x{S1}xf32>) {{ # noqa: E501
  ^bb0(%in: f32, %out: f32):
    %cst = arith.constant 0.797884583 : f32
    %cst2 = arith.constant 4.471500e-02 : f32
    %cst3 = arith.constant 5.000000e-01 : f32
    %cst4 = arith.constant 1.000000e+00 : f32
    %pow3 = arith.mulf %in, %in : f32
    %pow3b = arith.mulf %pow3, %in : f32
    %t1 = arith.mulf %cst2, %pow3b : f32
    %t2 = arith.addf %in, %t1 : f32
    %t3 = arith.mulf %cst, %t2 : f32
    %t4 = math.tanh %t3 : f32
    %t5 = arith.addf %cst4, %t4 : f32
    %t6 = arith.mulf %cst3, %t5 : f32
    %0 = arith.mulf %in, %t6 : f32
    linalg.yield %0 : f32
  }} -> tensor<{S0}x{S1}xf32>
  return %result : tensor<{S0}x{S1}xf32>
}}
""",
        "shapes": [
            {"S0": 1024, "S1": 3072, "name": "siglip_mlp_gelu"},
        ],
        "input_gen": lambda s: [np.random.randn(s["S0"], s["S1"]).astype(np.float32) * 0.5],
        "ref_fn": lambda inputs: inputs[0]
        * 0.5
        * (1.0 + np.tanh(0.7978845608 * (inputs[0] + 0.044715 * inputs[0] ** 3))),
    },
    "silu_f32": {
        "description": "SiLU/Swish activation (f32)",
        "mlir": """
func.func @silu(%arg0: tensor<{S0}x{S1}xf32>) -> tensor<{S0}x{S1}xf32> {{
  %empty = tensor.empty() : tensor<{S0}x{S1}xf32>
  %result = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%arg0 : tensor<{S0}x{S1}xf32>) outs(%empty : tensor<{S0}x{S1}xf32>) {{ # noqa: E501
  ^bb0(%in: f32, %out: f32):
    %neg = arith.negf %in : f32
    %exp = math.exp %neg : f32
    %cst = arith.constant 1.0 : f32
    %denom = arith.addf %cst, %exp : f32
    %0 = arith.divf %in, %denom : f32
    linalg.yield %0 : f32
  }} -> tensor<{S0}x{S1}xf32>
  return %result : tensor<{S0}x{S1}xf32>
}}
""",
        "shapes": [
            {"S0": 50, "S1": 2048, "name": "gemma_mlp_silu"},
        ],
        "input_gen": lambda s: [np.random.randn(s["S0"], s["S1"]).astype(np.float32)],
        "ref_fn": lambda inputs: inputs[0] / (1.0 + np.exp(-inputs[0])),
    },
}


def compile_and_run(mlir_text: str, func_name: str, inputs: list[np.ndarray]) -> np.ndarray:
    """Compile MLIR and execute with given inputs, return output."""
    vmfb = compiler.compile_str(mlir_text, target_backends=["llvm-cpu"])

    config = runtime.Config("local-task")
    ctx = runtime.SystemContext(config=config)
    vm_module = runtime.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)

    result = ctx.modules.module[func_name](*inputs)
    return np.array(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/SaturnNPU/golden_data/mlir_level"),
        help="Output directory for MLIR golden data",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    manifest = []

    for op_name, op_def in MLIR_OPS.items():
        print(f"\n=== {op_name}: {op_def['description']} ===")

        for shape_params in op_def["shapes"]:
            variant_name = shape_params["name"]
            shape_str = "_".join(f"{k}{v}" for k, v in shape_params.items() if k != "name")
            print(f"  {variant_name} ({shape_str})...")

            # Generate MLIR
            mlir_text = op_def["mlir"].format(**shape_params)

            # Get function name from MLIR
            import re

            func_match = re.search(r"@(\w+)\(", mlir_text)
            func_name = func_match.group(1) if func_match else "main"

            # Generate inputs
            inputs = op_def["input_gen"](shape_params)

            try:
                # Compile and execute via IREE
                iree_output = compile_and_run(mlir_text, func_name, inputs)

                # Compute reference
                ref_output = op_def["ref_fn"](inputs)

                # Compare
                max_diff = np.abs(iree_output - ref_output).max()
                print(f"    IREE output shape: {iree_output.shape}")
                print(f"    Max diff vs reference: {max_diff:.2e}")
                # Matmuls with large K dimension accumulate FP rounding errors
                tol = 1e-3 if "matmul" in op_name else 1e-4
                status = "PASS" if max_diff < tol else "WARN"
                print(f"    Status: {status}")

                # Save golden data
                out_dir = args.output_dir / f"{op_name}_{variant_name}"
                out_dir.mkdir(parents=True, exist_ok=True)

                for i, inp in enumerate(inputs):
                    np.save(out_dir / f"input_{i}.npy", inp)
                np.save(out_dir / "output_iree.npy", iree_output)
                np.save(out_dir / "output_reference.npy", ref_output)

                # Save metadata
                meta = {
                    "op_name": op_name,
                    "variant": variant_name,
                    "description": op_def["description"],
                    "shape_params": {k: v for k, v in shape_params.items() if k != "name"},
                    "input_shapes": [list(inp.shape) for inp in inputs],
                    "input_dtypes": [str(inp.dtype) for inp in inputs],
                    "output_shape": list(iree_output.shape),
                    "output_dtype": str(iree_output.dtype),
                    "max_diff_vs_reference": float(max_diff),
                    "status": status,
                }
                with open(out_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

                manifest.append(meta)

            except Exception as e:
                print(f"    ERROR: {e}")
                manifest.append(
                    {
                        "op_name": op_name,
                        "variant": variant_name,
                        "status": "ERROR",
                        "error": str(e),
                    }
                )

    # Save manifest
    manifest_path = args.output_dir / "mlir_golden_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    # Summary
    passed = sum(1 for m in manifest if m.get("status") == "PASS")
    total = len(manifest)
    print(f"\nResults: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
