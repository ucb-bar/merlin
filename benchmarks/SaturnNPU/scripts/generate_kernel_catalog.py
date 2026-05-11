#!/usr/bin/env python3
"""Generate a per-kernel catalog with standalone MLIR files and README.

For each kernel type we need to implement, creates:
  kernels/<kernel_type>/
    README.md          — description, shapes, PyTorch origin, how to test
    <variant>.mlir     — standalone compilable MLIR for each shape variant
    golden/            — .npy golden data (input/output) per variant

This gives kernel developers everything in one place: the MLIR to implement
against, the golden data to verify, and context about what it represents.

Usage:
    python tools/generate_kernel_catalog.py \
        --linalg-input build/compiled_models/smolVLA/.../phases/module.1.input.mlir \
        --global-opt build/compiled_models/smolVLA/.../phases/module.4.global-optimization.mlir \
        --output-dir benchmarks/SaturnNPU/kernels/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    HAS_IREE = True
except ImportError:
    HAS_IREE = False


SEED = 42


# ---------------------------------------------------------------------------
# MLIR snippet extraction
# ---------------------------------------------------------------------------


def _extract_snippet(lines: list[str], start: int, max_lines: int = 20) -> str:
    """Extract a clean MLIR snippet starting at `start`, up to linalg.yield or max_lines."""
    snippet_lines = []
    for j in range(start, min(start + max_lines, len(lines))):
        snippet_lines.append(lines[j].rstrip())
        if "linalg.yield" in lnines[j] or (j > start and lines[j].strip().startswith("}")):
            # grab the closing brace line too
            if j + 1 < len(lines) and lines[j + 1].strip().startswith("}"):
                snippet_lines.append(lines[j + 1].rstrip())
            break
    return "\n".join(snippet_lines)


def extract_representative_snippets(lines: list[str], classify_fn, op_re) -> dict[str, dict]:
    """Extract one representative MLIR snippet per classified op type.

    Returns {classification: {"snippet": str, "line": int, "shapes": list[str]}}
    """
    seen: dict[str, dict] = {}
    tensor_re = re.compile(r"tensor<([^>]+)>")

    for i, line in enumerate(lines):
        if op_re.search(line):
            op_name = op_re.search(line).group(1)
            if op_name == "linalg.generic":
                cls = classify_fn(lines, i)
            elif op_name == "linalg.batch_matmul":
                cls = "batch_matmul"
            elif op_name == "linalg.matmul":
                cls = "matmul"
            elif op_name == "linalg.softmax":
                cls = "softmax"
            elif op_name == "iree_linalg_ext.attention":
                cls = "fused_attention"
            elif op_name == "linalg.transpose":
                cls = "transpose"
            elif op_name == "linalg.fill":
                cls = "fill"
            else:
                cls = op_name

            shapes = tensor_re.findall(line)

            # For certain ops, prefer specific shape patterns for the representative
            if cls in seen:
                # For quantized_matmul_fp8, prefer snippets with f8E4M3FN in shapes
                if cls == "quantized_matmul_fp8" and "f8E4M3FN" not in str(seen[cls]["shapes"]):
                    if "f8E4M3FN" in str(shapes):
                        seen[cls] = {
                            "snippet": _extract_snippet(lines, i),
                            "line": i + 1,
                            "shapes": shapes[:4],
                        }
                continue

            seen[cls] = {
                "snippet": _extract_snippet(lines, i),
                "line": i + 1,
                "shapes": shapes[:4],
            }

    return seen


# ---------------------------------------------------------------------------
# Kernel type metadata
# ---------------------------------------------------------------------------

KERNEL_INFO = {
    "quantized_matmul_fp8": {
        "name": "Quantized MatMul (FP8)",
        "pytorch": "torch.aten.linear (with MX fp8 quantized weights)",
        "description": (
            "Matrix multiplication where weights are stored in fp8 (E4M3FN) format "
            "with block-wise bf16 scaling factors. The body dequantizes the weight "
            "inline: extf(fp8→bf16), multiply by scale, then multiply-accumulate "
            "with bf16 activation. This is 82.2% of total model compute."
        ),
        "body_pattern": "arith.extf(fp8→bf16) → arith.mulf(scale) → arith.mulf(act×wt) → arith.addf(accum)",
        "iterator_types": '["parallel", "parallel", "parallel", "reduction", "reduction"]',
    },
    "batch_matmul": {
        "name": "Batched MatMul (bf16/f32)",
        "pytorch": "torch.matmul inside attention (Q@K^T and attn@V)",
        "description": (
            "Batched matrix multiplication for attention score and value paths. "
            "Shape: [B, M, K] × [B, K, N] → [B, M, N] where B=num_heads."
        ),
    },
    "matmul": {
        "name": "MatMul (various dtypes)",
        "pytorch": "torch.aten.linear / torch.aten.mm",
        "description": "Standard 2D matrix multiplication. Used for projection layers.",
    },
    "fused_attention": {
        "name": "Fused Scaled Dot-Product Attention",
        "pytorch": "torch.aten.scaled_dot_product_attention",
        "description": (
            "Fused SDPA: computes Q@K^T, scales, applies mask, softmax, then @V. "
            "All in bf16. Shape: [H, S, D] for query/key/value."
        ),
    },
    "softmax": {
        "name": "Softmax",
        "pytorch": "torch.aten.softmax / F.softmax",
        "description": "Named linalg.softmax. Reduces along specified dimension.",
    },
    "rms_norm": {
        "name": "RMS Normalization",
        "pytorch": "GemmaRMSNorm / torch.aten.layer_norm",
        "description": (
            "Reciprocal square root normalization: rsqrt(mean(x²) + eps). "
            "Always appears as part of a larger norm sequence: "
            "reduce_sum → div → rsqrt → mul(scale)."
        ),
        "body_pattern": "math.rsqrt",
    },
    "gelu_tanh": {
        "name": "GELU (tanh approximation)",
        "pytorch": "torch.aten.gelu / F.gelu(approximate='tanh')",
        "description": "GELU activation using tanh approximation. Body: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³))).",  # noqa: E501
        "body_pattern": "math.tanh",
    },
    "silu": {
        "name": "SiLU / Swish",
        "pytorch": "torch.aten.silu",
        "description": "SiLU activation: x / (1 + exp(-x)). Used in Gemma MLP.",
        "body_pattern": "arith.negf → math.exp → arith.addf(1) → arith.divf",
    },
    "int8_quantize": {
        "name": "Dynamic Int8 Quantization",
        "pytorch": "Dynamic activation quantization (bf16 → i8)",
        "description": (
            "Quantizes bf16 activations to int8 with per-token scale and zero-point. "
            "Body: divf(scale) → addf(zp) → clamp → fptosi → trunci."
        ),
        "body_pattern": "arith.divf → arith.fptosi → arith.trunci",
    },
    "elementwise_add": {
        "name": "Element-wise Addition",
        "pytorch": "Residual connections, bias add",
        "description": "Parallel addition. No reduction dimensions.",
    },
    "elementwise_mul": {
        "name": "Element-wise Multiplication",
        "pytorch": "Scaling, gating (gate * up in MLP)",
        "description": "Parallel multiplication. Used for GeGLU combine and RMSNorm scaling.",
    },
    "elementwise_div": {
        "name": "Element-wise Division",
        "pytorch": "Normalization division (x / variance)",
        "description": "Parallel division. Part of norm computation.",
    },
    "elementwise_sub": {
        "name": "Element-wise Subtraction",
        "pytorch": "Centering (x - mean), softmax shift (x - max)",
        "description": "Parallel subtraction.",
    },
    "reduction_sum": {
        "name": "Sum Reduction",
        "pytorch": "Variance computation in norms",
        "description": "Reduces along one dimension by summing. Part of LayerNorm/RMSNorm.",
    },
    "type_conversion": {
        "name": "Type Conversion",
        "pytorch": "Precision casts (bf16 ↔ f32)",
        "description": "arith.extf or arith.truncf for precision conversion.",
    },
    "rope_frequency": {
        "name": "RoPE Frequency Computation",
        "pytorch": "GemmaRotaryEmbedding",
        "description": "Computes inverse frequencies for rotary position embedding. Body: math.fpowi.",
    },
    "bias_add_cast": {
        "name": "Bias Add with Cast",
        "pytorch": "Part of LayerNorm (add bias + truncate)",
        "description": "arith.addf + arith.truncf — adds bias then casts precision.",
    },
}


# ---------------------------------------------------------------------------
# Standalone MLIR generation
# ---------------------------------------------------------------------------


def make_standalone_mlir(cls: str, snippet_data: dict) -> str | None:
    """Create a standalone compilable MLIR function from a snippet.

    Returns None if the snippet can't be made standalone.
    """
    snippet = snippet_data["snippet"]
    shapes = snippet_data["shapes"]

    # For simple linalg.generic ops, wrap in a function
    if "linalg.generic" in snippet:
        # Extract tensor types from the snippet
        tensor_types = re.findall(r"tensor<[^>]+>", snippet)
        if not tensor_types:
            return None

        # Get unique input and output types
        # The pattern is: ins(...) outs(...) → result
        result_match = re.search(r"-> (tensor<[^>]+>)", snippet)

        if not result_match:
            return None

        # Build a minimal function
        return f"// Kernel: {cls}\n// Extracted from SmolVLA global-optimization IR\n\n{snippet}\n"

    # For named ops
    if "linalg.batch_matmul" in snippet:
        if len(shapes) >= 3:
            return f"// Kernel: batch_matmul\n" f"// Shape: {shapes[0]} × {shapes[1]} → {shapes[2]}\n\n" f"{snippet}\n"

    if "iree_linalg_ext.attention" in snippet:
        return f"// Kernel: fused_attention (SDPA)\n\n{snippet}\n"

    if "linalg.softmax" in snippet:
        return f"// Kernel: softmax\n\n{snippet}\n"

    return f"// Kernel: {cls}\n\n{snippet}\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_npu_graph import (
        _LINALG_KEY_OPS,
        _classify_generic_body,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linalg-input",
        type=Path,
        default=Path("build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases/module.1.input.mlir"),
    )
    parser.add_argument(
        "--global-opt",
        type=Path,
        default=Path(
            "build/compiled_models/smolVLA/gemmini_mx_RVV_smolVLA.q.fp8/phases/module.4.global-optimization.mlir"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/SaturnNPU/kernels"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read files
    print("Reading MLIR files...")
    input_lines = args.linalg_input.read_text(encoding="utf-8").splitlines()
    gopt_lines = args.global_opt.read_text(encoding="utf-8").splitlines()

    # Find global-opt function body start
    gopt_func_start = 0
    for i, gl in enumerate(gopt_lines):
        if "@main" in gl and "func" in gl and "async" in gl.lower():
            gopt_func_start = i
            break

    # Extract snippets from both levels
    print("Extracting MLIR snippets...")
    input_snippets = extract_representative_snippets(input_lines, _classify_generic_body, _LINALG_KEY_OPS)
    gopt_snippets = extract_representative_snippets(
        gopt_lines[gopt_func_start:], _classify_generic_body, _LINALG_KEY_OPS
    )
    # Adjust line numbers for global-opt
    for cls, data in gopt_snippets.items():
        data["line"] += gopt_func_start

    print(f"  Input-level snippets: {len(input_snippets)} types")
    print(f"  Global-opt snippets: {len(gopt_snippets)} types")

    # Generate per-kernel directories
    all_types = sorted(set(list(input_snippets.keys()) + list(gopt_snippets.keys())))

    for cls in all_types:
        info = KERNEL_INFO.get(cls, {})
        if not info:
            continue  # Skip types we don't have metadata for

        kernel_dir = args.output_dir / cls
        kernel_dir.mkdir(parents=True, exist_ok=True)

        # Write MLIR snippets
        if cls in input_snippets:
            snippet_path = kernel_dir / "input_level.mlir"
            snippet_path.write_text(
                f"// Kernel: {cls} — {info.get('name', cls)}\n"
                f"// Source: module.1.input.mlir, line {input_snippets[cls]['line']}\n"
                f"// Shapes: {input_snippets[cls]['shapes']}\n"
                f"// PyTorch: {info.get('pytorch', '?')}\n\n"
                f"{input_snippets[cls]['snippet']}\n"
            )

        if cls in gopt_snippets:
            snippet_path = kernel_dir / "global_opt_level.mlir"
            snippet_path.write_text(
                f"// Kernel: {cls} — {info.get('name', cls)}\n"
                f"// Source: module.4.global-optimization.mlir, line {gopt_snippets[cls]['line']}\n"
                f"// Shapes: {gopt_snippets[cls]['shapes']}\n"
                f"// This is the level kernel writers should implement against.\n\n"
                f"{gopt_snippets[cls]['snippet']}\n"
            )

        # Write README
        readme_lines = [
            f"# `{cls}` — {info.get('name', cls)}",
            "",
            f"**PyTorch origin**: {info.get('pytorch', 'N/A')}",
            "",
            f"**Description**: {info.get('description', '')}",
            "",
        ]

        if "body_pattern" in info:
            readme_lines.extend(
                [
                    f"**Body pattern**: `{info['body_pattern']}`",
                    "",
                ]
            )

        if cls in input_snippets:
            readme_lines.extend(
                [
                    "## Input Level MLIR (`module.1.input.mlir`)",
                    "",
                    f"Line {input_snippets[cls]['line']}, shapes: `{input_snippets[cls]['shapes']}`",
                    "",
                    "```mlir",
                    input_snippets[cls]["snippet"],
                    "```",
                    "",
                ]
            )

        if cls in gopt_snippets:
            readme_lines.extend(
                [
                    "## Global-Opt Level MLIR (`module.4.global-optimization.mlir`)",
                    "",
                    "This is the level to implement against. Dequant chains are hoisted, transposes eliminated.",
                    "",
                    f"Line {gopt_snippets[cls]['line']}, shapes: `{gopt_snippets[cls]['shapes']}`",
                    "",
                    "```mlir",
                    gopt_snippets[cls]["snippet"],
                    "```",
                    "",
                ]
            )

        readme_lines.extend(
            [
                "## Files",
                "",
                "- `input_level.mlir` — MLIR from `module.1.input.mlir`",
                "- `global_opt_level.mlir` — MLIR from `module.4.global-optimization.mlir` (implement against this)",
                "",
            ]
        )

        (kernel_dir / "README.md").write_text("\n".join(readme_lines))
        print(f"  {cls}/")

    # Write top-level index
    index_lines = [
        "# SmolVLA Kernel Catalog",
        "",
        "Each subdirectory contains standalone MLIR files and a README for one kernel type.",
        "",
        "| Kernel | Description | PyTorch Origin |",
        "| --- | --- | --- |",
    ]
    for cls in all_types:
        info = KERNEL_INFO.get(cls, {})
        if info:
            index_lines.append(f"| [`{cls}`]({cls}/) | {info.get('name', cls)} | {info.get('pytorch', '')} |")
    index_lines.append("")
    (args.output_dir / "README.md").write_text("\n".join(index_lines))

    print(f"\nKernel catalog: {args.output_dir}/")
    print(f"  {len([t for t in all_types if t in KERNEL_INFO])} kernel types with MLIR + README")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
