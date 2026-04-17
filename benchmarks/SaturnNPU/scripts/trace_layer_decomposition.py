#!/usr/bin/env python3
"""Trace PyTorch layer decomposition through all MLIR compilation levels.

Uses MLIR Python bindings to walk the IR at each level and map ops back
to their PyTorch origins. Produces a detailed per-layer trace showing:

  PyTorch layer → Torch-MLIR ops → Linalg/Input ops → Global-Opt ops

For each op: name, shape, body pattern (for generics), and PyTorch role.

Usage:
    python scripts/trace_layer_decomposition.py \
        --linalg-input .../phases/module.1.input.mlir \
        --global-opt .../phases/module.4.global-optimization.mlir \
        --output-dir benchmarks/SaturnNPU/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import iree.compiler.ir as ir

# ---------------------------------------------------------------------------
# Generic body classifier (using MLIR bindings, not regex)
# ---------------------------------------------------------------------------


def classify_generic(op) -> str:
    """Classify a linalg.generic op by inspecting its region body ops."""
    body_ops = set()
    it_str = str(op.attributes["iterator_types"] if "iterator_types" in op.attributes else "")
    has_reduction = "reduction" in it_str

    for region in op.regions:
        for block in region:
            for body_op in block:
                body_ops.add(body_op.name)

    body = body_ops - {"linalg.yield"}

    if "math.rsqrt" in body:
        return "rms_norm"
    if "math.tanh" in body:
        return "gelu_tanh"
    if "math.exp" in body and "arith.negf" in body:
        return "silu"
    if "math.exp" in body and not has_reduction:
        return "softmax_exp"
    if "math.fpowi" in body:
        return "rope_frequency"
    if "math.sin" in body:
        return "rope_sin"
    if "math.cos" in body:
        return "rope_cos"
    if "math.powf" in body:
        return "rope_base"
    if "arith.extf" in body and "arith.mulf" in body and "arith.addf" in body and has_reduction:
        return "quantized_matmul_fp8"
    if "arith.shli" in body:
        return "dequant_bitshift"
    if "arith.fptosi" in body and "arith.trunci" in body:
        return "int8_quantize"
    if "arith.extui" in body and not has_reduction:
        return "dequant_extend_uint"
    if "arith.cmpi" in body and "arith.extsi" in body:
        return "dequant_nan_detect"
    if "arith.select" in body and "arith.cmpf" in body:
        return "conditional_select"
    if "arith.select" in body and not has_reduction:
        return "mask_select"
    if "arith.maximumf" in body and has_reduction:
        return "softmax_max_reduce"
    if "arith.sitofp" in body and "arith.mulf" in body:
        return "rope_inv_freq"
    if "arith.sitofp" in body and len(body) == 1:
        return "int_to_float"
    if "arith.subf" in body and len(body) == 1:
        return "elementwise_sub"
    if "arith.andi" in body:
        return "mask_and"
    if "arith.truncf" in body and "arith.addf" in body:
        return "bias_add_cast"
    if "arith.truncf" in body and "arith.mulf" in body:
        return "scale_mul_cast"
    if "arith.extf" in body and "arith.mulf" in body:
        return "scale_mul_extend"
    if "tensor.extract" in body:
        return "gather_lookup"
    if "linalg.index" in body and "arith.index_cast" in body:
        return "index_generation"
    if body == {"arith.addf"} and not has_reduction:
        return "elementwise_add"
    if body == {"arith.mulf"} and not has_reduction:
        return "elementwise_mul"
    if body == {"arith.divf"} and not has_reduction:
        return "elementwise_div"
    if body == {"arith.addf"} and has_reduction:
        return "reduction_sum"
    if body == {"arith.mulf", "arith.addf"} and has_reduction:
        return "dot_product"
    if body <= {"arith.extf"}:
        return "type_conversion"
    if body <= {"arith.truncf"}:
        return "type_conversion"
    return "other"


# ---------------------------------------------------------------------------
# Parse a file into a classified op sequence
# ---------------------------------------------------------------------------


def parse_mlir_ops(path: Path) -> list[dict]:
    """Parse an MLIR file and return classified op sequence for main$async body."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True

    print(f"  Parsing {path.name}...")
    with open(path) as f:
        module = ir.Module.parse(f.read(), context=ctx)

    # Find main$async
    main_func = None
    for op in module.body:
        if "sym_name" in op.attributes:
            sym = str(op.attributes["sym_name"]).strip('"')
            if "main" in sym and "async" in sym:
                main_func = op
                break

    if main_func is None:
        print(f"    WARNING: no main$async found in {path.name}")
        return []

    block = list(main_func.regions[0])[0]

    ops = []
    for child in block:
        name = child.name
        result_types = [str(r.type) for r in child.results]
        operand_types = [str(o.type) for o in child.operands]

        cls = ""
        mlir_text = ""
        if name == "linalg.generic":
            cls = classify_generic(child)
            # Extract the MLIR text for key ops
            if cls in (
                "quantized_matmul_fp8",
                "rms_norm",
                "gelu_tanh",
                "silu",
                "int8_quantize",
                "softmax_exp",
                "softmax_max_reduce",
            ):
                mlir_text = str(child)[:500]

        elif name in (
            "iree_linalg_ext.attention",
            "linalg.batch_matmul",
            "linalg.matmul",
            "linalg.softmax",
        ):
            mlir_text = str(child)[:500]

        ops.append(
            {
                "name": name,
                "cls": cls,
                "result_types": result_types,
                "operand_types": operand_types,
                "mlir_text": mlir_text,
            }
        )

    print(f"    {len(ops)} ops in function body")
    return ops


# ---------------------------------------------------------------------------
# Identify layer boundaries and assign PyTorch roles
# ---------------------------------------------------------------------------

# Map of dequant-related classifications to skip in high-level view
DEQUANT_OPS = {
    "dequant_bitshift",
    "dequant_extend_uint",
    "dequant_nan_detect",
    "conditional_select",
    "mask_select",
}

INFRA_OPS = {
    "tensor.empty",
    "tensor.expand_shape",
    "tensor.collapse_shape",
    "arith.constant",
    "tensor.extract_slice",
    "tensor.insert_slice",
    "util.global.load",
    "hal.tensor.barrier",
    "hal.tensor.import",
    "hal.tensor.export",
    "linalg.fill",
    "hal.fence.create",
    "hal.fence.await",
    "util.call",
    "util.return",
    "iree_tensor_ext.dispatch_tensor.load",
    "iree_tensor_ext.dispatch_tensor.store",
    "iree_tensor_ext.compute_barrier.begin",
    "iree_tensor_ext.compute_barrier.end",
}


def build_layer_trace(ops: list[dict], level_name: str) -> list[dict]:
    """Build a per-layer decomposition from the classified op sequence.

    Returns a list of layer dicts, each with the ops assigned to it.
    """
    # Find key markers
    attn_idx = [i for i, o in enumerate(ops) if o["name"] == "iree_linalg_ext.attention"]
    _ = [i for i, o in enumerate(ops) if o["name"] == "linalg.batch_matmul"]
    _ = [i for i, o in enumerate(ops) if o["name"] == "linalg.matmul"]
    _ = [i for i, o in enumerate(ops) if o["name"] == "linalg.softmax"]
    _ = [i for i, o in enumerate(ops) if o["cls"] == "gelu_tanh"]
    silu_idx = [i for i, o in enumerate(ops) if o["cls"] == "silu"]

    layers = []

    # ----- SigLIP Encoder Layers -----
    # Each attention = 1 SigLIP layer's attention block
    # Each gelu between two attentions = that layer's MLP block
    for layer_i, ai in enumerate(attn_idx):
        # Find this layer's start (go back to find norm pattern)
        start = max(0, ai - 80)
        # Find end: next attention or next major section
        end = attn_idx[layer_i + 1] if layer_i + 1 < len(attn_idx) else ai + 80

        # Find gelu in this range (MLP activation)

        # Collect compute ops (skip infrastructure)
        compute_ops = []
        qm_count = 0
        for i in range(start, min(end, len(ops))):
            op = ops[i]
            if op["name"] in INFRA_OPS or op["name"] == "linalg.transpose":
                continue

            cls = op["cls"] or op["name"]

            # Skip dequant chain in high-level view (note it for context)
            if cls in DEQUANT_OPS:
                continue

            # Assign PyTorch role
            role = ""
            if op["name"] == "iree_linalg_ext.attention":
                role = "torch.aten.scaled_dot_product_attention"
            elif cls == "quantized_matmul_fp8":
                qm_count += 1
                role_map = {
                    1: "Q projection (torch.aten.linear)",
                    2: "K projection (torch.aten.linear)",
                    3: "V projection (torch.aten.linear)",
                    4: "O projection (torch.aten.linear)",
                    5: "FC1 expansion (torch.aten.linear)",
                    6: "FC2 contraction (torch.aten.linear)",
                }
                role = role_map.get(qm_count, f"linear #{qm_count}")
            elif cls == "gelu_tanh":
                role = "torch.aten.gelu"
            elif cls in ("rms_norm", "reduction_sum", "elementwise_div", "bias_add_cast"):
                role = "torch.aten.layer_norm (part)"
            elif cls == "elementwise_add":
                result = op["result_types"][0] if op["result_types"] else ""
                if "768" in result or "1024" in result:
                    role = "Residual connection"
            elif cls == "type_conversion":
                role = "Precision cast"
            elif cls == "elementwise_mul":
                role = "Scale/norm multiply"

            result = op["result_types"][0] if op["result_types"] else ""
            compute_ops.append(
                {
                    "index": i,
                    "op": cls,
                    "result_type": result[:60],
                    "pytorch_role": role,
                    "mlir_text": op.get("mlir_text", ""),
                }
            )

        layers.append(
            {
                "layer_name": f"SigLIP Encoder Layer {layer_i} (Image {layer_i // 12})",
                "layer_type": "siglip_encoder",
                "index_range": [start, end],
                "compute_ops": compute_ops,
            }
        )

    # ----- Gemma Decoder Layers -----
    # Each pair of batch_matmul = one Gemma attention layer
    for layer_i in range(0, len(bmatmul_idx), 2):
        if layer_i + 1 >= len(bmatmul_idx):
            break

        score_idx = bmatmul_idx[layer_i]
        value_idx = bmatmul_idx[layer_i + 1]

        # Extend range to include MLP after attention
        start = max(0, score_idx - 100)
        # Find silu after value_idx (Gemma MLP marker)
        layer_silu = [s for s in silu_idx if s > value_idx and s < value_idx + 200]

        if layer_silu:
            end = layer_silu[0] + 50
        else:
            end = value_idx + 100

        # Collect compute ops
        compute_ops = []
        matmul_count = 0
        qm_count = 0
        for i in range(start, min(end, len(ops))):
            op = ops[i]
            if op["name"] in INFRA_OPS or op["name"] == "linalg.transpose":
                continue
            cls = op["cls"] or op["name"] or ""
            if cls in DEQUANT_OPS:
                continue

            role = ""
            if op["name"] == "linalg.batch_matmul":
                if i == score_idx:
                    role = "Score MatMul: Q @ K^T (torch.matmul)"
                elif i == value_idx:
                    role = "Value MatMul: attn @ V (torch.matmul)"
            elif op["name"] == "linalg.matmul":
                matmul_count += 1
                roles = {1: "Q proj", 2: "K proj", 3: "V proj", 4: "O proj"}
                role = f"Int8 MatMul — {roles.get(matmul_count, f'proj #{matmul_count}')} (torch.aten.linear)"
            elif op["name"] == "linalg.softmax":
                role = "torch.aten.softmax"
            elif cls == "quantized_matmul_fp8":
                qm_count += 1
                role = f"FP8 MatMul #{qm_count} (torch.aten.linear)"
            elif cls == "silu":
                role = "torch.aten.silu (Gemma MLP)"
            elif cls == "int8_quantize":
                role = "Dynamic quantization (bf16 → i8)"
            elif cls == "rms_norm":
                role = "GemmaRMSNorm"
            elif cls == "softmax_max_reduce":
                role = "Softmax — max reduction"
            elif cls == "softmax_exp":
                role = "Softmax — exp"
            elif cls and cls.startswith("rope"):
                role = "RoPE position encoding"

            result = op["result_types"][0] if op["result_types"] else ""
            compute_ops.append(
                {
                    "index": i,
                    "op": cls,
                    "result_type": result[:60],
                    "pytorch_role": role,
                    "mlir_text": op.get("mlir_text", ""),
                }
            )

        layers.append(
            {
                "layer_name": f"Gemma Decoder Layer {layer_i // 2}",
                "layer_type": "gemma_decoder",
                "index_range": [start, end],
                "compute_ops": compute_ops,
            }
        )

    return layers


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_trace_report(
    input_layers: list[dict],
    gopt_layers: list[dict],
    output_dir: Path,
) -> None:
    """Write the decomposition trace as markdown + JSON."""
    lines = []
    lines.append("# SmolVLA Layer Decomposition Trace")
    lines.append("")
    lines.append("*Auto-generated using MLIR Python bindings (`iree.compiler.ir`)*")
    lines.append("")
    lines.append(
        "For each PyTorch layer, shows the MLIR ops it decomposes into at the "
        "input level (`module.1.input.mlir`) with shapes and PyTorch role annotations."
    )
    lines.append("")

    # Write input-level traces
    lines.append("## Input Level (`module.1.input.mlir`)")
    lines.append("")

    # Summary by layer type
    siglip_layers = [l for layer in input_layers if layer["layer_type"] == "siglip_encoder"]
    gemma_layers = [l for layer in input_layers if layer["layer_type"] == "gemma_decoder"]

    lines.append(f"- **{len(siglip_layers)} SigLIP Encoder Layers** (12 per image × 3 images)")
    lines.append(f"- **{len(gemma_layers)} Gemma Decoder Layers** (attention + MLP)")
    lines.append("")

    # Show first SigLIP layer in detail
    if siglip_layers:
        layer = siglip_layers[0]
        lines.append(f"### {layer['layer_name']} (representative)")
        lines.append("")
        lines.append(
            "**PyTorch**: `LayerNorm → Linear(Q) → Linear(K) → Linear(V) → SDPA "
            "→ Linear(O) → Residual → LayerNorm → Linear(FC1) → GELU → Linear(FC2) → Residual`"
        )
        lines.append("")
        lines.append("| Op | Result Type | PyTorch Role |")
        lines.append("| --- | --- | --- |")
        for op in layer["compute_ops"]:
            if op["pytorch_role"]:
                lines.append(f"| `{op['op']}` | `{op['result_type']}` | {op['pytorch_role']} |")
        lines.append("")

        # Show MLIR snippets for key ops
        for op in layer["compute_ops"]:
            if op["mlir_text"] and op["pytorch_role"]:
                lines.append(f"**{op['pytorch_role']}**:")
                lines.append("```mlir")
                lines.append(op["mlir_text"][:400])
                lines.append("```")
                lines.append("")

    # Show first Gemma layer in detail
    if gemma_layers:
        layer = gemma_layers[0]
        lines.append(f"### {layer['layer_name']} (representative)")
        lines.append("")
        lines.append(
            "**PyTorch**: `RMSNorm → Linear(Q) → Linear(K) → Linear(V) → RoPE "
            "→ Q@K^T → Softmax → attn@V → Linear(O) → Residual → RMSNorm "
            "→ Linear(gate) → Linear(up) → SiLU → gate*up → Linear(down) → Residual`"
        )
        lines.append("")
        lines.append("| Op | Result Type | PyTorch Role |")
        lines.append("| --- | --- | --- |")
        for op in layer["compute_ops"]:
            if op["pytorch_role"]:
                lines.append(f"| `{op['op']}` | `{op['result_type']}` | {op['pytorch_role']} |")
        lines.append("")

        for op in layer["compute_ops"]:
            if op["mlir_text"] and op["pytorch_role"]:
                lines.append(f"**{op['pytorch_role']}**:")
                lines.append("```mlir")
                lines.append(op["mlir_text"][:400])
                lines.append("```")
                lines.append("")

    # ---------------------------------------------------------------
    # Global-Opt Level
    # ---------------------------------------------------------------
    lines.append("## Global-Opt Level (`module.4.global-optimization.mlir`)")
    lines.append("")
    lines.append(
        "This is the vanilla global-opt (spacemit target, no accelerator plugins). "
        "**This is the level kernel writers implement against.**"
    )
    lines.append("")
    lines.append("Key changes from input level:")
    lines.append("- **Transposes eliminated** (fused into contractions)")
    lines.append("- **MX fp8 dequant chains hoisted** to compile-time initializers")
    lines.append("- **FP8 weights may be pre-dequantized** to bf16 in initializers")
    lines.append("- Named ops (`linalg.batch_matmul`, `linalg.softmax`, `iree_linalg_ext.attention`) still present")
    lines.append("")

    gopt_siglip = [l for layer in gopt_layers if layer["layer_type"] == "siglip_encoder"]
    gopt_gemma = [l for layer in gopt_layers if layer["layer_type"] == "gemma_decoder"]

    lines.append(f"- **{len(gopt_siglip)} SigLIP Encoder Layers**")
    lines.append(f"- **{len(gopt_gemma)} Gemma Decoder Layers**")
    lines.append("")

    if gopt_siglip:
        layer = gopt_siglip[0]
        lines.append(f"### {layer['layer_name']} — Global-Opt (representative)")
        lines.append("")
        lines.append("| Op | Result Type | PyTorch Role |")
        lines.append("| --- | --- | --- |")
        for op in layer["compute_ops"]:
            if op["pytorch_role"]:
                lines.append(f"| `{op['op']}` | `{op['result_type']}` | {op['pytorch_role']} |")
        lines.append("")

        for op in layer["compute_ops"]:
            if op["mlir_text"] and op["pytorch_role"]:
                lines.append(f"**{op['pytorch_role']}** (global-opt):")
                lines.append("```mlir")
                lines.append(op["mlir_text"][:400])
                lines.append("```")
                lines.append("")

    if gopt_gemma:
        layer = gopt_gemma[0]
        lines.append(f"### {layer['layer_name']} — Global-Opt (representative)")
        lines.append("")
        lines.append("| Op | Result Type | PyTorch Role |")
        lines.append("| --- | --- | --- |")
        for op in layer["compute_ops"]:
            if op["pytorch_role"]:
                lines.append(f"| `{op['op']}` | `{op['result_type']}` | {op['pytorch_role']} |")
        lines.append("")

        for op in layer["compute_ops"]:
            if op["mlir_text"] and op["pytorch_role"]:
                lines.append(f"**{op['pytorch_role']}** (global-opt):")
                lines.append("```mlir")
                lines.append(op["mlir_text"][:400])
                lines.append("```")
                lines.append("")

    # Op count summary across all layers
    lines.append("## Op Count Summary")
    lines.append("")

    all_ops_count = {}
    for layer in input_layers:
        for op in layer["compute_ops"]:
            key = op["op"]
            all_ops_count[key] = all_ops_count.get(key, 0) + 1

    lines.append("| Op Classification | Total Instances | Description |")
    lines.append("| --- | --- | --- |")
    for op_name, count in sorted(all_ops_count.items(), key=lambda x: -x[1]):
        lines.append(f"| `{op_name}` | {count} | |")
    lines.append("")

    report_path = output_dir / "LAYER_DECOMPOSITION_TRACE.md"
    report_path.write_text("\n".join(lines))
    print(f"Report: {report_path}")

    # Also save as JSON for machine consumption
    json_data = {
        "input_level_layers": input_layers,
        "global_opt_layers": gopt_layers,
    }

    # Strip mlir_text for JSON (too large)
    for level_layers in [json_data["input_level_layers"], json_data["global_opt_layers"]]:
        for layer in level_layers:
            for op in layer["compute_ops"]:
                op.pop("mlir_text", None)

    json_path = output_dir / "layer_decomposition.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
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
            "build/compiled_models/smolVLA/spacemit_x60_RVV_smolVLA.q.fp8/phases/module.4.global-optimization.mlir"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/SaturnNPU"),
    )
    args = parser.parse_args()

    print("Tracing layer decomposition using MLIR Python bindings...\n")

    # Parse input level
    print("Input level:")
    input_ops = parse_mlir_ops(args.linalg_input)
    input_layers = build_layer_trace(input_ops, "input")
    print(f"    Found {len(input_layers)} layers")

    # Parse global-opt level
    print("Global-opt level:")
    gopt_ops = parse_mlir_ops(args.global_opt)
    gopt_layers = build_layer_trace(gopt_ops, "global_opt")
    print(f"    Found {len(gopt_layers)} layers")

    # Write reports
    print("\nWriting trace reports...")
    write_trace_report(input_layers, gopt_layers, args.output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
