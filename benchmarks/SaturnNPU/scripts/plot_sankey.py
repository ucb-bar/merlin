#!/usr/bin/env python3
"""Generate a Sankey diagram showing how PyTorch modules decompose through MLIR.

Flow: PyTorch modules → sub-modules → Torch-MLIR ops → Linalg/Input → Global-Opt

Usage:
    python scripts/plot_sankey.py --output-dir benchmarks/SaturnNPU/plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/SaturnNPU/plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    sources = []
    targets = []
    values = []
    link_colors = []

    node_idx = {}

    def node(name):
        if name not in node_idx:
            node_idx[name] = len(labels)
            labels.append(name)
        return node_idx[name]

    def link(src, tgt, val, color="rgba(100,100,100,0.3)"):
        sources.append(node(src))
        targets.append(node(tgt))
        values.append(val)
        link_colors.append(color)

    # Colors
    BLUE = "rgba(52,152,219,0.4)"
    GREEN = "rgba(46,204,113,0.4)"
    PURPLE = "rgba(155,89,182,0.4)"
    ORANGE = "rgba(230,126,34,0.4)"
    RED = "rgba(231,76,60,0.3)"

    # ── Level 0 → 1: Top-level modules → sub-modules ──

    link("SigLIP Vision Tower", "SiglipAttention (×36)", 36, BLUE)
    link("SigLIP Vision Tower", "SiglipMLP (×36)", 36, BLUE)
    link("SigLIP Vision Tower", "LayerNorm (×72)", 72, BLUE)
    link("SigLIP Vision Tower", "Conv2d patch (×3)", 3, BLUE)

    link("Gemma Main LM", "GemmaAttention-main", 18, GREEN)
    link("Gemma Main LM", "GemmaMLP-main", 18, GREEN)
    link("Gemma Main LM", "GemmaRMSNorm", 36, GREEN)
    link("Gemma Main LM", "RotaryEmb", 18, GREEN)

    link("Gemma Expert", "GemmaAttention-expert", 18, PURPLE)
    link("Gemma Expert", "GemmaMLP-expert", 18, PURPLE)
    link("Gemma Expert", "GemmaRMSNorm", 36, PURPLE)

    link("Projections", "Linear-proj (×5)", 5, ORANGE)

    # ── Level 1 → 2: Sub-modules → Torch-MLIR ops ──

    link("SiglipAttention (×36)", "torch.aten.linear (382)", 144, BLUE)
    link("SiglipAttention (×36)", "torch.aten.sdpa (36)", 36, BLUE)
    link("SiglipMLP (×36)", "torch.aten.linear (382)", 72, BLUE)
    link("SiglipMLP (×36)", "torch.aten.gelu (36)", 36, BLUE)
    link("LayerNorm (×72)", "torch.aten.layer_norm (75)", 72, BLUE)
    link("Conv2d patch (×3)", "torch.aten.conv2d (3)", 3, BLUE)

    link("GemmaAttention-main", "torch.aten.linear (382)", 72, GREEN)
    link("GemmaAttention-main", "torch.aten.matmul (114)", 36, GREEN)
    link("GemmaAttention-main", "torch.aten.softmax (24)", 12, GREEN)
    link("GemmaAttention-expert", "torch.aten.linear (382)", 72, PURPLE)
    link("GemmaAttention-expert", "torch.aten.matmul (114)", 36, PURPLE)
    link("GemmaAttention-expert", "torch.aten.softmax (24)", 12, PURPLE)

    link("GemmaMLP-main", "torch.aten.linear (382)", 54, GREEN)
    link("GemmaMLP-main", "torch.aten.silu (33)", 16, GREEN)
    link("GemmaMLP-expert", "torch.aten.linear (382)", 54, PURPLE)
    link("GemmaMLP-expert", "torch.aten.silu (33)", 17, PURPLE)

    link("GemmaRMSNorm", "RMSNorm ops", 72, GREEN)
    link("RotaryEmb", "RoPE ops", 18, GREEN)

    link("Linear-proj (×5)", "torch.aten.linear (382)", 5, ORANGE)

    # ── Level 2 → 3: Torch-MLIR → Linalg/Input ──

    link("torch.aten.linear (382)", "quantized_matmul_fp8 (376)", 376, BLUE)
    link("torch.aten.linear (382)", "linalg.matmul i8 (67)", 67, GREEN)
    link("torch.aten.linear (382)", "MX fp8 dequant (~700)", 446, RED)
    link("torch.aten.linear (382)", "linalg.transpose (774)", 774, RED)

    link("torch.aten.sdpa (36)", "iree_linalg_ext.attention (36)", 36, BLUE)
    link("torch.aten.matmul (114)", "linalg.batch_matmul (46)", 46, GREEN)
    link("torch.aten.softmax (24)", "softmax decomposed", 24, GREEN)
    link("torch.aten.gelu (36)", "gelu_tanh (36)", 36, BLUE)
    link("torch.aten.silu (33)", "silu (32)", 32, GREEN)
    link("torch.aten.layer_norm (75)", "norm decomposed", 75, BLUE)
    link("RMSNorm ops", "norm decomposed", 72, GREEN)
    link("RoPE ops", "rope ops", 18, GREEN)
    link("torch.aten.conv2d (3)", "conv-like matmul", 3, BLUE)

    # ── Level 3 → 4: Linalg/Input → Global-Opt ──

    link("quantized_matmul_fp8 (376)", "G-Opt: quantized_matmul_fp8 (379)", 376, BLUE)
    link("iree_linalg_ext.attention (36)", "G-Opt: attention (36)", 36, BLUE)
    link("linalg.batch_matmul (46)", "G-Opt: batch_matmul (46)", 46, GREEN)
    link("linalg.matmul i8 (67)", "G-Opt: matmul (67)", 67, GREEN)
    link("softmax decomposed", "G-Opt: linalg.softmax (23)", 24, GREEN)
    link("gelu_tanh (36)", "G-Opt: gelu_tanh (36)", 36, BLUE)
    link("silu (32)", "G-Opt: silu (32)", 32, GREEN)
    link("norm decomposed", "G-Opt: rms_norm + reduce + div", 147, BLUE)
    link("rope ops", "G-Opt: rope_frequency + sin/cos", 18, GREEN)
    link("conv-like matmul", "G-Opt: conv matmul", 3, BLUE)

    link("MX fp8 dequant (~700)", "HOISTED (compile-time)", 446, RED)
    link("linalg.transpose (774)", "ELIMINATED (fused)", 774, RED)

    # Node colors
    node_colors = []
    for label in labels:
        if "SigLIP" in label or "Siglip" in label or "Conv2d" in label:
            node_colors.append("rgba(52,152,219,0.8)")
        elif "Gemma" in label and "Expert" not in label:
            node_colors.append("rgba(46,204,113,0.8)")
        elif "Expert" in label:
            node_colors.append("rgba(155,89,182,0.8)")
        elif "HOISTED" in label or "ELIMINATED" in label:
            node_colors.append("rgba(231,76,60,0.8)")
        elif "Projection" in label or "Linear-proj" in label:
            node_colors.append("rgba(230,126,34,0.8)")
        elif "G-Opt" in label:
            node_colors.append("rgba(39,174,96,0.9)")
        elif "torch.aten" in label:
            node_colors.append("rgba(149,165,166,0.8)")
        else:
            node_colors.append("rgba(127,140,141,0.8)")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=12,
                    thickness=18,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=(
            "SmolVLA: PyTorch Modules → Torch-MLIR → Linalg/Input → Global-Opt<br>"
            "<sup>Flow width proportional to op count. Red = eliminated/hoisted by compiler.</sup>"
        ),
        font_size=10,
        width=1800,
        height=1100,
    )

    # Save as HTML (interactive)
    html_path = args.output_dir / "sankey_decomposition.html"
    fig.write_html(str(html_path))
    print(f"Sankey: {html_path}")
    print(f"  Size: {html_path.stat().st_size / 1024:.0f} KB")
    print("  Open in browser for interactive view")

    # Try PNG
    try:
        png_path = args.output_dir / "sankey_decomposition.png"
        fig.write_image(str(png_path), scale=2)
        print(f"  PNG: {png_path}")
    except Exception:
        print("  (PNG export needs kaleido — HTML is the primary output)")


if __name__ == "__main__":
    main()
