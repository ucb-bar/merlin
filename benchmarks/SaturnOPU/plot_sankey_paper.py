"""Paper-figure Sankey for the hybrid-model case study (ASPLOS-style).

Purpose: a polished main-paper explanatory figure showing how different
regions of the hybrid model lower through MLIR and land on different
execution backends. **Not** a quantitative figure — numbers are shown
elsewhere in the paper.

Design constraints:
  * Exactly four columns: model region → op class → lowered pattern →
    execution path. No raw IR names, no phase numbers, no counts.
  * Link widths proportional to analytical compute share (from
    `per_model_summary.csv`) so dominant flows still feel dominant,
    but the numbers themselves never appear in the figure.
  * Color family by final execution path; left-side nodes are kept
    visually quiet (light grey) so the eye flows to the right column.
  * Paper-appropriate title, no subtitle, no in-figure caption.

This is a sibling to `plot_sankey.py` and `plot_sankey_tiered.py` — the
engineering-detail versions. Those are kept for reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go

BENCH_DIR = Path(__file__).resolve().parent


# -------- Color palette ------------------------------------------------
# Terminal-family colors (column 4 — made visually prominent). OPU gets
# a 4-shade teal family to distinguish tile / encoding variants; RVV
# side is warm (salmon → dark red); scalar is a single neutral grey.

C_OPU_32_RUN = "rgba(17,122,139,0.98)"  # dark teal — fast 32×32 runtime
C_OPU_NRW_RUN = "rgba(38,166,154,0.95)"  # medium teal — runtime narrow
C_OPU_16_ENC = "rgba(72,180,178,0.95)"  # lighter teal — encoded 16×16
C_OPU_NRW_ENC = (
    "rgba(31,97,141,1.00)"  # navy-teal — encoded narrow (distinct + saturated so the tiny ribbon is visible)
)
C_SCALAR_RED = "rgba(149,165,166,0.92)"  # neutral grey — scalar reductions
C_RVV_ELEM = "rgba(223,134,118,0.95)"  # muted salmon — RVV elementwise
C_DIRECT_CONV = "rgba(176,58,46,1.00)"  # dark red — direct conv

# Distinctive accent for the Final LayerNorm node (otherwise it sits as
# a thin grey sliver and is easy to miss).
C_FINAL_LN = "rgba(243,156,18,0.92)"  # amber

# Interior columns — quieter palette so the reader's eye flows rightward.
C_REGION = "rgba(189,195,199,0.85)"  # light grey — model regions (quiet)
C_OPCLASS = "rgba(166,172,175,0.85)"  # slightly darker grey
C_LOWER = "rgba(133,146,158,0.85)"  # medium grey


def _fade(rgba: str, alpha: float = 0.40) -> str:
    """Return the same rgba with a new alpha — used for link colors."""
    import re

    return re.sub(r"[\d.]+\)$", f"{alpha})", rgba)


# -------- Data (analytical compute share per model) -------------------
# Each MODEL_SPEC bundles the 4-column Sankey data:
#   TERM             — execution-path (col-4) totals
#   LOWER_TO_TERMS   — col-3 → col-4 fan-out
#   OPCLASS_TO_LOWER — col-2 → col-3 mapping + share
#   REGION_EDGES     — col-1 → col-2 edges
#   TITLE            — figure title

# -- HYBRID MODEL -------------------------------------------------------
HYBRID_TERM = {
    "Direct conv": 2_663_424,
    "RVV elementwise": 13_021_200,
    "Scalar / RVV reduction": 2_979_200,
    "OPU 32×32": 201_326_592,
    "OPU narrow": 25_165_824,
    "Enc. OPU 16×16": 16_777_216,
    "Enc. OPU narrow": 4_096,
}
HYBRID_LOWER_TO_TERMS: dict[str, list[tuple[str, float]]] = {
    "Convolution": [("Direct conv", 1.0)],
    "Elementwise": [("RVV elementwise", 1.0)],
    "Reduction / softmax": [("Scalar / RVV reduction", 1.0)],
    "Batched matmul": [
        ("OPU 32×32", 201_326_592 / 226_492_416),
        ("OPU narrow", 25_165_824 / 226_492_416),
    ],
    "Matmul": [
        ("Enc. OPU 16×16", 16_777_216 / 16_781_312),
        ("Enc. OPU narrow", 4_096 / 16_781_312),
    ],
}
HYBRID_OPCLASS_TO_LOWER: dict[str, tuple[str, float]] = {
    "Conv2d": ("Convolution", 1.0),
    "BatchNorm / activation": ("Elementwise", 1.0),
    "Attention projections": ("Batched matmul", 8 / 16),
    "Attention matmul": ("Batched matmul", 4 / 16),
    "FFN matmul": ("Batched matmul", 4 / 16),
    "LayerNorm": ("Reduction / softmax", 5 / 7),
    "Softmax": ("Reduction / softmax", 2 / 7),
    "Head matmul": ("Matmul", 1.0),
}
HYBRID_REGION_EDGES: list[tuple[str, str, float]] = [
    ("Conv stem", "Conv2d", 1.0),
    ("Conv stem", "BatchNorm / activation", 4 / 6),
    ("Transformer blocks", "BatchNorm / activation", 2 / 6),
    ("Transformer blocks", "Attention projections", 1.0),
    ("Transformer blocks", "Attention matmul", 1.0),
    ("Transformer blocks", "FFN matmul", 1.0),
    ("Transformer blocks", "LayerNorm", 4 / 5),
    ("Final LayerNorm", "LayerNorm", 1 / 5),
    ("Transformer blocks", "Softmax", 1.0),
    ("Classifier head", "Head matmul", 1.0),
]

# -- YOLOv8-nano --------------------------------------------------------
# Ops counts sourced from model_dispatch_decomposition.csv (yolov8_nano
# rows). Segments: runtime_8x8_tile (OPU via im2col-mmt4d for 1×1
# convs), direct_conv (RVV direct for 3×3 convs), elementwise
# (BatchNorm/SiLU/scaling), rvv_matmul (detection-head), rvv_reduction
# (NMS-adjacent), data_movement (packs, barely visible).
YOLO_TERM = {
    "RVV conv (3×3)": 78_464_512,  # 3×3 spatial conv, pure RVV loops
    "OPU 8×8": 338_995_200,  # 1×1 via mmt4d→OPU runtime
    "OPU narrow": 268_800,  # narrow-tile fallback
    "RVV matmul": 768_000,  # detection-head tiny matmuls
    "RVV elementwise": 4_370_800,  # BatchNorm / SiLU / scaling
    "Scalar / RVV reduction": 252_000,  # softmax/NMS helpers
}
# Col-3: pick the SINGLE dominant lowered pattern per segment. YOLO
# doesn't have distinct Encoded vs. Runtime categories — every matmul
# path is runtime-dispatch (no data tiling for IM2COL).
YOLO_LOWER_TO_TERMS = {
    "3×3 Convolution": [("RVV conv (3×3)", 1.0)],
    "1×1 Convolution": [
        ("OPU 8×8", YOLO_TERM["OPU 8×8"] / (YOLO_TERM["OPU 8×8"] + YOLO_TERM["OPU narrow"])),
        ("OPU narrow", YOLO_TERM["OPU narrow"] / (YOLO_TERM["OPU 8×8"] + YOLO_TERM["OPU narrow"])),
    ],
    "Elementwise": [("RVV elementwise", 1.0)],
    "Matmul": [("RVV matmul", 1.0)],
    "Reduction / softmax": [("Scalar / RVV reduction", 1.0)],
}
# Op classes grouped by their typical call-site identity in the YOLO
# PyTorch source: C2f blocks contain both Conv (3×3 strided/regular)
# and Bottleneck (1×1 + 3×3). SPPF has pooling + 1×1. Head: conv +
# regression matmul.
YOLO_OPCLASS_TO_LOWER = {
    "Conv 3×3": ("3×3 Convolution", 1.0),
    "Conv 1×1": ("1×1 Convolution", 1.0),
    "BatchNorm / SiLU": ("Elementwise", 1.0),
    "Upsample / Concat": ("Elementwise", 1.0),  # dest share below
    "Head regression": ("Matmul", 1.0),
    "Softmax / Reduction": ("Reduction / softmax", 1.0),
}
# Region: the Ultralytics YOLOv8n source has a recognizable 3-stage
# structure. Shares are approximate proportional to the op counts in
# each region (hand-estimated; the exact per-region split isn't in the
# CSV, only per-dispatch).
YOLO_REGION_EDGES = [
    ("Backbone (C2f)", "Conv 3×3", 0.70),
    ("Backbone (C2f)", "Conv 1×1", 0.55),
    ("Backbone (C2f)", "BatchNorm / SiLU", 0.70),
    ("Neck (FPN/SPPF)", "Conv 3×3", 0.15),
    ("Neck (FPN/SPPF)", "Conv 1×1", 0.30),
    ("Neck (FPN/SPPF)", "BatchNorm / SiLU", 0.20),
    ("Neck (FPN/SPPF)", "Upsample / Concat", 1.0),
    ("Detection heads", "Conv 3×3", 0.15),
    ("Detection heads", "Conv 1×1", 0.15),
    ("Detection heads", "BatchNorm / SiLU", 0.10),
    ("Detection heads", "Head regression", 1.0),
    ("Detection heads", "Softmax / Reduction", 1.0),
]

# -- ViT v2 (dim=512, 6 blocks, conv stem, all-token head) ---------------
# Compute breakdown (analytical MACs, from the exported ONNX):
# - Conv stem: 2 small 3×3 convs (3→64, 64→512) on 64×64 input
# - 6 × transformer blocks each with:
#     QKV projections:  3 × 256×512×512  = 201 M MACs
#     Q@K attention:    4 × 256×256×128  = 34 M MACs
#     S@V attention:    4 × 256×128×256  = 34 M MACs
#     O  projection:    256×512×512      = 67 M MACs
#     FFN up:           256×2048×512     = 268 M MACs
#     FFN down:         256×512×2048     = 268 M MACs
#     LayerNorm + softmax small
# - Head: 256×16×512 = 2 M MACs (N=16 → encoding 16×16 tile)
VIT_TERM = {
    "RVV conv (3×3)": 1_769_472 + 7_077_888,  # stem convs
    "RVV elementwise": 20_000_000,  # BN+ReLU+GELU+residuals
    "Scalar / RVV reduction": 1_500_000,  # LayerNorm+Softmax
    "AOT 32×32 OPU": (201 + 34 + 34 + 67 + 268 + 268) * 1_000_000 * 6,
    "AOT 16×16 OPU": 2_097_152,  # head 256×16×512
}
VIT_LOWER_TO_TERMS = {
    "3×3 Convolution": [("RVV conv (3×3)", 1.0)],
    "Elementwise": [("RVV elementwise", 1.0)],
    "Reduction / softmax": [("Scalar / RVV reduction", 1.0)],
    "Batched matmul": [("AOT 32×32 OPU", 1.0)],
    "Matmul (2D)": [
        (
            "AOT 32×32 OPU",
            (201 + 34 + 34 + 67 + 268 + 268)
            * 1_000_000
            * 6
            / ((201 + 34 + 34 + 67 + 268 + 268) * 1_000_000 * 6 + 2_097_152),
        ),
        ("AOT 16×16 OPU", 2_097_152 / ((201 + 34 + 34 + 67 + 268 + 268) * 1_000_000 * 6 + 2_097_152)),
    ],
}
# Attention matmul = batch_matmul; QKV/FFN/O/head = plain matmul
VIT_OPCLASS_TO_LOWER = {
    "Conv 3×3": ("3×3 Convolution", 1.0),
    "BatchNorm / ReLU": ("Elementwise", 0.4),
    "Attention QKV/O proj": ("Matmul (2D)", 0.42),  # 4 of 6 matmul families
    "Attention Q@K, S@V": ("Batched matmul", 1.0),
    "FFN up / down": ("Matmul (2D)", 0.54),  # 2 of 6 (larger)
    "Head projection": ("Matmul (2D)", 0.04),  # tiny
    "LayerNorm / Softmax": ("Reduction / softmax", 1.0),
    "GELU / residuals": ("Elementwise", 0.6),
}
VIT_REGION_EDGES = [
    ("Conv stem", "Conv 3×3", 1.0),
    ("Conv stem", "BatchNorm / ReLU", 0.5),
    ("Transformer blocks", "BatchNorm / ReLU", 0.5),  # (nominal — residual norms)
    ("Transformer blocks", "Attention QKV/O proj", 1.0),
    ("Transformer blocks", "Attention Q@K, S@V", 1.0),
    ("Transformer blocks", "FFN up / down", 1.0),
    ("Transformer blocks", "LayerNorm / Softmax", 1.0),
    ("Transformer blocks", "GELU / residuals", 1.0),
    ("Classification head", "Head projection", 1.0),
]

MODEL_SPECS = {
    "hybrid": {
        "title": "Hybrid Model Decomposition by Execution Path",
        "TERM": HYBRID_TERM,
        "LOWER_TO_TERMS": HYBRID_LOWER_TO_TERMS,
        "OPCLASS_TO_LOWER": HYBRID_OPCLASS_TO_LOWER,
        "REGION_EDGES": HYBRID_REGION_EDGES,
        "canvas_h": 1060,
        "top_margin": 170,
        "node_pad": 62,
    },
    "vit": {
        "title": "ViT Model Decomposition by Execution Path",
        "TERM": VIT_TERM,
        "LOWER_TO_TERMS": VIT_LOWER_TO_TERMS,
        "OPCLASS_TO_LOWER": VIT_OPCLASS_TO_LOWER,
        "REGION_EDGES": VIT_REGION_EDGES,
        "canvas_h": 820,
        "canvas_w": 1300,
        "top_margin": 130,
        "right_margin": 150,
        "node_pad": 35,
    },
    "yolov8": {
        "title": "YOLOv8-n Model Decomposition by Execution Path",
        "TERM": YOLO_TERM,
        "LOWER_TO_TERMS": YOLO_LOWER_TO_TERMS,
        "OPCLASS_TO_LOWER": YOLO_OPCLASS_TO_LOWER,
        "REGION_EDGES": YOLO_REGION_EDGES,
        # Fewer model regions (3 vs hybrid's 4) → shorter canvas and
        # tighter node pad so the flows fill the vertical space instead
        # of floating in a tall empty box.
        "canvas_h": 780,
        "canvas_w": 1200,
        "top_margin": 120,
        "right_margin": 120,
        "node_pad": 30,
    },
}

# Backward-compat aliases used by the rest of the module.
TERM = HYBRID_TERM
LOWER_TO_TERMS = HYBRID_LOWER_TO_TERMS
OPCLASS_TO_LOWER = HYBRID_OPCLASS_TO_LOWER
REGION_EDGES = HYBRID_REGION_EDGES


# -------- Build the Sankey --------------------------------------------


# -------- Scale transform ---------------------------------------------
# Link widths are the RAW analytical compute share, then passed through
# val**SCALE_POWER to compress the dynamic range. SCALE_POWER=1.0 is the
# true-proportional Sankey; smaller powers make thin ribbons visible
# while preserving the ordering. 0.55–0.65 is a common paper sweet spot.
SCALE_POWER = 0.55


def _scale(val: float) -> float:
    return val**SCALE_POWER if val > 0 else 0.0


def build_figure(model: str = "hybrid") -> go.Figure:
    spec = MODEL_SPECS[model]
    TERM = spec["TERM"]
    LOWER_TO_TERMS = spec["LOWER_TO_TERMS"]
    OPCLASS_TO_LOWER = spec["OPCLASS_TO_LOWER"]
    REGION_EDGES = spec["REGION_EDGES"]
    TITLE = spec["title"]

    # Resolve compute per (column-3 lowered pattern).
    lower_weight: dict[str, float] = {lower: sum(TERM[t] for t, _ in dests) for lower, dests in LOWER_TO_TERMS.items()}

    # Resolve compute per (column-2 op class).
    opclass_weight: dict[str, float] = {}
    for opclass, (lower, frac) in OPCLASS_TO_LOWER.items():
        opclass_weight[opclass] = lower_weight[lower] * frac

    # Percentages are computed from raw (unscaled) compute share so the
    # numbers printed on the figure reflect the true decomposition, not
    # the visual compression.
    TOTAL = float(sum(TERM.values()))
    term_pct = {t: 100.0 * v / TOTAL for t, v in TERM.items()}

    # Terminal color per execution path. OPU sub-paths get distinct
    # teal shades; RVV side is warm; scalar is neutral grey. The 8×8
    # shade is a slightly different teal so it can live in the same
    # figure alongside 32×32 without visual collision.
    TERM_COLOR = {
        "Direct conv": C_DIRECT_CONV,
        "RVV conv (3×3)": C_DIRECT_CONV,
        "RVV elementwise": C_RVV_ELEM,
        "RVV matmul": "rgba(230, 119, 91, 0.95)",  # warm salmon
        "Scalar / RVV reduction": C_SCALAR_RED,
        "OPU 32×32": C_OPU_32_RUN,
        "OPU 8×8": "rgba(26, 148, 149, 0.95)",  # distinct teal
        "OPU narrow": C_OPU_NRW_RUN,
        "Enc. OPU 16×16": C_OPU_16_ENC,
        "Enc. OPU narrow": C_OPU_NRW_ENC,
        # New ViT palette entries (AOT encoding path):
        "AOT 32×32 OPU": C_OPU_32_RUN,
        "AOT 16×16 OPU": C_OPU_16_ENC,
    }

    # Dominant terminal color per lowered pattern (used to tint upstream
    # ribbons so each flow keeps a single visual identity).
    def dominant_color(lower: str) -> str:
        dests = LOWER_TO_TERMS[lower]
        top = max(dests, key=lambda d: TERM[d[0]])
        return TERM_COLOR[top[0]]

    # Column x positions. Col-4 at 0.86 sits deeper into the canvas so
    # inline terminal labels ("OPU 32×32 77%") end near the right edge
    # without leaving a big dead zone after them.
    X = [0.05, 0.32, 0.58, 0.86]

    labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    idx: dict[str, int] = {}
    edges: list[tuple[int, int, float, str]] = []

    # Uniform font size per column — no per-node overrides. Readers
    # scan columns left-to-right; the same size within a column is
    # what makes the figure feel clean.
    FONT_PX = {X[0]: 28, X[1]: 20, X[2]: 20, X[3]: 22}

    # All four columns get inline labels. Col-3 uses SHORT display names
    # (3-5 chars) at a smaller font so they fit in the gap before col-4
    # without colliding. Without col-3 labels the bars look like
    # unexplained grey blocks — "empty spaces going nowhere."
    COL3_SHORT = {
        "Convolution": "Conv",
        "3×3 Convolution": "3×3",
        "1×1 Convolution": "1×1",
        "Elementwise": "Elem",
        "Batched matmul": "BMM",
        "Matmul": "MM",
        "Reduction / softmax": "Red/SM",
    }

    def _sized(name: str, x: float) -> str:
        disp = COL3_SHORT.get(name, name)
        size = FONT_PX[x]
        # Col-4 terminals append their compute-share percentage inline.
        if name in term_pct:
            pct = term_pct[name]
            pct_sz = max(18, int(size * 0.9))
            pct_txt = f"{pct:.1f}%" if pct < 1 else f"{pct:.0f}%"
            return (
                f"<span style='font-size:{size}px'><b>{disp}</b></span>"
                f"<span style='font-size:{pct_sz}px;color:#17202a'>"
                f"&nbsp;&nbsp;<b>{pct_txt}</b></span>"
            )
        return f"<span style='font-size:{size}px'><b>{disp}</b></span>"

    def node(name: str, color: str, x: float) -> int:
        # Key by name (not label) — empty/blank labels collide otherwise.
        if name not in idx:
            idx[name] = len(labels)
            labels.append(_sized(name, x))
            node_colors.append(color)
            node_x.append(x)
        return idx[name]

    def link(src: str, tgt: str, val: float, color: str, src_c: str, tgt_c: str, x_src: float, x_tgt: float) -> None:
        if val <= 0:
            return
        s = node(src, src_c, x_src)
        t = node(tgt, tgt_c, x_tgt)
        edges.append((s, t, val, color))

    # Per-node color override (only Final LayerNorm — make it pop with
    # an amber tone so the 1-op sliver is unmistakable).
    REGION_NODE_COLOR = {
        "Final LayerNorm": C_FINAL_LN,
    }

    # Column 3 → 4 (fan-out: matmul-family splits across OPU variants).
    for lower, dests in LOWER_TO_TERMS.items():
        for term, frac in dests:
            w = _scale(lower_weight[lower] * frac)
            color = TERM_COLOR[term]
            # Tiny ribbons (Enc. OPU narrow ≈ 4096 ops) need a higher
            # alpha or they vanish on the canvas.
            edge_alpha = 0.85 if term == "Enc. OPU narrow" else 0.55
            link(lower, term, w, _fade(color, edge_alpha), C_LOWER, color, X[2], X[3])

    # Column 2 → 3.
    for opclass, (lower, _frac) in OPCLASS_TO_LOWER.items():
        w = _scale(opclass_weight[opclass])
        color = _fade(dominant_color(lower), 0.38)
        link(opclass, lower, w, color, C_OPCLASS, C_LOWER, X[1], X[2])

    # Column 1 → 2.
    for region, opclass, frac in REGION_EDGES:
        lower, _ = OPCLASS_TO_LOWER[opclass]
        w = _scale(opclass_weight[opclass] * frac)
        color = _fade(dominant_color(lower), 0.28)
        region_color = REGION_NODE_COLOR.get(region, C_REGION)
        link(region, opclass, w, color, region_color, C_OPCLASS, X[0], X[1])

    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]
    values = [e[2] for e in edges]
    link_colors = [e[3] for e in edges]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                # align="right" puts every node label to the RIGHT of its bar.
                # Combined with X[3]=0.92 and a right margin big enough to fit
                # "Enc. OPU 16×16 (6%)", col-4 labels sit cleanly in the right
                # gutter and never collide with col-3 labels.
                # (The default "justify" puts col-4 labels on the LEFT — i.e.,
                # into the gap between col-3 and col-4 — which was colliding
                # with col-3's own right-flowing labels.)
                node=dict(
                    pad=spec.get("node_pad", 62),
                    thickness=34,
                    align="right",
                    line=dict(color="rgba(44,62,80,0.75)", width=0.8),
                    label=labels,
                    color=node_colors,
                    x=node_x,
                ),
                link=dict(source=sources, target=targets, value=values, color=link_colors),
                textfont=dict(size=30, color="#1b2631", family="Helvetica"),
            )
        ]
    )

    # Canvas: paper-figure aspect. Narrower (1400) than the earlier
    # full-bleed draft, and a bit shorter overall (1060 vs. the original
    # 1280) — the extra vertical room goes into node `pad` so each
    # label has breathing room rather than into white space around the
    # figure itself.
    CANVAS_W = spec.get("canvas_w", 1400)
    CANVAS_H = spec.get("canvas_h", 1060)
    TOP_MARGIN = spec.get("top_margin", 170)
    RIGHT_MARGIN = spec.get("right_margin", 180)

    # -- Column headers -------------------------------------------------
    header_annotations = [
        dict(
            x=x,
            y=1.04,
            xref="paper",
            yref="paper",
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=20, color="#2c3e50", family="Helvetica"),
            xanchor="center",
            yanchor="middle",
            bgcolor="rgba(236,240,241,0.92)",
            borderpad=6,
            borderwidth=0.9,
            bordercolor="rgba(133,146,158,0.55)",
        )
        for x, name in zip(
            X,
            ["Model region", "Op class", "Lowered pattern", "Execution path"],
        )
    ]

    fig.update_layout(
        title=dict(
            text=f"<b>{TITLE}</b>",
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=28, color="#17202a", family="Helvetica"),
        ),
        annotations=header_annotations,
        font=dict(size=22, family="Helvetica", color="#1b2631"),
        width=CANVAS_W,
        height=CANVAS_H,
        # Right margin tuned to fit the longest col-4 label + percentage
        # (~"Scalar / RVV reduction 0.1%") without leaving dead space.
        margin=dict(l=18, r=RIGHT_MARGIN, t=TOP_MARGIN, b=24),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "figures")
    parser.add_argument(
        "--model",
        choices=["hybrid", "yolov8", "vit", "all"],
        default="hybrid",
        help="Which model's Sankey to render (or 'all').",
    )
    parser.add_argument("--out-name", default=None, help="Output basename (default: sankey_<model>_paper).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = ["hybrid", "yolov8", "vit"] if args.model == "all" else [args.model]
    for model in models:
        fig = build_figure(model)
        basename = args.out_name or f"sankey_{model}_paper"
        html = args.out_dir / f"{basename}.html"
        png = args.out_dir / f"{basename}.png"
        fig.write_html(str(html))
        fig.write_image(str(png), width=fig.layout.width, height=fig.layout.height, scale=4)
        print(f"  [{model}] html: {html}  ({html.stat().st_size / 1024:.0f} KB)")
        print(f"  [{model}] png : {png}   ({png.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
