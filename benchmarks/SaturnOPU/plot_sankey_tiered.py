"""Tier-colored Sankey for a Saturn OPU model (paper-ready, 16:9).

Design choices — locked after user feedback:

  * Title: just "Sankey Decomposition of <Model>". No subtitle, no stats.
  * Column 1 (PyTorch blocks): each block gets a distinct NPU-style
    hue — blue / purple / green / orange. These reappear nowhere else.
  * Columns 2-4 (torch-mlir, linalg, global-opt): NEUTRAL grey. Flow
    through these nodes is mixed by construction, so the node colors
    must not commit to any tier.
  * Column 5 (Saturn OPU terminal): strong tier colors.
      - OPU  : teal (vivid → faded for 32×32 → narrow)
      - RVV  : red / orange / salmon family (Option-E reduction/softmax
               = deepest red; matmul/direct-conv = orange; elementwise +
               pack/copy = salmon). Everything that runs on RVV at
               runtime is in this family — including elementwise and
               data-movement kernels.
      - Fused: muted crimson (distinct from teal and green). This is
               the "compile-time eliminated" track: linalg.transpose and
               linalg.broadcast that exist in phase-1 but are gone by
               phase-4.
  * Edges: soft. Coming off a PyTorch block they carry the block's hue
    (fade = 0.25). At the terminal edge they carry the destination's
    tier color. Middle edges are neutral grey.

All edge weights are op counts, end-to-end. Counts come from grepping
phase-1 / phase-4 MLIR dumps and `model_dispatch_decomposition.csv`;
OPU opcode counts from the linked ELF `.s`. Nothing is hand-entered
except the PyTorch module structure (because the compiler loses it).

Kept as a sibling to `plot_sankey.py` — the user wants both styles.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import plotly.graph_objects as go

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent


MODEL_PATHS: dict[str, Path] = {
    "opu_bench_hybrid": REPO_ROOT / "build/compiled_models/opu_bench_suite/saturn_opu_OPU_LLM_opu_bench_hybrid.q.int8",
    "opu_bench_vit": REPO_ROOT / "build/compiled_models/opu_bench_suite/saturn_opu_OPU_LLM_opu_bench_vit.q.int8",
}


# -------- NPU-style palette --------------------------------------------

# Column-1 block colors (NPU plot hues). Avoid green/teal — those are
# reserved for OPU terminals and would read as "OPU" if they appeared
# on a PyTorch block.
BLOCK_COLORS = [
    "rgba(52,152,219,0.90)",  # blue
    "rgba(155,89,182,0.90)",  # purple
    "rgba(187,78,143,0.90)",  # magenta  (distinct from OPU teal)
    "rgba(230,126,34,0.90)",  # orange
    "rgba(93,109,126,0.90)",  # slate    (spare)
]

# Column 2 / 3 / 4 = neutral greys (no tier implied at intermediate nodes).
C_TORCH_MLIR = "rgba(149,165,166,0.90)"  # mid grey
C_LINALG = "rgba(127,140,141,0.92)"  # darker grey
C_GOPT = "rgba(39,174,96,0.90)"  # NPU dark green

# Terminal colors — three tier families.
# OPU (teal) — darker = better tile match.
C_OPU_32 = "rgba(22,160,133,1.00)"
C_OPU_16 = "rgba(22,160,133,0.72)"
C_OPU_8 = "rgba(22,160,133,0.52)"
C_OPU_NARROW = "rgba(22,160,133,0.38)"
C_OPU_QDQ = "rgba(26,188,156,0.85)"

# RVV (red family — strong red throughout so every RVV terminal
# visually groups together, including direct-conv which stays on the
# RVV vector unit).
C_RVV_MATMUL = "rgba(176,58,46,0.95)"  # deep crimson
C_RVV_CONV = "rgba(203,67,53,0.95)"  # strong red
C_RVV_ELEM = "rgba(231,76,60,0.90)"  # vivid red (elementwise)
C_RVV_PACK = "rgba(236,112,99,0.80)"  # lighter red (pack / copy)

# Scalar (grey) — dispatches that explicitly do *not* use RVV vector
# after our `-v` target-feature patch (reduction / softmax / LN in f32).
# Runs as plain scalar riscv64 — no RVV, no OPU.
C_SCALAR_F32 = "rgba(127,140,141,0.92)"  # medium grey

# Fused (muted crimson) — compile-time eliminated (currently unused).
C_FUSED = "rgba(136,78,160,0.80)"

# Tier classification for coloring ribbons at the terminal edge.
SEG_TIER: dict[str, str] = {
    "runtime_32x32_tile": "opu",
    "encoding_32x32_tile": "opu",
    "runtime_16x16_tile": "opu",
    "encoding_16x16_tile": "opu",
    "runtime_8x8_tile": "opu",
    "runtime_narrow_tile": "opu",
    "encoding_narrow_tile": "opu",
    "fused_qdq": "opu",
    "inline_vopacc": "opu",
    # Runtime tiers:
    #   rvv   → kernels that DO use the RVV vector unit (matmul, conv,
    #           elementwise, pack / copy)
    #   scalar→ kernels that explicitly do NOT use RVV after our -v
    #           patch (f32 reduction, softmax, layer-norm). These were
    #           originally on RVV vector f32; the patch devectorizes
    #           them to scalar riscv64 to sidestep the vector hang.
    "rvv_reduction_softmax_norm": "scalar",
    "rvv_matmul": "rvv",
    "direct_conv": "rvv",
    "elementwise_other": "rvv",
    "data_movement": "rvv",
    "fused_eliminated": "fused",
}

TERMINAL_COLOR: dict[str, str] = {
    "runtime_32x32_tile": C_OPU_32,
    "encoding_32x32_tile": C_OPU_32,
    "runtime_16x16_tile": C_OPU_16,
    "encoding_16x16_tile": C_OPU_16,
    "runtime_8x8_tile": C_OPU_8,
    "runtime_narrow_tile": C_OPU_NARROW,
    "encoding_narrow_tile": C_OPU_NARROW,
    "fused_qdq": C_OPU_QDQ,
    "inline_vopacc": C_OPU_QDQ,
    "rvv_reduction_softmax_norm": C_SCALAR_F32,
    "rvv_matmul": C_RVV_MATMUL,
    "direct_conv": C_RVV_CONV,
    "elementwise_other": C_RVV_ELEM,
    "data_movement": C_RVV_PACK,
    "fused_eliminated": C_FUSED,
}

TERMINAL_LABEL: dict[str, str] = {
    "runtime_32x32_tile": "OPU 32×32 (runtime)",
    "encoding_32x32_tile": "OPU 32×32 (encoded)",
    "runtime_16x16_tile": "OPU 16×16 (runtime)",
    "encoding_16x16_tile": "OPU 16×16 (encoded)",
    "runtime_8x8_tile": "OPU 8×8",
    "runtime_narrow_tile": "OPU narrow (runtime)",
    "encoding_narrow_tile": "OPU narrow (encoded)",
    "fused_qdq": "Fused QDQ (VOPACC)",
    "inline_vopacc": "Inline VOPACC",
    "rvv_reduction_softmax_norm": "Scalar f32 (reduction / softmax / LN)",
    "rvv_matmul": "RVV matmul",
    "direct_conv": "Direct conv",
    "elementwise_other": "Elementwise",
    "data_movement": "Pack / copy",
    "fused_eliminated": "Compile-time eliminated",
}

# Tier → soft edge color (for ribbons that "commit" to a tier at the
# right edge of the Sankey). Middle ribbons use light grey.
EDGE_TIER = {
    "opu": "rgba(22,160,133,0.38)",
    "rvv": "rgba(203,67,53,0.38)",
    "scalar": "rgba(127,140,141,0.38)",  # grey for scalar ribbons
    "fused": "rgba(136,78,160,0.32)",
    "neutral": "rgba(189,195,199,0.28)",
}


# -------- Hand-asserted PyTorch structure ------------------------------

HybridDecomp = {
    "title": "Sankey Decomposition of Hybrid Model",
    "blocks": {
        "Conv stem": [
            ("torch.aten.conv2d", 2),
            ("torch.aten.batch_norm", 2),
            ("torch.aten.relu", 2),
        ],
        "TransformerBlock ×2": [
            ("torch.aten.layer_norm", 4),
            ("torch.aten.linear (Q/K/V/O)", 8),
            ("torch.aten.matmul (attention)", 4),
            ("torch.aten.softmax", 2),
            ("torch.aten.linear (FFN)", 4),
            ("torch.aten.gelu", 2),
        ],
        "Final LayerNorm": [("torch.aten.layer_norm", 1)],
        "Classifier head": [("torch.aten.linear (head)", 1)],
    },
    "torch_to_linalg": {
        "torch.aten.conv2d": "linalg.conv_2d_nchw_fchw_q",
        "torch.aten.batch_norm": "linalg.generic (elementwise)",
        "torch.aten.relu": "linalg.generic (elementwise)",
        "torch.aten.layer_norm": "linalg.generic (reduction)",
        "torch.aten.linear (Q/K/V/O)": "linalg.quantized_batch_matmul",
        "torch.aten.linear (FFN)": "linalg.quantized_batch_matmul",
        "torch.aten.linear (head)": "linalg.quantized_matmul",
        "torch.aten.linear (patch)": "linalg.quantized_matmul",
        "torch.aten.matmul (attention)": "linalg.quantized_batch_matmul",
        "torch.aten.softmax": "linalg.generic (reduction)",
        "torch.aten.gelu": "linalg.generic (elementwise)",
    },
    "linalg_to_gopt": {
        "linalg.conv_2d_nchw_fchw_q": "no im2col lowering",
        "linalg.quantized_batch_matmul": "batch_matmul (i8)",
        "linalg.quantized_matmul": "matmul (i8)",
        "linalg.generic (reduction)": "reduction / softmax (f32)",
        "linalg.generic (elementwise)": "fused elementwise",
    },
    "gopt_cues": {
        "no im2col lowering": ["conv_2d"],
        "batch_matmul (i8)": ["batch_matmul"],
        "matmul (i8)": ["matmul_like", "matmul"],
        "reduction / softmax (f32)": ["reduction", "softmax"],
        "fused elementwise": ["elementwise", "_encoding_", "generic", "pack"],
    },
}

VitBigDecomp = {
    "title": "Sankey Decomposition of ViT Model",
    "blocks": {
        "Patch projection": [("torch.aten.linear (patch)", 1)],
        "TransformerBlock ×2": HybridDecomp["blocks"]["TransformerBlock ×2"],
        "Classifier head": [("torch.aten.linear (head)", 1)],
    },
    "torch_to_linalg": HybridDecomp["torch_to_linalg"],
    "linalg_to_gopt": {k: v for k, v in HybridDecomp["linalg_to_gopt"].items() if k != "linalg.conv_2d_nchw_fchw_q"},
    "gopt_cues": {k: v for k, v in HybridDecomp["gopt_cues"].items() if k != "Direct conv (unfused)"},
}

DECOMPS = {
    "opu_bench_hybrid": HybridDecomp,
    "opu_bench_vit": VitBigDecomp,
}


# -------- Parsing helpers ---------------------------------------------


def phase_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for m in re.finditer(r"linalg\.(\w+)", path.read_text()):
        counts[m.group(1)] += 1
    return counts


def load_dispatch_segments(csv_path: Path, model_key: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("include_in_model") != "1" or r["model_key"] != model_key:
                continue
            out.append((r["symbol"], r.get("segment", "")))
    return out


def count_opu_insn_per_function(asm_path: Path) -> dict[str, int]:
    if not asm_path.exists():
        return {}
    counts: dict[str, int] = {}
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if current is not None:
            counts[current] = sum(1 for line in buf if ".insn r 87" in line)

    label_re = re.compile(r"^([A-Za-z_][\w$.]*):$")
    for line in asm_path.read_text().splitlines():
        m = label_re.match(line)
        if m:
            flush()
            current = m.group(1)
            buf = []
        else:
            buf.append(line)
    flush()
    return counts


def _fade(rgba: str, alpha: float) -> str:
    return re.sub(r"[\d.]+\)$", f"{alpha})", rgba)


# -------- Sankey builder -----------------------------------------------


def build_sankey(model_key: str, model_dir: Path, csv_path: Path) -> go.Figure:
    dec = DECOMPS[model_key]
    phases = model_dir / "phases"
    prefix = next(phases.glob("*.1.input.mlir")).stem.rsplit(".1.input", 1)[0]
    p1 = phase_counts(phases / f"{prefix}.1.input.mlir")
    p4 = phase_counts(phases / f"{prefix}.4.global-optimization.mlir")

    dispatches = load_dispatch_segments(csv_path, model_key)
    asm = next((model_dir / "files").glob("*linked*riscv_64.s"), None)
    asm_opu_insn = count_opu_insn_per_function(asm) if asm else {}

    # Assign a fixed hue to each PyTorch block (NPU style).
    block_color = {block: BLOCK_COLORS[i % len(BLOCK_COLORS)] for i, block in enumerate(dec["blocks"])}

    # Bucket dispatches → gopt + segment counts.
    def bucket_for_symbol(sym: str) -> str:
        s = sym.lower()
        for bucket, cues in dec["gopt_cues"].items():
            if any(cue in s for cue in cues):
                return bucket
        return "fused elementwise"

    bucket_seg: dict[str, Counter[str]] = defaultdict(Counter)
    seg_dispatches: Counter[str] = Counter()
    seg_opu_insn: Counter[str] = Counter()
    for sym, seg in dispatches:
        b = bucket_for_symbol(sym)
        bucket_seg[b][seg] += 1
        seg_dispatches[seg] += 1
        seg_opu_insn[seg] += asm_opu_insn.get(sym, 0)

    # Op-count totals per column.
    torch_totals: Counter[str] = Counter()
    for _, ops in dec["blocks"].items():
        for op, n in ops:
            torch_totals[op] += n
    linalg_totals: Counter[str] = Counter()
    for op, n in torch_totals.items():
        linalg_totals[dec["torch_to_linalg"][op]] += n
    gopt_totals: Counter[str] = Counter()
    for lin, n in linalg_totals.items():
        gopt_totals[dec["linalg_to_gopt"][lin]] += n

    # Build nodes + edges.
    labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    idx: dict[str, int] = {}
    edges: list[tuple[int, int, float, str]] = []

    X = [0.03, 0.27, 0.50, 0.73, 0.93]

    def node(name: str, color: str, x: float, y: float | None = None) -> int:
        if name not in idx:
            idx[name] = len(labels)
            labels.append(name)
            node_colors.append(color)
            node_x.append(x)
            # Plotly requires concrete y values when x is given. 0.5 is
            # centered; the eliminated track (only place that passes a
            # non-None y) pins to near-bottom so it doesn't cross the
            # main flow.
            node_y.append(y if y is not None else 0.5)
        return idx[name]

    def link(
        src_key: str,
        tgt_key: str,
        val: float,
        src_color: str,
        tgt_color: str,
        edge_color: str,
        x_src: float,
        x_tgt: float,
        y_src: float | None = None,
        y_tgt: float | None = None,
    ) -> None:
        if val <= 0:
            return
        s = node(src_key, src_color, x_src, y_src)
        t = node(tgt_key, tgt_color, x_tgt, y_tgt)
        edges.append((s, t, val, edge_color))

    # Column 1 → 2 : PyTorch block → torch.aten op (edges carry block hue)
    for block, ops in dec["blocks"].items():
        bcol = block_color[block]
        for op, n in ops:
            link(f"<b>{block}</b>", f"<b>{op}</b>", n, bcol, C_TORCH_MLIR, _fade(bcol, 0.25), X[0], X[1])

    # Compute the dominant tier of every global-opt bucket so we can
    # tier-color column 4 (and column 3, since each linalg op maps to
    # exactly one bucket). The tier commits gradually: columns 3 and 4
    # already wear their destination color before the terminal column.
    def bucket_tier(bucket: str) -> str:
        segs = bucket_seg.get(bucket, {})
        tier_weight: Counter[str] = Counter()
        for seg, k in segs.items():
            tier_weight[SEG_TIER.get(seg, "rvv")] += k
        return tier_weight.most_common(1)[0][0] if tier_weight else "rvv"

    # Map: tier → (node color, edge color).
    TIER_COMMIT = {
        "opu": (C_OPU_16, EDGE_TIER["opu"]),
        "rvv": (C_RVV_CONV, EDGE_TIER["rvv"]),
        "scalar": (C_SCALAR_F32, EDGE_TIER["scalar"]),
    }

    bucket_commit = {b: TIER_COMMIT[bucket_tier(b)] for b in gopt_totals}
    # Each linalg op maps to exactly one bucket, so its committed tier
    # matches the bucket's.
    linalg_commit = {lin: bucket_commit[dec["linalg_to_gopt"][lin]] for lin in linalg_totals}

    # Column 2 → 3 : torch.aten → linalg (tier-colored linalg nodes)
    for op, n in torch_totals.items():
        lin = dec["torch_to_linalg"][op]
        lin_color, edge_color = linalg_commit[lin]
        link(f"<b>{op}</b>", f"<b>{lin}</b>", n, C_TORCH_MLIR, lin_color, edge_color, X[1], X[2])

    # Column 3 → 4 : linalg → gopt (tier-colored nodes + ribbon).
    # Every path passes through col 4 — the bucket name describes the
    # COMPILER DECISION at Global-Opt (e.g., "no im2col lowering" for
    # conv), distinct from the execution outcome in col 5.
    for lin, n in linalg_totals.items():
        bucket = dec["linalg_to_gopt"][lin]
        lin_color, _ = linalg_commit[lin]
        bucket_color, edge_color = bucket_commit[bucket]
        link(f"<b>{lin}</b>", f"<b>{bucket}</b>", n, lin_color, bucket_color, edge_color, X[2], X[3])

    # Column 4 → 5 : gopt → terminal (tier-colored).
    for bucket, segs in bucket_seg.items():
        total_disp = sum(segs.values())
        total_w = gopt_totals.get(bucket, 0)
        bucket_color, _ = bucket_commit.get(bucket, (C_GOPT, EDGE_TIER["neutral"]))
        for seg, k in segs.items():
            term_key = f"<b>{TERMINAL_LABEL.get(seg, seg)}</b>"
            term_color = TERMINAL_COLOR.get(seg, C_RVV_ELEM)
            tier = SEG_TIER.get(seg, "rvv")
            w = total_w * k / total_disp
            link(f"<b>{bucket}</b>", term_key, w, bucket_color, term_color, EDGE_TIER[tier], X[3], X[4])

    # (Compile-time eliminated side-flow removed per user request —
    # its pseudo-block felt out of place in the PyTorch-module column.)
    eliminated_count = 0

    # ---- Captions ---------------------------------------------------
    lin_caption = {
        "linalg.conv_2d_nchw_fchw_q": f"linalg.conv_2d  ({p1.get('conv_2d_nchw_fchw_q', 0)})",
        "linalg.quantized_batch_matmul": f"linalg.quant_batch_matmul  ({p1.get('quantized_batch_matmul', 0)})",
        "linalg.quantized_matmul": f"linalg.quant_matmul  ({p1.get('quantized_matmul', 0)})",
        "linalg.generic (reduction)": "linalg.generic reduction  (~15 LN/SM)",
        "linalg.generic (elementwise)": f"linalg.generic elementwise  ({p1.get('generic', 0)} bodies)",
    }
    for orig, annot in lin_caption.items():
        k = f"<b>{orig}</b>"
        if k in idx:
            labels[idx[k]] = f"<b>{annot}</b>"

    gopt_caption = {
        "batch_matmul (i8)": f"batch_matmul i8  ({p4.get('batch_matmul', 0)})",
        "matmul (i8)": f"matmul i8  ({p4.get('matmul', 0)})",
        "reduction / softmax (f32)": f"reduction + softmax  ({p4.get('softmax', 0)} SM)",
        "fused elementwise": f"fused elementwise  ({p4.get('generic', 0)} generics)",
    }
    for orig, annot in gopt_caption.items():
        k = f"<b>{orig}</b>"
        if k in idx:
            labels[idx[k]] = f"<b>{annot}</b>"

    # Terminal annotations — terse single-line labels.
    for seg, n_disp in seg_dispatches.items():
        term_key = f"<b>{TERMINAL_LABEL.get(seg, seg)}</b>"
        if term_key not in idx:
            continue
        opc = seg_opu_insn[seg]
        base = TERMINAL_LABEL.get(seg, seg)
        if SEG_TIER.get(seg) == "opu" and opc > 0:
            labels[idx[term_key]] = f"<b>{base} · {n_disp} disp · {opc} ops</b>"
        else:
            labels[idx[term_key]] = f"<b>{base} · {n_disp} disp</b>"
    if eliminated_count > 0:
        term_key = f"<b>{TERMINAL_LABEL['fused_eliminated']}</b>"
        if term_key in idx:
            labels[idx[term_key]] = f"<b>{TERMINAL_LABEL['fused_eliminated']} · {eliminated_count} ops</b>"

    # ---- Render ------------------------------------------------------
    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]
    values = [e[2] for e in edges]
    link_colors = [e[3] for e in edges]

    # Let Plotly auto-layout: don't pass x/y. The column alignment is
    # preserved because edges always flow L→R across exactly 4 levels.
    # Auto-layout produces better y-positioning than our x-hints did.
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=28,
                    thickness=28,
                    line=dict(color="rgba(44,62,80,0.9)", width=0.9),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(source=sources, target=targets, value=values, color=link_colors),
                textfont=dict(size=17, color="#1b2631", family="Helvetica"),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{dec['title']}</b>",
            x=0.5,
            xanchor="center",
            y=0.975,
            yanchor="top",
            font=dict(size=28, color="#17202a", family="Helvetica"),
        ),
        annotations=[
            # Column headers — positioned above the plot area, NOT
            # overlapping any node. Top margin is generous so top-of-
            # column node labels have room above them.
            dict(
                x=x,
                y=1.055,
                xref="paper",
                yref="paper",
                text=f"<b>{name}</b>",
                showarrow=False,
                font=dict(size=15, color="#17202a", family="Helvetica"),
                xanchor="center",
                bgcolor="rgba(236,240,241,0.92)",
                borderpad=5,
                borderwidth=0.7,
                bordercolor="rgba(133,146,158,0.60)",
            )
            for x, name in zip(
                X,
                ["PyTorch block", "torch-mlir op", "linalg op (Phase 1)", "Global-Opt (Phase 4)", "Execution"],
            )
        ],
        font=dict(size=16, family="Helvetica", color="#1b2631"),
        width=1600,
        height=900,
        margin=dict(l=16, r=16, t=180, b=26),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _write(fig: go.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{name}.html"
    png = out_dir / f"{name}.png"
    fig.write_html(str(html))
    fig.write_image(str(png), width=1600, height=900, scale=2)
    print(f"  html: {html}  ({html.stat().st_size / 1024:.0f} KB)")
    print(f"  png : {png}   ({png.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="opu_bench_hybrid", choices=list(DECOMPS.keys()))
    parser.add_argument("--dispatch-csv", type=Path, default=BENCH_DIR / "model_dispatch_decomposition.csv")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "figures")
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    fig = build_sankey(args.model, MODEL_PATHS[args.model], args.dispatch_csv)
    _write(fig, args.out_dir, args.out_name or f"sankey_{args.model}_tiered")


if __name__ == "__main__":
    main()
