"""NPU-style Sankey for a Saturn OPU model (paper-ready, 16:9).

Mirrors `benchmarks/SaturnNPU/scripts/plot_sankey.py`. Five columns,
left → right:

  1. PyTorch top-level blocks     (Conv stem, TransformerBlock×2, Head)
  2. Torch-MLIR ops               (conv2d, linear, layer_norm, softmax, gelu, …)
  3. Linalg ops                   (conv_2d_nchw_fchw_q, quantized_batch_matmul, …)
  4. Global-opt buckets           (Phase-4 post-fusion)
  5. Saturn OPU terminal          (OPU 32×32 / OPU narrow / Fused QDQ / RVV / …)

Edge widths = **op counts, in one unit end-to-end** (no mixing with
dispatch counts). Column 1 is hand-asserted from
`models/opu_bench_suite/opu_bench_models.py` (nn.Module structure is
lost after ONNX import); columns 2-4 come from grepped phase-1/phase-4
MLIR dumps; column 5 is `model_dispatch_decomposition.csv` with a
cross-check against `.insn r 87, …` opcode counts in the linked
embedded-ELF `.s`.

Output: PNG + HTML at 1600×900 (clean 16:9 for single-column paper
figure). Fonts and pad tuned for that canvas — do not resize without
re-inspecting.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent


# -------- Model registry -----------------------------------------------

MODEL_PATHS: dict[str, Path] = {
    "opu_bench_hybrid": REPO_ROOT / "build/compiled_models/opu_bench_suite/saturn_opu_OPU_LLM_opu_bench_hybrid.q.int8",
    "opu_bench_vit": REPO_ROOT / "build/compiled_models/opu_bench_suite/saturn_opu_OPU_LLM_opu_bench_vit.q.int8",
}


# -------- Paper palette ------------------------------------------------
# One hue per column (blue → purple → grey → green → teal).
# Paper figure = 16:9, single column. Keep saturation moderate so
# colors remain legible after b/w print. Teal family reserved for OPU
# terminals (the "good" destination); red for RVV-scalar (Option E path).

C_L1 = "rgba(52,152,219,0.90)"  # PyTorch block — blue
C_L2 = "rgba(155,89,182,0.85)"  # torch-mlir op — purple
C_L3 = "rgba(127,140,141,0.85)"  # linalg — grey
C_L4 = "rgba(39,174,96,0.90)"  # global-opt — green

# Terminal palette — 4-stop teal for OPU quality, others distinct.
C_OPU_32 = "rgba(26,138,160,0.98)"
C_OPU_16 = "rgba(26,138,160,0.78)"
C_OPU_8 = "rgba(26,138,160,0.58)"
C_OPU_NARROW = "rgba(26,138,160,0.38)"
C_FUSED_QDQ = "rgba(142,68,173,0.90)"
C_RVV_SCALAR = "rgba(231,76,60,0.92)"
C_RVV_MM = "rgba(230,126,34,0.90)"
C_CONV = "rgba(211,84,0,0.70)"
C_ELEM = "rgba(127,140,141,0.55)"
C_DATA = "rgba(189,195,199,0.45)"


def _fade(rgba: str, alpha: float = 0.28) -> str:
    return re.sub(r"[\d.]+\)$", f"{alpha})", rgba)


# -------- Hand-asserted PyTorch top-level decomposition ----------------
#
# Level 1 of the Sankey is 4 top-level nn.Module groups. Each group
# expands into Level 2 (torch-mlir op calls) with integer counts derived
# directly from reading the PyTorch class source.
#
# opu_bench_hybrid (HybridModel) — see models/opu_bench_suite/opu_bench_models.py
#   stem       = Conv2d(3,64,3,s=2) + BN + ReLU + Conv2d(64,128,3,s=2) + BN + ReLU
#   blocks     = ViTBlock × 2  (each: LN + MHA + LN + FFN + GELU + FFN)
#   norm       = LayerNorm(dim)
#   head       = Linear(dim, 16)

HybridDecomp = {
    "display_name": "HybridModel (opu_bench_hybrid)",
    "blocks": {
        "Conv stem": [
            ("torch.aten.conv2d", 2),
            ("torch.aten.batch_norm", 2),
            ("torch.aten.relu", 2),
        ],
        "TransformerBlock ×2": [
            ("torch.aten.layer_norm", 4),  # 2 LN × 2 blocks
            ("torch.aten.linear (Q/K/V/O)", 8),  # 4 linears × 2 blocks
            ("torch.aten.matmul (attention)", 4),  # 2 matmul × 2 blocks
            ("torch.aten.softmax", 2),  # 1 × 2 blocks
            ("torch.aten.linear (FFN)", 4),  # 2 linears × 2 blocks
            ("torch.aten.gelu", 2),
        ],
        "Final LayerNorm": [
            ("torch.aten.layer_norm", 1),
        ],
        "Classifier head": [
            ("torch.aten.linear (head)", 1),
        ],
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
        "linalg.conv_2d_nchw_fchw_q": "Direct conv (unfused)",
        "linalg.quantized_batch_matmul": "batch_matmul (i8)",
        "linalg.quantized_matmul": "matmul (i8)",
        "linalg.generic (reduction)": "reduction / softmax (f32)",
        "linalg.generic (elementwise)": "fused elementwise",
    },
    "gopt_cues": {
        "Direct conv (unfused)": ["conv_2d"],
        "batch_matmul (i8)": ["batch_matmul"],
        "matmul (i8)": ["matmul_like", "matmul"],
        "reduction / softmax (f32)": ["reduction", "softmax"],
        "fused elementwise": ["elementwise", "_encoding_", "generic", "pack"],
    },
}

VitBigDecomp = {
    "display_name": "ViTModel (opu_bench_vit)",
    "blocks": {
        "Patch projection": [
            ("torch.aten.linear (patch)", 1),
        ],
        "TransformerBlock ×2": HybridDecomp["blocks"]["TransformerBlock ×2"],
        "Classifier head": [
            ("torch.aten.linear (head)", 1),
        ],
    },
    "torch_to_linalg": HybridDecomp["torch_to_linalg"],
    "linalg_to_gopt": {k: v for k, v in HybridDecomp["linalg_to_gopt"].items() if k != "linalg.conv_2d_nchw_fchw_q"},
    "gopt_cues": {k: v for k, v in HybridDecomp["gopt_cues"].items() if k != "Direct conv (unfused)"},
}

DECOMPS = {
    "opu_bench_hybrid": HybridDecomp,
    "opu_bench_vit": VitBigDecomp,
}


# -------- Phase dump parsing -------------------------------------------


def phase_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for m in re.finditer(r"linalg\.(\w+)", path.read_text()):
        counts[m.group(1)] += 1
    return counts


# -------- CSV + ASM ----------------------------------------------------


SEGMENT_TERMINAL: dict[str, tuple[str, str]] = {
    "runtime_32x32_tile": ("OPU 32×32 (runtime)", C_OPU_32),
    "encoding_32x32_tile": ("OPU 32×32 (encoded)", C_OPU_32),
    "runtime_16x16_tile": ("OPU 16×16 (runtime)", C_OPU_16),
    "encoding_16x16_tile": ("OPU 16×16 (encoded)", C_OPU_16),
    "runtime_8x8_tile": ("OPU 8×8", C_OPU_8),
    "runtime_narrow_tile": ("OPU narrow (runtime)", C_OPU_NARROW),
    "encoding_narrow_tile": ("OPU narrow (encoded)", C_OPU_NARROW),
    "fused_qdq": ("Fused QDQ (VOPACC)", C_FUSED_QDQ),
    "inline_vopacc": ("Inline VOPACC", C_FUSED_QDQ),
    "rvv_reduction_softmax_norm": ("RVV f32 (Option E)", C_RVV_SCALAR),
    "rvv_matmul": ("RVV matmul", C_RVV_MM),
    "direct_conv": ("Direct conv RVV", C_CONV),
    "elementwise_other": ("Elementwise (RVV)", C_ELEM),
    "data_movement": ("Pack / copy", C_DATA),
}


@dataclass
class DispatchRow:
    symbol: str
    segment: str


def load_dispatch_rows(csv_path: Path, model_key: str) -> list[DispatchRow]:
    out: list[DispatchRow] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r.get("include_in_model") != "1" or r["model_key"] != model_key:
                continue
            out.append(DispatchRow(symbol=r["symbol"], segment=r.get("segment", "")))
    return out


def count_opu_insn_per_function(asm_path: Path) -> dict[str, int]:
    """Parse a linked ELF .s. Use the full label (matches CSV `symbol`)."""
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


# -------- Sankey builder -----------------------------------------------


def build_sankey(model_key: str, model_dir: Path, csv_path: Path) -> go.Figure:
    dec = DECOMPS.get(model_key)
    if dec is None:
        raise SystemExit(f"No decomposition asserted for {model_key!r}")

    phases = model_dir / "phases"
    stem = next(iter(phases.glob("*.1.input.mlir")), None)
    if stem is None:
        raise SystemExit(f"No phase dumps in {phases}")
    prefix = stem.stem.rsplit(".1.input", 1)[0]
    p1 = phase_counts(phases / f"{prefix}.1.input.mlir")
    p4 = phase_counts(phases / f"{prefix}.4.global-optimization.mlir")

    dispatches = load_dispatch_rows(csv_path, model_key)
    asm_path = next((model_dir / "files").glob("*linked*riscv_64.s"), None)
    asm_opu_insn = count_opu_insn_per_function(asm_path) if asm_path else {}

    labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    idx: dict[str, int] = {}
    edges: list[tuple[int, int, float, str]] = []

    def node(name: str, color: str, x: float) -> int:
        if name not in idx:
            idx[name] = len(labels)
            labels.append(name)
            node_colors.append(color)
            node_x.append(x)
        return idx[name]

    def link(src: str, tgt: str, val: float, c_src: str, c_tgt: str, x_src: float, x_tgt: float) -> None:
        if val <= 0:
            return
        s = node(src, c_src, x_src)
        t = node(tgt, c_tgt, x_tgt)
        edges.append((s, t, val, _fade(c_tgt, 0.30)))

    # 5 columns. Right edge kept at 0.92 so the terminal column labels
    # (which Plotly draws to the right of the node) fit within the canvas.
    X = [0.03, 0.27, 0.50, 0.73, 0.92]

    # Column 1 → 2 : PyTorch top-level block → torch.aten op
    torch_totals: Counter[str] = Counter()
    for block, ops in dec["blocks"].items():
        for op, n in ops:
            link(block, op, n, C_L1, C_L2, X[0], X[1])
            torch_totals[op] += n

    # Column 2 → 3 : torch.aten → linalg
    linalg_totals: Counter[str] = Counter()
    for op, n in torch_totals.items():
        lin = dec["torch_to_linalg"][op]
        link(op, lin, n, C_L2, C_L3, X[1], X[2])
        linalg_totals[lin] += n

    # Column 3 → 4 : linalg → global-opt bucket
    gopt_totals: Counter[str] = Counter()
    for lin, n in linalg_totals.items():
        bucket = dec["linalg_to_gopt"][lin]
        link(lin, bucket, n, C_L3, C_L4, X[2], X[3])
        gopt_totals[bucket] += n

    # Column 4 → 5 : global-opt → OPU terminal (split each g-opt bucket
    # proportionally across the segments of its dispatches)
    bucket_seg: dict[str, Counter[str]] = {bucket: Counter() for bucket in gopt_totals}
    seg_opu_insn: Counter[str] = Counter()
    seg_dispatches: Counter[str] = Counter()

    def bucket_for_symbol(sym: str) -> str | None:
        s = sym.lower()
        for bucket, cues in dec["gopt_cues"].items():
            if any(cue in s for cue in cues):
                return bucket
        return None

    for rec in dispatches:
        b = bucket_for_symbol(rec.symbol) or "fused elementwise"
        bucket_seg.setdefault(b, Counter())[rec.segment] += 1
        seg_dispatches[rec.segment] += 1
        seg_opu_insn[rec.segment] += asm_opu_insn.get(rec.symbol, 0)

    for bucket, total in gopt_totals.items():
        segs = bucket_seg.get(bucket, Counter())
        if not segs:
            continue
        n_disp = sum(segs.values())
        for seg, k in segs.items():
            term_label, term_color = SEGMENT_TERMINAL.get(seg, (seg, "rgba(127,140,141,0.7)"))
            link(bucket, term_label, total * k / n_disp, C_L4, term_color, X[3], X[4])

    # Annotate linalg / gopt nodes with grepped phase counts.
    lin_caption = {
        "linalg.conv_2d_nchw_fchw_q": f"linalg.conv_2d_nchw_fchw_q  ({p1['conv_2d_nchw_fchw_q']})",
        "linalg.quantized_batch_matmul": f"linalg.quantized_batch_matmul  ({p1['quantized_batch_matmul']})",
        "linalg.quantized_matmul": f"linalg.quantized_matmul  ({p1['quantized_matmul']})",
        "linalg.generic (reduction)": "linalg.generic reduction  (~15 bodies)",
        "linalg.generic (elementwise)": f"linalg.generic elementwise  ({p1['generic']} bodies)",
    }
    for orig, annotated in lin_caption.items():
        if orig in idx:
            labels[idx[orig]] = annotated

    gopt_caption = {
        "batch_matmul (i8)": f"batch_matmul i8  ({p4['batch_matmul']})",
        "matmul (i8)": f"matmul i8  ({p4['matmul']})",
        "reduction / softmax (f32)": f"reduction + softmax  ({p4['softmax']} softmax)",
        "fused elementwise": f"fused elementwise  ({p4['generic']} generics)",
    }
    for orig, annotated in gopt_caption.items():
        if orig in idx:
            labels[idx[orig]] = annotated

    # Terminal OPU annotation: dispatch count + OPU opcode count from .s.
    opu_insn_per_term: Counter[str] = Counter()
    disp_per_term: Counter[str] = Counter()
    for seg, k in seg_dispatches.items():
        term, _ = SEGMENT_TERMINAL.get(seg, (seg, ""))
        disp_per_term[term] += k
        opu_insn_per_term[term] += seg_opu_insn.get(seg, 0)
    for term, n_disp in disp_per_term.items():
        if term not in idx:
            continue
        opc = opu_insn_per_term[term]
        if (term.startswith("OPU") or "VOPACC" in term) and opc > 0:
            labels[idx[term]] = f"{term}  ({n_disp} disp · {opc} OPU ops)"
        else:
            labels[idx[term]] = f"{term}  ({n_disp} disp)"

    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]
    values = [e[2] for e in edges]
    link_colors = [e[3] for e in edges]

    # Paper-tuned layout for 1600×900 PNG.
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=26,
                    thickness=24,
                    line=dict(color="rgba(44,62,80,0.6)", width=0.6),
                    label=labels,
                    color=node_colors,
                    x=node_x,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
                textfont=dict(size=17, color="#2c3e50", family="Helvetica"),
            )
        ]
    )

    opu_disp = sum(
        k for seg, k in seg_dispatches.items() if seg.endswith("_tile") or seg in ("fused_qdq", "inline_vopacc")
    )
    total_disp = sum(seg_dispatches.values())
    fig.update_layout(
        title=dict(
            text=(
                f"<b style='font-size:22px'>{dec['display_name']}</b>  "
                f"<span style='color:#7f8c8d;font-size:14px'>"
                f"· {sum(p1.values())} phase-1 ops · "
                f"{total_disp} dispatches · "
                f"<b style='color:#16a085'>{opu_disp}/{total_disp}</b> OPU-matched</span>"
            ),
            x=0.5,
            xanchor="center",
            y=0.965,
            yanchor="top",
            font=dict(color="#2c3e50", family="Helvetica"),
        ),
        annotations=[
            dict(
                x=x,
                y=1.015,
                xref="paper",
                yref="paper",
                text=f"<b>{name}</b>",
                showarrow=False,
                font=dict(size=13, color="#5d6d7e", family="Helvetica"),
                xanchor="center",
            )
            for x, name in zip(
                X,
                [
                    "PyTorch block",
                    "torch-mlir op",
                    "linalg op (Phase 1)",
                    "Global-Opt (Phase 4)",
                    "Saturn OPU terminal",
                ],
            )
        ],
        font=dict(size=16, family="Helvetica", color="#2c3e50"),
        width=1600,
        height=900,
        margin=dict(l=14, r=14, t=140, b=22),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def _write_figure(fig: go.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{name}.html"
    png = out_dir / f"{name}.png"
    fig.write_html(str(html))
    fig.write_image(str(png), width=1600, height=900, scale=2)
    print(f"  html: {html}  ({html.stat().st_size / 1024:.0f} KB)")
    print(f"  png : {png}   ({png.stat().st_size / 1024:.0f} KB)")
    return png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="opu_bench_hybrid", choices=list(DECOMPS.keys()))
    parser.add_argument("--dispatch-csv", type=Path, default=BENCH_DIR / "model_dispatch_decomposition.csv")
    parser.add_argument("--out-dir", type=Path, default=BENCH_DIR / "figures")
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    fig = build_sankey(args.model, MODEL_PATHS[args.model], args.dispatch_csv)
    _write_figure(fig, args.out_dir, args.out_name or f"sankey_{args.model}")


if __name__ == "__main__":
    main()
