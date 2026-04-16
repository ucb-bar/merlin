"""Shared matplotlib palette for Saturn OPU paper figures."""

from __future__ import annotations

# One-column paper width.
COLUMN_WIDTH_IN: float = 3.5
TEXT_WIDTH_IN: float = 7.2

# Optimization journey after removing the uninformative initial mmt4d+OPU step.
JOURNEY_COLORS: list[str] = [
    "#d62728",  # RVV baseline
    "#e8b800",  # fixed tile query
    "#2ca02c",  # K-loop unroll
    "#17becf",  # const weights
    "#1f77b4",  # encoding resolver
]

RUNTIME_SUPPORT_COLOR = "#8e9aae"  # muted blue-grey — linked-in lib helpers


SEGMENT_COLORS: dict[str, str] = {
    # Confirmed OPU paths. Tile quality is encoded by alpha in SEGMENT_ALPHA:
    # darker = bigger tile = higher utilization.
    "encoding_32x32_tile": JOURNEY_COLORS[4],
    "encoding_16x16_tile": JOURNEY_COLORS[4],
    "encoding_narrow_tile": JOURNEY_COLORS[4],
    "encoding_other_tile": JOURNEY_COLORS[4],
    "runtime_32x32_tile": JOURNEY_COLORS[2],
    "runtime_16x16_tile": JOURNEY_COLORS[2],
    "runtime_8x8_tile": "#e8b800",
    "runtime_narrow_tile": "#e8b800",
    "runtime_other_tile": JOURNEY_COLORS[2],
    "fused_qdq": "#6a51a3",
    # Same 16×16 outer-product VOPACC as encoding_16x16_tile — just emitted
    # via a different compile path (LLVM pattern inside a generic vector
    # matmul body, instead of the structured encoding-resolver body).
    # Hardware-identical → same color.
    "inline_vopacc": JOURNEY_COLORS[4],
    # Non-OPU compute paths.
    "direct_conv": "#ff7f0e",
    "rvv_matmul": JOURNEY_COLORS[0],
    "rvv_reduction_softmax_norm": "#8f8f8f",
    # Non-OPU sub-categories. Quant/dequant share one yellow since they
    # are the same class of op (dtype conversion at matmul boundaries);
    # fused BN/scale/bias gets a deeper amber to stay in the yellow
    # family but stand out. Activation + transpose get muted non-grey
    # tones so the reader can read them at a glance without confusing
    # them with generic grey "scalar/other" residue.
    "quantize_f32_to_i8": "#f1c40f",  # yellow — quant/dequant family
    "dequantize_i8_to_f32": "#f1c40f",  # same — merges in legend
    "requantize_i32_to_i8": "#f1c40f",  # same — merges in legend
    "elementwise_multi_dtype": "#c9900c",  # deep amber — fused BN/scale/bias
    "activation": "#a4b493",  # muted sage — activation (ReLU/GELU)
    "transpose_reshape": "#b99a9a",  # muted dusty rose — transpose/reshape
    "elementwise_other": "#95a5a6",  # neutral grey — unmatched residue
    "data_movement": "#d7dbdd",  # very light grey — pack/unpack
    "runtime_support": "#8e9aae",  # muted blue-grey — linked-in helpers
}

SEGMENT_ALPHA: dict[str, float] = {
    # Darker (closer to 1.0) = larger tile = more OPU utilization.
    "encoding_32x32_tile": 1.0,
    "encoding_16x16_tile": 0.80,
    "encoding_8x8_tile": 0.60,
    "encoding_narrow_tile": 0.40,
    "encoding_other_tile": 0.55,
    "runtime_32x32_tile": 1.0,
    "runtime_16x16_tile": 0.80,
    "runtime_8x8_tile": 0.60,
    "runtime_narrow_tile": 0.40,
    "runtime_other_tile": 0.55,
    "fused_qdq": 0.9,
    "inline_vopacc": 0.80,  # same alpha as encoding_16x16_tile
}

SEGMENT_ORDER: list[str] = [
    "encoding_32x32_tile",
    "encoding_16x16_tile",
    "encoding_narrow_tile",
    "encoding_other_tile",
    "runtime_32x32_tile",
    "runtime_16x16_tile",
    "runtime_8x8_tile",
    "runtime_narrow_tile",
    "runtime_other_tile",
    "fused_qdq",
    "inline_vopacc",
    "rvv_matmul",
    "direct_conv",
    "rvv_reduction_softmax_norm",
    "quantize_f32_to_i8",
    "dequantize_i8_to_f32",
    "requantize_i32_to_i8",
    "elementwise_multi_dtype",
    "activation",
    "transpose_reshape",
    "elementwise_other",
    "data_movement",
    "runtime_support",
]

SEGMENT_LABELS: dict[str, str] = {
    "encoding_32x32_tile": "AOT 32×32 OPU",
    "encoding_16x16_tile": "AOT 16×16 OPU",
    "encoding_narrow_tile": "AOT Mx16 OPU",
    "encoding_other_tile": "AOT other OPU",
    "runtime_32x32_tile": "Runtime 32×32 OPU",
    "runtime_16x16_tile": "Runtime 16×16 OPU",
    "runtime_8x8_tile": "Runtime 8×8 OPU",
    "runtime_narrow_tile": "Runtime Mx16 OPU",
    "runtime_other_tile": "Runtime other OPU",
    "fused_qdq": "Fused QDQ OPU",
    # inline_vopacc = same 16×16 outer-product VOPACC as AOT 16×16, just
    # emitted by the LLVM vector-contract pattern instead of the encoding
    # resolver. Identical hardware work → same label → legend dedupes.
    "inline_vopacc": "AOT 16×16 OPU",
    "rvv_matmul": "RVV matmul",
    "direct_conv": "Direct conv (RVV)",
    "rvv_reduction_softmax_norm": "Reduction / softmax / norm",
    # Granular non-OPU elementwise categories. Quant/Dequant/Requant
    # share a single legend entry (they're the same quant-conversion
    # family); fused BN/scale/bias kept separate (distinct purpose).
    "quantize_f32_to_i8": "Quant / Dequant",
    "dequantize_i8_to_f32": "Quant / Dequant",
    "requantize_i32_to_i8": "Quant / Dequant",
    "elementwise_multi_dtype": "Fused BN / scale / bias",
    "activation": "Activation (ReLU/GELU)",
    "transpose_reshape": "Transpose / reshape",
    "elementwise_other": "Other elementwise",
    "data_movement": "Pack / unpack",
    "runtime_support": "Runtime support (linked)",
}

OPU_SEGMENTS: set[str] = {
    "encoding_32x32_tile",
    "encoding_16x16_tile",
    "encoding_narrow_tile",
    "encoding_other_tile",
    "runtime_32x32_tile",
    "runtime_16x16_tile",
    "runtime_8x8_tile",
    "runtime_narrow_tile",
    "runtime_other_tile",
    "fused_qdq",
    "inline_vopacc",
    # direct_conv is pure RVV spatial loops — NOT OPU
}

JOURNEY_FIGSIZE: tuple[float, float] = (COLUMN_WIDTH_IN, 1.85)
DECOMPOSITION_BASE_HEIGHT: float = 1.05
DECOMPOSITION_ROW_HEIGHT: float = 0.25


def apply_paper_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
