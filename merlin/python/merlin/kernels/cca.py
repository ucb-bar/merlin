"""Common Compute Abstraction (CCA) — the shareable, **target-agnostic** vocabulary every source
(framework kernel, our compiler output, DSE view) decompiles into, at any level (asm | source_ast
| mlir | dse).

NOT RVV-specific. A region carries a ``backend`` tag (``scalar|rvv|gemmini|npu|…``, or a list for
a composite region like NPU+RVV) and only the relevant **facets** are populated:
- ``compute`` (target-agnostic): op, contraction form, accumulator, widening, reduction, epilogue.
- ``vector`` (RVV/SIMD): SEW, LMUL, VL strategy, tail.
- ``spatial`` (Gemmini/systolic): PE-array dims, dataflow, accumulator residency.
- ``dataflow`` (NPU): engine ops, DMA pattern, on-chip-buffer residency.

Lifters reconstruct a CCA from each level (``lift_asm`` is primary, per-ISA). ``cca_agree``
cross-checks two CCAs (e.g. source-lifted vs asm-lifted) per populated facet field — the
"good reconstruction" validity gate. RVV fills the ``vector`` facet first; other targets add
their facets + lifters behind this same schema without a rewrite.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ComputeFacet:
    op: str | None = None
    contraction_form: str | None = None      # fused_fma | mul_add | outerproduct | dot | systolic
    accumulator_dtype: str | None = None      # f32 | i32 | f64 | ...
    widening: bool | None = None              # i8xi8->i32 widening MAC (vwmacc / array widen)
    reduction_form: str | None = None         # tree | vredsum | vfredusum | none
    register_block: tuple | None = None       # (mr, nr) or tile dims
    epilogue: str | None = None               # requant_narrow | none


@dataclass
class VectorFacet:
    sew: int | None = None
    lmul: float | None = None
    vl_strategy: str | None = None            # vsetvl_loop | vsetivli_fixed
    tail: str | None = None                   # ta | tu | none


@dataclass
class SpatialFacet:                            # Gemmini / systolic (populated by a future lifter)
    pe_rows: int | None = None
    pe_cols: int | None = None
    dataflow: str | None = None               # ws | os
    accumulator_resident: bool | None = None


@dataclass
class DataflowFacet:                           # NPU (populated by a future lifter)
    engine_ops: list[str] = field(default_factory=list)
    dma_pattern: str | None = None
    onchip_resident: str | None = None


@dataclass
class CCA:
    op: str
    backend: list[str]                         # ["rvv"], or ["npu","rvv"] for a composite region
    compute: ComputeFacet = field(default_factory=ComputeFacet)
    vector: VectorFacet | None = None
    spatial: SpatialFacet | None = None
    dataflow: DataflowFacet | None = None
    provenance: dict[str, Any] = field(default_factory=dict)   # level, source, confidence

    def to_dict(self) -> dict:
        return asdict(self)


# ---- lifters ------------------------------------------------------------------------

def _dominant_vtype(stream) -> tuple[int | None, float | None]:
    hist = stream.vtype_histogram()
    if not hist:
        return None, None
    top = max(hist.items(), key=lambda kv: kv[1])[0]   # e.g. "e32m2tama"
    sew = lmul = None
    if top.startswith("e"):
        i = 1
        while i < len(top) and top[i].isdigit():
            i += 1
        sew = int(top[1:i]) if i > 1 else None
        rest = top[i:]
        if rest.startswith("mf"):
            j = 2
            while j < len(rest) and rest[j].isdigit():
                j += 1
            lmul = 1.0 / int(rest[2:j]) if j > 2 else None
        elif rest.startswith("m"):
            j = 1
            while j < len(rest) and rest[j].isdigit():
                j += 1
            lmul = float(int(rest[1:j])) if j > 1 else None
    return sew, lmul


def lift_asm(stream, *, op: str, source: str, backend: str = "rvv") -> CCA:
    """Primary lifter: RVV/vector ``InsnStream`` (from ``decode.rvv``) -> CCA.

    Everything is read from the decoded instruction stream + tracked vtype — never guessed from a
    source substring. Other ISAs get their own ``lift_asm`` over their decoder's stream.
    """
    vfmacc = stream.count("vfmacc", "vmacc")
    vfmul = stream.count("vfmul", "vmul")
    vfadd = stream.count("vfadd", "vadd")
    widening = stream.count("vwmacc") > 0
    reduce_n = stream.count("vredsum", "vfredusum", "vfredsum", "vredosum")
    narrow = stream.count("vnclip", "vfncvt", "vncvt")
    sew, lmul = _dominant_vtype(stream)
    # VL strategy: vsetvli (register VL, polymorphic) in a loop = vsetvl_loop; else vsetivli fixed.
    has_setvli = stream.count("vsetvli") > 0
    vl_strategy = "vsetvl_loop" if (has_setvli and stream.has_loop()) else "vsetivli_fixed"
    contraction = ("fused_fma" if vfmacc > 0
                   else "mul_add" if (vfmul > 0 and vfadd > 0)
                   else None)
    return CCA(
        op=op, backend=[backend],
        compute=ComputeFacet(
            op=op, contraction_form=contraction,
            widening=widening,
            reduction_form=("vredsum_tree" if reduce_n > 0 else "none"),
            epilogue=("requant_narrow" if narrow > 0 else "none"),
        ),
        vector=VectorFacet(sew=sew, lmul=lmul, vl_strategy=vl_strategy),
        provenance={"level": "asm", "source": source, "confidence": "high"},
    )


def lift_dse(op_shape, *, source: str = "dse") -> CCA:
    """Partial lift from the DSE operator-geometry view (target-agnostic geometry + role).

    Reuses dse_guidance.operator_geometry.OperatorShape: gives op + register block hints, not the
    vector/spatial micro-decisions (those live at asm). Provenance flags the partial level.
    """
    op = getattr(op_shape, "semantic_class", None) or getattr(op_shape, "op", None) or "unknown"
    mnk = (getattr(op_shape, "M", None), getattr(op_shape, "N", None), getattr(op_shape, "K", None))
    return CCA(
        op=str(op), backend=[],
        compute=ComputeFacet(op=str(op), register_block=mnk if any(mnk) else None),
        provenance={"level": "dse", "source": source, "confidence": "low"},
    )


# ---- cross-level agreement (the validity gate) --------------------------------------

@dataclass
class AgreementReport:
    agree: bool
    disagreements: list[str] = field(default_factory=list)
    compared_fields: list[str] = field(default_factory=list)


def _facet_fields(a, b) -> dict[str, tuple]:
    """Pairs of (a_value, b_value) for fields BOTH populated (None on either side = not compared)."""
    out = {}
    if a is None or b is None:
        return out
    for k, va in asdict(a).items():
        vb = asdict(b).get(k)
        if va is not None and vb is not None:
            out[k] = (va, vb)
    return out


def cca_agree(a: CCA, b: CCA) -> AgreementReport:
    """Compare two CCAs (e.g. source-lifted vs asm-lifted) per populated facet field. A kernel
    whose levels disagree is quarantined from policy promotion until reconciled."""
    diffs: list[str] = []
    compared: list[str] = []
    for facet in ("compute", "vector", "spatial", "dataflow"):
        fa, fb = getattr(a, facet), getattr(b, facet)
        for k, (va, vb) in _facet_fields(fa, fb).items():
            compared.append(f"{facet}.{k}")
            if va != vb:
                diffs.append(f"{facet}.{k}: {a.provenance.get('level')}={va!r} vs "
                             f"{b.provenance.get('level')}={vb!r}")
    return AgreementReport(agree=not diffs, disagreements=diffs, compared_fields=compared)
