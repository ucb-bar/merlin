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
from pathlib import Path
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
    # SHARED, TARGET-AGNOSTIC accumulator-residency concept. "Does the output accumulator stay in
    # the fastest storage (vector regs / PE array / on-chip buffer) across the WHOLE reduction, and
    # commit ONCE after it?" — the property that distinguishes an expert micro-kernel from a lowering
    # that round-trips the accumulator through memory every reduction tile. It is the same question
    # on every backend, so it lives on ComputeFacet (RVV reads it from the asm via a vfmacc-chain /
    # no-in-loop-accumulator-spill analysis; SpatialFacet.accumulator_resident is the Gemmini-PE
    # view of the SAME concept; an NPU lifter reads it from on-chip-buffer residency). None = the
    # lifter could not determine it (e.g. a straight-line / fully-unrolled region with no loop).
    accumulator_resident: bool | None = None
    # Does the inner output width (NR) track the runtime vsetvlmax (VL-adaptive, scalable) rather
    # than a compile-time-fixed tile? True iff the kernel uses a vsetvli VL-loop (vl_strategy).
    nr_is_vsetvlmax: bool | None = None


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


# --- asm-level structural inference of the expert-win properties (no regex; read from the stream) -

# Whole-vector-register spill ops: vsNr.v stores / vlNre.v loads of a whole vector register group.
# When one of these touches the ACCUMULATOR inside the reduction loop, the accumulator is being
# round-tripped through (stack) memory each iteration — the opposite of register-resident.
_ACC_SPILL_STORE = ("vs1r", "vs2r", "vs4r", "vs8r")
_ACC_SPILL_LOAD = ("vl1re", "vl2re", "vl4re", "vl8re")
_FMA = ("vfmacc", "vmacc", "vfwmacc", "vwmacc")


def _fma_loop(stream):
    """The reduction loop where the MAC chain lives: among back-edge spans that contain at least one
    fma, the one with the MOST fma (the register-blocked MR-accumulator chain), breaking ties toward
    the TIGHTEST span (the innermost K-reduction loop, not an enclosing M/N loop). A register block
    of MR=1 (e.g. XNNPACK 1x4v) legitimately has one fma per K step, so the threshold is >=1, not
    >=2. Returns the (lo, hi) span or None if the region is straight-line (no loop with an fma)."""
    spans = stream.loop_spans()
    fma_spans = [(sp, stream.count_in(sp, *_FMA)) for sp in spans]
    fma_spans = [(sp, n) for sp, n in fma_spans if n >= 1]
    if not fma_spans:
        return None
    # most fma first, then tightest span (the register-block kernel loop)
    fma_spans.sort(key=lambda x: (-x[1], x[0][1] - x[0][0]))
    return fma_spans[0][0]


def _infer_accumulator_resident(stream) -> bool | None:
    """Read accumulator-residency from the decoded stream, target-agnostic in spirit, RVV in detail.

    RESIDENT (True): a vfmacc MAC chain runs in the reduction loop and the accumulator is NEVER
    spilled inside it — no whole-register acc store (vsNr.v) and no acc element-store (vse) of the
    MAC destination inside the loop body. The C accumulator stays in the vreg group across the whole
    reduction and commits once AFTER the loop (the expert micro-kernel shape).
    NOT RESIDENT (False): the MAC loop contains a whole-register acc spill-store/-load pair, i.e. the
    accumulator is loaded+stored THROUGH memory every reduction tile (the lowering's per-K roundtrip).
    None: no multi-fma reduction loop to judge (straight-line / fully-unrolled region).
    """
    sp = _fma_loop(stream)
    if sp is None:
        return None
    # accumulator round-trip = a whole-register spill store INSIDE the fma loop. (The expert stores
    # the accumulator once AFTER the loop; the non-resident lowering stores it inside, every step.)
    spill_store = stream.count_in(sp, *_ACC_SPILL_STORE)
    spill_load = stream.count_in(sp, *_ACC_SPILL_LOAD)
    return not (spill_store > 0 and spill_load > 0)


def _infer_register_block(stream, sew, lmul) -> tuple | None:
    """(MR, NR) register block read from the asm. MR = number of DISTINCT accumulator vregs fed by
    the broadcast MAC (vfmacc.vf — the register-blocking idiom: MR rows broadcast as scalars into MR
    accumulators). NR = effective vector lanes of the MAC's vtype (VLEN unknown at decode -> the
    LMUL-scaled width relative to SEW, expressed as a *lane* count via the standard VLEN guess). If
    there is no vfmacc.vf register block, MR falls back to the count of distinct accumulator dests of
    any fma in the loop. Returns None when there is no fma loop."""
    sp = _fma_loop(stream)
    if sp is None:
        return None
    acc_dests_vf: set[str] = set()
    acc_dests_any: set[str] = set()
    for i in stream.insns_in(sp):
        m = i.raw.mnemonic
        if any(m.startswith(p) for p in _FMA) and i.raw.operands:
            dest = i.raw.operands[0]
            acc_dests_any.add(dest)
            if m.startswith(("vfmacc.vf", "vmacc.vx")):   # broadcast (register-blocking) form
                acc_dests_vf.add(dest)
    mr = len(acc_dests_vf) or len(acc_dests_any) or None
    # NR = lanes in the accumulator vreg group = VLEN/SEW * LMUL. VLEN is a target constant not in
    # the asm; we report NR as the LMUL-relative lane multiplier (the shape the comparator/action
    # consume) rather than a fabricated absolute, keyed by SEW. With VLEN=256 (the mined K1 target)
    # and SEW given, callers can resolve NR = 256/sew*lmul; here we leave NR symbolic as lmul-scaled.
    nr = None
    if sew and lmul:
        nr = ("vsetvlmax", lmul)  # lane group is lmul-scaled vsetvlmax (VLEN-dependent, scalable)
    return (mr, nr) if mr else None


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
    # The expert-win properties, inferred structurally from the InsnStream (the gap the CCA used to
    # be blind to on the RVV path):
    acc_resident = _infer_accumulator_resident(stream)
    reg_block = _infer_register_block(stream, sew, lmul)
    # NR tracks vsetvlmax exactly when the kernel uses a polymorphic vsetvli VL-loop (VL-adaptive).
    nr_is_vsetvlmax = (vl_strategy == "vsetvl_loop") if contraction == "fused_fma" else None
    return CCA(
        op=op, backend=[backend],
        compute=ComputeFacet(
            op=op, contraction_form=contraction,
            widening=widening,
            reduction_form=("vredsum_tree" if reduce_n > 0 else "none"),
            epilogue=("requant_narrow" if narrow > 0 else "none"),
            register_block=reg_block,
            accumulator_resident=acc_resident,
            nr_is_vsetvlmax=nr_is_vsetvlmax,
        ),
        vector=VectorFacet(sew=sew, lmul=lmul, vl_strategy=vl_strategy),
        provenance={"level": "asm", "source": source, "confidence": "high"},
    )


def lift_source(facts, *, op: str, source: str, backend: str = "rvv") -> CCA:
    """Source-level lift from typed C-intrinsic facts (decode.clang_ast.SourceFacts) — the
    cross-check for the asm lift. Reads decisions from RESOLVED intrinsic types, not substrings."""
    sew, lmul = facts.dominant_vtype()
    contraction = ("fused_fma" if facts.has("vfmacc", "vmacc", "vfwmacc", "vwmacc")
                   else "mul_add" if (facts.has("vfmul", "vmul") and facts.has("vfadd", "vadd"))
                   else None)
    return CCA(
        op=op, backend=[backend],
        compute=ComputeFacet(op=op, contraction_form=contraction,
                             widening=facts.has("vwmacc", "vfwmacc") > 0,
                             reduction_form=("vredsum_tree" if facts.has("vredsum", "vfredusum") else None),
                             epilogue=("requant_narrow" if facts.has("vnclip", "vfncvt") else None)),
        vector=VectorFacet(sew=sew, lmul=lmul) if backend == "rvv" else None,
        provenance={"level": "source_ast", "source": source, "confidence": "medium"},
    )


def lift_spatial(op_counts: dict, *, op: str, source: str,
                 dataflow: str | None = None, pe_rows: int | None = None,
                 pe_cols: int | None = None, backend: str = "gemmini") -> CCA:
    """Spatial/systolic (Gemmini) lifter — fills the SPATIAL facet from decoded accelerator ops
    (e.g. targetgen.rocc_decode counts of preload/compute/mvin/mvout). Keeps the same CCA schema
    so a gemmini region compares against a gemmini expert just like RVV does for vector."""
    return CCA(
        op=op, backend=[backend],
        compute=ComputeFacet(op=op, contraction_form="systolic",
                             accumulator_dtype=op_counts.get("acc_dtype"),
                             widening=bool(op_counts.get("widening"))),
        spatial=SpatialFacet(pe_rows=pe_rows, pe_cols=pe_cols, dataflow=dataflow,
                             accumulator_resident=op_counts.get("acc_resident")),
        provenance={"level": "asm", "source": source, "confidence": "high"},
    )


def lift_npu(engine_ops: list[str], *, op: str, source: str,
             dma_pattern: str | None = None, backend: str = "npu") -> CCA:
    """NPU lifter — fills the DATAFLOW facet (engine ops + DMA). A region may pair this backend
    with rvv in a composite CCA (backend=['npu','rvv'])."""
    return CCA(
        op=op, backend=[backend],
        compute=ComputeFacet(op=op),
        dataflow=DataflowFacet(engine_ops=list(engine_ops), dma_pattern=dma_pattern),
        provenance={"level": "asm", "source": source, "confidence": "medium"},
    )


def particularities() -> dict:
    """Load the per-target runtime/ABI particularities (bf16 ABI reg class, VLEN, vsetvl
    semantics, fp-contract default) so the comparator can normalize runtime artifacts out."""
    import yaml
    p = Path(__file__).resolve().parent / "runtime_particularities.yaml"
    return yaml.safe_load(p.read_text()) if p.is_file() else {}


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
