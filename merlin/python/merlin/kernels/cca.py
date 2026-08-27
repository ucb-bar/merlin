"""Common Compute Abstraction (CCA) — the shareable, **target-agnostic** vocabulary every source
(framework kernel, our compiler output, DSE view) decompiles into, at any level (asm | source_ast
| mlir | dse).

NOT RVV-specific. A region carries a ``backend`` tag and only the relevant **facets** are populated.
Which facets those are is a property of the target's ENGINES, not of its name: a target is a SET of
compute engines (see :mod:`merlin.kernels.engines`), and each engine-scoped facet is populated exactly
when the target has that engine. Measured: 62% of atlas's engine-driving expert kernels touch its
vector engine, so describing it by one facet because it is "an NPU" would leave most of its corpus
unexpressed.

ENGINE-SCOPED — populated per :func:`merlin.kernels.engines.facet_families_for`:
- ``vector`` (a lane engine: RVV, or an accelerator's VPU): SEW, LMUL, VL strategy, tail.
- ``spatial`` (an array engine, systolic OR outer-product): PE-array dims, dataflow, accumulator
  residency. The two datapaths share this facet and differ in ``dataflow``'s VALUE.
- ``simt`` (a threads-of-control engine): warps, shared-memory residency, barriers, divergence.

ENGINE-AGNOSTIC — meaningful for any target:
- ``compute``: op, contraction form, accumulator, widening, reduction, epilogue.
- ``memory``: access pattern, panel reuse, operand broadcast.
- ``envelope``: what is emitted AROUND the loop.
- ``coverage``: whole-model claim (model-scoped, not per-kernel).

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
    # How a transcendental activation (GELU/sigmoid/SiLU/tanh) is evaluated: "vectorized_polynomial"
    # (an inline minimax vfmacc chain, no libm call) vs "scalar_libm_call" (a per-element convert-math-
    # to-libm call loop). None = the region is not an activation. Captured here so the CCA SEES this
    # gap (it used to be injected out-of-band); the compiler exposes it via
    # impr_features:vectorized_transcendental_activation.
    activation_vectorization: str | None = None
    # Does the register block's M extent SHRINK for a small-M region rather than staying at the tile
    # size? Routed in the action catalog with no field behind it, so the divergence could never be
    # raised. The measured case is a decode step at M=1 on a wide tile: correct, busy, and using one
    # row of the block.
    mr_adapts_to_m: bool | None = None


@dataclass
class VectorFacet:
    sew: int | None = None
    lmul: float | None = None
    vl_strategy: str | None = None            # vsetvl_loop | vsetivli_fixed
    tail: str | None = None                   # ta | tu | none


@dataclass
class MemoryFacet:                             # data-movement / packing (the #1 expert GEMM lever)
    # How the inner-loop operands are fetched — the packing/layout story the expert wins on (e.g.
    # XNNPACK's goi-prepacked contiguous B panel streamed by pointer-advance vs our strided
    # model-layout gather). "unit_stride" = packed contiguous panels (one vle per K, reused across the
    # MR accumulators); "strided" = vlse model-layout; "indexed" = gather. Lifted from decode.memory.
    access_pattern: str | None = None          # unit_stride | strided | indexed | none
    # Is one loaded operand panel REUSED across the MR register-block accumulators (the loads/FMA
    # amortization — expert ~1.1, unblocked baseline ~2.0)? True iff the K-loop broadcasts a single
    # vector load across multiple fma accumulators (the .vf register-block idiom).
    panel_reuse: bool | None = None
    # Is the A operand broadcast via vfmacc.vf (no per-step A-vector rebuild ladder) vs rebuilt with
    # vslideup/vrgather each step? True = .vf broadcast (a_broadcast_per_fma == 0).
    a_broadcast_vf: bool | None = None
    # --- ON-CHIP RESIDENCY: does the working set FIT, and what happens when it does not? ---
    # A compile-time feasibility predicate, not a performance hint. Radiance's SPAD_DEST() is exactly
    # this, and failing it produced 0/16384 correct elements SILENTLY -- the kernel ran, wrote nothing
    # correct, and nothing in the CCA could say why. `capacity_fit` measures the BACKEND's residency
    # strategy against the target's declared capacity, so a lowering that overruns is a divergence
    # rather than a mystery.
    capacity_fit: bool | None = None
    onchip_bytes_required: int | None = None   # METRIC: what the region's working set needs
    banks_used: int | None = None              # METRIC: distinct on-chip banks the region occupies
    spill_reason: str | None = None            # why it did not fit: operand | accumulator | both | none
    # Folded in from the retired DataflowFacet: bulk movement is data movement, which every target has.
    dma_pattern: str | None = None             # burst | strided | scatter_gather | none
    onchip_resident: str | None = None         # which operand stays on chip: a | b | both | none


@dataclass
class DispatchFacet:
    """HOW MANY commands the region issues to its endpoint, and what shape they take.

    The largest structural gap the CCA had, and it is engine-agnostic: every endpoint is driven by
    something. Gemmini's biggest expert win IS dispatch shape rather than arithmetic — a hardware-loop
    descriptor offloads the whole loop nest, ``matmul_ws`` elides re-preloads it does not need, and
    ``a_spad_id``/``b_spad_id`` double-buffer the scratchpad — and none of that is expressible in
    contraction form, vector config or access pattern. ``features/dispatch.py`` already extracts the
    counts and had nowhere to put them.

    ``n_dispatches`` is a METRIC (a count, not a choice). The rest are choices a schedule makes.
    """

    n_dispatches: int | None = None            # METRIC: commands issued to the endpoint
    config_fraction: float | None = None       # METRIC: share of them that only set state
    descriptor_reuse: bool | None = None       # state set once and inherited, vs re-set per tile
    loop_offloaded: bool | None = None         # a loop nest handed to the endpoint's own sequencer
    double_buffered_banks: int | None = None   # distinct on-chip banks alternated across tiles
    dma_overlap: bool | None = None            # bulk movement issued to overlap with compute


@dataclass
class LayoutFacet:
    """How operands are LAID OUT before the region runs, as opposed to how they are fetched.

    ``layout.transpose_materialized`` is already ROUTED in the action catalog with no backing facet, so
    the divergence it exists to answer could never be raised — a route with nothing able to trigger it.
    Distinct from ``MemoryFacet``, which describes the access pattern of a layout already chosen; this
    describes the choice.
    """

    transpose_materialized: bool | None = None  # a transpose written to memory vs folded into access
    operand_major: str | None = None            # k_major | m_major | n_major
    prepack_required: bool | None = None        # the endpoint needs an offline-packed operand panel


@dataclass
class EnvelopeFacet:                             # the code AROUND the inner loop (prologue/epilogue)
    """What the compiler emits around the compute loop, as opposed to inside it.

    Every other facet describes the inner loop. That left a blind spot: two kernels can agree on
    contraction form, vector config, register block and access pattern -- i.e. look IDENTICAL in CCA
    -- while one of them wraps each tile in a runtime call and the other does not. Measured on K1,
    that blind spot WAS the entire expert gap: our f32 GEMM hot loop is better per-FMA than
    XNNPACK's (3.0 vs 3.82 ins/FMA), but a per-tile ``memrefCopy`` added ~79 instructions per output
    element, which is ~77% of everything retired at N=128. No point in the tiling space could move
    it, which is why a full MR/NR/KC/unroll sweep read flat.

    Lifted structurally from the decoded stream (call sites scoped to loop bodies) plus, when
    available, the object's undefined runtime symbols -- never guessed from a source substring.
    """

    calls_in_loop: int | None = None            # call sites inside a loop body (expert GEMM: 0)
    runtime_calls: tuple[str, ...] | None = None  # runtime helpers the region calls, e.g. memrefCopy
    work_ins_per_mac: float | None = None       # METRIC: the N^3 coefficient (hot-loop efficiency)
    overhead_ins_per_output: float | None = None  # METRIC: the N^2 coefficient (per-tile overhead)


@dataclass
class SpatialFacet:
    """An ARRAY engine: a 2-D grid of cells with an accumulator (populated by a future lifter).

    Covers both array datapaths — a stationary-weight systolic wavefront and a rank-1 outer-product
    accumulate — because what a lifter reads out of the assembly is the same question for both. They
    are separate compute-unit KINDS (they carry different RTL fact families) and they differ here only
    in ``dataflow``'s value, which is the right place for it: on a systolic mesh the choice is a real
    lever, on an outer-product unit it is fixed, and ``leverable_axes`` already expresses "fixed" as
    simply not routing the axis.

    The geometry is what the compiler tiles TO, not something it picks — hence IDENTITY, not LEVER.
    """

    pe_rows: int | None = None
    pe_cols: int | None = None
    dataflow: str | None = None               # ws | os | outer_product
    accumulator_resident: bool | None = None


@dataclass
class SimtFacet:
    """A THREADS-OF-CONTROL engine (populated by a future lifter).

    Distinct from ``vector`` and not reducible to it: a lane engine is element-parallel *within one
    thread of control*, so it has a VL and no divergence; a SIMT engine has threads that can diverge,
    a barrier discipline, and a programmer-managed scratchpad. ``vector.lmul`` has no SIMT analogue
    and barrier placement has no lane analogue, which is why they are two engines rather than one.

    ``threads_per_warp`` is fixed hardware geometry (IDENTITY, like ``pe_rows``); the rest are choices
    the schedule makes and therefore candidate levers.
    """

    warps: int | None = None                  # warps cooperating on this region
    threads_per_warp: int | None = None       # fixed geometry, not a choice
    smem_resident: bool | None = None         # operand tile staged in shared memory across the reduction
    barriers_in_loop: int | None = None       # barriers INSIDE the reduction loop (0 = none needed)
    divergence: str | None = None             # uniform | divergent


@dataclass
class CoverageFacet:
    """WHOLE-MODEL coverage: how much of the work the schedule actually claims.

    Every other facet describes ONE kernel, lifted from its asm. That makes the largest measured
    losses structurally invisible to the CCA, because they are properties of the GRAPH rather than of
    a kernel: an entire contraction op class left for ``convert-linalg-to-loops`` (whisper_tiny's
    1500-wide encoder attention, dropped to scalar by one N=1 decode step in the same class), and the
    non-contraction ops — im2col gathers, pads, norms, elementwise — which the schedule never matches
    and which therefore run scalar on one core.

    A kernel-scoped CCA can compare our matmul to an expert's matmul and find nothing wrong while the
    model runs at a few percent of peak. Measured: spectformer 0.40 MAC/cycle, deepjscc 0.22,
    lstmnetvit 0.067, against ~8 MAC/cycle for a VLEN=128 int8 vwmacc datapath. None of that is
    expressible in ``compute``/``vector``/``memory``, so no divergence was ever raised and no action
    ever routed — the loop could not discover the optimization because it could not see the loss.

    Lifted from the IR + the schedule (no asm), so it works before anything is built.
    """

    #: MACs in contractions the schedule CLAIMS / total MACs in the model. 1.0 = every contraction is
    #: on the tiled+vectorized path; 0.6 = 40% of the arithmetic is in an unclaimed class.
    claimed_mac_fraction: float | None = None
    #: contraction op classes present in the model that the schedule does NOT match, so their
    #: contractions fall through to convert-linalg-to-loops (scalar). Empty = every class is claimed.
    unclaimed_op_classes: tuple[str, ...] = ()
    #: share of the model's ops that are NOT contractions — the elementwise/layout/reduction tail that
    #: the contraction-only schedule leaves scalar. High + low MAC/cycle = the tail dominates.
    non_contraction_op_fraction: float | None = None


@dataclass
class CCA:
    op: str
    backend: list[str]                         # ["rvv"], or ["npu","rvv"] for a composite region
    compute: ComputeFacet = field(default_factory=ComputeFacet)
    vector: VectorFacet | None = None
    memory: MemoryFacet | None = None
    envelope: EnvelopeFacet | None = None
    spatial: SpatialFacet | None = None
    simt: SimtFacet | None = None
    #: HOW the region drives its endpoint (engine-agnostic: every endpoint is driven by something).
    dispatch: DispatchFacet | None = None
    #: How operands are laid out BEFORE the region runs, as opposed to how they are fetched.
    layout: LayoutFacet | None = None
    #: whole-model coverage. Present only on a MODEL-level CCA (lifted from IR + schedule); None on a
    #: per-kernel CCA, where "what fraction of the model is claimed" is not a meaningful question.
    coverage: CoverageFacet | None = None
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


# A scalar activation calls out to a transcendental libm symbol; a vectorized one evaluates an inline
# polynomial. _LIBM_* are the callee symbols (read from resolved call-target operands, not a mnemonic
# guess); _ACTIVATION_OPS give the region context so a matmul is never misclassified as an activation.
_LIBM_TRANSCENDENTAL = ("exp", "erf", "tanh", "sinh", "cosh", "log", "pow")
_ACTIVATION_OPS = ("gelu", "silu", "sigmoid", "tanh", "erf", "exp", "softmax")
_CALL_MNEMONICS = ("jal", "jalr", "call")


def _has_transcendental_libm_call(stream) -> bool:
    """True iff a call instruction targets a transcendental libm symbol (e.g. ``jal ra, <expf>``).
    Reads the resolved call-target operand (the ``<sym>`` objdump renders) — structured, not regex."""
    for i in stream.insns:
        if not any(i.raw.mnemonic.startswith(c) for c in _CALL_MNEMONICS):
            continue
        blob = " ".join(i.raw.operands).lower()
        if any(f"<{s}" in blob for s in _LIBM_TRANSCENDENTAL):
            return True
    return False


def _infer_activation_vectorization(stream, op) -> str | None:
    """A transcendental activation evaluated as a SCALAR libm call loop vs a VECTORIZED minimax
    polynomial (vfmacc chain). None when the region is not an activation (no transcendental op/call),
    so a matmul/plain kernel is never misclassified."""
    trans_call = _has_transcendental_libm_call(stream)
    is_activation = bool(op) and any(s in op.lower() for s in _ACTIVATION_OPS)
    if not (trans_call or is_activation):
        return None
    if trans_call:
        return "scalar_libm_call"
    if stream.count("vfmacc", "vfmul", "vfadd") > 0:   # transcendental evaluated as a vector poly
        return "vectorized_polynomial"
    return None


def _dominant_tail(stream) -> str | None:
    """The tail policy (ta|tu) of the kernel's dominant vector vtype, read from the decoded vsetvl
    state (VType.tail) — not guessed. None when no vector insn carries a tail token."""
    from collections import Counter
    c = Counter(i.vtype.tail for i in stream.insns
                if i.is_vector and i.vtype and i.vtype.tail)
    return c.most_common(1)[0][0] if c else None


def _infer_accumulator_dtype(stream, sew) -> str | None:
    """Accumulator element type read from the MAC form (ISA-grounded, not a fitted rule).

    A WIDENING integer MAC (``vwmacc``) accumulates i8/i16 products in i32 by definition; a widening
    FLOAT MAC (``vfwmacc``) accumulates in the 2xSEW float (f16 inputs -> f32). A NON-widening float
    contraction accumulates in the element width itself (SEW). No contraction to judge -> None."""
    if stream.count("vwmacc") > 0:
        return "i32"
    if stream.count("vfwmacc") > 0:
        return "f32"
    if stream.count("vfmacc", "vfmul") > 0 and sew in (16, 32, 64):
        return {16: "f16", 32: "f32", 64: "f64"}[sew]
    return None


def _lift_memory(stream) -> "MemoryFacet | None":
    """The data-movement/packing facet, lifted from the existing decode.memory MemFacet (loads/FMA,
    unit-stride, A-broadcast) — the memory dimension the CCA used to be blind to. None if no FMA loop.
    Lazy import of decode.memory (which references cca._fma_loop) to avoid an import cycle."""
    from .decode.memory import analyze_memory
    m = analyze_memory(stream)
    if m is None:
        return None
    access = ("unit_stride" if m.unit_stride_only
              else "indexed" if m.vec_indexed_loads > 0
              else "strided" if m.vec_strided_loads > 0
              else "none")
    # panel reuse: loads/FMA well below the unblocked ~2.0 => one loaded panel is reused across the MR
    # register-block accumulators (the amortization the expert wins on).
    reuse = (m.loads_per_fma is not None and m.loads_per_fma < 1.5)
    a_vf = (m.a_broadcast_per_fma == 0) if m.a_broadcast_per_fma is not None else None
    return MemoryFacet(access_pattern=access, panel_reuse=reuse, a_broadcast_vf=a_vf)


#: Runtime helpers whose presence in a compute region is a codegen ESCAPE -- the compiler fell back
#: to a generic library routine instead of emitting the operation. ``memrefCopy`` is MLIR's
#: rank-generic strided copy: correct for any rank/stride, and ~5,000 instructions to move a 4x16
#: tile. An expert kernel calls none of these.
RUNTIME_ESCAPE_SYMBOLS = ("memrefCopy", "memcpy", "memmove", "memset", "malloc", "free")

_CALL_MNEMONICS = ("jal", "jalr", "call")


def _lift_envelope(stream, *, undefined_symbols=None) -> "EnvelopeFacet":
    """Lift the code AROUND the loop: call sites scoped to loop bodies + runtime escapes.

    ``calls_in_loop`` is the structural signal and needs no symbol table: a call inside a loop body
    is per-iteration overhead whatever it calls. Counted over the OUTER loop spans (a tile epilogue
    sits inside the M/N loops but outside the K loop), excluding the innermost, so a call in the
    reduction body and a call in the tile epilogue both register.

    ``runtime_calls`` names them when the object's undefined symbols are available (``llvm-nm -u``),
    which is what turns "there is a call" into "it is memrefCopy" -- the difference between knowing
    a gap exists and knowing which pass closes it.

    HONESTY: ``calls_in_loop`` is loop-scoped, so it is only as trustworthy as the decoded loop
    structure. On an UNRELOCATED object (an unlinked ``model.o``: every branch displacement is a zero
    placeholder, so each branch resolves to its own address and ``loop_spans()`` reads EMPTY) the
    count silently collapses to 0 -- a per-tile ``memrefCopy`` would read as "no calls in any loop".
    So when ``stream.spans_reliable()`` is False we report ``calls_in_loop=None`` (UNKNOWN), never a
    misleading 0. ``runtime_calls`` is UNAFFECTED: it comes from the object's undefined symbols, not
    from loop structure, and is the axis that stays trustworthy on a whole-model fork.
    """
    if stream.spans_reliable():
        # kernel_loop_spans(), not loop_spans(): a compiler-emitted support helper (MLIR's
        # dealloc_helper) contributes loops that are not model compute, and counting their calls as
        # per-tile overhead would attribute the deallocator's work to the kernel's tile epilogue.
        spans = stream.kernel_loop_spans()
        inner = stream.innermost_loop()
        outer = [sp for sp in spans if sp != inner] or spans
        calls_in_loop = 0
        for sp in outer:
            calls_in_loop += stream.count_in(sp, *_CALL_MNEMONICS)
    else:
        calls_in_loop = None                # loop structure unreadable -> honestly UNKNOWN, not 0
    if undefined_symbols is not None:
        undef = {str(x) for x in undefined_symbols}
        escapes = tuple(sorted(undef.intersection(RUNTIME_ESCAPE_SYMBOLS)))
    elif stream.count(*_CALL_MNEMONICS) == 0:
        # No symbol table, but the region contains NO call instruction at all -- so it provably
        # escapes to no runtime helper. Sound to report () rather than "unknown", and it is what
        # lets an expert kernel (hand-written asm, no object symbols) populate this axis at all.
        # Without it both sides stay None, the divergence never fires, and the beam cannot see the
        # very gap this facet exists to expose.
        escapes = ()
    else:
        escapes = None                      # calls exist but we cannot name them: honestly unknown
    return EnvelopeFacet(calls_in_loop=calls_in_loop, runtime_calls=escapes)


def lift_asm(stream, *, op: str, source: str, backend: str = "rvv",
             undefined_symbols: "Iterable[str] | None" = None) -> CCA:
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
            accumulator_dtype=_infer_accumulator_dtype(stream, sew),
            reduction_form=("vredsum_tree" if reduce_n > 0 else "none"),
            epilogue=("requant_narrow" if narrow > 0 else "none"),
            register_block=reg_block,
            accumulator_resident=acc_resident,
            nr_is_vsetvlmax=nr_is_vsetvlmax,
            activation_vectorization=_infer_activation_vectorization(stream, op),
        ),
        vector=VectorFacet(sew=sew, lmul=lmul, vl_strategy=vl_strategy, tail=_dominant_tail(stream)),
        memory=_lift_memory(stream),
        envelope=_lift_envelope(stream, undefined_symbols=undefined_symbols),
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
                 pe_cols: int | None = None, backend: str) -> CCA:
    """Spatial/systolic (Gemmini) lifter — fills the SPATIAL facet from decoded accelerator ops
    (e.g. targetgen.rocc.decode counts of preload/compute/mvin/mvout). Keeps the same CCA schema
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
        # `engine_ops` had no home after the fold: it is a LIST of op names, i.e. a coverage question
        # ("which ops reached the engine"), not a per-region property. Its movement half is what
        # mattered and it lives on memory now, beside the access pattern it belongs with.
        memory=MemoryFacet(dma_pattern=dma_pattern),
        dispatch=DispatchFacet(n_dispatches=len(list(engine_ops)) or None),
        provenance={"level": "asm", "source": source, "confidence": "medium"},
    )


def lift_asm_roles(decoded, endpoint, *, op: str, source: str,
                   geometry: dict | None = None,
                   accumulator_dtype: str | None = None,
                   widening: bool | None = None,
                   loop_spans=None) -> CCA:
    """A CCA lifted from a ROLE-tagged instruction stream — the target-agnostic generalization of
    :func:`lift_asm`.

    ``lift_asm`` counts RVV mnemonic literals, so it recognizes exactly one target. This takes a stream
    that some target's own decoder has already role-tagged (``kernels.decode.rocc`` /
    ``kernels.decode.muon`` / the matrix decoder) plus the :class:`kernels.endpoints.Endpoint` that
    supplied the roles, and fills the facets those roles license. Roles are the only vocabulary shared
    across targets, and they are the same roles that derive the engine set, so a target cannot end up
    with a lifted facet its declared silicon does not have.

    Which engine-scoped facet gets filled follows ``endpoint.engine``, not a target name. The
    engine-agnostic facets (``dispatch``, ``memory``, ``compute``) are filled for every endpoint,
    because every endpoint is driven by something and moves data.

    ``geometry`` carries the target's derived hardware constants (``pe_rows``/``pe_cols``,
    ``threads_per_warp``, ``sew``, ...). They are IDENTITY, not levers: the compiler tiles TO them.
    Absent, they stay None rather than being inferred from the stream, because a stream that happens to
    use a 16-wide operand says nothing about how wide the array is.

    ``loop_spans`` names the address ranges of reduction loops when the caller could resolve them.
    Residency is scoped to the LOOP: a looping reduction emits one accumulate statically, so "is there
    a readout between the first and the last accumulate" is vacuous for exactly the kernels that matter.
    Without spans, residency is reported UNKNOWN (None) rather than guessed -- an unlinked object
    resolves every branch to its own address, finds no span, and would otherwise collapse every
    loop-scoped count to a confident zero.
    """
    from . import roles as _roles

    counts: dict[str, int] = {}
    for d in decoded:
        for r in getattr(d, "roles", ()) or ():
            counts[r] = counts.get(r, 0) + 1
    geo = dict(geometry or {})
    total = sum(counts.values())

    # --- dispatch: engine-agnostic, and the largest gap this closes ---
    n_config = counts.get("config", 0)
    dispatch = DispatchFacet(
        n_dispatches=total or None,
        config_fraction=(n_config / total) if total else None,
        # State set ONCE and inherited by the rest of the stream, versus re-set per tile. One config
        # for many accumulates is reuse; one per accumulate is not.
        descriptor_reuse=(n_config <= 1) if counts.get("accumulate") else None,
        loop_offloaded=bool(counts.get("loop_descriptor")) if total else None,
        dma_overlap=bool(counts.get("dma")) if total else None,
    )

    # --- compute: shared across every engine ---
    resident = _accumulator_resident(decoded, loop_spans)
    compute = ComputeFacet(
        op=op, accumulator_dtype=accumulator_dtype, widening=widening,
        contraction_form=geo.get("dataflow"),
        accumulator_resident=resident,
    )

    # --- the engine-scoped facet, chosen by the ENDPOINT's engine ---
    spatial = simt = vector = None
    engine = getattr(endpoint, "engine", "")
    if engine == "spatial":
        spatial = SpatialFacet(pe_rows=geo.get("pe_rows"), pe_cols=geo.get("pe_cols"),
                               dataflow=geo.get("dataflow"), accumulator_resident=resident)
    elif engine == "simt":
        simt = SimtFacet(warps=geo.get("warps"), threads_per_warp=geo.get("threads_per_warp"),
                         # A barrier inside the reduction loop is the SIMT analogue of a readout
                         # inside it: both say the engine cannot keep its state across the reduction.
                         barriers_in_loop=counts.get("sync") if total else None,
                         smem_resident=resident,
                         divergence=geo.get("divergence"))
    elif engine == "vector":
        vector = VectorFacet(sew=geo.get("sew"), lmul=geo.get("lmul"),
                             vl_strategy=geo.get("vl_strategy"), tail=geo.get("tail"))

    # A stream where NOTHING was role-tagged is not a clean lift, it is a lift that saw nothing. The
    # completeness check only applies once at least one role was recognized; otherwise the honest
    # verdict is "no roles recognized", which reads as an unresolved decode rather than a good kernel.
    # The completeness check applies to a CONTRACTION, and a stream is one iff it accumulates. Judging
    # a lane engine's elementwise region by contraction completeness would report every correct
    # activation kernel as incomplete for lacking a weight load it has no use for.
    missing = _roles.missing_contraction_roles(counts) if counts.get("accumulate") else ()
    nothing_seen = not counts
    return CCA(
        op=op, backend=[getattr(endpoint, "target", "") or "unknown"],
        compute=compute, spatial=spatial, simt=simt, vector=vector, dispatch=dispatch,
        memory=MemoryFacet(dma_pattern="burst" if counts.get("dma") else None),
        provenance={"level": "asm", "source": source,
                    # A stream missing a role a complete contraction needs is reported at LOW
                    # confidence rather than as a clean lift. The measured failure it guards: an audit
                    # counted accumulates and passed a kernel that never drained its accumulator.
                    "confidence": "low" if (missing or nothing_seen) else "high",
                    "no_roles_recognized": nothing_seen,
                    "endpoint": getattr(endpoint, "name", ""),
                    "engine": engine,
                    "role_counts": dict(sorted(counts.items())),
                    "missing_contraction_roles": list(missing)},
    )


def _accumulator_resident(decoded, loop_spans) -> bool | None:
    """Does the accumulator stay in the endpoint across the reduction, committing once after it?

    None when it cannot be decided. Scoped to the reduction LOOP when spans are available; without
    them the question is genuinely undecidable from a static stream, and answering it anyway is how a
    lifter reports a clean kernel it could not see.
    """
    acc = [d for d in decoded if "accumulate" in (getattr(d, "roles", ()) or ())]
    out = [d for d in decoded if "readout" in (getattr(d, "roles", ()) or ())]
    if not acc:
        return None                               # the engine was not driven at all
    if not loop_spans:
        if len(acc) > 1:
            # Fully unrolled: a readout BETWEEN the first and last accumulate means the accumulator
            # round-trips mid-reduction.
            lo, hi = acc[0].index, acc[-1].index
            return not any(lo < d.index < hi for d in out)
        return None                               # one static accumulate and no spans: undecidable
    inside = lambda d: any(lo <= d.addr <= hi for lo, hi in loop_spans)   # noqa: E731
    return not any(inside(d) for d in out)


def particularities() -> dict:
    """Load the per-target runtime/ABI particularities (bf16 ABI reg class, VLEN, vsetvl
    semantics, fp-contract default) so the comparator can normalize runtime artifacts out."""
    import yaml
    p = Path(__file__).resolve().parent / "runtime_particularities.yaml"
    return yaml.safe_load(p.read_text()) if p.is_file() else {}


def lift_graph(record, *, source: str = "graph", backend: str = "rvv") -> CCA:
    """Flat-graph analyzer: compose a PARTIAL CCA from a model2MLIR ``MatmulRecord`` (the flattened
    exported graph + ``prov.*``). Reads only what the graph determines — the op and the dtype-derived
    datapath facets (widening / accumulator dtype) — NOT the micro-kernel decisions (those are asm-level,
    from ``lift_asm``). Deterministic; the second analyzer whose CCA ``cca_agree`` cross-checks against
    the asm-derived one, so a bad reconstruction on either side is quarantined."""
    prov = dict(getattr(record, "prov", {}) or {})
    op = prov.get("prov.op") or getattr(record, "kind", None) or "unknown"
    dt = (getattr(record, "dtype", "") or "").lower()
    is_int8 = "i8" in dt or "int8" in dt
    # 16-bit floats accumulate in f32 (lower_bf16_matmul_f32acc) via a WIDENING MAC, exactly as
    # int8 accumulates in i32 -- without this arm an f16/bf16 record inferred accumulator_dtype
    # None, so the compute.accumulator_dtype axis could never route to the 16-bit datapath.
    is_half = "f16" in dt or "float16" in dt or "bf16" in dt
    acc = ("i32" if is_int8 else
           "f32" if (is_half or "f32" in dt or "float32" in dt) else None)
    return CCA(
        op=op, backend=[backend],
        compute=ComputeFacet(op=op, accumulator_dtype=acc,
                             widening=True if (is_int8 or is_half) else None),
        provenance={"level": "graph", "source": source, "confidence": "medium"})


def lift_coverage(model_mlir, claimed_ops, *, op: str = "model", source: str = "coverage",
                  backend: str = "rvv") -> CCA:
    """Whole-model CCA: how much of the model's work the schedule's ``claimed_ops`` actually covers.

    ``claimed_ops`` is the set of contraction op classes the schedule matches (a package's
    ``op_match``, minus any class a shape policy declined). Everything else — a contraction in an
    unclaimed class, and every non-contraction op — lowers through ``convert-linalg-to-loops`` to
    scalar code on one core.

    This is an IR lifter on purpose: it needs no asm and no build, so the loop can see the loss while
    choosing a schedule rather than after measuring a disappointing wall-clock. It is also the only
    place the CCA can see it at all — see :class:`CoverageFacet`.

    MACs are weighted by ``prod(parallel) * prod(reduction)`` per contraction, so a class holding one
    huge attention matmul is not outvoted by a dozen tiny ones.

    Contractions are classified through :func:`kernels.shapes.contraction_shapes`, NOT by op name: the
    capture emits attention as a ``linalg.generic`` and the pipeline names it with
    ``linalg-specialize-generic-ops`` before the schedule runs, so counting op names would report every
    such contraction as an unclaimed ``linalg.generic`` and invent a loss that is not there.
    """
    from ..common.mlir_query import op_name, parse
    from .shapes import contraction_shapes

    claimed = {str(c) for c in claimed_ops}
    module = parse(model_mlir)
    fn = next((o for o in module.walk() if o.name == "func.func"), None)
    if fn is None or not fn.body.blocks:
        return CCA(op=op, backend=[backend], coverage=CoverageFacet(),
                   provenance={"level": "coverage", "source": source, "confidence": "low"})
    total_macs = claimed_macs = 0
    present: set[str] = set()
    shapes = contraction_shapes(module)
    for s in shapes:
        present.add(s.op)
        macs = 1
        for d in tuple(s.parallel) + tuple(getattr(s, "reduction", ()) or ()):
            macs *= int(d)
        total_macs += macs
        if s.op in claimed:
            claimed_macs += macs
    n_ops = sum(1 for o in fn.body.blocks[0].ops if op_name(o).startswith("linalg."))
    n_contraction = len(shapes)
    return CCA(
        op=op, backend=[backend],
        coverage=CoverageFacet(
            claimed_mac_fraction=(claimed_macs / total_macs) if total_macs else None,
            unclaimed_op_classes=tuple(sorted(present - claimed)),
            non_contraction_op_fraction=((n_ops - n_contraction) / n_ops) if n_ops else None),
        provenance={"level": "coverage", "source": source, "confidence": "high"})


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
    for facet in ("compute", "vector", "memory", "envelope", "spatial", "simt", "dispatch", "layout"):
        fa, fb = getattr(a, facet), getattr(b, facet)
        for k, (va, vb) in _facet_fields(fa, fb).items():
            compared.append(f"{facet}.{k}")
            if va != vb:
                diffs.append(f"{facet}.{k}: {a.provenance.get('level')}={va!r} vs "
                             f"{b.provenance.get('level')}={vb!r}")
    return AgreementReport(agree=not diffs, disagreements=diffs, compared_fields=compared)
