"""Named optimization registry with explicit, ordered default-off declarations.

Implementation modules define reusable strategy functions without registering on import.
Only this owner constructs the shared registry and executes the historical eager catalog.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from merlin.compiler.passes.features import (
    _PROMOTE_STACK_DEFAULT_BYTES as _PROMOTE_STACK_DEFAULT_BYTES,
)
from merlin.compiler.passes.features import (
    FUSE_AFTER_GENERALIZE_NAME as FUSE_AFTER_GENERALIZE_NAME,
)
from merlin.compiler.passes.features import (
    FUSE_ELEMENTWISE_ANCHOR as FUSE_ELEMENTWISE_ANCHOR,
)
from merlin.compiler.passes.features import (
    FUSE_ELEMENTWISE_NAME as FUSE_ELEMENTWISE_NAME,
)
from merlin.compiler.passes.features import (
    FUSE_ELEMENTWISE_STAGE as FUSE_ELEMENTWISE_STAGE,
)
from merlin.compiler.passes.features import (
    PROMOTE_STACK_BYTES_ENV as PROMOTE_STACK_BYTES_ENV,
)
from merlin.compiler.passes.features import (
    PROMOTE_STACK_NAME as PROMOTE_STACK_NAME,
)
from merlin.compiler.passes.features import (
    _fuse_after_generalize as _fuse_after_generalize,
)
from merlin.compiler.passes.features import (
    _fuse_elementwise_pipeline as _fuse_elementwise_pipeline,
)
from merlin.compiler.passes.features import (
    _promote_buffers_to_stack as _promote_buffers_to_stack,
)
from merlin.compiler.passes.features import (
    _promote_stack_bytes as _promote_stack_bytes,
)
from merlin.compiler.passes.features import (
    _reuse_tensor_destination as _reuse_tensor_destination,
)
from merlin.compiler.selection.features import CompositionError as CompositionError
from merlin.compiler.selection.features import FeatureRegistry, ImprFeature
from merlin.llvmlower.copy_expand import FEATURE as _EXPAND_COPY_FEATURE
from merlin.llvmlower.selfcopy import FEATURE as _SELF_COPY_FEATURE
from merlin.llvmlower.transpose_fuse import FEATURE as _FUSE_TRANSPOSE_FEATURE

_FEATURES = FeatureRegistry(lambda name: _try_lazy_register(name))

_REGISTRY = _FEATURES.entries

register = _FEATURES.register

from merlin.compiler.selection import shared as _shared_features

NAMED_INT8_CONTRACTION_NAME = _shared_features.NAMED_INT8_CONTRACTION_NAME

QUANTIZE_BEFORE_GATHER_NAME = _shared_features.QUANTIZE_BEFORE_GATHER_NAME

for _feature in _shared_features.eager_features():
    register(_feature)

from merlin.compiler.optimizations.discovery import (
    _try_lazy_register as _try_lazy_register,
)

get = _FEATURES.get


def known() -> list[str]:
    return sorted(set(_FEATURES.known()) | set(_shared_features.known()))


normalize = _FEATURES.normalize

apply_pipeline = _FEATURES.apply_pipeline

apply_schedule = _FEATURES.apply_schedule

apply_cflags = _FEATURES.apply_cflags

from merlin.compiler.optimizations.contraction import (
    _VFMACC_TILED_SCHEDULE as _VFMACC_TILED_SCHEDULE,
)
from merlin.compiler.optimizations.contraction import (
    _vfmacc_schedule_edit as _vfmacc_schedule_edit,
)

register(
    ImprFeature(
        name="lmul_widen_n",
        action_class="KNOB",
        description="mined lmul_grouping_policy: widen the matmul N tile/vector 8->16 so the emitted "
        "vector group uses a higher LMUL (m2->m4). Composes with other features in autotune.",
        edit_schedule=lambda t: t.replace("tile_sizes [4, 8, 1]", "tile_sizes [4, 16, 1]").replace(
            "vector_sizes [4, 8, 1]", "vector_sizes [4, 16, 1]"
        ),
    )
)

register(
    ImprFeature(
        name="fused_vfmacc_tiled",
        action_class="PASS",
        description="CORRECT, SCALABLE, bounded-code tiled vfmacc: tile matmul [MR=4,NR=16,KC=4] and "
        "batch_matmul [1,4,16,4] so M,N,K are all scf.for LOOPS, SCOPED-vectorize the tile "
        "only (no whole-func vectorize_children => whole-model-safe, no vector.extract "
        "explosion), rebuild the contract via reduction_to_contract, then "
        "outerproduct->vector.fma->vfmacc. Inner body = constant MR*KC=16 fma at any "
        "M/N/K => .text bounded (no JAL +-1MB wall) and applicable to whole models. "
        "Benchmarked vs the full-unroll fused_vfmacc_contraction.",
        edit_schedule=lambda _t: _VFMACC_TILED_SCHEDULE,
        schedule_replace=True,
    )
)

register(
    ImprFeature(
        name="fused_vfmacc_scalable",
        action_class="PASS",
        description="Alias of the fixed fused_vfmacc_tiled recipe: bounded-code (constant MR*KC inner "
        "body), K-as-loop, scoped-vectorize (whole-model-safe) tiled vfmacc. Correct + "
        "bit-exact at 32/64/128; the whole-model-safe vfmacc for e2e.",
        edit_schedule=lambda _t: _VFMACC_TILED_SCHEDULE,
        schedule_replace=True,
    )
)

from merlin.compiler.optimizations.contraction import (
    _register_tiled_grid as _register_tiled_grid,
)
from merlin.compiler.optimizations.contraction import (
    vfmacc_tiled_schedule as vfmacc_tiled_schedule,
)

TILED_GRID_NAMES: list[str] = _register_tiled_grid()

from merlin.compiler.optimizations.contraction import (
    _packed_eliminate_empties as _packed_eliminate_empties,
)
from merlin.compiler.optimizations.contraction import (
    vfmacc_packed_schedule as vfmacc_packed_schedule,
)

register(
    ImprFeature(
        name="vfmacc_packed",
        action_class="PASS",
        edit_pipeline=_packed_eliminate_empties,
        description="operand-PACKING tiled vfmacc (mined packed_rhs_policy = OpenBLAS ncopy/tcopy, "
        "XNNPACK goi-prepack): transform.structured.pack A/B/C into contiguous [MR,NR,KC] "
        "register-tile panels, lower pack/unpack to copy loops, fold unit dims, then the "
        "scoped-vectorize -> outerproduct -> vfmacc recipe on the packed op so the inner "
        "transfers are UNIT-STRIDE contiguous (no strided vector.transfer). Register-tile "
        "[4,16,16]. Default-off; baseline byte-identical.",
        edit_schedule=lambda _t: vfmacc_packed_schedule(4, 16, 16),
        schedule_replace=True,
    )
)

from merlin.compiler.optimizations.contraction import (
    _register_packed_grid as _register_packed_grid,
)

PACKED_GRID_NAMES: list[str] = _register_packed_grid()

from merlin.compiler.optimizations.resident import (
    WHOLEMODEL_VF_BMMPAD_NAME as WHOLEMODEL_VF_BMMPAD_NAME,
)
from merlin.compiler.optimizations.resident import (
    WHOLEMODEL_VF_CAPS as WHOLEMODEL_VF_CAPS,
)
from merlin.compiler.optimizations.resident import (
    WHOLEMODEL_VF_MR_MM as WHOLEMODEL_VF_MR_MM,
)
from merlin.compiler.optimizations.resident import (
    WHOLEMODEL_VF_NAME as WHOLEMODEL_VF_NAME,
)
from merlin.compiler.optimizations.resident import (
    WHOLEMODEL_VF_NR_BMM as WHOLEMODEL_VF_NR_BMM,
)
from merlin.compiler.optimizations.resident import (
    _register_accumulator_resident as _register_accumulator_resident,
)

ACCUM_RESIDENT_NAMES: list[str] = _register_accumulator_resident()

from merlin.compiler.optimizations.hoisting import (
    _register_accumulator_resident_v2 as _register_accumulator_resident_v2,
)

ACCUM_RESIDENT_V2_NAMES: list[str] = _register_accumulator_resident_v2()


from merlin.compiler.optimizations.blocks import (
    ensure_v3_kblocked_microkernel as ensure_v3_kblocked_microkernel,
)
from merlin.compiler.optimizations.blocks import (
    ensure_v3_microkernel as ensure_v3_microkernel,
)
from merlin.compiler.optimizations.blocks import (
    ensure_v3_perop_microkernel as ensure_v3_perop_microkernel,
)
from merlin.compiler.optimizations.blocks import (
    ensure_v3_scalable_microkernel as ensure_v3_scalable_microkernel,
)
from merlin.compiler.optimizations.blocks import (
    ensure_v3_unrolled_microkernel as ensure_v3_unrolled_microkernel,
)
from merlin.compiler.optimizations.blocks import (
    unclaimed_op_classes as unclaimed_op_classes,
)
from merlin.compiler.optimizations.loops import (
    scalable_lanes as scalable_lanes,
)
from merlin.compiler.optimizations.pointwise import (
    VEC_BYTEWISE_ATTR as VEC_BYTEWISE_ATTR,
)
from merlin.compiler.optimizations.pointwise import (
    VEC_BYTEWISE_MAX_INPUTS as VEC_BYTEWISE_MAX_INPUTS,
)
from merlin.compiler.optimizations.pointwise import (
    VEC_BYTEWISE_MIN_BITS as VEC_BYTEWISE_MIN_BITS,
)
from merlin.compiler.optimizations.pointwise import (
    VEC_NONCONTRACTION_LANES as VEC_NONCONTRACTION_LANES,
)
from merlin.compiler.optimizations.pointwise import (
    VEC_NONCONTRACTION_MAX_RANK as VEC_NONCONTRACTION_MAX_RANK,
)
from merlin.compiler.optimizations.pointwise import (
    VEC_NONCONTRACTION_MIN_RANK as VEC_NONCONTRACTION_MIN_RANK,
)
from merlin.compiler.optimizations.pointwise import (
    VEC_NONCONTRACTION_NAME as VEC_NONCONTRACTION_NAME,
)
from merlin.compiler.optimizations.pointwise import (
    _splice_vec_rank_arms as _splice_vec_rank_arms,
)
from merlin.compiler.optimizations.pointwise import (
    _vec_noncontraction_hygiene as _vec_noncontraction_hygiene,
)

register(
    ImprFeature(
        name=VEC_NONCONTRACTION_NAME,
        action_class="PASS",
        description=(
            "Bounded per-rank vectorize of the NON-CONTRACTION all-parallel linalg.generics "
            "(elementwise / layout / im2col gather / pad) that the contraction-only schedule leaves for "
            "convert-linalg-to-loops to emit as scalar loops. Measured share of the loss: 86-89% of the "
            "linalg ops of every captured workload, against whole-model MAC/cycle of 0.40 (spectformer), "
            "0.22 (deepjscc), 0.067 (lstmnetvit) versus ~8 for a VLEN=128 int8 vwmacc datapath. Needs the "
            "prepare pass's merlin.vec_r{rank} tags, which build_app enables when this feature is on. "
            "MEASURED on deepjscc int8 (spike): emits 394 -> 1945 vector instructions (4.9x, so the "
            "lever is NOT inert), output BIT-IDENTICAL, and 484,690,000 -> 621,555,001 cycles, i.e. "
            "1.28x SLOWER, flat at 0.78x across 8/16/32 lanes. That flatness was the tell that the "
            "width was never the variable: the tile-and-vectorize-on-tensors realization emits a "
            "`memref.copy %x, %x` of each tile INSIDE the innermost loop -- a memcpy of the bytes the "
            "vector store just wrote, per 8 elements. It went unattributed because those copies lower "
            "to `llvm.memcpy`, not to the `@memrefCopy` the earlier check counted. This feature now "
            "IMPLIES `erase_self_copy`, which removes them: measured on the host lowering, the copy "
            "call sites the lever adds inside `forward` go from +15 (deepjscc) / +33 (small_llama) to "
            "+0 on both, with the vector-instruction gain kept. The output was reported bit-identical "
            "at that point and it was NOT: on small_llama int8 the lever answered cos 0.968247 / rel "
            "0.46352 against a baseline 0.999966 / 0.00836, and answered DIFFERENTLY from the same "
            "object under a different process memory layout, because a `vector<8xi1>` mask tile is "
            "stored PACKED into a buffer every scalar reader addresses one byte per element (56 of 64 "
            "bytes left uninitialised, read straight into the attention mask). The arms now refuse a "
            "sub-byte element type on the destination or on any input, and with that refusal the "
            "output IS bit-identical to the baseline on all three int8 recaptures, in both arm "
            "placements, under three memory layouts. See `_vec_bytewise_matchers`. "
            "NOT yet claimed as a speedup -- that is a board measurement, and the cycle number above "
            "is the one it has to beat. Two known limits remain, both MEASURED and neither fixable "
            "from this file: (a) COVERAGE -- `func.func(linalg-specialize-generic-ops)`, which runs "
            "before the transform interpreter so the contraction arms can match named ops, rewrites "
            "the tagged generics into `linalg.broadcast` and DROPS their `merlin.vec_r{rank}` "
            "attribute, so only 15 of 93 tagged ops (deepjscc) ever reach an arm while the prepare "
            "pass reports 93; (b) the arms still allocate one extra destination buffer per vectorized "
            "op (+12 allocs / +1.05 MB of 50.19 MB cumulative on deepjscc), which is malloc calls "
            "rather than per-element traffic."
        ),
        edit_schedule=_splice_vec_rank_arms,
        implies=_vec_noncontraction_hygiene(),
    )
)

from merlin.compiler.optimizations.passes import (
    ensure_promote_stack as ensure_promote_stack,
)
from merlin.compiler.optimizations.perop import (
    PEROP_BLOCK_NAME as PEROP_BLOCK_NAME,
)
from merlin.compiler.optimizations.perop import (
    PEROP_NR_FILL_NAME as PEROP_NR_FILL_NAME,
)
from merlin.compiler.optimizations.perop import (
    _perop_nr_fill_unresolved as _perop_nr_fill_unresolved,
)
from merlin.compiler.optimizations.perop import (
    _perop_sentinel_unresolved as _perop_sentinel_unresolved,
)
from merlin.compiler.optimizations.pointwise import (
    ensure_vec_noncontraction as ensure_vec_noncontraction,
)
from merlin.compiler.optimizations.pointwise import (
    vec_noncontraction_lanes as vec_noncontraction_lanes,
)
from merlin.compiler.optimizations.pointwise import (
    vec_noncontraction_max_rank as vec_noncontraction_max_rank,
)

register(
    ImprFeature(
        name=PEROP_NR_FILL_NAME,
        action_class="KNOB",
        description="widen each contraction's per-op N cap until its NARROWEST element fills a whole "
        "vector register at the board's VLEN, instead of every op sharing one element count. "
        "MEASURED model-dependent on the K1: 1.160x faster on spectformer int8, 1.196x slower "
        "on small_llama int8 -- because the i32 accumulator is what sets LMUL, so a wider N "
        "tile can push it from m4 to m8 and spill (decoded: 0 -> 6 accumulator spill ops). A "
        "search knob, not a default. Default-off; baseline byte-identical.",
        edit_pipeline=_perop_nr_fill_unresolved,
        implies=frozenset({PEROP_BLOCK_NAME}),
    )
)

from merlin.compiler.optimizations.perop import (
    PEROP_MR_FILL_NAME as PEROP_MR_FILL_NAME,
)
from merlin.compiler.optimizations.perop import (
    _perop_mr_fill_unresolved as _perop_mr_fill_unresolved,
)

register(
    ImprFeature(
        name=PEROP_MR_FILL_NAME,
        action_class="KNOB",
        description="derive each contraction's per-op M cap from how many accumulator rows of ITS OWN "
        "block fit the board's vector register file, instead of every op in the model "
        "sharing one hand-set number. The N axis of this question has been derived per-op "
        "for a while (VLEN- and element-width-scaled); M was a single scalar, so two ops "
        "could differ only by gcd(M) clipping a shared cap. Derived from target facts only "
        "-- the VLEN built for, the op's own element triple, and the RVV register count. A "
        "bound on the architecture, not a promise about the allocator, so it is searched "
        "rather than defaulted. Default-off; baseline byte-identical.",
        edit_pipeline=_perop_mr_fill_unresolved,
        implies=frozenset({PEROP_BLOCK_NAME}),
    )
)

register(
    ImprFeature(
        name=PEROP_BLOCK_NAME,
        action_class="PASS",
        description="request PER-CONTRACTION register blocking: derive the widest block legal for each "
        "contraction's OWN extents (and its own narrowest element width) from the prepared "
        "IR, tag each contraction, and emit one tile+vectorize arm per distinct block. "
        "Resolved by prepare_for_lowering into a concrete, table-specific feature; a "
        "sentinel, so it must never reach lowering itself. Replaces the class-wide clamps "
        "(one degenerate extent in a class otherwise forces the whole class off the vector "
        "path). Default-off; baseline byte-identical.",
        edit_pipeline=_perop_sentinel_unresolved,
        schedule_replace=True,
    )
)

from merlin.compiler.optimizations.perop import (
    PEROP_MR_SENTINEL_PREFIX as PEROP_MR_SENTINEL_PREFIX,
)
from merlin.compiler.optimizations.perop import (
    PEROP_NR_SENTINEL_PREFIX as PEROP_NR_SENTINEL_PREFIX,
)
from merlin.compiler.optimizations.perop import (
    ensure_perop_block as ensure_perop_block,
)
from merlin.compiler.optimizations.perop import (
    parse_perop_mr_sentinel as parse_perop_mr_sentinel,
)
from merlin.compiler.optimizations.perop import (
    parse_perop_nr_sentinel as parse_perop_nr_sentinel,
)
from merlin.compiler.optimizations.perop import (
    perop_mr_sentinel as perop_mr_sentinel,
)
from merlin.compiler.optimizations.perop import (
    perop_nr_sentinel as perop_nr_sentinel,
)

PEROP_NR_LADDER: tuple[str, ...] = tuple(perop_nr_sentinel(_n) for _n in (2, 4, 8, 32))

PEROP_MR_LADDER: tuple[str, ...] = tuple(perop_mr_sentinel(_n) for _n in (1, 2, 8, 16))

from merlin.compiler.optimizations.variants import (
    _register_accumulator_resident_v3 as _register_accumulator_resident_v3,
)

ACCUM_RESIDENT_V3_NAMES: list[str] = _register_accumulator_resident_v3()

from merlin.compiler.optimizations.variants import (
    MRPAD_NAME as MRPAD_NAME,
)
from merlin.compiler.optimizations.variants import (
    ensure_mrpad_for_elem_types as ensure_mrpad_for_elem_types,
)

MRPAD_INT8_NAME = ensure_mrpad_for_elem_types("i8", "i8", "i32")

MRPAD_INT8_TILES: tuple[str, ...] = tuple(
    ensure_mrpad_for_elem_types("i8", "i8", "i32", MR=_mr, NR=_nr, NR_bmm=_nb)
    for _mr, _nr, _nb in (
        (1, 16, 8),
        (1, 32, 8),
        (1, 64, 16),  # the EXPERT's MR
        (2, 16, 8),
        (4, 16, 8),
        (8, 16, 8),
        (4, 32, 8),
        (4, 64, 16),
        (8, 32, 16),
        (8, 64, 16),
    )
)

register(
    ImprFeature(
        name="intrinsic_microkernel",
        action_class="CODEGEN",
        # HONEST LABEL: this is a CEILING REFERENCE, not a compiler-emitted feature. It is a marker with
        # NO MLIR schedule/pipeline edit (baseline byte-identical); the measured number comes from a
        # HAND-WRITTEN riscv_vector.h driver (ceiling_drivers/ours_intrinsic_gemm_driver.c), NOT from our
        # transform pipeline. It records the TARGET a dedicated RVV micro-kernel codegen pass should hit,
        # and quantifies how far the compiler-emitted accumulator_resident_microkernel still is from it.
        # The transform-dialect feature does NOT yet reach this (see accumulator_resident_microkernel:
        # the emitted asm still spills the carried accumulator through the stack inside the K loop —
        # vl4re8.v/vs4r.v of the accumulator per K-tile, so the CCA reads accumulator_resident=False, and
        # measured ~19x off this ceiling @64^3). Keeping the hand kernel ONLY as a labeled ceiling so the
        # gap is honest and the codegen work-item (action_catalog: compute.accumulator_resident ->
        # CODEGEN, forkable_now=False) is visible — never linked as if the compiler emitted it.
        description="CEILING REFERENCE (hand-written riscv_vector.h driver, NOT compiler-emitted): a "
        "register-blocked, accumulator-resident, K-streaming RVV GEMM micro-kernel (MR=4, "
        "NR=vsetvlmax) that keeps the MR x NR accumulator in vector registers across the "
        "whole K loop (vfmacc.vf chain, B row + A scalars streamed, C stored once). "
        "Spill-free, bit-exact 32/64/128, 1.7x faster than OpenBLAS on the spike proxy "
        "(pack-excluded). It is the TARGET for a dedicated RVV micro-kernel codegen pass; "
        "the transform-dialect accumulator_resident_microkernel does NOT yet reach it "
        "(~19x off @64^3 — still spills the accumulator per K-tile). Marker only (no "
        "schedule/pipeline edit); baseline byte-identical.",
        edit_pipeline=None,
        edit_schedule=None,
    )
)

from merlin.compiler.optimizations.passes import (
    _ACT_POLY_SCHEDULE as _ACT_POLY_SCHEDULE,
)

register(
    ImprFeature(
        name="vectorized_transcendental_activation",
        action_class="PASS",
        description="GENERAL vectorized-activation lowering, PRECISELY TARGETED by provenance: the "
        "act_poly rewriter (spliced into the lowering runner before the pass manager) "
        "replaces math.erf/exp/tanh with an inline minimax ARITH polynomial ONLY inside a "
        "linalg.generic the provenance marks as an elementwise ACTIVATION (gelu/silu/sigmoid/"
        "tanh) — NOT a softmax/normalization (whose exp stays on the exact libm path; "
        "blanket-rewriting it drove openvla whole-model cos to 0.541). It TAGS each targeted "
        "generic (merlin.act_vectorize) and the schedule vectorizes exactly those (no blanket "
        "foreach over every generic, no failures(suppress) -> no masked miscompile, no "
        "6+min/config compile blowup). So GELU (erf) and sigmoid/SiLU (exp) vectorize to a "
        "vector fmul/fadd (vfmacc) chain while softmax stays correct. Closes the activation "
        "gap vs XNNPACK's polynomial RVV kernels (coefficient/structure CEILING REFERENCE; we "
        "emit the MLIR). APPROXIMATION: cos>0.999 / max-abs-err <~6e-7 vs libm on REALISTIC "
        "ranges (gated on cos/rel error, not bit-exact). Default-off; baseline byte-identical.",
        edit_schedule=lambda _t: _ACT_POLY_SCHEDULE,
        schedule_replace=True,
        implies=frozenset({"approximate_transcendental_activation"}),
    )
)

register(
    ImprFeature(
        name="fused_vfmacc_contraction",
        action_class="PASS",
        description="mined fma_broadcast_policy: form a real vector.contract -> outerproduct(kind=add) "
        "-> vector.fma -> llvm.fmuladd -> vfmacc (vectorize_children + lower_contraction "
        "outerproduct + lower_outerproduct). Closes the separate-vfmul.vv+vfadd.vv gap. For "
        "kernel-sized contraction workloads (vectorize_children explodes on whole models).",
        edit_schedule=_vfmacc_schedule_edit,
        schedule_replace=True,
    )
)

from merlin.compiler.optimizations.passes import (
    vectorize_reduction_pipeline as vectorize_reduction_pipeline,
)
from merlin.compiler.optimizations.passes import (
    vectorize_reduction_schedule as vectorize_reduction_schedule,
)

register(
    ImprFeature(
        name="vectorize_reduction",
        action_class="PASS",
        description="Vectorize a standalone reduction (softmax/norm row-reduce, `linalg.reduce`) so it "
        "lowers to a HARDWARE vector reduction (`vfredusum.vs` for fp / `vredsum.vs` for int) "
        "instead of the scalar convert-linalg-to-loops accumulate the baseline emits. Matches "
        "the reduction `linalg.generic` by its reduction iterator (ranks 1/2/3), vectorizes it "
        "to `vector.multi_reduction`, lowers that via the inner-reduction strategy to "
        "`vector.reduction`, and reassociates the fp reduce so the backend picks the unordered "
        "vfredusum. The contraction schedule is left intact (matmul object byte-identical), so "
        "it is whole-model-safe. The route for the compute.reduction_form CCA lever (previously "
        "a bijection orphan). APPROXIMATION (fp reassociation, cos-gated). Default-off; baseline "
        "byte-identical.",
        edit_pipeline=vectorize_reduction_pipeline,
        edit_schedule=vectorize_reduction_schedule,
        schedule_replace=True,
    )
)

from merlin.llvmlower.reduce_vec import ensure_registered as _ensure_amax_reduction

VECTORIZE_AMAX_REDUCTION_NAME = _ensure_amax_reduction()

from merlin.llvmlower.bufferized_result_identity import ensure_registered as _ensure_buffer_identity

BUFFERIZED_RESULT_IDENTITY_NAME = _ensure_buffer_identity()

from merlin.llvmlower.scalar_squared_sum import ensure_registered as _ensure_squared_sum

SCALAR_SQUARED_SUM_NAME = _ensure_squared_sum()

from merlin.llvmlower.tensor_helper_inlining import ensure_registered as _ensure_tensor_helper_inlining

TENSOR_HELPER_INLINING_NAME = _ensure_tensor_helper_inlining()
