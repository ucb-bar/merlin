"""Named optimization registry with explicit, ordered default-off declarations.

Implementation modules define reusable strategy functions without registering on import.
Only this owner constructs the shared registry and executes the historical eager catalog.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from merlin.common.paths import module_source_path
from merlin.compiler.lower.native.policy_features import (
    _PROMOTE_STACK_DEFAULT_BYTES,
    FUSE_AFTER_GENERALIZE_NAME,
    FUSE_ELEMENTWISE_ANCHOR,
    FUSE_ELEMENTWISE_NAME,
    FUSE_ELEMENTWISE_STAGE,
    PROMOTE_STACK_BYTES_ENV,
    PROMOTE_STACK_NAME,
    _fuse_after_generalize,
    _fuse_elementwise_pipeline,
    _promote_buffers_to_stack,
    _promote_stack_bytes,
    _reuse_tensor_destination,
)
from merlin.compiler.lower.options.bufferized_result_identity import (
    ensure_registered as _ensure_buffer_identity,
)
from merlin.compiler.lower.options.copy_expand import FEATURE as _EXPAND_COPY_FEATURE
from merlin.compiler.lower.options.reduce_vec import ensure_registered as _ensure_amax_reduction
from merlin.compiler.lower.options.scalar_squared_sum import ensure_registered as _ensure_squared_sum
from merlin.compiler.lower.options.selfcopy import FEATURE as _SELF_COPY_FEATURE
from merlin.compiler.lower.options.tensor_helper_inlining import (
    ensure_registered as _ensure_tensor_helper_inlining,
)
from merlin.compiler.lower.options.transpose_fuse import FEATURE as _FUSE_TRANSPOSE_FEATURE
from merlin.compiler.optimizations.accumulator_residency import (
    WHOLEMODEL_VF_BMMPAD_NAME,
    WHOLEMODEL_VF_CAPS,
    WHOLEMODEL_VF_MR_MM,
    WHOLEMODEL_VF_NAME,
    WHOLEMODEL_VF_NR_BMM,
    _register_accumulator_resident,
    _register_accumulator_resident_v2,
)
from merlin.compiler.optimizations.contraction_loops import (
    _VFMACC_TILED_SCHEDULE,
    MRPAD_NAME,
    _packed_eliminate_empties,
    _register_accumulator_resident_v3,
    _register_packed_grid,
    _register_tiled_grid,
    _vfmacc_schedule_edit,
    ensure_mrpad_for_elem_types,
    scalable_lanes,
    vfmacc_packed_schedule,
    vfmacc_tiled_schedule,
)
from merlin.compiler.optimizations.register_blocking import (
    PEROP_BLOCK_NAME,
    PEROP_MR_FILL_NAME,
    PEROP_MR_SENTINEL_PREFIX,
    PEROP_NR_FILL_NAME,
    PEROP_NR_SENTINEL_PREFIX,
    _perop_mr_fill_unresolved,
    _perop_nr_fill_unresolved,
    _perop_sentinel_unresolved,
    ensure_perop_block,
    ensure_v3_kblocked_microkernel,
    ensure_v3_microkernel,
    ensure_v3_perop_microkernel,
    ensure_v3_scalable_microkernel,
    ensure_v3_unrolled_microkernel,
    parse_perop_mr_sentinel,
    parse_perop_nr_sentinel,
    perop_mr_sentinel,
    perop_nr_sentinel,
    unclaimed_op_classes,
)
from merlin.compiler.optimizations.vectorization import (
    _ACT_POLY_SCHEDULE,
    VEC_BYTEWISE_ATTR,
    VEC_BYTEWISE_MAX_INPUTS,
    VEC_BYTEWISE_MIN_BITS,
    VEC_NONCONTRACTION_LANES,
    VEC_NONCONTRACTION_MAX_RANK,
    VEC_NONCONTRACTION_MIN_RANK,
    VEC_NONCONTRACTION_NAME,
    _splice_vec_rank_arms,
    _vec_noncontraction_hygiene,
    _vec_noncontraction_point,
    ensure_promote_stack,
    ensure_vec_noncontraction,
    vec_noncontraction_lanes,
    vec_noncontraction_max_rank,
    vectorize_reduction_pipeline,
    vectorize_reduction_schedule,
)
from merlin.compiler.selection import features as _shared_features
from merlin.compiler.selection.features import CompositionError as CompositionError
from merlin.compiler.selection.features import FeatureRegistry, ImprFeature

# ---- discovery ---------------------------------------------------------------------------------
# Resolve existing feature names without importing every legacy implementation.

_LEVER_MODULES: "dict[str, str] | None" = None


def _single_name_lever_modules() -> "dict[str, str]":
    """Lever name -> module basename, for every sibling module declaring lever names as constants.

    Discovered by PARSING the sibling sources rather than importing them: at this point in the
    package every lever module imports THIS one to call :func:`register`, so importing them all
    eagerly is circular. Parsing is also structural, not a line-match -- the module-level
    ``FEATURE = "..."`` assignment is read out of the AST, and a module only qualifies if it also
    defines ``ensure_registered`` at module level, which is the thing that will be called.

    A module that declares no ``FEATURE`` (a family module carrying a ``FEATURE_PREFIX``, or a plain
    helper) simply does not appear, so families keep resolving through their own prefix/arity rules.

    ``<X>_FEATURE`` counts too, not only the bare ``FEATURE``. A module may own more than one point --
    ``requant_fuse`` registers the traversal removal and the variant that also reshapes the epilogue
    loop, so the board can price the reshape separately -- and ``perop_blocks`` has always owned
    ``CONV_ARM_FEATURE`` with no way to resolve it here. Reading every module-level string constant
    whose name ends in ``FEATURE`` keeps the mapping DERIVED, which is the property this function
    exists for; a name whose ``ensure_registered`` does not actually register it still falls through
    to the family rules below, because the caller re-checks the registry.
    """
    global _LEVER_MODULES
    if _LEVER_MODULES is None:
        import ast as _ast

        found: dict[str, str] = {}
        for src in sorted(module_source_path("merlin.compiler.lower.model").parent.glob("*.py")):
            try:
                tree = _ast.parse(src.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue  # unreadable/unparseable sibling is not a lever; never guess one
            features: list[str] = []
            has_ensure = False
            for node in tree.body:  # MODULE level only -- a nested FEATURE is not the lever's name
                if (
                    isinstance(node, _ast.Assign)
                    and isinstance(node.value, _ast.Constant)
                    and isinstance(node.value.value, str)
                    and any(
                        isinstance(t, _ast.Name) and (t.id == "FEATURE" or t.id.endswith("_FEATURE"))
                        for t in node.targets
                    )
                ):
                    features.append(node.value.value)
                elif isinstance(node, _ast.FunctionDef) and node.name == "ensure_registered":
                    has_ensure = True
            if has_ensure:
                for feature in features:
                    found.setdefault(feature, src.stem)
        _LEVER_MODULES = found
    return _LEVER_MODULES


def _try_lazy_register(name: str) -> bool:
    """Auto-register a v3 micro-kernel tuning point from its NAME (accum_resident_v3_<MR>_<NR>_<KC>).

    The register block is a continuous, beam-tunable knob space, but the lowering runs in a SUBPROCESS
    that re-imports this module — so a point registered on demand in the parent is invisible there and
    the feature fails to resolve (observed: every dynamically-registered MR failed at K2 while the
    pre-registered grid worked). Deriving the point from the name makes resolution reproducible in ANY
    process, which is what actually makes the space continuous.

    The same holds for the two other on-demand v3 families, which were previously NOT derivable from
    their name (their prefix is ``accum_resident_v3vl_`` / ``accum_resident_v3p_``, so the
    ``accum_resident_v3_`` test rejected them): a package that names one of those points directly in
    ``compiler_features`` -- rather than reaching it through a ``microkernel`` knob block, which
    registers it as a side effect of resolving -- failed with an "unknown impr feature" KeyError.
    Each family is registered from its own arity, so a name is either derivable or an honest error."""

    try:
        shared = _shared_features.get_shared_feature(name)
    except KeyError:
        pass
    else:
        register(shared)
        return True
    # A SINGLE-NAME lever lives in its own module and resolves by importing that module, rather than
    # through the arity table below. It has to resolve HERE and not only where it is ranked:
    # `wholemodel_proposer` registers it for the PROPOSAL path, but a package that names it in
    # `compiler_features` reaches `normalize` through `linux_cross_build.build_binary`, which imports no proposer
    # -- and through the lowering SUBPROCESS, which re-imports this module and sees no registration
    # the parent made at run time. Both raise "unknown impr feature" without this.
    #
    # This was a HAND-MAINTAINED list of per-module hooks, and it drifted exactly as such lists do:
    # it named two modules while THIRTEEN declared a lever. The other eleven resolved only by
    # ACCIDENT -- an unrelated module happening to import theirs (`pipeline` pulls in `prov_cse` and
    # `im2col_pack`), or a call-time `ensure_registered()` buried in `linux_cross_build.build_binary` (that is
    # what `prepack_weight_layout` relies on). Accidental resolution is not resolution: a fresh
    # process that imports only THIS module could not resolve seven of the thirteen, and the lowering
    # subprocess is exactly such a process. Removing an unrelated import, or reaching a lever by a
    # path that skips `linux_cross_build.build_binary`, silently turns a named lever into "unknown impr feature" --
    # or, where a caller swallows that, into a lever reported as applied and never applied.
    # The mapping is DERIVED from the modules themselves so that resolution does not depend on who
    # else happened to be imported first.
    _module = _single_name_lever_modules().get(name)
    if _module is not None:
        import importlib

        importlib.import_module(f"merlin.compiler.lower.model.{_module}").ensure_registered()
        return name in _REGISTRY
    # The parallel-GRAIN family carries its threshold in the name and lives in its own module, so
    # it is resolved first and by import rather than through the arity table below.
    from merlin.compiler.lower.options.parallel_grain import FEATURE_PREFIX as _PG_PREFIX

    if name.startswith(_PG_PREFIX):
        from merlin.compiler.lower.options.parallel_grain import ensure_registered as _pg_ensure

        try:
            _pg_ensure(int(name[len(_PG_PREFIX) :]))
        except ValueError:
            return False
        return name in _REGISTRY
    # Per-region OpenMP team policy is another name-derived continuous family.  Resolve it here so
    # a package and the lowering subprocess do not depend on the proposer having imported it first.
    from merlin.compiler.lower.options.parallel_team import FEATURE_PREFIX as _PT_PREFIX

    if name.startswith(_PT_PREFIX):
        from merlin.compiler.lower.options.parallel_team import ensure_registered as _pt_ensure

        try:
            _pt_ensure(int(name[len(_PT_PREFIX) :]))
        except ValueError:
            return False
        return name in _REGISTRY
    from merlin.compiler.lower.options.residual_parallel import FEATURE_PREFIX as _RP_PREFIX

    if name.startswith(_RP_PREFIX):
        from merlin.compiler.lower.options.residual_parallel import ensure_registered as _rp_ensure

        try:
            _rp_ensure(int(name[len(_RP_PREFIX) :]))
        except ValueError:
            return False
        return name in _REGISTRY
    point = _vec_noncontraction_point(name)
    if point is not None:
        # The non-contraction family's points are DERIVABLE FROM THE NAME (lanes and max rank), so
        # they resolve in any process -- which is the whole reason this hook exists: a point the
        # parent registered at run time is invisible to the lowering subprocess that re-imports this
        # module, and the build fails with "unknown impr feature" rather than with the lever off.
        ensure_vec_noncontraction(*point)
        return name in _REGISTRY
    parts = name.split("_")
    tails: dict[str, tuple[int, "Callable[..., str]"]] = {
        "accum_resident_v3_": (3, ensure_v3_microkernel),  # MR, NR, KC
        "accum_resident_v3vl_": (3, ensure_v3_scalable_microkernel),  # MR, NR, KC
        "accum_resident_v3p_": (5, ensure_v3_perop_microkernel),  # MR_mm, NR_mm, MR_bmm, NR_bmm, KC
    }
    for prefix, (arity, make) in tails.items():
        if not name.startswith(prefix):
            continue
        args = parts[len(prefix.rstrip("_").split("_")) :]
        if len(args) != arity:
            return False
        try:
            make(*(int(a) for a in args))
        except ValueError:
            return False
        return name in _REGISTRY
    # A feature IMPLEMENTED by the selected vector support provider resolves from that provider's own
    # declaration, in every process that selects it (the lowering subprocess inherits the selection).
    # With none selected the name cannot be honoured, and the refusal says why rather than only that
    # the name is unknown.
    from merlin.compiler.backend import providers as provided

    try:
        feature = provided.feature_definition(name)
    except provided.ProviderUnavailable as exc:
        raise KeyError(
            f"unknown impr feature {name!r}: Merlin declares no such feature, and no vector support provider "
            "is selected to declare it (name one on MERLIN_TARGET_PATH, or install its backend package)"
        ) from exc
    except KeyError:
        return False
    register(feature)
    return True


# ---- registry ----------------------------------------------------------------------------------

_FEATURES = FeatureRegistry(lambda name: _try_lazy_register(name))

_REGISTRY = _FEATURES.entries

register = _FEATURES.register


NAMED_INT8_CONTRACTION_NAME = _shared_features.NAMED_INT8_CONTRACTION_NAME

QUANTIZE_BEFORE_GATHER_NAME = _shared_features.QUANTIZE_BEFORE_GATHER_NAME

for _feature in _shared_features.eager_features():
    register(_feature)


get = _FEATURES.get


def known() -> list[str]:
    return sorted(set(_FEATURES.known()) | set(_shared_features.known()))


normalize = _FEATURES.normalize

apply_pipeline = _FEATURES.apply_pipeline

apply_schedule = _FEATURES.apply_schedule

apply_cflags = _FEATURES.apply_cflags


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


TILED_GRID_NAMES: list[str] = _register_tiled_grid()


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


PACKED_GRID_NAMES: list[str] = _register_packed_grid()


ACCUM_RESIDENT_NAMES: list[str] = _register_accumulator_resident()


ACCUM_RESIDENT_V2_NAMES: list[str] = _register_accumulator_resident_v2()


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


PEROP_NR_LADDER: tuple[str, ...] = tuple(perop_nr_sentinel(_n) for _n in (2, 4, 8, 32))

PEROP_MR_LADDER: tuple[str, ...] = tuple(perop_mr_sentinel(_n) for _n in (1, 2, 8, 16))


ACCUM_RESIDENT_V3_NAMES: list[str] = _register_accumulator_resident_v3()


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


VECTORIZE_AMAX_REDUCTION_NAME = _ensure_amax_reduction()


BUFFERIZED_RESULT_IDENTITY_NAME = _ensure_buffer_identity()


SCALAR_SQUARED_SUM_NAME = _ensure_squared_sum()


TENSOR_HELPER_INLINING_NAME = _ensure_tensor_helper_inlining()
