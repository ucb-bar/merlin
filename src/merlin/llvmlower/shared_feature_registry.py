"""Target-neutral feature declarations, independent of any vector provider.

Each shared rewrite registers on demand in its owning module. The registry is used by
scalar compilation and copied into a provider's private registry when selected.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from .copy_expand import FEATURE as _EXPAND_COPY_FEATURE
from .feature_registry import CompositionError, FeatureRegistry, ImprFeature
from .selfcopy import FEATURE as _SELF_COPY_FEATURE
from .shared_features import (
    FUSE_AFTER_GENERALIZE_NAME,
    FUSE_ELEMENTWISE_NAME,
    PROMOTE_STACK_NAME,
    _fuse_after_generalize,
    _fuse_elementwise_pipeline,
    _promote_buffers_to_stack,
    _reuse_tensor_destination,
)
from .transpose_fuse import FEATURE as _FUSE_TRANSPOSE_FEATURE

NAMED_INT8_CONTRACTION_NAME = "named_int8_contraction"
QUANTIZE_BEFORE_GATHER_NAME = "quantize_before_gather"

_SHARED_MODULES = frozenset(
    {
        "broadcast_fold",
        "broadcast_math_hoist",
        "bufferized_result_identity",
        "concat_dps",
        "epilogue_fusion",
        "fma_intrinsic",
        "im2col_identity_view",
        "int_softmax_table",
        "llvm_loop_outline",
        "masked_contraction",
        "named_broadcast_fold",
        "parallel_coarsen",
        "parallel_grain",
        "parallel_team",
        "prov_cse",
        "quant_round",
        "quant_hoist",
        "quant_scope",
        "residual_parallel",
        "roundeven_intrinsic",
        "scalar_contraction",
        "scalar_pointwise_packet",
        "scalar_pointwise_unroll",
        "scalar_squared_sum",
        "source_observation_stage",
        "transpose_maps",
        "weight_prequant",
    }
)
_NAME_MODULES: dict[str, str] | None = None


def _name_modules() -> dict[str, str]:
    global _NAME_MODULES
    if _NAME_MODULES is None:
        found: dict[str, str] = {}
        for stem in sorted(_SHARED_MODULES):
            tree = ast.parse((Path(__file__).with_name(stem + ".py")).read_text(encoding="utf-8"))
            names: list[str] = []
            has_ensure = False
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    if any(
                        isinstance(t, ast.Name) and (t.id == "FEATURE" or t.id.endswith("_FEATURE"))
                        for t in node.targets
                    ):
                        names.append(node.value.value)
                elif isinstance(node, ast.FunctionDef) and node.name == "ensure_registered":
                    has_ensure = True
            if has_ensure:
                for name in names:
                    found.setdefault(name, stem)
        _NAME_MODULES = found
    return _NAME_MODULES


def _lazy_register(name: str) -> bool:
    stem = _name_modules().get(name)
    if stem is not None:
        module = importlib.import_module(f".{stem}", __package__)
        if stem == "quant_round":
            module.ensure_shared_registered()
        else:
            module.ensure_registered()
        return name in _REGISTRY
    for stem in ("parallel_grain", "parallel_team", "residual_parallel"):
        module = importlib.import_module(f".{stem}", __package__)
        prefix = module.FEATURE_PREFIX
        if name.startswith(prefix):
            try:
                module.ensure_registered(int(name[len(prefix) :]))
            except ValueError:
                return False
            return name in _REGISTRY
    return False


_FEATURES = FeatureRegistry(_lazy_register)
_REGISTRY = _FEATURES.entries


def register(feature: ImprFeature) -> ImprFeature:
    return _FEATURES.register(feature)


def get_shared_feature(name: str) -> ImprFeature:
    """Resolve one shared feature without importing the legacy vector registry."""
    return _FEATURES.get(name)


def get(name: str) -> ImprFeature:
    return get_shared_feature(name)


def known() -> list[str]:
    return _FEATURES.known()


def normalize(features) -> frozenset[str]:
    return _FEATURES.normalize(features)


def apply_pipeline(passes: list[str], features: frozenset[str]) -> list[str]:
    return _FEATURES.apply_pipeline(passes, features)


def apply_schedule(schedule_text: str, features: frozenset[str]) -> str:
    return _FEATURES.apply_schedule(schedule_text, features)


def apply_cflags(cflags: list[str], features: frozenset[str]) -> list[str]:
    return _FEATURES.apply_cflags(cflags, features)


def eager_features() -> tuple[ImprFeature, ...]:
    """Return the generic baseline declarations for a separate compiler registry."""
    return tuple(_REGISTRY[name] for name in _EAGER_NAMES)


# Generic, eagerly available feature declarations.
register(
    ImprFeature(
        name=NAMED_INT8_CONTRACTION_NAME,
        action_class="PASS",
        description="Emit the canonical 2-D i8xi8->i32 contraction as a MIXED-TYPE linalg.matmul instead "
        "of a linalg.generic, so the named-op transform schedules can see it. The int8 quant "
        "pass otherwise leaves ZERO linalg.matmul in the module (measured: 15 -> 0 on "
        "small_llama_int8), and transform.structured.match on a name nothing carries returns "
        "an empty handle -- which makes every one of the 39 linalg.matmul/batch_matmul "
        "matchers in this file a vacuous no-op on int8 while still reporting as applied. An "
        "87-fork beam over those levers emitted only 21 distinct binaries. Batched, conv and "
        "non-canonical indexing keep the generic form: a named op ASSERTS an indexing "
        "convention, so claiming one the op does not have would be a correctness bug rather "
        "than a missed optimization. Default-off; baseline byte-identical.",
    )
)

register(
    ImprFeature(
        name=QUANTIZE_BEFORE_GATHER_NAME,
        action_class="PASS",
        description=(
            "When a contraction's f32 activation operand is produced by a PURE data-movement op (an "
            "all-parallel linalg.generic whose body only yields its input, i.e. an element copy), "
            "quantize the op's SOURCE with a per-tensor scale instead of quantizing its expanded result "
            "with a per-parallel-row one. Quantization is elementwise, so quantize(G(A)) == G(quantize(A)) "
            "exactly for a single shared scale; what blocks the commutation today is only the per-row "
            "scale, under which one element of A carries a different scale in every im2col column it "
            "appears in. This is the case that matters on every convolutional model here: model2MLIR "
            "expands every conv into im2col + matmul before merlin sees it (190 such ops in deepjscc "
            "int8, 175 in lstmnetvit int8, zero fused conv2d), so the operand being quantized IS the "
            "expanded matrix -- deepjscc enc.net.1 quantizes a 147x4096 f32 matrix, ~41x the "
            "1x3x70x70 activation it was gathered from. With the scale moved, the abs-max and the "
            "quantize both run on the activation and the gather itself moves i8, 4x less traffic for the "
            "same trip count, and the f32 expansion is erased entirely. The abs-max is EXACT in both "
            "modes: over the source when the indexing maps and bounds PROVE the gather reads every "
            "element, otherwise reduced through the gather's own map (same reads, scalar result, no "
            "materialization) -- so a strided or dilated gather that skips elements is handled, not "
            "approximated with the coarser amax(A) >= amax(G(A)). Refuses and counts the reason for a "
            "computed body, a shared intermediate, a dynamic extent or a non-gather producer. "
            "NOT bit-identical: the per-tensor activation scale is a genuine numeric change against the "
            "shipped per-row scheme and must pass the accuracy gate on its own. Default-off; with the "
            "feature absent the int8 datapath is byte-identical."
        ),
    )
)

register(
    ImprFeature(
        name=FUSE_AFTER_GENERALIZE_NAME,
        action_class="PASS",
        description="run elementwise fusion a second time, AFTER linalg-generalize-named-ops. The "
        "upstream order fuses (pos 2) before generalizing (pos 3), and fusion only sees "
        "linalg.generic -- so every NAMED op is invisible to it. MEASURED on small_llama "
        "int8: 39 linalg.broadcast ops at 23.9% of the model, each materialising a "
        "per-channel quantization scale into a full weight-sized tensor (a 344-element "
        "vector splatted to 44,032 elements, then read straight back). "
        "MEASURED AND REFUTED as a whole-model lever on that same model: 3,543,517 -> "
        "4,313,041 ns, 1.22x SLOWER (sustained, n=3, cos identical). It does what it says -- "
        "the scalar bucket falls 3.1 -> 2.8 ms -- but the contraction RISES 1.8 -> 2.8 ms, "
        "because fusing the dequant chain into the producers perturbs the shape the "
        "contraction was vectorized into and costs more than the broadcast it removes. Kept "
        "registered so the finding is reproducible and the lever is not re-attempted blindly; "
        "the broadcast waste is real but needs a TARGETED fold of the broadcast into its "
        "consumer's indexing map, not blanket elementwise fusion. Default-off; baseline "
        "byte-identical.",
        edit_pipeline=_fuse_after_generalize,
    )
)

register(
    ImprFeature(
        name=PROMOTE_STACK_NAME,
        action_class="PASS",
        description="promote small bufferization allocs to the stack after hoisting. Targets LOCALITY, "
        "not allocator cost: bufferization gives each intermediate its own memref.alloc (209 "
        "of them on small_llama int8, ~3% of wall), and scattered heap buffers mean every "
        "intermediate is written and re-read through cache misses. ExecuTorch's memory "
        "planner places EVERY activation of the same model into ONE 32,512-byte arena, small "
        "enough to stay in L1; we have no whole-model planner wired, and this upstream pass "
        "buys much of the same effect. Per-buffer cap (MERLIN_PROMOTE_STACK_BYTES, default "
        "16384) is a stack-overflow guard, not a tuning knob. Default-off; baseline "
        "byte-identical.",
        edit_pipeline=_promote_buffers_to_stack,
    )
)

register(
    ImprFeature(
        name="approximate_transcendental_activation",
        action_class="PASS",
        description="Explicit f32 activation polynomial rewrite, independent of host vectorization. "
        "Shares the provenance-scoped exp/erf/tanh approximation with the vectorized activation "
        "feature. Normalization and unsupported dtypes retain libm lowering. Changes numerical "
        "arithmetic and requires the caller's original model accuracy gate; default off.",
    )
)

register(
    ImprFeature(
        name="fuse_activation_polynomial_fma",
        action_class="PASS",
        description="Explicit fused evaluation of the selected f32 activation polynomial. "
        "Changes approximation arithmetic; requires the original model accuracy gate. "
        "Selects exact FMA intrinsics and leaves normalization expf unchanged; default off.",
        implies=frozenset({"approximate_transcendental_activation", "lower_fma_to_intrinsic"}),
    )
)

register(
    ImprFeature(
        name=_SELF_COPY_FEATURE,
        action_class="PASS",
        description="erase `memref.copy %x, %x` (a buffer copied onto itself) after bufferization/cse "
        "and before finalize-memref-to-llvm, where it would otherwise survive as an opaque "
        "@memrefCopy rank-generic runtime call costing ~79 retired instructions per OUTPUT "
        "ELEMENT. Removes the tile-epilogue copy; emits no new code.",
    )
)

register(
    ImprFeature(
        name=_EXPAND_COPY_FEATURE,
        action_class="PASS",
        description="rewrite every ranked, statically shaped `memref.copy` to a `linalg.copy` after "
        "bufferization and before finalize-memref-to-llvm, so the pipeline's own "
        "convert-linalg-to-loops emits an scf load/store nest instead of leaving a call to "
        "the rank-generic `@memrefCopy` runtime helper (or a copy-derived `memcpy`). "
        "Structure-keyed (ranked + static shape), no shape or model assumption; a copy it "
        "cannot prove static is left alone and counted. Default-off, baseline "
        "byte-identical.",
    )
)

register(
    ImprFeature(
        name=_FUSE_TRANSPOSE_FEATURE,
        action_class="PASS",
        description="fuse `matmul(A, transpose(B, [1,0]))` into a transpose-b `linalg.matmul` (repoint "
        "the B operand to the un-transposed weight + permute its indexing_map (k,n)->(n,k), "
        "then erase the dead transpose). Eliminates the standalone SCALAR weight transpose "
        "(393 ms / 57% of openvla) and its DRAM buffer with no op materialized; the matmul "
        "stays vectorized by the frozen schedule and reads B contiguously along k. "
        "Whole-model cross-op fusion; default-off, baseline byte-identical.",
    )
)

register(
    ImprFeature(
        name=FUSE_ELEMENTWISE_NAME,
        action_class="PASS",
        description=(
            "Run `linalg-fuse-elementwise-ops` (+ canonicalize/cse) after the transform schedule has "
            "matched and vectorized the contractions, so the elementwise producer->consumer chains that "
            "`linalg-specialize-generic-ops` un-fuses collapse again and each line is touched once. The "
            "stage already existed in build_rvv_pipeline but was reachable only through the "
            "MERLIN_FUSE_POST environment variable, which no fork can vary -- so the tuning loop could "
            "not select it. Attacks the linalg.generic long tail (16.5% of int8 whole-model runtime "
            "across 191 ops in 26 classes): the per-row activation-quantize and requant generics each "
            "read a scale that `specialize` materialized into a full-size temporary first. MEASURED on "
            "small_llama_int8_consistent at the current best config -- linalg.broadcast 50 -> 13, "
            "242,944 -> 60,160 bytes written per inference for zero arithmetic (13.9x amplification "
            "removed); emitted `forward` 35,253 -> 31,348 instructions (-11.1%, vector 14,053 -> 12,561 "
            "and scalar 13,802 from 15,319, so the vector FRACTION is flat at 0.478 -> 0.477 by "
            "construction), model.o 189,008 -> 170,664 bytes, stack alloca 118 -> 103 sites. Output "
            "BIT-IDENTICAL on spike (prefix digest cc60e8a90270ec1e either way; tiers ['fp32','w8a8'], "
            "tier_ok 'fp32_cos_only', ok True, every gate figure equal to the last digit). NOT the "
            "refuted `fuse_elementwise_after_generalize`, which runs the stage on the far side of the "
            "generalize anchor and measured 1.22x SLOWER by perturbing the already-vectorized "
            "contraction: here vwmacc stays 152 and vredmax 104 while vle32.v 2,116 -> 1,729 and "
            "vse32.v 809 -> 581. (fmul.s 0 -> 82 is a small scalar tail this introduces.) THE WALL IS "
            "UNMEASURED -- fewer instructions is a reason to measure, not a result: on this same model a "
            "transpose fold that removed ops and shrank the object measured 1.09x SLOWER. Structure-keyed "
            "(names one upstream pass and one anchor pass; no model, shape, dtype or target). "
            "Default-off; baseline byte-identical."
        ),
        edit_pipeline=_fuse_elementwise_pipeline,
    )
)

register(
    ImprFeature(
        name="reuse_tensor_destination",
        action_class="PASS",
        description="Forward tensor destinations through proved insert-slice chains before bufferization; default off.",
        edit_pipeline=_reuse_tensor_destination,
        implies=frozenset({_SELF_COPY_FEATURE}),
    )
)

register(
    ImprFeature(
        name="initialize_tensor_border_only",
        action_class="PASS",
        description="Fill only the proved exterior of a fresh tensor destination whose interior is fully overwritten; default off.",
        implies=frozenset({"reuse_tensor_destination"}),
    )
)

register(
    ImprFeature(
        name="fold_uniform_fill_copy",
        action_class="PASS",
        description="Fill proved copy destinations directly from a private uniform buffer; default off.",
        implies=frozenset({_SELF_COPY_FEATURE}),
    )
)

register(
    ImprFeature(
        name="specialize_contiguous_copy",
        action_class="PASS",
        description="Specialize disjoint static copies to contiguous suffix memcpy and outer strided loops; default off.",
        implies=frozenset({_SELF_COPY_FEATURE}),
    )
)

_EAGER_NAMES = tuple(_REGISTRY)
