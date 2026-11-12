"""Register existing optional passes and validate explicitly supplied effects."""

from __future__ import annotations

from merlin.compiler.passes.contract import PipelineError


def select_features(
    features: frozenset[str] | None, *, masked_contraction_effects=None, source_observation_effects=None
) -> frozenset[str]:
    from merlin.compiler.optimizations.registry import normalize

    # `normalize` REJECTS an unregistered name, and the lowering runs in forked/child processes that
    # import this module without importing `llvmlower.lower` (where the runner-gated features are
    # registered). Registering here, before the set is validated, is what keeps a feature the caller
    # legitimately asked for from failing as "unknown impr feature" in one process and working in
    # another. Idempotent.
    from merlin.llvmlower.transpose_maps import ensure_registered as _register_fold_weight_transpose

    _register_fold_weight_transpose()
    from merlin.llvmlower.weight_prepack import ensure_registered as _register_prepack_weight_layout

    _register_prepack_weight_layout()
    from merlin.llvmlower.weight_prequant import ensure_registered as _register_prequantize_constant_weights

    _register_prequantize_constant_weights()
    from merlin.llvmlower.broadcast_fold import ensure_registered as _register_fold_broadcast_into_generic

    _register_fold_broadcast_into_generic()
    from merlin.llvmlower.named_broadcast_fold import ensure_registered as _register_named_broadcast_fold

    _register_named_broadcast_fold()
    from merlin.llvmlower.concat_dps import ensure_registered as _register_concat_dps

    _register_concat_dps()
    # Features the selected vector support provider implements (its direct-conv block request among
    # them) need no registration here: `normalize` resolves each from that provider's declarations,
    # and the lowering child processes inherit the same selection.
    # The panel-packed im2col REQUEST, registered for the same reason: the rewrite happens in the
    # preparation step, and by lowering time the name only records that the arm was asked for.
    from merlin.llvmlower.im2col_pack import ensure_registered as _register_im2col_panel_pack

    _register_im2col_panel_pack()
    # This feature edits the pass list itself and lives outside impr_features. Direct callers of
    # this module therefore need the same registration that lower.py performs before normalization.
    from merlin.llvmlower.epilogue_fusion import ensure_registered as _register_epilogue_fusion

    _register_epilogue_fusion()
    from merlin.llvmlower.roundeven_intrinsic import ensure_registered as _register_roundeven_intrinsic

    _register_roundeven_intrinsic()
    from merlin.llvmlower.fma_intrinsic import ensure_registered as _register_fma_intrinsic

    _register_fma_intrinsic()
    from merlin.compiler.quantization.scope import ensure_registered as _register_quant_scope

    _register_quant_scope()
    from merlin.llvmlower.int_softmax_table import ensure_registered as _register_int_softmax_table

    _register_int_softmax_table()
    # `--pass` / `--no-pass` / MERLIN_PASSES add or remove feature-bound optional passes.
    from merlin.llvmlower.optional_passes import selected_features

    feats = normalize(selected_features(features))
    from merlin.llvmlower.masked_contraction import FEATURE as _MASKED_FEATURE
    from merlin.llvmlower.masked_contraction import MaskEffectContract

    if _MASKED_FEATURE in feats:
        if not isinstance(masked_contraction_effects, MaskEffectContract):
            raise PipelineError("closed-mask scheduling requires explicit MaskEffectContract")
        masked_contraction_effects.validate()
    elif masked_contraction_effects is not None:
        raise PipelineError("masked arithmetic effects supplied without masked-contraction policy")
    from merlin.llvmlower.source_expression_interval import IntervalEffectContract
    from merlin.llvmlower.source_observation_stage import FEATURE as _SOURCE_OBSERVATION_FEATURE

    if _SOURCE_OBSERVATION_FEATURE in feats:
        if not isinstance(source_observation_effects, IntervalEffectContract):
            raise PipelineError("source observation requires explicit IntervalEffectContract")
        source_observation_effects.validate()
    elif source_observation_effects is not None:
        raise PipelineError("source observation effects supplied without selected discovery feature")
    return feats
