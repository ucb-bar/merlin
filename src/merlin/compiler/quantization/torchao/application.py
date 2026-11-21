"""Apply the existing eager/PT2E recipe and retain its complete capture reports."""

from __future__ import annotations

import contextlib
import math
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from .calibration import (
    _as_tuple,
    _dynamic_summary,
    _RepeatedObservationSkip,
    _verify_dynamic_sites,
    _weight_rounding,
    calibration_limit,
)
from .eager import _plan_layers, build_fqn_config
from .effects import _pt2e_nested_graph_cleanup, preserve_fx_vmap_effects
from .graphs import _bias_onto_accumulator_grid, _has_floating_recipe_work, _rewrite_planes, spell_attention
from .portable import adopt, helper
from .pt2e import build_quantizer

_specs = helper("_recipe_specs")
DEFAULT_OBSERVER_EPSILON = _specs.DEFAULT_OBSERVER_EPSILON
ACTIVATION_CONTRACTION = _specs.ACTIVATION_CONTRACTION
RecipeError = _specs.RecipeError
_no_batch_norm_fold_needed = _specs._no_batch_norm_fold_needed
_require = _specs._require
_observer_name = _specs._observer_name
_recipe_operators = _specs._recipe_operators


@preserve_fx_vmap_effects()
def apply_recipe(
    model: Any,
    recipe: Mapping[str, Any],
    *,
    example_inputs: tuple,
    calibration_inputs: Iterable[Any] | None = None,
    calibration_samples: int = 100,
    original_frontend_snapshot: dict[str, Any] | None = None,
    source_layer_inventory: list[dict[str, Any]] | None = None,
) -> Any:
    """Quantize ``model`` under ``recipe`` and return the module the capture should lower."""
    import torch

    _weight, activation = _require(recipe)
    # Integral EXTERNAL inputs are not proof that a model is already quantized: a language model
    # accepts integer token IDs, then performs floating contractions after an embedding. Inspect
    # the operations covered by this recipe before skipping calibration. A graph with only integer
    # contractions (such as an authored integer seam) still needs no observer.
    tensors = [t for t in (example_inputs or ()) if isinstance(t, torch.Tensor)]
    integral_inputs = bool(tensors) and not any(torch.is_floating_point(t) for t in tensors)

    def skip_already_integer() -> Any:
        model._recipe_quantization_stats = {  # type: ignore[attr-defined]
            "applied": False,
            "why": (
                f"example inputs are integral ({sorted({str(t.dtype) for t in tensors})}) and no "
                "floating-point operation covered by the recipe needs quantization"
            ),
        }
        return model

    if activation.get("mode") == "dynamic":
        if recipe.get("software_admission") is not None:
            raise RecipeError("scoped SW admission is currently supported by static PT2E recipes only")
        # Dynamic TorchAO transforms floating Linear modules rather than observing the external
        # input. Integer token IDs must not suppress those internal projections either.
        if integral_inputs and not any(
            isinstance(module, (torch.nn.Linear, torch.nn.LazyLinear))
            and (weight := getattr(module, "weight", None)) is not None
            and torch.is_floating_point(weight)
            for module in model.modules()
        ):
            return skip_already_integer()
        from torchao.quantization import quantize_

        layer_plan = _plan_layers(recipe, model)
        config, notes = build_fqn_config(recipe, layer_plan)
        # filter_fn=None is REQUIRED, not tidiness: quantize_ defaults it to "is this a Linear",
        # and TorchAO refuses to hold both that and a per-layer mapping. Which layers are
        # quantized is the plan's verdict, derived from the target; being a Linear is not a reason.
        quantize_(model, config, filter_fn=None)
        placed = sum(1 for value in config.fqn_to_config.values() if value is not None)
        model._recipe_quantization_stats = {  # type: ignore[attr-defined]
            "api": "quantize_",
            "recipe_sha256": recipe.get("recipe_sha256"),
            "plan_sha256": layer_plan.get("plan_sha256"),
            "layers_seen": len(config.fqn_to_config),
            "layers_quantized": placed,
            # Every layer that is NOT in the target's numeric form, and whose limit put it there.
            "layers_on_host": notes,
            "refusals": layer_plan.get("refusals"),
            "weight_granularity": recipe["weight"]["granularity"],
            "activation_granularity": recipe["activation"]["granularity"],
        }
        return model

    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    if not example_inputs:
        raise RecipeError("a static recipe needs example inputs to export the model")
    # An ExportedProgram.module() GraphModule may return None from eval(); the
    # owned module, not that method's return value, remains the export input.
    model.eval()
    exported_program = torch.export.export(model, tuple(example_inputs))
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import (
            attach_original_identity,
            snapshot_exported_program,
            tuple_selection_trace_program,
        )

        actual = snapshot_exported_program(exported_program, stage="quantization_input")
        attach_original_identity(exported_program, original_frontend_snapshot, actual)
        # PT2E replays this exact ExportedProgram. Its proxy interpreter may
        # alias tuple selections onto the producer's result proxies; preserve
        # the selecting call-site lineage at that replay, not by guessing from
        # node order or tensor shape after quantization.
        exported_program = tuple_selection_trace_program(exported_program)
    exported = exported_program.module()
    # The selected frontend owns the semantic rewrite from string-padding
    # Conv2d to the numeric overload TorchAO observes. Apply it to the exact
    # exported graph before either the integral-input check or PT2E folding:
    # the recipe's operator table must never misread an unnormalized floating
    # Conv as absent. This does not rewrite the source model or its recipe.
    from m2m.capture.pt2e_padding import normalize_conv2d_padding

    padding_normalizations = normalize_conv2d_padding(exported)
    if original_frontend_snapshot is not None:
        source_nodes = {row["id"]: row for row in original_frontend_snapshot.get("nodes", ())}
        source_padding_schema = torch.ops.aten.conv2d.padding._schema.arguments
        padding_index = next(
            index for index, argument in enumerate(source_padding_schema) if argument.name == "padding"
        )
        padding_argument = source_padding_schema[padding_index]
        for row in padding_normalizations:
            origins = set(row["source_ids"]).intersection(source_nodes)
            # Precision staging may add intermediate identities. Exactly one
            # original string-padding Conv must own this rewritten node. The
            # selected source spelling must also agree on the literal padding
            # mode: an unrelated same-shaped Conv is not semantic ancestry.
            source = source_nodes[next(iter(origins))] if len(origins) == 1 else {}
            args, kwargs = source.get("args"), source.get("kwargs")
            if isinstance(args, list) and isinstance(kwargs, Mapping) and "padding" not in kwargs:
                source_padding = (
                    args[padding_index]
                    if len(args) > padding_index
                    else padding_argument.default_value
                    if padding_argument.has_default_value()
                    else None
                )
            elif isinstance(args, list) and isinstance(kwargs, Mapping) and len(args) <= padding_index:
                source_padding = kwargs.get("padding")
            else:
                source_padding = None
            if source.get("target") != "aten.conv2d.padding" or source_padding != row["padding"]:
                raise RecipeError("Conv2d string-padding normalization lost original source ancestry")
    # Only a recipe that quantizes activation x activation contractions needs a fused attention's
    # two matmuls in the graph; every other recipe sees the graph exactly as exported.
    attention = spell_attention(exported) if ACTIVATION_CONTRACTION in (recipe.get("families") or ()) else []
    if integral_inputs and not _has_floating_recipe_work(exported, recipe):
        return skip_already_integer()
    layer_plan = _plan_layers(recipe, model, source_layer_inventory=source_layer_inventory)
    policy = recipe.get("framework_capture_policy") or {}
    epsilon = policy.get("observer_epsilon", DEFAULT_OBSERVER_EPSILON)
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise RecipeError("observer epsilon must be finite and positive")
    fold_candidates = []
    fold_provenance_api = "not_applicable"
    if original_frontend_snapshot is not None:
        from m2m.capture import trace as capture_trace

        fold_api = getattr(capture_trace, "pt2e_conv_bn_fold_candidates", None)
        if callable(fold_api):
            fold_candidates = fold_api(exported, original_frontend_snapshot)
            fold_provenance_api = "selected_model2mlir"
        elif _no_batch_norm_fold_needed(original_frontend_snapshot, exported):
            # There is no BatchNorm source identity for a PT2E fold to carry.
            # This is an exact graph absence proof, not a family-level alias.
            fold_provenance_api = "not_needed_no_batch_norm"
        else:
            raise RecipeError(
                "selected model2MLIR lacks pt2e_conv_bn_fold_candidates; "
                "cannot prove PT2E Conv+BatchNorm source identity for this graph"
            )
    quantizer = build_quantizer(
        recipe,
        layer_plan=layer_plan,
        source_layer_inventory=source_layer_inventory,
        eps=epsilon,
        fold_candidates=fold_candidates,
    )
    if recipe.get("dynamic") and calibration_inputs is not None and iter(calibration_inputs) is calibration_inputs:
        calibration_inputs = list(calibration_inputs)  # the site check reads the same stream again
    calibration = calibration_inputs if calibration_inputs is not None else (tuple(example_inputs),)
    dynamic = quantizer.rewrite_dynamic(exported, calibration, calibration_limit(recipe, calibration_samples))
    if dynamic:
        _verify_dynamic_sites(exported, dynamic, example_inputs, calibration_inputs)
    prepared = prepare_pt2e(exported, quantizer)
    samples = calibration_inputs if calibration_inputs is not None else (tuple(example_inputs),)
    sample_limit = calibration_limit(recipe, calibration_samples)
    # GPTQ weight rounding, when the recipe asks for it: Hessians accumulate during this same
    # calibration pass, and the codes are set into the stored weights before convert folds them.
    weight_rounding = _weight_rounding(recipe, prepared)
    if weight_rounding is not None:
        samples = weight_rounding.track(samples)
    calibrated = 0
    epilogue = quantizer.epilogue
    tracking = epilogue.track(prepared) if epilogue is not None else contextlib.nullcontext()
    exact_integer_mm = helper("_capture_integer_mm").exact_integer_mm

    with torch.no_grad(), _RepeatedObservationSkip(prepared), tracking, exact_integer_mm(torch):
        for sample in samples:
            if calibrated >= sample_limit:
                break
            prepared(*_as_tuple(sample))
            calibrated += 1
    if calibrated == 0:
        raise RecipeError("static calibration saw no input sample")
    weight_rounding_report = weight_rounding.apply() if weight_rounding is not None else None
    if epilogue is not None and weight_rounding is None:
        epilogue.correct_biases(prepared)  # GPTQ chose the codes itself; its bias is left as calibrated
    quantizer.extensions.freeze(prepared)
    with _pt2e_nested_graph_cleanup() as nested_cleanup:
        quantized = convert_pt2e(prepared, use_reference_representation=False, fold_quantize=True)
    if quantizer.shared_sums is not None:
        quantizer.shared_sums.rewrite(quantized)
    island = epilogue.finish(quantized, recipe.get("float_island")) if epilogue is not None else None
    extension_reports = quantizer.extensions.finish(quantized)
    snap = recipe.get("bias_domain") == "accumulator" and epilogue is None
    bias_grid = _bias_onto_accumulator_grid(quantized) if snap else None
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import attach_quantization_boundaries

        operators = _recipe_operators(torch)
        selected = {op for family in recipe.get("families") or () for op in operators.get(family, ())}
        attach_quantization_boundaries(quantized, selected)
    planes = _rewrite_planes(quantized, recipe, quantizer.annotated_planes)
    pruned = None
    try:  # the capture library's own dead-state sweep, when present
        from m2m.capture.torchao_pipeline import _drop_unused_graph_state

        pruned = _drop_unused_graph_state(quantized)
    except Exception:  # noqa: BLE001 -- an optional tidy-up; the graph is correct without it
        pass
    try:
        from torch.ao.quantization import allow_exported_model_train_eval

        allow_exported_model_train_eval(quantized)
    except (ImportError, AttributeError):
        pass
    quantized._recipe_quantization_stats = {  # type: ignore[attr-defined]
        "api": "pt2e",
        "nested_graph_cleanup": nested_cleanup,
        "recipe_sha256": recipe.get("recipe_sha256"),
        **(
            {"activation_contraction_policy": dict(recipe["activation_contraction_policy"])}
            if "activation_contraction_policy" in recipe
            else {}
        ),
        "plan_sha256": layer_plan.get("plan_sha256"),
        "layers_on_host": [d for d in layer_plan.get("layers") or () if d.get("placement") == "host"],
        "operators_refused_by_placement": quantizer.refused_by_placement,
        # Every annotated contraction, whatever its operands: the capture's census compares this with
        # the contractions the integer rewrite saw. The activation x activation share is broken out.
        "annotated_contractions": quantizer.extensions.physical_contractions(
            quantizer.annotated + quantizer.annotated_pairs
        ),
        **(
            {"logical_annotated_contractions": quantizer.annotated + quantizer.annotated_pairs, **extension_reports}
            if extension_reports
            else {}
        ),
        **({"bias_on_accumulator_grid": bias_grid} if bias_grid is not None else {}),
        **(
            {
                "annotated_activation_contractions": quantizer.annotated_pairs,
                "activation_contraction_sites": quantizer.pair_sites,
                "attention_spelled": attention,
            }
            if ACTIVATION_CONTRACTION in (recipe.get("families") or ())
            else {}
        ),
        # Contractions the recipe's dynamic block rewrote into integer form (never PT2E-annotated), with
        # the Phase 2 cost they carry: device int8 MACs and calls, and the host's elementwise work
        # (absmax and mean passes, quantize, the outer-product rescale per output element and plane).
        **({"dynamic_contractions": _dynamic_summary(dynamic)} if recipe.get("dynamic") else {}),
        # Linears whose activation the static_planes block split into int8 digit planes (not counted
        # in annotated_contractions: integerization never sees them), with the same cost columns.
        **({"static_planes": _dynamic_summary(planes)} if recipe.get("static_planes") else {}),
        **({"integer_epilogue": epilogue.census(), "float_island": island} if epilogue is not None else {}),
        "annotated_sums": quantizer.annotated_sums,
        **({"shared_grid_operand_sums": quantizer.shared_sums.census()} if quantizer.shared_sums is not None else {}),
        "annotated_means": quantizer.annotated_means,
        # Sums and means the recipe's families cover and the graph does not requantize: left alone.
        "sums_and_means_left_in_float": quantizer.left_in_float,
        "calibration_samples": calibrated,
        **({"weight_rounding": weight_rounding_report} if weight_rounding_report is not None else {}),
        "pruned_dead_state_tensors": pruned,
        "weight_granularity": recipe["weight"]["granularity"],
        "activation_granularity": recipe["activation"]["granularity"],
        "software_admission": quantizer.software_decisions,
        "fold_provenance_api": fold_provenance_api,
        "conv2d_padding_normalizations": padding_normalizations,
        "framework_capture_policy": {
            "activation_observer": _observer_name(recipe["activation"], is_weight=False),
            "weight_observer": _observer_name(recipe["weight"], is_weight=True),
            "observer_epsilon": epsilon,
            "calibration_source": "explicit_calibration_stream" if calibration_inputs is not None else "example_inputs",
            "calibration_count": calibrated,
            "calibration_sample_limit": sample_limit,
        },
    }
    return quantized


# The application facade preserves the existing recipe API while these real
# responsibility owners remain separately readable and independently callable.

_owner = adopt(sys.modules[__name__])
sys.modules[__name__] = _owner
