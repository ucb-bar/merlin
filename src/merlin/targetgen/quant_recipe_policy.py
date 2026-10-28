"""Explicit numerical recipe selection using existing capability-derived arithmetic.

This composes ordinary recipe builders; it grants no capture, placement or
performance authority. Source recipes and their historical defaults are immutable.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from merlin.targetgen import quant_methodology as M
from merlin.targetgen import quant_recipe as Q

SCHEMA = "numerical_recipe_policy_v1"
_EXTENSIONS = ("accumulator_residual", "integer_mean", "projection_input_reuse", "operand_sum_grid")


def validate(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an explicit weight-scale/native-exit policy, without defaults."""
    fields = {
        "schema",
        "weight_granularity",
        "native_outputs",
        "dynamic_activation_planes",
        "float_quality",
        "integer_extensions",
    }
    if (not isinstance(policy, Mapping) or not fields <= set(policy)
            or set(policy) - fields - {"attention_context_output"} or policy.get("schema") != SCHEMA):
        raise ValueError("numerical recipe policy has an unknown schema or field")
    if (policy["weight_granularity"] not in ("tensor", "preserve")
            or policy["float_quality"] not in ("report_only", "preserve")):
        raise ValueError("this policy needs explicit weight-scale and float-quality handling")
    planes = policy["dynamic_activation_planes"]
    if planes is not None and (type(planes) is not int or planes != 1):
        raise ValueError("dynamic_activation_planes is null or the integer 1")
    if "attention_context_output" in policy and (
        policy["attention_context_output"] != "int8_value_grid" or planes != 1
    ):
        raise ValueError("attention_context_output selects int8_value_grid with single-plane attention")
    outputs = policy["native_outputs"]
    if not isinstance(outputs, Mapping) or set(outputs) != {"operators", "reader_policy"}:
        raise ValueError("native_outputs states operators and reader_policy")
    operators = outputs["operators"]
    if (
        not isinstance(operators, list)
        or not operators
        or any(type(op) is not str or op not in ("linear", "conv2d") for op in operators)
        or len(set(operators)) != len(operators)
        or outputs["reader_policy"] not in ("compute", "non_add_compute")
    ):
        raise ValueError("native_outputs has unsupported operators or reader policy")
    extensions = policy["integer_extensions"]
    if (
        not isinstance(extensions, Mapping)
        or set(extensions) - set(_EXTENSIONS)
        or any(not isinstance(value, Mapping) for value in extensions.values())
    ):
        raise ValueError("integer_extensions contains an unsupported extension")
    return copy.deepcopy(dict(policy))


def _extend(body: Mapping[str, Any], key: str, extension: Mapping[str, Any]) -> dict[str, Any]:
    if key != "operand_sum_grid":
        return Q.with_integer_extension(body, key=key, policy=extension)
    # The ordinary shared-grid constructor owns the schema, supported boundary
    # and residual selector. Do not accept additional policy fields silently.
    if "accumulator_residual" in body:
        raise ValueError("shared-grid sums and accumulator residuals have distinct graph ownership")
    selected = Q.with_shared_grid_operand_sums(
        body,
        output_boundary=extension.get("output_boundary", "quantized_readers"),
        residual_selection=extension.get("selection"),
    )
    if selected[key] != extension:
        raise ValueError("operand_sum_grid differs from the closed shared-grid policy")
    return selected


def select(
    source: Mapping[str, Any],
    methodology: Mapping[str, Any],
    output_path: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """Derive a new, explicitly selected reference contract from bound inputs.

    Weight scales use the existing observer and capability-derived epilogue;
    ``preserve`` retains the source scale policy. Dynamic activation planes are
    separate from static activation digit planes: an ambiguous combination refuses. Existing graph extensions
    retain their own legality checks. ``preserve`` retains the source recipe's
    float-quality thresholds; ``report_only`` removes only its site thresholds.
    Neither selection changes the separately selected end-to-end model gates
    or independent integer-reference equality.
    """
    selected_policy = validate(policy)
    if source.get("schema") != Q.SCHEMA or source.get("status") != Q.DERIVED:
        raise ValueError("numerical selection requires a derived recipe")
    if source.get("recipe_sha256") != Q.digest(source):
        raise ValueError("source recipe content identity differs")
    if source.get("accumulator_dtype") != "int32" or any(
        not isinstance(spec := source.get(operand), Mapping)
        or spec.get("dtype") != "int8"
        or spec.get("mode") != "static"
        or spec.get("symmetric") is not True
        for operand in ("weight", "activation")
    ):
        raise ValueError("numerical policy requires static symmetric I8 operands and I32 accumulation")
    method = M.validate(methodology)
    provenance = source.get("methodology") or {}
    method_sha = M.digest({k: v for k, v in method.items() if k != "model_variants"})
    if (
        provenance.get("methodology_sha256") != method_sha
        or provenance.get("output_path_sha256") != M.digest(output_path)
        or provenance.get("model") != model
    ):
        raise ValueError("selected methodology, model or output path differs from source recipe")
    if Q.STATIC_PLANES in source:
        raise ValueError("static activation digit planes need a separately selected numerical policy")
    derived_method = copy.deepcopy(method)
    if selected_policy["weight_granularity"] == "tensor":
        derived_method["scale_granularity"] = "per_tensor"
        for row in derived_method.get("model_variants", []):
            if row["model"] == model:
                row["changes"].pop("scale_granularity", None)
    body = M.derive(source, derived_method, output_path, model=model)
    if body.get("status") != Q.DERIVED:
        raise ValueError(f"selected numerical recipe is underivable: {body.get('underivable')}")
    if selected_policy["weight_granularity"] == "tensor":
        if body["weight"].get("granularity") != "tensor" or "channels_per_scale" in body["weight"]:
            raise ValueError("capability derivation did not select one weight scale per tensor")
    elif body["weight"] != source["weight"]:
        raise ValueError("capability derivation changed the preserved weight policy")
    original_epilogue = copy.deepcopy(source.get("epilogue"))
    derived_epilogue = copy.deepcopy(body.get("epilogue"))
    if not isinstance(original_epilogue, dict) or not isinstance(derived_epilogue, dict):
        raise ValueError("native selection requires the declared integer epilogue")
    original_epilogue.pop("output_quantization", None)
    if original_epilogue != derived_epilogue:
        raise ValueError("weight-scale derivation changed the physical output path")
    body = Q.with_native_output_quantization(body, **selected_policy["native_outputs"])
    for key in _EXTENSIONS:
        if key in source:
            body = _extend(body, key, source[key])
    # Reapply explicit extensions before this comparison: their families can
    # be more specific than a historical methodology's family selection.
    allowed = {"weight", "methodology", "recipe_sha256", "why", "epilogue"}
    if any(body.get(k) != source.get(k) for k in body.keys() | source.keys() if k not in allowed):
        raise ValueError("weight-scale derivation changed another arithmetic policy")
    for key, extension in selected_policy["integer_extensions"].items():
        body = _extend(body, key, extension)
    planes = selected_policy["dynamic_activation_planes"]
    if planes is not None:
        dynamic = source.get(Q.DYNAMIC)
        if not isinstance(dynamic, Mapping) or dynamic.get("schema") != Q.DYNAMIC_SCHEMA:
            raise ValueError("single-plane selection requires an existing dynamic contraction contract")
        if dynamic.get("families") != [Q.ACTIVATION_CONTRACTION]:
            raise ValueError("single-plane selection covers activation contractions only")
        known = {"schema", "families", "scales", "planes", "smooth", "context_output", Q.SITE_AGREEMENT}
        if set(dynamic) - known or dynamic.get("smooth") not in (None, Q.SOFTMAX_AXIS):
            raise ValueError("unsupported dynamic contraction policy")
        body = Q.with_dynamic_contractions(
            body,
            families=dynamic["families"],
            scales=dynamic["scales"],
            planes={key: planes for key in dynamic.get("planes", {})},
            smooth=dynamic.get("smooth") == Q.SOFTMAX_AXIS,
            context_output=selected_policy.get("attention_context_output", dynamic.get("context_output")),
        )
    if isinstance(body.get(Q.DYNAMIC), dict):
        Q.site_agreement_of(source)  # Malformed historical thresholds cannot disappear silently.
        if selected_policy["float_quality"] == "report_only":
            body[Q.DYNAMIC].pop(Q.SITE_AGREEMENT, None)
        elif (agreement := Q.site_agreement_of(source)) is not None:
            body[Q.DYNAMIC][Q.SITE_AGREEMENT] = agreement
    if selected_policy["float_quality"] == "report_only":
        body.setdefault("why", {}).pop(f"{Q.DYNAMIC}.{Q.SITE_AGREEMENT}", None)
    body["numerical_selection"] = {
        "schema": "numerical_recipe_selection_v1",
        "policy": selected_policy,
        "source_recipe_sha256": source["recipe_sha256"],
        "source_methodology_sha256": method_sha,
        "source_output_path_sha256": M.digest(output_path),
    }
    quality = (
        "source recipe float-quality thresholds retained"
        if selected_policy["float_quality"] == "preserve"
        else "recipe site float-quality thresholds report-only"
    )
    body["why"]["numerical_selection"] = (
        "explicit successor arithmetic; fresh calibrated captures and complete-chain quantized references required; "
        f"{quality}; separate model acceptance gates unchanged; no placement or speed claim"
    )
    body["recipe_sha256"] = Q.digest(body)
    return body


def derive(base, methodology, output_path, *, policy, model=None) -> dict[str, Any]:
    """Apply the same policy to an ordinary public capability-derived recipe.

    Public graph construction needs no private recipe or model identifier. An
    explicitly selected dynamic activation contract can be supplied by the
    methodology's existing variant mechanism; null planes introduce none.
    """
    if base.get("recipe_sha256") != Q.digest(base):
        raise ValueError("base recipe content identity differs")
    source = M.derive(base, methodology, output_path, model=model)
    return select(source, methodology, output_path, policy=policy, model=model)
