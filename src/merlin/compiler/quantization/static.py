"""Explicit static-grid, integer-boundary and weight-rounding recipe choices.

These constructors return a new numerical recipe. Framework application and
independent correctness/quality qualification remain separate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .recipe import (
    ACTIVATION_CONTRACTION as ACTIVATION_CONTRACTION,
)
from .recipe import (
    CANDIDATE_SCHEMA as CANDIDATE_SCHEMA,
)
from .recipe import (
    CONTRACTION as CONTRACTION,
)
from .recipe import (
    DERIVED as DERIVED,
)
from .recipe import (
    DYNAMIC as DYNAMIC,
)
from .recipe import (
    DYNAMIC_SCHEMA as DYNAMIC_SCHEMA,
)
from .recipe import (
    NON_REDUCTION as NON_REDUCTION,
)
from .recipe import (
    OPERAND as OPERAND,
)
from .recipe import (
    OPERAND_SUM as OPERAND_SUM,
)
from .recipe import (
    RESIDUAL_OPERAND as RESIDUAL_OPERAND,
)
from .recipe import (
    SCHEMA as SCHEMA,
)
from .recipe import (
    SITE_AGREEMENT as SITE_AGREEMENT,
)
from .recipe import (
    SITE_AGREEMENT_SCHEMA as SITE_AGREEMENT_SCHEMA,
)
from .recipe import (
    SOFTMAX_AXIS as SOFTMAX_AXIS,
)
from .recipe import (
    SOFTMAX_OUTPUT as SOFTMAX_OUTPUT,
)
from .recipe import (
    SOFTMAX_PARTNER as SOFTMAX_PARTNER,
)
from .recipe import (
    STATIC_HEADS as STATIC_HEADS,
)
from .recipe import (
    STATIC_PLANES as STATIC_PLANES,
)
from .recipe import (
    STATIC_PLANES_SCHEMA as STATIC_PLANES_SCHEMA,
)
from .recipe import (
    UNDERIVABLE as UNDERIVABLE,
)
from .recipe import (
    WEIGHT as WEIGHT,
)
from .recipe import (
    WINDOW_MEAN as WINDOW_MEAN,
)
from .recipe import (
    digest as digest,
)

_OBSERVERS = ("minmax", "histogram")


def with_native_output_quantization(
    recipe: Mapping[str, Any], *, operators: Sequence[str] = ("linear",), reader_policy: str = "non_add_compute"
) -> dict[str, Any]:
    """Select new contraction output grids before pure internal compute readers.

    The result is a different numerical recipe: float consumers receive the
    dequantized native output grid. Graph outputs and mutable/unknown readers
    retain their existing cut. Selection uses source operation/dataflow only.
    """
    from merlin.development.targetgen import _integer_epilogue as epilogue

    _static_int8(recipe, "native output quantization")
    if CONTRACTION not in (recipe.get("families") or ()) or recipe.get("accumulator_dtype") != "int32":
        raise ValueError("native output quantization requires integer contractions")
    if not isinstance(recipe.get("epilogue"), Mapping):
        raise ValueError("native output quantization requires a declared integer epilogue")
    body = json.loads(json.dumps(dict(recipe)))
    body["epilogue"]["output_quantization"] = {
        "schema": epilogue.OUTPUT_POLICY,
        "operators": list(operators),
        "reader_policy": reader_policy,
    }
    epilogue.check(body["epilogue"])
    body.setdefault("why", {})["epilogue.output_quantization"] = (
        "explicit new output grids before selected pure internal readers; native round/clamp is part of the reference"
    )
    body["recipe_sha256"] = digest(body)
    return body


def with_integer_extension(recipe: Mapping[str, Any], *, key: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Select one closed generic graph extension, with its own quantized reference.

    Existing target readout geometry is a legality input, never an instruction
    implementation or an execution claim. No extension is selected by default.
    """
    from merlin.development.targetgen import _integer_epilogue as epilogue

    activation = _static_int8(recipe, key)
    weight = recipe.get("weight") or {}
    if (
        activation.get("dtype") != "int8"
        or weight.get("dtype") != "int8"
        or weight.get("mode") != "static"
        or not weight.get("symmetric")
        or recipe.get("accumulator_dtype") != "int32"
        or CONTRACTION not in recipe.get("families", ())
    ):
        raise ValueError("integer extensions require static symmetric I8 contractions with I32 accumulation")
    block = epilogue.check(recipe.get("epilogue"))
    if not block["exits"]["requant"]:
        raise ValueError("integer extensions require an open native requantizing exit")
    if key == "accumulator_residual":
        from merlin.development.targetgen import _integer_residual as residual

        if "operand_sum_grid" in recipe:
            raise ValueError("accumulator residual and shared operand sums cannot own the same graph")
        residual.check(policy)
    elif key == "integer_mean":
        from merlin.development.targetgen import _integer_pool as pool

        pool.check(policy)
    elif key == "projection_input_reuse":
        from merlin.development.targetgen import _projection_input_reuse as projection

        projection.validate(policy, block["scale"]["columns_per_scale"])
    else:
        raise ValueError(f"unknown integer recipe extension {key!r}")
    body = json.loads(json.dumps(dict(recipe)))
    body[key] = dict(policy)
    if key == "integer_mean":
        body["families"] = list(dict.fromkeys([*body["families"], WINDOW_MEAN]))
        (body.get("unquantized") or {}).pop(WINDOW_MEAN, None)
    body.setdefault("why", {})[key] = (
        "explicit generic integer graph transform; its rewritten arithmetic defines a new reference; "
        "physical placement, reuse and performance require independent execution evidence"
    )
    body["recipe_sha256"] = digest(body)
    return body


def with_shared_grid_operand_sums(
    recipe: Mapping[str, Any], *, output_boundary: str = "quantized_readers", residual_selection: str | None = None
) -> dict[str, Any]:
    """Opt into one observed i8 grid per closed residual's inputs and output.

    The capture spells unit-grid add/ReLU/clamp, restoring real units only
    after the integer result. This is a new quantized recipe, not an accuracy
    or execution certificate. Existing independent placement and exact
    quantized-reference comparisons remain mandatory. Explicit ``dequantize``
    also permits a float consumer (including normalization) after the narrowed
    i8 result. Its float operations are unchanged, but its inputs and hence the
    quantized reference change. Prove the selected target's two unit-load/add/
    clamp path and qualify actual outputs before claiming device execution.
    ``residual_selection="distinct_contraction_frontiers"`` additionally requires
    disjoint first-contraction/input frontiers and at least one contraction;
    same-projection coordinate transforms are excluded without naming models.
    This selects a new recipe domain, not a guarantee of float-model accuracy.
    """
    from merlin.development.targetgen import _integer_operand_sum as sums

    body = json.loads(json.dumps(dict(recipe)))
    if CONTRACTION not in (body.get("families") or ()):
        raise ValueError("shared-grid sums need a recipe that quantizes contractions")
    if "accumulator_residual" in body:
        raise ValueError("accumulator residual and shared operand sums cannot own the same graph")
    body["families"] = list(dict.fromkeys([*(body.get("families") or ()), OPERAND_SUM]))
    if output_boundary not in {"quantized_readers", "dequantize"}:
        raise ValueError("shared-grid output boundary must be quantized_readers or dequantize")
    policy_schema = sums.FLOAT_BOUNDARY_SCHEMA if output_boundary == "dequantize" else sums.SCHEMA
    body[sums.KEY] = (
        {"schema": policy_schema, "output_boundary": "dequantize"}
        if output_boundary == "dequantize"
        else {"schema": policy_schema}
    )
    if residual_selection is not None:
        if output_boundary != "dequantize" or residual_selection != sums.DISTINCT_FRONTIERS:
            raise ValueError("residual selection requires dequantize and distinct_contraction_frontiers")
        body[sums.KEY]["selection"] = residual_selection
    sums.validate(body)
    body.setdefault("unquantized", {}).pop(OPERAND_SUM, None)
    body.setdefault("why", {})[OPERAND_SUM] = (
        "opted in: each closed residual's inputs and output share one observed activation grid; "
        "unit i8 load/readout multipliers preserve exact integer add, optional ReLU and saturation"
    )
    if output_boundary == "dequantize":
        body["why"][OPERAND_SUM] += (
            "; explicit V2 boundary dequantizes the narrowed result for float readers; "
            "normalization stays floating-point and target execution needs independent qualification"
        )
    if residual_selection is not None:
        body["why"][OPERAND_SUM] += (
            "; select disjoint first-contraction/input frontiers, excluding same-origin coordinate sums"
        )
    if "methodology" in body:
        extensions = body["methodology"].setdefault("extensions", [])
        if policy_schema not in extensions:
            extensions.append(policy_schema)
    body["recipe_sha256"] = digest(body)
    return body


def with_activation_contractions(
    recipe: Mapping[str, Any],
    *,
    minimum_reduction: int | None = None,
) -> dict[str, Any]:
    """``recipe`` (a derived ``quant_recipe_v1`` document) that ALSO quantizes activation contractions.

    The switch is the family :data:`ACTIVATION_CONTRACTION` in ``families``; everything else is the
    recipe's own. Both operands of such a contraction take the recipe's activation spec, so for
    attention ``Q``, ``K``, the softmax probabilities ``P`` and ``V`` each get one static, symmetric
    scale per tensor (per call site, shared by every head), calibrated by the activation observer
    like any other activation edge. The product of two per-tensor scales is again one scale, which
    is what a readout that holds one scale per tensor applies; a finer scale on either operand is a
    run-time vector the readout cannot hold. Only a static per-tensor integer recipe is extended:
    any other form is refused with the reason, never approximated. The content digest changes with
    the family, so a capture made under the variant is never served for one made without it.
    ``minimum_reduction`` explicitly replaces the historical >16 profitability screen; it
    does not change integer formats, software admission, or numerical quality bounds.
    """
    if recipe.get("schema") != SCHEMA or recipe.get("status") != DERIVED:
        raise ValueError("only a derived quant_recipe_v1 can be extended")
    activation = recipe.get("activation") or {}
    families = list(recipe.get("families") or ())
    refusal = (
        "the recipe quantizes no contraction"
        if CONTRACTION not in families
        else f"activations are {activation.get('mode')!r}, not static"
        if activation.get("mode") != "static"
        else f"activation scales are per {activation.get('granularity')!r}, not one per tensor"
        if activation.get("granularity") != "tensor"
        else "the activation format is not a symmetric integer"
        if not (activation.get("symmetric") and str(activation.get("dtype", "")).startswith("int"))
        else None
    )
    if refusal is not None:
        raise ValueError(f"activation contractions cannot extend this recipe: {refusal}")
    body = json.loads(json.dumps(dict(recipe)))
    if ACTIVATION_CONTRACTION not in families:
        body["families"] = [*families, ACTIVATION_CONTRACTION]
    (body.get("unquantized") or {}).pop(ACTIVATION_CONTRACTION, None)
    body.setdefault("why", {})[ACTIVATION_CONTRACTION] = (
        "opted in: a contraction of two run-time activations (attention scores and context) is "
        "int8 x int8 into the accumulator with both operands on the activation spec, one static "
        "scale per tensor each; the readout's one scale is their product"
    )
    if minimum_reduction is not None:
        body["activation_contraction_policy"] = {
            "schema": "activation_contraction_admission_v1",
            "minimum_reduction": minimum_reduction,
        }
    from merlin.development.targetgen._activation_contractions import activation_reduction_minimum

    activation_reduction_minimum(body)
    body["recipe_sha256"] = digest(body)
    return body


def with_gptq_weights(recipe: Mapping[str, Any], **parameters: Any) -> dict[str, Any]:
    """``recipe`` (a derived ``quant_recipe_v1`` document) whose weight codes are chosen by GPTQ.

    The switch is ``weight.rounding = "gptq"`` with ``weight.gptq`` holding every parameter of
    :data:`merlin.development.targetgen._gptq.PARAMETERS`. ``parameters`` override
    :data:`~merlin.development.targetgen._gptq.DEFAULT_PARAMETERS`. The grid stays the recipe's own: the same
    dtype, range, symmetry and scale granularity, with the same observed scales. GPTQ only picks
    which integer each weight takes on that grid, using the calibration inputs of its layer (see
    ``_gptq.py``). Without the key a recipe rounds to nearest and its digest is what it always was.
    With it the digest changes, so a capture made under GPTQ codes is never served for one made
    under nearest codes.

    Refused with the reason, never approximated: a recipe that quantizes no contraction, dynamic
    activations (``quantize_`` rounds the weights itself), a weight format that is not a symmetric
    integer, and a weight scale granularity that the static path cannot express.
    """
    from merlin.development.targetgen import _gptq

    if recipe.get("schema") != SCHEMA or recipe.get("status") != DERIVED:
        raise ValueError("only a derived quant_recipe_v1 can be extended")
    weight = recipe.get("weight") or {}
    activation = recipe.get("activation") or {}
    refusal = (
        "the recipe quantizes no contraction"
        if CONTRACTION not in (recipe.get("families") or ())
        else f"activations are {activation.get('mode')!r}; GPTQ codes are set on the static path"
        if activation.get("mode") != "static"
        else "the weight format is not a symmetric integer with a declared range"
        if not (
            weight.get("symmetric")
            and str(weight.get("dtype", "")).startswith("int")
            and isinstance(weight.get("quant_min"), int)
            and isinstance(weight.get("quant_max"), int)
        )
        else f"weight scales per {weight.get('granularity')!r} have no static spec to round onto"
        if weight.get("granularity") not in ("tensor", "channel")
        else None
    )
    if refusal is not None:
        raise ValueError(f"GPTQ weights cannot extend this recipe: {refusal}")
    chosen = _gptq.check_parameters({**_gptq.DEFAULT_PARAMETERS, **parameters})
    body = json.loads(json.dumps(dict(recipe)))
    body["weight"]["rounding"] = _gptq.GPTQ
    body["weight"]["gptq"] = chosen
    body.setdefault("why", {})["weight.rounding"] = (
        f"opted in: GPTQ picks each weight's code on the recipe's own grid, propagating rounding "
        f"error through the inverse Hessian of the layer's first {chosen['calibration_samples']} "
        f"calibration inputs (act order {chosen['act_order']}, damp {chosen['damp']}, block {chosen['block']})"
    )
    body["recipe_sha256"] = digest(body)
    return body


def _observer_valid(name: Any) -> bool:
    from merlin.development.targetgen._recipe_observers import percentile_of

    return name in _OBSERVERS or percentile_of(name) is not None


def _static_int8(recipe: Mapping[str, Any], what: str) -> Mapping[str, Any]:
    if recipe.get("schema") != SCHEMA or recipe.get("status") != DERIVED:
        raise ValueError("only a derived quant_recipe_v1 can be extended")
    activation = recipe.get("activation") or {}
    if activation.get("mode") != "static" or activation.get("granularity") != "tensor":
        raise ValueError(f"{what} needs static per-tensor activations")
    if not (activation.get("symmetric") and str(activation.get("dtype", "")).startswith("int")):
        raise ValueError(f"{what} needs a symmetric integer activation format")
    return activation


def static_planes_block(
    *, planes: int = 2, observer: str = "minmax", owners: Sequence[str] | None = None
) -> dict[str, Any]:
    """The validated, canonical framework policy for static activation digit planes."""
    if type(planes) is not int or not 2 <= planes <= 3:
        raise ValueError(f"static planes take 2 or 3 digits, not {planes!r}")
    if not _observer_valid(observer):
        raise ValueError(f"observer {observer!r} is not minmax, histogram or p<percent>")
    if owners is not None and (not owners or any(not isinstance(o, str) or not o for o in owners)):
        raise ValueError("owners are non-empty module-path prefixes")
    return {
        "schema": STATIC_PLANES_SCHEMA,
        "families": [CONTRACTION],
        "planes": planes,
        "observer": observer,
        "owners": None if owners is None else sorted(set(owners)),
    }


def with_static_planes(
    recipe: Mapping[str, Any],
    *,
    planes: int = 2,
    observer: str = "minmax",
    owners: Sequence[str] | None = None,
) -> dict[str, Any]:
    """``recipe`` whose Linear and Conv2d activations (in ``owners``' modules, default all) take ``planes`` digits.

    The activation keeps ONE static scale per tensor, on a grid ``planes`` digits wide: with the
    recipe's activation range ``[low, high]`` and ``base = high + 1``, the top digit spans
    ``[low, high]`` and each lower digit ``[0, base - 1]`` (two int8 digits: 15 bits). The device
    multiplies every digit plane by the same int8 weight; the host combines the accumulators
    (``base * acc_top + acc_low``) inside the per-column rescale it already applies, so a Linear
    costs ``planes`` device contractions and no extra host pass. ``observer`` calibrates the wide
    grid (``minmax``, ``histogram`` or ``p<percent>``); ``owners`` are module-path prefixes, so a
    model's recipe can widen one part of it. The digest changes with the block.
    """
    activation = _static_int8(recipe, "static planes")
    if CONTRACTION not in (recipe.get("families") or ()):
        raise ValueError("static planes need a recipe that quantizes contractions")
    block = static_planes_block(planes=planes, observer=observer, owners=owners)
    low, high = activation.get("quant_min"), activation.get("quant_max")
    if not (isinstance(low, int) and isinstance(high, int) and low < 0 < high):
        raise ValueError("static planes need the recipe's signed activation range")
    body = json.loads(json.dumps(dict(recipe)))
    body[STATIC_PLANES] = block
    body.setdefault("why", {})[STATIC_PLANES] = (
        "opted in: a Linear's activation on a static per-tensor grid of int8 digits, each digit plane "
        "multiplied by the same int8 weight on the unit; the host sums base-weighted accumulators in the "
        "per-column rescale it already applies"
    )
    body["recipe_sha256"] = digest(body)
    return body


def with_activation_observer(recipe: Mapping[str, Any], observer: str) -> dict[str, Any]:
    """``recipe`` whose static activation scales are calibrated by ``observer`` (``p99.999``: a percentile)."""
    _static_int8(recipe, "an activation observer")
    if not _observer_valid(observer):
        raise ValueError(f"observer {observer!r} is not minmax, histogram or p<percent>")
    body = json.loads(json.dumps(dict(recipe)))
    body["activation"]["observer"] = observer
    body.setdefault("why", {})["activation.observer"] = f"selected: activation ranges by {observer}"
    body["recipe_sha256"] = digest(body)
    return body
