"""Framework-independent recipe records, serialized vocabulary and content identity.

These describe numerical preparation intent, not observed framework realization,
placement, application quality or target execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# ---- recipe ------------------------------------------------------------------------------------

SCHEMA = "quant_recipe_v1"

CANDIDATE_SCHEMA = "quant_recipe_candidate_v1"

DERIVED = "derived"

UNDERIVABLE = "underivable"

CONTRACTION = "contraction"

OPERAND_SUM = "operand_sum"

WINDOW_MEAN = "window_mean"

ACTIVATION_CONTRACTION = "activation_contraction"

DYNAMIC = "dynamic"

DYNAMIC_SCHEMA = "dynamic_contractions_v1"

NON_REDUCTION = "non_reduction"

STATIC_HEADS = "static_heads"

SOFTMAX_OUTPUT = "softmax_output"

WEIGHT = "weight"

OPERAND = "operand"

SOFTMAX_PARTNER = "softmax_partner"

RESIDUAL_OPERAND = "residual_operand"

SOFTMAX_AXIS = "softmax_axis"

STATIC_PLANES = "static_planes"

STATIC_PLANES_SCHEMA = "static_planes_v1"

SITE_AGREEMENT = "site_agreement"

SITE_AGREEMENT_SCHEMA = "dynamic_site_agreement_v1"


@dataclass(frozen=True)
class TensorSpec:
    dtype: str  # a quant-format registry name
    granularity: str  # "tensor" | "channel" | "token" | "block"
    symmetric: bool
    quant_min: int | None
    quant_max: int | None
    block: int | None = None
    mode: str = "static"  # activations only: "static" | "dynamic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "granularity": self.granularity,
            "symmetric": self.symmetric,
            "quant_min": self.quant_min,
            "quant_max": self.quant_max,
            "block": self.block,
            "mode": self.mode,
        }


@dataclass
class QuantRecipe:
    target: str
    unit: str | None
    status: str = DERIVED
    weight: TensorSpec | None = None
    activation: TensorSpec | None = None
    bias_domain: str | None = None  # "accumulator" when the readout adds an integer bias
    families: tuple[str, ...] = (CONTRACTION,)
    unquantized: dict[str, str] = field(default_factory=dict)  # family -> why it stays float
    why: dict[str, str] = field(default_factory=dict)  # field -> the derivation behind it
    underivable: dict[str, str] = field(default_factory=dict)  # field -> why not

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "target": self.target,
            "unit": self.unit,
            "status": self.status,
            "families": list(self.families),
            "weight": self.weight.to_dict() if self.weight else None,
            "activation": self.activation.to_dict() if self.activation else None,
            "bias_domain": self.bias_domain,
            "unquantized": dict(sorted(self.unquantized.items())),
            "why": dict(sorted(self.why.items())),
            "underivable": dict(sorted(self.underivable.items())),
        }
        body["recipe_sha256"] = digest(body)
        return body


@dataclass(frozen=True)
class QuantCandidate:
    """One *declared* operand format, with the facts that do or do not license a recipe.

    This is an inventory of hardware-side candidates, not a promise that TorchAO can produce the
    format or that a compiler can lower it. In particular an alternative format on a multi-format
    unit must not borrow the readout facts derived for the unit's first/RTL-visible format.
    """

    target: str
    unit: str
    format: str
    recipe: QuantRecipe | None
    evidence: tuple[dict[str, str], ...]
    underivable: dict[str, str]

    @property
    def status(self) -> str:
        return DERIVED if self.recipe is not None else UNDERIVABLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_SCHEMA,
            "target": self.target,
            "unit": self.unit,
            "format": self.format,
            "status": self.status,
            "recipe": self.recipe.to_dict() if self.recipe is not None else None,
            "evidence": [dict(item) for item in self.evidence],
            "underivable": dict(sorted(self.underivable.items())),
            "framework_route": "not_evaluated",
            "compiler_route": "not_evaluated",
        }


def digest(body: Mapping[str, Any]) -> str:
    """Identity of a recipe's CONTENT: what a capture cache and a receipt key on."""
    payload = {k: v for k, v in body.items() if k not in ("recipe_sha256", "why")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# ---- dynamic -----------------------------------------------------------------------------------
# Explicit runtime-scale and site-agreement numerical recipe declarations.
#
# Dynamic qparams remain computations when realized; this data construction does
# not execute a framework, select placement or issue an accuracy verdict.

_MAX_PLANES = 4


def with_dynamic_contractions(
    recipe: Mapping[str, Any],
    *,
    families: Sequence[str] = (ACTIVATION_CONTRACTION,),
    planes: Mapping[str, int] | None = None,
    smooth: bool = False,
    static_families: Sequence[str] | None = None,
    scales: str = NON_REDUCTION,
    context_output: str | None = None,
) -> dict[str, Any]:
    """``recipe`` with a ``dynamic`` block: these families' contractions take run-time scales.

    Three general mechanisms, each named by the block and none by a model:

    * dynamic scales on non-reduction axes (:data:`NON_REDUCTION`): the host takes each row's absmax
      at run time, quantizes, the device runs a plain int8 matmul, and the host rescales the i32
      result by the outer product of the row and column scales. The readout holds one scale per
      tensor, so these scales are host work by construction;
    * planes: an operand selected by ``planes`` (:data:`SOFTMAX_OUTPUT`, :data:`WEIGHT`) is split into
      k int8 planes, each the quantized residual of the ones before, with its own row scales: k
      device matmuls against the same other operand, summed on the host;
    * the smooth shift (:data:`SOFTMAX_AXIS`), applied only where it preserves the function.

    ``families`` must be contraction families of this recipe (``activation_contraction`` is added
    when asked for); ``static_families`` (default: the recipe's own) are the families still
    quantized with the recipe's static spec. Families listed in both take the dynamic form.

    ``scales=STATIC_HEADS`` keeps the same mechanisms with scales fixed at capture: one calibrated
    minmax scale per head of each operand, the smooth shift a calibrated per-(head, row) mean (folded
    into the producing Linear's bias where only layout steps lie between), and a softmax-produced
    operand replaced by ``exp(s - rowmax)`` in planes with fixed scales ``1 / (q * (q + 1)**j)``, the
    rows divided by their sum after the contraction (``_static_attention``). ``planes[OPERAND] = k``
    puts every other operand on a static per-head grid of k int8 digits (two digits: 15 bits), each
    pair of digit planes one device call with one scale; ``planes[SOFTMAX_PARTNER] = k`` selects bounded
    round-and-clamp residual planes for the operand a softmax output multiplies, each successive grid
    ``quant_max + 1`` times finer. ``planes[RESIDUAL_OPERAND] = k`` uses those bounded residuals for
    every ordinary operand, including both score operands; it replaces the historical ``OPERAND``
    selector, whose radix arithmetic stays unchanged. Activation contractions of a static recipe only.

    ``context_output="int8_value_grid"`` explicitly selects single-plane normalized
    probabilities on a 1/127 grid and requantizes each P×V product onto its value
    operand's grid before joining heads. Q×K still leaves as INT32. This changes
    the numerical recipe; it is not equivalent to the exp-plane formulation.
    """
    if recipe.get("schema") != SCHEMA or recipe.get("status") != DERIVED:
        raise ValueError("only a derived quant_recipe_v1 can be extended")
    activation = recipe.get("activation") or {}
    if not (activation.get("symmetric") and str(activation.get("dtype", "")).startswith("int")):
        raise ValueError("dynamic contractions need a symmetric integer activation format")
    wanted = list(dict.fromkeys(families))
    if not wanted or any(f not in (CONTRACTION, ACTIVATION_CONTRACTION) for f in wanted):
        raise ValueError(f"dynamic scales apply to contraction families only, not {wanted}")
    selected = dict(planes or {})
    if any(
        k not in (SOFTMAX_OUTPUT, WEIGHT, OPERAND, SOFTMAX_PARTNER, RESIDUAL_OPERAND)
        or type(v) is not int
        or not 1 <= v <= _MAX_PLANES
        for k, v in selected.items()
    ):
        raise ValueError(
            f"planes take {SOFTMAX_OUTPUT!r}/{WEIGHT!r}/{OPERAND!r}/{SOFTMAX_PARTNER!r}/{RESIDUAL_OPERAND!r} "
            f"-> 1..{_MAX_PLANES}, not {selected}"
        )
    if OPERAND in selected and RESIDUAL_OPERAND in selected:
        raise ValueError("operand and residual_operand selectors are mutually exclusive")
    for digits in (OPERAND, SOFTMAX_PARTNER, RESIDUAL_OPERAND):
        if digits in selected and (scales != STATIC_HEADS or selected[digits] > 3):
            raise ValueError(f"{digits!r} digit planes (at most 3) belong to static per-head scales")
    if scales not in (NON_REDUCTION, STATIC_HEADS):
        raise ValueError(f"scales are {NON_REDUCTION!r} or {STATIC_HEADS!r}, not {scales!r}")
    if scales == STATIC_HEADS and (
        wanted != [ACTIVATION_CONTRACTION] or WEIGHT in selected or activation.get("mode") != "static"
    ):
        raise ValueError("static per-head scales cover activation contractions of a static recipe only")
    if context_output is not None and (
        context_output != "int8_value_grid"
        or scales != STATIC_HEADS
        or activation.get("dtype") != "int8"
        or (activation.get("quant_min"), activation.get("quant_max")) != (-128, 127)
        or any(value != 1 for value in selected.values())
    ):
        raise ValueError("int8_value_grid context readout requires static scales and single-plane operands")
    kept = list(recipe.get("families") or ()) if static_families is None else list(static_families)
    unknown = [f for f in kept if f not in (recipe.get("families") or ())]
    if unknown:
        raise ValueError(f"static families {unknown} are not the recipe's own")
    body = json.loads(json.dumps(dict(recipe)))
    body["families"] = list(dict.fromkeys([*kept, *wanted]))
    for family in wanted:
        (body.get("unquantized") or {}).pop(family, None)
    body[DYNAMIC] = {
        "schema": DYNAMIC_SCHEMA,
        "families": wanted,
        "scales": scales,
        "planes": dict(sorted(selected.items())),
        "smooth": SOFTMAX_AXIS if smooth else None,
    }
    body.setdefault("why", {}).pop("dynamic.context_output", None)
    if context_output is not None:
        body[DYNAMIC]["context_output"] = context_output
        body.setdefault("why", {})["dynamic.context_output"] = (
            "explicit numerical variant: quantize normalized probabilities on the 1/127 grid, "
            "perform an integer contraction and requantize each matrix result onto its value operand grid; "
            "no post-contraction row-sum division and no change to score readouts"
        )
    body.setdefault("why", {})[DYNAMIC] = (
        "opted in: run-time scales on non-reduction axes (rows of the streamed operand, columns of the "
        "other, a stored weight's output channels), applied by the host to the exact i32 accumulator "
        "as an outer product; the readout's one scale cannot hold them"
        if scales == NON_REDUCTION
        else "opted in: one calibrated scale per head of each operand, so every per-head device call reads "
        "one scale; softmax probabilities as fixed-scale planes of exp(s - rowmax), rows divided by their sum "
        "on the host"
    )
    if context_output is not None:
        body["why"][DYNAMIC] = body["why"]["dynamic.context_output"]
    body["recipe_sha256"] = digest(body)
    return body


def site_agreement_of(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """The validated per-site bar of ``recipe``'s dynamic block, or None when it declares none."""
    import math

    block = recipe.get(DYNAMIC) if isinstance(recipe, Mapping) else None
    bar = block.get(SITE_AGREEMENT) if isinstance(block, Mapping) else None
    if bar is None:
        return None
    if (
        not isinstance(bar, Mapping)
        or set(bar) != {"schema", "cosine_min", "rel_rmse_max"}
        or bar.get("schema") != SITE_AGREEMENT_SCHEMA
        or any(type(bar.get(key)) is not float or not math.isfinite(bar[key]) for key in ("cosine_min", "rel_rmse_max"))
        or not 0.0 < bar["cosine_min"] <= 1.0
        or not bar["rel_rmse_max"] > 0.0
    ):
        raise ValueError(f"the dynamic block's {SITE_AGREEMENT} is not an exact {SITE_AGREEMENT_SCHEMA} bar")
    return dict(bar)


def with_site_agreement(recipe: Mapping[str, Any], *, cosine_min: float, rel_rmse_max: float, basis: str) -> dict:
    """``recipe`` whose dynamic contraction sites are each held to a declared agreement bar.

    The bar is the recipe's own: a capture records each site's measured agreement with the float
    contraction of its operands, and the sealed issuer refuses a site below it. ``basis`` says where the
    numbers came from (e.g. a measured per-site distribution and its margin); it is recorded, not judged.
    """
    if recipe.get("schema") != SCHEMA or recipe.get("status") != DERIVED or not isinstance(recipe.get(DYNAMIC), dict):
        raise ValueError("a site agreement bar extends a derived recipe that has a dynamic block")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("a site agreement bar records its basis")
    body = json.loads(json.dumps(dict(recipe)))
    body[DYNAMIC][SITE_AGREEMENT] = {
        "schema": SITE_AGREEMENT_SCHEMA,
        "cosine_min": float(cosine_min),
        "rel_rmse_max": float(rel_rmse_max),
    }
    site_agreement_of(body)
    body.setdefault("why", {})[f"{DYNAMIC}.{SITE_AGREEMENT}"] = basis.strip()
    body["recipe_sha256"] = digest(body)
    return body


# ---- static ------------------------------------------------------------------------------------
# Explicit static-grid, integer-boundary and weight-rounding recipe choices.
#
# These constructors return a new numerical recipe. Framework application and
# independent correctness/quality qualification remain separate.

_OBSERVERS = ("minmax", "histogram")


def with_native_output_quantization(
    recipe: Mapping[str, Any], *, operators: Sequence[str] = ("linear",), reader_policy: str = "non_add_compute"
) -> dict[str, Any]:
    """Select new contraction output grids before pure internal compute readers.

    The result is a different numerical recipe: float consumers receive the
    dequantized native output grid. Graph outputs and mutable/unknown readers
    retain their existing cut. Selection uses source operation/dataflow only.
    """
    from merlin.compiler.quantization.torchao.rewrites import _integer_epilogue as epilogue

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
    from merlin.compiler.quantization.torchao.rewrites import _integer_epilogue as epilogue

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
        from merlin.compiler.quantization.torchao.rewrites import _integer_residual as residual

        if "operand_sum_grid" in recipe:
            raise ValueError("accumulator residual and shared operand sums cannot own the same graph")
        residual.check(policy)
    elif key == "integer_mean":
        from merlin.compiler.quantization.torchao.rewrites import _integer_pool as pool

        pool.check(policy)
    elif key == "projection_input_reuse":
        from merlin.compiler.quantization.torchao.rewrites import _projection_input_reuse as projection

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
    from merlin.compiler.quantization.torchao.rewrites import _integer_operand_sum as sums

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
    from merlin.compiler.quantization.torchao.rewrites._activation_contractions import (
        activation_reduction_minimum,
    )

    activation_reduction_minimum(body)
    body["recipe_sha256"] = digest(body)
    return body


def with_gptq_weights(recipe: Mapping[str, Any], **parameters: Any) -> dict[str, Any]:
    """``recipe`` (a derived ``quant_recipe_v1`` document) whose weight codes are chosen by GPTQ.

    The switch is ``weight.rounding = "gptq"`` with ``weight.gptq`` holding every parameter of
    :data:`merlin.compiler.quantization.torchao.rewrites._gptq.PARAMETERS`. ``parameters`` override
    :data:`~merlin.compiler.quantization.torchao.rewrites._gptq.DEFAULT_PARAMETERS`. The grid stays the recipe's own: the same
    dtype, range, symmetry and scale granularity, with the same observed scales. GPTQ only picks
    which integer each weight takes on that grid, using the calibration inputs of its layer (see
    ``_gptq.py``). Without the key a recipe rounds to nearest and its digest is what it always was.
    With it the digest changes, so a capture made under GPTQ codes is never served for one made
    under nearest codes.

    Refused with the reason, never approximated: a recipe that quantizes no contraction, dynamic
    activations (``quantize_`` rounds the weights itself), a weight format that is not a symmetric
    integer, and a weight scale granularity that the static path cannot express.
    """
    from merlin.compiler.quantization.torchao.rewrites import _gptq

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
    from merlin.compiler.quantization.torchao.rewrites._recipe_observers import percentile_of

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
