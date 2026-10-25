"""The quantization a model should be captured under, derived from the target that will run it.

A model used to be quantized under a scheme picked by name from a table keyed on a dtype token:
``int8`` meant one TorchAO recipe, whatever hardware the result was for. The recipe's defaults
then became the compiler's problem. Per-channel weight scales, the default of every static int8
recipe, meet a store path that holds one scale per command, and the requantization of every layer
lands on the host for the life of the program. Nothing chose that; a default did.

A recipe here is DATA, derived from what the target's readout was derived to hold
(:mod:`merlin.targetgen.readout_facet`):

* element formats come from the unit's datapath;
* the weight scale granularity is the finest one the readout holds, because a finer scale is a
  more accurate model and a coarser one is the only kind the hardware can absorb;
* symmetry follows from whether the readout carries a zero point;
* ranges follow from the element format and the readout's clamp;
* a family the target cannot absorb stays in floating point, and says so.

It crosses to the capture worker as JSON, where a generic quantizer builds the framework's own
specs from it. Nothing here imports a framework, names a target, or knows a scheme name. A field
the facet could not derive makes the recipe ``underivable`` with the reason; it is never filled
with a common default, because a common default is how this went wrong.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.common import quant_formats as qf

SCHEMA = "quant_recipe_v1"
CANDIDATE_SCHEMA = "quant_recipe_candidate_v1"
DERIVED = "derived"
UNDERIVABLE = "underivable"

#: Facet granularity -> the recipe's weight/activation granularity word. A recipe speaks about the
#: TENSOR being quantized ("one scale per output channel of the weight"), a facet about the
#: accumulator ("one scale per column"); this is the one place they meet.
_WEIGHT_GRANULARITY = {"column": "channel", "row": "channel", "rank1": "channel", "tensor": "tensor", "block": "block"}
#: Finest first: a recipe takes the first of these the readout holds.
_WEIGHT_PREFERENCE = ("block", "rank1", "column", "row", "tensor")
#: An activation scale can vary per row of the accumulator (per token) only where the readout
#: holds a row or rank-1 scale; otherwise it is one scale for the tensor.
_ACTIVATION_PREFERENCE = ("rank1", "row", "tensor")
_ACTIVATION_GRANULARITY = {"rank1": "token", "row": "token", "tensor": "tensor"}

#: Operator families a recipe quantizes. A contraction is the only family a matrix unit computes;
#: everything else is quantized only as far as it rides on a contraction's readout.
CONTRACTION = "contraction"
#: A sum of separately scaled tensors (a residual connection). Quantized only when a unit's load
#: multiplies; otherwise both operands stay in floating point and the add is host work by choice.
OPERAND_SUM = "operand_sum"
#: A mean over a window (an average pool). On a unit that contracts it is a contraction against a
#: constant one with the reciprocal of the count in the readout's scale, so its input is quantized
#: wherever contractions are and the readout multiplies.
WINDOW_MEAN = "window_mean"
#: A contraction of two RUN-TIME activations (attention's ``Q @ K^T`` and ``P @ V``, softmax
#: probabilities included). Never derived by default: no readout fact distinguishes it from a
#: weight contraction, so it is a numerical CHOICE a recipe opts into (:func:`with_activation_contractions`),
#: and a recipe that does not list it keeps every such contraction in floating point.
ACTIVATION_CONTRACTION = "activation_contraction"
#: An affine map of at most two int8 tensors (scalar add/sub/mul/div, neg, relu, a difference):
#: integer weights and an int32 constant in the accumulator, then one readout. Quantized only where
#: the unit declares a standalone elementwise path AND a standalone contraction (the exact integer
#: weights are constant c*I contractions; merlin.targetgen.recipe_numerics).
AFFINE_MAP = "affine_map"
#: Data motion between int8 tensors (copy, permute, cat, pads, scatters, repeat, fills). Quantized
#: where the unit declares a movement capability: a move needs no arithmetic, at most one load rounding.
MOVEMENT = "movement"


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
    #: Accelerator capture keeps exact integer semantics and full-width int32 device readout.
    readout_numerics: str = "integer_readout"
    #: Capture identity includes the declared policy; exact sums are the default.
    integer_sum_policy: str = "exact"
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
        body["readout_numerics"] = self.readout_numerics
        body["integer_sum_policy"] = self.integer_sum_policy
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


def _integer_range(bits: int, *, symmetric_restricted: bool) -> tuple[int, int]:
    high = (1 << (bits - 1)) - 1
    return (-high if symmetric_restricted else -high - 1), high


def _format_of(mlir_or_token: str | None) -> qf.QuantFormat | None:
    if not mlir_or_token:
        return None
    token = str(mlir_or_token)
    candidates = [token]
    if token[:1] == "i" and token[1:].isdigit():
        candidates.append("int" + token[1:])
    for candidate in candidates:
        if qf.has(candidate):
            return qf.get(candidate)
    return None


def _unit_families(recipe: QuantRecipe, capabilities: Sequence[Any] | None) -> None:
    """The non-contraction families a unit's DECLARED capabilities license on a derived recipe.

    Only under integer activations with one static scale per tensor, which is what one readout or
    one load scale holds. Absent capabilities (a caller without the contract) license nothing.
    """
    if capabilities is None or recipe.status != DERIVED or recipe.activation is None:
        return
    activation = recipe.activation
    fmt = _format_of(activation.dtype)
    integer = fmt is not None and fmt.kind == "int_affine"
    if not (integer and activation.granularity == "tensor" and activation.mode == "static"):
        for family in (AFFINE_MAP, MOVEMENT):
            recipe.unquantized[family] = (
                "an exact integer form needs integer activations under one static scale per tensor; "
                f"these are per {activation.granularity} ({activation.mode})"
            )
        return

    def standalone(family: str) -> bool:
        return any(cap.family == family and not cap.composed_with for cap in capabilities)

    if standalone("elementwise_map") and standalone(CONTRACTION):
        recipe.families = (*recipe.families, AFFINE_MAP)
        recipe.why[AFFINE_MAP] = (
            "the unit runs elementwise maps standalone and contracts, so integer weights apply as "
            "constant contractions into the accumulator and the map reads out once"
        )
    else:
        recipe.unquantized[AFFINE_MAP] = "the unit declares no standalone elementwise path beside a contraction"
    if any(cap.family == "movement" for cap in capabilities):
        recipe.families = (*recipe.families, MOVEMENT)
        recipe.why[MOVEMENT] = "the unit declares data movement, which needs at most one load rounding"
    else:
        recipe.unquantized[MOVEMENT] = "the unit declares no movement capability"


def derive(
    facet: Any,
    *,
    prefer_activation_mode: str | None = None,
    capabilities: Sequence[Any] | None = None,
    integer_sum_policy: str | None = None,
) -> QuantRecipe:
    """The recipe a unit's readout facet licenses.

    ``prefer_activation_mode`` is a workload's request ("dynamic" for a model whose activation
    range moves with its input). It is honoured where the facet allows and recorded where it does
    not: a recipe never promises a mode the hardware would push onto the host silently.
    ``integer_sum_policy`` records the flow's declared numeric policy for integer sums.
    Exact is the default. Target readouts still require exact sums and product
    proofs even under an explicitly retained legacy bounded declaration.
    A per_operand scheme needs a capture with explicit per-operand requantization
    nodes; it cannot be implemented by relaxing the exact readout facet.
    """
    policy = "exact" if integer_sum_policy is None else str(integer_sum_policy)
    if policy == "per_operand":
        raise ValueError(
            "per_operand requires an explicit per-operand requantization capture, not an exact readout facet"
        )
    if policy not in ("bounded", "exact"):
        raise ValueError(f"integer sum policy {integer_sum_policy!r} is not one of ('bounded', 'exact')")
    # Policy provenance never relaxes the target's exact sum and product proofs.
    recipe = QuantRecipe(target=facet.target, unit=facet.unit, integer_sum_policy=policy)

    element = _format_of(facet.element_dtype)
    if element is None:
        recipe.underivable["element_dtype"] = facet.unknown.get(
            "element_dtype", f"element dtype {facet.element_dtype!r} is not a registered format"
        )
    held = facet.scale_granularities
    if held is None:
        recipe.underivable["granularity"] = facet.unknown.get(
            "scale_granularities", "the readout's scale granularity is not derived"
        )
    if facet.zero_point_carried is None and (element is None or element.kind == "int_affine"):
        recipe.underivable["symmetric"] = facet.unknown.get(
            "zero_point_carried", "whether the readout carries a zero point is not derived"
        )
    if recipe.underivable:
        recipe.status = UNDERIVABLE
        return recipe

    integer = element.kind == "int_affine"
    symmetric = not facet.zero_point_carried if integer else True
    weight_held = (
        next(g for g in _WEIGHT_PREFERENCE if g in held) if any(g in held for g in _WEIGHT_PREFERENCE) else None
    )
    activation_held = next((g for g in _ACTIVATION_PREFERENCE if g in held), None)
    if weight_held is None or (activation_held is None and "block" not in held):
        recipe.status = UNDERIVABLE
        recipe.underivable["granularity"] = (
            f"the readout holds scales per {list(held)}, none of which a weight or activation scale can take"
        )
        return recipe

    block = facet.scale_block if weight_held == "block" else None
    if integer:
        w_min, w_max = _integer_range(element.element_bits, symmetric_restricted=symmetric)
        a_min, a_max = facet.clamp if facet.clamp else _integer_range(element.element_bits, symmetric_restricted=False)
    else:
        w_min = w_max = a_min = a_max = None
    recipe.weight = TensorSpec(
        dtype=element.name,
        granularity=_WEIGHT_GRANULARITY[weight_held],
        symmetric=symmetric,
        quant_min=w_min,
        quant_max=w_max,
        block=block,
    )
    recipe.why["weight.granularity"] = (
        f"the finest scale the readout holds is per {weight_held}"
        + (f" of {block}" if block else "")
        + "; a finer weight scale could only be applied on the host"
    )
    recipe.why["weight.range"] = (
        "symmetric weights use the restricted range so a negated weight is representable"
        if integer and symmetric
        else "the format's own range"
    )

    if weight_held == "block":
        activation_granularity, mode = "block", "dynamic"
        recipe.why["activation"] = "a block-scaled format scales activations per block at run time"
    else:
        activation_granularity = _ACTIVATION_GRANULARITY[activation_held]
        mode = "static"
        if prefer_activation_mode == "dynamic":
            if activation_granularity == "token":
                mode = "dynamic"
                recipe.why["activation.mode"] = (
                    "the workload asked for dynamic activations and the readout holds a per-row scale"
                )
            else:
                recipe.why["activation.mode"] = (
                    "the workload asked for dynamic activations; the readout holds one scale per "
                    "tensor, and computing it at run time is a host reduction over every "
                    "activation, so the recipe is static and the request is recorded"
                )
        else:
            recipe.why["activation.mode"] = "a scale the readout holds in a register is a compile-time constant"
    recipe.activation = TensorSpec(
        dtype=element.name,
        granularity=activation_granularity,
        symmetric=symmetric,
        quant_min=a_min,
        quant_max=a_max,
        block=block,
        mode=mode,
    )
    recipe.why["symmetric"] = (
        "the readout carries no zero point, so both scales are symmetric"
        if integer and symmetric
        else "the readout carries a zero point"
        if integer
        else "a floating format has no zero point"
    )

    _operand_sum(recipe, facet, activation_granularity=activation_granularity, mode=mode)
    if integer and activation_granularity == "tensor" and mode == "static":
        recipe.families = (*recipe.families, WINDOW_MEAN)
        recipe.why[WINDOW_MEAN] = (
            "a mean over a window is the unit's own contraction against a constant one, with the "
            "reciprocal of the count folded into the readout's one scale"
        )
    else:
        recipe.unquantized[WINDOW_MEAN] = (
            "a window mean folds 1/count into one compile-time readout scale, which needs integer "
            f"activations under one static scale per tensor; these are per {activation_granularity} ({mode})"
        )

    applied = {stage for readout in facet.readouts for stage in readout.get("applies") or ()}
    if applied:
        from merlin.runtime.commandbuffer import BIAS_STAGES

        recipe.bias_domain = "accumulator" if applied & set(BIAS_STAGES) else None
        recipe.why["bias_domain"] = (
            "the readout adds a bias in the accumulator's domain"
            if recipe.bias_domain
            else "no readout applies a bias, so a bias stays a host add"
        )
    _unit_families(recipe, capabilities)
    _readout_numerics(recipe, facet)
    return recipe


def _readout_numerics(recipe: QuantRecipe, facet: Any) -> None:
    """Capture semantics do not depend on a device's optional narrowing scale."""
    if recipe.status == DERIVED:
        recipe.readout_numerics = "integer_readout"
        recipe.why["readout_numerics"] = (
            "exact integer capture; full int32 device readout followed by exact rational host quantization"
        )


def _operand_sum(recipe: QuantRecipe, facet: Any, *, activation_granularity: str, mode: str) -> None:
    """Whether a sum of two tensors is quantized: only where a unit can compute it as integers.

    Quantizing an add the hardware cannot absorb buys nothing and costs accuracy, because the host
    would then dequantize, add and requantize where it used to add. So the family is listed only
    when the facet derived a scaled, accumulating load, and only under one static scale per
    tensor, which is what a load's single multiplier can carry.
    """
    licence = getattr(facet, "operand_sum", None)
    if not licence:
        recipe.unquantized[OPERAND_SUM] = (
            getattr(facet, "operand_sum_absent", None) or "the unit's load applies no scale"
        )
        return
    if activation_granularity != "tensor" or mode != "static":
        recipe.unquantized[OPERAND_SUM] = (
            f"a load carries one compile-time multiplier; activations here are scaled per "
            f"{activation_granularity} ({mode}), which it cannot hold"
        )
        return
    recipe.families = (*recipe.families, OPERAND_SUM)
    recipe.why[OPERAND_SUM] = (
        f"the unit's load multiplies each operand into the accumulator ({licence.get('operands')} operands, "
        f"rounded {licence.get('operand_rounding')}), so a sum of two tensors is integer work within "
        f"{facet.operand_sum_bound()} output step(s) of the single-rounding reference"
    )


def with_native_output_quantization(
    recipe: Mapping[str, Any], *, operators: Sequence[str] = ("linear",), reader_policy: str = "non_add_compute"
) -> dict[str, Any]:
    """Select new contraction output grids before pure internal compute readers.

    The result is a different numerical recipe: float consumers receive the
    dequantized native output grid. Graph outputs and mutable/unknown readers
    retain their existing cut. Selection uses source operation/dataflow only.
    """
    from merlin.targetgen import _integer_epilogue as epilogue

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
    from merlin.targetgen import _integer_epilogue as epilogue

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
        from merlin.targetgen import _integer_residual as residual

        if "operand_sum_grid" in recipe:
            raise ValueError("accumulator residual and shared operand sums cannot own the same graph")
        residual.check(policy)
    elif key == "integer_mean":
        from merlin.targetgen import _integer_pool as pool

        pool.check(policy)
    elif key == "projection_input_reuse":
        from merlin.targetgen import _projection_input_reuse as projection

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


def with_shared_grid_operand_sums(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Opt into one observed i8 grid per closed residual's inputs and output.

    The capture spells unit-grid add/ReLU/clamp, restoring real units only
    after the integer result. This is a new quantized recipe, not an accuracy
    or execution certificate. Existing independent placement and exact
    quantized-reference comparisons remain mandatory.
    """
    from merlin.targetgen import _integer_operand_sum as sums

    body = json.loads(json.dumps(dict(recipe)))
    if CONTRACTION not in (body.get("families") or ()):
        raise ValueError("shared-grid sums need a recipe that quantizes contractions")
    body["families"] = list(dict.fromkeys([*(body.get("families") or ()), OPERAND_SUM]))
    body[sums.KEY] = {"schema": sums.SCHEMA}
    sums.validate(body)
    body.setdefault("unquantized", {}).pop(OPERAND_SUM, None)
    body.setdefault("why", {})[OPERAND_SUM] = (
        "opted in: each closed residual's inputs and output share one observed activation grid; "
        "unit i8 load/readout multipliers preserve exact integer add, optional ReLU and saturation"
    )
    if "methodology" in body:
        extensions = body["methodology"].setdefault("extensions", [])
        if sums.SCHEMA not in extensions:
            extensions.append(sums.SCHEMA)
    body["recipe_sha256"] = digest(body)
    return body


def with_activation_contractions(recipe: Mapping[str, Any]) -> dict[str, Any]:
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
    body["recipe_sha256"] = digest(body)
    return body


#: The recipe block that quantizes contractions with RUN-TIME scales (``recipe["dynamic"]``).
DYNAMIC = "dynamic"
DYNAMIC_SCHEMA = "dynamic_contractions_v1"
#: Every scale sits on a NON-reduction axis: one per row of the streamed operand, one per column of the
#: other (a stored weight: one per output channel, fixed at capture). Each int8 product is then summed
#: exactly in the i32 accumulator and the scales apply afterwards as an outer product, on the host.
NON_REDUCTION = "non_reduction"
#: The static mode: one CALIBRATED scale per head of each operand (the readout's own form per device
#: call), a calibrated smooth shift, and fixed-scale planes of a softmax's ``exp(s - rowmax)``.
STATIC_HEADS = "static_heads"
#: Plane selectors: an operand produced by a row softmax, a stored weight, and (static per-head scales only)
#: every other operand, as k int8 digits of one static grid.
SOFTMAX_OUTPUT = "softmax_output"
WEIGHT = "weight"
OPERAND = "operand"
#: The operand a softmax output multiplies (attention's ``V``), as k successively finer static int8 grids.
#: ``P V`` is a convex combination of ``V``'s rows: when most of ``P`` sits on a row much smaller than the
#: head's largest (an attention sink), the result is small against ``V``'s grid step and needs more digits.
SOFTMAX_PARTNER = "softmax_partner"
#: All ordinary activation operands, on successively finer bounded residual grids.
RESIDUAL_OPERAND = "residual_operand"
#: The smooth shift: when a contraction reaches a softmax over its column axis only through scalar
#: factors and masks, the other operand's mean over that axis is subtracted first. That shifts each
#: score row by a constant, which the softmax ignores.
SOFTMAX_AXIS = "softmax_axis"
_MAX_PLANES = 4


def with_dynamic_contractions(
    recipe: Mapping[str, Any],
    *,
    families: Sequence[str] = (ACTIVATION_CONTRACTION,),
    planes: Mapping[str, int] | None = None,
    smooth: bool = False,
    static_families: Sequence[str] | None = None,
    scales: str = NON_REDUCTION,
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
    body.setdefault("why", {})[DYNAMIC] = (
        "opted in: run-time scales on non-reduction axes (rows of the streamed operand, columns of the "
        "other, a stored weight's output channels), applied by the host to the exact i32 accumulator "
        "as an outer product; the readout's one scale cannot hold them"
        if scales == NON_REDUCTION
        else "opted in: one calibrated scale per head of each operand, so every per-head device call reads "
        "one scale; softmax probabilities as fixed-scale planes of exp(s - rowmax), rows divided by their sum "
        "on the host"
    )
    body["recipe_sha256"] = digest(body)
    return body


def with_gptq_weights(recipe: Mapping[str, Any], **parameters: Any) -> dict[str, Any]:
    """``recipe`` (a derived ``quant_recipe_v1`` document) whose weight codes are chosen by GPTQ.

    The switch is ``weight.rounding = "gptq"`` with ``weight.gptq`` holding every parameter of
    :data:`merlin.targetgen._gptq.PARAMETERS`. ``parameters`` override
    :data:`~merlin.targetgen._gptq.DEFAULT_PARAMETERS`. The grid stays the recipe's own: the same
    dtype, range, symmetry and scale granularity, with the same observed scales. GPTQ only picks
    which integer each weight takes on that grid, using the calibration inputs of its layer (see
    ``_gptq.py``). Without the key a recipe rounds to nearest and its digest is what it always was.
    With it the digest changes, so a capture made under GPTQ codes is never served for one made
    under nearest codes.

    Refused with the reason, never approximated: a recipe that quantizes no contraction, dynamic
    activations (``quantize_`` rounds the weights itself), a weight format that is not a symmetric
    integer, and a weight scale granularity that the static path cannot express.
    """
    from merlin.targetgen import _gptq

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


#: The recipe block that gives selected Linears' and Conv2ds' activations a wider STATIC grid as int8 digit planes.
STATIC_PLANES = "static_planes"
STATIC_PLANES_SCHEMA = "static_planes_v1"
#: Observers a static recipe may name: TorchAO's two, or a percentile of |x| (``p99.999``).
_OBSERVERS = ("minmax", "histogram")


def _observer_valid(name: Any) -> bool:
    from merlin.targetgen._recipe_observers import percentile_of

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


#: The per-site bar of a ``dynamic`` block (``recipe["dynamic"]["site_agreement"]``): every rewritten
#: contraction's int8 result, held to the float contraction of its own operands on the recipe's
#: agreement samples, must reach ``cosine_min`` and stay within ``rel_rmse_max`` (a ratio of RMS values).
SITE_AGREEMENT = "site_agreement"
SITE_AGREEMENT_SCHEMA = "dynamic_site_agreement_v1"


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


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m merlin.targetgen.quant_recipe {activation-contractions|dynamic-contractions|static-planes|
    shared-grid-sums|native-outputs|site-agreement|activation-observer|gptq-weights} IN OUT``."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m merlin.targetgen.quant_recipe")
    sub = parser.add_subparsers(dest="command", required=True)
    variant = sub.add_parser("activation-contractions", help=with_activation_contractions.__doc__.splitlines()[0])
    variant.add_argument("recipe", type=Path)
    variant.add_argument("out", type=Path)
    residual = sub.add_parser("shared-grid-sums", help=with_shared_grid_operand_sums.__doc__.splitlines()[0])
    residual.add_argument("recipe", type=Path)
    residual.add_argument("out", type=Path)
    outputs = sub.add_parser("native-outputs", help=with_native_output_quantization.__doc__.splitlines()[0])
    outputs.add_argument("recipe", type=Path)
    outputs.add_argument("out", type=Path)
    outputs.add_argument("--operators", nargs="+", default=["linear"], choices=("linear", "conv2d"))
    outputs.add_argument("--reader-policy", default="non_add_compute", choices=("compute", "non_add_compute"))
    dynamic = sub.add_parser("dynamic-contractions", help=with_dynamic_contractions.__doc__.splitlines()[0])
    dynamic.add_argument("recipe", type=Path)
    dynamic.add_argument("out", type=Path)
    dynamic.add_argument("--families", default=ACTIVATION_CONTRACTION, help="comma-separated contraction families")
    dynamic.add_argument("--planes", default="", help=f"selector=k pairs, e.g. {SOFTMAX_OUTPUT}=2")
    dynamic.add_argument("--smooth", action="store_true", help="subtract the mean over a softmax's axis")
    dynamic.add_argument("--static-families", default=None, help="families kept static (default: the recipe's)")
    dynamic.add_argument("--scales", default=NON_REDUCTION, choices=(NON_REDUCTION, STATIC_HEADS))
    wide = sub.add_parser("static-planes", help=with_static_planes.__doc__.splitlines()[0])
    wide.add_argument("recipe", type=Path)
    wide.add_argument("out", type=Path)
    wide.add_argument("--planes", type=int, default=2)
    wide.add_argument("--observer", default="minmax")
    wide.add_argument("--owners", default=None, help="comma-separated module-path prefixes (default: all)")
    observe = sub.add_parser("activation-observer", help=with_activation_observer.__doc__.splitlines()[0])
    observe.add_argument("recipe", type=Path)
    observe.add_argument("out", type=Path)
    observe.add_argument("observer")
    bar = sub.add_parser("site-agreement", help=with_site_agreement.__doc__.splitlines()[0])
    bar.add_argument("recipe", type=Path)
    bar.add_argument("out", type=Path)
    bar.add_argument("--cosine-min", type=float, required=True, dest="cosine_min")
    bar.add_argument("--rel-rmse-max", type=float, required=True, dest="rel_rmse_max")
    bar.add_argument("--basis", required=True, help="where the bar came from (recorded in the recipe's why)")
    gptq = sub.add_parser("gptq-weights", help=with_gptq_weights.__doc__.splitlines()[0])
    gptq.add_argument("recipe", type=Path)
    gptq.add_argument("out", type=Path)
    gptq.add_argument("--calibration-samples", type=int, dest="calibration_samples")
    gptq.add_argument("--act-order", action=argparse.BooleanOptionalAction, dest="act_order")
    gptq.add_argument("--damp", type=float)
    gptq.add_argument("--block", type=int)
    args = parser.parse_args(argv)
    source = json.loads(args.recipe.read_text(encoding="utf-8"))
    if args.command == "shared-grid-sums":
        body = with_shared_grid_operand_sums(source)
    elif args.command == "native-outputs":
        body = with_native_output_quantization(source, operators=args.operators, reader_policy=args.reader_policy)
    elif args.command == "activation-contractions":
        body = with_activation_contractions(source)
    elif args.command == "static-planes":
        owners = None if args.owners is None else [o for o in args.owners.split(",") if o]
        body = with_static_planes(source, planes=args.planes, observer=args.observer, owners=owners)
    elif args.command == "gptq-weights":
        chosen = {name: getattr(args, name) for name in ("calibration_samples", "act_order", "damp", "block")}
        body = with_gptq_weights(source, **{name: value for name, value in chosen.items() if value is not None})
    elif args.command == "site-agreement":
        body = with_site_agreement(source, cosine_min=args.cosine_min, rel_rmse_max=args.rel_rmse_max, basis=args.basis)
    elif args.command == "activation-observer":
        body = with_activation_observer(source, args.observer)
    else:
        planes = {k: int(v) for k, _, v in (p.partition("=") for p in args.planes.split(",") if p)}
        static = None if args.static_families is None else [f for f in args.static_families.split(",") if f]
        body = with_dynamic_contractions(
            source,
            families=[f for f in args.families.split(",") if f],
            planes=planes,
            smooth=args.smooth,
            static_families=static,
            scales=args.scales,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.out}: recipe_sha256 {body['recipe_sha256']}")
    return 0


def for_target(
    target: str, *, facts: Mapping[str, Any] | None = None, prefer_activation_mode: str | None = None
) -> list[QuantRecipe]:
    """One recipe per compute unit that contracts, in the contract's unit order."""
    from merlin.targetgen import readout_facet

    return [
        derive(facet, prefer_activation_mode=prefer_activation_mode)
        for facet in readout_facet.for_target(target, facts=facts)
    ]


def derive_candidates(
    contract: Mapping[str, Any],
    facets: Sequence[Any],
    *,
    prefer_activation_mode: str | None = None,
    integer_sum_policy: str | None = None,
) -> list[QuantCandidate]:
    """Inventory every declared unit format without promoting a declaration to a proven recipe.

    The existing readout derivation produces one facet per unit, not one per format. A candidate
    therefore derives only when that facet's observed element format matches this candidate and
    its format-dependent evidence agrees. Other declared formats are present, but explicitly
    underivable until format-specific readout evidence exists. No candidate is selected for a
    whole-model capture here; :func:`select` retains its existing behavior.
    """
    from merlin.targetgen.compute_units import compute_units

    by_unit: dict[str, list[Any]] = {}
    for facet in facets:
        if facet.unit is not None:
            by_unit.setdefault(str(facet.unit), []).append(facet)

    candidates: list[QuantCandidate] = []
    for unit in compute_units(dict(contract)):
        for name in dict.fromkeys(unit.dtypes):
            fmt = qf.get(name)
            matches = by_unit.get(unit.name, [])
            facet = matches[0] if len(matches) == 1 else None
            target = (
                str(facet.target)
                if facet is not None
                else (str(facets[0].target) if facets else str(contract.get("target") or ""))
            )
            evidence: list[dict[str, str]] = [
                {
                    "field": "operand_format",
                    "rung": "contract",
                    "observed": f"compute unit {unit.name!r} declares {fmt.name!r}",
                }
            ]
            if facet is not None:
                evidence.extend(
                    {"field": item.field, "rung": item.rung, "observed": item.observed} for item in facet.evidence
                )
            reasons: dict[str, str] = {}
            if not matches:
                reasons["readout"] = f"no readout facet was derived for compute unit {unit.name!r}"
            elif len(matches) != 1:
                reasons["readout"] = f"{len(matches)} readout facets exist for compute unit {unit.name!r}"

            # A quantization recipe is about a contraction. A unit that declares only movement or
            # standalone elementwise work cannot license one merely by declaring a storage dtype.
            if not any(
                cap.family == CONTRACTION
                and not cap.composed_with
                and (not cap.dtypes or any(_format_of(dtype) == fmt for dtype in cap.dtypes))
                for cap in unit.semantic_capabilities
            ):
                reasons["contraction"] = (
                    f"the compute unit declares no standalone contraction capability for {fmt.name!r}"
                )

            if unit.accumulate:
                same_format_rules = [
                    rule for rule in unit.accumulate if _format_of(rule.inp) == fmt and _format_of(rule.weight) == fmt
                ]
                if not same_format_rules:
                    reasons["accumulate"] = (
                        f"the declared accumulate matrix has no {fmt.name} x {fmt.name} pairing; "
                        "mixed-operand quantization is not derivable by this recipe"
                    )
                elif (
                    facet is not None
                    and facet.accumulator_dtype is not None
                    and not any(rule.acc in (None, facet.accumulator_dtype) for rule in same_format_rules)
                ):
                    reasons["accumulate"] = (
                        f"the {fmt.name} x {fmt.name} pairing does not accumulate to the readout's "
                        f"observed {facet.accumulator_dtype!r}"
                    )

            if facet is not None:
                observed = _format_of(facet.element_dtype)
                if observed != fmt:
                    reasons["element_dtype"] = (
                        f"readout element {facet.element_dtype!r} is not declared format {fmt.name!r}; "
                        "a format-specific datapath/readout fact is required"
                    )
                abi = facet.scalar_abi or {}
                abi_output = _format_of(abi.get("output_dtype"))
                if abi.get("output_dtype") and abi_output != fmt:
                    reasons["scalar_abi"] = (
                        f"scalar readout ABI output {abi['output_dtype']!r} does not prove {fmt.name!r}"
                    )
                format_rung = "format" in facet.rungs_for("scale_granularities")
                if format_rung:
                    if not fmt.scale.is_block:
                        reasons["scale_granularities"] = (
                            "the facet's block-scale granularity was derived from a different "
                            "declared format, not this format's readout"
                        )
                    elif (facet.scale_block, facet.scale_dtype) != (fmt.scale.block, fmt.scale.dtype):
                        reasons["scale_granularities"] = (
                            f"the facet's block scale {(facet.scale_block, facet.scale_dtype)!r} "
                            f"does not match {fmt.name!r} scale {(fmt.scale.block, fmt.scale.dtype)!r}"
                        )
                if facet.operand_sum and _format_of(facet.operand_sum.get("operand_dtype")) != fmt:
                    reasons["operand_sum"] = (
                        "the load-scale operand format does not match this candidate; its "
                        "integer-sum family cannot be reused"
                    )

            recipe = None
            if not reasons and facet is not None:
                derived = derive(
                    facet,
                    prefer_activation_mode=prefer_activation_mode,
                    capabilities=unit.semantic_capabilities,
                    integer_sum_policy=integer_sum_policy,
                )
                if derived.status == DERIVED:
                    recipe = derived
                else:
                    reasons.update(derived.underivable)
            candidates.append(
                QuantCandidate(
                    target=target,
                    unit=unit.name,
                    format=fmt.name,
                    recipe=recipe,
                    evidence=tuple(evidence),
                    underivable=reasons,
                )
            )
    return candidates


def candidates_for_target(
    target: str,
    *,
    contract: Mapping[str, Any] | None = None,
    facts: Mapping[str, Any] | None = None,
    prefer_activation_mode: str | None = None,
) -> list[QuantCandidate]:
    """Target-facing candidate inventory; it does not change capture recipe selection."""
    from merlin.targetgen import readout_facet

    if contract is None:
        from merlin.targetgen.target_experiment import load_capability_manifest

        contract = load_capability_manifest(target).contract
    facets = readout_facet.for_target(target, contract=contract, facts=facts)
    return derive_candidates(contract, facets, prefer_activation_mode=prefer_activation_mode)


def select(recipes: Sequence[QuantRecipe]) -> QuantRecipe | None:
    """The recipe a whole-model capture uses: the first derived one, else the first with its reasons."""
    return next((r for r in recipes if r.status == DERIVED), recipes[0] if recipes else None)


if __name__ == "__main__":
    raise SystemExit(_main())
