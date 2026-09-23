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


def derive(facet: Any, *, prefer_activation_mode: str | None = None) -> QuantRecipe:
    """The recipe a unit's readout facet licenses.

    ``prefer_activation_mode`` is a workload's request ("dynamic" for a model whose activation
    range moves with its input). It is honoured where the facet allows and recorded where it does
    not: a recipe never promises a mode the hardware would push onto the host silently.
    """
    recipe = QuantRecipe(target=facet.target, unit=facet.unit)

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
    return recipe


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


def for_target(
    target: str, *, facts: Mapping[str, Any] | None = None, prefer_activation_mode: str | None = None
) -> list[QuantRecipe]:
    """One recipe per compute unit that contracts, in the contract's unit order."""
    from merlin.targetgen import readout_facet

    return [
        derive(facet, prefer_activation_mode=prefer_activation_mode)
        for facet in readout_facet.for_target(target, facts=facts)
    ]


def select(recipes: Sequence[QuantRecipe]) -> QuantRecipe | None:
    """The recipe a whole-model capture uses: the first derived one, else the first with its reasons."""
    return next((r for r in recipes if r.status == DERIVED), recipes[0] if recipes else None)
