"""Derive numerical recipe candidates from selected readout and format facts.

This is the historical public recipe namespace as well as the actual derivation
owner. Reexports preserve object identity; construction, realization, placement
and execution qualification retain distinct contracts.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

from merlin.common import quant_formats as qf

from .dynamic import (
    _MAX_PLANES as _MAX_PLANES,
)
from .dynamic import (
    site_agreement_of as site_agreement_of,
)
from .dynamic import (
    with_dynamic_contractions as with_dynamic_contractions,
)
from .dynamic import (
    with_site_agreement as with_site_agreement,
)
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
    QuantCandidate as QuantCandidate,
)
from .recipe import (
    QuantRecipe as QuantRecipe,
)
from .recipe import (
    TensorSpec as TensorSpec,
)
from .recipe import (
    digest as digest,
)
from .static import (
    _OBSERVERS as _OBSERVERS,
)
from .static import (
    _observer_valid as _observer_valid,
)
from .static import (
    _static_int8 as _static_int8,
)
from .static import (
    static_planes_block as static_planes_block,
)
from .static import (
    with_activation_contractions as with_activation_contractions,
)
from .static import (
    with_activation_observer as with_activation_observer,
)
from .static import (
    with_gptq_weights as with_gptq_weights,
)
from .static import (
    with_integer_extension as with_integer_extension,
)
from .static import (
    with_native_output_quantization as with_native_output_quantization,
)
from .static import (
    with_shared_grid_operand_sums as with_shared_grid_operand_sums,
)
from .static import (
    with_static_planes as with_static_planes,
)

_WEIGHT_GRANULARITY = {"column": "channel", "row": "channel", "rank1": "channel", "tensor": "tensor", "block": "block"}

_WEIGHT_PREFERENCE = ("block", "rank1", "column", "row", "tensor")

_ACTIVATION_PREFERENCE = ("rank1", "row", "tensor")

_ACTIVATION_GRANULARITY = {"rank1": "token", "row": "token", "tensor": "tensor"}


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


def derive_candidates(
    contract: Mapping[str, Any],
    facets: Sequence[Any],
    *,
    prefer_activation_mode: str | None = None,
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
                derived = derive(facet, prefer_activation_mode=prefer_activation_mode)
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


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m merlin.compiler.quantization.capabilities {activation-contractions|dynamic-contractions|static-planes|
    shared-grid-sums|native-outputs|site-agreement|activation-observer|gptq-weights} IN OUT``."""
    from .cli import main

    return main(argv, api=sys.modules[__name__])
