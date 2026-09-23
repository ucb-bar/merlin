"""Which of a model's layers the target's own numeric form covers, and why each other one does not.

A recipe (:mod:`merlin.targetgen.quant_recipe`) says what numeric form a target's datapath
implements. It says it once, for the target. A model is not one layer, and the answer is not the
same for all of them: a recipe that holds one scale per output channel still cannot place a layer
whose weight has no output-channel axis, and a block scale of 32 cannot tile a layer with 24 inputs.
Those layers are host work. They were host work before too — the difference is that nobody said so.

So this is the per-layer half of the derivation: recipe in, one decision per layer out, each
carrying the reason it got that decision. The reasons are separated by WHOSE limit they are, because
the two have different fixes and conflating them is how a framework's gap gets filed as a hardware
gap:

``family_not_absorbed``
    The target's readout cannot absorb this family of operation at all. The recipe already said so
    and this repeats its own words. Fixing it means different hardware.
``granularity_does_not_tile``
    The target's scale granularity cannot divide THIS layer's weight. The hardware absorbs the
    family; this layer's shape is the obstacle.
``no_weight_to_scale`` / ``unmapped_module``
    The layer has nothing the recipe describes, or is not a kind any family names.
``no_quantization_site``
    The layer stores no operand and names no family, so it carries no quantization of its OWN. It
    is not therefore host arithmetic: an activation is absorbed into a neighbouring contraction's
    readout wherever that readout applies it, which is a question about the readout, not about this
    layer. Reporting such a module as a host refusal is how a census comes to claim that a model's
    activations are host work when the hardware applies them on the way out of the accumulator.

Nothing here imports a framework, imports merlin, or names a target. It is pure data in and data
out, which is also what lets it be read on both sides of the capture boundary: the capture venv is
the only interpreter with torch, and it imports this module as a sibling by bare name, the same way
it imports the quantizer that consumes the plan.

A layer the plan cannot decide is never quantized by default. "Unknown" and "host" are recorded as
different things: the first says the derivation ran out, the second says it reached a refusal.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "quant_layer_plan_v1"

#: A layer runs in the target's numeric form, or on the host.
DEVICE = "device"
HOST = "host"

#: The recipe families, spelled as :mod:`merlin.targetgen.quant_recipe` spells them. Repeated rather
#: than imported: this module is read from the capture venv, which has no merlin on its path.
CONTRACTION = "contraction"
OPERAND_SUM = "operand_sum"
WINDOW_MEAN = "window_mean"

#: Framework module kind -> the recipe family its operation belongs to. This is a FRAMEWORK
#: vocabulary (the names torch gives its module classes), not a fact about any target: the target's
#: side of the decision is the recipe, and it is consulted below. A kind absent here is not guessed.
MODULE_FAMILY: dict[str, str] = {
    "Linear": CONTRACTION,
    "LazyLinear": CONTRACTION,
    "Conv1d": CONTRACTION,
    "Conv2d": CONTRACTION,
    "Conv3d": CONTRACTION,
    "ConvTranspose1d": CONTRACTION,
    "ConvTranspose2d": CONTRACTION,
    "ConvTranspose3d": CONTRACTION,
    "Embedding": CONTRACTION,
    "AvgPool1d": WINDOW_MEAN,
    "AvgPool2d": WINDOW_MEAN,
    "AvgPool3d": WINDOW_MEAN,
    "AdaptiveAvgPool1d": WINDOW_MEAN,
    "AdaptiveAvgPool2d": WINDOW_MEAN,
    "AdaptiveAvgPool3d": WINDOW_MEAN,
}

#: Refusal codes. A consumer groups by these; the prose beside them is for a person.
FAMILY_NOT_ABSORBED = "family_not_absorbed"
GRANULARITY_DOES_NOT_TILE = "granularity_does_not_tile"
NO_WEIGHT_TO_SCALE = "no_weight_to_scale"
UNMAPPED_MODULE = "unmapped_module"
NO_QUANTIZATION_SITE = "no_quantization_site"
RECIPE_UNDERIVABLE = "recipe_underivable"


@dataclass(frozen=True)
class LayerDecision:
    """Where one layer's arithmetic runs, and the reason it runs there."""

    fqn: str
    kind: str
    placement: str  # DEVICE | HOST
    family: str | None = None
    refusal: str | None = None  # a refusal code, when HOST
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fqn": self.fqn,
            "kind": self.kind,
            "placement": self.placement,
            "family": self.family,
            "refusal": self.refusal,
            "why": self.why,
        }


@dataclass
class LayerPlan:
    """Every layer's decision under one recipe, with the recipe's identity carried along."""

    target: str
    recipe_sha256: str | None
    recipe_status: str
    decisions: tuple[LayerDecision, ...] = ()
    unknown: dict[str, str] = field(default_factory=dict)

    @property
    def on_device(self) -> tuple[LayerDecision, ...]:
        return tuple(d for d in self.decisions if d.placement == DEVICE)

    @property
    def on_host(self) -> tuple[LayerDecision, ...]:
        return tuple(d for d in self.decisions if d.placement == HOST)

    def refusals(self) -> dict[str, int]:
        """How many layers each refusal code accounts for — the census this exists to make sayable."""
        counts: dict[str, int] = {}
        for decision in self.on_host:
            key = decision.refusal or "unrecorded"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "target": self.target,
            "recipe_sha256": self.recipe_sha256,
            "recipe_status": self.recipe_status,
            "layers": [d.to_dict() for d in self.decisions],
            "refusals": self.refusals(),
            "unknown": dict(sorted(self.unknown.items())),
        }
        body["plan_sha256"] = digest(body)
        return body


def digest(body: Mapping[str, Any]) -> str:
    """Identity of a plan's CONTENT: placements and refusal codes, not the prose beside them."""
    payload = {
        "schema": body.get("schema"),
        "recipe_sha256": body.get("recipe_sha256"),
        "recipe_status": body.get("recipe_status"),
        "layers": [
            {"fqn": layer.get("fqn"), "placement": layer.get("placement"), "refusal": layer.get("refusal")}
            for layer in body.get("layers") or ()
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _shape(layer: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw = layer.get("weight_shape")
    if raw is None:
        return None
    try:
        return tuple(int(d) for d in raw)
    except (TypeError, ValueError):
        return None


def _tiles(granularity: str, block: int | None, shape: tuple[int, ...] | None) -> tuple[bool, str]:
    """Whether the recipe's weight granularity divides THIS layer's weight, and the reason.

    A granularity is a promise about how many scales the readout holds per weight. Whether that
    promise can be kept for one layer is a property of the layer's own shape, so it is read off the
    shape and never assumed:

    * ``tensor`` — one scale for the whole weight. Any shape takes it.
    * ``channel`` — one scale per output channel, which is the weight's leading axis. A weight with
      no axes has no such axis.
    * ``block`` — one scale per fixed-length run along the REDUCED extent. For a weight of shape
      ``[out, *rest]`` the contraction reduces over all of ``rest``, so the extent is their product:
      ``[16, 24]`` reduces over 24, and a convolution's ``[8, 3, 3, 3]`` over 27, not over the 3 of
      its last axis. That extent must be a whole number of blocks; a partial block would be a scale
      covering elements that are not there.
    """
    if shape is None:
        return False, "the layer declares no weight shape, so no granularity can be checked against it"
    if granularity == "tensor":
        return True, "one scale covers the whole weight, which any shape admits"
    if granularity == "channel":
        if not shape:
            return False, "a per-channel scale indexes the weight's leading axis, and this weight has no axes"
        return True, f"one scale per output channel over the weight's leading axis ({shape[0]})"
    if granularity == "block":
        if not block or block <= 0:
            return False, "the recipe asks for a block scale but states no block length"
        if len(shape) < 2:
            return False, (
                f"a block scale runs along the extent a contraction reduces, and a weight of shape "
                f"{list(shape)} has no axis beyond its output one"
            )
        reduced = 1
        for extent in shape[1:]:
            reduced *= extent
        if reduced % block:
            return False, (
                f"a block scale covers {block} elements of the reduced extent, which is {reduced} "
                f"here ({'x'.join(str(d) for d in shape[1:])}) and is not a whole number of blocks "
                f"({reduced} % {block} = {reduced % block})"
            )
        return True, f"the reduced extent ({reduced}) is {reduced // block} whole block(s) of {block}"
    return False, f"granularity {granularity!r} is not one this plan knows how to check against a shape"


def plan(recipe: Mapping[str, Any], layers: Sequence[Mapping[str, Any]]) -> LayerPlan:
    """One decision per layer under ``recipe``.

    ``layers`` are plain mappings — ``fqn``, ``kind`` (the framework's module class name), an
    optional ``family`` where the caller already knows it, and an optional ``weight_shape``. They
    carry no framework objects, so this runs on either side of the capture boundary.

    An underivable recipe places every layer on the host, each saying which field of the target's
    readout was not derived. That is the fail-closed direction: a recipe that could not be derived
    must not quantize anything, and it must not be silent about why.
    """
    target = str(recipe.get("target") or "")
    status = str(recipe.get("status") or "")
    out = LayerPlan(target=target, recipe_sha256=recipe.get("recipe_sha256"), recipe_status=status)

    if status != "derived":
        reasons = recipe.get("underivable") or {}
        spelled = (
            "; ".join(f"{field_name}: {why}" for field_name, why in sorted(reasons.items())) or "no reason recorded"
        )
        out.unknown["recipe"] = spelled
        out.decisions = tuple(
            LayerDecision(
                fqn=str(layer.get("fqn") or ""),
                kind=str(layer.get("kind") or ""),
                placement=HOST,
                family=_family_of(layer),
                refusal=RECIPE_UNDERIVABLE,
                why=f"the target's readout licenses no recipe ({spelled})",
            )
            for layer in layers
        )
        return out

    families = tuple(recipe.get("families") or ())
    unquantized = recipe.get("unquantized") or {}
    weight = recipe.get("weight") or {}
    granularity = str(weight.get("granularity") or "")
    block = weight.get("block")

    decisions: list[LayerDecision] = []
    for layer in layers:
        fqn, kind = str(layer.get("fqn") or ""), str(layer.get("kind") or "")
        family = _family_of(layer)
        if family is None:
            # A module that stores no operand quantizes nothing of its own, whatever its kind; one
            # that stores an operand and names no family is a gap in the family vocabulary. The two
            # are different findings and only the second is something to go and fix.
            stores = _shape(layer) is not None
            decisions.append(
                LayerDecision(
                    fqn=fqn,
                    kind=kind,
                    placement=HOST,
                    family=None,
                    refusal=UNMAPPED_MODULE if stores else NO_QUANTIZATION_SITE,
                    why=(
                        f"module kind {kind!r} stores an operand but is not one any recipe family "
                        f"names, so no recipe describes what to do with it"
                        if stores
                        else f"module kind {kind!r} stores no operand and names no family, so it is "
                        f"no quantization site of its own; whether it is absorbed into a "
                        f"neighbouring readout is a question about that readout"
                    ),
                )
            )
            continue
        if family not in families:
            decisions.append(
                LayerDecision(
                    fqn=fqn,
                    kind=kind,
                    placement=HOST,
                    family=family,
                    refusal=FAMILY_NOT_ABSORBED,
                    why=str(
                        unquantized.get(family)
                        or f"the recipe quantizes {list(families)}, which does not include {family!r}"
                    ),
                )
            )
            continue
        shape = _shape(layer)
        if family == CONTRACTION and shape is None:
            decisions.append(
                LayerDecision(
                    fqn=fqn,
                    kind=kind,
                    placement=HOST,
                    family=family,
                    refusal=NO_WEIGHT_TO_SCALE,
                    why="a contraction is quantized by scaling its stored operand, and this layer declares none",
                )
            )
            continue
        if family == CONTRACTION:
            fits, why = _tiles(granularity, block, shape)
            if not fits:
                decisions.append(
                    LayerDecision(
                        fqn=fqn,
                        kind=kind,
                        placement=HOST,
                        family=family,
                        refusal=GRANULARITY_DOES_NOT_TILE,
                        why=f"the readout holds one scale per {granularity}, and {why}",
                    )
                )
                continue
        else:
            why = f"the recipe lists {family!r}, which rides on a readout this target holds"
        decisions.append(LayerDecision(fqn=fqn, kind=kind, placement=DEVICE, family=family, why=why))
    out.decisions = tuple(decisions)
    return out


def _family_of(layer: Mapping[str, Any]) -> str | None:
    declared = layer.get("family")
    if declared:
        return str(declared)
    return MODULE_FAMILY.get(str(layer.get("kind") or ""))
