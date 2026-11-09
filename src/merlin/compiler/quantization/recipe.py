"""Framework-independent recipe records, serialized vocabulary and content identity.

These describe numerical preparation intent, not observed framework realization,
placement, application quality or target execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
