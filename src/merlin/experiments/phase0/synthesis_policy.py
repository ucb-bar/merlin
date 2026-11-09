"""Apply explicitly authored model scheduling gates to freshly synthesized entries.

This does not change numerical acceptance, coverage requirements or retained inputs.
Policy is read only from the selected public recipe, never from generated sidecars.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import yaml


def apply_model_gates(result: dict, recipe: Path) -> dict:
    """Return synthesis with validated per-model gates and exact recipe attribution.

    An absent policy preserves the original result. Every declared name must select
    exactly one synthesized model; stale selectors and unsupported fields fail closed.
    The recipe must state a reason for each scheduling decision.
    """
    raw = recipe.read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, dict):
        raise ValueError("synthesis recipe must be a mapping")
    policy = document.get("synthesis_model_gates", {})
    if not isinstance(policy, dict):
        raise ValueError("synthesis_model_gates must be a mapping")
    if not policy:
        return result
    updated = deepcopy(result)
    applied = []
    for name, setting in policy.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("synthesis model gate names must be non-empty strings")
        if not isinstance(setting, dict) or set(setting) != {"after_op_pass_fraction", "reason"}:
            raise ValueError(f"{name}: model gate requires only after_op_pass_fraction and reason")
        fraction = setting["after_op_pass_fraction"]
        if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
            raise ValueError(f"{name}: after_op_pass_fraction must be a finite number in [0, 1]")
        reason = setting["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{name}: model gate reason must be non-empty")
        matches = [entry for entry in updated["capsules"] if entry.get("name") == name]
        if len(matches) != 1 or matches[0].get("kind") != "model":
            raise ValueError(f"{name}: model gate must select exactly one synthesized model")
        entry = matches[0]
        gate = entry.setdefault("gate", {})
        if not isinstance(gate, dict):
            raise ValueError(f"{name}: synthesized gate must be a mapping")
        previous = deepcopy(gate)
        gate["after_op_pass_fraction"] = fraction
        applied.append({"name": name, "previous_gate": previous, "gate": deepcopy(gate), "reason": reason})
    updated.setdefault("provenance", {})["authored_model_gates"] = {
        "recipe_sha256": sha256(raw).hexdigest(),
        "entries": applied,
    }
    return updated
