"""Validate target-selected oracle identities in frozen claim declarations.

The authored template names tiers, while Phase 0 freezes concrete simulator and
oracle-kind names.  An analyzer reconstructs the exact reviewed declaration
from those names; it never treats the declaration itself as its own template.
"""

from __future__ import annotations

import copy
import string
from collections.abc import Mapping
from typing import Any

_ENGINE_CHARS = frozenset(string.ascii_letters + string.digits + "_")
_ENGINE_HEAD = frozenset(string.ascii_letters)
_PLACEHOLDERS = {
    "correctness_simulator": "$target_oracle:L2",
    "timing_simulator": "$target_oracle:L3",
    "timing_oracle_kind": "$target_oracle_kind:L3",
}


def require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be an exact lowercase SHA-256")
    return value


def _engine(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value[0] not in _ENGINE_HEAD
        or not set(value) <= _ENGINE_CHARS
        or value in {"elaborated_rtl", "unknown"}
    ):
        raise ValueError(f"{field} must name a concrete simulator")
    return value


def selected_acceptance(template: Mapping[str, Any], declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct a reviewed template using only concrete frozen oracle identities.

    Exact equality with this result remains the caller's job.  In particular,
    thresholds and result-evidence rules are never read back from the candidate
    declaration to build their expected values.
    """
    evidence = declaration.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("acceptance evidence must be a mapping")
    correctness = _engine(evidence.get("correctness_simulator"), "L2 correctness simulator")
    timing = _engine(evidence.get("timing_simulator"), "L3 timing simulator")
    if correctness == timing:
        raise ValueError("L2 and L3 simulators must have distinct measurement identities")
    kind = evidence.get("timing_oracle_kind")
    if kind != f"rtl_{timing}":
        raise ValueError("L3 oracle kind must identify the selected concrete RTL simulator")
    fit = declaration.get("fit")
    if not isinstance(fit, Mapping) or fit.get("dependent_metric") != f"{timing}_L3_cycles":
        raise ValueError("L3 dependent metric must match the selected timing simulator")
    expected = copy.deepcopy(dict(template))
    selected = expected["evidence"]
    selected["correctness_simulator"] = correctness
    selected["timing_simulator"] = timing
    selected["timing_oracle_kind"] = kind
    selected["resolved_from"] = dict(_PLACEHOLDERS)
    expected["fit"]["dependent_metric"] = f"{timing}_L3_cycles"
    return expected
