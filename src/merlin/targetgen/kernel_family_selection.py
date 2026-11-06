"""Declarative kernel-family selection for authoring catalogs.

The kernel library is evidence for reusable schedules, not an implementation dependency.  This
module consequently reads only a selection contract and chooses a *strategy family* from a
semantic operation, its dtype/shape, and caller-derived hardware capabilities.  It never imports,
copies, links, or calls the reference kernels. The retained provider snapshot remains frozen; this
module owns only reusable declarative matching, including stricter shape-contract validation.

Selection is deliberately fail closed.  A missing dimension or hardware fact is a refusal, and
families marked experimental by the contract can never be selected.  The returned audit contains a
decision for every declared family so whole-corpus qualification can count coverage rather than
mistaking a silent omission for support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class KernelSelectionContractError(ValueError):
    """The declarative family registry is incomplete or malformed."""


@dataclass(frozen=True)
class KernelRequest:
    """Target-independent facts about one candidate region."""

    op: str
    dtype: str
    shape: Mapping[str, int]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KernelRequest":
        op = str(value.get("op") or "").strip()
        dtype = str(value.get("dtype") or "").strip()
        raw_shape = value.get("shape")
        if not op:
            raise KernelSelectionContractError("kernel request has no semantic op")
        if not dtype:
            raise KernelSelectionContractError("kernel request has no dtype")
        if not isinstance(raw_shape, Mapping):
            raise KernelSelectionContractError("kernel request shape is not a mapping")
        shape: dict[str, int] = {}
        for name, extent in raw_shape.items():
            if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
                raise KernelSelectionContractError(f"kernel request dimension {name!r} must be a positive integer")
            shape[str(name)] = extent
        return cls(op=op, dtype=dtype, shape=shape)


@dataclass(frozen=True)
class HardwareCapabilities:
    """Hardware facts already derived by the target introspector/contract loader."""

    features: frozenset[str]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HardwareCapabilities":
        raw = value.get("features")
        if not isinstance(raw, (list, tuple, set, frozenset)):
            raise KernelSelectionContractError("hardware capabilities need a features sequence")
        features = frozenset(str(item).strip() for item in raw if str(item).strip())
        return cls(features=features)


@dataclass(frozen=True)
class FamilyDecision:
    family: str
    status: str
    priority: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "status": self.status,
            "priority": self.priority,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SelectionReport:
    request: KernelRequest
    hardware_features: tuple[str, ...]
    selected_family: str | None
    decisions: tuple[FamilyDecision, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "semantic_kernel_selection_report_v1",
            "request": {
                "op": self.request.op,
                "dtype": self.request.dtype,
                "shape": dict(sorted(self.request.shape.items())),
            },
            "derived_hardware_features": list(self.hardware_features),
            "selected_family": self.selected_family,
            "selection_is_numeric_qualification": False,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def load_selection_contract(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise KernelSelectionContractError("kernel selection contract is not a mapping")
    return data


def _family_sets(contract: Mapping[str, Any]) -> tuple[set[str], dict[str, str]]:
    qualified = {str(item) for item in contract.get("qualified_families", ())}
    experimental: dict[str, str] = {}
    for item in contract.get("experimental_families", ()):
        if not isinstance(item, Mapping) or not item.get("path") or not item.get("reason"):
            raise KernelSelectionContractError("experimental family needs path and reason")
        experimental[str(item["path"])] = str(item["reason"])
    overlap = qualified & experimental.keys()
    if overlap:
        raise KernelSelectionContractError(f"families cannot be both qualified and experimental: {sorted(overlap)}")
    return qualified, experimental


def _rules(contract: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    block = contract.get("compiler_selection")
    if not isinstance(block, Mapping) or not isinstance(block.get("rules"), list):
        raise KernelSelectionContractError("contract has no compiler_selection.rules list")
    rules = block["rules"]
    qualified, experimental = _family_sets(contract)
    known = qualified | experimental.keys()
    names: list[str] = []
    for rule in rules:
        if not isinstance(rule, Mapping) or not rule.get("family"):
            raise KernelSelectionContractError("every compiler selection rule needs a family")
        family = str(rule["family"])
        names.append(family)
        if family not in known:
            raise KernelSelectionContractError(f"compiler selection rule names undeclared family {family!r}")
        if not isinstance(rule.get("semantic_ops"), list) or not rule["semantic_ops"]:
            raise KernelSelectionContractError(f"{family}: semantic_ops must be a non-empty list")
        if not isinstance(rule.get("dtypes"), list) or not rule["dtypes"]:
            raise KernelSelectionContractError(f"{family}: dtypes must be a non-empty list")
        if not isinstance(rule.get("requires_hardware", []), list):
            raise KernelSelectionContractError(f"{family}: requires_hardware must be a list")
        priority = rule.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise KernelSelectionContractError(f"{family}: priority must be an integer")
        raw_shape = rule.get("shape", {})
        if not isinstance(raw_shape, Mapping):
            raise KernelSelectionContractError(f"{family}: shape must be a mapping")
        # Validate constraint syntax independently of which dimensions a request supplies.
        shape_constraint_reasons({}, raw_shape)
    duplicate = sorted({name for name in names if names.count(name) > 1})
    if duplicate:
        raise KernelSelectionContractError(f"duplicate compiler selection rules: {duplicate}")
    missing = sorted(known - set(names))
    if missing:
        raise KernelSelectionContractError(f"declared families without compiler rules: {missing}")
    return rules


def shape_constraint_reasons(shape: Mapping[str, int], constraints: Mapping[str, Any]) -> list[str]:
    """Validate shape predicates and return auditable reasons for every unsatisfied constraint."""
    reasons: list[str] = []
    for dimension, raw_constraint in constraints.items():
        if isinstance(raw_constraint, bool):
            raise KernelSelectionContractError(f"shape constraint for {dimension!r} must not be boolean")
        constraint = raw_constraint if isinstance(raw_constraint, Mapping) else {"eq": raw_constraint}
        allowed = {
            "eq", "min", "max", "multiple_of", "one_of", "power_of_two",
            "multiple_of_dimension", "tile_product_max",
        }
        unknown = set(constraint) - allowed
        if unknown:
            raise KernelSelectionContractError(f"shape constraint for {dimension!r} has unknown keys {sorted(unknown)}")
        for key in ("eq", "min", "max", "multiple_of"):
            if key not in constraint:
                continue
            number = constraint[key]
            minimum = 1 if key == "multiple_of" else 0
            if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
                raise KernelSelectionContractError(
                    f"shape constraint {dimension}.{key} must be an integer >= {minimum}"
                )
        if "one_of" in constraint:
            choices = constraint["one_of"]
            if not isinstance(choices, list) or not choices or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in choices
            ):
                raise KernelSelectionContractError(
                    f"shape constraint {dimension}.one_of must be a non-empty positive integer list"
                )
        if "power_of_two" in constraint and constraint["power_of_two"] is not True:
            raise KernelSelectionContractError(f"shape constraint {dimension}.power_of_two must be true")
        if "multiple_of_dimension" in constraint:
            reference = constraint["multiple_of_dimension"]
            if not isinstance(reference, str) or not reference.strip():
                raise KernelSelectionContractError(
                    f"shape constraint {dimension}.multiple_of_dimension must be a non-empty string"
                )
        if "tile_product_max" in constraint:
            product = constraint["tile_product_max"]
            if not isinstance(product, Mapping) or set(product) != {"dimension", "tile", "max"}:
                raise KernelSelectionContractError(
                    f"shape constraint {dimension}.tile_product_max needs exactly dimension, tile, and max"
                )
            reference = product["dimension"]
            if not isinstance(reference, str) or not reference.strip():
                raise KernelSelectionContractError(
                    f"shape constraint {dimension}.tile_product_max.dimension must be a non-empty string"
                )
            for field in ("tile", "max"):
                number = product[field]
                if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                    raise KernelSelectionContractError(
                        f"shape constraint {dimension}.tile_product_max.{field} must be a positive integer"
                    )
        if dimension not in shape:
            reasons.append(f"shape dimension {dimension} is unknown")
            continue
        value = shape[dimension]
        if "one_of" in constraint and value not in constraint["one_of"]:
            reasons.append(f"shape {dimension}={value}, requires one of {constraint['one_of']}")
        if "power_of_two" in constraint and (value <= 0 or value & (value - 1)):
            reasons.append(f"shape {dimension}={value}, requires a power of two")
        if "multiple_of_dimension" in constraint:
            reference = constraint["multiple_of_dimension"]
            divisor = shape.get(reference)
            if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 0:
                reasons.append(f"shape dimension {reference} is unknown or not a positive integer")
            elif value % divisor:
                reasons.append(f"shape {dimension}={value}, requires a multiple of {reference}={divisor}")
        if "tile_product_max" in constraint:
            product = constraint["tile_product_max"]
            reference = product["dimension"]
            other = shape.get(reference)
            if isinstance(other, bool) or not isinstance(other, int) or other <= 0:
                reasons.append(f"shape dimension {reference} is unknown or not a positive integer")
            else:
                tile = product["tile"]
                tiles = ((value + tile - 1) // tile) * ((other + tile - 1) // tile)
                if tiles > product["max"]:
                    reasons.append(
                        f"shape tile product ceil({dimension}/{tile})*ceil({reference}/{tile})={tiles}, "
                        f"requires <= {product['max']}"
                    )
        if "eq" in constraint and value != int(constraint["eq"]):
            reasons.append(f"shape {dimension}={value}, requires {int(constraint['eq'])}")
        if "min" in constraint and value < int(constraint["min"]):
            reasons.append(f"shape {dimension}={value}, requires >= {int(constraint['min'])}")
        if "max" in constraint and value > int(constraint["max"]):
            reasons.append(f"shape {dimension}={value}, requires <= {int(constraint['max'])}")
        if "multiple_of" in constraint:
            divisor = int(constraint["multiple_of"])
            if divisor <= 0:
                raise KernelSelectionContractError(f"shape constraint {dimension}.multiple_of must be positive")
            if value % divisor:
                reasons.append(f"shape {dimension}={value}, requires a multiple of {divisor}")
    return reasons


def validate_selection_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return a machine-readable coverage census, raising on a registry hole."""
    qualified, experimental = _family_sets(contract)
    rules = _rules(contract)
    return {
        "schema": "semantic_kernel_family_coverage_v1",
        "qualified_families": len(qualified),
        "experimental_families": len(experimental),
        "compiler_rules": len(rules),
        "qualified": sorted(qualified),
        "experimental": [{"family": family, "reason": experimental[family]} for family in sorted(experimental)],
        "missing_rules": [],
        "selection_is_numeric_qualification": False,
    }


def select_kernel_family(
    request: KernelRequest | Mapping[str, Any],
    hardware: HardwareCapabilities | Mapping[str, Any],
    contract: Mapping[str, Any],
) -> SelectionReport:
    """Rank eligible qualified families and retain every refusal in the result."""
    req = request if isinstance(request, KernelRequest) else KernelRequest.from_mapping(request)
    hw = hardware if isinstance(hardware, HardwareCapabilities) else HardwareCapabilities.from_mapping(hardware)
    qualified, experimental = _family_sets(contract)
    rules = _rules(contract)
    provisional: list[FamilyDecision] = []
    eligible: list[tuple[int, str]] = []

    for rule in rules:
        family = str(rule["family"])
        priority = int(rule.get("priority", 0))
        if family in experimental:
            provisional.append(
                FamilyDecision(
                    family,
                    "disabled",
                    priority,
                    (f"experimental family is fail-closed: {experimental[family]}",),
                )
            )
            continue
        reasons: list[str] = []
        semantic_ops = {str(item) for item in rule["semantic_ops"]}
        dtypes = {str(item) for item in rule["dtypes"]}
        if req.op not in semantic_ops:
            reasons.append(f"semantic op {req.op!r} not in {sorted(semantic_ops)}")
        if req.dtype not in dtypes:
            reasons.append(f"dtype {req.dtype!r} not in {sorted(dtypes)}")
        required_hw = {str(item) for item in rule.get("requires_hardware", ())}
        missing_hw = sorted(required_hw - hw.features)
        if missing_hw:
            reasons.append(f"missing derived hardware capabilities {missing_hw}")
        raw_shape = rule.get("shape", {})
        if not isinstance(raw_shape, Mapping):
            raise KernelSelectionContractError(f"{family}: shape must be a mapping")
        reasons.extend(shape_constraint_reasons(req.shape, raw_shape))
        if reasons:
            provisional.append(FamilyDecision(family, "refused", priority, tuple(reasons)))
        else:
            provisional.append(FamilyDecision(family, "eligible", priority, ()))
            eligible.append((priority, family))

    selected = sorted(eligible, key=lambda item: (-item[0], item[1]))[0][1] if eligible else None
    decisions: list[FamilyDecision] = []
    for decision in provisional:
        if decision.family == selected:
            decisions.append(
                FamilyDecision(
                    decision.family,
                    "selected",
                    decision.priority,
                    ("highest-priority eligible strategy",),
                )
            )
        elif decision.status == "eligible":
            decisions.append(
                FamilyDecision(
                    decision.family,
                    "eligible_not_selected",
                    decision.priority,
                    (f"lower priority than selected family {selected}",),
                )
            )
        else:
            decisions.append(decision)
    assert not selected or selected in qualified
    return SelectionReport(req, tuple(sorted(hw.features)), selected, tuple(decisions))
