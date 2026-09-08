"""Declarative kernel-family selection for the Muon/MX compiler boundary.

The kernel library is evidence for reusable schedules, not an implementation dependency.  This
module consequently reads only its small selection contract and chooses a *strategy family* from a
semantic operation, its dtype/shape, and caller-derived hardware capabilities.  It never imports,
copies, links, or calls the reference kernels.

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
                raise KernelSelectionContractError(
                    f"kernel request dimension {name!r} must be a positive integer")
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
        raise KernelSelectionContractError(
            f"families cannot be both qualified and experimental: {sorted(overlap)}")
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
            raise KernelSelectionContractError(
                f"compiler selection rule names undeclared family {family!r}")
        if not isinstance(rule.get("semantic_ops"), list) or not rule["semantic_ops"]:
            raise KernelSelectionContractError(f"{family}: semantic_ops must be a non-empty list")
        if not isinstance(rule.get("dtypes"), list) or not rule["dtypes"]:
            raise KernelSelectionContractError(f"{family}: dtypes must be a non-empty list")
        if not isinstance(rule.get("requires_hardware", []), list):
            raise KernelSelectionContractError(f"{family}: requires_hardware must be a list")
        priority = rule.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise KernelSelectionContractError(f"{family}: priority must be an integer")
    duplicate = sorted({name for name in names if names.count(name) > 1})
    if duplicate:
        raise KernelSelectionContractError(f"duplicate compiler selection rules: {duplicate}")
    missing = sorted(known - set(names))
    if missing:
        raise KernelSelectionContractError(f"declared families without compiler rules: {missing}")
    return rules


def _constraint_reasons(shape: Mapping[str, int], constraints: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for dimension, raw_constraint in constraints.items():
        if dimension not in shape:
            reasons.append(f"shape dimension {dimension} is unknown")
            continue
        if isinstance(raw_constraint, bool):
            raise KernelSelectionContractError(
                f"shape constraint for {dimension!r} must not be boolean")
        constraint = raw_constraint if isinstance(raw_constraint, Mapping) else {"eq": raw_constraint}
        value = shape[dimension]
        allowed = {"eq", "min", "max", "multiple_of"}
        unknown = set(constraint) - allowed
        if unknown:
            raise KernelSelectionContractError(
                f"shape constraint for {dimension!r} has unknown keys {sorted(unknown)}")
        if "eq" in constraint and value != int(constraint["eq"]):
            reasons.append(f"shape {dimension}={value}, requires {int(constraint['eq'])}")
        if "min" in constraint and value < int(constraint["min"]):
            reasons.append(f"shape {dimension}={value}, requires >= {int(constraint['min'])}")
        if "max" in constraint and value > int(constraint["max"]):
            reasons.append(f"shape {dimension}={value}, requires <= {int(constraint['max'])}")
        if "multiple_of" in constraint:
            divisor = int(constraint["multiple_of"])
            if divisor <= 0:
                raise KernelSelectionContractError(
                    f"shape constraint {dimension}.multiple_of must be positive")
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
        "experimental": [
            {"family": family, "reason": experimental[family]}
            for family in sorted(experimental)
        ],
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
            provisional.append(FamilyDecision(
                family, "disabled", priority,
                (f"experimental family is fail-closed: {experimental[family]}",),
            ))
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
        reasons.extend(_constraint_reasons(req.shape, raw_shape))
        if reasons:
            provisional.append(FamilyDecision(family, "refused", priority, tuple(reasons)))
        else:
            provisional.append(FamilyDecision(family, "eligible", priority, ()))
            eligible.append((priority, family))

    selected = sorted(eligible, key=lambda item: (-item[0], item[1]))[0][1] if eligible else None
    decisions: list[FamilyDecision] = []
    for decision in provisional:
        if decision.family == selected:
            decisions.append(FamilyDecision(
                decision.family, "selected", decision.priority,
                ("highest-priority eligible strategy",),
            ))
        elif decision.status == "eligible":
            decisions.append(FamilyDecision(
                decision.family, "eligible_not_selected", decision.priority,
                (f"lower priority than selected family {selected}",),
            ))
        else:
            decisions.append(decision)
    assert not selected or selected in qualified
    return SelectionReport(req, tuple(sorted(hw.features)), selected, tuple(decisions))


_DTYPE_NAMES = {
    "f32": "fp32",
    "float32": "fp32",
    "f16": "fp16",
    "float16": "fp16",
    "bf16": "bf16",
    "i8": "int8",
    "int8": "int8",
    "mxfp4": "mxfp4",
    "mxfp6": "mxfp6",
    "mxfp8": "mxfp8",
}


def derive_hardware_capabilities(target_contract: Mapping[str, Any]) -> HardwareCapabilities:
    """Derive selector facts from a target contract without guessing absent capabilities.

    The standardized feature names in ``features`` are accepted as declarations.  Structural facts
    add only their direct meanings: a SIMT unit, an MX-capable systolic unit, block-E8M0 scaling, and
    shared memory.  In particular, math operations such as exp/tanh and mesh readback are *not*
    inferred from a target family or name.
    """
    features = {
        str(item).strip()
        for item in target_contract.get("features", ())
        if str(item).strip()
    }
    for unit in target_contract.get("compute_units", ()):
        if not isinstance(unit, Mapping):
            continue
        kind = str(unit.get("kind") or "").strip()
        if kind == "simt":
            features.add("simt")
        elif kind == "systolic":
            features.add("mx_mesh")
        if str(unit.get("scaling") or "").strip() == "block_e8m0":
            features.add("block_e8m0")
    memory = target_contract.get("memory_model")
    if isinstance(memory, Mapping) and memory.get("shared_memory") is True:
        features.add("shared_memory")
    return HardwareCapabilities(frozenset(features))


def request_from_command_buffer(cb: Mapping[str, Any]) -> KernelRequest:
    """Extract a semantic request from a command buffer at the Muon emission boundary.

    The production bridge recognizes standalone 2-D fp32 row-broadcast add and layer normalization.
    Shape, dtype, and operation are reconstructed from the command and tensor ABI; capsule identity and
    oracle data are never inspected.  More command families must add equally explicit extractors before
    they can opt into contract-driven family dispatch.
    """
    commands = cb.get("commands")
    tensors = cb.get("tensors")
    if not isinstance(commands, list) or not isinstance(tensors, Mapping):
        raise KernelSelectionContractError("command buffer has no commands/tensors selection facts")
    if (len(commands) == 2 and all(isinstance(command, Mapping) for command in commands)
            and all(str(command.get("opcode") or "").upper() == "RMSNORM"
                    for command in commands)):
        first_ops, second_ops = commands[0].get("operands"), commands[1].get("operands")
        if not isinstance(first_ops, Mapping) or not isinstance(second_ops, Mapping):
            raise KernelSelectionContractError("chained RMSNORM operands are not mappings")
        src, gamma1, intermediate = first_ops.get("src"), first_ops.get("gamma"), first_ops.get("dst")
        gamma2, dst = second_ops.get("gamma"), second_ops.get("dst")
        if second_ops.get("src") != intermediate:
            raise KernelSelectionContractError("two RMSNORM commands do not form a chain")
        if not all(isinstance(name, str) and name in tensors for name in (src, gamma1, gamma2, dst)):
            raise KernelSelectionContractError("four-norm operands are absent from the tensor ABI")
        src_spec, gamma1_spec, gamma2_spec, dst_spec = (
            tensors[src], tensors[gamma1], tensors[gamma2], tensors[dst])
        if not all(isinstance(spec, Mapping)
                   for spec in (src_spec, gamma1_spec, gamma2_spec, dst_spec)):
            raise KernelSelectionContractError("four-norm tensor specifications are not mappings")
        src_shape = src_spec.get("shape")
        if (not isinstance(src_shape, list) or len(src_shape) != 2
                or gamma1_spec.get("shape") not in ([src_shape[1]], [1, src_shape[1]])
                or gamma2_spec.get("shape") not in ([src_shape[1]], [1, src_shape[1]])
                or dst_spec.get("shape") != src_shape):
            raise KernelSelectionContractError("RMSNORM chain has incompatible tensor shapes")
        raw_dtype = str(dst_spec.get("dtype") or src_spec.get("dtype") or "").lower()
        dtype = _DTYPE_NAMES.get(raw_dtype)
        if dtype is None:
            raise KernelSelectionContractError(f"unsupported command-buffer dtype {raw_dtype!r}")
        return KernelRequest.from_mapping({
            "op": "decoder_four_norm", "dtype": dtype,
            "shape": {"rows": src_shape[0], "cols": src_shape[1]},
        })
    if len(commands) != 1 or not isinstance(commands[0], Mapping):
        raise KernelSelectionContractError(
            "semantic family extraction currently requires one command")
    command = commands[0]
    operands = command.get("operands")
    attrs = command.get("attributes") or {}
    opcode = str(command.get("opcode") or "").upper()
    if opcode == "LAYERNORM" and isinstance(operands, Mapping):
        src, gamma = operands.get("src"), operands.get("gamma")
        beta, dst = operands.get("beta"), operands.get("dst")
        if not all(isinstance(name, str) and name in tensors
                   for name in (src, gamma, beta, dst)):
            raise KernelSelectionContractError("layernorm operands are absent from the tensor ABI")
        src_spec, gamma_spec = tensors[src], tensors[gamma]
        beta_spec, dst_spec = tensors[beta], tensors[dst]
        if not all(isinstance(spec, Mapping)
                   for spec in (src_spec, gamma_spec, beta_spec, dst_spec)):
            raise KernelSelectionContractError("layernorm tensor specifications are not mappings")
        src_shape = src_spec.get("shape")
        if (not isinstance(src_shape, list) or len(src_shape) != 2
                or gamma_spec.get("shape") != [src_shape[1]]
                or beta_spec.get("shape") != [src_shape[1]]
                or dst_spec.get("shape") != src_shape):
            raise KernelSelectionContractError(
                "LAYERNORM is not a 2-D row normalization with matching affine vectors")
        raw_dtype = str(dst_spec.get("dtype") or src_spec.get("dtype") or "").lower()
        dtype = _DTYPE_NAMES.get(raw_dtype)
        if dtype is None:
            raise KernelSelectionContractError(f"unsupported command-buffer dtype {raw_dtype!r}")
        return KernelRequest.from_mapping({
            "op": "layernorm",
            "dtype": dtype,
            "shape": {"rows": src_shape[0], "cols": src_shape[1]},
        })
    if (str(command.get("opcode") or "").upper() != "VECTOR_MAP"
            or not isinstance(operands, Mapping)
            or not isinstance(attrs, Mapping)
            or attrs.get("combine", "add") != "add"):
        raise KernelSelectionContractError(
            "semantic family extraction currently supports LAYERNORM or row-broadcast VECTOR_MAP(add)")
    lhs, rhs, dst = operands.get("lhs"), operands.get("rhs"), operands.get("dst")
    if not all(isinstance(name, str) and name in tensors for name in (lhs, rhs, dst)):
        raise KernelSelectionContractError("bias-add operands are absent from the tensor ABI")
    lhs_spec, rhs_spec, dst_spec = tensors[lhs], tensors[rhs], tensors[dst]
    if not all(isinstance(spec, Mapping) for spec in (lhs_spec, rhs_spec, dst_spec)):
        raise KernelSelectionContractError("bias-add tensor specifications are not mappings")
    lhs_shape = lhs_spec.get("shape")
    rhs_shape = rhs_spec.get("shape")
    dst_shape = dst_spec.get("shape")
    if (not isinstance(lhs_shape, list) or len(lhs_shape) != 2
            or rhs_shape != [lhs_shape[1]] or dst_shape != lhs_shape):
        raise KernelSelectionContractError(
            "VECTOR_MAP(add) is not a 2-D row-broadcast bias add")
    raw_dtype = str(dst_spec.get("dtype") or lhs_spec.get("dtype") or "").lower()
    dtype = _DTYPE_NAMES.get(raw_dtype)
    if dtype is None:
        raise KernelSelectionContractError(f"unsupported command-buffer dtype {raw_dtype!r}")
    return KernelRequest.from_mapping({
        "op": "bias_add",
        "dtype": dtype,
        "shape": {"rows": lhs_shape[0], "cols": lhs_shape[1]},
    })


def select_command_buffer_family(
    cb: Mapping[str, Any],
    hardware_contract: Mapping[str, Any],
    selection_contract: Mapping[str, Any],
) -> SelectionReport:
    """Run semantic selection on a real emitted candidate using only compiler-visible facts."""
    return select_kernel_family(
        request_from_command_buffer(cb),
        derive_hardware_capabilities(hardware_contract),
        selection_contract,
    )
