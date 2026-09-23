"""Target-owned declarations for source-specific RTL fact readers.

Shared extractors can parse several hardware classes. The vocabulary locating a source's
instruction span and accumulator ports belongs to the selected target contract, not to Merlin.
Missing declarations disable only that optional fallback; malformed declarations fail closed.
"""

from __future__ import annotations

from typing import Any


def _rtl_extraction(target: str) -> dict[str, Any]:
    import yaml

    from .facts import target_contract_path

    path = target_contract_path(target)
    if not path.is_file():
        return {}
    contract = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: target contract must be a mapping")
    extraction = contract.get("rtl_extraction") or {}
    if not isinstance(extraction, dict):
        raise ValueError(f"{path}: rtl_extraction must be a mapping")
    return extraction


def scala_funct_block(target: str) -> tuple[str, str] | None:
    """The target's Scala funct-span anchors, or no header fallback when undeclared."""
    block = _rtl_extraction(target).get("scala_funct_block")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("scala_funct_block must be a mapping")
    start, stop = block.get("start_comment"), block.get("stop_declaration")
    if (
        not isinstance(start, str)
        or not start.strip().startswith("//")
        or not isinstance(stop, str)
        or not stop.isidentifier()
    ):
        raise ValueError("scala_funct_block needs a Scala comment and identifier stop_declaration")
    return start, stop


def accumulator_layout(target: str) -> dict[str, str] | None:
    """The target's optional accumulator HW-port vocabulary, never a guessed module name."""
    layout = _rtl_extraction(target).get("accumulator_memory")
    if layout is None:
        return None
    required = ("module", "memory_name", "address_port", "data_prefix", "data_suffix", "mask_prefix")
    if not isinstance(layout, dict) or any(not isinstance(layout.get(key), str) or not layout[key] for key in required):
        raise ValueError(f"accumulator_memory needs nonempty {', '.join(required)}")
    result = {key: layout[key] for key in required}
    field = layout.get("firrtl_data_field")
    if field is not None:
        if not isinstance(field, str) or not all(part.isidentifier() for part in field.split(".")):
            raise ValueError("accumulator_memory.firrtl_data_field must be a dotted FIRRTL port path")
        result["firrtl_data_field"] = field
    return result


def firrtl_role_probe(target: str) -> dict[str, Any] | None:
    """Optional source-specific role anchors declared by the selected support provider.

    The reader implements FIRRTL parsing; module names, source markers, and published role
    names belong to the target. Missing declarations leave the scoped census in charge.
    """
    probe = _rtl_extraction(target).get("firrtl_role_probe")
    if probe is None:
        return None
    if not isinstance(probe, dict):
        raise ValueError("firrtl_role_probe must be a mapping")
    shapes = {
        "array": ("name", "parent_module_contains", "child_module_prefix"),
        "memory": ("name", "declaration_contains", "source_contains", "datapath_name"),
    }
    for kind, fields in shapes.items():
        block = probe.get(kind)
        if block is not None and (
            not isinstance(block, dict)
            or any(not isinstance(block.get(field), str) or not block[field] for field in fields)
        ):
            raise ValueError(f"firrtl_role_probe.{kind} needs nonempty {', '.join(fields)}")
    memory = probe.get("memory")
    if memory is not None and not memory["declaration_contains"].endswith("UInt<"):
        raise ValueError("firrtl_role_probe.memory.declaration_contains must end in UInt<")
    interfaces = probe.get("interfaces", [])
    if not isinstance(interfaces, list) or any(
        not isinstance(item, dict)
        or any(not isinstance(item.get(field), str) or not item[field] for field in ("module", "name", "evidence"))
        for item in interfaces
    ):
        raise ValueError("firrtl_role_probe.interfaces needs module, name, and evidence strings")
    if not probe.get("array") and not memory and not interfaces:
        raise ValueError("firrtl_role_probe must declare at least one role anchor")
    return probe


def boolean_feature_gate(target: str) -> dict[str, str] | None:
    """One optional FIRRTL build gate whose feature name and source names are target-owned."""
    gate = _rtl_extraction(target).get("boolean_feature_gate")
    if gate is None:
        return None
    required = ("feature", "module", "node", "dynamic_operand_contains", "config_field")
    if not isinstance(gate, dict) or any(not isinstance(gate.get(key), str) or not gate[key] for key in required):
        raise ValueError(f"boolean_feature_gate needs nonempty {', '.join(required)}")
    return {key: gate[key] for key in required}
