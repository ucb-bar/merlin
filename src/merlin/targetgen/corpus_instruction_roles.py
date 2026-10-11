"""Explicit corpus role selections bound to a neutral command-class roster.

These are authored corpus requirements, not inferred hardware capabilities or
instruction schedules. The ordinary numerical and execution gates remain separate.
"""

from __future__ import annotations

import hashlib
import json

from .isa_taxonomy import required_classes_from_roles, required_role_slots

_ROLES = frozenset(
    {
        "memory",
        "weight_load",
        "matmul",
        "acc_readout",
        "acc_readout_scaled",
        "tensor_compute_unary",
        "tensor_compute_binary",
    }
)


def selected_classes(contract: dict):
    """Return an explicit role selector, or None when none was selected."""
    if "corpus_instruction_roles" not in contract:
        return None
    selection = contract["corpus_instruction_roles"]
    if (
        not isinstance(selection, dict)
        or set(selection)
        not in (
            {"version", "instruction_roster_sha256", "by_role"},
            {"version", "instruction_roster_sha256", "by_role", "epilogue_roles"},
            {"version", "instruction_roster_sha256", "by_role", "operation_roles"},
            {"version", "instruction_roster_sha256", "by_role", "epilogue_roles", "operation_roles"},
        )
        or type(selection["version"]) is not int
        or selection["version"] not in (1, 2)
    ):
        raise ValueError("invalid corpus_instruction_roles selection")
    if "operation_roles" in selection and selection["version"] != 2:
        raise ValueError("operation roles require corpus selection version 2")
    roster = (contract.get("rocc_operand_roles") or {}).get("instructions")
    if not isinstance(roster, list) or not roster:
        raise ValueError("corpus instruction roles require an explicit command roster")
    names = []
    for row in roster:
        if not isinstance(row, dict) or not isinstance(row.get("class"), str) or not row["class"]:
            raise ValueError("invalid command class in corpus role roster")
        names.append(row["class"])
    if len(names) != len(set(names)):
        raise ValueError("duplicate command class in corpus role roster")
    digest = hashlib.sha256(
        json.dumps(roster, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if selection["instruction_roster_sha256"] != digest:
        raise ValueError("corpus instruction roles disagree with selected command roster")
    roles = selection["by_role"]
    if not isinstance(roles, dict) or not roles or set(roles) - _ROLES:
        raise ValueError("invalid corpus instruction role vocabulary")
    epilogues = selection.get("epilogue_roles", {})
    if not isinstance(epilogues, dict) or any(
        not isinstance(stage, str) or not stage or not isinstance(mapping, dict) or not mapping or set(mapping) - _ROLES
        for stage, mapping in epilogues.items()
    ):
        raise ValueError("invalid corpus epilogue role selection")
    operations = selection.get("operation_roles", {})
    from .semantic_families import from_op

    if not isinstance(operations, dict) or any(
        not isinstance(op, str) or not from_op(op) or not isinstance(mapping, dict)
        or not mapping or set(mapping) - _ROLES
        for op, mapping in operations.items()
    ):
        raise ValueError("invalid corpus operation role selection")
    for classes in [*(classes for mapping in operations.values() for classes in mapping.values()), *roles.values(), *(classes for mapping in epilogues.values() for classes in mapping.values())]:
        if (
            not isinstance(classes, list)
            or not classes
            or any(not isinstance(name, str) or name not in names for name in classes)
            or len(classes) != len(set(classes))
        ):
            raise ValueError("corpus instruction role names an undeclared or duplicate class")
    # Snapshot caller-owned lists: a later mutation cannot change the selection.
    selected = {role: list(classes) for role, classes in roles.items()}
    selected_epilogues = {
        stage: {role: list(classes) for role, classes in mapping.items()} for stage, mapping in epilogues.items()
    }

    selected_operations = {
        op: {role: list(classes) for role, classes in mapping.items()} for op, mapping in operations.items()
    }

    def classes_for(**kwargs):
        from .semantic_families import from_op, primitives_of

        bound = {role: list(classes) for role, classes in selected.items()}
        for role, classes in selected_operations.get(kwargs.get("op", "matmul"), {}).items():
            if role in bound and bound[role] != classes:
                raise ValueError("conflicting corpus operation role selection")
            bound[role] = list(classes)
        # Fused stage requirements do not declare a general unary/binary unit.
        if "contraction" in primitives_of(from_op(kwargs.get("op", "matmul")) or ""):
            for stage in kwargs.get("epilogue", ()):
                for role, classes in selected_epilogues.get(stage, {}).items():
                    if role in bound and bound[role] != classes:
                        raise ValueError("conflicting corpus epilogue role selection")
                    bound[role] = list(classes)
        for slot in required_role_slots(**kwargs):
            if not any(role in bound for role in slot):
                raise ValueError(f"selected corpus roles lack required role slot {slot}")
        classes = required_classes_from_roles(bound, **kwargs)
        # Every explicitly selected operation class remains an obligation; the
        # generic role resolver chooses only the first class for each slot.
        for names in selected_operations.get(kwargs.get("op", "matmul"), {}).values():
            for name in names:
                if name not in classes:
                    classes.append(name)
        if not classes:
            raise ValueError("selected corpus roles yield no required instruction classes")
        return classes

    return classes_for
