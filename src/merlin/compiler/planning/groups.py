"""Retain one source-bound grouping decision before region materialization.

The current grouping algorithm owns numerical and target legality. This record
keeps its actual IR members, decisions and selected stored-argument identities
together, so outlining and reports do not independently choose groups again.
It is not an execution schedule or a hardware/ABI qualification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any


def _group_identity(groups, operations) -> str:
    ordinals = {id(op): ordinal for ordinal, op in enumerate(operations)}

    def ordinal(op):
        if op is None:
            return None
        if id(op) not in ordinals:
            raise ValueError("group plan refers to an operation outside its source")
        return ordinals[id(op)]

    rows = []
    for group in groups:
        row = {}
        for field in fields(group):
            value = getattr(group, field.name)
            if field.name == "members":
                value = [ordinal(op) for op in value]
            elif field.name in {"root", "stopped_at"}:
                value = ordinal(value)
            row[field.name] = value
        rows.append(row)
    return json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class GroupPlan:
    """Current source identities and the existing planner's selected groups.

    Group objects are retained for existing materializers. Every consumer checks
    them against the original source and decision snapshot before reuse. This
    consistency check does not turn serialized data into lowering authority.
    """

    source: Any
    target: str
    groups: tuple
    oracle: Any
    weight_args: frozenset[int] | None
    source_sha256: str
    _operations: tuple
    _decisions: str

    def require_current(self, module, target: str, *, weight_args=None) -> None:
        from merlin.transforms.certificates import ir_digest

        if module is not self.source or target != self.target:
            raise ValueError("group plan belongs to another source or target")
        current = tuple(module.walk())
        if len(current) != len(self._operations) or any(
            left is not right for left, right in zip(current, self._operations, strict=True)
        ) or ir_digest(module) != self.source_sha256:
            raise ValueError("group plan source changed; plan the realized program again")
        selected = None if weight_args is None else frozenset(weight_args)
        if selected != self.weight_args:
            raise ValueError("group plan stored-argument binding changed")
        if _group_identity(self.groups, current) != self._decisions:
            raise ValueError("group plan decisions changed; replan before materialization")

    def report(self) -> dict:
        """Describe the selected groups without another grouping decision."""
        from merlin.xdsl_dialects.lowering import compute_groups

        self.require_current(self.source, self.target, weight_args=self.weight_args)
        result = compute_groups.plan(self.source, self.target, oracle=self.oracle, groups=self.groups)
        self.require_current(self.source, self.target, weight_args=self.weight_args)
        return result


def plan_groups(module, target: str, *, oracle=None, weight_args=None) -> GroupPlan:
    """Select groups once using the existing source and target legality owner."""
    from merlin.transforms.certificates import ir_digest
    from merlin.xdsl_dialects.lowering import compute_groups

    before = ir_digest(module)
    operations = tuple(module.walk())
    oracle = oracle or compute_groups.TargetOracle(target)
    groups = tuple(compute_groups.form_groups(module, target, oracle=oracle))
    current = tuple(module.walk())
    if ir_digest(module) != before or len(current) != len(operations) or any(
        left is not right for left, right in zip(current, operations, strict=True)
    ):
        raise ValueError("group planning changed the source program")
    return GroupPlan(
        module, target, groups, oracle,
        None if weight_args is None else frozenset(weight_args),
        before, operations, _group_identity(groups, operations),
    )
