"""Validate synthesized plan dicts against the shared schemas (pre-write check)."""
from __future__ import annotations

from typing import Any

from ...common import schemas

# plan key (as produced by the pipeline) -> schema name
PLAN_SCHEMAS: dict[str, str] = {
    "target_contract": "target_contract",
    "dialect_plan": "dialect_plan",
    "runtime_adapter_plan": "runtime_adapter_plan",
    "zephyr_plan": "zephyr_plan",
    "llvm_extension_plan": "llvm_extension_plan",
}


def validate_plans(plans: dict[str, Any]) -> list[str]:
    """Validate every plan in ``plans`` against its schema. Empty list == valid."""
    problems: list[str] = []
    for key, obj in plans.items():
        schema_name = PLAN_SCHEMAS.get(key)
        if schema_name is None:
            problems.append(f"{key}: no schema mapping")
            continue
        problems.extend(f"{key}: {p}" for p in schemas.validate(obj, schema_name))
    return problems
