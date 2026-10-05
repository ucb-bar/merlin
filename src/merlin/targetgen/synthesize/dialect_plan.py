"""Synthesize a dialect_plan (validates against dialect_plan.schema.yaml).

Required fields: target, dialect_name, ops, types, lowering, tests.

A registry-selected support provider's authored plan is used verbatim. Without one, a contract that
advertises the Merlin tensor-resident interface is GENERATED into a usable plan from its own op/type
names. Everything else gets a review-flagged skeleton with no asserted ops. An absent provider plan
never permits borrowing another provider's plan.
"""

from __future__ import annotations

import copy
from typing import Any

from ..evidence.store import Evidence


def _conservative(evidence: Evidence) -> dict[str, Any]:
    concepts = sorted(evidence.concept_names())
    return {
        "target": evidence.target,
        "dialect_name": evidence.target,
        "ops": [],
        "types": [],
        "lowering": [],
        "tests": [],
        "detected_concepts": concepts,
        "notes": "Ops/types/lowerings are a human-review decision; do not auto-generate "
        "dialect ops directly from instruction names.",
        "confidence": "low",
        "requires_human_review": True,
    }


def _curated(target_name: str) -> dict[str, Any] | None:
    """Read only the selected support provider's optional authored plan.

    No plan means derive one from its contract below, not borrow an in-tree plan.
    A present but malformed or escaping resource is an error, never absence.
    """
    import yaml

    from ..providers import contained_resource
    from ..target_registry import resolve

    selected = resolve(target_name)
    path = selected.dialect_plan_path
    if not path.exists() and not path.is_symlink():
        return None
    if selected.kind == "external":
        path = contained_resource(selected.base, str(path.relative_to(selected.base)))
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError(f"{path}: dialect plan must be a mapping")
    if plan.get("target") != selected.name:
        raise ValueError(f"{path}: dialect plan target differs from selected provider {selected.name!r}")
    return plan


def _generate_from_operation_capabilities(target_contract: dict[str, Any]) -> dict[str, Any] | None:
    """Project reviewed typed signatures from the existing operation contract.

    The source registry remains authoritative. This projection does not infer
    semantics, Pure effects, or a lowering from a mnemonic. An incomplete
    explicitly typed declaration refuses rather than falling back to the
    legacy name-only dialect generator.
    """
    declaration = target_contract.get("operation_capabilities") or {}
    if not isinstance(declaration, dict):
        raise ValueError("operation_capabilities must be a mapping")
    rows = declaration.get("operations") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("operation_capabilities.operations must be a list")
    typed_rows = [
        row for row in rows
        if row.get("domain") == "dialect" and isinstance(row.get("semantics"), dict)
        and "mlir_signature" in row["semantics"]
    ]
    if not typed_rows:
        return None
    dialect = target_contract.get("dialect_name")
    if not isinstance(dialect, str) or not dialect:
        raise ValueError("typed operation contract requires dialect_name")
    selected = [
        row for row in rows
        if isinstance(row, dict) and row.get("domain") == "dialect"
        and row.get("dialect") == dialect
    ]
    if not selected or any(row not in selected for row in typed_rows):
        raise ValueError("typed operation dialect differs from selected dialect_name")
    if any(
        not isinstance(row.get("semantics"), dict)
        or "mlir_signature" not in row["semantics"] for row in selected
    ):
        raise ValueError("every selected dialect operation needs an explicit MLIR signature")
    types = target_contract.get("types", [])
    if not isinstance(types, list) or any(not isinstance(row, dict) for row in types):
        raise ValueError("typed operation contracts require explicit custom type declarations")
    ops = []
    for row in sorted(selected, key=lambda item: str(item.get("operation"))):
        if not isinstance(row.get("operation"), str) or not row["operation"]:
            raise ValueError("typed operation contract has no operation name")
        ops.append({
            "name": row["operation"],
            "summary": str((row.get("semantics") or {}).get("summary") or row["operation"]),
            "signature": copy.deepcopy(row["semantics"]["mlir_signature"]),
        })
    plan = {
        "target": target_contract["name"],
        "dialect_name": target_contract["dialect_name"],
        "types": copy.deepcopy(types),
        "ops": ops,
        "lowering": [],
        "tests": [],
        "generated_from_contract": True,
        "requires_human_review": True,
        "source_operation_ids": [
            {"domain": "dialect", "dialect": target_contract["dialect_name"], "operation": op["name"]}
            for op in ops
        ],
    }
    from ..generate.typed_mlir import validate

    validate(plan)
    return plan


# The Merlin tensor-resident interface a target dialect lowers. role -> (canonical op name, matcher,
# summary). A contract that advertises this shape gets a GENERATED, usable dialect_plan (not a stub).
_ROLES: list[tuple[str, str, Any, str]] = [
    ("resident_pack", "pack", lambda o: "pack" in o, "pack + make RHS resident"),
    ("matmul", "matmul", lambda o: "matmul" in o, "matmul vs resident tensor -> accumulator"),
    ("commit", "commit", lambda o: "commit" in o, "apply epilogue + commit accumulator"),
    ("resident_evict", "evict", lambda o: "evict" in o or "release" in o, "free resident storage"),
]

# Optional vector-lane roles — emitted only when the contract advertises non-matmul pointwise
# compute (relu/bias_add/elementwise), i.e. the target has vector/scalar lanes beyond the systolic
# matmul. A pure-matmul target omits them (its dialect stays the 4-op resident core).
_VECTOR_ROLES: list[tuple[str, Any, str]] = [
    (
        "vector_map",
        lambda o: "vector_map" in o,
        "elementwise combine (add/mul/identity) + activation on the vector lanes",
    ),
    ("vector_reduce", lambda o: "vector_reduce" in o, "reduce a tensor (sum) on the vector lanes"),
]


def _has_vector_lanes(tc: dict[str, Any]) -> bool:
    """True when the contract advertises non-matmul pointwise compute — a vector/scalar lane beyond
    the systolic matmul. Derived from its capability ops (never a target name): any capability op
    other than matmul (bias_add, relu, an elementwise add/mul, …) implies a vector datapath."""
    cap_ops = set((tc.get("capabilities") or {}).get("ops") or [])
    return bool(cap_ops - {"matmul"})


def _is_tensor_resident(tc: dict[str, Any]) -> bool:
    """A target that implements the Merlin tensor-resident interface (packs a resident weight,
    accumulates, commits via a command buffer). Detected from the contract's own declarations."""
    feats = set(tc.get("features") or [])
    ops = set(tc.get("ops") or []) | set((tc.get("capabilities") or {}).get("ops") or [])
    return (
        "command_buffer" in feats
        and ("resident_packed_tensor" in feats or "accumulator_commit" in feats)
        and "matmul" in ops
    )


def _generate(target_contract: dict[str, Any]) -> dict[str, Any]:
    """Generate a usable dialect_plan from a tensor-resident contract (its op/type names drive the
    dialect; the interface->target lowering is the canonical mapping). Feeds the dialect factory."""
    name = target_contract["name"]
    dname = name.replace("_", "")
    decl_ops = list(target_contract.get("ops") or [])
    role_op = {role: (next((o for o in decl_ops if match(o)), canon)) for role, canon, match, _ in _ROLES}
    # Vector-lane roles are conditional: a pure-matmul target keeps the 4-op resident core.
    vector_roles = []
    if _has_vector_lanes(target_contract):
        vector_roles = [
            (role, next((o for o in decl_ops if match(o)), role), summ) for role, match, summ in _VECTOR_ROLES
        ]
    types = target_contract.get("types") or ["resident_tensor", "accumulator"]
    plan = {
        "target": name,
        "dialect_name": dname,
        "ops": [
            {"name": role_op[r], "summary": summ, "source_interface": f"interface.{r}"} for r, _c, _m, summ in _ROLES
        ]
        + [{"name": op, "summary": summ, "source_interface": f"interface.{r}"} for r, op, summ in vector_roles],
        "types": [{"name": t} for t in types],
        "lowering": [{"from": f"interface.{r}", "to": f"{dname}.{role_op[r]}"} for r, _c, _m, _s in _ROLES]
        + [{"from": f"interface.{r}", "to": f"{dname}.{op}"} for r, op, _s in vector_roles],
        "tests": [{"lit": "pack_roundtrip"}, {"lit": "matmul_commit_epilogue"}, {"lit": "evict_after_use"}],
        "confidence": "medium",
        "requires_human_review": False,
        "generated_from_contract": True,
    }
    from ...common.schemas import validate_or_raise

    validate_or_raise(plan, "dialect_plan")
    return plan


def synthesize_dialect_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Use the selected support plan or derive from this contract when it has no plan.

    Malformed present plans refuse; absence permits a generated tensor-resident plan
    or a review-flagged conservative skeleton, never another provider's authored plan.
    """
    name = target_contract.get("name")
    typed = _generate_from_operation_capabilities(target_contract)
    curated = _curated(name)
    if typed is not None and curated is not None:
        raise ValueError("selected typed operation contract and separate dialect plan duplicate dialect authority")
    if curated is not None:
        return curated
    if typed is not None:
        return typed
    if _is_tensor_resident(target_contract):
        return _generate(target_contract)  # usable generated plan (was: empty _conservative stub)
    return _conservative(evidence)
