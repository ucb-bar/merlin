"""Derived Phase 0 stage: the target's instruction-ROLE taxonomy and the experiment's declared policy.

A rule such as "no hardware loop-descriptor instructions" is a statement in the closed role vocabulary
(:mod:`merlin.kernels.roles`), never a list of instruction names: which instructions carry a role is a
fact about the target, derived from its own RTL-derived encoding table through its declared compute
endpoints (:func:`merlin.perf.task_instruction_evidence.target_instruction_facts`). Phase 0 derives
that taxonomy once, resolves the experiment's ``prohibited_instruction_roles`` against it, and records
the result where Phase 1 grading and Phase 2 measurement read it. Enforcement -- scanning the WHOLE
linked program for a prohibited instruction -- is a separate owner; this module only declares and
derives, and fails closed: an unknown role is refused, an underivable taxonomy is ``UNKNOWN``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

TAXONOMY_SCHEMA = "merlin.phase0.instruction_roles.v1"
POLICY_SCHEMA = "merlin.phase0.instruction_policy.v1"
UNKNOWN = "UNKNOWN"


def validate_roles(roles: object) -> list[str]:
    """Declared role names, checked against the closed vocabulary; duplicates and unknowns refused."""
    from merlin.kernels.roles import ROLES

    if roles is None:
        return []
    if not isinstance(roles, (list, tuple)) or any(not isinstance(r, str) or not r for r in roles):
        raise ValueError("prohibited_instruction_roles must be a list of role names")
    if len(set(roles)) != len(roles):
        raise ValueError("prohibited_instruction_roles repeats a role")
    unknown = sorted(set(roles) - set(ROLES))
    if unknown:
        raise ValueError(f"unknown instruction role(s) {unknown}; the closed vocabulary is {sorted(ROLES)}")
    return list(roles)


def derive_role_taxonomy(target: str) -> dict[str, Any]:
    """``{role: [instruction]}`` for every instruction the target's own facts declare, or UNKNOWN.

    Call inside the frozen contract/facts scope (``observed_contract`` / ``observed_facts``) so the
    derivation reads the selected evidence rather than an ambient cache.
    """
    from merlin.kernels.decode.rocc import funct_table_for
    from merlin.kernels.endpoints import endpoints_for
    from merlin.kernels.roles import ROLES
    from merlin.perf.task_instruction_evidence import declared_instruction_set

    # The same composition ``target_instruction_facts`` makes, without requiring the target's runtime
    # backend: the taxonomy needs only the RTL-derived name table and the declared endpoint roles. The
    # ISA constants (the custom opcode) are the scanner's business and are recorded when available.
    try:
        names = funct_table_for(target).get("names") or {}
        endpoints = endpoints_for(target)
        if not endpoints:
            # No endpoint binds a role to any of this target's instructions, so every instruction would
            # come back role-less and a prohibited role would match nothing -- a policy that cannot fail,
            # reading as derived. Unknown, with the reason, instead.
            from merlin.kernels.endpoints import spec_path

            raise LookupError(f"no compute endpoint in {spec_path()} declares instruction roles for {target!r}")
        facts: dict[str, Any] = {
            "target": target,
            "instruction_names": names,
            "roles_by_selector": {
                str(selector): sorted({role for endpoint in endpoints for role in endpoint.roles_of(name)})
                for selector, name in names.items()
            },
            "role_sources": [endpoint.source for endpoint in endpoints],
        }
        declared = declared_instruction_set(facts)
    except Exception as exc:  # noqa: BLE001 -- recorded as UNKNOWN, never as an empty role set
        return {
            "schema": TAXONOMY_SCHEMA,
            "target": target,
            "status": UNKNOWN,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if declared.get("status") != "derived":
        return {"schema": TAXONOMY_SCHEMA, "target": target, "status": UNKNOWN, "reason": declared.get("reason")}
    by_role: dict[str, list[dict[str, str]]] = {role: [] for role in sorted(ROLES)}
    unroled: list[dict[str, str]] = []
    for row in declared.get("instructions") or ():
        instruction = {"selector": str(row["funct"]), "name": str(row["name"])}
        roles = row.get("roles")
        if not roles:
            unroled.append(instruction)
            continue
        for role in roles:
            by_role.setdefault(str(role), []).append(instruction)
    return {
        "schema": TAXONOMY_SCHEMA,
        "target": target,
        "status": "derived",
        "vocabulary": sorted(ROLES),
        "by_role": by_role,
        "instructions_without_roles": unroled,
        "role_sources": list(declared.get("role_sources") or ()),
        "derivation": "merlin.perf.task_instruction_evidence.target_instruction_facts",
    }


#: The one status a declared policy may be enforced under.
RESOLVED = "resolved"
NONE_DECLARED = "none_declared"


def resolve_policy(roles: Sequence[str], taxonomy: Mapping[str, Any]) -> dict[str, Any]:
    """The declared prohibition resolved against the derived taxonomy: what it forbids on THIS target.

    FAILS CLOSED ON A VACUOUS ROLE. A declared role that matches no instruction of this target forbids
    nothing, so a scan under it cannot refuse anything -- and a policy that cannot fail is not one.
    Recorded as ``vacuous_roles`` it used to sit beside ``status: resolved``, which every consumer read
    as "enforced": a sealed Phase 0 release prohibited ``loop_descriptor`` while matching zero
    instructions, because the endpoint declaration that binds the role was never read. Any vacuous role
    makes the policy ``UNKNOWN`` with the reason, so verified Phase 0 refuses it."""
    declared = validate_roles(list(roles))
    derived = taxonomy.get("status") == "derived"
    by_role = taxonomy.get("by_role") or {}
    prohibited = {role: copy.deepcopy(by_role.get(role) or []) for role in declared} if derived else {}
    vacuous = sorted(role for role in declared if derived and not prohibited.get(role))
    if not declared:
        status, reason = NONE_DECLARED, None
    elif not derived:
        status, reason = UNKNOWN, f"the target's instruction-role taxonomy was not derived: {taxonomy.get('reason')}"
    elif vacuous:
        sources = list(taxonomy.get("role_sources") or ())
        status, reason = (
            UNKNOWN,
            (
                f"declared prohibited role(s) {vacuous} match no instruction of target {taxonomy.get('target')!r} "
                f"(role sources: {sources or 'none'}); a prohibition that forbids nothing cannot be enforced"
            ),
        )
    else:
        status, reason = RESOLVED, None
    return {
        "schema": POLICY_SCHEMA,
        "prohibited_instruction_roles": declared,
        "status": status,
        **({"reason": reason} if reason else {}),
        "prohibited_instructions": prohibited,
        # A declared role no instruction of this target carries forbids nothing here: recorded by name,
        # and it makes the policy UNKNOWN (above) rather than resolved.
        "vacuous_roles": vacuous,
        "taxonomy_status": taxonomy.get("status"),
        **({"taxonomy_reason": taxonomy.get("reason")} if not derived else {}),
        "applies_to": ["phase1_capsule_elfs", "phase1_whole_model_elfs", "phase2_candidate_arm"],
        "exempt": ["phase2_vendor_reference_arm"],
        "enforcement": "whole_linked_program_scan",
    }


def enforcement_problems(policy: Mapping[str, Any] | None, roles: Sequence[str] | None = None) -> list[str]:
    """Why ``policy`` cannot be enforced for ``roles`` (its own declared roles when omitted); empty when it can.

    The one predicate every consumer of a sealed instruction policy applies -- verified Phase 0 before
    it seals, Phase 2 measured/cell/group modes before they spend machine time, champion export: the
    policy resolved, it declares every role the caller enforces, and each of those roles names at least
    one of the target's instructions."""
    if not isinstance(policy, Mapping):
        return ["no instruction policy was supplied"]
    declared = list(policy.get("prohibited_instruction_roles") or ())
    wanted = declared if roles is None else list(roles)
    if not wanted:
        return []
    problems = []
    if policy.get("status") != RESOLVED:
        why = policy.get("reason") or policy.get("taxonomy_reason") or "no reason recorded"
        problems.append(f"the instruction policy is {policy.get('status')!r}, not {RESOLVED!r}: {why}")
    missing = sorted(set(wanted) - set(declared))
    if missing:
        problems.append(f"the instruction policy does not declare role(s) {missing} (it declares {declared})")
    prohibited = policy.get("prohibited_instructions")
    prohibited = prohibited if isinstance(prohibited, Mapping) else {}
    empty = sorted(role for role in wanted if role in declared and not prohibited.get(role))
    if empty:
        problems.append(f"the instruction policy prohibits no instruction for role(s) {empty}")
    return problems


def require_enforceable(policy: Mapping[str, Any] | None, roles: Sequence[str] | None = None) -> dict[str, Any]:
    """``policy``, or :class:`ValueError` naming every reason it cannot be enforced for ``roles``."""
    problems = enforcement_problems(policy, roles)
    if problems:
        raise ValueError("; ".join(problems))
    return dict(policy or {})


def candidate_arm_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """The instruction policy a form-perf candidate arm carries: the declared roles, by value."""
    roles = list((policy or {}).get("prohibited_instruction_roles") or ())
    return {
        "prohibited_instruction_roles": roles,
        "source": "experiment.policy" if roles else "experiment.policy (none declared)",
    }
