"""Separate raw captured scope demand from Phase 2 emitter-compatible scope.

The source graph can contain a movement/contraction/map adjacency without the
accelerator supporting those exact operations or the current writer emitting
them. Neither family adjacency nor a synthetic program is a placement proof.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.development.targetgen.corpus.model_builders import scope_chain_region_ops
from merlin.target.spec.software import admit_operation

# ---- performance_scope -------------------------------------------------------------------------


def _dtype(row: dict) -> str | None:
    return next((item.get("dtype") for item in row.get("results") or [] if item.get("dtype")), None)


def _operand_dtype(instance: dict, row: dict) -> str | None:
    edge = next(
        (edge for edge in instance.get("edges") or [] if edge.get("consumer_operation_id") == row.get("operation_id")),
        None,
    )
    if edge is not None:
        return edge.get("dtype")
    return next((item.get("dtype") for item in row.get("operands") or [] if item.get("dtype")), None)


def derive_performance_scope(scope: dict, software_spec: dict) -> dict:
    """Classify exact typed chains for the *current* standalone PN emitter.

    A required Phase 2 cohort also needs exact source-body correspondence,
    which the current synthetic PN writer does not supply. Matching operation
    names and dtypes is not sufficient. Unknown constraints remain unresolved;
    explicit SW refusals remain Phase 1 source demand, not invented accelerator
    performance obligations.
    """
    typed = scope.get("typed_required_instances") or {}
    if typed.get("schema") != "merlin.phase0.typed_scope_instances.v1":
        raise ValueError("performance scope needs exact typed source instances")
    excluded, unresolved = [], []
    for instance in typed.get("instances") or []:
        regions = instance.get("regions") or []
        signature = instance.get("signature")
        families = [row.get("semantic_family") for row in regions]
        if (
            len(regions) < 2
            or any(not isinstance(family, str) or not family for family in families)
            or signature != " -> ".join(families)
        ):
            raise ValueError("typed scope instance has inconsistent region signature")
        try:
            emitted_ops = scope_chain_region_ops(families)
        except ValueError:
            unresolved.append(
                {
                    "instance_id": instance["instance_id"],
                    "signature": signature,
                    "reason": "no current standalone scope-chain emitter accepts this family sequence",
                    "status": "emitter_unimplemented",
                }
            )
            continue
        accumulator_dtype = _dtype(regions[1])
        decisions = []
        for row in regions:
            results = row.get("results") or []
            shape = results[0].get("shape") if results else None
            signature_observed = {
                "family": row["semantic_family"],
                "operand_dtype": _operand_dtype(instance, row),
                "accum_dtype": accumulator_dtype,
                "rank": len(shape) if isinstance(shape, list) else None,
            }
            decision = admit_operation(software_spec, row["source_op"], signature_observed, "accelerator")
            decisions.append({"operation_id": row["operation_id"], "source_op": row["source_op"], **decision})
        row = {
            "instance_id": instance["instance_id"],
            "application": instance["application"],
            "capture_sha256": instance["capture_sha256"],
            "signature": signature,
            "source_operations": [region["source_op"] for region in regions],
            "emitted_operations": emitted_ops,
            "source_edge_dtypes": [edge["dtype"] for edge in instance.get("edges") or []],
            "software_decisions": decisions,
        }
        refused = [decision for decision in decisions if decision["status"] == "unsupported"]
        if refused:
            excluded.append(
                {
                    **row,
                    "status": "software_refused",
                    "reason": "; ".join(f"{decision['source_op']}: {decision['reason']}" for decision in refused),
                }
            )
            continue
        # The current emitter keeps every post-contraction map in accumulator
        # precision. It cannot stand in for a captured cast or floating map.
        same_operations = row["source_operations"] == emitted_ops
        same_precision = accumulator_dtype is not None and all(
            _operand_dtype(instance, region) == accumulator_dtype and _dtype(region) == accumulator_dtype
            for region in regions[2:]
        )
        if not same_operations or not same_precision:
            unresolved.append(
                {
                    **row,
                    "status": "emitter_unimplemented",
                    "reason": "current emitted operations or map precision differ from exact captured regions",
                }
            )
        elif any(decision["status"] != "admitted" for decision in decisions):
            unresolved.append(
                {
                    **row,
                    "status": "software_unreviewed",
                    "reason": "one or more exact source operations has unresolved SW constraints or review",
                }
            )
        else:
            # Matching op names and dtypes cannot prove that a captured map
            # computes this writer's hard-coded scalar `+1`, or that the
            # movement has the same permutation. The current PN writer is
            # synthetic and has no source-body correspondence witness.
            unresolved.append(
                {
                    **row,
                    "status": "emitter_unimplemented",
                    "reason": "standalone scope emitter has no exact source-body semantic correspondence",
                }
            )
    required: list[dict] = []
    result = {
        "schema": "merlin.phase0.performance_scope.v1",
        "status": "unresolved" if unresolved else "ready" if required else "no_eligible_chain",
        "required": required,
        "excluded": excluded,
        "unresolved": unresolved,
        "qualification": (
            "exact source/SW/emitter preselection only; compiler placement, target execution and "
            "performance measurement remain separate gates"
        ),
    }
    validate_performance_scope({**scope, "performance": result})
    return result


def validate_performance_scope(scope: dict) -> dict:
    """Reject a Phase 2 selection that silently drops exact source instances."""
    typed, performance = scope.get("typed_required_instances") or {}, scope.get("performance") or {}
    if typed.get("schema") != "merlin.phase0.typed_scope_instances.v1":
        raise ValueError("Phase 2 scope lacks exact typed source instances")
    if performance.get("schema") != "merlin.phase0.performance_scope.v1":
        raise ValueError("Phase 2 scope lacks exact SW/emitter-derived selection")
    instances = typed.get("instances")
    if not isinstance(instances, list):
        raise ValueError("typed source instance inventory is invalid")
    index = {row.get("instance_id"): row for row in instances if isinstance(row, dict)}
    if len(index) != len(instances) or None in index:
        raise ValueError("typed source instance IDs are missing or repeated")
    raw = scope.get("required") or []
    if (
        not isinstance(raw, list)
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("signature"), str)
            or type(row.get("occurrences")) is not int
            or row["occurrences"] < 1
            for row in raw
        )
        or len({row["signature"] for row in raw}) != len(raw)
    ):
        raise ValueError("raw source scope requirement is invalid or duplicated")
    if Counter(row.get("signature") for row in instances) != Counter(
        {row["signature"]: row["occurrences"] for row in raw}
    ):
        raise ValueError("typed source instances differ from raw scope demand")
    selected: set[str] = set()

    def take(identity: str, signature: str) -> None:
        if identity not in index or index[identity].get("signature") != signature or identity in selected:
            raise ValueError("Phase 2 scope loses, duplicates or mislabels a typed source instance")
        selected.add(identity)

    for key in ("excluded", "unresolved"):
        rows = performance.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"Phase 2 scope {key} inventory is invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Phase 2 scope {key} row is invalid")
            take(row.get("instance_id"), row.get("signature"))
    required = performance.get("required")
    if not isinstance(required, list):
        raise ValueError("Phase 2 required scope inventory is invalid")
    signatures = set()
    for row in required:
        if not isinstance(row, dict) or row.get("signature") in signatures:
            raise ValueError("Phase 2 required scope has duplicate or invalid signatures")
        signature = row["signature"]
        signatures.add(signature)
        members = row.get("instance_ids")
        if (
            not isinstance(members, list)
            or type(row.get("occurrences")) is not int
            or len(members) != row["occurrences"]
        ):
            raise ValueError("Phase 2 required scope lacks exact source-instance witnesses")
        for identity in members:
            take(identity, signature)
    if selected != set(index):
        raise ValueError("Phase 2 scope leaves typed source instances unclassified")
    expected_status = "unresolved" if performance["unresolved"] else "ready" if required else "no_eligible_chain"
    if performance.get("status") != expected_status:
        raise ValueError("Phase 2 scope status differs from exact classified source instances")
    return performance


# ---- typed_scope -------------------------------------------------------------------------------
# Exact typed source instances behind derived Phase 0 adjacency requirements.
#
# These records preserve what the selected capture actually says. They do not
# assign host/device placement, assert fusion, or qualify a Phase 2 performance
# cohort. The raw ``scope.required`` axis remains a separate source-demand census.


def typed_required_instances(scope: dict, inventory: dict) -> dict:
    """Bind every required chain occurrence to raw MLIR operation IDs and SSA types."""
    from merlin.compiler.ir import mlir_query as mq
    from merlin.development.targetgen.applications.operation_plan import chains
    from merlin.development.targetgen.conformance.model_coverage import region_ops

    required = scope.get("required") or []
    wanted = {row["signature"]: row["occurrences"] for row in required}
    if len(wanted) != len(required) or any(type(count) is not int or count < 1 for count in wanted.values()):
        raise ValueError("scope requirement has duplicate or invalid occurrence counts")
    instances = []
    observed: Counter[str] = Counter()
    for label, application in sorted((inventory.get("applications") or {}).items()):
        path = Path(application["capture_source_path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"selected scope capture is missing or symlinked: {label}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != application["capture_sha256"]:
            raise ValueError(f"capture bytes changed while deriving typed scope: {label}")
        graph = (application.get("operation_graph") or {}).get("capture_graph") or {}
        if graph.get("mlir_sha256") != digest:
            raise ValueError(f"source graph differs from selected capture bytes: {label}")
        module = mq.parse(raw.decode("utf-8"))
        operations = list(mq.walk(module))
        graph_rows = graph.get("operations") or []
        if len(operations) != len(graph_rows):
            raise ValueError(f"source operation graph membership changed: {label}")
        ordinals = {id(op): index for index, op in enumerate(operations)}
        regions = region_ops(module)
        for chain in chains(module, max_length=64):
            if chain.signature not in wanted:
                continue
            observed[chain.signature] += 1
            selected = []
            for region_index, family in zip(chain.indices, chain.families, strict=True):
                op = regions[region_index]
                ordinal = ordinals[id(op)]
                source = graph_rows[ordinal]
                operation_id = f"mlir:{digest}:{ordinal}"
                if (
                    source.get("ordinal") != ordinal
                    or source.get("operation_id") != operation_id
                    or source.get("mlir_operation") != mq.op_name(op)
                ):
                    raise ValueError(f"source operation identity changed: {label}/{ordinal}")
                provenance = source.get("provenance") or {}
                selected.append(
                    {
                        "region_index": region_index,
                        "operation_id": operation_id,
                        "ordinal": ordinal,
                        "mlir_operation": source["mlir_operation"],
                        "source_op": provenance.get("prov.op") or source["mlir_operation"],
                        "semantic_family": family,
                        "source_node_ids": source.get("source_node_ids") or [],
                        "origin_node_ids": source.get("origin_node_ids") or [],
                        "operands": source.get("operands") or [],
                        "results": source.get("results") or [],
                    }
                )
            edges = []
            for producer, consumer in zip(selected, selected[1:]):
                crossing = [
                    operand
                    for operand in consumer["operands"]
                    if operand.get("producer_operation_id") == producer["operation_id"]
                ]
                if not crossing:
                    raise ValueError(f"source chain lost its typed SSA edge: {label}/{chain.signature}")
                edges.extend(
                    {
                        "producer_operation_id": producer["operation_id"],
                        "consumer_operation_id": consumer["operation_id"],
                        "value_id": operand["value_id"],
                        "operand_index": operand["index"],
                        "type": operand["type"],
                        "shape": operand["shape"],
                        "dtype": operand["dtype"],
                    }
                    for operand in crossing
                )
            operation_ids = [row["operation_id"] for row in selected]
            identity = json.dumps({"capture_sha256": digest, "operation_ids": operation_ids}, sort_keys=True)
            instances.append(
                {
                    "instance_id": hashlib.sha256(identity.encode()).hexdigest(),
                    "application": label,
                    "capture_sha256": digest,
                    "capture_receipt_sha256": (application.get("capture_receipt") or {}).get("receipt_sha256"),
                    "signature": chain.signature,
                    "operation_ids": operation_ids,
                    "regions": selected,
                    "edges": edges,
                }
            )
    if dict(observed) != wanted:
        raise ValueError(f"typed scope occurrences differ from raw requirement: {dict(observed)} != {wanted}")
    return {
        "schema": "merlin.phase0.typed_scope_instances.v1",
        "status": "typed_source_only",
        "instances": instances,
        "qualification": "exact source adjacency and types; no host/device placement, compiler lowering or execution",
    }


# ---- instruction_roles -------------------------------------------------------------------------
# Derived Phase 0 stage: the target's instruction-ROLE taxonomy and the experiment's declared policy.
#
# A rule such as "no hardware loop-descriptor instructions" is a statement in the closed role vocabulary
# (:mod:`merlin.analysis.observation.instruction_roles`), never a list of instruction names: which instructions carry a role is a
# fact about the target, derived from its own RTL-derived encoding table through its declared compute
# endpoints (:func:`merlin.analysis.programs.task_instructions.target_instruction_facts`). Phase 0 derives
# that taxonomy once, resolves the experiment's ``prohibited_instruction_roles`` against it, and records
# the result where Phase 1 grading and Phase 2 measurement read it. Enforcement -- scanning the WHOLE
# linked program for a prohibited instruction -- is a separate owner; this module only declares and
# derives, and fails closed: an unknown role is refused, an underivable taxonomy is ``UNKNOWN``.

TAXONOMY_SCHEMA = "merlin.phase0.instruction_roles.v1"
POLICY_SCHEMA = "merlin.phase0.instruction_policy.v1"
UNKNOWN = "UNKNOWN"


def validate_roles(roles: object) -> list[str]:
    """Declared role names, checked against the closed vocabulary; duplicates and unknowns refused."""
    from merlin.analysis.observation.instruction_roles import ROLES

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
    from merlin.analysis.observation.compute_endpoints import endpoints_for
    from merlin.analysis.observation.instruction_roles import ROLES
    from merlin.analysis.observation.rocc_commands import funct_table_for, policy_instruction_declarations
    from merlin.analysis.programs.task_instructions import declared_instruction_set

    # The same composition ``target_instruction_facts`` makes, without requiring the target's runtime
    # backend: the taxonomy needs only the RTL-derived name table and the declared endpoint roles. The
    # ISA constants (the custom opcode) are the scanner's business and are recorded when available.
    try:
        names, provenance = policy_instruction_declarations(funct_table_for(target))
        endpoints = endpoints_for(target)
        if not endpoints:
            # No endpoint binds a role to any of this target's instructions, so every instruction would
            # come back role-less and a prohibited role would match nothing -- a policy that cannot fail,
            # reading as derived. Unknown, with the reason, instead.
            from merlin.analysis.observation.compute_endpoints import spec_path

            raise LookupError(f"no compute endpoint in {spec_path()} declares instruction roles for {target!r}")
        facts: dict[str, Any] = {
            "target": target,
            "instruction_names": names,
            "instruction_provenance": provenance,
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
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in sorted(ROLES)}
    unroled: list[dict[str, Any]] = []
    for row in declared.get("instructions") or ():
        instruction = {"selector": str(row["funct"]), "name": str(row["name"])}
        if "provenance" in row:
            instruction["provenance"] = copy.deepcopy(row["provenance"])
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
        "derivation": "merlin.analysis.programs.task_instructions.target_instruction_facts",
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
