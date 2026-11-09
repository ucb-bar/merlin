"""Separate raw captured scope demand from Phase 2 emitter-compatible scope.

The source graph can contain a movement/contraction/map adjacency without the
accelerator supporting those exact operations or the current writer emitting
them. Neither family adjacency nor a synthetic program is a placement proof.
"""

from __future__ import annotations

from collections import Counter

from merlin.targetgen.corpus_spec import scope_chain_region_ops
from merlin.targetgen.software_spec import admit_operation


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
            unresolved.append({
                "instance_id": instance["instance_id"], "signature": signature,
                "reason": "no current standalone scope-chain emitter accepts this family sequence",
                "status": "emitter_unimplemented",
            })
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
            excluded.append({**row, "status": "software_refused", "reason": "; ".join(
                f"{decision['source_op']}: {decision['reason']}" for decision in refused
            )})
            continue
        # The current emitter keeps every post-contraction map in accumulator
        # precision. It cannot stand in for a captured cast or floating map.
        same_operations = row["source_operations"] == emitted_ops
        same_precision = (
            accumulator_dtype is not None
            and all(_operand_dtype(instance, region) == accumulator_dtype and _dtype(region) == accumulator_dtype
                    for region in regions[2:])
        )
        if not same_operations or not same_precision:
            unresolved.append({
                **row, "status": "emitter_unimplemented",
                "reason": "current emitted operations or map precision differ from exact captured regions",
            })
        elif any(decision["status"] != "admitted" for decision in decisions):
            unresolved.append({
                **row, "status": "software_unreviewed",
                "reason": "one or more exact source operations has unresolved SW constraints or review",
            })
        else:
            # Matching op names and dtypes cannot prove that a captured map
            # computes this writer's hard-coded scalar `+1`, or that the
            # movement has the same permutation. The current PN writer is
            # synthetic and has no source-body correspondence witness.
            unresolved.append({
                **row, "status": "emitter_unimplemented",
                "reason": "standalone scope emitter has no exact source-body semantic correspondence",
            })
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
        or any(not isinstance(row, dict) or not isinstance(row.get("signature"), str)
               or type(row.get("occurrences")) is not int or row["occurrences"] < 1 for row in raw)
        or len({row["signature"] for row in raw}) != len(raw)
    ):
        raise ValueError("raw source scope requirement is invalid or duplicated")
    if Counter(row.get("signature") for row in instances) != Counter({
        row["signature"]: row["occurrences"] for row in raw
    }):
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
