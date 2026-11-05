"""Project public source operation obligations onto available capsule writers."""

from __future__ import annotations


def _application_operation_plan(demands: dict | None, *, exact_entries: list[dict] | None = None) -> dict:
    """Plan operation obligations without equating family cells to frontend coverage.

    The selected spec carries a compact operation projection, not the complete capture inventory.
    Its groups identify missing writers and host seams. Exact integer-matmul candidates are admitted
    only when digest-bound detailed signatures account for every source ordinal. Emission and compiler
    execution remain explicit, unverified obligations.
    """

    from .corpus_synth import (
        SynthesisError,
        _cost,
        _exact_int_mm_group_candidates,
        _op_family_map,
        _writer_for,
        available_ops,
    )

    if not isinstance(demands, dict):
        return {
            "status": "not_recorded",
            "coverage_status": "unverified",
            "reason": "this conformance spec predates the application operation inventory",
            "obligations": [],
        }

    groups = demands.get("operation_groups")
    base = {
        "status": "unverified",
        "coverage_status": "unverified",
        "inventory_status": demands.get("status"),
        "full_inventory_sha256": demands.get("full_inventory_sha256"),
        "sidecar": demands.get("sidecar"),
        "n_operations": demands.get("n_operations", 0),
        "basis": (
            "operation groups are planning hints from declared application captures; a family/dtype "
            "cell or a writer candidate does not prove frontend lowering, capsule coverage, or execution"
        ),
    }
    if demands.get("status") == "not_declared":
        return {**base, "status": "not_declared", "obligations": []}
    if not isinstance(groups, list):
        return {
            **base,
            "status": "blocked",
            "reason": "compact spec has no operation groups; inspect the digest-checked full inventory sidecar",
            "missing_mapping": ["application signature -> generic writer", "application signature -> exact capsule"],
            "obligations": [],
        }

    from merlin.targetgen.conformance import capsule_dtype

    pool = available_ops()
    families = _op_family_map()
    exact_entries = list(exact_entries or ())
    obligations: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            raise SynthesisError("application_demands.operation_groups contains a non-object row")
        operation = str(group.get("operation") or "")
        mlir_operation = str(group.get("mlir_operation") or "")
        family = str(group.get("semantic_family") or "")
        fmt = group.get("operand_format")
        dtype = capsule_dtype(str(fmt)) if fmt else None
        disposition = str(group.get("disposition") or "unclassified")
        # A captured frontend op is not equivalent to the linalg op left after frontend lowering.
        # Only an exact vocabulary name is an operation-specific writer candidate. A named linalg op
        # may use its final segment when no frontend/provenance operation survives in the projection.
        writer_op = operation if operation in pool else None
        if writer_op is None and operation == mlir_operation and mlir_operation.startswith("linalg."):
            leaf = mlir_operation.rpartition(".")[2]
            writer_op = leaf if leaf in pool else None
        writer = _writer_for({"op": writer_op, "operand_dtype": dtype}) if writer_op and dtype else None
        family_writers = sorted(
            (
                op
                for op in pool
                if families.get(op) == family and dtype and _writer_for({"op": op, "operand_dtype": dtype})
            ),
            key=_cost,
        )
        row = {
            "operation": operation,
            "mlir_operation": mlir_operation,
            "semantic_family": family or None,
            "operand_format": fmt,
            "shape_class": group.get("shape_class"),
            "disposition": disposition,
            "count": int(group.get("count") or 0),
            "sources": group.get("sources") or [],
            "coverage_status": "unverified",
        }
        if disposition == "hardware_admitted":
            row["lane"] = "accelerator"
            exact_candidates = (
                _exact_int_mm_group_candidates(
                    group,
                    exact_entries,
                    demands.get("full_inventory_sha256"),
                    tuple(sorted((demands.get("applications") or {}).keys())),
                )
                if (
                    operation == "aten._int_mm.default"
                    and mlir_operation == "linalg.generic"
                    and family == "contraction"
                    and fmt == "int8"
                    and group.get("shape_class") == "contraction:rank_3"
                )
                else []
            )
            if exact_candidates:
                row.update(
                    {
                        "status": "candidate_unverified",
                        "obligation": "exact_signature_capsule_and_compile_route",
                        "writer_candidate": {"op": "matmul", "source": "pytorch", "capture_op": "int_matmul"},
                        "capsule_candidates": exact_candidates,
                        "mapped_occurrences": row["count"],
                        "missing_mapping": [
                            "capture and emit every exact PyTorch integer-matmul capsule candidate",
                            "verify each emitted linalg signature against its digest-bound source",
                            "generated capsule -> executable whole-model compiler route",
                        ],
                    }
                )
            elif writer:
                row.update(
                    {
                        "status": "unverified",
                        "obligation": "exact_signature_capsule_and_compile_route",
                        "writer_candidate": {"op": writer_op, "source": writer},
                        "missing_mapping": [
                            "captured frontend signature -> writer semantics and operand ABI",
                            "writer + exact dtype/shape/layout/quantization -> generated capsule",
                            "generated capsule -> executable whole-model compiler route",
                        ],
                    }
                )
            else:
                row.update(
                    {
                        "status": "refused",
                        "obligation": "no_exact_generic_writer",
                        "writer_candidate": None,
                        "family_writer_candidates_not_equivalent": family_writers,
                        "action": (
                            "add an operation-specific generic writer or a reviewed frontend decomposition "
                            "at this dtype; a family-level capsule cannot stand in for this operation"
                        ),
                    }
                )
        elif disposition == "host_required":
            row.update(
                {
                    "lane": "host",
                    "status": "required_unverified",
                    "obligation": "host_lowering_and_boundary",
                    "action": "retain a host implementation and verify value transfer at the accelerator/host seam",
                }
            )
        elif disposition == "support_required":
            row.update(
                {
                    "lane": "support",
                    "status": "required_unverified",
                    "obligation": "graph_support_lowering",
                    "action": "lower this graph-plumbing operation; it is not a separate compute capsule",
                }
            )
        elif disposition == "unclassified":
            row.update(
                {
                    "lane": "unresolved",
                    "status": "blocked_unclassified",
                    "obligation": "resolve_operation_semantics",
                    "action": "resolve the operation, callee, shape or format before assigning a hardware or host lane",
                }
            )
        else:
            # Structural containers and nested linalg-body components are counted by the inventory,
            # but are not independent capsule demands.
            continue
        obligations.append(row)

    obligations.sort(
        key=lambda row: (
            row["disposition"],
            row["operation"],
            row["mlir_operation"],
            str(row["operand_format"]),
            str(row["shape_class"]),
        )
    )
    excluded = sum(
        int((app.get("counts") or {}).get(kind) or 0)
        for app in (demands.get("applications") or {}).values()
        for kind in ("structural", "component")
    )
    planned = sum(row["count"] for row in obligations)
    expected = int(demands.get("n_operations") or 0) - excluded
    blocked = [row for row in obligations if row["status"] in {"refused", "blocked_unclassified"}]
    return {
        **base,
        "status": (
            "blocked"
            if demands.get("status") != "inventoried" or expected <= 0 or planned != expected or blocked
            else "obligations_pending"
        ),
        "blocked_operations": sum(row["count"] for row in blocked),
        "obligations": obligations,
        "n_planned_operations": planned,
        "n_non_demand_operations": excluded,
        "projection_status": "complete" if planned == expected else "incomplete",
        "projection_note": (
            "every non-structural parsed operation appears in an obligation group"
            if planned == expected
            else f"operation groups account for {planned} of {expected} non-structural parsed operations; "
            "inspect the full inventory before relying on this plan"
        ),
        "counts": {
            key: sum(row["count"] for row in obligations if row["obligation"] == key)
            for key in sorted({row["obligation"] for row in obligations})
        },
        "missing_mapping": ["digest-checked full capture signature -> exact generated capsule and compiler route"],
    }


def _cost(op: str) -> tuple:
    """Rank candidate ops for a family: fewest operands, then smallest default footprint, then name.

    A derived tie-break rather than a preference list, so adding an op cannot silently change which
    capsule a cell gets unless that op is genuinely cheaper.
    """
    from merlin.targetgen.corpus_spec import BUILDERS

    # Operand ROLES, not tensor count: `linear` carries an optional bias role that `matmul` does not,
    # so the bare contraction is the cheaper way to evidence a contraction cell.
    operands = {
        "matmul": 2,
        "linear": 3,
        "movement": 1,
        "rmsnorm": 1,
        "softmax": 1,
        "layernorm": 1,
        "add": 2,
        "reduce_sum": 1,
        "gelu": 1,
        "silu": 1,
        "bias_add": 2,
    }
    return (operands.get(op, 3), 0 if op in BUILDERS else 1, op)
