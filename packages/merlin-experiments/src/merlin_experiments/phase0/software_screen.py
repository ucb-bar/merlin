"""Intersect backend candidates with authored SW intent, without certifying either."""

from __future__ import annotations

import copy

from merlin.targetgen.interface_observations import epilogue_carrier
from merlin.targetgen.semantic_families import from_op
from merlin.targetgen.software_spec import admit_operation


def _screen_scope_chain(spec: dict, entry: dict, capsule: dict | None) -> dict:
    """Admit an emitted region chain one operation at a time, not by container name.

    The builder emits separate movement, contraction and map regions. Adjacency
    does not establish fused placement: each region needs an accelerator SW
    declaration of its own. The pre-emission entry lacks an operation inventory.
    """
    attrs = ((capsule or {}).get("operation") or {}).get("attributes") or {}
    families, operations = attrs.get("scope_families"), attrs.get("scope_region_ops")
    if capsule is None or not isinstance(families, list) or not isinstance(operations, list):
        return {
            "status": "unknown",
            "constraints_status": "unknown",
            "decisions": [],
            "reason": "scope chain requires the emitted per-region operation inventory",
        }
    selected = entry.get("scope_families")
    if (
        len(families) < 3
        or len(families) != len(operations)
        or families != selected
        or attrs.get("scope_signature") != " -> ".join(families)
        or any(from_op(op) != family for op, family in zip(operations, families, strict=True))
    ):
        return {
            "status": "unsupported",
            "constraints_status": "refused",
            "decisions": [],
            "reason": "emitted scope regions differ from the selected operation-family chain",
        }
    operands = [row for row in capsule.get("inputs") or [] if row.get("role") in {"input", "weight"}]
    operand_dtypes = {row.get("dtype") for row in operands}
    operand_dtype = next(iter(operand_dtypes)) if len(operand_dtypes) == 1 else None
    accumulator_dtype = attrs.get("output_dtype")
    dimensions = {axis: attrs[axis] for axis in ("M", "K", "N") if type(attrs.get(axis)) is int}
    decisions = []
    for index, (op, family) in enumerate(zip(operations, families, strict=True)):
        signature = {
            "family": family,
            "operand_dtype": operand_dtype if family != "elementwise_map" else accumulator_dtype,
            "accum_dtype": accumulator_dtype,
            "rank": 2,
            "dimensions": dimensions,
        }
        decision = admit_operation(spec, op, signature, "accelerator")
        decisions.append({"role": "region", "index": index, "family": family, **decision})
    refused = [row for row in decisions if row["status"] == "unsupported"]
    unknown = [row for row in decisions if row["status"] == "unknown"]
    return {
        "status": "unsupported" if refused else "unknown" if unknown else "admitted",
        "constraints_status": "refused"
        if refused
        else "unknown"
        if any(row.get("constraints_status") != "matched" for row in decisions)
        else "matched",
        "reason": "; ".join(f"region {row['index']} ({row['op']}): {row['reason']}" for row in refused or unknown)
        or "all emitted region SW constraints match",
        "decisions": decisions,
    }


def screen_entry(
    spec: dict,
    entry: dict,
    *,
    defaults: dict | None = None,
    capsule: dict | None = None,
    host_capabilities: dict | None = None,
) -> dict:
    """Screen a selected carrier and every explicitly carried epilogue separately.

    Precision comes from the selected entry/binding or emitted capsule, not the
    backend's full format list. Missing layout/placement observations stay unknown.
    Whole-program containers need their per-operation inventory, not a model-name
    admission invented here.
    """
    defaults = defaults or {}
    if capsule is None and entry.get("component_coverage"):
        return {
            "status": "unknown",
            "constraints_status": "unknown",
            "decisions": [],
            "reason": "component obligation requires the concrete emitted per-operation program screen",
        }
    operation = (capsule or {}).get("operation") or {}
    op = operation.get("op") or entry.get("op", "unknown")
    attrs = operation.get("attributes") or {}
    kind = (capsule or {}).get("kind") or entry.get("kind")
    if kind == "model_slice" and op == "scope_chain":
        return _screen_scope_chain(spec, entry, capsule)
    performance = (capsule or {}).get("performance") or entry.get("performance") or {}
    knobs = (performance.get("emitter") or {}).get("knobs") or {}
    declared_regions = knobs.get("accelerator_regions") if knobs.get("operation") == op else None
    observed_regions = attrs.get("accelerator_contractions")
    # A registered emitter's multi-region program shape is not a single unknown
    # operation. A model_slice label or unfamiliar name alone is insufficient:
    # ordinary slices still undergo the same carrier and epilogue refusals below.
    compound_program = (
        kind == "model_slice"
        and from_op(op) is None
        and any(type(count) is int and count > 1 for count in (declared_regions, observed_regions))
    )
    if compound_program:
        from merlin.targetgen.corpus_spec import BUILDERS

        compound_program = op in BUILDERS
    if (
        kind == "model"
        or op in {"model", "micro_model"}
        or entry.get("micro_model") is True
        or entry.get("materialized_capture")
        or compound_program
    ):
        return {
            "status": "unknown",
            "constraints_status": "unknown",
            "decisions": [],
            "reason": "whole-program container requires the saved per-operation admission inventory",
        }
    signature = {
        "operand_dtype": entry.get("operand_dtype") or entry.get("capture_dtype") or defaults.get("operand_dtype"),
        "accum_dtype": entry.get("accum_dtype") or defaults.get("accumulator_dtype"),
        "dimensions": {key: value for key, value in entry.items() if type(value) is int},
    }
    signature.update(entry.get("operation_signature") or {})
    # An entry's layout request is explicit; its presence is not execution proof.
    if entry.get("layout") is not None:
        signature["layout"] = entry["layout"]
    if capsule is not None:
        operands = [row for row in capsule.get("inputs") or [] if row.get("role") == "input"]
        dtypes = {row.get("dtype") for row in operands}
        if len(dtypes) == 1:
            signature["operand_dtype"] = dtypes.pop()
        if attrs.get("output_dtype"):
            signature["readout_dtype"] = attrs["output_dtype"]
    placement = entry.get("placement", "unknown")
    generalization = (capsule or {}).get("semantic") or entry.get("generalization") or {}
    host_only = generalization.get("must_accelerate") is False and generalization.get("eligible") is False
    if host_only:
        from merlin.targetgen.host_capabilities import admit_host_operation

        # A probe built from an observed host operation names the frontend operator it reproduces;
        # declarations that name exact operators can then recognize it before it is written.
        decision = admit_host_operation(
            host_capabilities, {"mlir_operation": op, "frontend_op": entry.get("frontend_op")}, signature
        )
        if decision["status"] == "unsupported" and from_op(op) == "movement":
            # Data movement is a support lowering, not a host compute operation, and the written
            # program's own admission (program_admission.screen_written) reads it that way. Before
            # writing, the absence of a host COMPUTE declaration for it decides nothing.
            decision = {
                **decision,
                "status": "unknown",
                "reason": "data movement: decided by the written program's admission, not a host compute declaration",
            }
        return {
            **decision,
            "constraints_status": "refused" if decision["status"] == "unsupported" else "unknown",
            "scope": "explicit host-only probe; no accelerator support claim",
            "decisions": [{"role": "host", **decision}],
        }
    carrier = admit_operation(spec, op, signature, placement)
    decisions = [{"role": "carrier", **carrier}]
    stages = attrs.get("epilogue", entry.get("epilogue") or [])
    family = from_op(op)
    producer = epilogue_carrier(op, family)
    for stage in stages:
        # Pre-emission screening must project the same carrier and data type as
        # the command-buffer observation. Missing readout type stays unknown;
        # an accumulator type is not evidence of an elementwise stage's input.
        stage_dtype = (
            signature.get("operand_dtype")
            if family == "contraction"
            else attrs.get("output_dtype") or entry.get("readout_dtype") or signature.get("operand_dtype")
        )
        stage_signature = {
            **signature,
            "operand_dtype": stage_dtype,
            "epilogues": [stage],
            "composed_with": [producer] if producer is not None else None,
        }
        # An epilogue is explicitly part of the accelerator carrier, not a
        # standalone host operation with a coincidentally matching family.
        decision = admit_operation(spec, stage, stage_signature, "fused_accelerator")
        decisions.append({"role": "epilogue", "stage": stage, **decision})
    refused = [row for row in decisions if row["status"] == "unsupported"]
    unknown = [row for row in decisions if row["status"] == "unknown"]
    status = "unsupported" if refused else "unknown" if unknown else "admitted"
    return {
        "status": status,
        "constraints_status": "refused"
        if refused
        else "unknown"
        if any(row.get("constraints_status") != "matched" for row in decisions)
        else "matched",
        "reason": "; ".join(row["reason"] for row in refused or unknown) or "all selected SW constraints match",
        "decisions": decisions,
    }


def diagnostic_entry(entry: dict, decision: dict) -> dict:
    """Retain an explicitly refused probe, but outside functional/performance roles."""
    result = copy.deepcopy(entry)
    result["cat"] = "_diagnostic"
    result["software_screen"] = copy.deepcopy(decision)
    prefix = "Backend probe explicitly refused by the selected SW spec; diagnostic only. "
    source_reference = str(entry.get("source_reference") or "")
    result["source_reference"] = source_reference if source_reference.startswith(prefix) else prefix + source_reference
    return result


def intersect_requirement(requirement: dict, spec: dict, contract: dict) -> dict:
    """Remove explicit SW refusals from required device cells, never from the census.

    Backend-declared capabilities remain inspectable in diagnostic obligations.
    Unknown typed constraints remain required obligations rather than being
    silently omitted or treated as reviewed support.
    """
    from merlin.targetgen.eligibility import capability_map_from_contract

    result = copy.deepcopy(requirement)
    capabilities = capability_map_from_contract(contract) or {}
    defaults = spec["numerical_semantics"]
    rejected, retained = [], []
    for cell in result.get("cells") or []:
        family = cell.get("family")
        capability = capabilities.get(family)
        composed = list(getattr(capability, "composed_with", ()) or ())
        signature = {"family": family, "operand_dtype": cell.get("dtype"), "accum_dtype": defaults["accumulator_dtype"]}
        if composed:
            signature["composed_with"] = composed
        # Exact-op declarations cannot be disproved by a hypothetical family
        # label; the real synthesized carrier is screened independently below.
        exact = [op for row in spec["operations"] for op in row.get("ops", []) if from_op(op) == family]
        decisions = [
            admit_operation(spec, op, signature, "fused_accelerator" if composed else "accelerator")
            for op in exact or [str(family)]
        ]
        if all(row["status"] == "unsupported" for row in decisions):
            rejected.append({"kind": "cell", "backend_requirement": cell, "software_decisions": decisions})
        else:
            retained.append(cell)
    result["cells"] = retained
    epilogue = result.get("epilogue") or {}
    required = []
    for row in epilogue.get("required") or []:
        decision = screen_entry(spec, {"op": "matmul", "epilogue": [row["stage"]]}, defaults=defaults)
        if decision["status"] == "unsupported":
            rejected.append({"kind": "epilogue", "backend_requirement": row, "software_decision": decision})
        else:
            required.append(row)
    if "epilogue" in result:
        result["epilogue"]["required"] = required
    result["software_intersection"] = {
        "status": "diagnostic",
        "rejected_backend_capabilities": rejected,
        "qualification": "authored constraint screen; retained unknowns are not certified support",
    }
    return result
