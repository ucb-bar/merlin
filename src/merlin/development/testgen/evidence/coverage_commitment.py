"""Bind application obligations to an actual admitted cohort, not its source pool.

This is a pre-compiler completeness check. It does not execute a capsule, certify
hardware, or turn declared numerical semantics into a compiler correctness claim.
All observations travel with the corpus and are independently snapshot-bound.
"""

from __future__ import annotations

import copy
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_bytes
from merlin.development.testgen.evidence.witness_basis import build_witness_basis

# ---- coverage_commitment -----------------------------------------------------------------------


def build_commitment(
    inputs: dict | None, capsules: list[dict], *, phase: str = "phase1", conformance_coverage: dict | None = None
) -> dict:
    """Match every source obligation against only these byte-identified capsules.

    Capsule observations must be produced by ``observe_cohort`` or already bound
    by a containing snapshot. Typed signature presence is pre-compiler coverage,
    never semantic equivalence or a successful target execution.
    """
    from merlin.development.testgen.evidence.coverage_obligations import (
        INPUT_SCHEMA,
        SCHEMA,
        _conformance_blockers,
        _digest,
        _framework_versions_complete,
        _graph_totality,
        _placement,
        _source_complete,
        attestation_failure,
        signature_identity,
    )

    if phase not in {"phase1", "phase2"}:
        raise ValueError("coverage commitment phase must be phase1 or phase2")
    blockers = []
    rows = sorted(copy.deepcopy(capsules), key=lambda item: item["name"])
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("admitted coverage cohort contains duplicate capsule identities")
    if not rows:
        blockers.append({"component": "cohort", "reason": "admitted cohort is empty"})
    for row in rows:
        if not is_sha256(row.get("sha256")) or not is_sha256(row.get("program_sha256")):
            blockers.append(
                {"component": "capsule", "capsule": row["name"], "reason": "capsule byte commitments are absent"}
            )
        if row.get("status") != "inventoried":
            blockers.append(
                {"component": "capsule", "capsule": row["name"], "reason": "capsule program is not inventoried"}
            )
    document = inputs or {}
    if document.get("schema") != INPUT_SCHEMA:
        blockers.append({"component": "inputs", "reason": "new manifest-bound coverage inputs are absent"})
    spec = document.get("software_spec") or {}
    if spec.get("status") != "reviewed":
        blockers.append({"component": "software_spec", "reason": "software declarations are not reviewed"})
    if document.get("schema") == INPUT_SCHEMA:
        from merlin.development.targetgen.spec_fact_drift import blockers as drift_blockers

        # A reviewed spec can still forbid what the facts establish, or claim what they refuse.
        blockers.extend(drift_blockers(document.get("spec_fact_drift")))
    accounting = document.get("accounting") or {}
    applications = accounting.get("applications") or {}
    if not applications:
        blockers.append({"component": "applications", "reason": "no complete declared application inventory"})
    signature_witnesses: dict[str, list[str]] = {}
    for capsule in rows:
        for signature in capsule.get("signatures") or []:
            key = signature_identity(signature)
            if key is not None:
                signature_witnesses.setdefault(key, []).append(capsule["name"])
    application_reports = {}
    for label, application in sorted(applications.items()):
        completeness = application.get("completeness") or {}
        graph_accounting, graph_reasons = _graph_totality(application, completeness)
        for reason in graph_reasons:
            blockers.append({"component": "graph_totality", "application": label, "reason": reason})
        receipt = application.get("capture_receipt") or {}
        # The materialized receipt never establishes closure by itself; an admitted
        # issuer's attestation, re-verified from disk against these bytes, does.
        closure = receipt.get("source_closure_verified") is True or attestation_failure(application) is None
        if receipt.get("status") != "verified_materialized" or not closure:
            blockers.append(
                {
                    "component": "capture_provenance",
                    "application": label,
                    "reason": "materialized capture or its source closure is not verified",
                }
            )
        trace = completeness.get("source_trace") or {}
        if not _source_complete(trace):
            blockers.append(
                {
                    "component": "source_trace",
                    "application": label,
                    "reason": "original/quantized source counts and exact lowering correspondence are incomplete",
                }
            )
        if not _framework_versions_complete(application):
            blockers.append(
                {
                    "component": "framework_versions",
                    "application": label,
                    "reason": (
                        "available selected catalog and original/quantized/prepared runtime versions do not all match"
                    ),
                }
            )
        obligations = completeness.get("operation_obligations")
        if not isinstance(obligations, list) or not obligations:
            blockers.append(
                {
                    "component": "operations",
                    "application": label,
                    "reason": "exact per-operation obligations are absent",
                }
            )
            obligations = []
        selected_placements, selected_signatures, operation_reports = {}, {}, []
        for obligation in obligations:
            identifier = obligation.get("id")
            if obligation.get("role") == "support_lowering":
                precision = obligation.get("precision") or {}
                evidence = obligation.get("support_lowering_evidence") or {}
                operand_shapes = [row.get("shape") for row in precision.get("ordered_operand_types") or []]
                result_shapes = [row.get("shape") for row in precision.get("ordered_result_types") or []]
                static_source_shapes = all(
                    isinstance(shape, list) and all(type(dimension) is int and dimension >= 0 for dimension in shape)
                    for shape in [*operand_shapes, *result_shapes]
                )
                reasons = []
                if obligation.get("status") != "resolved" or precision.get("status") != "resolved":
                    reasons.append("exact source operand/result types or shapes are unresolved")
                if (
                    evidence.get("source_capture_sha256") != application.get("capture_sha256")
                    or evidence.get("source_operation_id") != identifier
                    or evidence.get("operand_types") != precision.get("ordered_storage_types")
                    or evidence.get("result_types") != precision.get("result_types")
                    or evidence.get("operand_shapes") != operand_shapes
                    or evidence.get("result_shapes") != result_shapes
                    or evidence.get("source_shape_status")
                    != ("static" if static_source_shapes else "dynamic_or_unknown")
                ):
                    reasons.append("support-lowering observation differs from the selected typed source operation")
                # No selected compiler artifact/verifier is bound by these Phase 0
                # inputs. A self-asserted status cannot certify a lowering or its
                # shape/value mapping; keep this obligation open for Phase 1.
                reasons.append("artifact-backed typed lowering and shape/value preservation are not verified")
                key = signature_identity(obligation.get("observed_signature"))
                witnesses = sorted(set(signature_witnesses.get(key, []))) if key is not None else []
                if not witnesses:
                    reasons.append("no admitted capsule has the exact observed typed operation signature")
                operation_reports.append(
                    {
                        "id": identifier,
                        "role": "support_lowering",
                        "source_operation_ids": obligation.get("source_operation_ids"),
                        "mlir_ordinals": obligation.get("mlir_ordinals"),
                        "placement": None,
                        "support_lowering_evidence": copy.deepcopy(evidence),
                        "signature_sha256": key,
                        "witnesses": witnesses,
                        "status": "missing",
                        "reasons": reasons,
                    }
                )
                blockers.append(
                    {
                        "component": "support_lowering",
                        "application": label,
                        "obligation": identifier,
                        "reason": "; ".join(reasons),
                    }
                )
                continue
            placement = _placement(obligation)
            observed = obligation.get("observed_signature")
            if observed is None:
                matching = [
                    entry.get("observed_signature")
                    for entry in application.get("signatures") or []
                    if set(obligation.get("mlir_ordinals") or ()) <= set(entry.get("ordinals") or ())
                ]
                observed = matching[0] if len(matching) == 1 else None
            key = signature_identity(observed)
            witnesses = sorted(set(signature_witnesses.get(key, []))) if key is not None else []
            reasons = []
            if placement is None:
                reasons.append("reviewed host/accelerator placement is unresolved or ambiguous")
            if (obligation.get("precision") or {}).get("status") != "resolved":
                reasons.append("operand/compute/accumulator/result precision is unresolved")
            numerical_contract = ((obligation.get("precision") or {}).get("numerical_contracts") or {}).get(
                placement
            ) or {}
            if numerical_contract.get("status") != "resolved":
                reasons.append("selected placement has no resolved reviewed operation-specific numerical declaration")
            if obligation.get("status") != "resolved":
                reasons.append("source operation obligation is unresolved")
            if not witnesses:
                reasons.append("no admitted capsule has the exact observed typed operation signature")
            for operation_id in obligation.get("operation_ids") or []:
                selected_placements[operation_id] = placement
                selected_signatures[operation_id] = key
            operation_reports.append(
                {
                    "id": identifier,
                    "role": "compute_placement",
                    "source_operation_ids": obligation.get("source_operation_ids"),
                    "mlir_ordinals": obligation.get("mlir_ordinals"),
                    "placement": placement,
                    "numerical_contract": copy.deepcopy(numerical_contract),
                    "signature_sha256": key,
                    "witnesses": witnesses,
                    "status": "missing" if reasons else "covered",
                    "reasons": reasons,
                }
            )
            if reasons:
                blockers.append(
                    {
                        "component": "operation",
                        "application": label,
                        "obligation": identifier,
                        "reason": "; ".join(reasons),
                    }
                )
        transfers = completeness.get("transfer_obligations")
        if not isinstance(transfers, list):
            blockers.append(
                {"component": "transfers", "application": label, "reason": "typed SSA-edge inventory is absent"}
            )
            transfers = []
        transfer_reports = []
        pending_transfers = 0
        for transfer in transfers:
            producer, consumer = transfer.get("producer_operation_id"), transfer.get("consumer_operation_id")
            from_lane, to_lane = selected_placements.get(producer), selected_placements.get(consumer)
            if from_lane is not None and from_lane == to_lane:
                continue
            witnesses = []
            for capsule in rows:
                graph = capsule.get("operation_graph") or {}
                operation_keys = capsule.get("operation_signature_ids") or {}
                for edge in graph.get("edges") or []:
                    if (
                        selected_signatures.get(producer) is not None
                        and selected_signatures.get(consumer) is not None
                        and operation_keys.get(edge.get("producer_operation_id")) == selected_signatures.get(producer)
                        and operation_keys.get(edge.get("consumer_operation_id")) == selected_signatures.get(consumer)
                        and edge.get("type") == transfer.get("type")
                    ):
                        witnesses.append(capsule["name"])
            if from_lane is None or to_lane is None:
                pending_transfers += 1
                transfer_reports.append(
                    {
                        "id": transfer.get("id"),
                        "from": from_lane,
                        "to": to_lane,
                        "type": transfer.get("type"),
                        "declaration": {"status": "not_screened", "reason": "endpoint placement is unresolved"},
                        "witnesses": sorted(set(witnesses)),
                        "status": "pending_placement",
                        "reasons": ["typed edge endpoint placement is unresolved"],
                    }
                )
                continue
            # Only an actual selected lane crossing can be screened against
            # typed transfer declarations. An unknown endpoint is not evidence
            # that the declaration is absent.
            from merlin.development.targetgen.software_spec import screen_transfer_contract

            route = screen_transfer_contract(
                spec,
                source_placement=from_lane,
                destination_placement=to_lane,
                operand_dtype=transfer.get("operand_dtype", transfer.get("dtype")),
                result_dtype=transfer.get("result_dtype", transfer.get("dtype")),
                operand_layout=transfer.get("operand_layout"),
                result_layout=transfer.get("result_layout"),
            )
            reasons = []
            if route.get("status") != "admitted" or route.get("reviewed") is not True:
                reasons.append("reviewed typed transfer/conversion declaration is absent")
            if not witnesses:
                reasons.append("no admitted capsule observes the exact typed producer/consumer edge")
            transfer_reports.append(
                {
                    "id": transfer.get("id"),
                    "from": from_lane,
                    "to": to_lane,
                    "type": transfer.get("type"),
                    "declaration": route,
                    "witnesses": sorted(set(witnesses)),
                    "status": "missing" if reasons else "covered",
                    "reasons": reasons,
                }
            )
            if reasons:
                blockers.append(
                    {
                        "component": "transfer",
                        "application": label,
                        "obligation": transfer.get("id"),
                        "reason": "; ".join(reasons),
                    }
                )
        if pending_transfers:
            blockers.append(
                {
                    "component": "transfer",
                    "application": label,
                    "count": pending_transfers,
                    "reason": "conditional SSA uses await reviewed endpoint placement before transfer screening",
                }
            )
        pending_support_dependencies = sum(
            edge.get("accounting") == "support_dependency"
            for edge in graph_accounting.get("edges") or []
            if isinstance(edge, dict)
        )
        if pending_support_dependencies:
            blockers.append(
                {
                    "component": "support_dependency",
                    "application": label,
                    "count": pending_support_dependencies,
                    "reason": "compute islands connected through support lowering await a compiler-owned typed route",
                }
            )
        application_reports[label] = {
            "capture_sha256": application.get("capture_sha256"),
            "capture_receipt": copy.deepcopy(receipt),
            "capture_execution_attestation": copy.deepcopy(application.get("capture_execution_attestation")),
            "graph_accounting": copy.deepcopy(graph_accounting),
            "semantic_scope": {
                "compiler_reference": "selected quantized/prepared capture when quantization is present",
                "original_graph_sha256": (trace.get("graphs", {}).get("original") or {}).get("sha256"),
                "quantized_graph_sha256": (trace.get("graphs", {}).get("quantized") or {}).get("sha256"),
                "capture_quantization": application.get("capture_quantization"),
                "original_vs_quantized_equivalence": "not_established_by_capsule_coverage",
            },
            "source_trace": trace,
            "framework_version_match": copy.deepcopy(
                (application.get("framework_universe") or {}).get("source_graph_version_match")
            ),
            "operations": operation_reports,
            "transfers": transfer_reports,
        }
    blockers.extend(_conformance_blockers(conformance_coverage))
    return {
        "schema": SCHEMA,
        "phase": phase,
        "status": "complete" if not blockers else "incomplete",
        "target": document.get("target"),
        "inputs_sha256": _digest(inputs) if inputs is not None else None,
        "cohort": {
            "capsules": [{key: row.get(key) for key in ("name", "sha256", "program_sha256")} for row in rows],
            "n_capsules": len(rows),
            "sha256": _digest([{key: row.get(key) for key in ("name", "sha256", "program_sha256")} for row in rows]),
            "inventories": [
                {
                    "name": row["name"],
                    "kind": row.get("inventory_kind"),
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                    "interface_command_buffer": row.get("interface_command_buffer"),
                }
                for row in rows
            ],
        },
        "applications": application_reports,
        "phase1_witness_basis": build_witness_basis(
            application_reports, rows, coverage_status="complete" if not blockers else "incomplete"
        ),
        "conformance": conformance_coverage,
        "blockers": blockers,
        "independent_evidence": document.get("evidence"),
        "qualification": "pre-compiler completeness; not execution, numerical or RTL certification",
        "workload_scope": "derivation applications only; held-out validation needs separate full-model execution",
    }


def require_complete(report: dict) -> None:
    from merlin.development.testgen.evidence.coverage_obligations import SCHEMA

    if (
        report.get("schema") != SCHEMA
        or report.get("phase") != "phase1"
        or report.get("status") != "complete"
        or report.get("blockers") != []
        or not report.get("applications")
        or (report.get("cohort") or {}).get("n_capsules", 0) <= 0
    ):
        raise ValueError(
            "verified whole-workload Phase 1 requires complete source/precision/transfer and admitted-capsule coverage"
        )


def build_phase0_readiness(report: dict) -> dict:
    """Separate deterministic corpus readiness from compiler-owned qualification.

    A generated corpus can be ready to hand to Phase 1 before a compiler exists.
    That does *not* certify support-lowering, emitted host/device composition, or
    target execution. Only those exact obligations are deferred. Every other
    unknown remains a Phase 0 blocker, including capture execution provenance.
    """
    from merlin.development.testgen.evidence.coverage_obligations import READINESS_SCHEMA, SCHEMA, attestation_failure

    if not isinstance(report, dict):
        raise ValueError("Phase 0 readiness requires a coverage commitment")
    blockers = []
    deferred = []
    applications = report.get("applications") or {}
    cohort = report.get("cohort") or {}
    if report.get("schema") != SCHEMA or report.get("phase") != "phase1":
        blockers.append({"component": "coverage", "reason": "unsupported selected Phase 1 coverage commitment"})
    if not is_sha256(report.get("inputs_sha256")) or not is_sha256(cohort.get("sha256")):
        blockers.append({"component": "selection", "reason": "selected input or cohort byte identity is absent"})
    if not applications or type(cohort.get("n_capsules")) is not int or cohort["n_capsules"] <= 0:
        blockers.append({"component": "scope", "reason": "iteration applications or admitted capsules are absent"})
    if (report.get("independent_evidence") or {}).get("status") != "verified":
        blockers.append({"component": "evidence", "reason": "selected Phase 0 evidence is not verified"})

    # A materialized M2M receipt is producer-authored. Even a true-looking
    # closure flag cannot stand in for a separately verified fresh issuer.
    for label, application in sorted(applications.items()):
        reason = attestation_failure(application)
        if reason is not None:
            blockers.append({"component": "capture_execution", "application": label, "reason": reason})

    conformance = report.get("conformance") or {}
    if (
        type(conformance.get("n_required")) is not int
        or type(conformance.get("n_covered")) is not int
        or conformance["n_required"] < 0
        or conformance["n_covered"] != conformance["n_required"]
        or conformance.get("uncovered") != []
    ):
        blockers.append({"component": "conformance", "reason": "finite admitted-cohort cell coverage is incomplete"})

    artifact_only = "artifact-backed typed lowering and shape/value preservation are not verified"
    support_route_only = "compute islands connected through support lowering await a compiler-owned typed route"
    pending_transfer_only = "conditional SSA uses await reviewed endpoint placement before transfer screening"
    for blocker in report.get("blockers") or []:
        component = blocker.get("component")
        label = blocker.get("application")
        application = applications.get(label) or {}
        if component == "support_lowering":
            operations = application.get("operations") or []
            matching = [op for op in operations if op.get("id") == blocker.get("obligation")]
            if (
                len(matching) == 1
                and matching[0].get("role") == "support_lowering"
                and matching[0].get("reasons") == [artifact_only]
                and matching[0].get("witnesses")
            ):
                deferred.append(blocker)
                continue
        elif component == "support_dependency" and blocker.get("reason") == support_route_only:
            edges = (application.get("graph_accounting") or {}).get("edges") or []
            count = sum(edge.get("accounting") == "support_dependency" for edge in edges)
            if count > 0 and blocker.get("count") == count:
                deferred.append(blocker)
                continue
        elif component == "transfer" and blocker.get("reason") == pending_transfer_only:
            transfers = application.get("transfers") or []
            count = sum(row.get("status") == "pending_placement" for row in transfers)
            if count > 0 and blocker.get("count") == count:
                deferred.append(blocker)
                continue
        elif (
            component == "conformance.composition"
            and blocker.get("reason") == "required coverage axis was not measured"
        ):
            composition = conformance.get("composition") or {}
            if (
                composition.get("status") == "not_measured"
                and composition.get("phase") == "phase0"
                and composition.get("required")
                and set(composition.get("phase1_receipt_required") or {})
                == {
                    "selected_capture",
                    "selected_capsule",
                    "compiler_execution",
                    "lowering_correspondence",
                    "execution",
                }
            ):
                deferred.append(blocker)
                continue
        blockers.append(blocker)

    return {
        "schema": READINESS_SCHEMA,
        "status": "ready" if not blockers else "incomplete",
        "inputs_sha256": report.get("inputs_sha256"),
        "cohort_sha256": cohort.get("sha256"),
        "blockers": blockers,
        "deferred_phase1": deferred,
        "qualification": (
            "deterministic Phase 0 corpus handoff only; deferred compiler routes, numerical "
            "execution and target correctness require independent Phase 1 evidence"
        ),
    }


def phase0_readiness_identity(readiness: dict, *, required: bool) -> dict:
    """Aggregate-only identity for a private readiness report bound to a release."""
    from merlin.development.testgen.evidence.coverage_obligations import READINESS_SCHEMA, _digest

    if not isinstance(readiness, dict) or readiness.get("schema") != READINESS_SCHEMA:
        raise ValueError("unsupported Phase 0 readiness report")
    return {
        "schema": READINESS_SCHEMA,
        "status": readiness.get("status"),
        "required": required,
        "cohort_sha256": readiness.get("cohort_sha256"),
        "inputs_sha256": readiness.get("inputs_sha256"),
        "n_blockers": len(readiness.get("blockers") or []),
        "n_deferred_phase1": len(readiness.get("deferred_phase1") or []),
        "report_sha256": _digest(readiness),
    }


def require_phase0_ready(readiness: dict) -> None:
    """Require only the honest, source-closed deterministic Phase 0 handoff."""
    from merlin.development.testgen.evidence.coverage_obligations import READINESS_SCHEMA

    if (
        not isinstance(readiness, dict)
        or readiness.get("schema") != READINESS_SCHEMA
        or readiness.get("status") != "ready"
        or readiness.get("blockers") != []
        or not is_sha256(readiness.get("inputs_sha256"))
        or not is_sha256(readiness.get("cohort_sha256"))
    ):
        raise ValueError("reviewed Phase 0 corpus requires verified source, declarations and capsule coverage")


def _selected_program(capsule: dict, directory: Path) -> Path | None:
    """Read only the capsule's authoritative IR, never a nearby substitute.

    ISA and direct-template layer/model-slice capsules can provide the shared interface
    grammar rather than Linalg. Inventorying it does not translate its operations
    or make it a witness for an unrelated source signature.
    """
    selected = capsule.get("linalg_mlir")
    if "linalg_mlir" not in capsule and capsule.get("kind") in {"isa", "layer", "model_slice"}:
        selected = capsule.get("interface_mlir")
    if not isinstance(selected, str) or not selected:
        return None
    relative = Path(selected)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        return None
    current = directory
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    return current if current.is_file() else None


def verify_cohort_binding(report: dict, inputs: dict | None, roots, *, contract: Path | None = None) -> None:
    """Verify a frozen report's actual bytes without recapturing or rebuilding it."""
    from merlin.development.backend.source_inputs import fingerprint
    from merlin.development.targetgen.capsule_common import discover_capsules
    from merlin.development.testgen.evidence.coverage_obligations import SCHEMA, _digest

    if report.get("schema") != SCHEMA or report.get("inputs_sha256") != (
        _digest(inputs) if inputs is not None else None
    ):
        raise ValueError("coverage report selected-input identity changed")
    members = []
    for capsule in discover_capsules(roots, labels={"public", "dev"}, contract=contract):
        directory = Path(capsule["__dir__"])
        program = _selected_program(capsule, directory)
        members.append(
            {
                "name": capsule["name"],
                "sha256": fingerprint(directory),
                "program_sha256": sha256_bytes(program.read_bytes()) if program is not None else None,
            }
        )
    members.sort(key=lambda item: item["name"])
    saved = report.get("cohort") or {}
    if (
        saved.get("capsules") != members
        or saved.get("n_capsules") != len(members)
        or saved.get("sha256") != _digest(members)
    ):
        raise ValueError("coverage report no longer binds the exact admitted-cohort bytes")


def admitted_source_roots(source_roots, public_root: Path, *, contract: Path | None = None) -> list[Path]:
    """Select the authored bytes of exactly the admitted, materialized cohort.

    Oracle tier ceilings rewrite the public execution view's capsule metadata.
    Workload readiness instead binds the source program and its original capsule
    bytes, while the ordinary materializer and snapshot bind the execution view.
    """
    from merlin.development.targetgen.capsule_common import discover_capsules

    admitted = discover_capsules(public_root, labels={"public", "dev"}, contract=contract)
    source = discover_capsules(source_roots, labels={"public", "dev"}, contract=contract)
    selected_names = [capsule["name"] for capsule in admitted]
    source_names = [capsule["name"] for capsule in source]
    if len(selected_names) != len(set(selected_names)) or len(source_names) != len(set(source_names)):
        raise ValueError("admitted source cohort has duplicate capsule names")
    by_name = {capsule["name"]: Path(capsule["__dir__"]) for capsule in source}
    if not selected_names or set(selected_names) - set(by_name):
        raise ValueError("materialized cohort is empty or names capsules absent from its source")
    return [by_name[name] for name in sorted(selected_names)]


def observe_cohort(
    inputs: dict | None, roots, *, target: str, contract: Path | None = None, phase: str = "phase1"
) -> dict:
    """Inventory only the selected admitted files, without running a compiler or Torch."""
    from merlin.development.backend.source_inputs import fingerprint
    from merlin.development.targetgen.application_inventory import application_demand_inventory
    from merlin.development.targetgen.capsule_common import discover_capsules
    from merlin.development.testgen.evidence.coverage_obligations import signature_identity
    from merlin.experiments.corpus.coverage import selected_cohort_coverage

    roots = [Path(roots)] if isinstance(roots, (str, Path)) else list(roots)
    capsules = discover_capsules(roots, labels={"public", "dev"}, contract=contract)
    observations = []
    for capsule in capsules:
        directory = Path(capsule["__dir__"])
        row = {"name": capsule["name"], "sha256": fingerprint(directory), "status": "not_available"}
        program = _selected_program(capsule, directory)
        if program is None:
            row["reason"] = "capsule has no contained regular lowering program"
        else:
            row["program_sha256"] = sha256_bytes(program.read_bytes())
            try:
                if "linalg_mlir" not in capsule:
                    from merlin.development.targetgen.contract.interface_emit import parse_interface_mlir

                    # The frozen contract grammar has its own canonical parser.
                    # It is not a source model and cannot lend Linalg signatures
                    # or fabricated SSA transfers to the workload witness set.
                    parsed = parse_interface_mlir(program.read_text())
                    if parsed.get("target") != target:
                        raise ValueError("selected interface program names a different target")
                    row.update(
                        status="inventoried",
                        inventory_kind="interface_command_buffer",
                        interface_command_buffer=parsed,
                        signatures=[],
                    )
                    observations.append(row)
                    if fingerprint(directory) != row["sha256"]:
                        raise ValueError("admitted capsule changed during coverage observation")
                    continue
                capability = (inputs or {}).get("capability_contract")
                if not isinstance(capability, dict):
                    raise ValueError("selected capability contract is absent; no ambient lookup is permitted")
                inventory = application_demand_inventory(
                    {capsule["name"]: program},
                    target,
                    detailed=True,
                    capability_contract=capability,
                    include_graph=True,
                )
                observed = inventory["applications"][capsule["name"]]
                graph = (observed.get("operation_graph") or {}).get("normalized_graph") or {}
                row.update(
                    status="inventoried",
                    inventory_kind="source_mlir",
                    signatures=observed["signatures"],
                    operation_graph=graph,
                )
                operation_ids = {node["ordinal"]: node["operation_id"] for node in graph.get("operations") or []}
                row["operation_signature_ids"] = {
                    operation_ids[ordinal]: signature_identity(signature)
                    for signature in observed["signatures"]
                    for ordinal in signature["ordinals"]
                    if ordinal in operation_ids
                }
            except (OSError, ValueError, RuntimeError) as exc:
                row["reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        if fingerprint(directory) != row["sha256"]:
            raise ValueError("admitted capsule changed during coverage observation")
        observations.append(row)
    coverage = None
    requirement = (inputs or {}).get("conformance")
    if isinstance(requirement, dict):
        try:
            coverage = selected_cohort_coverage(
                requirement,
                sorted({Path(capsule["__dir__"]) for capsule in capsules}),
                inputs=inputs,
                phase=phase,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            coverage = {"status": "not_measured", "reason": f"{type(exc).__name__}: {str(exc)[:500]}"}
    return build_commitment(inputs, observations, phase=phase, conformance_coverage=coverage)
