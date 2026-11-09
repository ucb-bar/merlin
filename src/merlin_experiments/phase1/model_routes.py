"""Report exact whole-program host/device routing evidence without inventing a compiler.

The Phase 0 ledger screens *candidate* placements. A host LLVM file and an OOT
command buffer are separate products; neither proves the other was stitched into
the model or that the result is numerically correct. Keep that boundary explicit
until a compiler supplies per-operation lowering and typed crossing receipts.
"""

from __future__ import annotations

from collections import Counter

from merlin_experiments.phase0.coverage_commitment import (  # noqa: PLC2701 -- reuse the admission authority
    _graph_totality,
    _placement,
    _source_complete,
)

SCHEMA = "merlin.phase1.model_routes.v1"


def _ledger_observation(application: dict | None, model_sha256: str) -> tuple[dict, list[str]]:
    blockers: list[str] = []
    summary = {"status": "absent", "n_operations": None, "candidate_lanes": {}, "unresolved_obligations": None}
    if not isinstance(application, dict):
        return summary, ["selected per-program operation ledger is absent"]
    if application.get("capture_sha256") != model_sha256:
        blockers.append("operation ledger belongs to different captured MLIR bytes")
    receipt = application.get("capture_receipt") or {}
    if receipt.get("status") != "verified_materialized" or receipt.get("source_closure_verified") is not True:
        blockers.append("capture source closure is not verified")
    completeness = application.get("completeness") or {}
    try:
        _, graph_reasons = _graph_totality(application, completeness)
    except (KeyError, TypeError, ValueError) as exc:
        graph_reasons = [f"malformed exact operation/SSA graph: {type(exc).__name__}: {exc}"]
    blockers.extend(graph_reasons)
    if not _source_complete(completeness.get("source_trace") or {}):
        blockers.append("frontend-to-normalized-MLIR source correspondence is incomplete")
    obligations = completeness.get("operation_obligations")
    if not isinstance(obligations, list):
        obligations = []
        blockers.append("per-operation placement obligations are absent")
    lanes = Counter()
    selected_lanes = {}
    unresolved = 0
    support_pending = 0
    for row in obligations:
        if not isinstance(row, dict):
            unresolved += 1
            continue
        if row.get("role") == "support_lowering":
            # The Phase 0 source observation is not an executable/compiler
            # lowering receipt. Support nodes have no independent lane or
            # arithmetic contract, and cannot be transfer endpoints here.
            support_pending += 1
            continue
        lane = _placement(row)
        numerical = ((row.get("precision") or {}).get("numerical_contracts") or {}).get(lane) or {}
        if (
            lane is None
            or row.get("status") != "resolved"
            or (row.get("precision") or {}).get("status") != "resolved"
            or numerical.get("status") != "resolved"
        ):
            unresolved += 1
        else:
            lanes[lane] += 1
            for operation_id in row.get("operation_ids") or []:
                selected_lanes[operation_id] = lane
    if unresolved:
        blockers.append(f"{unresolved} operation placement/precision/numerical obligations remain unresolved")
    if support_pending:
        blockers.append(f"{support_pending} typed support-lowering/shape obligations lack compiler verification")
    transfers = completeness.get("transfer_obligations")
    if not isinstance(transfers, list):
        blockers.append("typed host/device transfer obligations are absent")
        transfers = []
    # These are compute SSA edges, not necessarily crossings. Only reviewed,
    # resolved endpoint placements can turn an edge into a required transfer.
    transfer_counts = Counter()
    for edge in transfers:
        if not isinstance(edge, dict):
            transfer_counts["pending_placement"] += 1
            continue
        producer = selected_lanes.get(edge.get("producer_operation_id"))
        consumer = selected_lanes.get(edge.get("consumer_operation_id"))
        if producer is None or consumer is None:
            transfer_counts["pending_placement"] += 1
        elif producer == consumer:
            transfer_counts["same_lane"] += 1
        else:
            transfer_counts["required_crossing"] += 1
    if transfer_counts["required_crossing"]:
        blockers.append(
            f"{transfer_counts['required_crossing']} selected host/device crossings "
            "lack typed transfer lowering witnesses"
        )
    transfer_status = (
        "pending_placement"
        if transfer_counts["pending_placement"]
        else "requires_lowering"
        if transfer_counts["required_crossing"]
        else "same_lane_or_none"
    )
    support_edges = sum(
        edge.get("accounting") == "support_dependency"
        for edge in (completeness.get("graph_accounting") or {}).get("edges") or []
        if isinstance(edge, dict)
    )
    if support_edges:
        blockers.append(f"{support_edges} support-mediated SSA dependencies lack compiler-owned route evidence")
    summary = {
        "status": "screened_not_compiled" if not blockers else "incomplete",
        "n_operations": application.get("n_mlir_operations"),
        "candidate_lanes": dict(sorted(lanes.items())),
        "unresolved_obligations": unresolved,
        "support_lowering": {"pending": support_pending, "support_dependencies": support_edges},
        "conditional_ssa_edges": {
            "status": transfer_status,
            "count": len(transfers),
            "pending_placement": transfer_counts["pending_placement"],
            "same_lane": transfer_counts["same_lane"],
            "required_crossing": transfer_counts["required_crossing"],
        },
        "qualification": (
            "declaration screening only; support lowering and compiler-owned routes require separate evidence"
        ),
    }
    return summary, blockers


def summarize_model_routes(
    workflow: dict, accounting: dict | None, compiler_observations: list[dict], native_lowerings: list[dict]
) -> list[dict]:
    """Bind model-route diagnostics to each declared program's exact captured bytes.

    This is intentionally not an executable plan. An OOT package that declines a
    whole captured model cannot be turned into accelerator work by a host LLVM
    file, and a nonempty buffer alone cannot prove that every model operation or
    host/device edge was lowered.
    """
    applications = (accounting or {}).get("applications") or {}
    compiler = {(row.get("program"), row.get("entrypoint")): row for row in compiler_observations}
    native = {row.get("program"): row for row in native_lowerings}
    result = []
    for program in workflow["programs"]:
        name, model_sha256 = program["name"], program["model_sha256"]
        ledger, blockers = _ledger_observation(applications.get(name), model_sha256)
        command = (compiler.get((name, "emit_command_buffer")) or {}).get("command_buffer") or {}
        command_status = command.get("status", "not_emitted")
        route_status = {
            "emitted": "unverified_candidate",
            "declined": "observed_decline",
            "invalid": "invalid_observation",
        }.get(command_status, "unresolved")
        if command_status != "emitted":
            blockers.append(f"selected OOT compiler command buffer is {command_status}")
        else:
            blockers.append("emitted commands have no whole-model operation/transfer correspondence witness")
        host = native.get(name) or {}
        host_status = host.get("status", "not_requested")
        if host_status != "llvm_ir_emitted":
            blockers.append(f"complete native-host LLVM lowering is {host_status}")
        blockers.append("whole-program numerical execution and target-visible seam are not verified")
        result.append(
            {
                "schema": SCHEMA,
                "program": name,
                "capture_sha256": model_sha256,
                "status": route_status,
                "whole_model_compiler_verified": False,
                "ledger": ledger,
                "host": {
                    "status": host_status,
                    "llvm_ir_sha256": host.get("llvm_ir_sha256"),
                    "qualification": "native LLVM IR only; not selected target-host execution",
                },
                "accelerator": {
                    "status": command_status,
                    "commands_count": command.get("commands_count"),
                    "qualification": "OOT payload observation only; not complete-model offload",
                },
                "blockers": sorted(set(blockers)),
            }
        )
    return result
