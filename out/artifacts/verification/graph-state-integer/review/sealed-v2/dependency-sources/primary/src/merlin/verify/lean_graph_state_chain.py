"""Actual adjacent schedule boundaries with a fresh complete final correspondence.

Every stage is rechecked, but the composed claim comes from a fresh original to
final GraphState theorem in ONE source projection. Independent SSA numberings
are never silently equated. This leaf issues no pipeline broker acceptance.
"""

from __future__ import annotations

from pathlib import Path

from . import lean_graph_state as state


def check_chain(boundaries, destination, *, reference_root, contract, original_contract_sha256):
    """Check every actual handoff and direct final relation under one host contract.

    A saved stage receipt or candidate projection is not input authority. The
    original semantic identity is required; its independent admission is still
    external. Stages must describe the exact same byte artifact at each handoff.
    """
    if type(boundaries) not in (list, tuple) or not 1 <= len(boundaries) <= 32:
        raise ValueError("explicit bounded nonempty actual boundary chain required")
    if any(type(edge) not in (list, tuple) or len(edge) != 2 for edge in boundaries):
        raise ValueError("each actual boundary needs complete before and after artifacts")
    if type(original_contract_sha256) is not str or len(original_contract_sha256) != 64:
        raise ValueError("original selected effect contract identity required")
    selected = [(state.file_ref(edge[0]), state.file_ref(edge[1])) for edge in boundaries]
    for left, right in zip(selected, selected[1:]):
        if any(left[1][key] != right[0][key] for key in ("sha256", "bytes")):
            raise ValueError("adjacent boundary handoff is not the identical actual intermediate artifact")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    own_source = state.file_ref(__file__)
    contract_source = state.file_ref(contract)
    reports = []
    for index, (before, after) in enumerate(boundaries):
        report = state.check_boundary(
            before,
            after,
            destination / f"stage-{index:03d}",
            reference_root=reference_root,
            contract=contract,
            original_contract_sha256=original_contract_sha256,
            expected_before_sha256=selected[index][0]["sha256"],
        )
        reports.append(report)
        if report["status"] != "graph_model_checked":
            result = {
                "schema": "merlin.graph-state.actual-chain.v1",
                "status": report["status"],
                "failed_stage": index,
                "reason": report.get("reason", "stage kernel/audit unavailable"),
                "boundaries": selected,
                "stage_reports": [state.file_ref(destination / f"stage-{i:03d}/report.json") for i in range(index + 1)],
                "complete_chain_checked": False,
                "public_acceptance_issued": False,
                "source_contract_approved": False,
            }
            state._write(destination / "chain.json", result)
            return result
    direct = state.check_boundary(
        boundaries[0][0],
        boundaries[-1][1],
        destination / "complete",
        reference_root=reference_root,
        contract=contract,
        original_contract_sha256=original_contract_sha256,
        expected_before_sha256=selected[0][0]["sha256"],
    )
    dependencies = [own_source, contract_source, *(row for edge in selected for row in edge)]
    for report in [*reports, direct]:
        dependencies.extend(row["original"] for row in report.get("source_closure", []))
    if not state.fresh(dependencies):
        raise ValueError("actual chain artifact, contract or measured checker source changed")
    result = {
        "schema": "merlin.graph-state.actual-chain.v1",
        "status": direct["status"],
        "complete_chain_checked": direct["status"] == "graph_model_checked",
        "boundaries": selected,
        "original_contract_sha256": original_contract_sha256,
        "contract": contract_source,
        "owned_source": own_source,
        "stage_reports": [state.file_ref(destination / f"stage-{i:03d}/report.json") for i in range(len(reports))],
        "direct_report": state.file_ref(destination / "complete/report.json"),
        "theorem": direct.get("theorem"),
        "existing_root": "Merlin.StateEffects.preserveExchanges",
        "composition": "fresh complete original-to-final projection and theorem in the original source cell domain",
        "stage_state_numberings_identified": False,
        "relation": "all-state equality in every sequential context under the original admitted total Action model",
        "public_acceptance_issued": False,
        "source_contract_approved": False,
        "native_fidelity_proved": False,
        "numerical_gates_discharged": False,
        "pipeline_broker_integration": "pending agreed live custody interface; this is development evidence",
    }
    state._write(destination / "chain.json", result)
    return result
