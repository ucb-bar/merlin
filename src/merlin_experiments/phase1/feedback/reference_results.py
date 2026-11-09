"""Render bounded diagnostic responses with explicit unknowns and immutable detail identities."""

from __future__ import annotations

import json

from merlin.verification.lean_contract import document_digest

from . import candidate_request as candidate_verification
from . import reference_bank, reference_experiments, schedule_attribution

SCHEMA = "merlin.reference_analysis.v1"


ACTION = "reference-analysis"


def _brief_analysis(value, depth=3):
    """Bound presentation only; the existing detail transport retains full evidence."""
    if isinstance(value, str):
        return value if len(value) <= 768 else value[:768] + " [see detail]"
    if isinstance(value, list):
        return [_brief_analysis(row, depth - 1) for row in value[:4]] if depth else []
    if isinstance(value, dict):
        if not depth:
            return dict(detail="see full detail", fields=list(value)[:12], fields_total=len(value))
        return {key: _brief_analysis(row, depth - 1) for key, row in list(value.items())[:12]}
    return value


def _analysis_summary(report):
    from merlin.kernels.instruction_observations import compact_comparison

    summary = {
        key: _brief_analysis(report[key])
        for key in (
            "input",
            "total",
            "offset",
            "next_offset",
            "count_basis",
            "count_scope",
            "source_binding",
            "source_command_join",
            "declared_contraction_alignment",
            "command_role_comparison",
            "unmatched_candidate_commands",
            "tile_dataflow",
            "semantic_equivalence",
            "semantic_compatibility",
            "numerical_status",
            "completion",
            "overlap",
            "performance",
            "schedule_consumption",
            "next_action",
            "gaps",
            "upstream_semantics",
            "producer_correspondence",
            "instruction_comparison",
            "operand_comparison",
            "machine_comparison",
            "emitted_comparison",
            "mechanism_observations",
            "mechanism_alignment",
            "consumer_structure",
            "producer_binding",
            "construction_receipts",
            "construction_comparison",
            "candidate_build_publication",
            "function_view_input",
            "bank",
            "reference_entry",
            "interface",
        )
        if key in report
    }
    summary.update(
        kind=report.get("kind"),
        projection="at most four list entries and bounded nested fields; full evidence in detail",
    )
    if report.get("kind") == "cca_bank_inventory":
        # Every unit and occurrence locator on the page plus a bounded card summary; full cards in the detail.
        def _unit(row):
            brief = dict(row.get("summary") or {})
            names = brief.get("observation_fields") or []
            if len(names) > 24:
                brief.update(observation_fields=names[:24], observation_fields_omitted=len(names) - 24)
            return dict(
                unit_id=row["unit_id"],
                card=row["card"],
                summary=brief,
                occurrences=[occurrence["region_id"] for occurrence in row["occurrences"]],
            )

        summary["units"] = [_unit(row) for row in report.get("units", [])]
        summary["projection"] = "every unit on this page with a bounded card summary; full cards in detail"
    comparison = report.get("cca")
    if isinstance(report.get("observation_comparison"), dict):
        # This helper already has a finite field/value budget. Recursive generic
        # truncation used to erase actual values behind "see full detail".
        summary["observation_comparison"] = compact_comparison(report["observation_comparison"])
    for name in ("machine_comparison", "emitted_comparison"):
        child = report.get(name) or {}
        if isinstance(child.get("observation_comparison"), dict):
            summary[name] = dict(
                kind=child.get("kind"), observation_comparison=compact_comparison(child["observation_comparison"])
            )
    if isinstance(comparison, dict):
        summary["cca"] = {
            key: _brief_analysis(comparison[key])
            for key in ("status", "denominator", "compared_axes", "divergences")
            if key in comparison
        }
        summary["cca"]["unknown_axes"] = len(comparison.get("unknowns", []))
    experiment = report.get("experiment")
    if isinstance(experiment, dict):
        summary["experiment"] = dict(
            status=experiment.get("status"),
            identity=experiment.get("identity"),
            total_tasks=len(experiment.get("tasks", [])),
            tasks=[
                {
                    key: _brief_analysis(task[key])
                    for key in ("task_id", "status", "goals", "summary", "next_action")
                    if key in task
                }
                | dict(request_ids=[row["request_id"] for row in task.get("requests", [])[:2] if "request_id" in row])
                for task in experiment.get("tasks", [])[:4]
            ],
        )
    return summary


def _bounded_response(response, *, workspace, pretty=False):
    """Bound the final transport without republishing analysis or replacing its detail."""
    # The lean writer uses indent=2 plus a newline; compact JSON alone can fit
    # while the actual channel file exceeds the bound after decoration.
    indent = 2 if pretty else None
    if len(json.dumps(response, indent=indent).encode()) + 1 <= 64 << 10:
        return response
    from .result_details import MAX_DETAIL_BYTES, compact_row, load_row

    metadata = {
        key: value for key, value in response.items() if key not in ("report", "reference_experiments", "next_steps")
    }
    try:
        if workspace is None:
            raise ValueError("analysis_detail_workspace_unavailable")
        if response.get("report_in_detail"):
            full = load_row(response, workspace=workspace)
            if (
                full is response
                or full.get("report_identity") != response.get("report_identity")
                or document_digest(full.get("report", {})) != response.get("report_identity")
            ):
                raise ValueError("analysis_detail_unavailable_or_changed")
            detail = response["detail"]
        else:
            if len(json.dumps(response).encode()) > MAX_DETAIL_BYTES:
                raise ValueError("analysis_detail_exceeds_bound")
            full = response
            detail = compact_row(dict(name="reference_analysis", **response), workspace=workspace)["detail"]
    except (OSError, ValueError, TypeError, KeyError):
        # A genuine artifact refusal remains a gap; summary size alone is not one.
        metadata.update(
            status="unknown",
            reason="analysis_report_detail_unavailable",
            next_action="Restore the hash-pinned analysis detail before retrying.",
        )
        return metadata
    report = full.get("report", {})
    compact = dict(
        metadata,
        report=_analysis_summary(report),
        detail=detail,
        report_in_detail=True,
        next_action="Read the hash-pinned detail for all evidence and experiment requests.",
    )
    feedback = response.get("reference_experiments")
    if isinstance(feedback, dict):
        compact["reference_experiments"] = {
            key: _brief_analysis(feedback[key])
            for key in (
                "schema",
                "status",
                "packet_identity",
                "execution_context_identity",
                "task_id",
                "task_status",
                "next_action",
                "experiment_status",
                "emitted_effects",
                "observed_requests",
                "advisory_only",
                "qualification_issued",
                "completion_authority",
            )
            if key in feedback
        }
        compact["reference_experiments"].update(
            projection="bounded current feedback; full packet/outcomes stay owned",
            required_selfchecks=[
                {key: row[key] for key in ("request_id", "role", "capsule", "argv") if key in row}
                for row in feedback.get("required_selfchecks", [])[:2]
            ],
        )
    if isinstance(response.get("next_steps"), dict):
        compact["next_steps"] = {
            key: _brief_analysis(response["next_steps"][key])
            for key in ("schema", "advisory_only", "lines")
            if key in response["next_steps"]
        }
    if len(json.dumps(compact, indent=indent).encode()) + 1 > 64 << 10:
        # Large individual scalar/request fields cannot turn completed analysis
        # into UNKNOWN. Keep exact pointers; do not fabricate a smaller full row.
        compact["report"] = dict(
            kind=report.get("kind"), projection="read full detail; compact evidence exceeded transport bound"
        )
        if isinstance(report.get("observation_comparison"), dict):
            from merlin.kernels.instruction_observations import compact_comparison

            observed = compact_comparison(report["observation_comparison"])
            observed["field_differences"] = observed.get("field_differences", [])[:1]
            compact["report"]["observation_comparison"] = observed
        if isinstance(feedback, dict):
            compact["reference_experiments"] = {
                key: feedback[key]
                for key in ("packet_identity", "execution_context_identity", "task_id", "task_status")
                if key in feedback
            }
            compact["reference_experiments"]["request_ids"] = [
                row["request_id"] for row in feedback.get("required_selfchecks", [])[:2] if "request_id" in row
            ]
        compact.pop("next_steps", None)
    return compact


def _response(base, packet, report, compiler_sha256, *, workspace=None):
    experiment = None
    if workspace is not None and compiler_sha256:
        try:
            experiment = reference_experiments.record_analysis(
                report, workspace=workspace, compiler_sha256=compiler_sha256
            )
        except (OSError, ValueError, TypeError, KeyError):
            experiment = dict(
                status="unknown", reason="current_analysis_experiment_binding_unavailable", tasks=[], requests=[]
            )
    if experiment is not None and (
        experiment.get("tasks") or report.get("gaps") or experiment.get("status") == "unknown"
    ):
        report = dict(report, experiment=experiment)
    elif report.get("gaps"):
        # Reuse the one experiment engine and its exact selected context. A
        # diagnostic cannot authorize a new evaluator, knob or capsule binding.
        from . import reference_views

        report = dict(
            report, experiment=reference_views.experiment(report, packet=packet, compiler_sha256=compiler_sha256)
        )
    producer = report.get("producer_binding") or (report.get("candidate") or {}).get("producer_binding") or {}
    artifact_binding = "unknown"
    if producer.get("status") == "observed_original_build_consumption" and compiler_sha256:
        artifact_binding = (
            "same_compiler_snapshot_digest"
            if producer["compiler_sha256"] == compiler_sha256
            else "different_original_compiler_snapshot"
        )
    response = dict(
        base,
        status="analyzed",
        selection_identity=packet["identity"],
        report=report,
        report_identity=document_digest(report),
        current_compiler_sha256=compiler_sha256,
        artifact_to_current_compiler_binding=artifact_binding,
    )
    return _bounded_response(response, workspace=workspace)


_GAPS = {
    "analysis_dependency_unavailable": "Check admitted paths, format, parser/ISA dependencies and bounds.",
    "analysis_input_unavailable_invalid_or_unsupported": "Check admitted paths, format, parser/ISA dependencies and bounds.",
    "analysis_capture_selection_invalid": "Reopen the original host-pinned CCA capture selection "
    "and its reviewed observer.",
    "analysis_capture_source_mismatch": "The selected capture belongs to different source/function bytes or tensor "
    "storage; recapture through the existing owner.",
    "analysis_capture_artifact_mismatch": "A selected capture, observer, fact, contract or ISA artifact changed; "
    "reopen the frozen original inputs.",
    "analysis_capture_decoder_mismatch": "Replay requires the same original selected facts, contract "
    "and operand protocol target.",
    "analysis_capture_records_invalid": "Use the complete bounded original issue stream, including its buffer "
    "declarations and completion marker.",
    "analysis_capture_replay_incomplete": "The existing tile/dataflow owner could not recover complete ordered "
    "reductions and output coverage; inspect its witness.",
    "analysis_elf_symbol_boundary_unavailable": "Select a unique sized function retained in the ELF symbol table; "
    "stripped or unsupported boundaries cannot establish a complete function span.",
    "analysis_symbol_not_unique": "Select one exact function symbol; inspect an inventory page first.",
    "analysis_region_not_unique": "Use the stable region identity from this artifact's inventory page.",
    "analysis_elf_invalid": "Use a complete ELF with bounded valid executable-section and symbol metadata.",
    "analysis_cpu_isa_unsupported": "This CPU operand frontend requires a little-endian RV64 ELF; "
    "select a supported CPU frontend.",
    "analysis_elf_file_exceeds_bound": "The ELF exceeds the 4 GiB file budget; select an original bounded artifact.",
    "analysis_elf_code_exceeds_bound": "Executable sections exceed the 32 MiB code budget.",
    "analysis_elf_section_unavailable": "Select an executable section retained in the original ELF.",
    "analysis_locator_unsupported": "Use a function, listing lines, or a two-element half-open PC range.",
    "analysis_region_empty": "Recheck the selected file and region against the reference map.",
    "analysis_entry_unavailable": "Select an entry ID from the pinned REFERENCE_MAP.json.",
    "analysis_entry_path_mismatch": "Use the mapped file or omit --reference-path with --reference-entry.",
    "analysis_compare_requires_one_region": "Select one region on each side with a symbol, entry, or nonzero offset.",
    "analysis_compare_domains_differ": "Compare like representations; inspect scheduling intent separately.",
    "analysis_observation_fields_unsupported": "Select machine or emitted instruction views for field comparisons.",
    "analysis_schedule_empty": "Emit an actual schedule.plan through the selected scheduling library.",
    "analysis_schedule_invalid": "Repair the schedule dialect syntax/verifier error using the selected scheduling API.",
    "analysis_semantic_contract_invalid": "Serialize a valid SemanticRegion with the existing kernel evidence API.",
    "analysis_source_parse_failed": "Use the declared format and resolve the parser diagnostic before mining.",
    "analysis_source_parser_unavailable": "Make the existing kernels-ast dependency available to the selected "
    "host parser, then retry; a missing parser is not evidence that a symbol is absent.",
    "analysis_artifact_views_schema_invalid": "Create the pinned source/command_buffer/emitted manifest with "
    "agent_selfcheck.py --write-reference-views MANIFEST SOURCE COMMAND_BUFFER EMITTED.",
    "analysis_artifact_views_role_invalid": "Each source, command_buffer and emitted role needs exactly path "
    "and sha256; use --write-reference-views to pin existing completed artifacts.",
    "analysis_schedule_is_emitted_llvm": "This is emitted LLVM, not schedule.plan. Use --write-reference-views "
    "with the original source, command buffer and emitted LLVM; inspect the actual schedule.plan separately.",
    "analysis_artifact_pin_mismatch": "Reopen the original source, command buffer and emitted bytes; "
    "update the selected artifact pins only after the new compilation finishes.",
    "analysis_emitted_invalid": "Repair the actual emitted LLVM parser/verifier error before comparing activity.",
    "analysis_source_readout_geometry_mismatch": "Reconcile the original product, epilogue and declared result "
    "shape; matching candidate metadata cannot make an inconsistent source contract valid.",
    "analysis_source_readout_geometry_unknown": "Supply the original complete readout geometry, including fused "
    "pool parameters; an unresolved result domain supplies no contract comparison credit.",
    "input_outside_selected_visible_roots": "Use reference-kernels/ or an owned submission/, variants/, "
    "selfcheck_out/ or artifacts/ file. For before/after, both compare inputs may be owned artifacts.",
    "linked_or_foreign_analysis_input": "Use an ordinary file in the admitted workspace roots.",
    "analysis_input_changed": "Finish writing the artifact and retry; retain its original source and content pins.",
    "invalid_analysis_request": "Use the documented reference-analysis options with bounded selectors.",
    "invalid_analysis_operation_or_page": "Select a supported operation and a page limit from 1 through 16.",
    **reference_bank.GAPS,
    **candidate_verification.GAPS,
    **schedule_attribution.GAPS,
}


# Preserve established callable pickle identities through the explicit public exports.
for _export in (
    _brief_analysis,
    _analysis_summary,
    _bounded_response,
    _response,
):
    _export.__module__ = "merlin_experiments.phase1.feedback.reference_analysis"
