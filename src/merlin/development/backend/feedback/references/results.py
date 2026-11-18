"""Render bounded diagnostic responses with explicit unknowns and immutable detail identities."""

from __future__ import annotations

import json
from collections import Counter

from merlin.development.backend.feedback.candidates import comparison as candidate_verification
from merlin.development.backend.feedback.guidance import composition_brief as schedule_attribution
from merlin.development.backend.feedback.references import (
    experiments as reference_experiments,
)
from merlin.development.backend.feedback.references import (
    reference_bank,
)
from merlin.verification.lean.contract import document_digest

# ---- reference_results -------------------------------------------------------------------------

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
    from merlin.analysis.kernels.cca.comparison import compact_comparison

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
    from merlin.development.backend.feedback.selfcheck.results import MAX_DETAIL_BYTES, compact_row, load_row

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
            from merlin.analysis.kernels.cca.comparison import compact_comparison

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
        from merlin.development.backend.feedback.references import views as reference_views

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

for _export in (
    _brief_analysis,
    _analysis_summary,
    _bounded_response,
    _response,
):
    _export.__module__ = "merlin.development.backend.feedback.references.analysis"


# ---- reference_representations -----------------------------------------------------------------
# Independent upstream, schedule and emitted views with explicit join limits.


def emitted_view(raw, pin, *, decoder, symbol=""):
    from xdsl.dialects import llvm
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.analysis.programs.emitted_work import assembly_site_kind, summarize_emitted_work
    from merlin.compiler.ir.llvm_context import make_llvm_context
    from merlin.compiler.ir.xdsl_io import IR_LOCK

    if len(raw) > 32 << 20:
        raise ValueError("analysis_emitted_invalid")
    try:
        with IR_LOCK:
            module = Parser(make_llvm_context(), raw.decode()).parse_module()
            module.verify()
    except (ParseError, VerifyException) as error:
        raise ValueError("analysis_emitted_invalid") from error
    functions = [op for op in module.walk() if isinstance(op, llvm.FuncOp) and op.body.blocks]
    if not functions or sum(1 for _ in module.walk()) > 100_000:
        raise ValueError("analysis_emitted_invalid")
    subject = module
    if symbol:
        functions = [op for op in functions if op.sym_name.data == symbol]
        if len(functions) != 1:
            raise ValueError("analysis_symbol_not_unique")
        # The existing counted observer consumes a module, including when the
        # selected function itself has several CFG blocks. Keep original
        # globals/declarations but remove other function bodies from its scope.
        subject = module.clone()
        for operation in tuple(subject.body.block.ops):
            if isinstance(operation, llvm.FuncOp) and operation.sym_name.data != symbol:
                subject.body.block.erase_op(operation)
    sites = Counter(assembly_site_kind(op) for op in subject.walk() if op.name == "llvm.inline_asm")
    from merlin.analysis.mining.references.capture import consumer_structure

    report = dict(
        kind="emitted_ir_analysis",
        input=pin,
        functions=[op.sym_name.data for op in functions],
        scope="selected_function" if symbol else "complete_emitted_artifact",
        instruction_sites=dict(sites),
        consumer_structure=consumer_structure(module, symbol=symbol),
        source_correspondence="unknown",
        numerical_equivalence="unknown",
        performance="unknown",
    )
    if decoder.get("semantics") is not None:
        from merlin.analysis.kernels.effects.instruction_comparison import OperandSummary
        from merlin.analysis.kernels.effects.machine_evidence import _finite
        from merlin.target.isa.rocc.decode import decode_module

        trace = _finite(decode_module(subject, target=decoder["target"], semantics=decoder["semantics"]))
        operands = OperandSummary()
        for row in trace["instructions"]:
            operands.observe(
                row["class"], _finite(row.get("decoded", {})), _finite({name: row.get(name) for name in ("rs1", "rs2")})
            )
        report["operand_summary"] = operands.report()
        report["operand_summary"].update(
            count_basis="static_ir_sites",
            decoder_inputs=decoder.get("decoder_inputs", []),
            instruction_classes=trace["summary"]["class_histogram"],
        )
        report["selected_instruction_decode"] = dict(
            classes=trace["summary"]["class_histogram"],
            witnesses=trace["instructions"][:4],
            omitted=max(0, len(trace["instructions"]) - 4),
        )
    if len(functions) == 1:
        try:
            report["counted_work"] = dict(
                status="observed_counted_cfg", **summarize_emitted_work(functions[0], command_kind=assembly_site_kind)
            )
        except ValueError as error:
            report["counted_work"] = dict(status="unknown", reason=str(error), command_counts=None)
    return report, None


def compare_bundles(reference, candidate, expert, ours, *, compare_machine):
    from merlin.analysis.mining.references.evidence import SemanticRegion, compare_regions
    from merlin.development.backend.feedback.references.applicability import align_questions

    left, right = reference["views"], candidate["views"]
    a, b = left.get("source", {}).get("semantic_region"), right.get("source", {}).get("semantic_region")
    alignment = (
        compare_regions(SemanticRegion.from_record(a), SemanticRegion.from_record(b)).to_dict()
        if a and b
        else dict(status="unknown", reason="complete_upstream_semantic_regions_required")
    )
    machine = None
    if left.get("binary", {}).get("kind") == right.get("binary", {}).get("kind") == "static_instruction_analysis":
        machine = compare_machine(left["binary"], right["binary"], expert, ours)
    emitted = None
    if left.get("emitted", {}).get("kind") == right.get("emitted", {}).get("kind") == "emitted_ir_analysis":
        emitted = compare_machine(left["emitted"], right["emitted"], None, None)
    # Bundle pins identify artifacts. They do not attest that either binary
    # implements the supplied source or schedule.
    return dict(
        kind="representation_comparison",
        reference=reference,
        candidate=candidate,
        upstream_semantics=alignment,
        machine_comparison=machine,
        emitted_comparison=emitted,
        mechanism_alignment=align_questions(reference, candidate),
        semantic_compatibility="unknown",
        producer_correspondence="unknown",
        performance="unknown",
        next_action="Bind source-to-emission producer records or original capture correspondence; "
        "then test supported differences through the ordinary experiment owner.",
    )


# ---- reference_source_structure ----------------------------------------------------------------
# Observe ordinary source IR without inventing its command-buffer lowering.


def source_structure(raw, *, pin, reduction_lanes=None):
    from xdsl.dialects.builtin import TensorType
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.analysis.kernels.mechanisms.inference import source_mechanisms
    from merlin.analysis.mining.references.evidence import semantic_region_from_ir
    from merlin.analysis.optimization.candidates.candidate_source import source_observation
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
    from merlin.compiler.ir.xdsl_io import IR_LOCK

    try:
        with IR_LOCK:
            module = parse_mlir_text(raw.decode())
            module.verify()
    except (ParseError, VerifyException) as error:
        raise ValueError("analysis_source_parse_failed") from error
    if sum(1 for _ in module.walk()) > 100000:
        raise ValueError("analysis_source_operation_budget_exhausted")

    def shape(value):
        return (
            dict(shape=list(value.get_shape()), dtype=str(value.get_element_type()))
            if isinstance(value, TensorType)
            else dict(type=str(value))
        )

    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
    if not functions:
        raise ValueError("analysis_source_defined_function_missing")
    try:
        region = semantic_region_from_ir(module, region_id="source-" + pin["sha256"][:24], complete=False).to_dict()
        region_status = dict(status="observed_source_structure", complete=False)
    except ValueError as error:
        region = None
        region_status = dict(status="unknown", reason=str(error)[:256])
    return dict(
        status="observed_source_structure",
        input=pin,
        groups=[],
        products=None,
        outputs={},
        argument_order=None,
        observation=source_observation(raw.decode()),
        functions=[
            dict(
                symbol=op.sym_name.data,
                inputs=[shape(value) for value in op.function_type.inputs],
                outputs=[shape(value) for value in op.function_type.outputs],
            )
            for op in functions
        ],
        semantic_region=region,
        semantic_region_status=region_status,
        mechanism_observations=source_mechanisms(module, evidence_id="source:" + pin["sha256"], lanes=reduction_lanes),
        source_semantics="parsed_original_ir; command-buffer correspondence unknown",
        command_buffer_correspondence="unknown",
        numerical_equivalence="unknown",
        emitted_mechanism="unknown",
    )
