"""Lift one admitted source or machine representation into structural diagnostic evidence."""

from __future__ import annotations

import hashlib
from collections import Counter

from merlin.verification.lean_contract import document_digest

from .reference_decoder import (  # noqa: F401 -- preserve established imports at this boundary
    _instructions,
    _listing_symbol_span,
    _selected_roles_of,
    _table,
    _trip_counts,
)
from .reference_inputs import (  # noqa: F401 -- preserve established imports at this boundary
    _AnalysisGap,
)


def _source_summary(raw, format, offset, limit, *, symbol="", locator=None):
    from merlin.kernels.reference_inventory import inspect_source

    result = inspect_source(raw, format)
    functions = result.get("functions", [])
    if not functions and result.get("diagnostics"):
        code = (
            "analysis_source_parser_unavailable"
            if result.get("parser") == "unavailable"
            else "analysis_source_parse_failed"
        )
        diagnostics = [
            {key: str(row[key])[:256] for key in ("kind", "reason", "next_action") if key in row}
            for row in result["diagnostics"][:8]
        ]
        raise _AnalysisGap(code, parser=result.get("parser"), diagnostics=diagnostics)
    if symbol:
        functions = [row for row in functions if row.get("symbol") == symbol]
        if len(functions) != 1:
            raise ValueError("analysis_symbol_not_unique")
    elif (locator or {}).get("lines"):
        start, end = locator["lines"]
        functions = [
            row
            for row in functions
            if type(row.get("line")) is int
            and type(row.get("end_line")) is int
            and start <= row["line"]
            and row["end_line"] <= end
        ]
        if not functions:
            raise ValueError("analysis_region_empty")
    elif (locator or {}).get("address_range"):
        raise ValueError("analysis_locator_unsupported")
    rows = []
    for function in functions[offset : offset + limit]:
        rows.append({key: function[key] for key in ("symbol", "line", "end_line", "body_sha256") if key in function})
        rows[-1].update(
            loop_count=len(function.get("loops", [])),
            loop_witnesses=function.get("loops", [])[:4],
            calls=dict(list(function.get("calls", {}).items())[:24]),
            parameters=[str(p)[:160] for p in function.get("parameters", [])[:16]],
        )
    return dict(
        kind="source_syntax",
        parser=result.get("parser"),
        total=len(functions),
        functions=rows,
        next_offset=offset + len(rows) if offset + len(rows) < len(functions) else None,
        diagnostics=result.get("diagnostics", [])[:8],
        semantic_region_status="unknown",
    )


def _lift(
    raw,
    format,
    pin,
    *,
    facts,
    offset,
    limit,
    candidate,
    symbol="",
    locator=None,
    ccas=None,
    mapped_symbol=False,
    region="",
    decoder=None,
    instruction_offset=0,
    summary_only=False,
    entry_contexts=None,
):
    from merlin.analysis.rocc_commands import audit_decoded, decode_stream
    from merlin.kernels.kernel_evidence import ArtifactPin, KernelEvidence, cca_projection
    from merlin.perf.pc_instruction_listing import (
        control_flow,
        lift_schedule,
        listing_regions,
        normalized_text,
    )

    locator = locator or {}
    extent = None
    if format == "elf" and symbol:
        from merlin.perf.compiled_static_features import linked_entry_extent

        try:
            from merlin.analysis.artifact_reader import PinnedELF

            extent = raw.function_extent(symbol) if isinstance(raw, PinnedELF) else linked_entry_extent(raw, symbol)
            if extent["size_bytes"] <= 0:
                raise ValueError("function has no retained extent")
        except ValueError as error:
            raise ValueError("analysis_elf_symbol_boundary_unavailable") from error
    if locator.get("lines") and format != "elf":
        start, end = locator["lines"]
        raw = b"\n".join(raw.splitlines()[start - 1 : end])
    elif locator.get("lines") and not (symbol or locator.get("address_range")):
        raise ValueError("analysis_locator_unsupported")
    declared_isa = (decoder or {}).get("host_isa")
    rows, expanded, decoder_isa = _instructions(raw, format, extent=extent, declared_isa=declared_isa)
    if extent is not None:
        start, end = extent["address"], extent["address"] + extent["size_bytes"]
        rows = tuple(row for row in rows if start <= row.addr < end)
    address = locator.get("address_range")
    if address is not None:
        if not isinstance(address, list) or len(address) != 2:
            raise ValueError("analysis_locator_unsupported")
        start, end = (int(value, 0) if isinstance(value, str) else value for value in address)
        if not 0 <= start < end:
            raise ValueError("analysis_locator_unsupported")
        rows = tuple(row for row in rows if start <= row.addr < end)
    if expanded and locator.get("function") and not (locator.get("lines") or address):
        raise ValueError("analysis_locator_unsupported")
    if not rows:
        raise ValueError("analysis_region_empty")
    table, fact_pin = _table(facts)
    if extent is not None:
        # A local relocation/debug label does not end an STT_FUNC. Keep all
        # instructions in the exact retained extent as one static scope.
        scopes = [(dict(region_id=0, symbol=symbol, entry_pc=extent["address"], static_function_extent=extent), rows)]
    elif expanded:
        scopes = [(r.to_dict(), [rows[index] for index in r.indices]) for r in listing_regions(rows)]
    elif symbol and format == "disassembly" and any("<" in row.section for row in rows):
        body = _listing_symbol_span(rows, symbol)
        scopes = [
            (
                dict(
                    region_id=0,
                    symbol=symbol,
                    entry_pc=body[0].addr,
                    symbol_status="listing_label_inferred_static_span",
                ),
                body,
            )
        ]
    elif any(getattr(row, "section_name", "") for row in rows) and not symbol and not locator:
        from merlin.analysis.riscv_machine import region_index

        indexed = region_index(rows, artifact_sha256=pin["sha256"], custom_opcode=table.get("custom_opcode"))
        scopes = [(scope, body) for scope, body in indexed if scope["accelerator_sites"]]
        if not scopes:
            scopes = indexed
    else:
        # Static ELF symbols are provided by the existing tokenizer. No PC
        # adjacency is promoted to dynamic call order or executed multiplicity.
        grouped = {}
        for row in rows:
            grouped.setdefault(row.section or "unlabelled", []).append(row)
        scopes = [
            (
                dict(
                    region_id=i,
                    symbol=label.partition("<")[2][:-1] if "<" in label and label.endswith(">") else label,
                    entry_pc=body[0].addr,
                ),
                body,
            )
            for i, (label, body) in enumerate(grouped.items())
        ]
    for scope, body in scopes:
        scope.setdefault(
            "accelerator_sites",
            sum(
                1 for row in body if len(row.hexcode) == 8 and int(row.hexcode, 16) & 127 == table.get("custom_opcode")
            ),
        )
        scope.setdefault(
            "stable_id",
            scope["region_id"]
            if isinstance(scope["region_id"], str)
            else "region-"
            + document_digest(
                dict(
                    artifact=pin["sha256"],
                    section=getattr(body[0], "section_name", "") or body[0].section,
                    start=body[0].addr,
                    end=body[-1].addr + len(body[-1].hexcode) // 2,
                )
            )[:24],
        )
    if region:
        scopes = [(scope, body) for scope, body in scopes if region in (scope["region_id"], scope["stable_id"])]
        if len(scopes) != 1:
            raise ValueError("analysis_region_not_unique")
    if symbol and mapped_symbol and (locator.get("lines") or address) and all(not row.section for row in rows):
        # The map labels an explicit span; its function is not a recovered symbol.
        scopes = [(dict(scope, mapped_function=symbol, symbol_status="not_in_listing"), body) for scope, body in scopes]
    elif symbol:
        scopes = [(scope, body) for scope, body in scopes if scope.get("symbol") == symbol]
        if len(scopes) != 1:
            raise ValueError("analysis_symbol_not_unique")
    result = []
    for scope, body in scopes[offset : offset + limit]:
        from merlin.kernels.instruction_comparison import instruction_summary

        if summary_only:
            result.append(
                dict(
                    **scope,
                    listed_instructions=len(body),
                    semantic_region_status="unknown",
                    next_action="Lift this stable region to recover selected ISA operands and CCA witnesses.",
                )
            )
            continue
        text = normalized_text(body)
        pins = [
            ArtifactPin("capture", pin["path"], pin["sha256"], pin["size_bytes"]),
            ArtifactPin(
                "disassembly",
                pin["path"] + "#normalized-region-" + str(scope["region_id"]),
                hashlib.sha256(text.encode()).hexdigest(),
            ),
        ]
        if fact_pin:
            pins.append(ArtifactPin("facts", fact_pin["path"], fact_pin["sha256"], fact_pin["size_bytes"]))
        evidence = KernelEvidence(
            "visible-analysis-" + pin["sha256"][:16] + "-" + str(scope["region_id"]),
            artifacts=tuple(pins),
            capture_kind="static_disassembly",
            complete=False,
            missing=("dynamic_loop_multiplicity", "register_values", "memory_values", "timing", "semantic_region"),
            metadata=dict(
                count_basis="listed_instruction_sites",
                selected_target_correspondence="unknown",
                decoder_isa=decoder_isa,
            ),
        )
        # The existing analyzer refuses absent selected facts rather than
        # inventing an ISA. Evidence-only decoding remains available as a gap.
        model = (
            lift_schedule(body, table=table, evidence=evidence, roles_of=_selected_roles_of(decoder, table))
            if fact_pin
            else None
        )
        cca = cca_projection(model, op="unknown", source="candidate" if candidate else "reference") if model else None
        projection = None
        context = (entry_contexts or {}).get(scope["region_id"]) or {}
        alternates = []
        if model and decoder:
            from merlin.kernels.machine_evidence import machine_projection, projection_summary

            projected, projection = machine_projection(
                body,
                model=model,
                evidence_id=evidence.evidence_id,
                decoder=decoder,
                instruction_offset=instruction_offset,
                entry_state=context.get("entry_state"),
                call_summaries=context.get("call_summaries"),
            )
            if projected is not None:
                cca = projected
            # Distinct caller argument tuples are evidence within this one region, not new kernels.
            for row in context.get("contexts") or ():
                _, alternate = machine_projection(
                    body,
                    model=model,
                    evidence_id=evidence.evidence_id + "#" + row["context_id"],
                    decoder=decoder,
                    entry_state=row.get("registers") or None,
                    call_summaries=context.get("call_summaries"),
                )
                alternates.append(dict(row, observations=projection_summary(alternate)))
        if ccas is not None:
            ccas.append(cca)
        cfg = control_flow(body)
        audit = audit_decoded(decode_stream(body, table), table=table).to_dict()
        audit["unaccounted_total"] = len(audit["unaccounted"])
        audit["unaccounted"] = audit["unaccounted"][:8]
        structure = None
        if model is not None:
            structure = dict(
                identity=model.identity,
                source_kind=model.source_kind,
                command_sites=model.command_count,
                complete=model.complete,
                unknowns=model.unknowns,
                role_counts=dict(model.role_counts),
                stage_witnesses=[
                    dict(index=s.index, identity=s.identity, kind=s.kind, roles=s.roles, parameters=dict(s.parameters))
                    for s in model.stages[:8]
                ],
                omitted_stages=max(0, len(model.stages) - 8),
                dependencies=len(model.dependencies),
                storage_lifetimes=len(model.storage),
                realization="unknown",
            )
        extra = {}
        if context.get("entry_context") is not None:
            extra["entry_context"] = dict(context["entry_context"], contexts=alternates)
        result.append(
            dict(
                **scope,
                **extra,
                listed_instructions=len(body),
                instruction_view=instruction_summary(body, decoder=decoder, offset=instruction_offset),
                mnemonics=dict(Counter(row.mnemonic for row in body)),
                selected_decode=audit,
                evidence=evidence.to_dict(),
                cfg=dict(
                    blocks=len(cfg["blocks"]),
                    natural_loops=len(cfg["natural_loops"]),
                    unresolved_edges=len(cfg["unresolved"]),
                    trip_counts=_trip_counts(projection),
                    status="unknown" if any(not row.hexcode for row in body) else "analyzed",
                    missing_encodings=sum(not row.hexcode for row in body),
                ),
                schedule_model=structure,
                cca=None if cca is None else cca.to_dict(),
                mechanism_projection=projection
                or dict(
                    status="unknown",
                    code="selected_decoder_unavailable" if fact_pin is None else "runtime_operand_projection_required",
                    missing_frontend="control-flow-aware machine-register known-bits analysis",
                    library_routes=[
                        "merlin.targetgen.rocc.selected_operand_protocol:bind_original_operand_selection",
                        "merlin.targetgen.rocc.static_operands:known_integer_bits",
                        "merlin.perf.dma_volume:propagate_constants",
                    ],
                    next_action="Use read-command-buffer and actual schedule consumer receipts; instruction sites alone "
                    "do not establish residency, overlap or execution counts.",
                ),
            )
        )
    return dict(
        kind="static_instruction_analysis",
        count_basis="listed_instruction_sites",
        listed_instructions=len(rows),
        unique_pcs=len({r.addr for r in rows}),
        total=len(scopes),
        offset=offset,
        regions=result,
        next_offset=offset + len(result) if offset + len(result) < len(scopes) else None,
        coverage=dict(
            accelerator_sites=sum(
                1 for row in rows if len(row.hexcode) == 8 and int(row.hexcode, 16) & 127 == table.get("custom_opcode")
            ),
            selected_region_sites=sum(
                scope.get(
                    "accelerator_sites",
                    sum(
                        1
                        for row in body
                        if len(row.hexcode) == 8 and int(row.hexcode, 16) & 127 == table.get("custom_opcode")
                    ),
                )
                for scope, body in scopes
            ),
            semantic_readiness="unknown",
            count_basis="listed_instruction_sites",
        ),
        selected_facts=fact_pin,
        selected_target_correspondence="unknown",
        decoder_isa=decoder_isa,
    )


# Preserve established callable pickle identities through the explicit public exports.
for _export in (
    _source_summary,
    _lift,
):
    _export.__module__ = "merlin_experiments.phase1.feedback.reference_analysis"
