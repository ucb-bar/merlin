"""Analyze explicitly selected input representations using the caller-bound operand decoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .reference_inputs import (  # noqa: F401 -- preserve established imports at this boundary
    _MAX_TEXT,
    _AnalysisGap,
    _entry,
    _member,
    _path,
    _read,
)
from .reference_lifting import (  # noqa: F401 -- preserve established imports at this boundary
    _lift,
    _source_summary,
)


def _analyze(
    value,
    *,
    workspace,
    reference_root,
    facts,
    candidate=False,
    compare=False,
    contract=None,
    target=None,
    descriptor=None,
    candidate_build_manifest=None,
    _operand_decoder,
):
    entry = None if candidate else _entry(value)
    label = "reference-kernels/" + entry["file"] if entry else value.get("path", "reference-kernels")
    path, label = _path(workspace, label, candidate=candidate, reference_root=reference_root)
    raw, pin = _read(path, label, allow_elf=True)
    from merlin.analysis.artifact_reader import PinnedELF

    try:
        return _analyze_open(
            value,
            raw=raw,
            pin=pin,
            label=label,
            entry=entry,
            workspace=workspace,
            reference_root=reference_root,
            facts=facts,
            candidate=candidate,
            compare=compare,
            contract=contract,
            target=target,
            descriptor=descriptor,
            candidate_build_manifest=candidate_build_manifest,
            _operand_decoder=_operand_decoder,
        )
    finally:
        if isinstance(raw, PinnedELF):
            raw.close()


def _analyze_open(
    value,
    *,
    raw,
    pin,
    label,
    entry,
    workspace,
    reference_root,
    facts,
    candidate=False,
    compare=False,
    contract=None,
    target=None,
    descriptor=None,
    candidate_build_manifest=None,
    _operand_decoder,
):
    member = value.get("member")
    if member:
        raw = _member(raw, member)
        pin = dict(
            path=label + "#" + member, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw), container=pin
        )
    format = value.get("format", "auto")
    if format == "auto":
        suffix = Path(member or label).suffix.lower()
        format = (
            "elf"
            if raw[:4] == b"\x7fELF"
            else "c_source"
            if suffix in (".c", ".h")
            else "python_schedule"
            if suffix == ".py"
            else "interface_mlir"
            if suffix == ".mlir" and b"merlin_iface." in raw
            else "schedule_mlir"
            if suffix == ".mlir"
            else "artifact_views"
            if suffix == ".json" and raw.lstrip().startswith(b"{")
            else "listing"
            if raw.lstrip().startswith(b"0x")
            else "disassembly"
        )
        if format == "artifact_views":
            document = json.loads(raw)
            if isinstance(document, dict) and {"tensors", "commands"} <= document.keys():
                format = "command_buffer"
            elif isinstance(document, dict) and document.get("schema") == "merlin.candidate_build_view.v1":
                format = "candidate_build"
    offset, limit = (value.get("offset", 0), 1 if compare else value.get("limit", 8))
    locator = entry.get("locator", {}) if entry else {}
    symbol = value.get("symbol") or locator.get("function", "")
    ccas = []
    if format == "candidate_build":
        if not candidate:
            raise ValueError("candidate_build_requires_owned_candidate")
        from .candidate_build_views import analyze

        return analyze(
            value,
            pin,
            workspace=workspace,
            bundle_manifest=candidate_build_manifest,
            facts=facts,
            contract=contract,
            target=target,
            descriptor=descriptor,
        )
    if format == "interface_mlir":
        from .reference_mechanisms import selected_lanes
        from .reference_views import source_view

        lanes, lane_status, lane_inputs = selected_lanes(facts=facts, target=target)
        report = source_view(raw, reduction_lanes=lanes)
        report["mechanism_observations"].update(reduction_geometry=lane_status, selected_inputs=lane_inputs)
        report.pop("command_buffer")
        return (dict(report, kind="interface_source_analysis", input=pin), None)
    if format == "source_mlir":
        from xdsl.utils.exceptions import ParseError, VerifyException

        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.kernels.kernel_evidence import semantic_region_from_ir

        if len(raw) > _MAX_TEXT:
            raise ValueError("analysis_source_parse_failed")
        try:
            module = parse_mlir_text(raw.decode("utf-8"))
            module.verify()
        except (ParseError, VerifyException) as error:
            raise ValueError("analysis_source_parse_failed") from error
        if sum((1 for _ in module.walk())) > 100000:
            raise ValueError("analysis_source_parse_failed")
        observed = semantic_region_from_ir(module, region_id="source-" + pin["sha256"][:24], complete=False)
        from .reference_mechanisms import source_report

        return (
            dict(
                kind="upstream_ir_analysis",
                input=pin,
                semantic_region=observed.to_dict(),
                mechanism_observations=source_report(module, pin, facts=facts, target=target),
                semantic_region_status="observed_source_structure",
                numerical_contract="unknown",
                emitted_correspondence="unknown",
            ),
            observed,
        )
    if format == "emitted_mlir":
        from .reference_representations import emitted_view

        return emitted_view(
            raw, pin, decoder=_operand_decoder(contract, facts, target, descriptor=descriptor), symbol=symbol
        )
    if format == "artifact_views":
        document = json.loads(raw)
        if isinstance(document, dict) and document.get("schema") == "merlin.reference_artifact_views.v2":
            return _representation_bundle(
                document,
                pin,
                workspace=workspace,
                reference_root=reference_root,
                facts=facts,
                candidate=candidate,
                contract=contract,
                target=target,
                descriptor=descriptor,
                _operand_decoder=_operand_decoder,
            )
        from .reference_views import artifact_views

        try:
            document = json.loads(raw)
        except ValueError as error:
            raise ValueError("analysis_artifact_views_schema_invalid") from error
        if (
            not isinstance(document, dict)
            or document.get("schema") != "merlin.reference_artifact_views.v1"
            or set(document) - {"schema", "source", "command_buffer", "emitted", "entry_symbol"}
        ):
            raise ValueError("analysis_artifact_views_schema_invalid")
        inputs, input_pins = ({}, {})
        for role in ("source", "command_buffer", "emitted"):
            selected = document.get(role)
            if not isinstance(selected, dict) or set(selected) != {"path", "sha256"}:
                raise _AnalysisGap(
                    "analysis_artifact_views_role_invalid", role=role, required_fields=["path", "sha256"]
                )
            artifact, visible = _path(workspace, selected["path"], candidate=candidate, reference_root=reference_root)
            payload, observed = _read(artifact, visible)
            if observed["sha256"] != selected["sha256"]:
                raise ValueError("analysis_artifact_pin_mismatch")
            inputs[role] = payload
            input_pins[role] = observed
        decoder = _operand_decoder(contract, facts, target, descriptor=descriptor, include_structure=True)
        from .reference_mechanisms import selected_lanes

        lanes, lane_status, lane_inputs = selected_lanes(facts=facts, target=target)
        report = artifact_views(
            source_mlir=inputs["source"],
            command_buffer=inputs["command_buffer"],
            emitted_mlir=inputs["emitted"],
            entry_symbol=document.get("entry_symbol", "kernel_entry"),
            reduction_lanes=lanes,
            **{key: decoder[key] for key in ("semantics", "target", "decoder_inputs") if key in decoder},
        )
        report["source"]["mechanism_observations"].update(reduction_geometry=lane_status, selected_inputs=lane_inputs)
        from .reference_capture_views import candidate_view

        projected, cca = candidate_view(
            inputs, entry_symbol=document.get("entry_symbol", "kernel_entry"), decoder=decoder
        )
        return (dict(report, input=pin, artifact_inputs=input_pins, captured_reference_axes=projected), cca)
    if format == "command_buffer":
        from merlin.perf.command_buffer_diagnostics import representation_activity

        document = json.loads(raw)
        if (
            not isinstance(document, dict)
            or not isinstance(document.get("tensors"), dict)
            or (not isinstance(document.get("commands"), list))
        ):
            raise ValueError("invalid_analysis_request")
        return (
            dict(
                kind="declared_command_buffer",
                input=pin,
                declared_representations=representation_activity(document),
                source_binding="unknown",
                emitted_mechanism="unknown",
                semantic_equivalence="unknown",
                next_action="Join original source and emitted LLVM with --write-reference-views; declared commands alone do not establish emitted work.",
            ),
            None,
        )
    if format == "semantic_region":
        from merlin.kernels.kernel_evidence import SemanticRegion

        try:
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError("not a semantic region record")
            region = SemanticRegion.from_record(document)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError("analysis_semantic_contract_invalid") from error
        report = dict(kind="declared_semantic_region", region=region.to_dict(), input=pin)
        return (report, region)
    if format == "schedule_mlir":
        from xdsl.utils.exceptions import ParseError, VerifyException

        from merlin.analysis.schedule_structure import analyze_ir_schedule
        from merlin.compiler.schedule.schedule_ir import parse_dense_schedule_plan

        try:
            module = parse_dense_schedule_plan(raw.decode("utf-8"))
            module.verify()
        except (ParseError, VerifyException) as error:
            from xdsl.dialects import llvm
            from xdsl.parser import Parser

            from merlin.common.ir_lock import IR_LOCK
            from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

            try:
                with IR_LOCK:
                    emitted = Parser(make_llvm_context(), raw.decode("utf-8")).parse_module()
                    emitted.verify()
                is_llvm = any((isinstance(op, llvm.FuncOp) for op in emitted.walk()))
            except (ParseError, VerifyException):
                is_llvm = False
            if is_llvm:
                raise ValueError("analysis_schedule_is_emitted_llvm") from error
            raise ValueError("analysis_schedule_invalid") from error
        model = analyze_ir_schedule(module)
        if not model.decisions:
            raise ValueError("analysis_schedule_empty")
        report = dict(
            kind="schedule_decisions",
            total=len(model.decisions),
            input=pin,
            schedule_identity=model.identity,
            decisions=model.decisions[offset : offset + limit],
            unknowns=model.unknowns,
            realized=False,
            source_kind=model.source_kind,
            next_action="Consume this schedule in emission and bind the artifact; annotations alone are intent.",
        )
        return (report, None)
    if format in ("c_source", "python_schedule"):
        if format == "c_source" and (not candidate):
            from .reference_capture_views import lift_selected

            captured = lift_selected(
                raw,
                pin,
                symbol=symbol,
                workspace=workspace,
                reference_root=reference_root,
                facts=facts,
                contract=contract,
                target=target,
                descriptor=descriptor,
                limit=limit,
            )
            if captured is not None:
                return captured
        report = _source_summary(raw, format, offset, limit, symbol=symbol, locator=locator)
        if report.get("diagnostics") and (not report["functions"]):
            raise ValueError("analysis_source_parse_failed")
    else:
        if format == "listing" and locator and (not value.get("symbol")):
            symbol = ""
        report = _lift(
            raw,
            format,
            pin,
            facts=facts,
            offset=offset,
            limit=limit,
            candidate=candidate,
            symbol=symbol,
            locator=locator,
            ccas=ccas,
            mapped_symbol=entry is not None and (not value.get("symbol")),
            region=value.get("region", ""),
            instruction_offset=value.get("instruction_offset", 0),
            summary_only=value.get("operation") == "inventory",
            decoder=_operand_decoder(contract, facts, target, descriptor=descriptor, include_structure=True),
        )
    if compare and report.get("total") != 1 and (not (symbol or offset or value.get("region"))):
        raise ValueError("analysis_compare_requires_one_region")
    if compare and len(report.get("regions", report.get("functions", []))) != 1:
        raise ValueError("analysis_compare_requires_one_region")
    report["input"] = pin
    if entry:
        report["reference_entry"] = dict(
            id=entry["id"],
            locator=locator,
            operation=entry["operation"],
            confidence=entry["confidence"],
            semantic_compatibility="unknown",
        )
    return (report, ccas[0] if len(ccas) == 1 else None)


def _representation_bundle(
    document, pin, *, workspace, reference_root, facts, candidate, contract, target, descriptor, _operand_decoder
):
    """Reopen each domain independently; supplied associations grant no producer authority."""
    if set(document) != {"schema", "views"} or not isinstance(document["views"], dict):
        raise ValueError("analysis_artifact_views_schema_invalid")
    views = document["views"]
    if not 2 <= len(views) <= 4 or set(views) - {"source", "schedule", "emitted", "binary"}:
        raise ValueError("analysis_artifact_views_schema_invalid")
    reports, ccas = ({}, {})
    formats = dict(
        source={"c_source", "python_schedule", "interface_mlir", "source_mlir", "semantic_region"},
        schedule={"schedule_mlir", "python_schedule"},
        emitted={"emitted_mlir"},
        binary={"elf", "disassembly", "listing"},
    )
    for role, selected in views.items():
        if (
            not isinstance(selected, dict)
            or not {"path", "sha256", "format"} <= set(selected)
            or set(selected) - {"path", "sha256", "format", "symbol", "region"}
            or (selected["format"] not in formats[role])
        ):
            raise _AnalysisGap("analysis_artifact_views_role_invalid", role=role)
        request = {key: selected[key] for key in ("path", "format", "symbol", "region") if key in selected}
        request["limit"] = 1
        report, cca = _analyze(
            request,
            workspace=workspace,
            reference_root=reference_root,
            facts=facts,
            candidate=candidate,
            contract=contract,
            target=target,
            descriptor=descriptor,
            _operand_decoder=_operand_decoder,
        )
        if report.get("input", {}).get("sha256") != selected["sha256"]:
            raise ValueError("analysis_artifact_pin_mismatch")
        if report.get("total", 1) != 1 and (not (selected.get("region") or selected.get("symbol"))):
            raise ValueError("analysis_compare_requires_one_region")
        reports[role], ccas[role] = (report, cca)
    result = dict(
        kind="representation_bundle",
        input=pin,
        views=reports,
        representation_links=[
            dict(source=a, destination=b, status="supplied_artifact_association", producer_correspondence="unknown")
            for a, b in zip(reports, list(reports)[1:])
        ],
        semantic_compatibility="unknown",
        emitted_correspondence="unknown",
        next_action="Check each representation and bind original producer/capture correspondence.",
    )
    return (result, ccas.get("binary", ccas.get("emitted")))
