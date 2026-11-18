"""Analyze explicitly selected input representations using the caller-bound operand decoder."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import subprocess
import tempfile
from pathlib import Path

from merlin.development.backend.feedback.references.lifting import (
    _MAX_ROWS,
    _MAX_TEXT,
    _AnalysisGap,
    _entry,
    _lift,
    _member,
    _path,
    _read,
    _source_summary,
)
from merlin.verification.lean.contract import document_digest

# ---- reference_formats -------------------------------------------------------------------------


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
    # Optional authored selector defaults; never copied into the resulting evidence report.
    value = {"path": "reference-kernels", "format": "auto", "offset": 0, "limit": 8, "instruction_offset": 0, **value}
    entry = None if candidate else _entry(value)
    label = "reference-kernels/" + entry["file"] if entry else value["path"]
    path, label = _path(workspace, label, candidate=candidate, reference_root=reference_root)
    raw, pin = _read(path, label, allow_elf=True)
    from merlin.analysis.observation.elf_artifacts import PinnedELF

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
    # Direct callers may omit request paging and format options. These select work, not evidence.
    value = {"format": "auto", "offset": 0, "limit": 8, "instruction_offset": 0, **value}
    member = value.get("member")
    if member:
        raw = _member(raw, member)
        pin = dict(
            path=label + "#" + member, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw), container=pin
        )
    format = value["format"]
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
    offset, limit = (value["offset"], 1 if compare else value["limit"])
    locator = entry.get("locator", {}) if entry else {}
    symbol = value.get("symbol") or locator.get("function", "")
    ccas = []
    if format == "candidate_build":
        if not candidate:
            raise ValueError("candidate_build_requires_owned_candidate")
        from merlin.development.backend.feedback.candidates.candidate_build_views import analyze

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
        from merlin.development.backend.feedback.references.applicability import selected_lanes
        from merlin.development.backend.feedback.references.views import source_view

        lanes, lane_status, lane_inputs = selected_lanes(facts=facts, target=target)
        report = source_view(raw, reduction_lanes=lanes)
        report["mechanism_observations"].update(reduction_geometry=lane_status, selected_inputs=lane_inputs)
        report.pop("command_buffer")
        return (dict(report, kind="interface_source_analysis", input=pin), None)
    if format == "source_mlir":
        from xdsl.utils.exceptions import ParseError, VerifyException

        from merlin.analysis.mining.references.evidence import semantic_region_from_ir
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

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
        from merlin.development.backend.feedback.references.applicability import source_report

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
        from merlin.development.backend.feedback.references.results import emitted_view

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
        from merlin.development.backend.feedback.references.views import artifact_views

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
        from merlin.development.backend.feedback.references.applicability import selected_lanes

        lanes, lane_status, lane_inputs = selected_lanes(facts=facts, target=target)
        # Legacy artifact-view requests select the public kernel_entry ABI when no symbol is authored.
        entry_symbol = document["entry_symbol"] if "entry_symbol" in document else "kernel_entry"
        report = artifact_views(
            source_mlir=inputs["source"],
            command_buffer=inputs["command_buffer"],
            emitted_mlir=inputs["emitted"],
            entry_symbol=entry_symbol,
            reduction_lanes=lanes,
            **{key: decoder[key] for key in ("semantics", "target", "decoder_inputs") if key in decoder},
        )
        report["source"]["mechanism_observations"].update(reduction_geometry=lane_status, selected_inputs=lane_inputs)
        from merlin.development.backend.feedback.references.capture_views import candidate_view

        projected, cca = candidate_view(inputs, entry_symbol=entry_symbol, decoder=decoder)
        return (dict(report, input=pin, artifact_inputs=input_pins, captured_reference_axes=projected), cca)
    if format == "command_buffer":
        from merlin.analysis.programs.command_buffers import representation_activity

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
        from merlin.analysis.mining.references.evidence import SemanticRegion

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

        from merlin.analysis.observation.schedule_structure import analyze_ir_schedule
        from merlin.compiler.schedule.dense.plans import parse_dense_schedule_plan

        try:
            module = parse_dense_schedule_plan(raw.decode("utf-8"))
            module.verify()
        except (ParseError, VerifyException) as error:
            from xdsl.dialects import llvm
            from xdsl.parser import Parser

            from merlin.compiler.ir.llvm_context import make_llvm_context
            from merlin.compiler.ir.xdsl_io import IR_LOCK

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
            from merlin.development.backend.feedback.references.capture_views import lift_selected

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
            instruction_offset=value["instruction_offset"],
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
        if report.get("total") != 1 and (not (selected.get("region") or selected.get("symbol"))):
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


# ---- reference_decoder -------------------------------------------------------------------------
# Decode declared instruction and operand structure from the selected machine evidence.


def _table(path):
    if path is None:
        return {}, None
    from merlin.common.target_data import interface

    raw = Path(path).read_bytes()
    if len(raw) > 16 << 20:
        raise ValueError("selected_facts_exceed_bound")
    document = json.loads(raw)
    body = document.get("facts", document)
    return interface(body, "funct_decode_table") or {}, dict(
        path="selected_rtl_facts", sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )


def _selected_roles_of(decoder, table):
    """Derived command name -> roles, from the selected decoder's role map, or ``None``.

    The role map (``structure["instruction_roles"]``) is keyed by the selected ISA's normalized
    command class; each derived funct name reaches it through that ISA's own ``FUNCT_CLASS``.
    A name the map does not bind stays role-unresolved.
    """
    roles = ((decoder or {}).get("structure") or {}).get("instruction_roles")
    semantics = (decoder or {}).get("semantics")
    if not roles or semantics is None or not decoder.get("target"):
        return None
    classes = semantics.isa_constants(decoder["target"]).get("FUNCT_CLASS") or {}
    by_name = {}
    for code, name in (table.get("names") or {}).items():
        selected = classes.get(int(code))
        if selected is not None and roles.get(selected):
            by_name[str(name)] = tuple(roles[selected])
    return (lambda name: by_name.get(name, ())) if by_name else None


_LISTING_DECODER_ISA = dict(value=None, source="supplied_listing", status="not_selected_by_reader", options=[])


def _instructions(raw, format, *, extent=None, declared_isa=None):
    from merlin.analysis.mining.references.instruction_listing import read_listing
    from merlin.analysis.observation.disassembly import objdump_bin, tokenize_text
    from merlin.analysis.observation.elf_artifacts import PinnedELF

    if isinstance(raw, PinnedELF):
        if declared_isa is not None:
            raw.declare_isa(declared_isa)
        if extent is not None:
            selected = raw.function_disassembly(extent["symbol"])
            return tuple(tokenize_text(selected["text"])), False, dict(raw.decoder_isa)
        rows = raw.instructions()
        return rows, False, dict(raw.decoder_isa)

    if format == "elf":
        temporary_root = os.environ.get("TMPDIR")
        if not temporary_root or not Path(temporary_root).is_absolute():
            raise ValueError("analysis_requires_owner_task_temporary_root")
        with tempfile.TemporaryDirectory(prefix="reference-analysis-", dir=temporary_root) as directory:
            source = Path(directory) / "input.elf"
            source.write_bytes(raw)
            if extent is not None:
                from merlin.analysis.observation.elf_artifacts import open_ordinary

                selected = PinnedELF(open_ordinary(source), "selected_elf", declared_isa=declared_isa)
                try:
                    listed = selected.function_disassembly(extent["symbol"])
                    return tuple(tokenize_text(listed["text"])), False, listed["decoder_isa"]
                finally:
                    selected.close()

            def bound_output():
                resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_TEXT, _MAX_TEXT))

            from merlin.analysis.observation.elf_artifacts import open_ordinary

            try:
                probe = PinnedELF(open_ordinary(source), "selected_elf", declared_isa=declared_isa)
            except ValueError as error:
                # The pinned reader refuses this container; select no ISA rather than guess one.
                decoder_isa = dict(value=None, source=None, status="unknown", options=[], reason=str(error))
            else:
                decoder_isa = dict(probe.decoder_isa)
                probe.close()
            options = list(decoder_isa["options"])
            with tempfile.TemporaryFile(dir=directory) as output, tempfile.TemporaryFile(dir=directory) as errors:
                command = [objdump_bin(), "-d", "-M", "no-aliases", *options]
                if extent is not None:
                    # Relocatable functions in different sections can share a
                    # PC. Select the symbol's ELF-owned section before its range.
                    command.append("--section=" + extent["section"])
                process = subprocess.run(
                    [*command, str(source)], stdout=output, stderr=errors, timeout=30, preexec_fn=bound_output
                )
                output.seek(0)
                raw = output.read(_MAX_TEXT + 1)
            if process.returncode:
                raise ValueError("selected_disassembler_could_not_read_input")
    if len(raw) > _MAX_TEXT:
        raise ValueError("instruction_text_exceeds_bound")
    text = raw.decode("utf-8")
    if format == "listing":
        if text.count("\n") > _MAX_ROWS:
            raise ValueError("listing_row_budget_exceeded")
        return read_listing(text.splitlines()), True, _LISTING_DECODER_ISA
    rows = tokenize_text(text)
    if len(rows) > _MAX_ROWS or not rows:
        raise ValueError("instruction_view_empty_or_exceeds_bound")
    return rows, False, (decoder_isa if format == "elf" else _LISTING_DECODER_ISA)


def _listing_symbol_span(rows, symbol):
    """Keep local listing labels inside a selected named static span.

    Text has no STT_FUNC authority. This only avoids treating LLVM local
    relocation/branch and RISC-V mapping labels as new functions; a following
    different nonlocal symbol still ends the span. ELF extents use their
    separate byte-checked owner above.
    """

    def label(row):
        value = row.section.partition("<")[2]
        return value[:-1] if value.endswith(">") else ""

    starts = [
        i for i, row in enumerate(rows) if label(row) == symbol and (i == 0 or rows[i - 1].section != row.section)
    ]
    if len(starts) != 1:
        raise ValueError("analysis_symbol_not_unique")
    first = starts[0]
    section = rows[first].section_name
    end = first + 1
    while end < len(rows):
        current = rows[end]
        name = label(current)
        if current.section_name != section or (
            name and name != symbol and not name.startswith((".L", "$x", "$d", symbol + "+0x"))
        ):
            break
        end += 1
    return rows[first:end]


def _trip_counts(projection):
    """Lifter-proved per-loop trip counts; ``unknown`` when the lifter proves none."""
    nest = (projection or {}).get("loop_nest") or {}
    levels = (nest.get("value") or {}).get("levels") or ()
    proved = [
        dict(header_pc=level["header_pc"], nest_depth=level.get("nest_depth"), value=level["trip_count"]["value"])
        for level in levels
        if (level.get("trip_count") or {}).get("status") == "proved" and "value" in level["trip_count"]
    ]
    if not proved:
        return "unknown"
    return dict(
        status="proved" if len(proved) == len(levels) else "partial",
        loops=len(levels),
        proved=proved,
        basis="lifter_loop_analysis",
    )


def _without_pinned_locations(value):
    if isinstance(value, dict):
        pinned = "sha256" in value and "path" in value
        return {key: _without_pinned_locations(item) for key, item in value.items() if not (pinned and key == "path")}
    if isinstance(value, list):
        return [_without_pinned_locations(item) for item in value]
    return value


def _protocol_decoder_structure(selection):
    """Field kinds and layout roster from the selected, byte-pinned operand protocol.

    The protocol's own declarations win. Without a declared roster, the derived
    layout roster is computed from the sidecar's RTL layouts; absent field kinds
    stay absent (unknown), never guessed.
    """
    if not isinstance(selection, dict) or "sidecar" not in selection:
        return {}
    from merlin.target.isa.rocc import operand_protocol as protocol

    declared = getattr(protocol, "selected_decoder_structure", None)
    if declared is not None:
        result = {key: value for key, value in declared(selection).items() if value}
    else:
        from merlin.analysis.kernels.effects.command_streams import read_pinned_sidecar

        sidecar = read_pinned_sidecar(selection) or {}
        result = {key: sidecar[key] for key in ("operand_field_kinds", "operand_layout_roster") if sidecar.get(key)}
    if "operand_layout_roster" not in result:
        from merlin.analysis.kernels.effects.command_streams import read_pinned_sidecar
        from merlin.target.isa.rocc.operand_recipe import layout_field_roster

        sidecar = read_pinned_sidecar(selection) or {}
        try:
            result["operand_layout_roster"] = layout_field_roster(sidecar)
        except (KeyError, TypeError, AttributeError):
            logging.getLogger(__name__).warning(
                "Selected operand layout roster unavailable; remains unknown", exc_info=True
            )
    return result


def _operand_decoder(contract, facts, target, *, descriptor=None, include_structure=False):
    """Original host selections only; the artifact bundle cannot supply these inputs."""
    if descriptor is not None:
        from merlin.target.registry.descriptor_loading import (
            load_target_experiment,
            selected_experiment_contract,
        )

        experiment = load_target_experiment(descriptor)
        if experiment.target != target:
            raise ValueError("selected descriptor and decoder target differ")
        # A capsule contract directory is a distinct input role. The existing
        # descriptor owner resolves the actual capability contract and refuses
        # conflicting process selections; neither the request nor bundle does.
        contract = selected_experiment_contract(experiment, environment=os.environ)
    if contract is None or facts is None or not target:
        return {}
    import yaml

    from merlin.target.isa.rocc.semantics import bind

    contract_raw, contract_pin = _read(Path(contract).absolute(), "selected_target_contract")
    facts_raw, facts_pin = _read(Path(facts).absolute(), "selected_rtl_facts")
    contract_document = yaml.safe_load(contract_raw)
    # Decoder identity is the contract's content: a release relocates content-pinned inputs (an entry
    # with both path and sha256), so their locations never distinguish two selections.
    contract_pin = dict(
        path=contract_pin["path"],
        sha256=document_digest(_without_pinned_locations(contract_document)),
        identity="contract content; locations of sha256-pinned entries excluded",
    )
    selected = bind(target=target, contract=contract_document, facts=json.loads(facts_raw))
    observer = selected.operand_observer
    structure = {}
    if include_structure and observer is not None:
        from merlin.analysis.observation.instruction_roles import from_isa_role, is_role

        table, _ = _table(facts)
        names = table.get("names") or {}
        role_bindings = (contract_document.get("corpus_instruction_roles") or {}).get("by_role") or {}
        isa = observer.isa_constants(target)
        normalized = isa["FUNCT_CLASS"]
        roles = {}
        for raw_role, instructions in role_bindings.items():
            role = from_isa_role(raw_role) or (raw_role if is_role(raw_role) else None)
            if role is None:
                continue
            for code, name in names.items():
                if name in instructions and int(code) in normalized:
                    roles.setdefault(normalized[int(code)], []).append(role)
        from merlin.analysis.observation.compute_endpoints import roles_from_table, spec_path

        endpoint_path = spec_path().absolute()
        endpoint_missing = {}
        endpoint_pins = []
        if endpoint_path.is_file():
            endpoint_raw, endpoint_pin = _read(endpoint_path, "selected_compute_endpoints")
            raw_roles, endpoint_missing = roles_from_table(yaml.safe_load(endpoint_raw), target=target, table=table)
            endpoint_pins.append(endpoint_pin)
            for code, name in names.items():
                if int(code) in normalized:
                    bound = roles.setdefault(normalized[int(code)], [])
                    bound.extend(role for role in raw_roles.get(name, ()) if role not in bound)
        # Configuration refinements inherit the selected base command's roles.
        for name in isa.get("CONFIG_SUBTYPE", {}).values():
            bound = roles.setdefault(name, [])
            bound.extend(role for role in roles.get(isa.get("CONFIG_CLASS"), ()) if role not in bound)
        # The generic legality protocol supplies roles for normalized classes,
        # including commands omitted from a minimal operation-presence census.
        # These are protocol role names, never target instruction spellings.
        protocol_roles = {
            "movement_in": ("operand_load",),
            "movement_out": ("readout", "commit"),
            "compute": ("accumulate",),
            "preload": ("weight_load",),
            "config_compute": ("config",),
            "config_load": ("config",),
            "config_store": ("config",),
            "fence": ("sync",),
        }
        checks = contract_document.get("rtl_checks")
        # Older contracts declare a list of checks, without a role protocol.
        declared_classes = checks.get("class_roles", {}) if isinstance(checks, dict) else {}
        normalized_names = set(normalized.values()) | set(isa.get("CONFIG_SUBTYPE", {}).values())
        for name, declared_roles in declared_classes.items():
            if name not in normalized_names:
                continue
            bound = roles.setdefault(name, [])
            for declared_role in declared_roles:
                for role in protocol_roles.get(declared_role, ()):
                    if role not in bound:
                        bound.append(role)
        structure = {key: contract_document.get(key) for key in ("compute_units", "output_epilogue")}
        structure["instruction_roles"] = roles
        structure["operand_protocol_roles"] = declared_classes
        structure["missing_declared_instruction_roles"] = endpoint_missing
        structure.update(_protocol_decoder_structure(contract_document.get("rocc_operand_selection")))
    else:
        endpoint_pins = []
    from merlin.analysis.observation.elf_artifacts import contract_host_isa

    host_isa = contract_host_isa(contract_document)
    return (
        dict(
            semantics=observer,
            target=target,
            decoder_inputs=[contract_pin, facts_pin],
            host_isa=None if host_isa is None else dict(host_isa, contract_sha256=contract_pin["sha256"]),
            **(dict(structure=structure, role_inputs=endpoint_pins) if include_structure else {}),
        )
        if observer is not None
        else {}
    )


for _export in (
    _table,
    _selected_roles_of,
    _instructions,
    _listing_symbol_span,
    _trip_counts,
    _without_pinned_locations,
    _protocol_decoder_structure,
    _operand_decoder,
):
    _export.__module__ = "merlin.development.backend.feedback.references.analysis"
