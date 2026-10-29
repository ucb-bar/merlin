"""Selected, read-only analysis of author-visible references and candidate bytes.

This is an existing selfcheck diagnostic action, not an evaluator or compiler
runner. It imports no candidate code and reads no oracle, golden or score cache.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import resource
import stat
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from merlin.verify.lean_contract import document_digest

from . import reference_experiments

SCHEMA = "merlin.reference_analysis.v1"
ACTION = "reference-analysis"
_FORMATS = ("auto", "c_source", "python_schedule", "listing", "disassembly", "elf", "schedule_mlir",
            "semantic_region", "interface_mlir", "source_mlir", "emitted_mlir", "command_buffer", "artifact_views",
            "candidate_build")
_MAX_BYTES = 256 << 20
_MAX_TEXT = 32 << 20
_MAX_ROWS = 2_000_000


class _AnalysisGap(ValueError):
    """Bounded actionable parser/schema detail, separate from source evidence."""

    def __init__(self, code, **detail):
        super().__init__(code)
        self.detail = detail


def broker_response(request, *, workspace, facts, compiler_sha256, rid, started_ns,
                    bundle_manifest=None, repo=None, contract=None, target=None, descriptor=None):
    """Decorate a diagnostic through the existing broker without grading it."""
    from ..brokers.selfcheck import _decorate_response

    body = handle(request, workspace=workspace, facts=facts, compiler_sha256=compiler_sha256,
                  bundle_manifest=bundle_manifest, repo=repo, contract=contract, target=target, descriptor=descriptor)
    document = _decorate_response(json.dumps(body), request, rid, compiler_sha256,
                                  started_ns=started_ns, completed_ns=time.time_ns())
    document["official"] = False
    return _bounded_response(document, workspace=workspace, pretty=True)


def _relative(value):
    if not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value:
        raise ValueError("input_requires_a_visible_relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in (".", "..") for part in path.parts):
        raise ValueError("input_requires_a_visible_relative_path")
    return path


def _path(workspace, value, *, candidate=False, reference_root=None):
    relative = _relative(value)
    parts = relative.parts
    allowed = (parts[0] in ("reference-kernels", "cca-evidence") if not candidate else
               parts[0] in ("submission", "selfcheck_out", "artifacts") or
               (len(parts) > 2 and parts[0] == "variants" and parts[1] not in (".", "..")))
    if not allowed:
        raise ValueError("input_outside_selected_visible_roots")
    root = workspace if candidate or reference_root is None or parts[0] == "cca-evidence" else reference_root
    path = workspace / relative if root == workspace else root.joinpath(*parts[1:])
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.resolve().is_relative_to(root):
        raise ValueError("linked_or_foreign_analysis_input")
    return path, relative.as_posix()


def _reference_root(workspace, bundle_manifest, repo):
    """Resolve the original read-only grant to frozen bytes, never its live alias."""
    if bundle_manifest is None:
        # Physical copies remain useful for explicit copy-mode diagnostics.
        # A sandbox alias cannot acquire authority through this fallback.
        if (workspace / "reference-kernels").is_symlink():
            raise ValueError("reference_alias_requires_original_bundle")
        return None
    import yaml

    from merlin.targetgen.sandbox import bwrap as BW

    manifest = Path(bundle_manifest)
    if (manifest.is_relative_to(workspace) or manifest.is_symlink() or not manifest.is_file()
            or manifest.stat().st_size > 4 << 20 or repo is None):
        raise ValueError("reference_analysis_requires_original_bundle")
    bundle = yaml.safe_load(manifest.read_text())
    grants = [row for row in bundle.get("allowed", []) if row.get("as") == "reference-kernels"]
    if len(grants) != 1 or grants[0].get("mode") != "ro":
        raise ValueError("reference_analysis_requires_one_readonly_reference_grant")
    source = BW.resolve_grant(grants[0]["path"], Path(repo)).absolute()
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [source], repo=Path(repo))
    alias = workspace / "reference-kernels"
    if not alias.is_symlink() or alias.readlink() != source:
        raise ValueError("reference_alias_does_not_match_original_grant")
    return frozen


def _read(path, label, *, allow_elf=False):
    # Pin each ancestor by directory FD. A concurrently replaced workspace
    # parent cannot redirect the host reader through a symlink into private data.
    from merlin.compile.analysis.artifact_reader import PinnedELF, open_ordinary

    descriptor = open_ordinary(path)
    try:
        is_elf = allow_elf and os.pread(descriptor, 4, 0) == b"\x7fELF"
    except BaseException:
        os.close(descriptor)
        raise
    if is_elf:
        artifact = PinnedELF(descriptor, label)
        return artifact, artifact.pin
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_BYTES:
            raise ValueError("input_not_a_bounded_ordinary_file")
        raw = stream.read(_MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(raw) > _MAX_BYTES or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("analysis_input_changed")
    return raw, dict(path=label, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _member(raw, name):
    _relative(name)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        matches = [row for row in archive.infolist() if row.filename == name]
        if (len(matches) != 1 or matches[0].is_dir() or matches[0].file_size > _MAX_BYTES
                or stat.S_ISLNK(matches[0].external_attr >> 16)):
            raise ValueError("archive_member_not_unique_bounded_or_ordinary")
        with archive.open(matches[0]) as stream:
            payload = stream.read(_MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES:
            raise ValueError("archive_member_exceeds_bound")
        return payload


def _source_summary(raw, format, offset, limit, *, symbol="", locator=None):
    from merlin.kernels.reference_inventory import inspect_source

    result = inspect_source(raw, format)
    functions = result.get("functions", [])
    if not functions and result.get("diagnostics"):
        code = ("analysis_source_parser_unavailable" if result.get("parser") == "unavailable"
                else "analysis_source_parse_failed")
        diagnostics = [{key: str(row[key])[:256] for key in ("kind", "reason", "next_action") if key in row}
                       for row in result["diagnostics"][:8]]
        raise _AnalysisGap(code, parser=result.get("parser"), diagnostics=diagnostics)
    if symbol:
        functions = [row for row in functions if row.get("symbol") == symbol]
        if len(functions) != 1:
            raise ValueError("analysis_symbol_not_unique")
    elif (locator or {}).get("lines"):
        start, end = locator["lines"]
        functions = [row for row in functions if start <= row.get("line", 0)
                     and row.get("end_line", end + 1) <= end]
        if not functions:
            raise ValueError("analysis_region_empty")
    elif (locator or {}).get("address_range"):
        raise ValueError("analysis_locator_unsupported")
    rows = []
    for function in functions[offset:offset + limit]:
        rows.append({key: function[key] for key in ("symbol", "line", "end_line", "body_sha256") if key in function})
        rows[-1].update(loop_count=len(function.get("loops", [])),
            loop_witnesses=function.get("loops", [])[:4],
            calls=dict(list(function.get("calls", {}).items())[:24]),
            parameters=[str(p)[:160] for p in function.get("parameters", [])[:16]])
    return dict(kind="source_syntax", parser=result.get("parser"), total=len(functions), functions=rows,
                next_offset=offset + len(rows) if offset + len(rows) < len(functions) else None,
                diagnostics=result.get("diagnostics", [])[:8], semantic_region_status="unknown")


def _table(path):
    if path is None:
        return {}, None
    from merlin.common.facts_view import interface

    raw = Path(path).read_bytes()
    if len(raw) > 16 << 20:
        raise ValueError("selected_facts_exceed_bound")
    document = json.loads(raw)
    body = document.get("facts", document)
    return interface(body, "funct_decode_table") or {}, dict(
        path="selected_rtl_facts", sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _instructions(raw, format, *, extent=None):
    from merlin.compile.analysis.artifact_reader import PinnedELF
    from merlin.compile.analysis.disassembly import objdump_bin, tokenize_text
    from merlin.perf.pc_instruction_listing import read_listing

    if isinstance(raw, PinnedELF):
        if extent is not None:
            selected = raw.function_disassembly(extent["symbol"])
            return tuple(tokenize_text(selected["text"])), False
        return raw.instructions(), False

    if format == "elf":
        temporary_root = os.environ.get("TMPDIR")
        if not temporary_root or not Path(temporary_root).is_absolute():
            raise ValueError("analysis_requires_owner_task_temporary_root")
        with tempfile.TemporaryDirectory(prefix="reference-analysis-", dir=temporary_root) as directory:
            source = Path(directory) / "input.elf"
            source.write_bytes(raw)
            if extent is not None:
                from merlin.compile.analysis.artifact_reader import open_ordinary

                selected = PinnedELF(open_ordinary(source), "selected_elf")
                try:
                    return tuple(tokenize_text(selected.function_disassembly(extent["symbol"])["text"])), False
                finally:
                    selected.close()
            def bound_output():
                resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_TEXT, _MAX_TEXT))
            with tempfile.TemporaryFile(dir=directory) as output, tempfile.TemporaryFile(dir=directory) as errors:
                command = [objdump_bin(), "-d", "-M", "no-aliases"]
                if extent is not None:
                    # Relocatable functions in different sections can share a
                    # PC. Select the symbol's ELF-owned section before its range.
                    command.append("--section=" + extent["section"])
                process = subprocess.run([*command, str(source)],
                    stdout=output, stderr=errors, timeout=30, preexec_fn=bound_output)
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
        return read_listing(text.splitlines()), True
    rows = tokenize_text(text)
    if len(rows) > _MAX_ROWS or not rows:
        raise ValueError("instruction_view_empty_or_exceeds_bound")
    return rows, False


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

    starts = [i for i, row in enumerate(rows) if label(row) == symbol
              and (i == 0 or rows[i - 1].section != row.section)]
    if len(starts) != 1:
        raise ValueError("analysis_symbol_not_unique")
    first = starts[0]
    section = rows[first].section_name
    end = first + 1
    while end < len(rows):
        current = rows[end]
        name = label(current)
        if current.section_name != section or (name and name != symbol
                and not name.startswith((".L", "$x", "$d", symbol + "+0x"))):
            break
        end += 1
    return rows[first:end]


def _lift(raw, format, pin, *, facts, offset, limit, candidate, symbol="", locator=None, ccas=None,
          mapped_symbol=False, region="", decoder=None, instruction_offset=0, summary_only=False):
    from merlin.compile.analysis.rocc_commands import audit_decoded, decode_stream
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
            from merlin.compile.analysis.artifact_reader import PinnedELF

            extent = raw.function_extent(symbol) if isinstance(raw, PinnedELF) else linked_entry_extent(raw, symbol)
            if extent["size_bytes"] <= 0:
                raise ValueError("function has no retained extent")
        except ValueError as error:
            raise ValueError("analysis_elf_symbol_boundary_unavailable") from error
    if locator.get("lines") and format != "elf":
        start, end = locator["lines"]
        raw = b"\n".join(raw.splitlines()[start - 1:end])
    elif locator.get("lines") and not (symbol or locator.get("address_range")):
        raise ValueError("analysis_locator_unsupported")
    rows, expanded = _instructions(raw, format, extent=extent)
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
        scopes = [(dict(region_id=0, symbol=symbol, entry_pc=extent["address"],
                        static_function_extent=extent), rows)]
    elif expanded:
        scopes = [(r.to_dict(), [rows[index] for index in r.indices]) for r in listing_regions(rows)]
    elif symbol and format == "disassembly" and any("<" in row.section for row in rows):
        body = _listing_symbol_span(rows, symbol)
        scopes = [(dict(region_id=0, symbol=symbol, entry_pc=body[0].addr,
                        symbol_status="listing_label_inferred_static_span"), body)]
    elif any(getattr(row, "section_name", "") for row in rows) and not symbol and not locator:
        from merlin.compile.analysis.riscv_machine import region_index

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
        scopes = [(dict(region_id=i,
                        symbol=label.partition("<")[2][:-1] if "<" in label and label.endswith(">") else label,
                        entry_pc=body[0].addr), body)
                  for i, (label, body) in enumerate(grouped.items())]
    for scope, body in scopes:
        scope.setdefault("accelerator_sites", sum(1 for row in body if len(row.hexcode) == 8
                         and int(row.hexcode, 16) & 127 == table.get("custom_opcode")))
        scope.setdefault("stable_id", scope["region_id"] if isinstance(scope["region_id"], str) else
                         "region-" + document_digest(dict(
            artifact=pin["sha256"], section=getattr(body[0], "section_name", "") or body[0].section,
            start=body[0].addr, end=body[-1].addr + len(body[-1].hexcode) // 2))[:24])
    if region:
        scopes = [(scope, body) for scope, body in scopes if region in (scope["region_id"], scope["stable_id"])]
        if len(scopes) != 1:
            raise ValueError("analysis_region_not_unique")
    if symbol and mapped_symbol and (locator.get("lines") or address) and all(not row.section for row in rows):
        # The map labels an explicit span; its function is not a recovered symbol.
        scopes = [(dict(scope, mapped_function=symbol, symbol_status="not_in_listing"), body)
                  for scope, body in scopes]
    elif symbol:
        scopes = [(scope, body) for scope, body in scopes if scope.get("symbol") == symbol]
        if len(scopes) != 1:
            raise ValueError("analysis_symbol_not_unique")
    result = []
    for scope, body in scopes[offset:offset + limit]:
        from merlin.kernels.instruction_comparison import instruction_summary

        if summary_only:
            result.append(dict(**scope, listed_instructions=len(body),
                               semantic_region_status="unknown",
                               next_action="Lift this stable region to recover selected ISA operands "
                                           "and CCA witnesses."))
            continue
        text = normalized_text(body)
        pins = [ArtifactPin("capture", pin["path"], pin["sha256"], pin["size_bytes"]),
                ArtifactPin("disassembly", pin["path"] + "#normalized-region-" + str(scope["region_id"]),
                            hashlib.sha256(text.encode()).hexdigest())]
        if fact_pin:
            pins.append(ArtifactPin("facts", fact_pin["path"], fact_pin["sha256"], fact_pin["size_bytes"]))
        evidence = KernelEvidence("visible-analysis-" + pin["sha256"][:16] + "-" + str(scope["region_id"]),
            artifacts=tuple(pins), capture_kind="static_disassembly", complete=False,
            missing=("dynamic_loop_multiplicity", "register_values", "memory_values", "timing", "semantic_region"),
            metadata=dict(count_basis="listed_instruction_sites", selected_target_correspondence="unknown"))
        # The existing analyzer refuses absent selected facts rather than
        # inventing an ISA. Evidence-only decoding remains available as a gap.
        model = lift_schedule(body, table=table, evidence=evidence) if fact_pin else None
        cca = cca_projection(model, op="unknown", source="candidate" if candidate else "reference") if model else None
        projection = None
        if model and decoder:
            from merlin.kernels.machine_evidence import machine_projection

            projected, projection = machine_projection(body, model=model, evidence_id=evidence.evidence_id,
                                                       decoder=decoder, instruction_offset=instruction_offset)
            if projected is not None:
                cca = projected
        if ccas is not None:
            ccas.append(cca)
        cfg = control_flow(body)
        audit = audit_decoded(decode_stream(body, table), table=table).to_dict()
        audit["unaccounted_total"] = len(audit["unaccounted"])
        audit["unaccounted"] = audit["unaccounted"][:8]
        structure = None
        if model is not None:
            structure = dict(identity=model.identity, source_kind=model.source_kind,
                command_sites=model.command_count, complete=model.complete, unknowns=model.unknowns,
                role_counts=dict(model.role_counts),
                stage_witnesses=[dict(index=s.index, identity=s.identity, kind=s.kind, roles=s.roles,
                                     parameters=dict(s.parameters)) for s in model.stages[:8]],
                omitted_stages=max(0, len(model.stages) - 8),
                dependencies=len(model.dependencies), storage_lifetimes=len(model.storage),
                realization="unknown")
        result.append(dict(**scope, listed_instructions=len(body),
            instruction_view=instruction_summary(body, decoder=decoder, offset=instruction_offset),
            mnemonics=dict(Counter(row.mnemonic for row in body)),
            selected_decode=audit, evidence=evidence.to_dict(),
            cfg=dict(blocks=len(cfg["blocks"]), natural_loops=len(cfg["natural_loops"]),
                     unresolved_edges=len(cfg["unresolved"]), trip_counts="unknown",
                     status="unknown" if any(not row.hexcode for row in body) else "analyzed",
                     missing_encodings=sum(not row.hexcode for row in body)),
            schedule_model=structure, cca=None if cca is None else cca.to_dict(),
            mechanism_projection=projection or dict(status="unknown",
                code="selected_decoder_unavailable" if fact_pin is None else "runtime_operand_projection_required",
                missing_frontend="control-flow-aware machine-register known-bits analysis",
                library_routes=["merlin.targetgen.rocc.selected_operand_protocol:bind_original_operand_selection",
                                "merlin.targetgen.rocc.static_operands:known_integer_bits",
                                "merlin.perf.dma_volume:propagate_constants"],
                next_action="Use read-command-buffer and actual schedule consumer receipts; instruction sites alone "
                            "do not establish residency, overlap or execution counts.")))
    return dict(kind="static_instruction_analysis", count_basis="listed_instruction_sites",
                listed_instructions=len(rows), unique_pcs=len({r.addr for r in rows}),
                total=len(scopes), offset=offset, regions=result,
                next_offset=offset + len(result) if offset + len(result) < len(scopes) else None,
                coverage=dict(accelerator_sites=sum(1 for row in rows if len(row.hexcode) == 8
                    and int(row.hexcode, 16) & 127 == table.get("custom_opcode")),
                    selected_region_sites=sum(scope.get("accelerator_sites", sum(
                        1 for row in body if len(row.hexcode) == 8
                        and int(row.hexcode, 16) & 127 == table.get("custom_opcode"))) for scope, body in scopes),
                    semantic_readiness="unknown", count_basis="listed_instruction_sites"),
                selected_facts=fact_pin, selected_target_correspondence="unknown")


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
    summary = {key: _brief_analysis(report[key]) for key in (
        "input", "total", "offset", "next_offset", "count_basis", "count_scope", "source_binding",
        "source_command_join", "declared_contraction_alignment", "command_role_comparison",
        "unmatched_candidate_commands", "tile_dataflow", "semantic_equivalence", "semantic_compatibility",
        "numerical_status", "completion", "overlap", "performance", "schedule_consumption", "next_action",
        "gaps", "upstream_semantics", "producer_correspondence", "instruction_comparison", "operand_comparison",
        "machine_comparison", "emitted_comparison", "mechanism_observations", "mechanism_alignment",
        "consumer_structure", "producer_binding", "construction_receipts", "construction_comparison",
        "candidate_build_publication", "function_view_input") if key in report}
    summary.update(kind=report.get("kind"),
                   projection="at most four list entries and bounded nested fields; full evidence in detail")
    comparison = report.get("cca")
    if isinstance(comparison, dict):
        summary["cca"] = {key: _brief_analysis(comparison[key]) for key in (
            "status", "denominator", "compared_axes", "divergences") if key in comparison}
        summary["cca"]["unknown_axes"] = len(comparison.get("unknowns", []))
    experiment = report.get("experiment")
    if isinstance(experiment, dict):
        summary["experiment"] = dict(status=experiment.get("status"), identity=experiment.get("identity"),
            total_tasks=len(experiment.get("tasks", [])), tasks=[{
                key: _brief_analysis(task[key]) for key in ("task_id", "status", "goals", "summary", "next_action")
                if key in task} | dict(request_ids=[row["request_id"] for row in task.get("requests", [])[:2]
                                                   if "request_id" in row])
                for task in experiment.get("tasks", [])[:4]])
    return summary


def _bounded_response(response, *, workspace, pretty=False):
    """Bound the final transport without republishing analysis or replacing its detail."""
    # The lean writer uses indent=2 plus a newline; compact JSON alone can fit
    # while the actual channel file exceeds the bound after decoration.
    indent = 2 if pretty else None
    if len(json.dumps(response, indent=indent).encode()) + 1 <= 64 << 10:
        return response
    from .result_details import MAX_DETAIL_BYTES, compact_row, load_row

    metadata = {key: value for key, value in response.items()
                if key not in ("report", "reference_experiments", "next_steps")}
    try:
        if workspace is None:
            raise ValueError("analysis_detail_workspace_unavailable")
        if response.get("report_in_detail"):
            full = load_row(response, workspace=workspace)
            if (full is response or full.get("report_identity") != response.get("report_identity")
                    or document_digest(full.get("report", {})) != response.get("report_identity")):
                raise ValueError("analysis_detail_unavailable_or_changed")
            detail = response["detail"]
        else:
            if len(json.dumps(response).encode()) > MAX_DETAIL_BYTES:
                raise ValueError("analysis_detail_exceeds_bound")
            full = response
            detail = compact_row(dict(name="reference_analysis", **response), workspace=workspace)["detail"]
    except (OSError, ValueError, TypeError, KeyError):
        # A genuine artifact refusal remains a gap; summary size alone is not one.
        metadata.update(status="unknown", reason="analysis_report_detail_unavailable",
                        next_action="Restore the hash-pinned analysis detail before retrying.")
        return metadata
    report = full.get("report", {})
    compact = dict(metadata, report=_analysis_summary(report), detail=detail, report_in_detail=True,
        next_action="Read the hash-pinned detail for all evidence and experiment requests.")
    feedback = response.get("reference_experiments")
    if isinstance(feedback, dict):
        compact["reference_experiments"] = {key: _brief_analysis(feedback[key]) for key in (
            "schema", "status", "packet_identity", "execution_context_identity", "task_id", "task_status",
            "next_action", "experiment_status", "emitted_effects", "observed_requests", "advisory_only",
            "qualification_issued", "completion_authority") if key in feedback}
        compact["reference_experiments"].update(projection="bounded current feedback; full packet/outcomes stay owned",
            required_selfchecks=[{key: row[key] for key in ("request_id", "role", "capsule", "argv") if key in row}
                                 for row in feedback.get("required_selfchecks", [])[:2]])
    if isinstance(response.get("next_steps"), dict):
        compact["next_steps"] = {key: _brief_analysis(response["next_steps"][key])
            for key in ("schema", "advisory_only", "lines") if key in response["next_steps"]}
    if len(json.dumps(compact, indent=indent).encode()) + 1 > 64 << 10:
        # Large individual scalar/request fields cannot turn completed analysis
        # into UNKNOWN. Keep exact pointers; do not fabricate a smaller full row.
        compact["report"] = dict(kind=report.get("kind"),
                                 projection="read full detail; compact evidence exceeded transport bound")
        if isinstance(feedback, dict):
            compact["reference_experiments"] = {key: feedback[key] for key in (
                "packet_identity", "execution_context_identity", "task_id", "task_status") if key in feedback}
            compact["reference_experiments"]["request_ids"] = [
                row["request_id"] for row in feedback.get("required_selfchecks", [])[:2] if "request_id" in row]
        compact.pop("next_steps", None)
    return compact


def _response(base, packet, report, compiler_sha256, *, workspace=None):
    experiment = None
    if workspace is not None and compiler_sha256:
        try:
            experiment = reference_experiments.record_analysis(
                report, workspace=workspace, compiler_sha256=compiler_sha256)
        except (OSError, ValueError, TypeError, KeyError):
            experiment = dict(status="unknown", reason="current_analysis_experiment_binding_unavailable",
                              tasks=[], requests=[])
    if experiment is not None and (
            experiment.get("tasks") or report.get("gaps") or experiment.get("status") == "unknown"):
        report = dict(report, experiment=experiment)
    elif report.get("gaps"):
        # Reuse the one experiment engine and its exact selected context. A
        # diagnostic cannot authorize a new evaluator, knob or capsule binding.
        from . import reference_views

        report = dict(report, experiment=reference_views.experiment(
            report, packet=packet, compiler_sha256=compiler_sha256))
    producer = report.get("producer_binding") or (report.get("candidate") or {}).get("producer_binding") or {}
    artifact_binding = "unknown"
    if producer.get("status") == "observed_original_build_consumption" and compiler_sha256:
        artifact_binding = ("same_compiler_snapshot_digest" if producer["compiler_sha256"] == compiler_sha256
                            else "different_original_compiler_snapshot")
    response = dict(base, status="analyzed", selection_identity=packet["identity"], report=report,
                    report_identity=document_digest(report), current_compiler_sha256=compiler_sha256,
                    artifact_to_current_compiler_binding=artifact_binding)
    return _bounded_response(response, workspace=workspace)


_GAPS = {
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
}


def _entry(value):
    """A selected map is a locator/hypothesis source, never a semantic proof."""
    identifier = value.get("entry")
    if not identifier:
        return None
    from . import reference_map

    document = reference_map.selected()
    entries = [row for row in (document or {}).get("entries", []) if row["id"] == identifier]
    if len(entries) != 1:
        raise ValueError("analysis_entry_unavailable")
    entry = entries[0]
    path = "reference-kernels/" + entry["file"]
    if value.get("path") and value["path"] != path:
        raise ValueError("analysis_entry_path_mismatch")
    return entry


def _analyze(value, *, workspace, reference_root, facts, candidate=False, compare=False,
             contract=None, target=None, descriptor=None, candidate_build_manifest=None):
    entry = None if candidate else _entry(value)
    label = "reference-kernels/" + entry["file"] if entry else value.get("path", "reference-kernels")
    path, label = _path(workspace, label, candidate=candidate, reference_root=reference_root)
    raw, pin = _read(path, label, allow_elf=True)
    from merlin.compile.analysis.artifact_reader import PinnedELF

    try:
        return _analyze_open(value, raw=raw, pin=pin, label=label, entry=entry, workspace=workspace,
                             reference_root=reference_root, facts=facts, candidate=candidate, compare=compare,
                             contract=contract, target=target, descriptor=descriptor,
                             candidate_build_manifest=candidate_build_manifest)
    finally:
        if isinstance(raw, PinnedELF):
            raw.close()


def _analyze_open(value, *, raw, pin, label, entry, workspace, reference_root, facts,
                  candidate=False, compare=False, contract=None, target=None, descriptor=None,
                  candidate_build_manifest=None):
    member = value.get("member")
    if member:
        raw = _member(raw, member)
        pin = dict(path=label + "#" + member, sha256=hashlib.sha256(raw).hexdigest(),
                   size_bytes=len(raw), container=pin)
    format = value.get("format", "auto")
    if format == "auto":
        suffix = Path(member or label).suffix.lower()
        format = ("elf" if raw[:4] == b"\x7fELF" else "c_source" if suffix in (".c", ".h")
                  else "python_schedule" if suffix == ".py"
                  else "interface_mlir" if suffix == ".mlir" and b"merlin_iface." in raw
                  else "schedule_mlir" if suffix == ".mlir"
                  else "artifact_views" if suffix == ".json" and raw.lstrip().startswith(b"{")
                  else "listing" if raw.lstrip().startswith(b"0x") else "disassembly")
        if format == "artifact_views":
            document = json.loads(raw)
            if isinstance(document, dict) and {"tensors", "commands"} <= document.keys():
                format = "command_buffer"
            elif isinstance(document, dict) and document.get("schema") == "merlin.candidate_build_view.v1":
                format = "candidate_build"
    offset, limit = value.get("offset", 0), 1 if compare else value.get("limit", 8)
    locator = entry.get("locator", {}) if entry else {}
    symbol = value.get("symbol") or locator.get("function", "")
    ccas = []
    if format == "candidate_build":
        if not candidate:
            raise ValueError("candidate_build_requires_owned_candidate")
        from .candidate_build_views import analyze

        return analyze(value, pin, workspace=workspace, bundle_manifest=candidate_build_manifest,
                       facts=facts, contract=contract, target=target, descriptor=descriptor)
    if format == "interface_mlir":
        from .reference_mechanisms import selected_lanes
        from .reference_views import source_view

        lanes, lane_status, lane_inputs = selected_lanes(facts=facts, target=target)
        report = source_view(raw, reduction_lanes=lanes)
        report["mechanism_observations"].update(reduction_geometry=lane_status, selected_inputs=lane_inputs)
        report.pop("command_buffer")
        return dict(report, kind="interface_source_analysis", input=pin), None
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
        if sum(1 for _ in module.walk()) > 100_000:
            raise ValueError("analysis_source_parse_failed")
        observed = semantic_region_from_ir(module, region_id="source-" + pin["sha256"][:24], complete=False)
        from .reference_mechanisms import source_report

        return dict(kind="upstream_ir_analysis", input=pin, semantic_region=observed.to_dict(),
                    mechanism_observations=source_report(module, pin, facts=facts, target=target),
                    semantic_region_status="observed_source_structure",
                    numerical_contract="unknown", emitted_correspondence="unknown"), observed
    if format == "emitted_mlir":
        from .reference_representations import emitted_view

        return emitted_view(raw, pin, decoder=_operand_decoder(contract, facts, target, descriptor=descriptor),
                            symbol=symbol)
    if format == "artifact_views":
        document = json.loads(raw)
        if isinstance(document, dict) and document.get("schema") == "merlin.reference_artifact_views.v2":
            return _representation_bundle(document, pin, workspace=workspace, reference_root=reference_root,
                                          facts=facts, candidate=candidate, contract=contract, target=target,
                                          descriptor=descriptor)
        from .reference_views import artifact_views

        try:
            document = json.loads(raw)
        except ValueError as error:
            raise ValueError("analysis_artifact_views_schema_invalid") from error
        if (not isinstance(document, dict) or document.get("schema") != "merlin.reference_artifact_views.v1"
                or set(document) - {"schema", "source", "command_buffer", "emitted", "entry_symbol"}):
            raise ValueError("analysis_artifact_views_schema_invalid")
        inputs, input_pins = {}, {}
        for role in ("source", "command_buffer", "emitted"):
            selected = document.get(role)
            if not isinstance(selected, dict) or set(selected) != {"path", "sha256"}:
                raise _AnalysisGap("analysis_artifact_views_role_invalid", role=role,
                                   required_fields=["path", "sha256"])
            artifact, visible = _path(workspace, selected["path"], candidate=candidate,
                                      reference_root=reference_root)
            payload, observed = _read(artifact, visible)
            if observed["sha256"] != selected["sha256"]:
                raise ValueError("analysis_artifact_pin_mismatch")
            inputs[role] = payload
            input_pins[role] = observed
        decoder = _operand_decoder(contract, facts, target, descriptor=descriptor, include_structure=True)
        from .reference_mechanisms import selected_lanes

        lanes, lane_status, lane_inputs = selected_lanes(facts=facts, target=target)
        report = artifact_views(source_mlir=inputs["source"], command_buffer=inputs["command_buffer"],
            emitted_mlir=inputs["emitted"], entry_symbol=document.get("entry_symbol", "kernel_entry"),
            reduction_lanes=lanes,
            **{key: decoder[key] for key in ("semantics", "target", "decoder_inputs") if key in decoder})
        report["source"]["mechanism_observations"].update(reduction_geometry=lane_status, selected_inputs=lane_inputs)
        from .reference_capture_views import candidate_view

        projected, cca = candidate_view(inputs, entry_symbol=document.get("entry_symbol", "kernel_entry"),
                                        decoder=decoder)
        return dict(report, input=pin, artifact_inputs=input_pins, captured_reference_axes=projected), cca
    if format == "command_buffer":
        from merlin.perf.command_buffer_diagnostics import representation_activity

        document = json.loads(raw)
        if (not isinstance(document, dict) or not isinstance(document.get("tensors"), dict)
                or not isinstance(document.get("commands"), list)):
            raise ValueError("invalid_analysis_request")
        return dict(kind="declared_command_buffer", input=pin,
            declared_representations=representation_activity(document), source_binding="unknown",
            emitted_mechanism="unknown", semantic_equivalence="unknown",
            next_action="Join original source and emitted LLVM with --write-reference-views; "
                        "declared commands alone do not establish emitted work."), None
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
        return report, region
    if format == "schedule_mlir":
        from xdsl.utils.exceptions import ParseError, VerifyException

        from merlin.compile.analysis.schedule_structure import analyze_ir_schedule
        from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan

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
                is_llvm = any(isinstance(op, llvm.FuncOp) for op in emitted.walk())
            except (ParseError, VerifyException):
                is_llvm = False
            if is_llvm:
                raise ValueError("analysis_schedule_is_emitted_llvm") from error
            raise ValueError("analysis_schedule_invalid") from error
        model = analyze_ir_schedule(module)
        if not model.decisions:
            raise ValueError("analysis_schedule_empty")
        report = dict(kind="schedule_decisions", total=len(model.decisions), input=pin,
            schedule_identity=model.identity, decisions=model.decisions[offset:offset + limit],
            unknowns=model.unknowns, realized=False, source_kind=model.source_kind,
            next_action="Consume this schedule in emission and bind the artifact; annotations alone are intent.")
        return report, None
    if format in ("c_source", "python_schedule"):
        if format == "c_source" and not candidate:
            from .reference_capture_views import lift_selected

            captured = lift_selected(raw, pin, symbol=symbol, workspace=workspace,
                reference_root=reference_root, facts=facts, contract=contract,
                target=target, descriptor=descriptor, limit=limit)
            if captured is not None:
                return captured
        report = _source_summary(raw, format, offset, limit, symbol=symbol, locator=locator)
        if report.get("diagnostics") and not report["functions"]:
            raise ValueError("analysis_source_parse_failed")
    else:
        # A listing does not carry ELF symbols. Its pinned line/PC locator selects the region instead.
        if format == "listing" and locator and not value.get("symbol"):
            symbol = ""
        report = _lift(raw, format, pin, facts=facts, offset=offset, limit=limit, candidate=candidate,
                       symbol=symbol, locator=locator, ccas=ccas,
                       mapped_symbol=entry is not None and not value.get("symbol"),
                       region=value.get("region", ""),
                       instruction_offset=value.get("instruction_offset", 0),
                       summary_only=value.get("operation") == "inventory",
                       decoder=_operand_decoder(contract, facts, target, descriptor=descriptor, include_structure=True))
    if compare and report.get("total") != 1 and not (symbol or offset or value.get("region")):
        raise ValueError("analysis_compare_requires_one_region")
    if compare and len(report.get("regions", report.get("functions", []))) != 1:
        raise ValueError("analysis_compare_requires_one_region")
    report["input"] = pin
    if entry:
        report["reference_entry"] = dict(id=entry["id"], locator=locator, operation=entry["operation"],
                                        confidence=entry["confidence"], semantic_compatibility="unknown")
    return report, ccas[0] if len(ccas) == 1 else None


def _representation_bundle(document, pin, *, workspace, reference_root, facts, candidate,
                           contract, target, descriptor):
    """Reopen each domain independently; supplied associations grant no producer authority."""
    if set(document) != {"schema", "views"} or not isinstance(document["views"], dict):
        raise ValueError("analysis_artifact_views_schema_invalid")
    views = document["views"]
    if not 2 <= len(views) <= 4 or set(views) - {"source", "schedule", "emitted", "binary"}:
        raise ValueError("analysis_artifact_views_schema_invalid")
    reports, ccas = {}, {}
    formats = dict(source={"c_source", "python_schedule", "interface_mlir", "source_mlir", "semantic_region"},
                   schedule={"schedule_mlir", "python_schedule"}, emitted={"emitted_mlir"},
                   binary={"elf", "disassembly", "listing"})
    for role, selected in views.items():
        if (not isinstance(selected, dict) or not {"path", "sha256", "format"} <= set(selected)
                or set(selected) - {"path", "sha256", "format", "symbol", "region"}
                or selected["format"] not in formats[role]):
            raise _AnalysisGap("analysis_artifact_views_role_invalid", role=role)
        request = {key: selected[key] for key in ("path", "format", "symbol", "region") if key in selected}
        request["limit"] = 1
        report, cca = _analyze(request, workspace=workspace, reference_root=reference_root, facts=facts,
                              candidate=candidate, contract=contract, target=target, descriptor=descriptor)
        if report.get("input", {}).get("sha256") != selected["sha256"]:
            raise ValueError("analysis_artifact_pin_mismatch")
        if report.get("total", 1) != 1 and not (selected.get("region") or selected.get("symbol")):
            raise ValueError("analysis_compare_requires_one_region")
        reports[role], ccas[role] = report, cca
    result = dict(kind="representation_bundle", input=pin, views=reports,
                representation_links=[dict(source=a, destination=b, status="supplied_artifact_association",
                                           producer_correspondence="unknown")
                                      for a, b in zip(reports, list(reports)[1:])],
                semantic_compatibility="unknown", emitted_correspondence="unknown",
                next_action="Check each representation and bind original producer/capture correspondence.")
    return result, ccas.get("binary", ccas.get("emitted"))


def _compare(left, right, expert, ours):
    """Freshly analyzed operands, never author-supplied CCA summaries or requested settings."""
    from merlin.kernels.cca_compare import compare_evidence
    from merlin.kernels.kernel_evidence import compare_regions

    if left["kind"] == right["kind"] == "representation_bundle":
        from .reference_representations import compare_bundles

        return compare_bundles(left, right, expert, ours, compare_machine=_compare)

    if left["kind"] == "captured_reference_analysis" and right["kind"] == "artifact_cross_view_analysis":
        from .reference_capture_views import compare_candidate

        return compare_candidate(left, right, expert, ours)
    if left["kind"] != right["kind"]:
        raise ValueError("analysis_compare_domains_differ")
    if left["kind"] == "artifact_cross_view_analysis":
        from .reference_views import compare_views

        return compare_views(left, right)
    if left["kind"] == "declared_semantic_region":
        return dict(kind="declared_semantic_comparison", reference=left["input"], candidate=right["input"],
            declared_contracts=compare_regions(expert, ours).to_dict(),
            artifact_origin="unverified", semantic_equivalence="unknown",
            next_action="Bind both declared contracts to the actual frontend/lowering products before ranking cycles.")
    if left["kind"] == "emitted_ir_analysis":
        from merlin.kernels.instruction_comparison import compare_operand_summaries

        return dict(kind="emitted_instruction_comparison", reference=left, candidate=right,
                    operand_comparison=compare_operand_summaries(left.get("operand_summary", {}),
                                                                 right.get("operand_summary", {})),
                    semantic_compatibility="unknown", schedule_consumption="unknown", performance="unknown")
    if left["kind"] != "static_instruction_analysis":
        raise ValueError("analysis_compare_domains_differ")
    ref, cand = left["regions"][0], right["regions"][0]
    from merlin.kernels.instruction_comparison import compare_instruction_summaries
    comparison = compare_evidence(expert, ours).to_dict() if expert is not None and ours is not None else dict(
        status="unknown", denominator=0, compared_axes=[], divergences=[],
        unknowns=[dict(reason="selected_instruction_decoder_unavailable")])
    ref_counts, cand_counts = ref["selected_decode"]["counts"], cand["selected_decode"]["counts"]
    command_mix = {name: dict(reference=ref_counts.get(name, 0), candidate=cand_counts.get(name, 0))
                   for name in sorted(ref_counts.keys() | cand_counts.keys())
                   if ref_counts.get(name) or cand_counts.get(name)}
    from .reference_schedule_receipts import compare_schedule_receipts

    return dict(kind="reference_candidate_comparison", reference=left, candidate=right, cca=comparison,
        instruction_comparison=compare_instruction_summaries(ref, cand),
        command_site_comparison=command_mix,
        construction_comparison=compare_schedule_receipts(left.get("construction_receipts", {}),
                                                         right.get("construction_receipts", {})),
        instruction_site_delta=cand["listed_instructions"] - ref["listed_instructions"],
        count_basis="listed_instruction_sites; no runtime multiplicity or latency implication",
        mechanism_status="unknown", schedule_consumption="unknown", semantic_compatibility="unknown",
        required_evidence=["same source graph, fusion/live-outs, shapes, layouts and complete numerical contract",
                           "actual schedule.plan consumer and emitted mechanism witness",
                           "source-bound numerical selfcheck and matched complete-region cycle measurements"],
        next_action="Compile alternatives with selected library APIs; reanalyze emission and record missing adapters.")


def _operand_decoder(contract, facts, target, *, descriptor=None, include_structure=False):
    """Original host selections only; the artifact bundle cannot supply these inputs."""
    if descriptor is not None:
        from merlin.targetgen.target_experiment import load_target_experiment, selected_experiment_contract

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

    from merlin.targetgen.rocc.semantics import bind

    contract_raw, contract_pin = _read(Path(contract).absolute(), "selected_target_contract")
    facts_raw, facts_pin = _read(Path(facts).absolute(), "selected_rtl_facts")
    contract_document = yaml.safe_load(contract_raw)
    selected = bind(target=target, contract=contract_document, facts=json.loads(facts_raw))
    observer = selected.operand_observer
    structure = {}
    if include_structure and observer is not None:
        from merlin.compile.analysis.instruction_roles import from_isa_role, is_role

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
        from merlin.compile.analysis.compute_endpoints import roles_from_table, spec_path

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
            "movement_in": ("operand_load",), "movement_out": ("readout", "commit"),
            "compute": ("accumulate",), "preload": ("weight_load",),
            "config_compute": ("config",), "config_load": ("config",), "config_store": ("config",),
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
    else:
        endpoint_pins = []
    return (dict(semantics=observer, target=target, decoder_inputs=[contract_pin, facts_pin],
                 **(dict(structure=structure, role_inputs=endpoint_pins) if include_structure else {}))
            if observer is not None else {})


def handle(request, *, workspace, facts=None, compiler_sha256=None, bundle_manifest=None, repo=None,
           contract=None, target=None, descriptor=None):
    """Whitelisted diagnostic entrypoint called by the existing host broker."""
    base = dict(schema=SCHEMA, all_pass=None, numeric_assessment="not_run", official=False,
                qualification_issued=False, semantic_equivalence="unknown", performance="unknown")
    try:
        selected = reference_experiments._current()
        if selected is None:
            return dict(base, status="denied", reason="reference_guidance_disabled")
        packet, _ = reference_experiments._load(selected)
        value = request.get("reference_analysis")
        if not isinstance(value, dict) or set(value) - {
                "operation", "path", "format", "member", "offset", "limit", "symbol", "entry", "region",
                "instruction_offset", "candidate"}:
            raise ValueError("invalid_analysis_request")
        operation, format = value.get("operation"), value.get("format", "auto")
        offset, limit = value.get("offset", 0), value.get("limit", 8)
        if (operation not in ("inventory", "lift", "candidate", "compare") or format not in _FORMATS
                or type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16):
            raise ValueError("invalid_analysis_operation_or_page")
        for key in ("symbol", "entry", "region"):
            if not isinstance(value.get(key, ""), str) or len(value.get(key, "")) > 512:
                raise ValueError("invalid_analysis_request")
        if type(value.get("instruction_offset", 0)) is not int or value.get("instruction_offset", 0) < 0:
            raise ValueError("invalid_analysis_request")
        other = value.get("candidate")
        if operation == "compare":
            if (not isinstance(other, dict)
                    or set(other) - {"path", "format", "symbol", "offset", "region", "instruction_offset"}
                    or not isinstance(other.get("path"), str) or other.get("format", "auto") not in _FORMATS
                    or not isinstance(other.get("symbol", ""), str) or len(other.get("symbol", "")) > 512
                    or not isinstance(other.get("region", ""), str) or len(other.get("region", "")) > 512
                    or type(other.get("instruction_offset", 0)) is not int or other.get("instruction_offset", 0) < 0
                    or type(other.get("offset", 0)) is not int or other.get("offset", 0) < 0):
                raise ValueError("invalid_analysis_request")
        elif other or operation == "candidate" and value.get("entry"):
            raise ValueError("invalid_analysis_request")
        workspace = Path(workspace).resolve()
        # A before/after baseline is candidate evidence too. It cannot acquire
        # a map entry's authority merely by occupying the left comparison slot.
        own_left = operation == "candidate" or (operation == "compare" and not value.get("entry")
            and isinstance(value.get("path"), str) and value["path"].split("/", 1)[0] != "reference-kernels")
        reference_root = _reference_root(workspace, bundle_manifest, repo) if not own_left else None
        if operation != "inventory":
            left, expert = _analyze(value, workspace=workspace, reference_root=reference_root, facts=facts,
                                    candidate=own_left, compare=operation == "compare",
                                    contract=contract, target=target, descriptor=descriptor,
                                    candidate_build_manifest=bundle_manifest)
            left["input_role"] = "candidate_artifact" if own_left else "selected_reference"
            if operation == "compare":
                right, ours = _analyze(other, workspace=workspace, reference_root=None, facts=facts,
                                       candidate=True, compare=True, contract=contract, target=target,
                                       descriptor=descriptor, candidate_build_manifest=bundle_manifest)
                right["input_role"] = "candidate_artifact"
                left = _compare(left, right, expert, ours)
            return _response(base, packet, left, compiler_sha256, workspace=workspace)
        path, label = _path(workspace, value.get("path", "reference-kernels"),
                            candidate=operation == "candidate", reference_root=reference_root)
        if operation == "inventory" and path.is_dir():
            entries = sorted(p for p in path.rglob("*") if p.is_file() or p.is_symlink())
            if len(entries) > 4096:
                raise ValueError("visible_reference_inventory_exceeds_bound")
            page = []
            for entry in entries[offset:offset + limit]:
                relative = (PurePosixPath(label) / entry.relative_to(path)).as_posix()
                _path(workspace, relative, reference_root=reference_root)
                payload, pin = _read(entry, relative, allow_elf=True)
                from merlin.compile.analysis.artifact_reader import PinnedELF

                if isinstance(payload, PinnedELF):
                    payload.close()
                page.append(pin)
            report = dict(kind="visible_reference_files", total=len(entries), files=page,
                          next_offset=offset + len(page) if offset + len(page) < len(entries) else None)
        elif path.suffix.lower() == ".zip" and not value.get("member"):
            raw, pin = _read(path, label)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                report = dict(kind="visible_archive_members", input=pin, total=len(entries),
                    members=[dict(member=r.filename[:1024], size_bytes=r.file_size, directory=r.is_dir())
                             for r in entries[offset:offset + limit]],
                    next_offset=offset + limit if offset + limit < len(entries) else None)
        else:
            report, _ = _analyze(value, workspace=workspace, reference_root=reference_root, facts=facts,
                                  contract=contract, target=target, descriptor=descriptor)
        return _response(base, packet, report, compiler_sha256, workspace=workspace)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, RecursionError,
            subprocess.SubprocessError, zipfile.BadZipFile) as error:
        code = str(error) if isinstance(error, ValueError) and str(error) in _GAPS else (
            "analysis_dependency_unavailable" if isinstance(error, ImportError)
            else "analysis_input_unavailable_invalid_or_unsupported")
        return dict(base, status="unknown", reason=code,
                    next_action=_GAPS.get(code, "Check admitted paths, format, parser/ISA dependencies and bounds."),
                    gap=dict(kind="evidence_or_adapter_gap", code=code, retryable=code == "analysis_input_changed",
                             **(dict(detail=error.detail) if isinstance(error, _AnalysisGap) else {})))
