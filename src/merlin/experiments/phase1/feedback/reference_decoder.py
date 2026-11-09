"""Decode declared instruction and operand structure from the selected machine evidence."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import resource
import subprocess
import tempfile
from pathlib import Path

from merlin.verification.lean_contract import document_digest

from .reference_inputs import (  # noqa: F401 -- preserve established imports at this boundary
    _MAX_ROWS,
    _MAX_TEXT,
    _read,
)


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
    from merlin.analysis.artifact_reader import PinnedELF
    from merlin.analysis.disassembly import objdump_bin, tokenize_text
    from merlin.perf.pc_instruction_listing import read_listing

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
                from merlin.analysis.artifact_reader import open_ordinary

                selected = PinnedELF(open_ordinary(source), "selected_elf", declared_isa=declared_isa)
                try:
                    listed = selected.function_disassembly(extent["symbol"])
                    return tuple(tokenize_text(listed["text"])), False, listed["decoder_isa"]
                finally:
                    selected.close()

            def bound_output():
                resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_TEXT, _MAX_TEXT))

            from merlin.analysis.artifact_reader import open_ordinary

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
    from merlin.targetgen.rocc import selected_operand_protocol as protocol

    declared = getattr(protocol, "selected_decoder_structure", None)
    if declared is not None:
        result = {key: value for key, value in declared(selection).items() if value}
    else:
        from merlin.kernels.decoded_effects import read_pinned_sidecar

        sidecar = read_pinned_sidecar(selection) or {}
        result = {key: sidecar[key] for key in ("operand_field_kinds", "operand_layout_roster") if sidecar.get(key)}
    if "operand_layout_roster" not in result:
        from merlin.kernels.decoded_effects import read_pinned_sidecar
        from merlin.targetgen.rocc.operand_selection_recipe import layout_field_roster

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
        from merlin.analysis.instruction_roles import from_isa_role, is_role

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
        from merlin.analysis.compute_endpoints import roles_from_table, spec_path

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
    from merlin.analysis.host_isa import contract_host_isa

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


# Preserve established callable pickle identities through the explicit public exports.
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
    _export.__module__ = "merlin.experiments.phase1.feedback.reference_analysis"
