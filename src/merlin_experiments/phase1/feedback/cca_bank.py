"""Pinned reference-bank inputs, decoder selection and emitted-kernel lifting."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

_KERNEL = "package_kernel.elf"


_MAX_JSON = 96 << 20


class Skip(Exception):
    """A stated reason the section is not computed."""


def open_selected_bank(workspace, pin, root=None):
    """The arm's verified bank (``root``, else ``cca-evidence/`` with its alias resolved) and its pinned decoder."""
    from . import reference_bank as RB

    if root is None:
        alias = Path(workspace) / RB.GRANT
        if not alias.exists():
            raise Skip("no cca-evidence/ bank in this workspace")
        root = alias
    elif not Path(root).exists():
        raise Skip(f"bank root {root} does not exist")
    root = Path(os.path.realpath(root))
    try:
        bank = RB.verify(root, pin)
        decoder = RB._bank_decoder(root, bank)
    except (OSError, ValueError) as error:
        raise Skip(f"bank not verified: {error}") from error
    if decoder is None:
        raise Skip("bank carries no pinned operand decoder")
    return root, bank, decoder


def _card_symbols(root, bank, unit):
    """``{occurrence id: symbol}`` from the unit's pinned card; ``{}`` when the card names none."""
    from . import reference_bank as RB

    if not unit.get("card") or unit["card"] not in bank.get("files", {}):
        return {}
    card = json.loads(RB._pinned(root, bank, unit["card"])[0])
    return {
        row.get("occurrence_id"): row["symbol"]
        for row in card.get("occurrences") or ()
        if isinstance(row, dict) and isinstance(row.get("symbol"), str) and row["symbol"]
    }


def bank_units(root, bank):
    """Every bank occurrence as ``{unit_id, region_id, original_id, symbol, analysis, signature}``.

    ``symbol`` is the original's own name for the body (the card's symtab name, else the record's), or
    None when neither names it."""
    from merlin.mining.reference_subset_cards import family_signature

    from . import reference_bank as RB

    rows = []
    for unit in sorted(bank["units"], key=lambda unit: unit["kernel_id"]):
        symbols = _card_symbols(root, bank, unit)
        for occurrence in RB._occurrences(unit):
            analysis = json.loads(RB._pinned(root, bank, occurrence["files"]["analysis.json"])[0])
            symbol = (
                symbols.get(occurrence["original_id"] + ":" + occurrence["region_id"])
                or next(iter(sorted(symbols.values())), None)
                or analysis.get("symbol")
                or (analysis.get("static_function_extent") or {}).get("symbol")
                or None
            )
            rows.append(
                dict(
                    unit_id=unit["kernel_id"],
                    region_id=occurrence["region_id"],
                    original_id=occurrence["original_id"],
                    symbol=symbol,
                    analysis=analysis,
                    signature=family_signature(analysis),
                    operation=None,
                )
            )
    return rows


def selected_decoder(decoder_files, target):
    """The bank's operand decoder and effect structure, bound exactly as the bank build bound them."""
    import yaml

    from merlin.kernels.decoded_effects import selected_structure

    from . import reference_analysis as A

    decoder = A._operand_decoder(decoder_files["contract"], decoder_files["facts"], target, include_structure=True)
    if not decoder or decoder.get("semantics") is None:
        raise Skip("selected operand decoder unavailable")
    structure = selected_structure(decoder, yaml.safe_load(Path(decoder_files["contract"]).read_text()))
    return decoder, structure


def lift_kernel(path, *, decoder, facts, structure):
    """Accelerator regions of one linked kernel ELF, lifted like a bank original; nothing is written."""
    from merlin.analysis.artifact_reader import PinnedELF, open_ordinary
    from merlin.analysis.riscv_machine import region_index
    from merlin.mining.reference_database_analysis import (
        interprocedural_contexts,
        observations_only,
        region_effects,
        resolve_callees,
        transfer_edges,
    )

    from . import reference_analysis as A

    elf = PinnedELF(open_ordinary(Path(path)), Path(path).name, declared_isa=decoder.get("host_isa"))
    try:
        rows = elf.instructions()
        table, _ = A._table(facts)
        if table.get("custom_opcode") is None:
            raise Skip("selected ISA has no derived accelerator opcode")
        indexed = region_index(rows, artifact_sha256=elf.pin["sha256"], custom_opcode=table["custom_opcode"])
        spans, edges = [], []
        for scope, body in indexed:
            start, stop = scope["address_range"]
            spans.append(dict(id=scope["region_id"], address=start, size_bytes=stop - start))
            edges.extend(transfer_edges(body, start, stop, scope["region_id"]))
        contexts = interprocedural_contexts(
            {scope["region_id"]: (scope, body) for scope, body in indexed}, resolve_callees(edges, spans)
        )
        lifted = A._lift(
            elf,
            "elf",
            elf.pin,
            facts=facts,
            offset=0,
            limit=len(indexed),
            candidate=True,
            decoder=decoder,
            entry_contexts=contexts,
        )
        reports = {tuple(row.get("address_range", ())): row for row in lifted["regions"]}
        out = []
        for scope, body in indexed:
            report = reports.get(tuple(scope["address_range"]))
            if not scope["accelerator_sites"] or report is None or report.get("cca") is None:
                continue
            start, stop = scope["address_range"]
            (section,) = [entry for entry in elf.sections if entry[0] == scope["section"]]
            payload = elf.read_at(section[1] + start - section[3], stop - start)
            report = dict(
                report,
                derived_effects=region_effects(
                    report, body, decoder, structure, scope["region_id"], context=contexts.get(scope["region_id"])
                ),
            )
            out.append(
                dict(
                    region_id=scope["region_id"],
                    identity="machine-" + hashlib.sha256(payload).hexdigest(),
                    accelerator_sites=scope["accelerator_sites"],
                    analysis=observations_only(report),
                )
            )
        return out
    finally:
        elf.close()


def _read_json(path):
    try:
        if Path(path).stat().st_size > _MAX_JSON:
            return None
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def grade_kernels(run_dir, workspace):
    """``{case: kernel ELF}`` of the newest completed grade's rows (each row's own grade token)."""
    grade = _read_json(Path(workspace) / "qa" / "latest_completed_grade.json") or {}
    work = Path(run_dir) / "_qa_work"
    out = {}
    for row in grade.get("per_capsule") or ():
        case, token = row.get("capsule"), (row.get("graded") or {}).get("token")
        if not isinstance(case, str) or not isinstance(token, str) or "/" in case or "/" in token:
            continue
        runs = work / ("fruns_" + token) / "runs"
        try:
            benches = sorted(path for path in runs.iterdir() if path.is_dir())
        except OSError as error:
            logging.getLogger(__name__).warning(
                "CCA completed-grade build inventory unavailable: %s", type(error).__name__
            )
            continue
        for bench in benches:
            path = bench / case / "generated" / _KERNEL
            if path.is_file():
                out[case] = path
                break
    return out


def latest_cycles(workspace, *, limit=8 << 20):
    """``{case: cycles}`` from the newest timing row per case in ``qa/perf_history.jsonl``."""
    path = Path(workspace) / "qa" / "perf_history.jsonl"
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - limit))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return {}
    out = {}
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError as error:
            logging.getLogger(__name__).warning("CCA timing history row unreadable: %s", type(error).__name__)
            continue
        if isinstance(row, dict) and isinstance(row.get("test_case"), str) and type(row.get("cycles")) is int:
            out[row["test_case"]] = row["cycles"]
    return out


def case_semantics(workspace, cases):
    """``({case: operation class}, {case: declared extents})`` from each case's declaration."""
    from . import family_rollup

    members = family_rollup.inventory(workspace)
    operations, shapes = {}, {}
    for case in cases:
        try:
            family, shape = family_rollup.describe(case, members.get(case), workspace)
        except Exception:  # noqa: BLE001 -- an undescribable case has no operation class
            family, shape = None, None
        operations[case] = family
        extents = (shape or {}).get("extents") or {}
        shapes[case] = {key: value for key, value in extents.items() if type(value) is int and value > 0} or None
    return operations, shapes


# Preserve the established callable pickle identities through explicit facade exports.
for _export in (
    Skip,
    open_selected_bank,
    _card_symbols,
    bank_units,
    selected_decoder,
    lift_kernel,
    _read_json,
    grade_kernels,
    latest_cycles,
    case_semantics,
):
    _export.__module__ = "merlin_experiments.phase1.feedback.cca_families"
