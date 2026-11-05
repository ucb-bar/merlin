"""CCA families in the grade feedback of a CCA bank arm: your kernels beside the nearest expert unit.

Measured on the CCA coverage arms: authors made about 10,000 tool calls and asked the bank service
twice, both ``inventory``. The service was one paragraph near the end of TASK.md and absent from the
feedback the authors read. This module brings its comparison into ``qa/round_brief.md``.

After a published grade, the emitted kernels of the grade (each case's linked ``package_kernel.elf``)
are lifted with the bank's own pinned decoder, through the same lifter and the same effect derivation
that built the bank, and grouped three ways, all derived by tooling:

* **identical**: the SHA256 of the linked accelerator-region bytes (the bank's own unit identity);
* **family**: :func:`merlin.mining.reference_subset_cards.family_signature` (decoded roles,
  instruction classes, configured-field set, loop-nest shape, control status, non-numeric dataflow
  facts), the same function over bank units and over authors' kernels; counts and shapes excluded;
* **operation class**: the case's declared source semantics (:func:`.family_rollup.describe`), used
  only to shortlist bank units when units carry one; bank units carry none today, which is stated.

Per family: the nearest bank unit by signature agreement, with the fields that agree and differ;
representatives (the most frequent body, and the slowest member by the latest modeled cycles when any
member has them); the applicability comparison of each representative with that unit
(:func:`merlin.kernels.mechanism_applicability.applicability`); a member-by-member comparison with the
representative (:func:`~merlin.kernels.mechanism_applicability.mechanism_differences`) whose deviating
members are listed as exceptions; and, per member, the shape-dependent route admissions.

Facts only: nothing is ranked by assumed speed, structural agreement is never semantic equivalence, and
performance stays unknown. An arm without a round brief gets the same comparison for the cases of one
self-check from :func:`selfcheck_section` (bounded text, full record in a named detail file, cached by
ELF digest). Enabled only for a CCA bank profile. It runs off the grading path in one
background thread with a wall-clock budget; work left at the budget is skipped with that reason. It
never raises into a grade. No target name, regex or opcode literal.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

SCHEMA = "merlin.cca_families.v1"
RECORD = "cca_families.json"
#: Wall-clock budget of one family computation (seconds).
BUDGET_S = 300.0
#: Comparison kinds (``mechanism_differences`` effect names, matched by prefix) shown per family. Edit
#: this list to re-target the section; an empty tuple shows every compared effect.
FACT_KINDS = (
    "operand_load_per_compute",
    "weight_load_per_compute",
    "role_transfer_width",
    "operand_buffering",
    "issue_overlap",
    "prefetch_distance",
    "prefetch_compute_ahead",
    "output_group_extent",
    "operand_transfer_extent",
    "config",
    "loop",
)
#: Families rendered in the brief; the record keeps all of them.
SHOWN_FAMILIES = 8
SHOWN_ROWS = 6
SHOWN_CASES = 4
_KERNEL = "package_kernel.elf"
_MAX_JSON = 96 << 20
_LOCK = threading.Lock()


class Skip(Exception):
    """A stated reason the section is not computed."""


def _kind(effect):
    return not FACT_KINDS or any(str(effect).startswith(kind) for kind in FACT_KINDS)


# --- inputs -------------------------------------------------------------------------------------------


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
    return {row.get("occurrence_id"): row["symbol"] for row in card.get("occurrences") or ()
            if isinstance(row, dict) and isinstance(row.get("symbol"), str) and row["symbol"]}


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
            symbol = (symbols.get(occurrence["original_id"] + ":" + occurrence["region_id"])
                      or next(iter(sorted(symbols.values())), None) or analysis.get("symbol")
                      or (analysis.get("static_function_extent") or {}).get("symbol") or None)
            rows.append(dict(unit_id=unit["kernel_id"], region_id=occurrence["region_id"],
                             original_id=occurrence["original_id"], symbol=symbol, analysis=analysis,
                             signature=family_signature(analysis), operation=None))
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
    from merlin.compile.analysis.artifact_reader import PinnedELF, open_ordinary
    from merlin.compile.analysis.riscv_machine import region_index
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
        contexts = interprocedural_contexts({scope["region_id"]: (scope, body) for scope, body in indexed},
                                            resolve_callees(edges, spans))
        lifted = A._lift(elf, "elf", elf.pin, facts=facts, offset=0, limit=len(indexed), candidate=True,
                         decoder=decoder, entry_contexts=contexts)
        reports = {tuple(row.get("address_range", ())): row for row in lifted["regions"]}
        out = []
        for scope, body in indexed:
            report = reports.get(tuple(scope["address_range"]))
            if not scope["accelerator_sites"] or report is None or report.get("cca") is None:
                continue
            start, stop = scope["address_range"]
            (section,) = [entry for entry in elf.sections if entry[0] == scope["section"]]
            payload = elf.read_at(section[1] + start - section[3], stop - start)
            report = dict(report, derived_effects=region_effects(report, body, decoder, structure, scope["region_id"],
                                                                 context=contexts.get(scope["region_id"])))
            out.append(dict(region_id=scope["region_id"], identity="machine-" + hashlib.sha256(payload).hexdigest(),
                            accelerator_sites=scope["accelerator_sites"], analysis=observations_only(report)))
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
        except OSError:
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
        except ValueError:
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


# --- mechanism summary --------------------------------------------------------------------------------

#: Movement roles named in a summary: operand (A), weight (W) and output.
SUMMARY_ROLES = dict(operand_load="A", weight_load="W", readout="output")


def _per_role(entries, project):
    """``{A|W|output[:operand]: value}``; per-operand entries only where the role entry has no value."""
    out = {}
    for key, entry in sorted(entries.items()):
        role, _, operand = key.partition(":")
        if operand and SUMMARY_ROLES.get(role, role) in out:
            continue
        try:
            value = project(entry)
        except (ArithmeticError, LookupError, TypeError, ValueError):
            value = None
        if value is not None:
            out[SUMMARY_ROLES.get(role, role) + (":" + operand if operand else "")] = value
    return out


def mechanism_summary(analysis):
    """Per-unit mechanism facts, each ``{"status", "value"}`` or ``{"status": "unknown", "reason"}``.

    Roles (A/W/output sites), transfer width per role, operand buffering, load/compute issue overlap,
    prefetch distance, output panel extent and operand residency, read from the region's derived effects.
    A fact the record does not establish stays unknown with the record's own reason."""
    import merlin.kernels.mechanism_applicability as MA

    effects = analysis.get("derived_effects") or {}
    sites = (analysis.get("mechanism_projection") or {}).get("role_sites") or {}
    out = dict(roles=dict(status="observed", value={label: sites.get(role, 0) for role, label in SUMMARY_ROLES.items()})
               if sites else dict(status="unknown", reason="no decoded role sites"))

    def fact(name, project):
        record = MA.stream_fact(effects, name) if name in MA.STREAM_EFFECTS else (effects.get("facts") or {}).get(name)
        if not isinstance(record, dict) or record.get("status") in (None, "unknown"):
            return dict(status="unknown", reason=str((record or {}).get("reason") or "not derived")[:120])
        try:
            value = project(record.get("value"))
        except (ArithmeticError, LookupError, TypeError, ValueError):
            value = None
        if value in (None, {}, []):
            return dict(status="unknown", reason="derived record has no summarizable value")
        return dict(status=record.get("status"), value=value)

    out["transfer_width"] = fact("role_transfer_width", lambda v: _per_role(
        MA._role_entries(v)[0], lambda e: (MA._width(e) or {}).get("elements_per_transfer")))
    out["buffering"] = fact("operand_buffering", lambda v: _per_role(
        MA._role_entries(v)[0], lambda e: (lambda b: None if not b or b.get("disjoint_windows") is None else
                                           dict(windows=b["disjoint_windows"], alternating=b["alternating"]))(
            MA._buffering(e))))
    out["overlap"] = fact("issue_overlap", lambda v: {SUMMARY_ROLES.get(k, k): row["fraction"]
                                                      for k, row in MA._overlap(v)[0].items()})
    out["prefetch_distance"] = fact("prefetch_distance", lambda v: {k: row["median_loads_between"]
                                                                   for k, row in MA._distance(v).items()})
    out["output_panel"] = fact("output_group_extent", lambda v: {k: v[k] for k in (
        "median_readout_elements_per_block", "column_extents", "row_extents") if isinstance(v, dict) and k in v}
        or (v if isinstance(v, (int, float)) else None))
    out["residency"] = fact("operand_retention", lambda v: {k: v[k] for k in sorted(v)
                                                           if isinstance(v.get(k), (int, float, bool))}
                            if isinstance(v, dict) else None)
    return out


def _summary_text(summary):
    parts = []
    for name, row in summary.items():
        if row.get("status") == "unknown":
            parts.append(f"{name} unknown")
        else:
            parts.append(f"{name} {json.dumps(row['value'], sort_keys=True, separators=(',', ':'), default=str)}")
    return "; ".join(parts)


# --- grouping and comparison --------------------------------------------------------------------------


def group(members):
    """``{family key: [members]}``; each member carries its signature and identical-body identity."""
    from merlin.mining.reference_subset_cards import family_key, family_signature

    families = {}
    for member in members:
        member.setdefault("signature", family_signature(member["analysis"]))
        member["family"] = family_key(member["signature"])
        families.setdefault(member["family"], []).append(member)
    return families


#: Nearest bank units whose mechanisms are listed per family.
EXPERT_UNITS = 3


def nearest_units(signature, units, count):
    """Up to ``count`` distinct bank units by signature agreement (``[(score, occurrence)]``)."""
    from merlin.mining.reference_subset_cards import signature_similarity

    best = {}
    for unit in units:
        score = signature_similarity(signature, unit["signature"])["score"]
        if unit["unit_id"] not in best or score > best[unit["unit_id"]][0]:
            best[unit["unit_id"]] = (score, unit)
    return sorted(best.values(), key=lambda item: (-item[0], item[1]["unit_id"]))[:count]


def nearest(signature, units, operation=None):
    """The bank occurrence whose signature agrees most, with the basis; ties are listed, never broken silently."""
    from merlin.mining.reference_subset_cards import signature_similarity

    shortlist = [unit for unit in units if operation is not None and unit.get("operation") == operation]
    basis = ("units of the same operation class" if shortlist else
             "all bank units (bank units carry no operation class)" if all(u.get("operation") is None for u in units)
             else "all bank units (none shares this operation class)")
    scored = [(signature_similarity(signature, unit["signature"]), unit) for unit in (shortlist or units)]
    if not scored:
        return None
    best = max(row["score"] for row, _ in scored)
    tied = [(row, unit) for row, unit in scored if row["score"] == best]
    row, unit = min(tied, key=lambda item: (item[1]["unit_id"], item[1]["region_id"]))
    return dict(unit_id=unit["unit_id"], region_id=unit["region_id"], original_id=unit["original_id"],
                symbol=unit.get("symbol"), agreement=row["score"], agree=row["agree"],
                differ={name: dict(only_yours=value["only_left"], only_unit=value["only_right"])
                        for name, value in row["differ"].items()},
                tied_units=sorted({u["unit_id"] for _, u in tied} - {unit["unit_id"]}), shortlist=basis,
                semantic_equivalence="unknown", performance="unknown")


#: The one summary number shown for a compared value, first key present.
_SUMMARY_KEYS = ("value", "elements_per_transfer", "fraction", "disjoint_windows", "median_loads_between",
                 "column_extent_if_one_row_band", "largest_elements")


def _value(row, side):
    from .reference_applicability import _side

    value = _side(row, side)
    if isinstance(value, dict):
        key = next((key for key in _SUMMARY_KEYS if isinstance(value.get(key), (int, float, bool))), None)
        if key is not None:
            return value[key] if key == "value" else {key: value[key]}
    text = json.dumps(value, sort_keys=True, default=str)
    return value if len(text) <= 80 else text[:77] + "..."


def _directions(rows):
    """``{effect label: direction}`` of the shown comparison kinds."""
    return {" ".join(str(row[key]) for key in ("effect", "role", "operand") if row.get(key)): row.get("direction")
            for row in rows if _kind(row["effect"])}


def _rows(rows):
    """The shown differing rows, order facts first: rows whose two sides are comparable numbers (they
    carry ``comparison``) lead, the rest keep their fixed effect order."""
    shown = [row for row in rows if _kind(row["effect"]) and row["difference"] == "differs"]
    return [dict(effect=row["effect"], difference=row["difference"],
                 **{key: row[key] for key in ("role", "operand", "direction", "comparison") if row.get(key)},
                 unit=_value(row, "reference"), yours=_value(row, "candidate"),
                 routes=[route["capability"] for route in row.get("routes") or ()
                         if route.get("family_match") is not False])
            for row in sorted(shown, key=lambda row: not row.get("comparison"))]


def _unknown_groups(rows):
    from .reference_applicability import summary

    return [row for row in summary([row for row in rows if _kind(row["effect"])])["unknown_by_reason"]]


def _admissions(result):
    return {route["capability"] + ":" + row["effect"]: route["admission"].get("status")
            for row in result["differences"] for route in row.get("routes") or ()}


def build(members, units, *, cycles=None, operations=None, shapes=None, geometry=None, geometry_reason=None,
          structure=None, deadline=None, clock=time.monotonic):
    """The family record. ``deadline`` is a ``clock()`` value; work past it is skipped with that reason."""
    from merlin.kernels.mechanism_applicability import applicability, mechanism_differences

    cycles, operations, shapes = cycles or {}, operations or {}, shapes or {}
    families = group(members)
    out, skipped = [], []

    def late():
        return deadline is not None and clock() > deadline

    for key, rows in sorted(families.items(), key=lambda item: (-len(item[1]), item[0])):
        identical = Counter(member["identity"] for member in rows)
        cases = sorted({member["case"] for member in rows})
        frequent_identity = min(identical, key=lambda identity: (-identical[identity], identity))
        frequent = min((m for m in rows if m["identity"] == frequent_identity), key=lambda m: m["case"])
        timed = [m for m in rows if m["case"] in cycles]
        slowest = max(timed, key=lambda m: (cycles[m["case"]], m["case"])) if timed else None
        family = dict(key=key, members=len(rows), cases=cases, identical_bodies=len(identical),
                      operation_classes=sorted({str(operations.get(c)) for c in cases}),
                      signature=rows[0]["signature"],
                      representative_basis=("most frequent body and slowest member by latest modeled cycles" if timed
                                            else "most frequent body only (no member has timing in the latest grade)"))
        if late():
            family["status"] = "skipped: budget exhausted"
            skipped.append(key)
            out.append(family)
            continue
        operation = operations.get(frequent["case"])
        match = nearest(frequent["signature"], units, operation)
        family["nearest_unit"] = match
        # The expert's choices across its units nearest this family, next to the members' own shapes: a
        # cross-shape view that needs no equal-shape comparison.
        family["expert_units"] = [dict(unit_id=row["unit_id"], region_id=row["region_id"], agreement=score,
                                       mechanisms=mechanism_summary(row["analysis"]))
                                  for score, row in nearest_units(frequent["signature"], units, EXPERT_UNITS)]
        family["member_mechanisms"] = [dict(case=m["case"], region_id=m["region_id"], shape=shapes.get(m["case"]),
                                            cycles=cycles.get(m["case"]), mechanisms=mechanism_summary(m["analysis"]))
                                       for m in sorted(rows, key=lambda m: (m["case"], m["region_id"]))]
        unit = next((u for u in units if match and (u["unit_id"], u["region_id"]) ==
                     (match["unit_id"], match["region_id"])), None)
        representatives = [("most_frequent", frequent)]
        if slowest is not None and slowest is not frequent:
            representatives.append(("slowest", slowest))
        family["representatives"] = []
        base, base_directions = None, None
        for role, member in representatives:
            entry = dict(role=role, case=member["case"], region_id=member["region_id"],
                         cycles=cycles.get(member["case"]))
            if unit is None or late():
                entry["status"] = "skipped: budget exhausted" if unit is not None else "no bank unit"
            else:
                result = applicability(unit["analysis"], member["analysis"], shape=shapes.get(member["case"]),
                                       geometry=geometry, geometry_reason=geometry_reason, structure=structure)
                entry.update(status="compared", differences=_rows(result["differences"]),
                             unknown_by_reason=_unknown_groups(result["differences"]),
                             same=sum(row["difference"] == "same" for row in result["differences"]),
                             unknown=sum(row["difference"] == "unknown" for row in result["differences"]),
                             missing_capabilities=len(result["missing_capabilities"]))
                if role == "most_frequent":
                    base = _admissions(result)
                    base_directions = _directions(result["differences"])
            family["representatives"].append(entry)
        exceptions, shape_rows, pending = [], [], []
        for member in sorted(rows, key=lambda m: (m["case"], m["region_id"])):
            if member is frequent:
                continue
            if late():
                pending.append(member["case"])
                continue
            if member["identity"] != frequent["identity"]:
                if base_directions is not None:
                    # Relative to the same unit: where this member stands differently from the representative
                    # (e.g. it already matches the unit's buffering while the representative does not).
                    mine = _directions(mechanism_differences(unit["analysis"], member["analysis"],
                                                             structure=structure))
                    deviating = sorted(label for label in set(mine) | set(base_directions)
                                       if mine.get(label) != base_directions.get(label))
                else:
                    deviating = sorted({row["effect"] for row in mechanism_differences(
                        frequent["analysis"], member["analysis"], structure=structure)
                        if row["difference"] == "differs" and _kind(row["effect"])})
                if deviating:
                    exceptions.append(dict(case=member["case"], region_id=member["region_id"], differs_in=deviating))
            if unit is not None and base is not None and shapes.get(member["case"]) != shapes.get(frequent["case"]):
                admitted = _admissions(applicability(unit["analysis"], member["analysis"],
                                                     shape=shapes.get(member["case"]), geometry=geometry,
                                                     geometry_reason=geometry_reason, structure=structure))
                changed = sorted(name for name in set(base) | set(admitted) if base.get(name) != admitted.get(name))
                if changed:
                    shape_rows.append(dict(case=member["case"], shape=shapes.get(member["case"]),
                                           admissions={name: admitted.get(name) for name in changed}))
        family.update(status="compared", exceptions=exceptions, shape_dependent=shape_rows,
                      exception_basis=("direction relative to the nearest unit differs from the most frequent "
                                       "representative's" if base_directions is not None else
                                       "static comparison with the most frequent representative's body"),
                      **({"skipped_members": dict(cases=pending, reason="budget exhausted")} if pending else {}))
        out.append(family)
    return dict(schema=SCHEMA, families=out, kernels=len(members), skipped_families=skipped,
                fact_kinds=list(FACT_KINDS), ordering="families by member count; nothing is ranked by speed",
                semantic_equivalence="unknown", performance="unknown")


# --- rendering and wiring -----------------------------------------------------------------------------


def _short(identity):
    return identity if len(identity) <= 28 else identity[:28] + "..."


def _diff_line(row):
    label = row["effect"] + "".join(f" {row[key]}" for key in ("role", "operand") if row.get(key))
    routes = f"; library routes: {', '.join(row['routes'][:3])}" if row.get("routes") else ""
    return (f"    - {label}: unit {json.dumps(row['unit'], default=str)} vs yours "
            f"{json.dumps(row['yours'], default=str)} ({row['difference']}){routes}")


def render(record):
    """Compact Markdown section for ``qa/round_brief.md``."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return []
    head = ["## CCA families: your kernels beside the nearest expert unit", "",
            f"The CCA bank service is enabled for this arm (bank `{str(record.get('bank_sha256'))[:12]}`, "
            f"{record.get('bank_units')} units).", ""]
    if record.get("status") == "skipped":
        return head + [f"Not computed for grade {record.get('grade_token')}: {record.get('reason')}.", ""]
    families = record.get("families") or []
    lines = head + [
        f"Grade {record.get('grade_token')}: {record.get('kernels')} emitted accelerator kernels in "
        f"{len(families)} families (same decoded roles, configured fields, loop-nest shape and dataflow; shapes "
        f"and counts ignored). Each family is compared with the bank unit whose structure agrees most: your "
        "loads, buffering, overlap and prefetch beside the expert's. Full record: `qa/" + RECORD + "`.", ""]
    for index, family in enumerate(families[:SHOWN_FAMILIES], 1):
        cases = family["cases"]
        named = ", ".join(f"`{case}`" for case in cases[:SHOWN_CASES]) + (
            f" +{len(cases) - SHOWN_CASES}" if len(cases) > SHOWN_CASES else "")
        lines.append(f"- **F{index}** `{family['key']}`: {family['members']} kernels, {len(cases)} cases "
                     f"({named}), {family['identical_bodies']} distinct bodies; operation classes "
                     f"{', '.join(family['operation_classes'])}.")
        if family.get("status") != "compared":
            lines.append(f"  {family.get('status')}.")
            continue
        match = family.get("nearest_unit")
        if match:
            differ = "; ".join(f"{name} (yours only: {len(v['only_yours'])}, unit only: {len(v['only_unit'])})"
                               for name, v in match["differ"].items()) or "none"
            lines.append(f"  Nearest unit `{_short(match['unit_id'])}` region `{match['region_id']}` "
                         f"(structural agreement {match['agreement']}; agree: {', '.join(match['agree']) or 'none'}; "
                         f"differ: {differ}" + (f"; tied: {len(match['tied_units'])} more" if match["tied_units"]
                                                else "") + ").")
        for rep in family.get("representatives") or ():
            cycles = f", {rep['cycles']} modeled cycles" if rep.get("cycles") is not None else ""
            if rep.get("status") != "compared":
                lines.append(f"  {rep['role']} `{rep['case']}`{cycles}: {rep.get('status')}.")
                continue
            lines.append(f"  {rep['role']} `{rep['case']}`{cycles}: {len(rep['differences'])} differences, "
                         f"{rep['same']} same, {rep['unknown']} unknown.")
            lines += [_diff_line(row) for row in rep["differences"][:SHOWN_ROWS]]
            lines += [f"    - unknown ({group['reason']}): {', '.join(group['effects'][:6])}"
                      + (f" +{len(group['effects']) - 6}" if len(group["effects"]) > 6 else "")
                      for group in (rep.get("unknown_by_reason") or ())[:3]]
        for unit in (family.get("expert_units") or ())[:EXPERT_UNITS]:
            lines.append(f"  expert `{_short(unit['unit_id'])}` (agreement {unit['agreement']}): "
                         + _summary_text(unit["mechanisms"]))
        for row in (family.get("member_mechanisms") or ())[:SHOWN_CASES]:
            shape = ",".join(f"{k}={v}" for k, v in (row.get("shape") or {}).items()) or "shape undeclared"
            lines.append(f"  yours `{row['case']}` ({shape}): " + _summary_text(row["mechanisms"]))
        if family.get("exceptions"):
            lines.append("  Exceptions (stand differently from the unit than the most frequent representative): "
                + "; ".join(f"`{row['case']}` in {', '.join(sorted({label.split(' ')[0] for label in row['differs_in']}))}"
                            for row in family["exceptions"][:SHOWN_CASES])
                + (f" +{len(family['exceptions']) - SHOWN_CASES}" if len(family["exceptions"]) > SHOWN_CASES else "")
                + ".")
        if family.get("shape_dependent"):
            lines.append(f"  Shape-dependent route admissions differ for {len(family['shape_dependent'])} members "
                         "(see the record).")
        if family.get("skipped_members"):
            lines.append(f"  {len(family['skipped_members']['cases'])} members not compared: budget exhausted.")
        if match:
            lines.append(f"  Ask: `python agent_selfcheck.py --reference-analysis applicability --reference-entry "
                         f"{match['unit_id']} --reference-region {match['region_id']} --candidate-path "
                         f"{(Path('artifacts') / 'FILE').as_posix()} --candidate-shape m=M,k=K,n=N`.")
    if len(families) > SHOWN_FAMILIES:
        lines.append(f"- {len(families) - SHOWN_FAMILIES} more families in the record.")
    return lines + ["", "Structural agreement is not semantic equivalence; performance effects are unknown until "
                    "measured.", ""]


def brief_lines(workspace):
    """The cached section; ``[]`` outside a bank arm or before the first record."""
    if _bank_pin() is None:
        return []
    return render(_read_json(Path(workspace) / "qa" / RECORD))


def _bank_pin():
    try:
        from .reference_bank import selected_pin

        return selected_pin()
    except Exception:  # noqa: BLE001 -- no profile, no section
        return None


def _jsonable(value):
    """String keys throughout: derived records may key a mapping by operand index or by ``None``."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(workspace, record):
    path = Path(workspace) / "qa" / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{RECORD}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, default=str))
    os.replace(temporary, path)


def compute(run_dir, workspace, *, pin, target, budget_s=BUDGET_S, clock=time.monotonic):
    """One bounded record for the newest completed grade; a stated skip instead of an exception."""
    started = clock()
    deadline = started + budget_s
    grade = _read_json(Path(workspace) / "qa" / "latest_completed_grade.json") or {}
    token = (grade.get("performance_headline") or {}).get("grade", {}).get("token")
    base = dict(schema=SCHEMA, grade_token=token, budget_s=budget_s)
    try:
        if not target:
            raise Skip("no selected target registered in this process")
        root, bank, decoder_files = open_selected_bank(workspace, pin)
        decoder, structure = selected_decoder(decoder_files, target)
        units = bank_units(root, bank)
        kernels = grade_kernels(run_dir, workspace)
        if not kernels:
            raise Skip("the newest completed grade retained no emitted kernel ELFs")
        members, failed = [], {}
        for case, path in sorted(kernels.items()):
            if clock() > deadline:
                failed[case] = "budget exhausted before lifting"
                continue
            try:
                for region in lift_kernel(path, decoder=decoder, facts=decoder_files["facts"], structure=structure):
                    members.append(dict(region, case=case))
            except Exception as error:  # noqa: BLE001 -- one unliftable kernel is reported, not fatal
                failed[case] = f"{type(error).__name__}: {str(error)[:120]}"
        operations, shapes = case_semantics(workspace, sorted({m["case"] for m in members}))
        from merlin.kernels.mechanism_applicability import selected_geometry

        geometry, reason = selected_geometry(target, json.loads(Path(decoder_files["facts"]).read_text()))
        record = build(members, units, cycles=latest_cycles(workspace), operations=operations, shapes=shapes,
                       geometry=geometry, geometry_reason=reason, structure=structure, deadline=deadline, clock=clock)
        record.update(base, bank_sha256=pin, bank_units=len({u["unit_id"] for u in units}), not_lifted=failed,
                      seconds=round(clock() - started, 1))
        return _jsonable(record)
    except Skip as reason:
        return dict(base, status="skipped", reason=str(reason))
    except Exception as error:  # noqa: BLE001 -- feedback never fails a grade
        return dict(base, status="skipped", reason=f"{type(error).__name__}: {str(error)[:200]}")


def refresh_async(run_dir, workspace, *, target=None, on_done=None):
    """Start one background computation after a grade; returns the thread, or None when not applicable.

    At most one runs at a time; a grade published meanwhile is picked up by the next refresh."""
    pin = _bank_pin()
    if pin is None or not _LOCK.acquire(blocking=False):
        return None
    if target is None:
        try:
            from . import case_declarations

            target = case_declarations.target()
        except Exception:  # noqa: BLE001 -- compute() states the missing target
            target = None

    def run():
        try:
            _write(workspace, compute(run_dir, workspace, pin=pin, target=target))
            if on_done is not None:
                on_done()
        except Exception as error:  # noqa: BLE001
            print(f"[grade] cca family feedback skipped: {type(error).__name__}: {error}", flush=True)
        finally:
            _LOCK.release()

    thread = threading.Thread(target=run, name="cca-families", daemon=True)
    thread.start()
    return thread


# --- self-check reply section ---------------------------------------------------------------------------
#
# An arm without a round brief (the small study) shows the same comparison right after each self-check of
# the author's dev cases. The kernels are the self-check's own linked ELFs; everything below reuses the
# family pipeline above. Results are cached by ELF digest, so re-checking unchanged kernels costs one hash
# per case. The wiring into the reply belongs to the broker; this is the library half.

SELFCHECK_SCHEMA = "merlin.cca_families.selfcheck.v1"
#: Wall-clock budget of one self-check section (seconds); the reply waits for it.
SELFCHECK_BUDGET_S = 60.0
#: Upper bound of the section text (UTF-8 bytes); the full record is the detail file the text names.
SELFCHECK_TEXT_BYTES = 2048
SELFCHECK_DIFFERENCES = 4
SELFCHECK_UNKNOWN_GROUPS = 3
SELFCHECK_MEMBERS = 3
#: Mechanism facts shown per case, in order, with their labels (keys of :func:`mechanism_summary`).
SELFCHECK_MECHANISMS = (("roles", "roles"), ("transfer_width", "width"), ("buffering", "buffering"),
                        ("overlap", "overlap"), ("prefetch_distance", "prefetch distance"),
                        ("output_panel", "output group"), ("residency", "residency"))
_MAX_ELF = 64 << 20
_VALUE_CHARS = 56
_DECODERS: dict = {}
_UNITS: dict = {}


def arm_enabled(arm):
    """``(enabled, bank pin or None)`` of an explicit arm flag (bool), arm mapping or author profile."""
    if isinstance(arm, bool) or arm is None:
        return bool(arm), None
    if isinstance(arm, dict):
        pin = arm.get("bank_sha256") or arm.get("cca_bank_sha256")
        return bool(arm.get("cca", pin is not None)), pin
    pin = getattr(arm, "bank_sha256", None)
    return pin is not None, pin


def selfcheck_kernels(selfcheck_dir, cases=None):
    """``({case: linked kernel ELF}, {case: reason})`` of one self-check's raw runs.

    The self-check grades each case in ``<runs root>/runs/<suite>/<case>/`` (``capsule_result.json``
    beside ``generated/package_kernel.elf``, as :mod:`.selfcheck` reads them); the runs root, its
    ``runs/`` directory or the suite directory is accepted. ``cases`` limits the answer to those cases."""
    root, results = Path(selfcheck_dir), []
    for directory, pattern in ((root, "*/capsule_result.json"), (root, "*/*/capsule_result.json"),
                               (root / "runs", "*/*/capsule_result.json")):
        results = sorted(directory.glob(pattern))
        if results:
            break
    found = {}
    for result in results:
        found.setdefault(result.parent.name, result.parent / "generated" / _KERNEL)
    kernels, missing = {}, {}
    for case in sorted(found if cases is None else set(cases)):
        path = found.get(case)
        if path is None:
            missing[case] = "no result for this case in the self-check output"
        elif not path.is_file():
            missing[case] = "the self-check retained no linked kernel ELF for this case"
        else:
            kernels[case] = path
    return kernels, missing


def _digest(path):
    if Path(path).stat().st_size > _MAX_ELF:
        raise Skip(f"{Path(path).name} exceeds {_MAX_ELF} bytes")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(*parts):
    return hashlib.sha256(json.dumps(_jsonable(parts), sort_keys=True, default=str).encode()).hexdigest()


def _write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, default=str))
    os.replace(temporary, path)


def _facts_document(facts):
    if facts is None or isinstance(facts, dict):
        return facts
    return json.loads(Path(facts).read_text())


def _decoder_for(decoder_files, target):
    key = (str(decoder_files["contract"]), str(decoder_files["facts"]), target)
    if key not in _DECODERS:
        _DECODERS.clear()
        _DECODERS[key] = selected_decoder(decoder_files, target)
    return _DECODERS[key]


def _units_for(root, bank, pin):
    key = (str(root), pin)
    if key not in _UNITS:
        _UNITS.clear()
        _UNITS[key] = bank_units(root, bank)
    return _UNITS[key]


def _lifted(path, digest, *, pin, target, decoder, decoder_files, structure, cache_dir):
    """Lifted regions of one ELF, cached by its digest; the JSON form both ways, so cold equals cached."""
    cache = Path(cache_dir) / f"lift-{_key(SCHEMA, pin, target, digest)[:32]}.json"
    cached = _read_json(cache)
    if isinstance(cached, dict) and cached.get("elf_sha256") == digest and isinstance(cached.get("regions"), list):
        return cached["regions"], True
    regions = json.loads(json.dumps(_jsonable(lift_kernel(path, decoder=decoder, facts=decoder_files["facts"],
                                                          structure=structure)), default=str))
    try:
        _write_json(cache, dict(elf_sha256=digest, regions=regions))
    except OSError:
        pass  # an unwritable cache costs a re-lift, never the section
    return regions, False


def _family_view(family, units):
    """One built family plus the nearest unit's own mechanisms and the representative compared with it."""
    match = family.get("nearest_unit")
    unit = next((u for u in units if match and (u["unit_id"], u["region_id"]) ==
                 (match["unit_id"], match["region_id"])), None)
    reps = [rep for rep in family.get("representatives") or () if rep.get("status") == "compared"]
    return dict(family, nearest_mechanisms=mechanism_summary(unit["analysis"]) if unit else None,
                compared=reps[0] if reps else None)


def _selfcheck_record(workspace, selfcheck_dir, *, bank_root, facts, target, pin, cases, budget_s, cache_dir,
                      started, deadline, clock):
    if target is None:
        try:
            from . import case_declarations

            target = case_declarations.target()
        except Exception:  # noqa: BLE001 -- the skip below states it
            target = None
    if not target:
        raise Skip("no selected target registered in this process")
    kernels, missing = selfcheck_kernels(selfcheck_dir, cases)
    if not kernels:
        raise Skip("no linked kernel ELF in this self-check's output"
                   + "".join(f"; `{case}`: {why}" for case, why in sorted(missing.items())[:3]))
    digests = {case: _digest(path) for case, path in kernels.items()}
    from .reference_bank import GRANT

    root = Path(bank_root) if bank_root is not None else Path(workspace) / GRANT
    pin_basis = "supplied by the arm"
    if pin is None:
        try:
            pin = hashlib.sha256((Path(os.path.realpath(root)) / "bank.json").read_bytes()).hexdigest()
        except OSError as error:
            raise Skip(f"no bank.json under {root}") from error
        pin_basis = "digest of bank.json (the arm supplied no pin)"
    facts_doc = _facts_document(facts)
    operations, shapes = case_semantics(workspace, sorted(kernels))
    key = _key(SELFCHECK_SCHEMA, SCHEMA, pin, target, digests, operations, shapes, FACT_KINDS, facts_doc)
    cached = _read_json(Path(cache_dir) / f"record-{key[:32]}.json")
    if isinstance(cached, dict) and cached.get("key") == key and cached.get("status") == "compared":
        return dict(cached, cache="hit", not_lifted=dict(cached.get("not_lifted") or {}, **missing))
    if clock() > deadline:
        raise Skip(f"budget of {budget_s}s exhausted before the bank was opened")
    root, bank, decoder_files = open_selected_bank(workspace, pin, root=bank_root)
    decoder, structure = _decoder_for(decoder_files, target)
    units = _units_for(root, bank, pin)
    members, failed, lifts = [], {}, {}
    for case, path in sorted(kernels.items()):
        if clock() > deadline:
            failed[case] = "budget exhausted before lifting"
            continue
        try:
            regions, hit = _lifted(path, digests[case], pin=pin, target=target, decoder=decoder,
                                   decoder_files=decoder_files, structure=structure, cache_dir=cache_dir)
        except Exception as error:  # noqa: BLE001 -- one unliftable kernel is reported, not fatal
            failed[case] = f"{type(error).__name__}: {str(error)[:120]}"
            continue
        lifts[case] = "cached" if hit else "lifted"
        if not regions:
            failed[case] = "no accelerator region in the linked kernel"
        members.extend(dict(region, case=case) for region in regions)
    from merlin.kernels.mechanism_applicability import selected_geometry

    geometry, reason = selected_geometry(target, facts_doc if facts_doc is not None
                                         else json.loads(Path(decoder_files["facts"]).read_text()))
    full = build(members, units, operations=operations, shapes=shapes, geometry=geometry, geometry_reason=reason,
                 structure=structure, deadline=deadline, clock=clock)
    labels = {family["key"]: f"F{index}" for index, family in enumerate(full["families"], 1)}
    per_case = {case: dict(elf=str(path), elf_sha256=digests[case], lift=lifts.get(case),
                           kernels=[dict(region_id=m["region_id"], identity=m["identity"],
                                         family=labels.get(m["family"]))
                                    for m in sorted(members, key=lambda m: m["region_id"]) if m["case"] == case])
                for case, path in sorted(kernels.items())}
    budget_cut = [case for case, why in failed.items() if why.startswith("budget")]
    complete = (not budget_cut and not full["skipped_families"]
                and not any(family.get("skipped_members") for family in full["families"]))
    record = dict(schema=SELFCHECK_SCHEMA, key=key, status="compared" if complete else "partial",
                  target=target, budget_s=budget_s, cases=per_case, not_lifted=dict(failed, **missing),
                  bank=dict(sha256=pin, pin_basis=pin_basis, root=str(root),
                            units=len({u["unit_id"] for u in units})),
                  families=[dict(_family_view(family, units), label=labels[family["key"]])
                            for family in full["families"]],
                  fact_kinds=full["fact_kinds"], ordering=full["ordering"],
                  semantic_equivalence="unknown", performance="unknown", cache="miss")
    record = _jsonable(record)
    if complete:
        try:
            _write_json(Path(cache_dir) / f"record-{key[:32]}.json", record)
        except OSError:
            pass  # an unwritable cache costs a recomputation, never the section
    return record


def _flat(value):
    """``k=v`` pairs, nested mappings in parentheses; floats to three decimals."""
    if isinstance(value, float):
        return str(round(value, 3))
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, dict):
        return " ".join(f"{key}={_flat(item)}" if not isinstance(item, dict) else f"{key}({_flat(item)})"
                        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_flat(item) for item in value) + "]"
    return str(value)


def _compact(value, chars=_VALUE_CHARS):
    text = _flat(value)
    return text if len(text) <= chars else text[:chars - 3] + "..."


def _mechanism(summary, name):
    row = (summary or {}).get(name) or {}
    if row.get("status") in (None, "unknown"):
        return "unknown"
    value = row.get("value")
    if name == "buffering" and isinstance(value, dict):
        # windows per role, "alt" when the windows alternate
        value = {role: f"{(entry or {}).get('windows')}w" + (" alt" if (entry or {}).get("alternating") else "")
                 for role, entry in value.items()}
    elif name == "output_panel" and isinstance(value, dict):
        value = {key.split("_")[0] if key.endswith("_extents") else "per_block": (
            sorted(item, key=str) if isinstance(item, dict) else item) for key, item in value.items()}
    return _compact(value)


def _unknown_groups_of(family):
    """Unknowns of one family grouped by reason: comparison effects and unestablished mechanism facts."""
    groups = {}
    for group in ((family.get("compared") or {}).get("unknown_by_reason") or ()):
        groups.setdefault(str(group.get("reason")), []).extend(group.get("effects") or ())
    sides = [("unit", family.get("nearest_mechanisms"))] + [
        (f"`{row['case']}`", row.get("mechanisms")) for row in family.get("member_mechanisms") or ()]
    for side, summary in sides:
        for name, label in SELFCHECK_MECHANISMS:
            row = (summary or {}).get(name) or {}
            if row.get("status") == "unknown":
                groups.setdefault(str(row.get("reason")), []).append(f"{side} {label}")
    return sorted(([reason, effects] for reason, effects in groups.items()), key=lambda g: (-len(g[1]), g[0]))


def _family_lines(family):
    cases = ", ".join(f"`{case}`" for case in family["cases"])
    lines = [f"{family['label']} `{family['key']}`: {family['members']} kernel(s) in {cases}."]
    if family.get("status") != "compared":
        return lines + [f"  {family.get('status')}."]
    match = family.get("nearest_unit")
    if not match:
        return lines + ["  No bank unit to compare with."]
    symbol = f"`{match['symbol']}`" if match.get("symbol") else "(symbol unknown: its card names none)"
    differ = ", ".join(f"{name} (yours only {len(v['only_yours'])}, unit only {len(v['only_unit'])})"
                       for name, v in match["differ"].items()) or "none"
    tied = f", {len(match['tied_units'])} tied" if match.get("tied_units") else ""
    lines.append(f"  Nearest unit {symbol} `{_short(match['unit_id'])}` {match['region_id']} (agreement "
                 f"{match['agreement']}{tied}); agree: {', '.join(match['agree']) or 'none'}; differ: {differ}.")
    members = (family.get("member_mechanisms") or ())[:SELFCHECK_MEMBERS]
    names = [f"`{row['case']}`" + (f" {row['region_id'][:15]}" if len(members) > 1 else "") for row in members]
    lines.append("  Mechanisms, unit | " + " | ".join(names) + ":")
    lines += [f"    {label}: {_mechanism(family.get('nearest_mechanisms'), name)} | "
              + " | ".join(_mechanism(row.get("mechanisms"), name) for row in members)
              for name, label in SELFCHECK_MECHANISMS]
    compared = family.get("compared")
    if compared:
        rows = compared.get("differences") or ()
        lines.append(f"  Differences (unit vs yours, order facts first): {len(rows)} differ, "
                     f"{compared.get('same')} same, {compared.get('unknown')} unknown.")
        for row in rows[:SELFCHECK_DIFFERENCES]:
            label = row["effect"] + "".join(f" {row[key]}" for key in ("role", "operand") if row.get(key))
            lines.append(f"    - {label}: {_compact(row['unit'])} vs {_compact(row['yours'])}")
    for reason, effects in _unknown_groups_of(family)[:SELFCHECK_UNKNOWN_GROUPS]:
        shown = ", ".join(effects[:4]) + (f" +{len(effects) - 4}" if len(effects) > 4 else "")
        lines.append(f"  Unknown ({reason[:120]}): {shown}.")
    if family.get("exceptions"):
        lines.append("  Exceptions: " + "; ".join(
            f"`{row['case']}` in {', '.join(sorted({label.split(' ')[0] for label in row['differs_in']}))}"
            for row in family["exceptions"][:SELFCHECK_MEMBERS]) + ".")
    return lines


def selfcheck_text(record, detail, *, limit=SELFCHECK_TEXT_BYTES):
    """The compact section for one self-check reply, at most ``limit`` UTF-8 bytes."""
    head = "CCA families (this arm's CCA bank service)"
    if record.get("status") == "skipped":
        return f"{head}: not computed for this self-check: {record.get('reason')}."[:limit]
    bank = record.get("bank") or {}
    families, cases = record.get("families") or [], record.get("cases") or {}
    kernels = sum(len(row["kernels"]) for row in cases.values())
    lines = [f"{head}, bank `{str(bank.get('sha256'))[:12]}` ({bank.get('units')} units): {len(cases)} case(s), "
             f"{kernels} kernel(s), {len(families)} famil{'y' if len(families) == 1 else 'ies'}"
             + (", cached (kernels unchanged)" if record.get("cache") == "hit" else "")
             + f". Full record: `{detail}`."]
    if record.get("status") == "partial":
        lines.append(f"Partial: the {record.get('budget_s')}s budget was exhausted; the rest is skipped.")
    for case, row in sorted(cases.items()):
        lines.append(f"- `{case}`: " + (", ".join(f"{k['region_id'][:15]} -> {k['family']}" for k in row["kernels"])
                                         or "no accelerator region"))
    for case, why in sorted((record.get("not_lifted") or {}).items()):
        if case not in cases or not cases[case]["kernels"]:
            lines.append(f"- `{case}`: not compared ({why[:120]}).")
    for family in families:
        lines += _family_lines(family)
    tail = "Structural agreement is not semantic equivalence; performance is unknown until measured."
    out, used = [], len(tail.encode()) + 1
    for index, line in enumerate(lines):
        size = len(line.encode()) + 1
        if index and used + size > limit - 64:
            out.append(f"... {len(lines) - index} more lines in the full record.")
            break
        out.append(line)
        used += size
    return "\n".join(out + [tail]).encode()[:limit].decode(errors="ignore")


def _summary_record(record, detail):
    """The JSON half of the section: what the text shows, small; the detail file holds the rest."""
    if record.get("status") == "skipped":
        return dict(schema=SELFCHECK_SCHEMA, status="skipped", reason=record.get("reason"),
                    budget_s=record.get("budget_s"), seconds=record.get("seconds"))
    families = []
    for family in record.get("families") or ():
        match = family.get("nearest_unit") or {}
        compared = family.get("compared") or {}
        families.append(dict(
            label=family["label"], key=family["key"], cases=family["cases"], members=family["members"],
            status=family.get("status"),
            nearest_unit=dict(unit_id=match.get("unit_id"), symbol=match.get("symbol"),
                              region_id=match.get("region_id"), agreement=match.get("agreement"),
                              agree=match.get("agree"), differ=sorted(match.get("differ") or ()))
            if match else None,
            top_differences=[dict(effect=row["effect"], **{k: row[k] for k in ("role", "operand", "comparison")
                                                           if row.get(k)}, unit=row["unit"], yours=row["yours"])
                             for row in (compared.get("differences") or ())[:SELFCHECK_DIFFERENCES]],
            unknown_by_reason=[dict(reason=reason, count=len(effects))
                               for reason, effects in _unknown_groups_of(family)]))
    bank = record.get("bank") or {}
    return dict(schema=SELFCHECK_SCHEMA, status=record["status"], cache=record.get("cache"),
                seconds=record.get("seconds"), budget_s=record.get("budget_s"), detail=detail,
                bank=dict(sha256=bank.get("sha256"), units=bank.get("units")),
                cases={case: dict(elf_sha256=row["elf_sha256"],
                                  kernels=[dict(region_id=k["region_id"], family=k["family"]) for k in row["kernels"]])
                       for case, row in (record.get("cases") or {}).items()},
                not_lifted=record.get("not_lifted") or {}, families=families,
                semantic_equivalence="unknown", performance="unknown")


def selfcheck_section(workspace, selfcheck_dir, *, arm, bank_root=None, facts=None, target=None, pin=None,
                      cases=None, budget_s=SELFCHECK_BUDGET_S, detail_dir=None, cache_dir=None,
                      limit=SELFCHECK_TEXT_BYTES, clock=time.monotonic):
    """CCA family feedback for the cases one self-check built; ``None`` when the arm has no CCA bank.

    ``arm`` is the explicit enablement: ``True``/``False``, a mapping ``{"cca": bool, "bank_sha256": pin}``
    or an author profile (enabled when it pins a bank). ``selfcheck_dir`` is the self-check's raw runs root
    (see :func:`selfcheck_kernels`), read before the check releases its scratch; ``cases`` the requested
    dev cases. ``bank_root`` defaults to the workspace's ``cca-evidence/``; ``pin`` to the arm's pin, else
    the digest of its ``bank.json`` (stated in the record). ``facts`` (path or document) selects the RTL
    geometry for route admission, default the bank's pinned decoder facts; ``target`` defaults to the
    registered selected target.

    Returns ``{"text", "record", "detail_path"}``: ``text`` at most ``limit`` bytes, ``record`` its compact
    JSON, ``detail_path`` the full record (``selfcheck_out/details/`` by default; ``None`` when skipped).
    Bounded by ``budget_s``: work past it is skipped with that reason. Records and lifts are cached by ELF
    digest in ``cache_dir`` (default ``qa/.cca_selfcheck_cache/``). Never raises."""
    enabled, arm_pin = arm_enabled(arm)
    if not enabled:
        return None
    started = clock()
    workspace = Path(workspace)
    detail_dir = Path(detail_dir) if detail_dir is not None else workspace / "selfcheck_out" / "details"
    cache_dir = Path(cache_dir) if cache_dir is not None else workspace / "qa" / ".cca_selfcheck_cache"
    try:
        record = _selfcheck_record(workspace, selfcheck_dir, bank_root=bank_root, facts=facts, target=target,
                                   pin=pin or arm_pin, cases=cases, budget_s=budget_s, cache_dir=cache_dir,
                                   started=started, deadline=started + budget_s, clock=clock)
    except Skip as reason:
        record = dict(schema=SELFCHECK_SCHEMA, status="skipped", reason=str(reason), budget_s=budget_s)
    except Exception as error:  # noqa: BLE001 -- feedback never fails a self-check
        record = dict(schema=SELFCHECK_SCHEMA, status="skipped", budget_s=budget_s,
                      reason=f"{type(error).__name__}: {str(error)[:200]}")
    record["seconds"] = round(clock() - started, 3)
    detail_path, detail = None, None
    if record.get("status") != "skipped":
        detail_path = detail_dir / f"cca_families_{record['key'][:16]}.json"
        try:
            _write_json(detail_path, record)
            detail = (str(detail_path.relative_to(workspace)) if detail_path.is_relative_to(workspace)
                      else str(detail_path))
        except OSError as error:
            detail_path, detail = None, f"not written ({type(error).__name__})"
    return dict(text=selfcheck_text(record, detail, limit=limit), record=_summary_record(record, detail),
                detail_path=None if detail_path is None else str(detail_path))
