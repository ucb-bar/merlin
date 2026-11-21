"""CCA family grouping and admitted mechanism comparisons, without runtime coordination."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path

from merlin.development.backend.feedback.references import cca_report as _cca_selfcheck
from merlin.development.backend.feedback.references.cca_bank import (
    _KERNEL,
    _MAX_JSON,
    Skip,
    _card_symbols,
    _read_json,
    bank_units,
    case_semantics,
    grade_kernels,
    latest_cycles,
    lift_kernel,
    open_selected_bank,
    selected_decoder,
)
from merlin.development.backend.feedback.references.cca_report import (
    _MAX_ELF,
    _VALUE_CHARS,
    RECORD,
    SELFCHECK_DIFFERENCES,
    SELFCHECK_MECHANISMS,
    SELFCHECK_MEMBERS,
    SELFCHECK_SCHEMA,
    SELFCHECK_TEXT_BYTES,
    SELFCHECK_UNKNOWN_GROUPS,
    SHOWN_CASES,
    SHOWN_FAMILIES,
    SHOWN_ROWS,
    _compact,
    _diff_line,
    _digest,
    _facts_document,
    _family_lines,
    _family_view,
    _flat,
    _jsonable,
    _key,
    _mechanism,
    _short,
    _summary_record,
    _unknown_groups_of,
    _write_json,
    render,
    selfcheck_kernels,
    selfcheck_text,
)

# ---- cca_comparison ----------------------------------------------------------------------------

SCHEMA = "merlin.cca_families.v1"

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


def _kind(effect):
    return not FACT_KINDS or any(str(effect).startswith(kind) for kind in FACT_KINDS)


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
    import merlin.analysis.kernels.mechanisms.applicability as MA

    effects = analysis.get("derived_effects") or {}
    sites = (analysis.get("mechanism_projection") or {}).get("role_sites") or {}
    out = dict(
        roles=dict(
            status="observed",
            value={label: sites[role] if role in sites else 0 for role, label in SUMMARY_ROLES.items()},
        )
        if sites
        else dict(status="unknown", reason="no decoded role sites")
    )

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

    out["transfer_width"] = fact(
        "role_transfer_width",
        lambda v: _per_role(MA._role_entries(v)[0], lambda e: (MA._width(e) or {}).get("elements_per_transfer")),
    )
    out["buffering"] = fact(
        "operand_buffering",
        lambda v: _per_role(
            MA._role_entries(v)[0],
            lambda e: (
                lambda b: (
                    None
                    if not b or b.get("disjoint_windows") is None
                    else dict(windows=b["disjoint_windows"], alternating=b["alternating"])
                )
            )(MA._buffering(e)),
        ),
    )
    out["overlap"] = fact(
        "issue_overlap", lambda v: {SUMMARY_ROLES.get(k, k): row["fraction"] for k, row in MA._overlap(v)[0].items()}
    )
    out["prefetch_distance"] = fact(
        "prefetch_distance", lambda v: {k: row["median_loads_between"] for k, row in MA._distance(v).items()}
    )
    out["output_panel"] = fact(
        "output_group_extent",
        lambda v: (
            {
                k: v[k]
                for k in ("median_readout_elements_per_block", "column_extents", "row_extents")
                if isinstance(v, dict) and k in v
            }
            or (v if isinstance(v, (int, float)) else None)
        ),
    )
    out["residency"] = fact(
        "operand_retention",
        lambda v: (
            {k: v[k] for k in sorted(v) if isinstance(v.get(k), (int, float, bool))} if isinstance(v, dict) else None
        ),
    )
    return out


def _summary_text(summary):
    parts = []
    for name, row in summary.items():
        if row.get("status") == "unknown":
            parts.append(f"{name} unknown")
        else:
            parts.append(f"{name} {json.dumps(row['value'], sort_keys=True, separators=(',', ':'), default=str)}")
    return "; ".join(parts)


def group(members):
    """``{family key: [members]}``; each member carries its signature and identical-body identity."""
    from merlin.analysis.mining.references.subsets import family_key, family_signature

    families = {}
    for member in members:
        member.setdefault("signature", family_signature(member["analysis"]))
        member["family"] = family_key(member["signature"])
        families.setdefault(member["family"], []).append(member)
    return families


EXPERT_UNITS = 3


def nearest_units(signature, units, count):
    """Up to ``count`` distinct bank units by signature agreement (``[(score, occurrence)]``)."""
    from merlin.analysis.mining.references.subsets import signature_similarity

    best = {}
    for unit in units:
        score = signature_similarity(signature, unit["signature"])["score"]
        if unit["unit_id"] not in best or score > best[unit["unit_id"]][0]:
            best[unit["unit_id"]] = (score, unit)
    return sorted(best.values(), key=lambda item: (-item[0], item[1]["unit_id"]))[:count]


def nearest(signature, units, operation=None):
    """The bank occurrence whose signature agrees most, with the basis; ties are listed, never broken silently."""
    from merlin.analysis.mining.references.subsets import signature_similarity

    shortlist = [unit for unit in units if operation is not None and unit.get("operation") == operation]
    basis = (
        "units of the same operation class"
        if shortlist
        else "all bank units (bank units carry no operation class)"
        if all(u.get("operation") is None for u in units)
        else "all bank units (none shares this operation class)"
    )
    scored = [(signature_similarity(signature, unit["signature"]), unit) for unit in (shortlist or units)]
    if not scored:
        return None
    best = max(row["score"] for row, _ in scored)
    tied = [(row, unit) for row, unit in scored if row["score"] == best]
    row, unit = min(tied, key=lambda item: (item[1]["unit_id"], item[1]["region_id"]))
    return dict(
        unit_id=unit["unit_id"],
        region_id=unit["region_id"],
        original_id=unit["original_id"],
        symbol=unit.get("symbol"),
        agreement=row["score"],
        agree=row["agree"],
        differ={
            name: dict(only_yours=value["only_left"], only_unit=value["only_right"])
            for name, value in row["differ"].items()
        },
        tied_units=sorted({u["unit_id"] for _, u in tied} - {unit["unit_id"]}),
        shortlist=basis,
        semantic_equivalence="unknown",
        performance="unknown",
    )


_SUMMARY_KEYS = (
    "value",
    "elements_per_transfer",
    "fraction",
    "disjoint_windows",
    "median_loads_between",
    "column_extent_if_one_row_band",
    "largest_elements",
)


def _value(row, side):
    from merlin.development.backend.feedback.references.applicability import _side

    value = _side(row, side)
    if isinstance(value, dict):
        key = next((key for key in _SUMMARY_KEYS if isinstance(value.get(key), (int, float, bool))), None)
        if key is not None:
            return value[key] if key == "value" else {key: value[key]}
    text = json.dumps(value, sort_keys=True, default=str)
    return value if len(text) <= 80 else text[:77] + "..."


def _directions(rows):
    """``{effect label: direction}`` of the shown comparison kinds."""
    return {
        " ".join(str(row[key]) for key in ("effect", "role", "operand") if row.get(key)): row.get("direction")
        for row in rows
        if _kind(row["effect"])
    }


def _rows(rows):
    """The shown differing rows, order facts first: rows whose two sides are comparable numbers (they
    carry ``comparison``) lead, the rest keep their fixed effect order."""
    shown = [row for row in rows if _kind(row["effect"]) and row["difference"] == "differs"]
    return [
        dict(
            effect=row["effect"],
            difference=row["difference"],
            **{key: row[key] for key in ("role", "operand", "direction", "comparison") if row.get(key)},
            unit=_value(row, "reference"),
            yours=_value(row, "candidate"),
            routes=[route["capability"] for route in row.get("routes") or () if route.get("family_match") is not False],
        )
        for row in sorted(shown, key=lambda row: not row.get("comparison"))
    ]


def _unknown_groups(rows):
    from merlin.development.backend.feedback.references.applicability import summary

    return [row for row in summary([row for row in rows if _kind(row["effect"])])["unknown_by_reason"]]


def _admissions(result):
    return {
        route["capability"] + ":" + row["effect"]: route["admission"].get("status")
        for row in result["differences"]
        for route in row.get("routes") or ()
    }


def build(
    members,
    units,
    *,
    cycles=None,
    operations=None,
    shapes=None,
    geometry=None,
    geometry_reason=None,
    structure=None,
    deadline=None,
    clock=time.monotonic,
):
    """The family record. ``deadline`` is a ``clock()`` value; work past it is skipped with that reason."""
    from merlin.analysis.kernels.mechanisms.applicability import applicability, mechanism_differences

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
        family = dict(
            key=key,
            members=len(rows),
            cases=cases,
            identical_bodies=len(identical),
            operation_classes=sorted({str(operations.get(c)) for c in cases}),
            signature=rows[0]["signature"],
            representative_basis=(
                "most frequent body and slowest member by latest modeled cycles"
                if timed
                else "most frequent body only (no member has timing in the latest grade)"
            ),
        )
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
        family["expert_units"] = [
            dict(
                unit_id=row["unit_id"],
                region_id=row["region_id"],
                agreement=score,
                mechanisms=mechanism_summary(row["analysis"]),
            )
            for score, row in nearest_units(frequent["signature"], units, EXPERT_UNITS)
        ]
        family["member_mechanisms"] = [
            dict(
                case=m["case"],
                region_id=m["region_id"],
                shape=shapes.get(m["case"]),
                cycles=cycles.get(m["case"]),
                mechanisms=mechanism_summary(m["analysis"]),
            )
            for m in sorted(rows, key=lambda m: (m["case"], m["region_id"]))
        ]
        unit = next(
            (u for u in units if match and (u["unit_id"], u["region_id"]) == (match["unit_id"], match["region_id"])),
            None,
        )
        representatives = [("most_frequent", frequent)]
        if slowest is not None and slowest is not frequent:
            representatives.append(("slowest", slowest))
        family["representatives"] = []
        base, base_directions = None, None
        for role, member in representatives:
            entry = dict(
                role=role, case=member["case"], region_id=member["region_id"], cycles=cycles.get(member["case"])
            )
            if unit is None or late():
                entry["status"] = "skipped: budget exhausted" if unit is not None else "no bank unit"
            else:
                result = applicability(
                    unit["analysis"],
                    member["analysis"],
                    shape=shapes.get(member["case"]),
                    geometry=geometry,
                    geometry_reason=geometry_reason,
                    structure=structure,
                )
                entry.update(
                    status="compared",
                    differences=_rows(result["differences"]),
                    unknown_by_reason=_unknown_groups(result["differences"]),
                    same=sum(row["difference"] == "same" for row in result["differences"]),
                    unknown=sum(row["difference"] == "unknown" for row in result["differences"]),
                    missing_capabilities=len(result["missing_capabilities"]),
                )
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
                    mine = _directions(mechanism_differences(unit["analysis"], member["analysis"], structure=structure))
                    deviating = sorted(
                        label
                        for label in set(mine) | set(base_directions)
                        if mine.get(label) != base_directions.get(label)
                    )
                else:
                    deviating = sorted(
                        {
                            row["effect"]
                            for row in mechanism_differences(
                                frequent["analysis"], member["analysis"], structure=structure
                            )
                            if row["difference"] == "differs" and _kind(row["effect"])
                        }
                    )
                if deviating:
                    exceptions.append(dict(case=member["case"], region_id=member["region_id"], differs_in=deviating))
            if unit is not None and base is not None and shapes.get(member["case"]) != shapes.get(frequent["case"]):
                admitted = _admissions(
                    applicability(
                        unit["analysis"],
                        member["analysis"],
                        shape=shapes.get(member["case"]),
                        geometry=geometry,
                        geometry_reason=geometry_reason,
                        structure=structure,
                    )
                )
                changed = sorted(name for name in set(base) | set(admitted) if base.get(name) != admitted.get(name))
                if changed:
                    shape_rows.append(
                        dict(
                            case=member["case"],
                            shape=shapes.get(member["case"]),
                            admissions={name: admitted.get(name) for name in changed},
                        )
                    )
        family.update(
            status="compared",
            exceptions=exceptions,
            shape_dependent=shape_rows,
            exception_basis=(
                "direction relative to the nearest unit differs from the most frequent representative's"
                if base_directions is not None
                else "static comparison with the most frequent representative's body"
            ),
            **({"skipped_members": dict(cases=pending, reason="budget exhausted")} if pending else {}),
        )
        out.append(family)
    return dict(
        schema=SCHEMA,
        families=out,
        kernels=len(members),
        skipped_families=skipped,
        fact_kinds=list(FACT_KINDS),
        ordering="families by member count; nothing is ranked by speed",
        semantic_equivalence="unknown",
        performance="unknown",
    )


for _export in (
    _kind,
    _per_role,
    mechanism_summary,
    _summary_text,
    group,
    nearest_units,
    nearest,
    _value,
    _directions,
    _rows,
    _unknown_groups,
    _admissions,
    build,
):
    _export.__module__ = "merlin.development.backend.feedback.references.cca_families"


# ---- cca_families ------------------------------------------------------------------------------
# CCA families in the grade feedback of a CCA bank arm: your kernels beside the nearest expert unit.
#
# Measured on the CCA coverage arms: authors made about 10,000 tool calls and asked the bank service
# twice, both ``inventory``. The service was one paragraph near the end of TASK.md and absent from the
# feedback the authors read. This module brings its comparison into ``qa/round_brief.md``.
#
# After a published grade, the emitted kernels of the grade (each case's linked ``package_kernel.elf``)
# are lifted with the bank's own pinned decoder, through the same lifter and the same effect derivation
# that built the bank, and grouped three ways, all derived by tooling:
#
# * **identical**: the SHA256 of the linked accelerator-region bytes (the bank's own unit identity);
# * **family**: :func:`merlin.analysis.mining.references.subsets.family_signature` (decoded roles,
#   instruction classes, configured-field set, loop-nest shape, control status, non-numeric dataflow
#   facts), the same function over bank units and over authors' kernels; counts and shapes excluded;
# * **operation class**: the case's declared source semantics (:func:`.family_rollup.describe`), used
#   only to shortlist bank units when units carry one; bank units carry none today, which is stated.
#
# Per family: the nearest bank unit by signature agreement, with the fields that agree and differ;
# representatives (the most frequent body, and the slowest member by the latest modeled cycles when any
# member has them); the applicability comparison of each representative with that unit
# (:func:`merlin.analysis.kernels.mechanisms.applicability.applicability`); a member-by-member comparison with the
# representative (:func:`~merlin.analysis.kernels.mechanisms.applicability.mechanism_differences`) whose deviating
# members are listed as exceptions; and, per member, the shape-dependent route admissions.
#
# Facts only: nothing is ranked by assumed speed, structural agreement is never semantic equivalence, and
# performance stays unknown. An arm without a round brief gets the same comparison for the cases of one
# self-check from :func:`selfcheck_section` (bounded text, full record in a named detail file, cached by
# ELF digest). Enabled only for a CCA bank profile. It runs off the grading path in one
# background thread with a wall-clock budget; work left at the budget is skipped with that reason. It
# never raises into a grade. No target name, regex or opcode literal.

BUDGET_S = 300.0

_LOCK = threading.Lock()


def brief_lines(workspace):
    """The cached section; ``[]`` outside a bank arm or before the first record."""
    if _bank_pin() is None:
        return []
    return render(_read_json(Path(workspace) / "qa" / RECORD))


def _bank_pin():
    try:
        from merlin.development.backend.feedback.references.reference_bank import selected_pin

        return selected_pin()
    except Exception:  # noqa: BLE001 -- no profile, no section
        return None


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
        from merlin.analysis.kernels.mechanisms.applicability import selected_geometry

        geometry, reason = selected_geometry(target, json.loads(Path(decoder_files["facts"]).read_text()))
        record = build(
            members,
            units,
            cycles=latest_cycles(workspace),
            operations=operations,
            shapes=shapes,
            geometry=geometry,
            geometry_reason=reason,
            structure=structure,
            deadline=deadline,
            clock=clock,
        )
        record.update(
            base,
            bank_sha256=pin,
            bank_units=len({u["unit_id"] for u in units}),
            not_lifted=failed,
            seconds=round(clock() - started, 1),
        )
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
            from merlin.development.backend.feedback import disclosure as case_declarations

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


SELFCHECK_BUDGET_S = 60.0

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
    regions = json.loads(
        json.dumps(
            _jsonable(lift_kernel(path, decoder=decoder, facts=decoder_files["facts"], structure=structure)),
            default=str,
        )
    )
    try:
        _write_json(cache, dict(elf_sha256=digest, regions=regions))
    except OSError as error:
        logging.getLogger(__name__).warning("CCA lift cache not written: %s", type(error).__name__)
    return regions, False


def selfcheck_section(
    workspace,
    selfcheck_dir,
    *,
    arm,
    bank_root=None,
    facts=None,
    target=None,
    pin=None,
    cases=None,
    budget_s=SELFCHECK_BUDGET_S,
    detail_dir=None,
    cache_dir=None,
    limit=SELFCHECK_TEXT_BYTES,
    clock=time.monotonic,
):
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
        record = _selfcheck_record(
            workspace,
            selfcheck_dir,
            bank_root=bank_root,
            facts=facts,
            target=target,
            pin=pin or arm_pin,
            cases=cases,
            budget_s=budget_s,
            cache_dir=cache_dir,
            started=started,
            deadline=started + budget_s,
            clock=clock,
        )
    except Skip as reason:
        record = dict(schema=SELFCHECK_SCHEMA, status="skipped", reason=str(reason), budget_s=budget_s)
    except Exception as error:  # noqa: BLE001 -- feedback never fails a self-check
        record = dict(
            schema=SELFCHECK_SCHEMA,
            status="skipped",
            budget_s=budget_s,
            reason=f"{type(error).__name__}: {str(error)[:200]}",
        )
    record["seconds"] = round(clock() - started, 3)
    detail_path, detail = None, None
    if record.get("status") != "skipped":
        detail_path = detail_dir / f"cca_families_{record['key'][:16]}.json"
        try:
            _write_json(detail_path, record)
            detail = (
                str(detail_path.relative_to(workspace)) if detail_path.is_relative_to(workspace) else str(detail_path)
            )
        except OSError as error:
            detail_path, detail = None, f"not written ({type(error).__name__})"
    return dict(
        text=selfcheck_text(record, detail, limit=limit),
        record=_summary_record(record, detail),
        detail_path=None if detail_path is None else str(detail_path),
    )


def _selfcheck_record(
    workspace, selfcheck_dir, *, bank_root, facts, target, pin, cases, budget_s, cache_dir, started, deadline, clock
):
    """Retain the existing live selected-source callback boundary."""
    return _cca_selfcheck._selfcheck_record(
        workspace,
        selfcheck_dir,
        bank_root=bank_root,
        facts=facts,
        target=target,
        pin=pin,
        cases=cases,
        budget_s=budget_s,
        cache_dir=cache_dir,
        started=started,
        deadline=deadline,
        clock=clock,
        open_selected_bank=lambda *args, **kwargs: open_selected_bank(*args, **kwargs),
        _decoder_for=lambda *args, **kwargs: _decoder_for(*args, **kwargs),
        _units_for=lambda *args, **kwargs: _units_for(*args, **kwargs),
        case_semantics=lambda *args, **kwargs: case_semantics(*args, **kwargs),
        _lifted=lambda *args, **kwargs: _lifted(*args, **kwargs),
    )
