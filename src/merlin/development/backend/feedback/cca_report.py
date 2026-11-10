"""Bounded CCA family and selfcheck presentation, without grading or input selection."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.development.backend.feedback.cca_comparison import (  # noqa: F401 -- preserve established imports at this boundary
    EXPERT_UNITS,
    SCHEMA,
    _summary_text,
    group,
)

RECORD = "cca_families.json"


SHOWN_FAMILIES = 8


SHOWN_ROWS = 6


SHOWN_CASES = 4


def _short(identity):
    return identity if len(identity) <= 28 else identity[:28] + "..."


def _diff_line(row):
    label = row["effect"] + "".join(f" {row[key]}" for key in ("role", "operand") if row.get(key))
    routes = f"; library routes: {', '.join(row['routes'][:3])}" if row.get("routes") else ""
    return (
        f"    - {label}: unit {json.dumps(row['unit'], default=str)} vs yours "
        f"{json.dumps(row['yours'], default=str)} ({row['difference']}){routes}"
    )


def render(record):
    """Compact Markdown section for ``qa/round_brief.md``."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return []
    head = [
        "## CCA families: your kernels beside the nearest expert unit",
        "",
        f"The CCA bank service is enabled for this arm (bank `{str(record.get('bank_sha256'))[:12]}`, "
        f"{record.get('bank_units')} units).",
        "",
    ]
    if record.get("status") == "skipped":
        return head + [f"Not computed for grade {record.get('grade_token')}: {record.get('reason')}.", ""]
    families = record.get("families") or []
    lines = head + [
        f"Grade {record.get('grade_token')}: {record.get('kernels')} emitted accelerator kernels in "
        f"{len(families)} families (same decoded roles, configured fields, loop-nest shape and dataflow; shapes "
        f"and counts ignored). Each family is compared with the bank unit whose structure agrees most: your "
        "loads, buffering, overlap and prefetch beside the expert's. Full record: `qa/" + RECORD + "`.",
        "",
    ]
    for index, family in enumerate(families[:SHOWN_FAMILIES], 1):
        cases = family["cases"]
        named = ", ".join(f"`{case}`" for case in cases[:SHOWN_CASES]) + (
            f" +{len(cases) - SHOWN_CASES}" if len(cases) > SHOWN_CASES else ""
        )
        lines.append(
            f"- **F{index}** `{family['key']}`: {family['members']} kernels, {len(cases)} cases "
            f"({named}), {family['identical_bodies']} distinct bodies; operation classes "
            f"{', '.join(family['operation_classes'])}."
        )
        if family.get("status") != "compared":
            lines.append(f"  {family.get('status')}.")
            continue
        match = family.get("nearest_unit")
        if match:
            differ = (
                "; ".join(
                    f"{name} (yours only: {len(v['only_yours'])}, unit only: {len(v['only_unit'])})"
                    for name, v in match["differ"].items()
                )
                or "none"
            )
            lines.append(
                f"  Nearest unit `{_short(match['unit_id'])}` region `{match['region_id']}` "
                f"(structural agreement {match['agreement']}; agree: {', '.join(match['agree']) or 'none'}; "
                f"differ: {differ}" + (f"; tied: {len(match['tied_units'])} more" if match["tied_units"] else "") + ")."
            )
        for rep in family.get("representatives") or ():
            cycles = f", {rep['cycles']} modeled cycles" if rep.get("cycles") is not None else ""
            if rep.get("status") != "compared":
                lines.append(f"  {rep['role']} `{rep['case']}`{cycles}: {rep.get('status')}.")
                continue
            lines.append(
                f"  {rep['role']} `{rep['case']}`{cycles}: {len(rep['differences'])} differences, "
                f"{rep['same']} same, {rep['unknown']} unknown."
            )
            lines += [_diff_line(row) for row in rep["differences"][:SHOWN_ROWS]]
            lines += [
                f"    - unknown ({group['reason']}): {', '.join(group['effects'][:6])}"
                + (f" +{len(group['effects']) - 6}" if len(group["effects"]) > 6 else "")
                for group in (rep.get("unknown_by_reason") or ())[:3]
            ]
        for unit in (family.get("expert_units") or ())[:EXPERT_UNITS]:
            lines.append(
                f"  expert `{_short(unit['unit_id'])}` (agreement {unit['agreement']}): "
                + _summary_text(unit["mechanisms"])
            )
        for row in (family.get("member_mechanisms") or ())[:SHOWN_CASES]:
            shape = ",".join(f"{k}={v}" for k, v in (row.get("shape") or {}).items()) or "shape undeclared"
            lines.append(f"  yours `{row['case']}` ({shape}): " + _summary_text(row["mechanisms"]))
        if family.get("exceptions"):
            lines.append(
                "  Exceptions (stand differently from the unit than the most frequent representative): "
                + "; ".join(
                    f"`{row['case']}` in {', '.join(sorted({label.split(' ')[0] for label in row['differs_in']}))}"
                    for row in family["exceptions"][:SHOWN_CASES]
                )
                + (f" +{len(family['exceptions']) - SHOWN_CASES}" if len(family["exceptions"]) > SHOWN_CASES else "")
                + "."
            )
        if family.get("shape_dependent"):
            lines.append(
                f"  Shape-dependent route admissions differ for {len(family['shape_dependent'])} members "
                "(see the record)."
            )
        if family.get("skipped_members"):
            lines.append(f"  {len(family['skipped_members']['cases'])} members not compared: budget exhausted.")
        if match:
            lines.append(
                f"  Ask: `python agent_selfcheck.py --reference-analysis applicability --reference-entry "
                f"{match['unit_id']} --reference-region {match['region_id']} --candidate-path "
                f"{(Path('artifacts') / 'FILE').as_posix()} --candidate-shape m=M,k=K,n=N`."
            )
    if len(families) > SHOWN_FAMILIES:
        lines.append(f"- {len(families) - SHOWN_FAMILIES} more families in the record.")
    return lines + [
        "",
        "Structural agreement is not semantic equivalence; performance effects are unknown until measured.",
        "",
    ]


def _jsonable(value):
    """String keys throughout: derived records may key a mapping by operand index or by ``None``."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


SELFCHECK_SCHEMA = "merlin.cca_families.selfcheck.v1"


SELFCHECK_TEXT_BYTES = 2048


SELFCHECK_DIFFERENCES = 4


SELFCHECK_UNKNOWN_GROUPS = 3


SELFCHECK_MEMBERS = 3


SELFCHECK_MECHANISMS = (
    ("roles", "roles"),
    ("transfer_width", "width"),
    ("buffering", "buffering"),
    ("overlap", "overlap"),
    ("prefetch_distance", "prefetch distance"),
    ("output_panel", "output group"),
    ("residency", "residency"),
)


_VALUE_CHARS = 56


def _flat(value):
    """``k=v`` pairs, nested mappings in parentheses; floats to three decimals."""
    if isinstance(value, float):
        return str(round(value, 3))
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, dict):
        return " ".join(
            f"{key}={_flat(item)}" if not isinstance(item, dict) else f"{key}({_flat(item)})"
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_flat(item) for item in value) + "]"
    return str(value)


def _compact(value, chars=_VALUE_CHARS):
    text = _flat(value)
    return text if len(text) <= chars else text[: chars - 3] + "..."


def _mechanism(summary, name):
    row = (summary or {}).get(name) or {}
    if row.get("status") in (None, "unknown"):
        return "unknown"
    value = row.get("value")
    if name == "buffering" and isinstance(value, dict):
        # windows per role, "alt" when the windows alternate
        value = {
            role: f"{(entry or {}).get('windows')}w" + (" alt" if (entry or {}).get("alternating") else "")
            for role, entry in value.items()
        }
    elif name == "output_panel" and isinstance(value, dict):
        value = {
            key.split("_")[0] if key.endswith("_extents") else "per_block": (
                sorted(item, key=str) if isinstance(item, dict) else item
            )
            for key, item in value.items()
        }
    return _compact(value)


def _unknown_groups_of(family):
    """Unknowns of one family grouped by reason: comparison effects and unestablished mechanism facts."""
    groups = {}
    for group in (family.get("compared") or {}).get("unknown_by_reason") or ():
        groups.setdefault(str(group.get("reason")), []).extend(group.get("effects") or ())
    sides = [("unit", family.get("nearest_mechanisms"))] + [
        (f"`{row['case']}`", row.get("mechanisms")) for row in family.get("member_mechanisms") or ()
    ]
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
    differ = (
        ", ".join(
            f"{name} (yours only {len(v['only_yours'])}, unit only {len(v['only_unit'])})"
            for name, v in match["differ"].items()
        )
        or "none"
    )
    tied = f", {len(match['tied_units'])} tied" if match.get("tied_units") else ""
    lines.append(
        f"  Nearest unit {symbol} `{_short(match['unit_id'])}` {match['region_id']} (agreement "
        f"{match['agreement']}{tied}); agree: {', '.join(match['agree']) or 'none'}; differ: {differ}."
    )
    members = (family.get("member_mechanisms") or ())[:SELFCHECK_MEMBERS]
    names = [f"`{row['case']}`" + (f" {row['region_id'][:15]}" if len(members) > 1 else "") for row in members]
    lines.append("  Mechanisms, unit | " + " | ".join(names) + ":")
    lines += [
        f"    {label}: {_mechanism(family.get('nearest_mechanisms'), name)} | "
        + " | ".join(_mechanism(row.get("mechanisms"), name) for row in members)
        for name, label in SELFCHECK_MECHANISMS
    ]
    compared = family.get("compared")
    if compared:
        rows = compared.get("differences") or ()
        lines.append(
            f"  Differences (unit vs yours, order facts first): {len(rows)} differ, "
            f"{compared.get('same')} same, {compared.get('unknown')} unknown."
        )
        for row in rows[:SELFCHECK_DIFFERENCES]:
            label = row["effect"] + "".join(f" {row[key]}" for key in ("role", "operand") if row.get(key))
            lines.append(f"    - {label}: {_compact(row['unit'])} vs {_compact(row['yours'])}")
    for reason, effects in _unknown_groups_of(family)[:SELFCHECK_UNKNOWN_GROUPS]:
        shown = ", ".join(effects[:4]) + (f" +{len(effects) - 4}" if len(effects) > 4 else "")
        lines.append(f"  Unknown ({reason[:120]}): {shown}.")
    if family.get("exceptions"):
        lines.append(
            "  Exceptions: "
            + "; ".join(
                f"`{row['case']}` in {', '.join(sorted({label.split(' ')[0] for label in row['differs_in']}))}"
                for row in family["exceptions"][:SELFCHECK_MEMBERS]
            )
            + "."
        )
    return lines


def selfcheck_text(record, detail, *, limit=SELFCHECK_TEXT_BYTES):
    """The compact section for one self-check reply, at most ``limit`` UTF-8 bytes."""
    head = "CCA families (this arm's CCA bank service)"
    if record.get("status") == "skipped":
        return f"{head}: not computed for this self-check: {record.get('reason')}."[:limit]
    bank = record.get("bank") or {}
    families, cases = record.get("families") or [], record.get("cases") or {}
    kernels = sum(len(row["kernels"]) for row in cases.values())
    lines = [
        f"{head}, bank `{str(bank.get('sha256'))[:12]}` ({bank.get('units')} units): {len(cases)} case(s), "
        f"{kernels} kernel(s), {len(families)} famil{'y' if len(families) == 1 else 'ies'}"
        + (", cached (kernels unchanged)" if record.get("cache") == "hit" else "")
        + f". Full record: `{detail}`."
    ]
    if record.get("status") == "partial":
        lines.append(f"Partial: the {record.get('budget_s')}s budget was exhausted; the rest is skipped.")
    for case, row in sorted(cases.items()):
        lines.append(
            f"- `{case}`: "
            + (", ".join(f"{k['region_id'][:15]} -> {k['family']}" for k in row["kernels"]) or "no accelerator region")
        )
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
        return dict(
            schema=SELFCHECK_SCHEMA,
            status="skipped",
            reason=record.get("reason"),
            budget_s=record.get("budget_s"),
            seconds=record.get("seconds"),
        )
    families = []
    for family in record.get("families") or ():
        match = family.get("nearest_unit") or {}
        compared = family.get("compared") or {}
        families.append(
            dict(
                label=family["label"],
                key=family["key"],
                cases=family["cases"],
                members=family["members"],
                status=family.get("status"),
                nearest_unit=dict(
                    unit_id=match.get("unit_id"),
                    symbol=match.get("symbol"),
                    region_id=match.get("region_id"),
                    agreement=match.get("agreement"),
                    agree=match.get("agree"),
                    differ=sorted(match.get("differ") or ()),
                )
                if match
                else None,
                top_differences=[
                    dict(
                        effect=row["effect"],
                        **{k: row[k] for k in ("role", "operand", "comparison") if row.get(k)},
                        unit=row["unit"],
                        yours=row["yours"],
                    )
                    for row in (compared.get("differences") or ())[:SELFCHECK_DIFFERENCES]
                ],
                unknown_by_reason=[
                    dict(reason=reason, count=len(effects)) for reason, effects in _unknown_groups_of(family)
                ],
            )
        )
    bank = record.get("bank") or {}
    return dict(
        schema=SELFCHECK_SCHEMA,
        status=record["status"],
        cache=record.get("cache"),
        seconds=record.get("seconds"),
        budget_s=record.get("budget_s"),
        detail=detail,
        bank=dict(sha256=bank.get("sha256"), units=bank.get("units")),
        cases={
            case: dict(
                elf_sha256=row["elf_sha256"],
                kernels=[dict(region_id=k["region_id"], family=k["family"]) for k in row["kernels"]],
            )
            for case, row in (record.get("cases") or {}).items()
        },
        not_lifted=record.get("not_lifted") or {},
        families=families,
        semantic_equivalence="unknown",
        performance="unknown",
    )


# Preserve the established callable pickle identities through explicit facade exports.
for _export in (
    _short,
    _diff_line,
    render,
    _jsonable,
    _flat,
    _compact,
    _mechanism,
    _unknown_groups_of,
    _family_lines,
    selfcheck_text,
    _summary_record,
):
    _export.__module__ = "merlin.development.backend.feedback.cca_families"
