"""CCA family grouping and admitted mechanism comparisons, without runtime coordination."""

from __future__ import annotations

import json
import time
from collections import Counter

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
    import merlin.kernels.mechanism_applicability as MA

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
    from merlin.mining.reference_subset_cards import family_key, family_signature

    families = {}
    for member in members:
        member.setdefault("signature", family_signature(member["analysis"]))
        member["family"] = family_key(member["signature"])
        families.setdefault(member["family"], []).append(member)
    return families


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
    from .reference_applicability import summary

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


# Preserve the established callable pickle identities through explicit facade exports.
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
    _export.__module__ = "merlin.experiments.phase1.feedback.cca_families"
