"""Turn whole-model measurements into feedback an optimizing agent can act on.

A headline cycle count names no edit.  What does: for EACH group, how many cycles ours took against
how many the reference implementation took ON THE SAME MACHINE, which route each took (a native
convolution or a materialised im2col, the library's call or the package's own kernel), how much of
it was a host-side gather, and whether it was right.  The groups holding the gap, sorted by how much
of it they hold, are the work list; nothing here says what to change in them, because that is the
agent's job and a hint written here would be a guess dressed as a measurement.

THE COMPARISON REFUSES UNLIKE MACHINES.  A reference measured on a different device (a different
emulator binary, a different bitstream) is not a divisor: the two designs differ in exactly the
readout paths whole-model programs exercise.  A reference that is not itself MEASURED (valid) is not
a bar either -- a wrong reference is a number, not a target.

``distance_to_bar`` is always stated when both sides are measured: cycles, the ratio, and which
groups hold the gap.  When ours is INVALID the distance is still shown (a wrong program's time is
where its time went) but it is labelled as not an achievement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.perf import whole_model_verdict as V

SCHEMA = "merlin_whole_model_feedback_v1"


def _device_key(result: Mapping[str, Any]) -> tuple[str, str]:
    device = result.get("device") or {}
    return (str(device.get("machine") or ""), str(device.get("binary_sha256") or ""))


def _routes(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    build = result.get("build") or {}
    rows = build.get("groups") or []
    return {str(row.get("group")): row for row in rows if isinstance(row, Mapping) and row.get("group") is not None}


def _rows(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["group"]): row for row in V.group_table(result.get("verdict") or {})}


def reference_admissible(result: Mapping[str, Any], reference: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(reference, Mapping):
        return False, "no reference measurement was supplied"
    if reference.get("timing_status") not in (V.TIMING_MEASURED, V.TIMING_MEASURED_INVALID):
        return False, f"the reference is {reference.get('timing_status')}, not a measurement"
    if _device_key(result) != _device_key(reference):
        return False, (
            f"the reference ran on {_device_key(reference)} and this measurement on {_device_key(result)}; "
            "cycle counts from different devices are never compared"
        )
    return True, "same device, reference valid"


def _emitted_by(attribution: Mapping[str, Any] | None, group: str) -> dict[str, Any] | None:
    """Where in the package a group's lowering came from, as the attribution recorded it."""
    groups = (attribution or {}).get("groups") if isinstance(attribution, Mapping) else None
    entry = (groups or {}).get(group) if isinstance(groups, Mapping) else None
    if not isinstance(entry, Mapping):
        return None
    if entry.get("same_lowering_as"):
        return {"same_lowering_as": entry["same_lowering_as"]}
    if entry.get("specific_functions") or entry.get("specific_files"):
        return {
            "files": list(entry.get("specific_files") or [])[:6],
            "functions": list(entry.get("specific_functions") or [])[:8],
        }
    if entry.get("components"):
        return {"components": list(entry["components"])[:8]}
    return {"unavailable": entry.get("reason")} if entry.get("reason") else None


def failing_groups(
    verdict: Mapping[str, Any],
    routes: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None,
    attribution: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Every FAILED group, in model order: its op, what is wrong in words, its route, and its code."""
    if not isinstance(routes, Mapping):
        routes = {str(r.get("group")): r for r in routes or () if isinstance(r, Mapping)}
    rows = []
    for row in V.group_table(verdict):
        if row.get("state") != V.GROUP_FAILED:
            continue
        group = str(row["group"])
        route = routes.get(group) or {}
        rows.append(
            {
                "group": group,
                "op": route.get("op") or row.get("kind"),
                "kind": row.get("kind"),
                "signature": row.get("detail"),
                "compare": row.get("compare"),
                "worst_element": row.get("worst_element"),
                "inherited_from": row.get("inherited_from"),
                "route": _route_brief(route),
                "emitted_by": _emitted_by(attribution, group),
            }
        )
    return rows


def failing_line(entry: Mapping[str, Any]) -> str:
    """``g6 residual_add (sum) FAILED: <what is wrong> | route ... | emitted by ...`` -- one line."""
    route = entry.get("route") or {}
    kind = f" ({entry['kind']})" if entry.get("kind") and entry.get("kind") != entry.get("op") else ""
    text = f"g{entry['group']} {entry.get('op')}{kind} FAILED: {entry.get('signature') or 'no detail recorded'}"
    text += f" | route {route.get('on')}/{route.get('lowering') or route.get('call')}"
    emitted = entry.get("emitted_by") or {}
    if emitted.get("same_lowering_as"):
        text += f" | emitted by the same code as g{emitted['same_lowering_as']}"
    elif emitted.get("functions"):
        text += " | emitted by " + ", ".join(emitted["functions"][:4])
    elif emitted.get("components"):
        text += " | package components " + ", ".join(emitted["components"][:4])
    return text


#: The route a group takes when the candidate's own package compiled it.
PACKAGE_ROUTE = "package"


def package_authored(result: Mapping[str, Any], reference: Mapping[str, Any] | None) -> dict[str, Any]:
    """How much of the model the PACKAGE compiled: groups it answered, and their share of the work,
    priced by the same-machine reference's own per-group cycles (the vendor library's cost of each
    group). A group declined to the library is the library's cycles, not the package's."""
    routes = {
        str(r.get("group")): r for r in ((result.get("build") or {}).get("groups") or ()) if isinstance(r, Mapping)
    }
    answered = sorted((g for g, r in routes.items() if r.get("on") == PACKAGE_ROUTE), key=V._order)
    price = {
        str(row["group"]): int(row.get("cycles") or 0) for row in V.group_table((reference or {}).get("verdict") or {})
    }
    total = sum(price.values())
    priced = sum(price.get(g, 0) for g in answered)
    return {
        "groups": answered,
        "groups_answered": len(answered),
        "groups_total": len(routes),
        "priced_cycles": priced if total else None,
        "priced_share": round(priced / total, 4) if total else None,
        "pricing": "the same-machine reference's own cycles per group",
    }


def coverage_text(authored: Mapping[str, Any] | None) -> str:
    """``54/71 groups, 93.2%`` -- the package-authored share printed beside a cycle count."""
    authored = authored or {}
    share = authored.get("priced_share")
    return (
        f"{authored.get('groups_answered')}/{authored.get('groups_total')} groups, "
        f"{'-' if share is None else f'{100 * float(share):.1f}%'}"
    )


def compare(result: Mapping[str, Any], reference: Mapping[str, Any] | None, *, top: int = 15) -> dict[str, Any]:
    """Per-group ours-vs-reference on one machine, the distance to the bar, and the gap holders."""
    status = result.get("timing_status")
    verdict = result.get("verdict") or {}
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "package_sha256": result.get("package_sha256"),
        "timing_status": status,
        "machine": (result.get("device") or {}).get("machine"),
        "device_artifact": (result.get("device") or {}).get("artifact"),
        "device_binary_sha256": (result.get("device") or {}).get("binary_sha256"),
        "objective_cycles": result.get("objective_cycles"),
        "whole_window_cycles": verdict.get("whole_window_cycles"),
        "correctness": verdict.get("correctness"),
    }
    if status not in (V.TIMING_MEASURED, V.TIMING_MEASURED_INVALID):
        document["reason"] = result.get("refusal") or verdict.get("refusal") or "no admissible reading"
        return document
    ours = _rows(result)
    routes = _routes(result)
    document["package_authored"] = package_authored(result, reference)
    document["failing_groups"] = failing_groups(verdict, routes, result.get("attribution"))
    admissible, why = reference_admissible(result, reference)
    theirs = _rows(reference) if admissible and reference is not None else {}
    reference_routes = _routes(reference) if admissible and reference is not None else {}
    table = []
    for key, row in sorted(ours.items(), key=lambda item: V._order(item[0])):
        other = theirs.get(key)
        entry: dict[str, Any] = {
            "group": key,
            "kind": row.get("kind"),
            "cycles": row.get("cycles"),
            "gather_cycles": row.get("gather_cycles"),
            "kernel_cycles": row.get("kernel_cycles"),
            "correct": row.get("correct"),
            "state": row.get("state"),
            "correctness_detail": row.get("detail"),
            "route": routes.get(key),
        }
        if other is not None:
            entry["reference_cycles"] = other.get("cycles")
            entry["reference_route"] = reference_routes.get(key)
            entry["delta_cycles"] = int(row["cycles"]) - int(other["cycles"])
            entry["ratio"] = round(int(row["cycles"]) / int(other["cycles"]), 4) if other.get("cycles") else None
        table.append(entry)
    whole = verdict.get("whole_window_cycles")
    gather = sum(int(row.get("gather_cycles") or 0) for row in ours.values())
    document["host_gather_cycles"] = gather
    document["host_gather_share"] = round(gather / whole, 4) if whole else None
    document["groups"] = table
    gaps = roofline_gaps(result, ours, theirs)
    if gaps:
        document["roofline_gaps"] = gaps
        document["distance_to_roofline"] = {
            "ours_cycles": sum(g["ours"] for g in gaps),
            "roofline_cycles": sum(g["roofline"] for g in gaps),
            "ratio": round(sum(g["ours"] for g in gaps) / sum(g["roofline"] for g in gaps), 3),
            "groups": len(gaps),
        }
    document["by_kind"] = _by_kind(table)
    document["by_kind_and_route"] = _by_kind_and_route(table)
    document["reference"] = {"admissible": admissible, "reason": why}
    ours_elf = (result.get("build") or {}).get("elf_sha256")
    if (
        admissible
        and reference is not None
        and ours_elf
        and ours_elf == (reference.get("build") or {}).get("elf_sha256")
    ):
        # THE SAME PROGRAM AS THE REFERENCE.  A package that answers no group links the library for every
        # one, and its "measurement" is the reference's under the package's digest -- a result that
        # would read as parity while measuring nothing the package did.
        document["identical_to_reference_program"] = True
    if admissible and reference is not None:
        bar = int((reference.get("verdict") or {}).get("whole_window_cycles"))
        document["reference"].update(
            package_sha256=reference.get("package_sha256"),
            label=reference.get("label"),
            whole_window_cycles=bar,
        )
        gap = int(whole) - bar
        holders = sorted(
            (entry for entry in table if isinstance(entry.get("delta_cycles"), int)),
            key=lambda entry: -int(entry["delta_cycles"]),
        )
        positive = sum(max(0, int(entry["delta_cycles"])) for entry in holders)
        document["reference"]["correct"] = reference.get("timing_status") == V.TIMING_MEASURED
        document["reference"]["role"] = "context_only"
        document["distance_to_bar"] = {
            "role": "context_only: the vendor reference's cycles orient; the roofline is the machine's bound",
            "ours_whole_window_cycles": int(whole),
            "bar_whole_window_cycles": bar,
            "gap_cycles": gap,
            "ratio": round(int(whole) / bar, 4) if bar else None,
            "achieved": status == V.TIMING_MEASURED and gap <= 0,
            "counts_as_achievement": status == V.TIMING_MEASURED,
            "note": " ".join(
                note
                for note in (
                    None
                    if status == V.TIMING_MEASURED
                    else "this program is INVALID (wrong output); its distance is where its time went, not a result.",
                    None
                    if reference.get("timing_status") == V.TIMING_MEASURED
                    else "The reference arm is itself INVALID on this machine; its cycles orient, they are not a bar.",
                )
                if note
            )
            or None,
            "uncounted_delta_cycles": (int(whole) - int(verdict.get("bracketed_sum_cycles") or whole))
            - (bar - int((reference.get("verdict") or {}).get("bracketed_sum_cycles") or bar)),
        }
        census_by_group = ((result.get("build") or {}).get("isa_census") or {}).get("per_group") or {}
        document["gap_holders"] = [
            {
                "group": entry["group"],
                "kind": entry["kind"],
                "ours": entry["cycles"],
                "reference": entry["reference_cycles"],
                "delta_cycles": entry["delta_cycles"],
                "share_of_positive_gap": round(int(entry["delta_cycles"]) / positive, 4) if positive else None,
                "gather_cycles": entry["gather_cycles"],
                "route": _route_brief(entry.get("route")),
                "op": (entry.get("route") or {}).get("op") or entry["kind"],
                "emitted_by": _emitted_by(result.get("attribution"), entry["group"]),
                "reference_route": _route_brief(entry.get("reference_route")),
                "correct": entry["correct"],
                "state": entry.get("state"),
                "worst_element": (ours.get(entry["group"]) or {}).get("worst_element"),
                # DIAGNOSTIC ONLY -- what issued, never what should have. Counts by the target's own
                # derived instruction role (mvin/preload/compute/mvout/fence and the like) and the
                # symbol that emitted each, read from the same disassembly the no-FSM gate already
                # paid for; a group with no entry here answered no instruction of its own (it is
                # answered by the library or the host).
                "instruction_census": census_by_group.get(entry["group"]),
            }
            for entry in holders[:top]
            if int(entry["delta_cycles"]) > 0
        ]
        document["groups_faster_than_reference"] = [
            entry["group"] for entry in holders if int(entry["delta_cycles"]) < 0
        ]
    return document


def roofline_gaps(
    result: Mapping[str, Any], ours: Mapping[str, Mapping[str, Any]], theirs: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Every group with a derived roofline (``result["diagnostics"]``), ranked by ours minus the roofline:
    INCLUDING the groups already faster than the reference, because the reference is another
    implementation's cycles and the roofline is the machine's. Each carries its efficiency numbers --
    what its program issued against the floor -- and never a suggested change."""
    per_group = ((result.get("diagnostics") or {}).get("per_group")) or {}
    gaps = []
    for key, row in ours.items():
        found = per_group.get(str(key)) or {}
        roofline = found.get("roofline") or {}
        bound = roofline.get("roofline_cycles") if roofline.get("status") == "derived" else None
        cycles = row.get("cycles")
        if not isinstance(bound, int) or bound <= 0 or not isinstance(cycles, int):
            continue
        gaps.append(
            {
                "group": str(key),
                "kind": row.get("kind"),
                "ours": cycles,
                "roofline": bound,
                "over_roofline": round(cycles / bound, 3),
                "gap_cycles": cycles - bound,
                "limiter": roofline.get("limiter"),
                "reference": (theirs.get(key) or {}).get("cycles"),
                "efficiency": found.get("efficiency"),
            }
        )
    return sorted(gaps, key=lambda g: -g["gap_cycles"])


def roofline_lines(document: Mapping[str, Any], *, top: int = 15) -> list[str]:
    """The "distance to the machine" block: ours against each group's derived roofline, with what each
    program issued. Numbers only -- the reference is shown beside them for orientation, not as a goal."""
    from merlin.perf import group_efficiency as E

    total = document.get("distance_to_roofline")
    if not total:
        return []
    lines = [
        f"  DISTANCE TO THE MACHINE'S ROOFLINE (derived from the target's own facts and each group's shape; "
        f"the reference arm is another implementation's cycles, not a ceiling): ours {total['ours_cycles']:,} "
        f"vs roofline {total['roofline_cycles']:,} = {total['ratio']}x over {total['groups']} group(s)"
    ]
    for gap in (document.get("roofline_gaps") or [])[:top]:
        reference = f" ref {gap['reference']:,}" if isinstance(gap.get("reference"), int) else ""
        lines.append(
            f"    g{gap['group']:<3} {str(gap.get('kind')):<12} ours {gap['ours']:>11,} "
            f"roofline {gap['roofline']:>11,} ({gap['over_roofline']}x, {gap.get('limiter')}-bound){reference}"
        )
        efficiency = gap.get("efficiency") or {}
        if efficiency.get("computes_issued") is not None or efficiency.get("per_output_tile"):
            lines.append(f"         issued: {E.describe(efficiency)}")
        elif efficiency.get("refusal"):
            lines.append(f"         issued: not counted ({str(efficiency['refusal'])[:160]})")
    return lines


def _route_brief(route: Mapping[str, Any] | None) -> Any:
    if not isinstance(route, Mapping):
        return None
    return {
        key: route[key]
        for key in ("on", "op", "lowering", "shape", "gather", "commands", "call", "why")
        if key in route
    }


def _by_kind(table: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    kinds: dict[str, dict[str, Any]] = {}
    for entry in table:
        row = kinds.setdefault(
            str(entry.get("kind")),
            {"kind": entry.get("kind"), "groups": 0, "cycles": 0, "reference_cycles": 0, "gather_cycles": 0},
        )
        row["groups"] += 1
        row["cycles"] += int(entry.get("cycles") or 0)
        row["gather_cycles"] += int(entry.get("gather_cycles") or 0)
        if isinstance(entry.get("reference_cycles"), int):
            row["reference_cycles"] += int(entry["reference_cycles"])
    for row in kinds.values():
        row["ratio"] = round(row["cycles"] / row["reference_cycles"], 4) if row["reference_cycles"] else None
    return sorted(kinds.values(), key=lambda row: -int(row["cycles"]))


def _by_kind_and_route(table: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The per-kind totals split by WHERE each group ran (the package's kernel, the host core, the
    library): one kind can hold groups on different routes, and a kind total that mixes them hides
    which part of it is the package's own code. Groups are listed when a split holds a few."""
    splits: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in table:
        on = str((entry.get("route") or {}).get("on") or "unrouted")
        row = splits.setdefault(
            (str(entry.get("kind")), on),
            {"kind": entry.get("kind"), "on": on, "groups": [], "cycles": 0, "reference_cycles": 0},
        )
        row["groups"].append(str(entry.get("group")))
        row["cycles"] += int(entry.get("cycles") or 0)
        if isinstance(entry.get("reference_cycles"), int):
            row["reference_cycles"] += int(entry["reference_cycles"])
    rows = []
    for row in splits.values():
        row["delta_cycles"] = row["cycles"] - row["reference_cycles"] if row["reference_cycles"] else None
        rows.append(row)
    return sorted(rows, key=lambda row: -(row["delta_cycles"] or 0))


#: What the instruments cost, stated beside the gaps they would be used on (never a remedy).
TOOLS_BY_COST = (
    "  to time an edit: a board run covers every group and is the only number that counts; the "
    "functional-model grade lands first and says only whether the bytes are correct"
)


def largest_gaps(document: Mapping[str, Any], *, top: int = 10) -> list[str]:
    """The ranked "where the cycles are against the reference" block, read off this measurement's own
    per-group table: kinds by (ours - reference), then the ``top`` groups by the same, each with its op,
    lowering, route and the package code that emitted it. Empty when there is no admissible reference."""
    bar = document.get("distance_to_bar")
    if not bar:
        return []
    lines = [
        "  LARGEST GAPS vs the same-machine VENDOR reference (context only -- another implementation's cycles, "
        "not a target; this measurement's per-group table):"
    ]
    kinds = [row for row in document.get("by_kind") or [] if row.get("reference_cycles")]
    for row in sorted(kinds, key=lambda r: -(int(r["cycles"]) - int(r["reference_cycles"]))):
        delta = int(row["cycles"]) - int(row["reference_cycles"])
        lines.append(
            f"    kind {str(row.get('kind')):<10} ours {int(row['cycles']):>12,} "
            f"ref {int(row['reference_cycles']):>12,} delta {delta:>+12,}"
        )
    splits = [row for row in document.get("by_kind_and_route") or [] if row.get("reference_cycles")]
    if len({row.get("kind") for row in splits}) < len(splits):
        # A kind that runs on more than one route is shown split, so the package's own share of it reads
        # directly instead of being mixed with groups the host or the library ran.
        lines.append("    by kind and where it ran:")
        for row in splits:
            members = row["groups"]
            named = ",".join("g" + g for g in members) if len(members) <= 4 else f"{len(members)} groups"
            lines.append(
                f"      {str(row.get('kind')):<10} on {str(row.get('on')):<9} {named:<16} "
                f"ours {int(row['cycles']):>12,} ref {int(row['reference_cycles']):>12,} "
                f"delta {int(row['delta_cycles']):>+12,}"
            )
    for holder in (document.get("gap_holders") or [])[:top]:
        route = holder.get("route") or {}
        emitted = holder.get("emitted_by") or {}
        code = (
            ", ".join(emitted.get("functions") or [])[:160]
            if emitted.get("functions")
            else (f"same code as g{emitted['same_lowering_as']}" if emitted.get("same_lowering_as") else "-")
        )
        lines.append(
            f"    g{holder['group']:<3} {str(holder.get('op')):<12} "
            f"{route.get('on')}/{route.get('lowering') or route.get('call')} "
            f"ours {int(holder['ours']):>11,} ref {int(holder['reference']):>11,} "
            f"delta {int(holder['delta_cycles']):>+11,} | {code}"
        )
        census = holder.get("instruction_census")
        if census:
            by_kind = sorted((census.get("by_kind") or {}).items(), key=lambda kv: -kv[1])
            counted = ",".join(f"{kind}={count}" for kind, count in by_kind[:6])
            lines.append(f"         issued {census.get('total')} instructions: {counted}")
    lines.append(TOOLS_BY_COST)
    return lines


def render(document: Mapping[str, Any], *, rows: int = 15) -> str:
    """The same feedback as a short text table, for a prompt."""
    lines = [
        f"whole-model measurement {str(document.get('package_sha256'))[:12]} on {document.get('machine')} "
        f"({document.get('device_artifact')}): {document.get('timing_status')}"
    ]
    if document.get("reason"):
        lines.append(f"  no admissible reading: {document['reason']}")
        return "\n".join(lines)
    authored = document.get("package_authored") or {}
    if authored:
        lines.append(
            f"  package-authored: {coverage_text(authored)} of the work (priced by the reference's own cycles)"
            + (f"  [{document['coverage_regression']}]" if document.get("coverage_regression") else "")
        )
    if document.get("roofline_unavailable"):
        lines.append(f"  no derived roofline in this feedback: {document['roofline_unavailable']}")
    failing = document.get("failing_groups") or []
    if failing:
        # CORRECTNESS LEADS. A wrong program's cycles are where its time went, not a result; the groups
        # that make it wrong are the first thing to fix, each named with what is wrong and whose code.
        # A group that only inherited different inputs is folded into one line after the originators.
        lines.append(f"  {len(failing)} FAILING GROUP(S) -- this run is INVALID until every one is fixed:")
        origin = [entry for entry in failing if not entry.get("inherited_from")]
        inherited = [entry for entry in failing if entry.get("inherited_from")]
        lines.extend(f"    {failing_line(entry)}" for entry in origin[:40])
        if inherited:
            lines.append(
                f"    and {len(inherited)} downstream group(s) that only inherited different inputs: "
                + ", ".join("g" + str(entry["group"]) for entry in inherited[:60])
            )
    else:
        # CORRECT: the cycles are the work, so the ranked gaps lead -- the machine's roofline first.
        lines.extend(roofline_lines(document))
        lines.extend(largest_gaps(document))
    correctness = document.get("correctness") or {}
    lines.append(
        f"  whole-window {document.get('whole_window_cycles'):,} cycles"
        + (f" (package-authored {coverage_text(authored)})" if authored else "")
        + "; correctness "
        f"{correctness.get('status')} (failed groups {correctness.get('groups_failed')}, argmax "
        f"{(correctness.get('argmax') or {}).get('observed')} vs oracle "
        f"{(correctness.get('argmax') or {}).get('oracle')}); "
        f"host gather {document.get('host_gather_cycles'):,} cycles ({document.get('host_gather_share')})"
    )
    kinds = document.get("by_kind") or []
    if kinds:
        # WHERE THE CYCLES ARE, by op kind, with the HOST share of each: a conv whose time is mostly a
        # host-side gather is a different problem from a conv whose kernel is slow.
        lines.append(
            f"  {'kind':<12} {'groups':>6} {'ours':>12} {'host gather':>12} {'gather%':>7} "
            f"{'reference':>12} {'ratio':>6}"
        )
        for row in kinds:
            cycles = int(row.get("cycles") or 0)
            gather = int(row.get("gather_cycles") or 0)
            lines.append(
                f"  {str(row.get('kind')):<12} {row.get('groups'):>6} {cycles:>12,} {gather:>12,} "
                f"{(100.0 * gather / cycles if cycles else 0):>6.1f}% {int(row.get('reference_cycles') or 0):>12,} "
                f"{row.get('ratio') if row.get('ratio') is not None else '-':>6}"
            )
    bar = document.get("distance_to_bar")
    if bar:
        lines.append(
            f"  vendor reference, same machine (context only, not a target) {bar['bar_whole_window_cycles']:,}; "
            f"ours {bar['gap_cycles']:+,} cycles, ratio {bar['ratio']}"
            + (f"  [{bar['note']}]" if bar.get("note") else "")
        )
        lines.append(f"  {'group':>5} {'kind':<8} {'ours':>11} {'reference':>11} {'delta':>11} {'gather':>10}  route")
        for holder in (document.get("gap_holders") or [])[:rows]:
            route = holder.get("route") or {}
            lines.append(
                f"  {holder['group']:>5} {str(holder['kind']):<8} {holder['ours']:>11,} {holder['reference']:>11,} "
                f"{holder['delta_cycles']:>+11,} {int(holder.get('gather_cycles') or 0):>10,}  "
                f"{route.get('on')}/{route.get('lowering') or route.get('call')}"
                + ("" if holder.get("correct") else f"  {str(holder.get('state') or 'not verified').upper()}")
            )
    else:
        lines.append(f"  no bar: {(document.get('reference') or {}).get('reason')}")
    return "\n".join(lines)


def _read(path: Any) -> Any:
    import json
    from pathlib import Path

    try:
        return json.loads(Path(str(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render_early(early: Mapping[str, Any] | None) -> str | None:
    """The correctness known before the timing: the capsule screen and the functional model's grade."""
    if not isinstance(early, Mapping):
        return None
    lines: list[str] = []
    screen = early.get("capsule_screen")
    if isinstance(screen, Mapping):
        lines.extend(render_capsule_report(screen))
    local = early.get("whole_model_functional")
    if isinstance(local, Mapping):
        correctness = local.get("correctness") or {}
        machine = (local.get("machine") or {}).get("machine") or (local.get("machine") or {}).get("artifact")
        lines.append(
            f"whole model on the functional model ({machine}): {str(local.get('status')).upper()}"
            + (f" -- {local.get('refusal')}" if local.get("refusal") else "")
            + f"; failed groups {correctness.get('groups_failed')}; argmax "
            f"{(correctness.get('argmax') or {}).get('observed')} vs oracle "
            f"{(correctness.get('argmax') or {}).get('oracle')}"
        )
        verdict = {"groups": local.get("groups") or []}
        lines.extend(
            f"  {failing_line(entry)}"
            for entry in failing_groups(verdict, local.get("routes"), local.get("attribution"))[:40]
        )
        if local.get("status") == "incorrect" and not local.get("attribution"):
            lines.append("  (locating each failing group's code in the package; call again in a minute)")
    return "\n".join(lines) or None


def render_capsule_report(report: Mapping[str, Any]) -> list[str]:
    """A capsule self-check report as lines: pass count, then each failing capsule's mismatch summary."""
    rows = [r for r in report.get("per_capsule") or [] if isinstance(r, Mapping)]
    failed = [r for r in rows if r.get("pass") is not True]
    lines = [
        f"capsule screen: {len(rows) - len(failed)} of {len(rows)} capsule(s) pass"
        + (f" -- {report.get('error')}" if report.get("error") else "")
    ]
    for row in failed[:30]:
        numeric = row.get("numeric") if isinstance(row.get("numeric"), Mapping) else {}
        failure = row.get("failure") if isinstance(row.get("failure"), Mapping) else {}
        if row.get("declined") or failure.get("category"):
            # NOT a numeric failure: the backend declined, or the run stopped before a comparison. Its
            # own words are the finding -- "None mismatches" would read as a numeric verdict it is not.
            declined = row.get("declined") if isinstance(row.get("declined"), Mapping) else {}
            lines.append(
                f"  {row.get('capsule')} {failure.get('category') or 'DECLINED'}: "
                f"{str(declined.get('reason') or failure.get('detail') or '')[:300]}"
            )
            continue
        lines.append(
            f"  {row.get('capsule')} FAILED: {numeric.get('mismatch_count')} mismatches, max_abs "
            f"{numeric.get('max_abs_diff')} ({numeric.get('policy')}), tiers {row.get('tiers')}"
        )
    return lines


__all__ = [
    "SCHEMA",
    "coverage_text",
    "render_stagnation",
    "stagnation",
    "compare",
    "failing_groups",
    "failing_line",
    "largest_gaps",
    "package_authored",
    "reference_admissible",
    "render",
    "render_capsule_report",
    "render_early",
]


def stagnation(
    history: Sequence[tuple[float, Mapping[str, Any]]],
    groups: Sequence[str],
    *,
    since_epoch: float,
    noise: float,
) -> dict[str, Any]:
    """Whether the top gap-holding groups' cycles MOVED during this session: per group, the lowest
    correct count measured before the session began against the lowest measured since.

    ``history`` is ``(finished_epoch, {group: cycles})`` for correct measurements only.  A group moved
    when its best since the session began is below its best before by more than ``noise`` (a fraction).
    DIAGNOSTIC ONLY: which of the largest gaps this session's measurements changed and which they did
    not -- never why, never what to change."""
    before = [counts for epoch, counts in history if epoch < since_epoch]
    after = [counts for epoch, counts in history if epoch >= since_epoch]

    def lowest(rows: Sequence[Mapping[str, Any]], group: str) -> int | None:
        values = [int(r[group]) for r in rows if isinstance(r.get(group), int)]
        return min(values) if values else None

    rows = []
    for group in groups:
        was, now = lowest(before, str(group)), lowest(after, str(group))
        moved = None if was is None or now is None else now < was * (1.0 - noise)
        rows.append({"group": str(group), "before_session": was, "this_session": now, "moved": moved})
    return {
        "since_epoch": since_epoch,
        "measured_this_session": len(after),
        "noise": noise,
        "groups": rows,
        "unmoved": [r["group"] for r in rows if r["moved"] is False],
    }


def render_stagnation(document: Mapping[str, Any] | None) -> str | None:
    """One line per top gap holder: its best count before this session and since, and whether it moved."""
    if not document or not document.get("groups"):
        return None
    lines = [
        f"  THIS SESSION vs BEFORE IT (top gap holders; {document.get('measured_this_session', 0)} correct "
        "measurement(s) this session):"
    ]
    for row in document["groups"]:
        was, now = row.get("before_session"), row.get("this_session")
        state = "not measured this session" if now is None else ("moved" if row.get("moved") else "unchanged")
        lines.append(
            f"    g{row['group']:<4} before {was if was is not None else '-':>12} this session "
            f"{now if now is not None else '-':>12}  {state}"
        )
    return "\n".join(lines)
