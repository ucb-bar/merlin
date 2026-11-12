"""Match proved movement roles to original consumer ports."""

from __future__ import annotations

from merlin.analysis.kernels.mechanism_metrics import _SEPARATE, stream_fact

PORT, REGISTER = "consumer_port", "argument_register"


def _first(entry, *keys):
    return next((entry[k] for k in keys if k in entry), None)


def _role_entries(value):
    """``({role: entry}, separate)``: data-movement roles, and the preload entry kept apart."""
    if not isinstance(value, dict):
        return {}, None
    if isinstance(value.get("data_movement"), dict):
        # Movement role, then each tensor argument it addresses (``role:operand``); local staging apart.
        out = {}
        for role, entry in value["data_movement"].items():
            out[str(role)] = entry
            for operand, sub in ((entry or {}).get("by_operand") or {}).items():
                out[f"{role}:{operand}"] = sub
        return out, value.get("local_staging")
    if value and all(
        isinstance(v, dict) and v and all(isinstance(w, dict) and "disjoint_windows" in w for w in v.values())
        for v in value.values()
    ):
        return {f"{role}:{operand}": w for role, v in value.items() for operand, w in v.items()}, None
    inner = next((value[k] for k in ("by_role", "roles", "per_role") if isinstance(value.get(k), dict)), value)
    separate = next(
        (_first(value, k) if k in value else inner.get(k) for k in _SEPARATE if k in value or k in inner), None
    )
    return ({str(k): v for k, v in inner.items() if isinstance(v, (dict, list)) and k not in _SEPARATE}, separate)


def _width(entry):
    rows = entry if isinstance(entry, list) else _first(entry, "shapes", "values", "distinct", "widths")
    if not isinstance(rows, list) or not rows:
        return None
    table = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        r, c = row.get("rows"), row.get("cols")
        n = _first(row, "count", "transfers", "sites")
        e = _first(row, "elements", "elements_per_transfer")
        if type(e) is not int:
            e = r * c if type(r) is int and type(c) is int else None
        if type(n) is not int or e is None:
            return None
        table[(r, c, e)] = table.get((r, c, e), 0) + n
    transfers = sum(table.values())
    elements = sum(e * n for (_, _, e), n in table.items())
    return dict(
        shapes=[dict(rows=r, cols=c, elements=e, count=n) for (r, c, e), n in sorted(table.items())],
        transfers=transfers,
        elements=elements,
        elements_per_transfer=round(elements / transfers, 6) if transfers else None,
    )


def _buffering(entry):
    if not isinstance(entry, dict):
        return None
    disjoint = _first(entry, "disjoint", "disjoint_windows", "disjoint_count")
    alternating = _first(entry, "alternating", "successive_phases_alternate", "alternates", "alternation")
    if isinstance(alternating, dict):
        alternating = alternating.get("value")
    if type(disjoint) is not int:
        return dict(disjoint_windows=None, alternating=None, reason=entry["reason"]) if entry.get("reason") else None
    # The definition the windows were counted under; an entry without one predates it.
    return dict(
        disjoint_windows=disjoint,
        alternating=alternating if isinstance(alternating, bool) else None,
        windows=entry.get("windows"),
        definition=entry.get("definition") or "phase_windows_v1",
    )


def _distance_value(entry):
    """Load-to-first-reader distance summary of one operand (or the pooled ``all`` entry)."""
    if not isinstance(entry, dict) or not isinstance(entry.get("median_loads_between"), (int, float)):
        return None
    return {
        k: entry.get(k)
        for k in (
            "median_loads_between",
            "min_loads_between",
            "fraction_with_compute_ahead",
            "loads",
            "consumed_loads",
            "unconsumed_loads",
        )
    }


def _distance(value):
    """``{operand or all: distance summary}`` of a ``prefetch_distance`` value."""
    if not isinstance(value, dict):
        return {}
    out = {str(k): _distance_value(v) for k, v in (value.get("by_operand") or {}).items()}
    out["all"] = _distance_value(value.get("all"))
    return {k: v for k, v in out.items() if v is not None}


def _overlap_value(item, total=None):
    if isinstance(item, dict):
        fraction = _first(item, "fraction", "frac", "loads_with_compute_before_barrier")
        count = _first(item, "count", "value")
        total = _first(item, "total", "loads") or total
    elif isinstance(item, float):
        fraction, count = item, None
    else:
        fraction, count = None, item
    if fraction is None and type(count) is int and type(total) is int and total:
        fraction = count / total
    if not isinstance(fraction, (int, float)):
        return None
    return dict(
        count=count if type(count) is int else None,
        total=total if type(total) is int else None,
        fraction=round(float(fraction), 6),
    )


def _overlap(value):
    """``({role: overlap}, run structure)``; role ``all`` when the stream reports one pooled value."""
    if not isinstance(value, dict):
        return {}, {}
    structure = {
        k: value[k]
        for k in (
            "compute_run_median",
            "load_run_median",
            "median_compute_run",
            "median_load_run",
            "fences",
            "fence_count",
            "computes_per_fence",
        )
        if k in value
    }
    loads = value.get("loads_with_compute_before_barrier")
    if isinstance(value.get("by_operand"), dict):
        out = {str(k): _overlap_value(v) for k, v in value["by_operand"].items() if v}
        out["all"] = _overlap_value(loads, value.get("loads"))
    elif isinstance(loads, dict) and not ({"count", "fraction", "frac", "total"} & set(loads)):
        if isinstance(loads.get("by_role"), dict):
            loads = loads["by_role"]
        out = {str(role): _overlap_value(item) for role, item in loads.items() if role not in _SEPARATE}
    else:
        out = {"all": _overlap_value(loads, _first(value, "loads", "load_count", "loads_total"))}
    return {k: v for k, v in out.items() if v is not None}, structure


FALLBACK = "role_level_fallback"


def _operand_groups(name, value, mapping):
    """``{role key: {operand: row key}}`` of one side's per-operand entries.

    Width and buffering keys are ``role:operand``; the role-level key is the bare role. Overlap's
    per-operand keys are the operands themselves, pooled under ``all``."""
    groups = {}
    if name in ("issue_overlap", "prefetch_distance"):
        if isinstance(value, dict) and isinstance(value.get("by_operand"), dict):
            groups["all"] = {str(k): str(k) for k in mapping if str(k) != "all"}
        return {k: v for k, v in groups.items() if v}
    for key in mapping:
        role, sep, operand = key.partition(":")
        if sep and operand != "all":
            groups.setdefault(role, {})[operand] = key
    return groups


def _aggregate(name, entries):
    """One role-level entry from summarized per-operand entries, or ``None`` when they do not combine."""
    if not entries or any(e is None for e in entries):
        return None
    if name == "role_transfer_width":
        return _width([shape for e in entries for shape in e["shapes"]])
    if name == "prefetch_distance":
        return None  # medians of separate operands do not combine
    if name == "issue_overlap":
        totals = [e.get("total") for e in entries]
        if any(type(t) is not int for t in totals) or not sum(totals):
            return None
        hits = sum(e["fraction"] * t for e, t in zip(entries, totals, strict=True))
        return dict(count=None, total=sum(totals), fraction=round(hits / sum(totals), 6))
    # Buffering: the union of every operand's listed windows; alternation is per operand, so only one
    # operand's flag carries over.
    windows = []
    for entry in entries:
        listed = entry.get("windows")
        if not isinstance(listed, list) or len(listed) != entry["disjoint_windows"]:
            return None
        windows += [(w["lo"], w["hi"] + 1) for w in listed if type(w.get("lo")) is int and type(w.get("hi")) is int]
    merged = []
    for lo, hi in sorted(windows):
        if merged and lo < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    definitions = {entry.get("definition") for entry in entries}
    if len(definitions) != 1:
        return None
    return dict(
        disjoint_windows=len(merged),
        alternating=entries[0]["alternating"] if len(entries) == 1 else None,
        windows=[dict(lo=lo, hi=hi - 1) for lo, hi in merged],
        definition=definitions.pop(),
    )


def _role_fallback(name, ref_fact, cand_fact, ref_map, cand_map, summarize, registers=({}, {})):
    """Role-level comparisons where the two sides' per-operand entries do not all match.

    Returns ``(labels, added, notes)``: role-level keys both sides already report (labelled), role-level
    entries aggregated from the per-operand entries when no such key exists, and the per-operand keys
    whose rows stay unknown because they have no counterpart. ``registers`` maps a re-keyed entry back to
    each side's argument registers, which label the attribution."""
    ref_groups, cand_groups = (
        _operand_groups(name, f.get("value"), m) for f, m in ((ref_fact, ref_map), (cand_fact, cand_map))
    )
    labels, added, notes = {}, {}, {}
    for role in sorted(set(ref_groups) & set(cand_groups)):
        ref_ops, cand_ops = ref_groups[role], cand_groups[role]
        if set(ref_ops) == set(cand_ops):
            continue
        attribution = {
            side: sorted(r for operand, key in ops.items() for r in names.get(key, [operand]))
            for side, ops, names in (("reference", ref_ops, registers[0]), ("candidate", cand_ops, registers[1]))
        }
        key = (
            role
            if name in ("issue_overlap", "prefetch_distance") or (role in ref_map and role in cand_map)
            else f"{role}:all"
        )
        if key in ref_map and key in cand_map:
            labels[key] = attribution
        else:
            added[key] = (
                attribution,
                *(
                    _aggregate(name, [summarize(m[k]) for k in groups.values()])
                    for m, groups in ((ref_map, ref_ops), (cand_map, cand_ops))
                ),
            )
        for operand in set(ref_ops) ^ set(cand_ops):
            notes[ref_ops.get(operand) or cand_ops[operand]] = (key, attribution)
    return labels, added, notes


def _consumer_ports(name, value):
    """``{per-operand row key: consuming compute port}`` reported by one side's stream fact."""
    ports = {}
    if not isinstance(value, dict):
        return ports
    if name in ("issue_overlap", "prefetch_distance"):
        for operand, entry in (value.get("by_operand") or {}).items():
            if isinstance(entry, dict):
                ports[str(operand)] = entry.get("consumer_port")
    elif name == "role_transfer_width":
        for role, entry in (value.get("data_movement") or {}).items():
            for operand, sub in ((entry or {}).get("by_operand") or {}).items():
                ports[f"{role}:{operand}"] = (sub or {}).get("consumer_port")
    elif name == "operand_buffering":
        for role, entry in value.items():
            for operand, sub in (entry if isinstance(entry, dict) else {}).items():
                if isinstance(sub, dict):
                    ports[f"{role}:{operand}"] = sub.get("consumer_port")
    return ports


def _operand_gaps(record):
    """``{operand: why}`` from one side's buffering entries, which say why a load's rows stay unresolved."""
    value = stream_fact(record.get("derived_effects"), "operand_buffering").get("value")
    gaps = {}
    for entries in value.values() if isinstance(value, dict) else ():
        for operand, entry in entries.items() if isinstance(entries, dict) else ():
            if isinstance(entry, dict) and isinstance(entry.get("reason"), str):
                gaps.setdefault(str(operand), entry["reason"])
    return gaps


_SIDES = ("reference", "candidate")


def _port_match(name, ref_fact, cand_fact, ref_map, cand_map, gaps=({}, {})):
    """Key both sides' per-operand entries by the compute port that reads the loaded rows.

    Every entry whose consuming port is derived, and not shared with another operand of its side, is
    re-keyed ``role:port`` (``port`` for the pooled-overlap keys); its argument register is kept only as a
    label. An entry without such a port keeps its register key: it is compared by register only when the
    other side's entry of that register has no usable port either (older records, shared ports), and
    otherwise stays unknown with the reason. Returns ``(ref_map, cand_map, info, registers)``: ``info``
    is each row key's attribution, basis and (for a row that cannot be compared) reason; ``registers``
    maps each side's re-keyed key back to its argument registers."""
    ports = [_consumer_ports(name, f.get("value")) for f in (ref_fact, cand_fact)]
    maps = [dict(ref_map), dict(cand_map)]
    groups = [{}, {}]
    for side, mapping in enumerate(maps):
        for key in mapping:
            if key in ports[side]:
                role, sep, _ = key.partition(":")
                groups[side].setdefault(role if sep else "all", []).append(key)
    info, registers = {}, ({}, {})

    def operand(group, key):
        return key.partition(":")[2] if group != "all" else key

    for group in sorted(set(groups[0]) | set(groups[1])):
        named = [{key: ports[side][key] for key in groups[side].get(group, [])} for side in (0, 1)]
        kept = [{}, {}]  # register key -> why its port is not usable on that side
        for side in (0, 1):
            counts = {}
            for port in named[side].values():
                counts[port] = counts.get(port, 0) + 1
            for key, port in named[side].items():
                if not port:
                    gap = gaps[side].get(operand(group, key))
                    kept[side][key] = (
                        f"no consuming compute port is derived for {operand(group, key)} on the "
                        f"{_SIDES[side]} side" + (f" ({gap})" if gap else "")
                    )
                elif counts[port] > 1:
                    kept[side][key] = f"two operands share one consuming port ({port}) on the {_SIDES[side]} side"
        by_port = [{}, {}]
        for side in (0, 1):
            for key, port in named[side].items():
                if key in kept[side]:
                    continue
                new = f"{group}:{port}" if group != "all" else port
                maps[side][new] = maps[side].pop(key)
                registers[side][new] = [operand(group, key)]
                by_port[side][port] = new
        for port in sorted(set(by_port[0]) | set(by_port[1])):
            new = by_port[0].get(port) or by_port[1][port]
            row = dict(
                attribution=PORT,
                consumer_port=port,
                operand_attribution={_SIDES[s]: registers[s].get(new, []) for s in (0, 1)},
                attribution_basis="matched by the compute port that reads the loaded rows; argument "
                "registers are labels only",
            )
            absent = [s for s in (0, 1) if port not in by_port[s]]
            if absent:
                other = absent[0]
                row["reason"] = (
                    f"no {_SIDES[other]} load of this role is attributed to port {port}: "
                    + "; ".join(kept[other].values())
                    if kept[other]
                    else f"no {_SIDES[other]} load of this role is read by port {port}"
                )
            info[new] = row
        for side in (0, 1):
            other = 1 - side
            for key, why in kept[side].items():
                counterpart = key in kept[other]
                row = dict(
                    attribution=REGISTER,
                    operand_attribution={_SIDES[side]: [operand(group, key)]},
                    attribution_basis=("matched by argument register: " if counterpart else "argument register only: ")
                    + why,
                )
                if counterpart:
                    row["operand_attribution"][_SIDES[other]] = [operand(group, key)]
                    row["attribution_basis"] += "; " + kept[other][key] if kept[other][key] != why else ""
                else:
                    row["reason"] = why + "; not matched to any " + _SIDES[other] + " operand"
                info.setdefault(key, row)
    return maps[0], maps[1], info, registers


def _summarizer(name):
    return {"role_transfer_width": _width, "operand_buffering": _buffering}.get(name, lambda item: item)


def _side_maps(name, fact):
    """``(per-role/per-operand entries, separate preload entry, run structure)`` of one side's fact."""
    if name == "issue_overlap":
        mapping, structure = _overlap(fact.get("value"))
        return mapping, None, structure
    if name == "prefetch_distance":
        return _distance(fact.get("value")), None, None
    mapping, separate = _role_entries(fact.get("value"))
    return mapping, separate, None


# Preserve the original public lookup and pickle identities.
for _definition in (
    _first,
    _role_entries,
    _width,
    _buffering,
    _distance_value,
    _distance,
    _overlap_value,
    _overlap,
    _operand_groups,
    _aggregate,
    _role_fallback,
    _consumer_ports,
    _operand_gaps,
    _port_match,
    _summarizer,
    _side_maps,
):
    _definition.__module__ = "merlin.analysis.kernels.mechanism_applicability"
del _definition
