"""Compare proved command-stream movement, grouping and resource facts."""

from __future__ import annotations

from merlin.analysis.kernels.mechanism_metrics import (
    _SEPARATE,
    NO_STREAM,
    ROLE_FACTS,
    STREAM_FACTS,
    STREAM_SCOPE,
    _order,
    stream_fact,
)
from merlin.analysis.kernels.mechanism_ports import (
    FALLBACK,
    _consumer_ports,
    _first,
    _operand_gaps,
    _port_match,
    _role_fallback,
    _side_maps,
    _summarizer,
)

STREAM_EFFECTS = STREAM_FACTS + ("prefetch_interleave", "prefetch_compute_ahead")


_STREAM_BASIS = dict(
    role_transfer_width="elements per issued transfer of this movement role (proved stream)",
    operand_buffering="disjoint local windows of this role and whether consecutive loads alternate (proved stream)",
    issue_overlap="fraction of this role's loads with compute issued after them before the next barrier "
    "(proved stream)",
    prefetch_distance="median loads issued between each load of this operand and the first issue reading its "
    "rows (proved stream; a measured distance)",
    prefetch_compute_ahead="fraction of this operand's consumed loads with at least one compute issued between "
    "the load and its first reader (proved stream)",
)


_STREAM_METRIC = dict(
    role_transfer_width="elements_per_transfer",
    operand_buffering="disjoint_windows",
    issue_overlap="fraction",
    prefetch_distance="median_loads_between",
    prefetch_compute_ahead="fraction_with_compute_ahead",
)


def _stream_direction(name, ref, cand, ref_structure=None, cand_structure=None):
    """Comparison fields (see :data:`COMPARISON_SEMANTICS`) of the candidate against the reference for one role."""
    unknown = dict(direction="unknown", needed_change=None)
    if name == "operand_buffering" and ref.get("definition") != cand.get("definition"):
        return unknown
    metric = _STREAM_METRIC[name]
    r, c = ref.get(metric), cand.get(metric)
    if r is None or c is None:
        return unknown
    fields = _order(name, r, c)
    if fields is None:
        return unknown
    fields["comparison"] = f"{metric}: " + fields["comparison"]
    if r != c:
        return fields
    if name == "operand_buffering":
        a, b = ref["alternating"], cand["alternating"]
        if a == b:
            return fields
        if None in (a, b):
            return unknown
        # Same window count: alternation is then the compared quantity (alternating counts as more).
        alternation = _order(name, int(a), int(b))
        alternation["comparison"] = (
            f"disjoint_windows: reference {r}, candidate {c}, same; alternating: reference "
            f"{'yes' if a else 'no'}, candidate {'yes' if b else 'no'}"
        )
        return alternation
    if name == "role_transfer_width" and ref["shapes"] != cand["shapes"]:
        # Same mean width, different shape composition: a difference with no side.
        return dict(
            fields,
            direction="candidate_differs",
            beyond_reference=None,
            comparison=fields["comparison"] + "; shape composition differs",
        )
    if name == "issue_overlap" and ref_structure != cand_structure:
        return dict(
            fields,
            direction="candidate_differs",
            beyond_reference=None,
            comparison=fields["comparison"] + "; run structure differs",
        )
    return fields


def _stream_rows(reference, candidate, ref_id, cand_id):
    rows = []
    gaps = (_operand_gaps(reference), _operand_gaps(candidate))
    for name in ROLE_FACTS:
        ref_fact, cand_fact = (stream_fact(r.get("derived_effects"), name) for r in (reference, candidate))
        pointer = [f"{i}#derived_effects.facts.{name}" for i in (ref_id, cand_id)]
        effects = (name, "prefetch_compute_ahead") if name == "prefetch_distance" else (name,)
        if "unknown" in (ref_fact["status"], cand_fact["status"]):
            sides = [s for s, f in (("reference", ref_fact), ("candidate", cand_fact)) if f["status"] == "unknown"]
            reason = (
                (ref_fact.get("reason") or cand_fact.get("reason") or NO_STREAM)
                + " on the "
                + " and ".join(sides)
                + " side"
            )
            known = [(s, f) for s, f in (("reference", ref_fact), ("candidate", cand_fact)) if f["status"] != "unknown"]
            for effect in effects:
                rows.append(
                    dict(
                        effect=effect,
                        role=None,
                        status="unknown",
                        direction="unknown",
                        needed_change=None,
                        reason=reason,
                        scope=STREAM_SCOPE,
                        evidence=pointer,
                    )
                )
                if len(known) != 1:
                    continue
                # The established side still describes itself; nothing is compared.
                side, fact = known[0]
                mapping, _, _ = _side_maps(name, fact)
                ports = _consumer_ports(name, fact.get("value"))
                for key in sorted(mapping):
                    rows.append(
                        dict(
                            effect=effect,
                            role=key,
                            status="unknown",
                            direction="unknown",
                            needed_change=None,
                            reason=reason,
                            scope=STREAM_SCOPE,
                            evidence=pointer,
                            basis=_STREAM_BASIS[effect],
                            **{side: _summarizer(name)(mapping[key])},
                            **({"consumer_port": ports[key]} if ports.get(key) else {}),
                        )
                    )
            continue
        (ref_map, ref_sep, ref_structure), (cand_map, cand_sep, cand_structure) = (
            _side_maps(name, f) for f in (ref_fact, cand_fact)
        )
        summarize, extra = _summarizer(name), {}
        structures = (ref_structure, cand_structure) if name == "issue_overlap" else (None, None)
        if name == "issue_overlap":
            extra = dict(reference_structure=ref_structure, candidate_structure=cand_structure)
        if name == "role_transfer_width" and (ref_sep is not None or cand_sep is not None):
            extra = dict(
                preload=dict(
                    reference=ref_sep, candidate=cand_sep, basis="kept separate from data-movement roles; not compared"
                )
            )
        if name == "role_transfer_width":
            extra["route_effect"] = "operand_transfer_extent"
        ref_map, cand_map, matched, registers = _port_match(name, ref_fact, cand_fact, ref_map, cand_map, gaps)
        labels, added, notes = _role_fallback(name, ref_fact, cand_fact, ref_map, cand_map, summarize, registers)

        def compare(effect, role, ref, cand, absent):
            row = dict(
                effect=effect,
                role=role,
                status="observed",
                reference=ref,
                candidate=cand,
                basis=_STREAM_BASIS[effect],
                scope=STREAM_SCOPE,
                evidence=pointer,
                **extra,
            )
            if ref is None or cand is None:
                missing = [s for s, v in (("reference", ref), ("candidate", cand)) if v is None]
                row.update(
                    status="unknown",
                    direction="unknown",
                    needed_change=None,
                    reason=f"role {role} {absent} on the " + " and ".join(missing) + " side",
                )
                return row
            # Run structure belongs to the pooled row; a per-operand row compares that operand alone.
            pooled = structures if role == "all" else (None, None)
            row.update(_stream_direction(effect, ref, cand, *pooled))
            if row["direction"] == "unknown":
                why = [v["reason"] for v in (ref, cand) if isinstance(v, dict) and v.get("reason")]
                if effect == "operand_buffering" and ref.get("definition") != cand.get("definition"):
                    why = [
                        f"the sides counted windows under different definitions (reference: "
                        f"{ref.get('definition')}; candidate: {cand.get('definition')}); re-derive the older "
                        "record"
                    ]
                row.update(
                    status="unknown", reason="; ".join(why) or _STREAM_METRIC[effect] + " not established on both sides"
                )
            return row

        def attributed(attribution):
            return dict(
                attribution=FALLBACK,
                operand_attribution=attribution,
                attribution_basis="the sides' per-operand entries do not all match (by consuming compute "
                "port, or by argument register where neither side derived a port); "
                "compared over all of the role's loads; argument registers are labels",
            )

        def label(row, key):
            if key in matched:
                found = matched[key]
                row.update({k: v for k, v in found.items() if k != "reason"})
                if row["status"] == "unknown" and found.get("reason"):
                    row["reason"] = found["reason"] + (
                        f"; compared at role level in role {notes[key][0]}" if key in notes else ""
                    )
            elif key in labels:
                row.update(attributed(labels[key]))
            elif key in notes:
                role_key, attribution = notes[key]
                row.update(
                    attribution="per_operand",
                    operand_attribution=attribution,
                    reason=f"operand attribution differs between the sides (reference: "
                    f"{', '.join(attribution['reference'])}; candidate: "
                    f"{', '.join(attribution['candidate'])}); compared at role level in role "
                    f"{role_key}",
                )
            return row

        for effect in effects:
            for role in sorted((set(ref_map) | set(cand_map)) - set(added)):
                row = compare(
                    effect,
                    role,
                    *(summarize(m[role]) if role in m else None for m in (ref_map, cand_map)),
                    "absent or unreadable",
                )
                rows.append(label(row, role))
            for role, (attribution, ref, cand) in sorted(added.items()):
                rows.append(
                    dict(
                        compare(effect, role, ref, cand, "per-operand entries do not combine"),
                        **attributed(attribution),
                    )
                )
        if name == "issue_overlap":
            rows += _interleave_rows(ref_fact.get("value") or {}, cand_fact.get("value") or {}, pointer)
    rows += _output_group_rows(reference, candidate, ref_id, cand_id)
    return rows + _extra_rows(reference, candidate, ref_id, cand_id)


def _scalar(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        if type(value.get("rows")) is int and type(value.get("cols")) is int and "elements" not in value:
            return value["rows"] * value["cols"]
        found = _first(value, "elements", "median", "value")
        return found if isinstance(found, (int, float)) and not isinstance(found, bool) else None
    return None


def _scalar_map(value):
    """``{role: number}`` from a scalar, an extent record or a per-role mapping (role ``all`` when pooled)."""
    if _scalar(value) is not None:
        return {"all": _scalar(value)}
    if isinstance(value, dict):
        inner = value.get("by_role") if isinstance(value.get("by_role"), dict) else value
        return {str(k): _scalar(v) for k, v in inner.items() if _scalar(v) is not None and k not in _SEPARATE}
    return {}


_INTERLEAVE = (
    (
        "prefetch_interleave",
        (
            "compute_run_between_load_bands",
            "median_compute_run_between_load_bands",
            "compute_run_median",
            "median_compute_run",
        ),
        "median compute run between load bands; loads_per_compute_run is listed beside it (proved stream)",
    ),
)


_GROUP_BASIS = (
    "median output columns completed per barrier-delimited block: distinct column start positions "
    "of the block's readouts (modulo the configured readout row pitch) x their column extent "
    "(proved stream)"
)


def _group_columns(fact, overlap):
    """``(columns, reason)`` of one side: its ``output_group_extent`` fact, else an extent an older
    ``issue_overlap`` value carried; ``None`` with the fact's reason when neither establishes it."""
    value = fact.get("value") if fact.get("status") not in (None, "unknown") else None
    if isinstance(value, dict):
        if isinstance(value.get("median_output_columns"), (int, float)):
            return value["median_output_columns"], None
        return None, (
            "output_group_extent does not establish output columns per block (readout row pitch or a "
            "uniform column extent not derived, or the record predates output_columns)"
        )
    legacy = _scalar((overlap or {}).get("output_group_extent")) if isinstance(overlap, dict) else None
    if legacy is not None:
        return legacy, None
    return None, fact.get("reason") or NO_STREAM


def _output_group_rows(reference, candidate, ref_id, cand_id):
    """One row comparing output columns per block, read from each side's own ``output_group_extent`` fact."""
    facts = [stream_fact(r.get("derived_effects"), "output_group_extent") for r in (reference, candidate)]
    overlaps = [(stream_fact(r.get("derived_effects"), "issue_overlap").get("value")) for r in (reference, candidate)]
    (ref, ref_reason), (cand, cand_reason) = (_group_columns(f, o) for f, o in zip(facts, overlaps, strict=True))
    pointer = [f"{i}#derived_effects.facts.output_group_extent" for i in (ref_id, cand_id)]
    row = dict(
        effect="output_group_extent",
        role="all",
        status="observed",
        reference=ref,
        candidate=cand,
        basis=_GROUP_BASIS,
        scope=STREAM_SCOPE,
        evidence=pointer,
        needed_change=None,
    )
    if ref is None or cand is None:
        reasons = [
            f"{side}: {why}"
            for side, value, why in (("reference", ref, ref_reason), ("candidate", cand, cand_reason))
            if value is None
        ]
        row.update(status="unknown", direction="unknown", reason="; ".join(reasons))
    else:
        row.update(_order("output_group_extent", ref, cand))
    return [row]


def _interleave_rows(ref_value, cand_value, pointer):
    rows = []
    for effect, keys, basis in _INTERLEAVE:
        ref, cand = (_scalar_map(_first(v, *keys)) for v in (ref_value, cand_value))
        if not ref and not cand:
            rows.append(
                dict(
                    effect=effect,
                    role=None,
                    status="unknown",
                    direction="unknown",
                    needed_change=None,
                    reason="not reported by the proved stream on either side",
                    scope=STREAM_SCOPE,
                    evidence=pointer,
                )
            )
            continue
        for role in sorted(set(ref) | set(cand)):
            r, c = ref.get(role), cand.get(role)
            row = dict(
                effect=effect,
                role=role,
                status="observed",
                reference=r,
                candidate=c,
                basis=basis,
                scope=STREAM_SCOPE,
                evidence=pointer,
                needed_change=None,
            )
            if effect == "prefetch_interleave":
                row.update(
                    reference_loads_per_compute_run=ref_value.get("loads_per_compute_run"),
                    candidate_loads_per_compute_run=cand_value.get("loads_per_compute_run"),
                )
            if r is None or c is None:
                row.update(status="unknown", direction="unknown", reason=f"role {role} absent on one side")
            else:
                row.update(_order(effect, r, c))
            rows.append(row)
    return rows


def _number(entry, *keys):
    value = _first(entry, *keys) if isinstance(entry, dict) else entry
    if isinstance(value, bool):
        return int(value)
    return value if isinstance(value, (int, float)) else None


def _capacity_fraction(entry):
    if not isinstance(entry, dict):
        return None
    found = _number(entry, "fraction", "use_fraction")
    if found is not None:
        return found
    used, capacity = _number(entry, "rows", "used_rows"), _number(entry, "capacity", "capacity_rows")
    return round(used / capacity, 6) if used is not None and capacity else None


def _in_loop(entry):
    if isinstance(entry, dict) and isinstance(entry.get("placement"), str):
        return {"in_loop": 1, "hoisted": 0}.get(entry["placement"])
    return _number(entry, "in_loop")


def _on_accelerator(entry):
    if isinstance(entry, dict) and isinstance(entry.get("placement"), str):
        return {"on_accelerator": 1, "accelerator": 1, "host": 0, "off_accelerator": 0}.get(entry["placement"])
    return _number(entry, "on_accelerator")


EXTRA_FACTS = (
    (
        "host_issue_rate",
        "host_issue_rate",
        lambda e: _number(e, "rate", "issue_rate", "value", "median"),
        "accelerator commands issued per host instruction (proved stream)",
    ),
    (
        "reconfiguration",
        "reconfiguration",
        lambda e: _number(e, "config_per_block", "per_block", "value"),
        "configuration commands per block (proved stream)",
    ),
    (
        "reconfiguration",
        "reconfiguration_in_loop",
        _in_loop,
        "configuration issued inside the loop (1) or hoisted (0) (proved stream)",
    ),
    (
        "local_capacity_use",
        "local_capacity_use",
        _capacity_fraction,
        "local rows used over the declared capacity (proved stream)",
    ),
    (
        "local_capacity_use",
        "same_bank_halves",
        lambda e: _number(e, "same_bank_halves", "same_bank"),
        "alternating halves placed in the same bank (1) or not (0) (proved stream)",
    ),
    (
        "dram_contiguity",
        "dram_contiguity",
        lambda e: _number(e, "contiguous_fraction", "contiguous", "fraction"),
        "fraction of transfer rows contiguous in memory (proved stream)",
    ),
    (
        "epilogue_placement",
        "epilogue_placement",
        _on_accelerator,
        "output epilogue on the accelerator (1) or not (0) (proved stream)",
    ),
    (
        "resource_occupancy",
        "queue_occupancy",
        lambda e: _number(e, "max_outstanding_before_dependency"),
        "largest count of commands routed to one dispatch queue before a sync or dependent consumer (proved stream)",
    ),
    (
        "resource_occupancy",
        "queue_points_past_entries",
        lambda e: _number(e, "points_exceeding_dispatch_entries"),
        "issue points past the queue's RTL dispatch entries (proved stream)",
    ),
    (
        "transfer_legality",
        "transfer_exceeds_bound",
        lambda e: _number(e, "exceeds_bound"),
        "transfers past an RTL transfer bound (proved stream)",
    ),
    (
        "post_store_reuse",
        "post_store_reuse",
        lambda e: (
            _number(e, "read_again") + _number(e, "accumulated_into")
            if _number(e, "read_again") is not None and _number(e, "accumulated_into") is not None
            else None
        ),
        "stored local rows read or accumulated into again before a rewrite (proved stream)",
    ),
)


STREAM_EFFECTS = STREAM_EFFECTS + tuple(effect for _, effect, *_ in EXTRA_FACTS)


def _per_role(value, metric):
    if not isinstance(value, dict):
        return {}
    inner = next(
        (value[k] for k in ("by_role", "roles", "per_role", "per_queue") if isinstance(value.get(k), dict)), None
    )
    if inner is None:
        pooled = metric(value)
        return {"all": pooled} if pooled is not None else {}
    return {str(k): metric(v) for k, v in inner.items() if k not in _SEPARATE and metric(v) is not None}


def _extra_rows(reference, candidate, ref_id, cand_id):
    """Rows only for facts the stream reports on either side; an older stepper adds no rows."""
    rows = []
    for name, effect, metric, *_, basis in EXTRA_FACTS:
        ref_fact, cand_fact = (
            ((r.get("derived_effects") or {}).get("facts") or {}).get(name) for r in (reference, candidate)
        )
        if ref_fact is None and cand_fact is None:
            continue
        pointer = [f"{i}#derived_effects.facts.{name}" for i in (ref_id, cand_id)]
        known = [f for f in (ref_fact, cand_fact) if isinstance(f, dict) and f.get("status") not in (None, "unknown")]
        if len(known) < 2:
            rows.append(
                dict(
                    effect=effect,
                    role=None,
                    status="unknown",
                    direction="unknown",
                    needed_change=None,
                    reason=next(
                        (f.get("reason") for f in (ref_fact, cand_fact) if isinstance(f, dict) and f.get("reason")),
                        NO_STREAM,
                    ),
                    scope=STREAM_SCOPE,
                    evidence=pointer,
                )
            )
            continue
        ref, cand = (_per_role(f.get("value"), metric) for f in (ref_fact, cand_fact))
        if not ref and not cand:
            continue  # this fact reports no value for this metric on either side
        for role in sorted(set(ref) | set(cand)):
            r, c = ref.get(role), cand.get(role)
            row = dict(
                effect=effect,
                fact=name,
                role=role,
                status="observed",
                reference=r,
                candidate=c,
                basis=basis,
                scope=STREAM_SCOPE,
                evidence=pointer,
                needed_change=None,
            )
            if r is None or c is None:
                row.update(status="unknown", direction="unknown", reason=f"role {role} absent on one side")
            else:
                row.update(
                    _order(effect, r, c)
                    or dict(status="unknown", direction="unknown", reason="values are not comparable numbers")
                )
            rows.append(row)
    return rows


# Preserve the original public lookup and pickle identities.
for _definition in (
    _stream_direction,
    _stream_rows,
    _scalar,
    _scalar_map,
    _group_columns,
    _output_group_rows,
    _interleave_rows,
    _number,
    _capacity_fraction,
    _in_loop,
    _on_accelerator,
    _per_role,
    _extra_rows,
):
    _definition.__module__ = "merlin.analysis.kernels.mechanism_applicability"
del _definition
