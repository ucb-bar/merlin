"""Mechanism differences between a reference occurrence and a candidate, with applicable library routes.

Inputs are two static occurrence records (``analysis.json``-shaped: ``schedule_model.role_counts``,
``cfg`` and ``derived_effects`` from :mod:`merlin.kernels.decoded_effects`), the scheduling
catalog's declared effect table (``api_catalog.CAPABILITY_EFFECTS``), the selected RTL facts and the
candidate's own source shape. Every difference carries an evidence pointer and a status:
``observed`` (decoded static sites or fields), ``inferred`` (a static site ratio read as a per-compute
rate inside loops whose trip counts are unknown) or ``unknown``.

For each difference the routes whose declared effect moves the candidate toward the reference are
listed with their library admission predicate evaluated on the candidate's shape and the selected
facts, plus the catalog's own usage and requirement text. A difference no route addresses is reported
as ``missing_capability`` naming that effect. Nothing is ranked or chosen: routes keep catalog order,
admission witnesses are construction conditions, and the author selects and composes any change.
"""

from __future__ import annotations

from fractions import Fraction

SCHEMA = "merlin.mechanism_applicability.v1"
_MAX_N_TILES = 256  # Bounded per-width admission evaluation; larger extents report the bound as unknown.
_DENSE = ("m", "k", "n")
_CONV = ("kh", "kw", "c", "oc")

#: How every compared effect reads, defined once: ``(what a larger value means, more_when)``. ``more_when``
#: names the change of the value that adds more of the mechanism the effect measures (and that the catalog's
#: routes for it declare): ``"increases"`` when a larger value is more of it, ``"reduces"`` when a smaller
#: value is, ``None`` when the effect has no such side. Every row states its numbers in plain words
#: (``comparison``), ``needed_change`` is ALWAYS the change that moves the candidate's value toward the
#: reference's, and ``beyond_reference`` says whether the candidate already has more of the mechanism than
#: the reference (then no route is sought: a route would move it further from the reference).
COMPARISON_SEMANTICS = dict(
    operand_load_per_compute=("more operand-load sites per accumulate site (less operand reuse)", "reduces"),
    weight_load_per_compute=("more weight-load sites per accumulate site (less weight reuse)", "reduces"),
    operand_transfer_extent=("more elements in the largest operand transfer", "increases"),
    broadcast_load_sites=("more movement sites issued under a stride-zero load configuration", "increases"),
    configured_load_states=("more distinct load-state ports given a stride configuration", None),
    config_per_transfer=("more configuration sites per transfer site", "reduces"),
    sync_sites=("more synchronization sites", "reduces"),
    loop_descriptor_sites=("more hardware loop-descriptor sites", None),
    role_transfer_width=("more elements per issued transfer (wider transfers)", "increases"),
    operand_buffering=("more disjoint local windows (then alternation) for this role's loads", "increases"),
    issue_overlap=("a larger fraction of loads with compute issued before the next barrier", "increases"),
    prefetch_distance=("more loads issued between a load and the first issue reading its rows", "increases"),
    prefetch_compute_ahead=("a larger fraction of loads with compute issued before their first reader",
                            "increases"),
    output_group_extent=("more output columns completed per barrier-delimited block", "increases"),
    prefetch_interleave=("longer compute runs between load bands (coarser load/compute interleaving)", "reduces"),
    host_issue_rate=("more accelerator commands per host instruction", "increases"),
    reconfiguration=("more configuration commands per block", "reduces"),
    reconfiguration_in_loop=("configuration issued inside the loop rather than hoisted", "reduces"),
    local_capacity_use=("a larger fraction of the declared local capacity used", "increases"),
    same_bank_halves=("alternating halves placed in the same bank", "reduces"),
    dram_contiguity=("a larger fraction of transfer rows contiguous in memory", "increases"),
    epilogue_placement=("the output epilogue placed on the accelerator", "increases"),
    queue_occupancy=("more commands outstanding on one dispatch queue before a dependency", "increases"),
    queue_points_past_entries=("more issue points past the queue's RTL dispatch entries", "reduces"),
    transfer_exceeds_bound=("more transfers past an RTL transfer bound", "reduces"),
    post_store_reuse=("more stored local rows read or accumulated into again before a rewrite", "increases"),
)


def _plain(value):
    return value if not isinstance(value, float) else round(value, 6)


def _order(effect, reference, candidate):
    """The comparison fields of one row whose two sides are comparable numbers (``None`` when not)."""
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (reference, candidate)):
        return None
    larger, more_when = COMPARISON_SEMANTICS.get(effect, (None, None))
    if reference == candidate:
        word, change = "same", None
    else:
        word, change = (("lower", "increases") if candidate < reference else ("higher", "reduces"))
    fields = dict(direction="candidate_matches" if change is None else "candidate_differs",
                  comparison=f"reference {_plain(reference)}, candidate {_plain(candidate)}, candidate {word}",
                  needed_change=change, beyond_reference=None if change is None or more_when is None
                  else change != more_when)
    if larger is not None:
        fields["larger_means"] = larger
    return fields


def _ratio(numerator, denominator):
    if not denominator:
        return None
    value = Fraction(numerator, denominator)
    return dict(numerator=numerator, denominator=denominator, value=float(value))


def _fact(effects, name):
    return ((effects or {}).get("facts") or {}).get(name) or dict(status="unknown", reason="fact not derived")


def _evidence_id(record):
    return ((record.get("derived_effects") or {}).get("evidence_id")
            or (record.get("evidence") or {}).get("evidence_id") or record.get("stable_id") or "occurrence")


def _loops(record):
    cfg = record.get("cfg") or {}
    return cfg.get("natural_loops")


def _rate_status(reference, candidate):
    """Static site ratios are per-compute rates only for straight-line spans; loops make them inferred."""
    loops = (_loops(reference), _loops(candidate))
    if any(value is None for value in loops):
        return "unknown"
    return "observed" if loops == (0, 0) else "inferred"


def _roles(record):
    return dict((record.get("schedule_model") or {}).get("role_counts") or {})


def _direction(reference, candidate):
    if reference is None or candidate is None or reference == candidate:
        return None
    return "reduces" if candidate > reference else "increases"


def _role_rate(name, role, reference, candidate, ref_id, cand_id):
    ref_roles, cand_roles = _roles(reference), _roles(candidate)
    ref = _ratio(ref_roles.get(role, 0), ref_roles.get("accumulate", 0))
    cand = _ratio(cand_roles.get(role, 0), cand_roles.get("accumulate", 0))
    status = _rate_status(reference, candidate) if ref and cand else "unknown"
    return dict(
        effect=name, status=status,
        reference=ref, candidate=cand,
        needed_change=_direction(ref and ref["value"], cand and cand["value"]) if status != "unknown" else None,
        basis=f"static {role}-role sites per accumulate-role site over the selected role roster",
        limits=["listed instruction sites, not dynamic issue counts; loop trip counts are unknown"]
               if status == "inferred" else [],
        evidence=[f"{ref_id}#schedule_model.role_counts", f"{cand_id}#schedule_model.role_counts",
                  f"{ref_id}#cfg.natural_loops", f"{cand_id}#cfg.natural_loops"])


def _extent(effects, structure):
    """Static transfer shapes of operand_load-role sites, kept per instruction class (one class per operand
    port) when the selected role roster names the classes; pooling every load under one maximum hid a
    narrow operand behind a wide one. Without a roster the pool stays one ``operand_load`` key."""
    fact = _fact(effects, "transfer_tiles")
    value = fact.get("value") or {}
    pooled = (value.get("by_role") or {}).get("operand_load") or []
    if fact.get("status") == "unknown" or not pooled:
        return fact.get("status", "unknown"), {}
    roles = (structure or {}).get("instruction_roles") or {}
    classes = {klass: rows for klass, rows in (value.get("by_class") or {}).items()
               if rows and "operand_load" in (roles.get(klass) or ())}
    return fact["status"], classes or {"operand_load": pooled}


def _transfer(reference, candidate, ref_id, cand_id, structure=None):
    ref_status, ref_classes = _extent(reference.get("derived_effects"), structure)
    cand_status, cand_classes = _extent(candidate.get("derived_effects"), structure)
    pointer = [f"{ref_id}#derived_effects.facts.transfer_tiles", f"{cand_id}#derived_effects.facts.transfer_tiles"]
    keys = sorted(set(ref_classes) | set(cand_classes)) or ["operand_load"]
    out = []
    for key in keys:
        ref_shapes, cand_shapes = ref_classes.get(key) or [], cand_classes.get(key) or []
        ref_max = max((r["rows"] * r["cols"] for r in ref_shapes), default=None)
        cand_max = max((r["rows"] * r["cols"] for r in cand_shapes), default=None)
        known = ref_max is not None and cand_max is not None
        status = ("observed" if ref_status == cand_status == "observed" else "observed_partial") if known else "unknown"
        unresolved = [side for side, value in (("reference", ref_max), ("candidate", cand_max)) if value is None]
        out.append(dict(
            effect="operand_transfer_extent", operand=key, status=status,
            **(dict(reason=f"no resolved operand_load transfer extent for {key} on the " + " and ".join(unresolved)
                    + " side (register values not statically known, or the class is absent there)")
               if unresolved else {}),
            reference=dict(largest_elements=ref_max, shapes=ref_shapes),
            candidate=dict(largest_elements=cand_max, shapes=cand_shapes),
            needed_change=_direction(ref_max, cand_max) if known else None,
            basis="decoded rows x cols of operand_load-role transfers per instruction class "
                  "(declared transfer-bound fields; static sites, not issued transfers)",
            limits=[] if status == "observed" else ["some transfer extents are unresolved"], evidence=pointer))
    return out


def _broadcast(effects):
    fact = _fact(effects, "broadcast_loads")
    if fact.get("status") == "unknown":
        return None, None
    value = fact.get("value") or {}
    states = sorted(set(value.get("nonzero_by_state") or {}) | set(value.get("stride_zero_by_state") or {}))
    return sum((value.get("broadcast_movement_sites") or {}).values()), states


def _broadcasts(reference, candidate, ref_id, cand_id):
    (ref_sites, ref_states), (cand_sites, cand_states) = (_broadcast(r.get("derived_effects"))
                                                          for r in (reference, candidate))
    known = ref_sites is not None and cand_sites is not None
    pointer = [f"{i}#derived_effects.facts.broadcast_loads" for i in (ref_id, cand_id)]
    rows = [dict(effect="broadcast_load_sites", status="observed" if known else "unknown",
                 reference=ref_sites, candidate=cand_sites,
                 needed_change=_direction(ref_sites, cand_sites) if known else None,
                 basis="movement sites issued under a stride-zero load configuration (static listing order)",
                 limits=["a stride inherited from before the occurrence stays unknown"], evidence=pointer)]
    known = ref_states is not None and cand_states is not None
    rows.append(dict(effect="configured_load_states", status="observed" if known else "unknown",
                     reference=ref_states, candidate=cand_states,
                     needed_change=_direction(len(ref_states), len(cand_states)) if known else None,
                     basis="distinct declared load-state ports given a stride configuration",
                     limits=[], evidence=pointer))
    return rows


def _count(name, role, reference, candidate, ref_id, cand_id):
    ref, cand = _roles(reference).get(role, 0), _roles(candidate).get(role, 0)
    return dict(effect=name, status="observed", reference=ref, candidate=cand, needed_change=_direction(ref, cand),
                basis=f"static {role}-role sites in the selected span", limits=[],
                evidence=[f"{ref_id}#schedule_model.role_counts", f"{cand_id}#schedule_model.role_counts"])


def _config(reference, candidate, ref_id, cand_id):
    def rate(record):
        roles = _roles(record)
        return _ratio(roles.get("config", 0), sum(roles.get(r, 0) for r in ("operand_load", "readout")))

    ref, cand = rate(reference), rate(candidate)
    status = _rate_status(reference, candidate) if ref and cand else "unknown"
    return dict(effect="config_per_transfer", status=status, reference=ref, candidate=cand,
                needed_change=_direction(ref and ref["value"], cand and cand["value"]) if status != "unknown" else None,
                basis="static config-role sites per operand_load/readout-role site", limits=[],
                evidence=[f"{ref_id}#schedule_model.role_counts", f"{cand_id}#schedule_model.role_counts"])


def _loop(reference, candidate, ref_id, cand_id):
    def sites(record):
        fact = _fact(record.get("derived_effects"), "hardware_loop")
        return None if fact.get("status") == "unknown" else (fact.get("value") or {}).get("loop_descriptor_sites")

    ref, cand = sites(reference), sites(candidate)
    known = ref is not None and cand is not None
    return dict(effect="loop_descriptor_sites", status="observed" if known else "unknown", reference=ref,
                candidate=cand, needed_change=_direction(ref, cand) if known else None,
                basis="sites whose selected role is loop_descriptor (dispatch structure)", limits=[],
                evidence=[f"{i}#derived_effects.facts.hardware_loop" for i in (ref_id, cand_id)])


def _unknowns(reference, candidate, ref_id, cand_id):
    """Facts either side could not establish stay explicit; they are never compared by default."""
    rows = []
    for name in ("operand_retention", "local_address_windows", "dataflow", "operation"):
        ref, cand = (_fact(r.get("derived_effects"), name) for r in (reference, candidate))
        if ref.get("status") in ("observed", "declared") and cand.get("status") in ("observed", "declared"):
            if ref.get("value") == cand.get("value"):
                continue
            status = "observed"
        else:
            status = "unknown"
        rows.append(dict(effect=name, status=status, reference_status=ref.get("status"),
                         candidate_status=cand.get("status"),
                         reason=ref.get("reason") or cand.get("reason") or (ref.get("limits") or [None])[0],
                         needed_change=None, evidence=[f"{i}#derived_effects.facts.{name}" for i in (ref_id, cand_id)]))
    return rows


# --- facts of a proved dynamic command stream ----------------------------------------------------

#: Per-role facts of a proved stream, then the per-block output-group fact.
ROLE_FACTS = ("role_transfer_width", "operand_buffering", "issue_overlap", "prefetch_distance")
STREAM_FACTS = ROLE_FACTS + ("output_group_extent",)
STREAM_EFFECTS = STREAM_FACTS + ("prefetch_interleave", "prefetch_compute_ahead")
STREAM_SCOPE = "proved_dynamic_command_stream"
NO_STREAM = "no_proved_command_stream"
_SEPARATE = ("preload", "preloads", "array_preload")  # kept apart from data-movement roles by the contract
#: What each stream comparison measures; how its values read is :data:`COMPARISON_SEMANTICS`. A
#: comparison, not a speed claim.
_STREAM_BASIS = dict(
    role_transfer_width="elements per issued transfer of this movement role (proved stream)",
    operand_buffering="disjoint local windows of this role and whether consecutive loads alternate (proved stream)",
    issue_overlap="fraction of this role's loads with compute issued after them before the next barrier "
                  "(proved stream)",
    prefetch_distance="median loads issued between each load of this operand and the first issue reading its "
                      "rows (proved stream; a measured distance)",
    prefetch_compute_ahead="fraction of this operand's consumed loads with at least one compute issued between "
                           "the load and its first reader (proved stream)")
_STREAM_METRIC = dict(role_transfer_width="elements_per_transfer", operand_buffering="disjoint_windows",
                      issue_overlap="fraction", prefetch_distance="median_loads_between",
                      prefetch_compute_ahead="fraction_with_compute_ahead")
#: Attribution labels of a per-operand row: matched by the consuming compute port, or by argument register.
PORT, REGISTER = "consumer_port", "argument_register"


def stream_fact(effects, name):
    """One stream fact, or an explicit unknown when no proved command stream produced it."""
    facts = (effects or {}).get("facts") or {}
    fact = facts.get(name)
    if fact is None and any(isinstance(facts.get(other), dict) and facts[other].get("status") not in
                            (None, "unknown") for other in ROLE_FACTS):
        return dict(status="unknown", scope=STREAM_SCOPE,
                    reason=f"{name} absent from a record whose other stream facts were derived "
                           "(derived before this fact existed; re-derive the record)")
    if not isinstance(fact, dict) or fact.get("status") in (None, "unknown"):
        return dict(status="unknown", reason=(fact or {}).get("reason") or NO_STREAM, scope=STREAM_SCOPE)
    return fact


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
    if value and all(isinstance(v, dict) and v and all(isinstance(w, dict) and "disjoint_windows" in w
                                                       for w in v.values()) for v in value.values()):
        return {f"{role}:{operand}": w for role, v in value.items() for operand, w in v.items()}, None
    inner = next((value[k] for k in ("by_role", "roles", "per_role") if isinstance(value.get(k), dict)), value)
    separate = next((_first(value, k) if k in value else inner.get(k) for k in _SEPARATE
                     if k in value or k in inner), None)
    return ({str(k): v for k, v in inner.items() if isinstance(v, (dict, list)) and k not in _SEPARATE},
            separate)


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
    return dict(shapes=[dict(rows=r, cols=c, elements=e, count=n) for (r, c, e), n in sorted(table.items())],
                transfers=transfers, elements=elements,
                elements_per_transfer=round(elements / transfers, 6) if transfers else None)


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
    return dict(disjoint_windows=disjoint, alternating=alternating if isinstance(alternating, bool) else None,
                windows=entry.get("windows"), definition=entry.get("definition") or "phase_windows_v1")


def _distance_value(entry):
    """Load-to-first-reader distance summary of one operand (or the pooled ``all`` entry)."""
    if not isinstance(entry, dict) or not isinstance(entry.get("median_loads_between"), (int, float)):
        return None
    return {k: entry.get(k) for k in ("median_loads_between", "min_loads_between", "fraction_with_compute_ahead",
                                      "loads", "consumed_loads", "unconsumed_loads")}


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
    return dict(count=count if type(count) is int else None, total=total if type(total) is int else None,
                fraction=round(float(fraction), 6))


def _overlap(value):
    """``({role: overlap}, run structure)``; role ``all`` when the stream reports one pooled value."""
    if not isinstance(value, dict):
        return {}, {}
    structure = {k: value[k] for k in ("compute_run_median", "load_run_median", "median_compute_run",
                                       "median_load_run", "fences", "fence_count", "computes_per_fence")
                 if k in value}
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
        alternation["comparison"] = (f"disjoint_windows: reference {r}, candidate {c}, same; alternating: reference "
                                     f"{'yes' if a else 'no'}, candidate {'yes' if b else 'no'}")
        return alternation
    if name == "role_transfer_width" and ref["shapes"] != cand["shapes"]:
        # Same mean width, different shape composition: a difference with no side.
        return dict(fields, direction="candidate_differs", beyond_reference=None,
                    comparison=fields["comparison"] + "; shape composition differs")
    if name == "issue_overlap" and ref_structure != cand_structure:
        return dict(fields, direction="candidate_differs", beyond_reference=None,
                    comparison=fields["comparison"] + "; run structure differs")
    return fields


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
    return dict(disjoint_windows=len(merged), alternating=entries[0]["alternating"] if len(entries) == 1 else None,
                windows=[dict(lo=lo, hi=hi - 1) for lo, hi in merged], definition=definitions.pop())


def _role_fallback(name, ref_fact, cand_fact, ref_map, cand_map, summarize, registers=({}, {})):
    """Role-level comparisons where the two sides' per-operand entries do not all match.

    Returns ``(labels, added, notes)``: role-level keys both sides already report (labelled), role-level
    entries aggregated from the per-operand entries when no such key exists, and the per-operand keys
    whose rows stay unknown because they have no counterpart. ``registers`` maps a re-keyed entry back to
    each side's argument registers, which label the attribution."""
    ref_groups, cand_groups = (_operand_groups(name, f.get("value"), m)
                               for f, m in ((ref_fact, ref_map), (cand_fact, cand_map)))
    labels, added, notes = {}, {}, {}
    for role in sorted(set(ref_groups) & set(cand_groups)):
        ref_ops, cand_ops = ref_groups[role], cand_groups[role]
        if set(ref_ops) == set(cand_ops):
            continue
        attribution = {side: sorted(r for operand, key in ops.items() for r in names.get(key, [operand]))
                       for side, ops, names in (("reference", ref_ops, registers[0]),
                                                ("candidate", cand_ops, registers[1]))}
        key = (role if name in ("issue_overlap", "prefetch_distance") or (role in ref_map and role in cand_map)
               else f"{role}:all")
        if key in ref_map and key in cand_map:
            labels[key] = attribution
        else:
            added[key] = (attribution, *(_aggregate(name, [summarize(m[k]) for k in groups.values()])
                                         for m, groups in ((ref_map, ref_ops), (cand_map, cand_ops))))
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
    for entries in (value.values() if isinstance(value, dict) else ()):
        for operand, entry in (entries.items() if isinstance(entries, dict) else ()):
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
                    kept[side][key] = (f"no consuming compute port is derived for {operand(group, key)} on the "
                                       f"{_SIDES[side]} side" + (f" ({gap})" if gap else ""))
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
            row = dict(attribution=PORT, consumer_port=port,
                       operand_attribution={_SIDES[s]: registers[s].get(new, []) for s in (0, 1)},
                       attribution_basis="matched by the compute port that reads the loaded rows; argument "
                                         "registers are labels only")
            absent = [s for s in (0, 1) if port not in by_port[s]]
            if absent:
                other = absent[0]
                row["reason"] = (f"no {_SIDES[other]} load of this role is attributed to port {port}: "
                                 + "; ".join(kept[other].values()) if kept[other] else
                                 f"no {_SIDES[other]} load of this role is read by port {port}")
            info[new] = row
        for side in (0, 1):
            other = 1 - side
            for key, why in kept[side].items():
                counterpart = key in kept[other]
                row = dict(attribution=REGISTER, operand_attribution={_SIDES[side]: [operand(group, key)]},
                           attribution_basis=("matched by argument register: " if counterpart
                                              else "argument register only: ") + why)
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


def _stream_rows(reference, candidate, ref_id, cand_id):
    rows = []
    gaps = (_operand_gaps(reference), _operand_gaps(candidate))
    for name in ROLE_FACTS:
        ref_fact, cand_fact = (stream_fact(r.get("derived_effects"), name) for r in (reference, candidate))
        pointer = [f"{i}#derived_effects.facts.{name}" for i in (ref_id, cand_id)]
        effects = (name, "prefetch_compute_ahead") if name == "prefetch_distance" else (name,)
        if "unknown" in (ref_fact["status"], cand_fact["status"]):
            sides = [s for s, f in (("reference", ref_fact), ("candidate", cand_fact)) if f["status"] == "unknown"]
            reason = ((ref_fact.get("reason") or cand_fact.get("reason") or NO_STREAM)
                      + " on the " + " and ".join(sides) + " side")
            known = [(s, f) for s, f in (("reference", ref_fact), ("candidate", cand_fact)) if f["status"] != "unknown"]
            for effect in effects:
                rows.append(dict(effect=effect, role=None, status="unknown", direction="unknown", needed_change=None,
                                 reason=reason, scope=STREAM_SCOPE, evidence=pointer))
                if len(known) != 1:
                    continue
                # The established side still describes itself; nothing is compared.
                side, fact = known[0]
                mapping, _, _ = _side_maps(name, fact)
                ports = _consumer_ports(name, fact.get("value"))
                for key in sorted(mapping):
                    rows.append(dict(effect=effect, role=key, status="unknown", direction="unknown",
                                     needed_change=None, reason=reason, scope=STREAM_SCOPE, evidence=pointer,
                                     basis=_STREAM_BASIS[effect], **{side: _summarizer(name)(mapping[key])},
                                     **({"consumer_port": ports[key]} if ports.get(key) else {})))
            continue
        (ref_map, ref_sep, ref_structure), (cand_map, cand_sep, cand_structure) = (
            _side_maps(name, f) for f in (ref_fact, cand_fact))
        summarize, extra = _summarizer(name), {}
        structures = (ref_structure, cand_structure) if name == "issue_overlap" else (None, None)
        if name == "issue_overlap":
            extra = dict(reference_structure=ref_structure, candidate_structure=cand_structure)
        if name == "role_transfer_width" and (ref_sep is not None or cand_sep is not None):
            extra = dict(preload=dict(reference=ref_sep, candidate=cand_sep,
                                      basis="kept separate from data-movement roles; not compared"))
        if name == "role_transfer_width":
            extra["route_effect"] = "operand_transfer_extent"
        ref_map, cand_map, matched, registers = _port_match(name, ref_fact, cand_fact, ref_map, cand_map, gaps)
        labels, added, notes = _role_fallback(name, ref_fact, cand_fact, ref_map, cand_map, summarize, registers)

        def compare(effect, role, ref, cand, absent):
            row = dict(effect=effect, role=role, status="observed", reference=ref, candidate=cand,
                       basis=_STREAM_BASIS[effect], scope=STREAM_SCOPE, evidence=pointer, **extra)
            if ref is None or cand is None:
                missing = [s for s, v in (("reference", ref), ("candidate", cand)) if v is None]
                row.update(status="unknown", direction="unknown", needed_change=None,
                           reason=f"role {role} {absent} on the " + " and ".join(missing) + " side")
                return row
            # Run structure belongs to the pooled row; a per-operand row compares that operand alone.
            pooled = structures if role == "all" else (None, None)
            row.update(_stream_direction(effect, ref, cand, *pooled))
            if row["direction"] == "unknown":
                why = [v["reason"] for v in (ref, cand) if isinstance(v, dict) and v.get("reason")]
                if effect == "operand_buffering" and ref.get("definition") != cand.get("definition"):
                    why = [f"the sides counted windows under different definitions (reference: "
                           f"{ref.get('definition')}; candidate: {cand.get('definition')}); re-derive the older "
                           "record"]
                row.update(status="unknown", reason="; ".join(why) or
                           _STREAM_METRIC[effect] + " not established on both sides")
            return row

        def attributed(attribution):
            return dict(attribution=FALLBACK, operand_attribution=attribution,
                        attribution_basis="the sides' per-operand entries do not all match (by consuming compute "
                                          "port, or by argument register where neither side derived a port); "
                                          "compared over all of the role's loads; argument registers are labels")

        def label(row, key):
            if key in matched:
                found = matched[key]
                row.update({k: v for k, v in found.items() if k != "reason"})
                if row["status"] == "unknown" and found.get("reason"):
                    row["reason"] = found["reason"] + (f"; compared at role level in role {notes[key][0]}"
                                                       if key in notes else "")
            elif key in labels:
                row.update(attributed(labels[key]))
            elif key in notes:
                role_key, attribution = notes[key]
                row.update(attribution="per_operand", operand_attribution=attribution,
                           reason=f"operand attribution differs between the sides (reference: "
                                  f"{', '.join(attribution['reference'])}; candidate: "
                                  f"{', '.join(attribution['candidate'])}); compared at role level in role "
                                  f"{role_key}")
            return row

        for effect in effects:
            for role in sorted((set(ref_map) | set(cand_map)) - set(added)):
                row = compare(effect, role, *(summarize(m[role]) if role in m else None for m in (ref_map, cand_map)),
                              "absent or unreadable")
                rows.append(label(row, role))
            for role, (attribution, ref, cand) in sorted(added.items()):
                rows.append(dict(compare(effect, role, ref, cand, "per-operand entries do not combine"),
                                 **attributed(attribution)))
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


#: Sub-facts of ``issue_overlap``: (effect, value keys, basis); read under :data:`COMPARISON_SEMANTICS`.
_INTERLEAVE = (
    ("prefetch_interleave", ("compute_run_between_load_bands", "median_compute_run_between_load_bands",
                             "compute_run_median", "median_compute_run"),
     "median compute run between load bands; loads_per_compute_run is listed beside it (proved stream)"),
)
_GROUP_BASIS = ("median output columns completed per barrier-delimited block: distinct column start positions "
                "of the block's readouts (modulo the configured readout row pitch) x their column extent "
                "(proved stream)")


def _group_columns(fact, overlap):
    """``(columns, reason)`` of one side: its ``output_group_extent`` fact, else an extent an older
    ``issue_overlap`` value carried; ``None`` with the fact's reason when neither establishes it."""
    value = fact.get("value") if fact.get("status") not in (None, "unknown") else None
    if isinstance(value, dict):
        if isinstance(value.get("median_output_columns"), (int, float)):
            return value["median_output_columns"], None
        return None, ("output_group_extent does not establish output columns per block (readout row pitch or a "
                      "uniform column extent not derived, or the record predates output_columns)")
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
    row = dict(effect="output_group_extent", role="all", status="observed", reference=ref, candidate=cand,
               basis=_GROUP_BASIS, scope=STREAM_SCOPE, evidence=pointer, needed_change=None)
    if ref is None or cand is None:
        reasons = [f"{side}: {why}" for side, value, why in (("reference", ref, ref_reason),
                                                             ("candidate", cand, cand_reason)) if value is None]
        row.update(status="unknown", direction="unknown", reason="; ".join(reasons))
    else:
        row.update(_order("output_group_extent", ref, cand))
    return [row]


def _interleave_rows(ref_value, cand_value, pointer):
    rows = []
    for effect, keys, basis in _INTERLEAVE:
        ref, cand = (_scalar_map(_first(v, *keys)) for v in (ref_value, cand_value))
        if not ref and not cand:
            rows.append(dict(effect=effect, role=None, status="unknown", direction="unknown", needed_change=None,
                             reason="not reported by the proved stream on either side", scope=STREAM_SCOPE,
                             evidence=pointer))
            continue
        for role in sorted(set(ref) | set(cand)):
            r, c = ref.get(role), cand.get(role)
            row = dict(effect=effect, role=role, status="observed", reference=r, candidate=c, basis=basis,
                       scope=STREAM_SCOPE, evidence=pointer, needed_change=None)
            if effect == "prefetch_interleave":
                row.update(reference_loads_per_compute_run=ref_value.get("loads_per_compute_run"),
                           candidate_loads_per_compute_run=cand_value.get("loads_per_compute_run"))
            if r is None or c is None:
                row.update(status="unknown", direction="unknown", reason=f"role {role} absent on one side")
            else:
                row.update(_order(effect, r, c))
            rows.append(row)
    return rows


# --- further proved-stream facts, compared only when present on both sides ------------------------

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


#: (fact, compared effect, metric of one role entry, basis); each effect reads under :data:`COMPARISON_SEMANTICS`
EXTRA_FACTS = (
    ("host_issue_rate", "host_issue_rate", lambda e: _number(e, "rate", "issue_rate", "value", "median"),
     "accelerator commands issued per host instruction (proved stream)"),
    ("reconfiguration", "reconfiguration", lambda e: _number(e, "config_per_block", "per_block", "value"),
     "configuration commands per block (proved stream)"),
    ("reconfiguration", "reconfiguration_in_loop", _in_loop,
     "configuration issued inside the loop (1) or hoisted (0) (proved stream)"),
    ("local_capacity_use", "local_capacity_use", _capacity_fraction,
     "local rows used over the declared capacity (proved stream)"),
    ("local_capacity_use", "same_bank_halves", lambda e: _number(e, "same_bank_halves", "same_bank"),
     "alternating halves placed in the same bank (1) or not (0) (proved stream)"),
    ("dram_contiguity", "dram_contiguity", lambda e: _number(e, "contiguous_fraction", "contiguous", "fraction"),
     "fraction of transfer rows contiguous in memory (proved stream)"),
    ("epilogue_placement", "epilogue_placement", _on_accelerator,
     "output epilogue on the accelerator (1) or not (0) (proved stream)"),
    ("resource_occupancy", "queue_occupancy", lambda e: _number(e, "max_outstanding_before_dependency"),
     "largest count of commands routed to one dispatch queue before a sync or dependent "
     "consumer (proved stream)"),
    ("resource_occupancy", "queue_points_past_entries", lambda e: _number(e, "points_exceeding_dispatch_entries"),
     "issue points past the queue's RTL dispatch entries (proved stream)"),
    ("transfer_legality", "transfer_exceeds_bound", lambda e: _number(e, "exceeds_bound"),
     "transfers past an RTL transfer bound (proved stream)"),
    ("post_store_reuse", "post_store_reuse", lambda e: _number(e, "read_again") + _number(e, "accumulated_into")
     if _number(e, "read_again") is not None and _number(e, "accumulated_into") is not None else None,
     "stored local rows read or accumulated into again before a rewrite (proved stream)"),
)
STREAM_EFFECTS = STREAM_EFFECTS + tuple(effect for _, effect, *_ in EXTRA_FACTS)


def _per_role(value, metric):
    if not isinstance(value, dict):
        return {}
    inner = next((value[k] for k in ("by_role", "roles", "per_role", "per_queue") if isinstance(value.get(k), dict)),
                 None)
    if inner is None:
        pooled = metric(value)
        return {"all": pooled} if pooled is not None else {}
    return {str(k): metric(v) for k, v in inner.items() if k not in _SEPARATE and metric(v) is not None}


def _extra_rows(reference, candidate, ref_id, cand_id):
    """Rows only for facts the stream reports on either side; an older stepper adds no rows."""
    rows = []
    for name, effect, metric, *_, basis in EXTRA_FACTS:
        ref_fact, cand_fact = (((r.get("derived_effects") or {}).get("facts") or {}).get(name)
                               for r in (reference, candidate))
        if ref_fact is None and cand_fact is None:
            continue
        pointer = [f"{i}#derived_effects.facts.{name}" for i in (ref_id, cand_id)]
        known = [f for f in (ref_fact, cand_fact) if isinstance(f, dict) and f.get("status") not in (None, "unknown")]
        if len(known) < 2:
            rows.append(dict(effect=effect, role=None, status="unknown", direction="unknown", needed_change=None,
                             reason=next((f.get("reason") for f in (ref_fact, cand_fact)
                                          if isinstance(f, dict) and f.get("reason")), NO_STREAM),
                             scope=STREAM_SCOPE, evidence=pointer))
            continue
        ref, cand = (_per_role(f.get("value"), metric) for f in (ref_fact, cand_fact))
        if not ref and not cand:
            continue  # this fact reports no value for this metric on either side
        for role in sorted(set(ref) | set(cand)):
            r, c = ref.get(role), cand.get(role)
            row = dict(effect=effect, fact=name, role=role, status="observed", reference=r, candidate=c,
                       basis=basis, scope=STREAM_SCOPE, evidence=pointer, needed_change=None)
            if r is None or c is None:
                row.update(status="unknown", direction="unknown", reason=f"role {role} absent on one side")
            else:
                row.update(_order(effect, r, c) or dict(status="unknown", direction="unknown",
                                                        reason="values are not comparable numbers"))
            rows.append(row)
    return rows


def _summary_value(value):
    """The compared number of one side of a summary row: a rate, a largest extent, a count or a set size."""
    if isinstance(value, dict):
        value = value.get("value", value.get("largest_elements"))
    if isinstance(value, list):
        return len(value)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _position(row):
    """Comparison fields of one summary row, consistent with its ``needed_change``."""
    if row["status"] == "unknown":
        return dict(direction="unknown")
    fields = _order(row["effect"], _summary_value(row.get("reference")), _summary_value(row.get("candidate")))
    if fields is None or fields["needed_change"] != row.get("needed_change"):
        return dict(direction="candidate_differs" if row.get("needed_change") else "candidate_matches")
    if row["effect"] == "configured_load_states":
        fields["comparison"] = "state count: " + fields["comparison"]
    return fields


def partial_recipe(rows):
    """Set when one role or effect matches the reference while a role's width, buffering or overlap differs."""
    stream = [r for r in rows if r["effect"] in STREAM_EFFECTS]
    if not stream or all(r["direction"] == "unknown" for r in stream):
        return dict(status="unknown", reason=NO_STREAM, matched=[], still_differs=[])

    def key(row):
        return dict(effect=row["effect"], role=row.get("role") or row.get("operand"), direction=row["direction"],
                    **({"needed_change": row["needed_change"]} if row.get("needed_change") else {}),
                    **({"comparison": row["comparison"]} if row.get("comparison") else {}))

    roled = [r for r in rows if r.get("role") or r.get("operand")]
    matched = [key(r) for r in roled if r.get("direction") == "candidate_matches"]
    beyond = [key(r) for r in roled if r.get("direction") == "candidate_differs" and r.get("beyond_reference")]
    differs = [key(r) for r in stream if r["direction"] == "candidate_differs" and not r.get("beyond_reference")]
    return dict(status=bool((matched or beyond) and differs), matched=matched, beyond_reference=beyond,
                still_differs=differs, unknown=[key(r) for r in stream if r["direction"] == "unknown"],
                basis="per-role rows of the proved command stream and per-operand transfer extents; a factual "
                      "comparison with the reference, not a statement about speed")


def mechanism_differences(reference, candidate, *, structure=None):
    """Every compared effect; equal values are kept as ``same`` so absence is never silent."""
    ref_id, cand_id = _evidence_id(reference), _evidence_id(candidate)
    rows = [
        _role_rate("operand_load_per_compute", "operand_load", reference, candidate, ref_id, cand_id),
        _role_rate("weight_load_per_compute", "weight_load", reference, candidate, ref_id, cand_id),
        *_transfer(reference, candidate, ref_id, cand_id, structure),
        *_broadcasts(reference, candidate, ref_id, cand_id),
        _config(reference, candidate, ref_id, cand_id),
        _count("sync_sites", "sync", reference, candidate, ref_id, cand_id),
        _loop(reference, candidate, ref_id, cand_id),
    ]
    for row in rows:
        row["difference"] = ("unknown" if row["status"] == "unknown"
                             else "differs" if row["needed_change"] else "same")
        row.update(_position(row))
    for row in _stream_rows(reference, candidate, ref_id, cand_id):
        row["difference"] = ("unknown" if row["direction"] == "unknown"
                             else "same" if row["direction"] == "candidate_matches" else "differs")
        rows.append(row)
    return rows + [dict(row, difference="unknown" if row["status"] == "unknown" else "differs",
                        direction="unknown" if row["status"] == "unknown" else "candidate_differs")
                   for row in _unknowns(reference, candidate, ref_id, cand_id)]


def command_stream(rows, decoder, *, max_steps=8_000_000, max_commands=1_000_000):
    """Decoded commands of a proved bounded scalar execution, or ``(None, reason)``.

    Feature-detected: until the machine stepper records commands and ``decode_stream`` exists, the
    stream facts stay unknown with :data:`NO_STREAM`."""
    import inspect

    from merlin.compile.analysis import riscv_machine
    from merlin.kernels import decoded_effects

    counts = getattr(riscv_machine, "finite_control_counts", None)
    decode = getattr(decoded_effects, "decode_stream", None)
    if (counts is None or decode is None or "record_commands" not in inspect.signature(counts).parameters
            or not decoder or decoder.get("semantics") is None):
        return None, NO_STREAM
    opcode = decoder["semantics"].isa_constants(decoder["target"]).get("CUSTOM_OPCODE")
    if opcode is None:
        return None, NO_STREAM + ": the selected ISA declares no custom opcode"
    counted = counts(list(rows), custom_opcode=opcode, max_steps=max_steps, max_commands=max_commands,
                     record_commands=True)
    if counted.get("status") != "proved":
        return None, NO_STREAM + ": " + str(counted.get("reason") or counted.get("status"))
    stream = next((counted[k] for k in ("command_stream", "recorded_commands", "stream") if k in counted), None)
    if stream is None:
        return None, NO_STREAM + ": the stepper recorded no stream"
    if counted.get("command_stream_truncated"):
        return None, NO_STREAM + ": the recorded stream reached its bound (truncated)"
    return decode(stream, decoder=decoder), None


def stream_effects(analysis, structure, stream, *, evidence_id=None):
    """``derive_effects`` facts of one decoded proved stream, by whichever parameter the deriver takes."""
    import inspect

    from merlin.kernels.decoded_effects import derive_effects

    accepted = inspect.signature(derive_effects).parameters
    options = dict(stream=stream) if "stream" in accepted else dict(commands=stream)
    return derive_effects(analysis, structure, evidence_id=evidence_id, **options).get("facts") or {}


# --- admission over the candidate's own shape and the selected RTL facts -------------------------

def selected_geometry(target, facts):
    """Block edge, operand/accumulator rows and element widths from the selected RTL facts, or a reason."""
    from merlin.compile.scheduling.block_schedule import BlockScheduleError
    from merlin.compile.scheduling.derive import geometry_from_address_space
    from merlin.targetgen.address_space import accumulator_store, derive_address_space, operand_store

    if not target or facts is None:
        return None, "selected target or RTL facts unavailable"
    space = derive_address_space(target, facts=facts)
    try:
        geometry = geometry_from_address_space(space)
    except BlockScheduleError as error:
        return None, str(error)
    operand, accumulator = operand_store(space).store, accumulator_store(space).store
    if None in (operand, accumulator) or None in (operand.element_bits, accumulator.element_bits,
                                                  operand.row_bytes, accumulator.row_bytes):
        return None, "operand or accumulator element width/row bytes not derived from the facts"
    return dict(block=geometry.block, operand_rows=geometry.operand_rows, accumulator_rows=geometry.accumulator_rows,
                operand_bank_rows=geometry.operand_bank_rows, operand_store=operand.name,
                accumulator_store=accumulator.name, operand_row_bytes=operand.row_bytes,
                accumulator_row_bytes=accumulator.row_bytes, operand_element_bits=operand.element_bits,
                accumulator_element_bits=accumulator.element_bits, sources=dict(geometry.sources)), None


def _family(shape):
    keys = set(shape or ())
    if set(_DENSE) <= keys:
        return "dense"
    if set(_CONV) <= keys:
        return "convolution"
    return None


def _dense(shape, geometry):
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.transforms.dense_planning import DenseProblem, DenseResources

    edge = geometry["block"]
    problem = DenseProblem(shape["m"], shape["k"], shape["n"], edge, edge, edge,
                           geometry["operand_element_bits"] // 8, geometry["operand_element_bits"] // 8,
                           geometry["accumulator_element_bits"] // 8)
    mt, kt, nt = problem.tiles
    resources = DenseResources(geometry["operand_rows"] * geometry["operand_row_bytes"],
                               geometry["accumulator_rows"] * geometry["accumulator_row_bytes"],
                               max_m_blocks=mt, max_n_blocks=nt, max_k_blocks=kt,
                               partial_output_tiles=True, partial_output_blocks=True)

    def storage(first, first_rows):
        """One disjoint witness split of the operand store: ``first`` gets exactly ``first_rows``."""
        rest = geometry["operand_rows"] - first_rows
        if first_rows <= 0 or rest <= 0:
            return None
        other = "rhs" if first == "lhs" else "lhs"
        made = {first: DenseMemoryView(first, geometry["operand_store"], geometry["operand_row_bytes"], 0, first_rows),
                other: DenseMemoryView(other, geometry["operand_store"], geometry["operand_row_bytes"],
                                       first_rows, rest)}
        return DenseLoopStorage(made["lhs"], made["rhs"], DenseMemoryView(
            "acc", geometry["accumulator_store"], geometry["accumulator_row_bytes"], 0, geometry["accumulator_rows"]))

    return problem, resources, storage


def _m_panel(shape, geometry):
    from merlin.compile.scheduling.dense_panel_pipeline import dense_m_panel_capacity

    problem, resources, storage = _dense(shape, geometry)
    mt, kt, nt = problem.tiles
    if nt > _MAX_N_TILES:
        return dict(status="unknown", missing=[f"bounded evaluation covers at most {_MAX_N_TILES} N tiles"])
    rows, satisfied, blockers = [], [], {}
    for n_tiles in range(1, nt + 1):
        rhs_rows = kt * n_tiles * problem.tile_k
        witness = storage("rhs", rhs_rows)
        if witness is None:
            blockers["rhs_private_rows"] = blockers.get("rhs_private_rows", 0) + 1
            rows.append(dict(n_tiles=n_tiles, maximum_m_tiles=0, limiting_conditions=["rhs_private_rows"],
                             retained_rhs_rows=rhs_rows))
            continue
        result = dense_m_panel_capacity(problem, resources, witness, n_tiles=n_tiles)
        rows.append(dict(n_tiles=n_tiles, maximum_m_tiles=result["maximum_m_tiles"],
                         limiting_conditions=result["limiting_conditions"],
                         retained_rhs_rows=result["retained_rhs_rows"]))
        if result["maximum_m_tiles"] >= 1 and all(result["conditions"].values()):
            satisfied.append(n_tiles)
        for name, value in result["conditions"].items():
            if not value:
                blockers[name] = blockers.get(name, 0) + 1
    return dict(status="satisfied" if satisfied else "violated",
                predicate="dense_panel_pipeline.dense_m_panel_capacity per original N width",
                satisfied_n_tiles=satisfied, violated_bounds=sorted(blockers),
                violated_bound_counts=blockers, per_n_tiles=rows, source_tiles=dict(m=mt, k=kt, n=nt),
                witness=("for each N width the RHS view receives exactly its retained full-K rows and the LHS "
                         "view the remaining operand rows; a construction condition, not an assignment choice"))


def _segmented(shape, geometry):
    from merlin.compile.scheduling.dense_segmented_pipeline import dense_segmented_blocks

    problem, resources, storage = _dense(shape, geometry)
    mt, kt, nt = problem.tiles
    lhs_rows = kt * problem.tile_m  # one complete M tile of the full-K LHS panel
    witness = storage("lhs", lhs_rows)
    if witness is None:
        return dict(status="violated", predicate="dense_segmented_pipeline.dense_segmented_blocks",
                    violated_bounds=["full_lhs_panel_rows"], required_lhs_rows=lhs_rows,
                    operand_rows=geometry["operand_rows"])
    choices = dense_segmented_blocks(problem, resources, witness)
    blocks = sorted({(c.m, c.k, c.n) for c in choices})
    return dict(status="satisfied" if blocks else "violated",
                predicate="dense_segmented_pipeline.dense_segmented_blocks",
                admitted_blocks=[dict(m=m, k=k, n=n) for m, k, n in blocks[:32]], admitted_block_count=len(blocks),
                ordering="block dimensions, not preference",
                violated_bounds=[] if blocks else ["no M/N block holds a full-K LHS panel and two RHS K segments"],
                required_lhs_rows_per_m_tile=lhs_rows, source_tiles=dict(m=mt, k=kt, n=nt),
                witness="LHS view holds exactly one full-K M tile; RHS view the remaining operand rows")


def _convolution(shape, geometry):
    from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency

    edge = geometry["block"]
    chunks = shape["kh"] * shape["kw"] * -(-shape["c"] // edge)
    k_rows, n_cols = chunks * edge, -(-shape["oc"] // edge) * edge
    layout = plan_dense_rhs_residency(1, k_rows, n_cols, block=edge, operand_rows=geometry["operand_rows"],
                                      accumulator_rows=geometry["accumulator_rows"], rhs_base=edge,
                                      max_static_rhs_tiles=geometry["operand_rows"] // edge)
    required = edge + chunks * (n_cols // edge) * edge
    return dict(status="satisfied" if layout is not None else "violated",
                predicate="convolution_exploration.plan_convolution_rhs_residency capacity owner "
                          "(dense_rhs_residency.plan_dense_rhs_residency)",
                required_operand_rows=required, operand_rows=geometry["operand_rows"],
                violated_bounds=[] if layout is not None else ["full_rhs_rows_exceed_private_operand_rows"],
                witness=("RHS base directly after one LHS tile; one column interval per output-channel tile. The "
                         "original scalar-safe column intervals and any phased fallback stay the source's"))


def _transfer_bound(differences, structure):
    from merlin.kernels.decoded_effects import _Protocol

    shapes = [shape for d in differences if d["effect"] == "operand_transfer_extent"
              for shape in (d.get("reference") or {}).get("shapes") or []]
    if not shapes or not structure:
        return dict(status="unknown", missing=["reference transfer extents or selected operand protocol"])
    proto = _Protocol(structure)
    limits = {}
    for (klass, name), binding in proto.bindings.items():
        if "operand_load" not in ((structure.get("instruction_roles") or {}).get(klass) or ()):
            continue
        for axis in ("rows", "cols"):
            if name == proto.extent_field(klass, axis):
                width = proto.widths.get((binding.get("bundle"), binding.get("field")))
                if type(width) is int:
                    limits.setdefault(axis, {})[klass] = (1 << width) - 1
    if set(limits) != {"rows", "cols"}:
        return dict(status="unknown", missing=["declared transfer-bound field widths for operand_load classes"])
    needed = dict(rows=max(s["rows"] for s in shapes), cols=max(s["cols"] for s in shapes))
    fits = {axis: max(limits[axis].values()) >= needed[axis] for axis in needed}
    return dict(status="satisfied" if all(fits.values()) else "violated",
                predicate="selected RTL transfer-bound field widths encode the reference's observed extents",
                requested=needed, encodable=limits,
                violated_bounds=[f"{axis}_field_width" for axis, ok in fits.items() if not ok],
                limits=["encodability only; the address law and source pitch remain separate requirements"])


def _admission(route, shape, geometry, differences, structure):
    name = route["admission"]
    if name is None:
        return dict(status="not_declared", capacity_predicate=None,
                    reason="the capability declares execution laws only; no capacity predicate exists")
    if name == "selected_transfer_extent_field":
        return _transfer_bound(differences, structure)
    if geometry is None:
        return dict(status="unknown", missing=["selected RTL geometry"])
    family = _family(shape)
    if family is None:
        return dict(status="unknown", predicate=name,
                    missing=["candidate shape: dense m,k,n or convolution kh,kw,c,oc (--candidate-shape)"])
    if family not in route["families"]:
        return dict(status="not_applicable", predicate=name,
                    reason=f"route consumes {route['families']}; the candidate shape is {family}")
    if name.endswith("dense_m_panel_capacity"):
        return _m_panel(shape, geometry)
    if name.endswith("dense_segmented_blocks"):
        return _segmented(shape, geometry)
    if name.endswith("plan_convolution_rhs_residency"):
        return _convolution(shape, geometry)
    return dict(status="unknown", missing=[f"no evaluator for declared predicate {name}"])


def _guarded_admission(route, shape, geometry, differences, structure):
    """A library predicate that raises on these inputs is reported, never mistaken for a decline."""
    try:
        return _admission(route, shape, geometry, differences, structure)
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError) as error:
        return dict(status="unknown", predicate=route["admission"],
                    missing=[f"predicate refused these inputs: {type(error).__name__}: {str(error)[:200]}"])


def applicability(reference, candidate, *, shape=None, geometry=None, geometry_reason=None, structure=None,
                  version=24):
    """Differences, routes per difference with admission, and missing capabilities. Never ranked."""
    from merlin.compile.scheduling.api_catalog import capability_effects

    differences = mechanism_differences(reference, candidate, structure=structure)
    routes = capability_effects(version)
    family = _family(shape)
    out, missing = [], []
    for row in differences:
        entry = dict(row)
        effect = row.get("route_effect") or row["effect"]
        if row["difference"] != "differs" or not row.get("needed_change") or row.get("beyond_reference"):
            entry["routes"] = []
            if row.get("beyond_reference"):
                entry["routes_basis"] = ("the candidate is already past the reference in the direction this "
                                         "effect's routes move it (larger means "
                                         + (row.get("larger_means") or "unknown")
                                         + "); none is sought, since a route would move it further away")
            if row["difference"] == "unknown":
                # Which routes would address this effect once both sides establish it; nothing is evaluated.
                entry["routes_if_established"] = [r["name"] for r in routes if effect in r["changes"]]
            out.append(entry)
            continue
        offered = [r for r in routes if r["changes"].get(effect) == row["needed_change"]]
        entry["routes"] = [dict(capability=r["name"], changes={effect: row["needed_change"]},
                                families=r["families"],
                                family_match=("unknown" if family is None else family in r["families"]),
                                symbols=r["symbols"], usage=r["usage"],
                                composition_and_scheduling_requirements=r["requires"],
                                admission=_guarded_admission(r, shape, geometry, differences, structure),
                                laws="required by the capability; not established by this report")
                           for r in offered]
        if not offered:
            entry["missing_capability"] = dict(effect=effect, needed_change=row["needed_change"],
                **({"role": row["role"]} if row.get("role") else {}),
                reason="no catalog capability declares changing this effect in this direction")
        elif family is not None and not any(family in r["families"] for r in offered):
            entry["missing_capability"] = dict(effect=effect, needed_change=row["needed_change"],
                family=family, reason="the capabilities that change this effect consume other source families")
        if "missing_capability" in entry:
            missing.append(entry["missing_capability"])
        out.append(entry)
    return dict(schema=SCHEMA, differences=out, missing_capabilities=missing, partial_recipe=partial_recipe(out),
                candidate_shape=dict(value=shape, family=family,
                                     source="author_declared" if shape else "not_supplied"),
                selected_geometry=geometry if geometry is not None else dict(status="unknown", reason=geometry_reason),
                catalog=dict(api="merlin.compile.scheduling.api_catalog:capability_effects", inventory_version=version),
                ordering="differences in fixed effect order; routes in catalog order; nothing is ranked",
                choice="selecting, composing and implementing a route is the author's work",
                semantic_equivalence="unknown", performance="unknown")
