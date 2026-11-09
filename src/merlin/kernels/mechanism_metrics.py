"""Compare static occurrence mechanisms and retain evidence status."""

from __future__ import annotations

from fractions import Fraction

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
    prefetch_compute_ahead=("a larger fraction of loads with compute issued before their first reader", "increases"),
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
        word, change = ("lower", "increases") if candidate < reference else ("higher", "reduces")
    fields = dict(
        direction="candidate_matches" if change is None else "candidate_differs",
        comparison=f"reference {_plain(reference)}, candidate {_plain(candidate)}, candidate {word}",
        needed_change=change,
        beyond_reference=None if change is None or more_when is None else change != more_when,
    )
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
    return (
        (record.get("derived_effects") or {}).get("evidence_id")
        or (record.get("evidence") or {}).get("evidence_id")
        or record.get("stable_id")
        or "occurrence"
    )


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
        effect=name,
        status=status,
        reference=ref,
        candidate=cand,
        needed_change=_direction(ref and ref["value"], cand and cand["value"]) if status != "unknown" else None,
        basis=f"static {role}-role sites per accumulate-role site over the selected role roster",
        limits=["listed instruction sites, not dynamic issue counts; loop trip counts are unknown"]
        if status == "inferred"
        else [],
        evidence=[
            f"{ref_id}#schedule_model.role_counts",
            f"{cand_id}#schedule_model.role_counts",
            f"{ref_id}#cfg.natural_loops",
            f"{cand_id}#cfg.natural_loops",
        ],
    )


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
    classes = {
        klass: rows
        for klass, rows in (value.get("by_class") or {}).items()
        if rows and "operand_load" in (roles.get(klass) or ())
    }
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
        out.append(
            dict(
                effect="operand_transfer_extent",
                operand=key,
                status=status,
                **(
                    dict(
                        reason=f"no resolved operand_load transfer extent for {key} on the "
                        + " and ".join(unresolved)
                        + " side (register values not statically known, or the class is absent there)"
                    )
                    if unresolved
                    else {}
                ),
                reference=dict(largest_elements=ref_max, shapes=ref_shapes),
                candidate=dict(largest_elements=cand_max, shapes=cand_shapes),
                needed_change=_direction(ref_max, cand_max) if known else None,
                basis="decoded rows x cols of operand_load-role transfers per instruction class "
                "(declared transfer-bound fields; static sites, not issued transfers)",
                limits=[] if status == "observed" else ["some transfer extents are unresolved"],
                evidence=pointer,
            )
        )
    return out


def _broadcast(effects):
    fact = _fact(effects, "broadcast_loads")
    if fact.get("status") == "unknown":
        return None, None
    value = fact.get("value") or {}
    states = sorted(set(value.get("nonzero_by_state") or {}) | set(value.get("stride_zero_by_state") or {}))
    return sum((value.get("broadcast_movement_sites") or {}).values()), states


def _broadcasts(reference, candidate, ref_id, cand_id):
    (ref_sites, ref_states), (cand_sites, cand_states) = (
        _broadcast(r.get("derived_effects")) for r in (reference, candidate)
    )
    known = ref_sites is not None and cand_sites is not None
    pointer = [f"{i}#derived_effects.facts.broadcast_loads" for i in (ref_id, cand_id)]
    rows = [
        dict(
            effect="broadcast_load_sites",
            status="observed" if known else "unknown",
            reference=ref_sites,
            candidate=cand_sites,
            needed_change=_direction(ref_sites, cand_sites) if known else None,
            basis="movement sites issued under a stride-zero load configuration (static listing order)",
            limits=["a stride inherited from before the occurrence stays unknown"],
            evidence=pointer,
        )
    ]
    known = ref_states is not None and cand_states is not None
    rows.append(
        dict(
            effect="configured_load_states",
            status="observed" if known else "unknown",
            reference=ref_states,
            candidate=cand_states,
            needed_change=_direction(len(ref_states), len(cand_states)) if known else None,
            basis="distinct declared load-state ports given a stride configuration",
            limits=[],
            evidence=pointer,
        )
    )
    return rows


def _count(name, role, reference, candidate, ref_id, cand_id):
    ref, cand = _roles(reference).get(role, 0), _roles(candidate).get(role, 0)
    return dict(
        effect=name,
        status="observed",
        reference=ref,
        candidate=cand,
        needed_change=_direction(ref, cand),
        basis=f"static {role}-role sites in the selected span",
        limits=[],
        evidence=[f"{ref_id}#schedule_model.role_counts", f"{cand_id}#schedule_model.role_counts"],
    )


def _config(reference, candidate, ref_id, cand_id):
    def rate(record):
        roles = _roles(record)
        return _ratio(roles.get("config", 0), sum(roles.get(r, 0) for r in ("operand_load", "readout")))

    ref, cand = rate(reference), rate(candidate)
    status = _rate_status(reference, candidate) if ref and cand else "unknown"
    return dict(
        effect="config_per_transfer",
        status=status,
        reference=ref,
        candidate=cand,
        needed_change=_direction(ref and ref["value"], cand and cand["value"]) if status != "unknown" else None,
        basis="static config-role sites per operand_load/readout-role site",
        limits=[],
        evidence=[f"{ref_id}#schedule_model.role_counts", f"{cand_id}#schedule_model.role_counts"],
    )


def _loop(reference, candidate, ref_id, cand_id):
    def sites(record):
        fact = _fact(record.get("derived_effects"), "hardware_loop")
        return None if fact.get("status") == "unknown" else (fact.get("value") or {}).get("loop_descriptor_sites")

    ref, cand = sites(reference), sites(candidate)
    known = ref is not None and cand is not None
    return dict(
        effect="loop_descriptor_sites",
        status="observed" if known else "unknown",
        reference=ref,
        candidate=cand,
        needed_change=_direction(ref, cand) if known else None,
        basis="sites whose selected role is loop_descriptor (dispatch structure)",
        limits=[],
        evidence=[f"{i}#derived_effects.facts.hardware_loop" for i in (ref_id, cand_id)],
    )


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
        rows.append(
            dict(
                effect=name,
                status=status,
                reference_status=ref.get("status"),
                candidate_status=cand.get("status"),
                reason=ref.get("reason") or cand.get("reason") or (ref.get("limits") or [None])[0],
                needed_change=None,
                evidence=[f"{i}#derived_effects.facts.{name}" for i in (ref_id, cand_id)],
            )
        )
    return rows


ROLE_FACTS = ("role_transfer_width", "operand_buffering", "issue_overlap", "prefetch_distance")


STREAM_FACTS = ROLE_FACTS + ("output_group_extent",)


STREAM_SCOPE = "proved_dynamic_command_stream"


NO_STREAM = "no_proved_command_stream"


_SEPARATE = ("preload", "preloads", "array_preload")  # kept apart from data-movement roles by the contract


def stream_fact(effects, name):
    """One stream fact, or an explicit unknown when no proved command stream produced it."""
    facts = (effects or {}).get("facts") or {}
    fact = facts.get(name)
    if fact is None and any(
        isinstance(facts.get(other), dict) and facts[other].get("status") not in (None, "unknown")
        for other in ROLE_FACTS
    ):
        return dict(
            status="unknown",
            scope=STREAM_SCOPE,
            reason=f"{name} absent from a record whose other stream facts were derived "
            "(derived before this fact existed; re-derive the record)",
        )
    if not isinstance(fact, dict) or fact.get("status") in (None, "unknown"):
        return dict(status="unknown", reason=(fact or {}).get("reason") or NO_STREAM, scope=STREAM_SCOPE)
    return fact


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


# Preserve the original public lookup and pickle identities.
for _definition in (
    _plain,
    _order,
    _ratio,
    _fact,
    _evidence_id,
    _loops,
    _rate_status,
    _roles,
    _direction,
    _role_rate,
    _extent,
    _transfer,
    _broadcast,
    _broadcasts,
    _count,
    _config,
    _loop,
    _unknowns,
    stream_fact,
    _summary_value,
    _position,
):
    _definition.__module__ = "merlin.kernels.mechanism_applicability"
del _definition
