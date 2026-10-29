"""Bounded same-window modeled source lifetimes, independent of reservations.

No timing, qualification or physical utilization authority. Consumers must keep
command envelopes, service unions and finite-credit residency distinct.
"""
from . import contracts as C
from . import observations as O

SCHEMA = "merlin.scoped_source_actor_activity.v1"

def _count(value, name):
    if type(value) is not int or not 0 <= value < 2**64:
        raise C.ContractError(name + " must be an unsigned 64-bit integer")
    return value

def project(report):
    pins = {k: C.sha(report.get(k), k) for k in ("model_sha256", "elf_sha256", "scope_sha256")}
    pins["scope_binding_sha256"] = report.get("scope_binding_sha256")
    result = dict(schema="merlin.scoped_source_actor_feedback.v1", **pins,
        status="UNKNOWN", actors=[], overlaps=[], timing_qualification_issued=False,
        physical_occupancy=False, additive=False)
    raw = report.get("scoped_source_actor_activity")
    if raw is None:
        result["reason"] = "selected native observer did not capture source lifetimes"
        return result
    C.sha(pins["scope_binding_sha256"], "scope_binding_sha256")
    if raw.get("schema") != SCHEMA or raw.get("source_kind") != "modeled_source_actor" or raw.get("unit") != "model_cycles":
        raise C.ContractError("unrecognized source lifetime producer")
    if any(raw.get(k) != v for k, v in pins.items()):
        raise C.ContractError("source lifetimes belong to another model, ELF or original counter scope")
    observation = report.get("feature_observation") or {}
    windows = report.get("counter_windows") or []
    index = observation.get("window_index")
    window = windows[index] if type(index) is int and 0 <= index < len(windows) else None
    cycles = report.get("estimated_cycles")
    if not (O.feature_complete(report) and report.get("execution_completed") is True
        and report.get("timing_complete") is True and report.get("unresolved") == []
        and report.get("functional_status") == "PASS" and report.get("raw_functional_exit_code") == 0
        and type(cycles) is int and cycles > 0 and observation.get("scope_kind") == "counter_pc_pair"
        and window and window.get("estimated_cycles") == cycles and raw.get("complete") is True):
        result["reason"] = raw.get("unavailable_reason") or "complete original numerical/count/cycle scope and source capture required"
        return result
    begin, end = _count(raw.get("begin_cycle"), "source ROI begin"), _count(raw.get("end_cycle"), "source ROI end")
    if end - begin != cycles:
        raise C.ContractError("source lifetime interval differs from original modeled counter window")
    indexed = {}
    rows = raw.get("actors")
    if type(rows) is not list or not 1 <= len(rows) <= 64:
        raise C.ContractError("source actor roster missing or unbounded")
    for row in rows:
        name, meaning = row.get("name"), row.get("meaning")
        if type(name) is not str or not 0 < len(name) <= 128 or name in indexed or type(meaning) is not str or not 0 < len(meaning) <= 512:
            raise C.ContractError("invalid or duplicate source actor")
        if row.get("availability") != "modeled":
            if row.get("availability") != "UNKNOWN" or any(row.get(k) is not None for k in ("active_cycles", "resident_unit_cycles", "peak_units", "starts_in_scope")):
                raise C.ContractError("unavailable source actor must not claim zero activity")
            result["actors"].append(dict(name=name, meaning=meaning, availability="UNKNOWN", reason="selected route has no source lifetime producer"))
            indexed[name] = None
            continue
        active = _count(row.get("active_cycles"), "source active cycles")
        units = _count(row.get("resident_unit_cycles"), "source resident unit cycles")
        peak = _count(row.get("peak_units"), "source peak units")
        starts = _count(row.get("starts_in_scope"), "source starts")
        capacity = row.get("capacity")
        if capacity is not None:
            capacity = _count(capacity, "selected source capacity")
            if not capacity or peak > capacity:
                raise C.ContractError("source occupancy exceeds selected capacity")
        if active > cycles or not active <= units <= peak * active:
            raise C.ContractError("source lifetime arithmetic contradicts its original scope")
        projected = dict(name=name, meaning=meaning, availability="modeled", active_cycles=active,
            resident_unit_cycles=units, peak_units=peak, starts_in_scope=starts, capacity=capacity,
            active_fraction=active/cycles, average_resident_units=units/cycles,
            capacity_normalized_residency=units/(capacity*cycles) if capacity else None)
        indexed[name] = projected
        result["actors"].append(projected)
    pairs = raw.get("overlaps")
    if type(pairs) is not list or len(pairs) > 32:
        raise C.ContractError("source overlap roster unbounded")
    seen = set()
    for pair in pairs:
        left, right, active = pair.get("left"), pair.get("right"), pair.get("active_cycles")
        if left not in indexed or right not in indexed or left == right:
            raise C.ContractError("source overlap lacks original actor")
        key = tuple(sorted((left, right)))
        if key in seen:
            raise C.ContractError("duplicate source overlap")
        seen.add(key)
        if indexed[left] is None or indexed[right] is None:
            if active is not None:
                raise C.ContractError("unsupported actor overlap must remain UNKNOWN")
            result["overlaps"].append(dict(left=left, right=right, availability="UNKNOWN", active_cycles=None, fraction_of_window=None))
            continue
        active = _count(active, "source overlap")
        if active > min(indexed[left]["active_cycles"], indexed[right]["active_cycles"]):
            raise C.ContractError("source overlap exceeds original lifetime union")
        result["overlaps"].append(dict(left=left, right=right, availability="modeled", active_cycles=active, fraction_of_window=active/cycles))
    result.update(status="modeled", estimated_cycles_same_window=cycles, begin_cycle=begin, end_cycle=end,
        unknown=raw.get("not_available") or [], qualification=raw.get("qualification"))
    return result
