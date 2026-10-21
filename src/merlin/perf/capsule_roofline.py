"""Conservative workload floors from explicit RTL arithmetic and memory facts.

Compute charges useful MACs against the maximum structural multiplier capacity.
Every discovered array must state its rows, columns and multipliers per element;
geometry alone establishes neither a MAC rate nor stationary-row scheduling.
The bound assumes all those sites can issue simultaneously, so unavailable sites,
fill/drain, tails and instruction overhead can only make execution slower.

Movement charges declared interface reads/writes against the elaborated path's
widths. This term requires those bytes actually to cross that interface; logical
intermediates retained locally must not be charged as compulsory interface bytes.
The maximum of known terms is a lower bound, not a runtime prediction or an
assertion of compute/movement overlap. Missing facts remain UNKNOWN, and a measured
count below a floor refutes it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA",
    "compute_floor",
    "compose_service_floor",
    "compute_peak_from_facts",
    "machine_bounds",
    "position",
    "partial_roofline",
    "roofline",
]

SCHEMA = "merlin_capsule_roofline_v1"
#: Recorded on every document: what the floors deliberately do not charge (each omission lowers them).
OMITTED = ("array fill/drain delay", "fixed per-invocation cost")


def _ceil(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def compute_floor(m: int, k: int, n: int, *, peak_macs_per_cycle: int) -> dict[str, Any]:
    """Useful work divided by an explicitly qualified structural rate upper bound."""
    if any(_positive_int(value) is None for value in (m, k, n, peak_macs_per_cycle)):
        raise ValueError("positive contraction extents and a qualified MAC rate are required")
    macs = m * k * n
    return {"cycles": _ceil(macs, peak_macs_per_cycle), "macs": macs}


def compute_peak_from_facts(document: Mapping[str, Any]) -> tuple[int | None, str]:
    """Optimistic simultaneous issue capacity; never infer multipliers from geometry.

    Summing all discovered sites avoids restricting a generic target to the first
    array. An incomplete site's arithmetic count makes the rate unknown, because
    dropping that capacity could create a false lower bound.
    """
    body = document.get("facts", document)
    arrays = body.get("arrays") if isinstance(body, Mapping) else None
    if not isinstance(arrays, (list, tuple)) or not arrays:
        return None, "no complete arithmetic-site inventory was extracted"
    peak = 0
    for array in arrays:
        if not isinstance(array, Mapping):
            return None, "an extracted arithmetic site is not a mapping"
        idiom = array.get("mac_idiom")
        values = (array.get("rows"), array.get("cols"), idiom.get("muls") if isinstance(idiom, Mapping) else None)
        if any(_positive_int(value) is None for value in values):
            return None, "every array must explicitly state positive rows, columns and mac_idiom.muls"
        peak += math.prod(values)
    return peak, "sum of extracted rows * columns * explicit mac_idiom.muls; simultaneous issue upper bound"


def machine_bounds(
    target: str,
    *,
    facts: Mapping[str, Any] | None = None,
    elaboration: str | Path | None = None,
) -> dict[str, Any]:
    """``{"array_rows", "array_cols", "read_bytes_per_cycle", "write_bytes_per_cycle", "basis",
    "peak_macs_per_cycle", "unresolved"}`` for ``target`` -- explicit arithmetic facts
    and the memory path from the exact
    elaboration the timing engine was built from. Every value is derived or None with a reason."""
    from merlin.perf.decompose import is_unknown  # noqa: PLC0415
    from merlin.perf.derived_bound import machine_from_facts  # noqa: PLC0415

    unresolved: dict[str, str] = {}
    basis: dict[str, str] = {}
    if facts is None:
        from merlin.targetgen.rtl.facts import load_facts  # noqa: PLC0415

        facts = load_facts(target)
    machine = machine_from_facts(target, facts=facts, measure_fill=False)
    peak, peak_basis = compute_peak_from_facts(facts)
    if peak is None:
        unresolved["compute"] = peak_basis
    else:
        basis["compute"] = peak_basis
    rows, cols = machine.array_rows, machine.array_cols
    if is_unknown(rows) or is_unknown(cols):
        rows = cols = None
    read_width = write_width = None
    if elaboration is None:
        read_width, write_width = _facts_movement(facts, basis, unresolved)
    else:
        from merlin.targetgen.rtl import introspect  # noqa: PLC0415

        try:
            found = introspect.memory_path(Path(elaboration))
        except OSError as exc:
            found = {"status": "unknown", "reason": f"the elaboration is unreadable ({type(exc).__name__})"}
        if found.get("status") == "derived":
            read_width = _positive_int(found.get("read_bytes_per_cycle"))
            write_width = _positive_int(found.get("write_bytes_per_cycle"))
            basis["movement"] = str(found.get("evidence") or "elaborated memory path")
        if read_width is None or write_width is None:
            circuit = str(found.get("reason") or "the memory path widths are not positive")
            read_width, write_width = _facts_movement(facts, basis, unresolved)
            if read_width is None:
                unresolved["movement"] = f"{circuit}; {unresolved.get('movement', '')}"
    return {
        "array_rows": None if rows is None else int(rows),
        "array_cols": None if cols is None else int(cols),
        "peak_macs_per_cycle": peak,
        "read_bytes_per_cycle": read_width,
        "write_bytes_per_cycle": write_width,
        "basis": basis,
        "unresolved": unresolved,
    }


def _facts_movement(
    facts: Mapping[str, Any], basis: dict[str, str], unresolved: dict[str, str]
) -> tuple[int | None, int | None]:
    """Read/write beat peaks from the facts' own timing structure, when no circuit was walked."""
    from merlin.perf.movement_peak import movement_peak  # noqa: PLC0415

    ports = movement_peak(facts)
    read, write = ports.get("read_bytes_per_cycle"), ports.get("write_bytes_per_cycle")
    if ports["status"] != "derived" or read is None or write is None:
        unresolved["movement"] = "no elaborated circuit was walked and " + "; ".join(
            f"{key}: {value}" for key, value in sorted(ports["unresolved"].items())
        )
        return None, None
    basis["movement"] = (
        f"facts.timing_structure.memory_clients beat peaks (read {read:g}, write {write:g} byte(s)/cycle); "
        "the in-flight window is not applied without a fitted round trip"
    )
    return int(read), int(write)


def roofline(
    contractions: Sequence[Sequence[int]] | None,
    read_bytes: int | None,
    write_bytes: int | None,
    machine: Mapping[str, Any] | None,
    *,
    omitted: Sequence[str] = (),
) -> dict[str, Any]:
    """One workload's roofline document. ``roofline_cycles`` is None only when no term resolved.

    ``contractions`` are the declared ``(M, K, N)`` extents (None: the compute term is unresolved);
    ``read_bytes`` / ``write_bytes`` the compulsory bytes the workload declares (None: that direction
    is not charged, which lowers the floor and is recorded)."""
    machine = machine if isinstance(machine, Mapping) else {}
    unresolved = dict(machine.get("unresolved") or {})
    if not machine:
        unresolved.setdefault("machine", "no machine bounds were derived")
    peak = _positive_int(machine.get("peak_macs_per_cycle"))
    terms: dict[str, float] = {}
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "compute_floor_cycles": None,
        "movement_floor_cycles": None,
        "compulsory_read_bytes": read_bytes,
        "compulsory_write_bytes": write_bytes,
    }
    if contractions is None:
        unresolved.setdefault("compute", "the declared work has no derivable contraction extents")
    elif not contractions:
        doc["compute_floor_cycles"] = 0  # known absence of contraction work, not a timing term
    elif any(len(extents) != 3 or any(_positive_int(v) is None for v in extents) for extents in contractions):
        unresolved["compute"] = "contraction extents are not positive integer triples"
    elif peak is not None:
        # Concurrent independent contractions may share a cycle; round only once.
        work = sum(math.prod(extents) for extents in contractions)
        cycles = _ceil(work, peak)
        terms["compute"] = cycles
        doc["compute_floor_cycles"] = cycles
    else:
        unresolved.setdefault("compute", "no explicit arithmetic-site rate upper bound was derived")
    read_width, write_width = machine.get("read_bytes_per_cycle"), machine.get("write_bytes_per_cycle")
    charged = []
    if read_width and isinstance(read_bytes, int):
        charged.append(read_bytes / read_width)
    elif read_bytes is None:
        unresolved.setdefault("read_bytes", "the declared operands do not state their read volume")
    if write_width and isinstance(write_bytes, int):
        charged.append(write_bytes / write_width)
    elif write_bytes is None:
        unresolved.setdefault("write_bytes", "the declaration does not state its result volume")
    if charged and read_width and write_width:
        terms["movement"] = max(charged)
        doc["movement_floor_cycles"] = round(max(charged), 1)
    doc["omitted"] = list(omitted) + list(OMITTED)
    doc["unresolved"] = unresolved
    if terms:
        limiter = max(terms, key=lambda t: terms[t])
        doc.update(roofline_cycles=math.ceil(terms[limiter]), limiter=limiter, status="derived")
    else:
        doc.update(roofline_cycles=None, limiter=None, status="unknown")
    return doc


def position(doc: Mapping[str, Any], cycles: Any) -> float | None:
    """``cycles / roofline_cycles`` (>= 1 for any program the bound describes), or None."""
    bound = doc.get("roofline_cycles")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        return None
    if isinstance(bound, bool) or not isinstance(bound, int) or bound <= 0:
        return None
    return round(cycles / bound, 4)


def _request_ideal_service(read_bytes, write_bytes, machine):
    service = {}
    for direction, demand in (("read", read_bytes), ("write", write_bytes)):
        if demand is not None and (type(demand) is not int or not 0 <= demand < 2**64):
            raise ValueError("partial demand must be a bounded nonnegative integer or UNKNOWN")
        rate = _positive_int(machine.get(direction + "_bytes_per_cycle"))
        service[direction] = {
            "requested_bytes": demand,
            "client_peak_bytes_per_cycle": rate,
            "ideal_requested_service_cycles": (
                _service_number(Fraction(demand, rate)) if demand is not None and rate else None
            ),
        }
    return service


def partial_roofline(
    contractions,
    machine,
    *,
    logical_read_bytes=None,
    logical_write_bytes=None,
    padded_macs=None,
    padding_basis=None,
    store_requests=None,
    client_requests=None,
):
    """Keep known work/floors visible without pricing missing whole-call terms.

    Logical bytes supply only ideal *requested* client service. They are never
    passed to ``roofline`` as compulsory interface traffic. Padding and storage
    requests are explicit caller premises, not inferred schedules or physical
    ownership. Separately supplied client requests retain their own provenance;
    callers must reopen their original count owner. This projection issues
    neither measurements nor qualification.
    """
    machine = machine if isinstance(machine, Mapping) else {}
    work = None
    if contractions is not None:
        if not isinstance(contractions, (list, tuple)) or len(contractions) > 1024 or any(
            not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(_positive_int(x) is None for x in shape) for shape in contractions
        ):
            raise ValueError("partial roofline needs bounded positive contraction triples")
        work = sum(math.prod(shape) for shape in contractions)
        if work >= 2**64:
            raise ValueError("partial roofline work exceeds its arithmetic bound")
    if padded_macs is not None and (type(padded_macs) is not int or not 0 <= padded_macs < 2**64):
        raise ValueError("padded work must be a bounded nonnegative integer or UNKNOWN")
    if (padded_macs is None) != (padding_basis is None) or (
        padded_macs is not None and (
            not isinstance(padding_basis, str) or not padding_basis
            or (work is not None and padded_macs < work)
        )
    ):
        raise ValueError("padded work needs explicit provenance and cannot omit useful work")
    compute = roofline(contractions, None, None, machine)
    peak = _positive_int(machine.get("peak_macs_per_cycle"))
    movement = _request_ideal_service(logical_read_bytes, logical_write_bytes, machine)
    provided_requests = None
    if client_requests is not None:
        if (
            not isinstance(client_requests, Mapping)
            or set(client_requests) != {"read_bytes", "write_bytes", "provenance"}
            or not isinstance(client_requests["provenance"], str) or not client_requests["provenance"]
        ):
            raise ValueError("client requests need explicit directions and provenance")
        provided_requests = {
            "provenance": client_requests["provenance"],
            "service": _request_ideal_service(
                client_requests["read_bytes"], client_requests["write_bytes"], machine
            ),
        }
    if store_requests is None:
        store_requests = {}
    if not isinstance(store_requests, Mapping) or len(store_requests) > 1024:
        raise ValueError("partial storage needs a bounded explicit request inventory")
    storage = {}
    for name, row in store_requests.items():
        if (
            not isinstance(name, str) or not name or not isinstance(row, Mapping)
            or set(row) != {"capacity_bytes", "requested_bytes", "provenance"}
            or not isinstance(row["provenance"], str) or not row["provenance"]
            or (row["capacity_bytes"] is not None and (
                _positive_int(row["capacity_bytes"]) is None or row["capacity_bytes"] >= 2**64
            ))
            or (row["requested_bytes"] is not None and (
                type(row["requested_bytes"]) is not int or not 0 <= row["requested_bytes"] < 2**64
            ))
        ):
            raise ValueError("storage request needs explicit capacity, demand and provenance")
        capacity, requested = row["capacity_bytes"], row["requested_bytes"]
        known = capacity is not None and requested is not None
        storage[name] = {
            **dict(row),
            "pressure_ratio": _service_number(Fraction(requested, capacity)) if known else None,
            "conditional_request_fits": requested <= capacity if known else None,
            "physical_placement_proven": False,
        }
    known = any(x is not None for x in (work, logical_read_bytes, logical_write_bytes, padded_macs))
    return {
        "schema": "merlin.perf.partial_roofline.v1",
        "status": "partial" if known or storage or provided_requests else "unknown",
        "useful_macs": work,
        "structural_peak_macs_per_cycle": peak,
        "structural_compute_floor_cycles": compute["compute_floor_cycles"],
        "padded_work": {
            "macs": padded_macs,
            "basis": padding_basis,
            "conditional_compute_service_cycles": _ceil(padded_macs, peak)
            if padded_macs is not None and peak else None,
            "compulsory_workload_floor": False,
        },
        "logical_request_ideal_service": movement,
        "provided_client_request_ideal_service": provided_requests,
        "storage_requests": storage,
        "missing_terms": [
            "actual compulsory interface traffic and transfer amplification",
            "sustained bandwidth, contention, translation and arbitration",
            "command issue, host preparation, layout and readout costs",
            "array fill/drain and fixed invocation costs",
            "physical placement, ports, queues, dependencies and legal overlap",
            "independent complete-cost measurements and timing qualification",
        ],
        "machine_unresolved": {
            **dict(machine.get("unresolved") or {}),
            **({"compute": "no explicit complete arithmetic-site rate upper bound"} if peak is None else {}),
        },
        "whole_call_cycles": None,
        "complete_attainment": None,
        "convergence_eligible": False,
        "authority": "none; logical request ideals are not physical DRAM or sustained service",
    }


def _service_number(value):
    if value is None:
        return None
    if value.denominator == 1:
        return value.numerator
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("service arithmetic exceeds finite report bounds")
    return result


def compose_service_floor(resources, events, *, completion_chains=(), unresolved=None, scope="workload"):
    """Compose original-owner compulsory demands against structural rate ceilings.

    This is pure arithmetic, not a proof or qualification issuer. Resource rows
    supply ``unit``, ``capacity_per_cycle_upper`` (None means UNKNOWN), and
    ``provenance``. Events supply unique ``id``, ``demand`` by resource (None
    means UNKNOWN), ``compulsory`` and ``provenance``. Shared read/write service
    must use the *same* resource key. No port independence follows from names.

    A completion chain is an explicitly proved sequence of *nonoverlapping*
    event services, with ``events`` and ``provenance``. A data-availability edge
    or issue order is insufficient. Individual event services use max, shared
    resource totals use sum, proven chains use sum, and the final bound uses max.
    No candidate schedule waste can become a fixed workload floor: callers must
    explicitly select ``emitted_schedule`` for noncompulsory dynamic requests.
    """
    if scope not in {"workload", "emitted_schedule"} or not isinstance(resources, Mapping):
        raise ValueError("select explicit workload or emitted-schedule service scope")
    if len(resources) > 1024 or not isinstance(events, (list, tuple)) or len(events) > 100000:
        raise ValueError("service-floor inventory exceeds bounded limits")
    missing = dict(unresolved or {})
    rates, totals, provenance = {}, {}, {}
    for key, row in resources.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(row, Mapping)
            or set(row) != {"unit", "capacity_per_cycle_upper", "provenance"}
            or not isinstance(row["unit"], str)
            or not row["unit"]
            or not isinstance(row["provenance"], str)
            or not row["provenance"]
        ):
            raise ValueError("resource requires an explicit unit, structural rate ceiling and provenance")
        rate = row["capacity_per_cycle_upper"]
        if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0):
            raise ValueError("resource rate must be finite positive or UNKNOWN")
        rates[key], totals[key], provenance[key] = (None if rate is None else Fraction(rate)), 0, dict(row)
        if rate is None:
            missing["resource:" + key] = "structural service-rate upper bound unavailable"
    durations = {}
    for event in events:
        if (
            not isinstance(event, Mapping)
            or set(event) != {"id", "demand", "compulsory", "provenance"}
            or not isinstance(event["id"], str)
            or not event["id"]
            or event["id"] in durations
            or not isinstance(event["demand"], Mapping)
            or not event["demand"]
            or type(event["compulsory"]) is not bool
            or (scope == "workload" and event["compulsory"] is not True)
            or not isinstance(event["provenance"], str)
            or not event["provenance"]
        ):
            raise ValueError("event needs unique identity, explicit demand/provenance and correct compulsory scope")
        services = []
        for resource, units in event["demand"].items():
            if resource not in rates:
                raise ValueError("event names an undeclared physical service resource")
            if units is None:
                missing["event:" + event["id"] + ":" + resource] = "compulsory demand unavailable"
                continue
            if type(units) is not int or not 0 <= units < 2**64:
                raise ValueError("service demand must be bounded nonnegative integer units or UNKNOWN")
            totals[resource] += units
            if rates[resource] is not None:
                services.append(units / rates[resource])
        durations[event["id"]] = max(services) if services else None
    terms = {"resource:" + key: units / rates[key] for key, units in totals.items() if rates[key] is not None}
    if not isinstance(completion_chains, (list, tuple)) or len(completion_chains) > 10000:
        raise ValueError("completion chains require a bounded explicit inventory")
    chains = []
    for chain in completion_chains:
        if (
            not isinstance(chain, Mapping)
            or set(chain) != {"events", "provenance"}
            or not isinstance(chain["events"], (list, tuple))
            or not chain["events"]
            or any(not isinstance(key, str) or key not in durations for key in chain["events"])
            or len(set(chain["events"])) != len(chain["events"])
            or not isinstance(chain["provenance"], str)
            or not chain["provenance"]
        ):
            raise ValueError("serial addition needs distinct known events and explicit completion proof provenance")
        cycles = sum(durations[key] for key in chain["events"] if durations[key] is not None)
        terms["completion_chain:" + str(len(chains))] = cycles
        chains.append({**dict(chain), "cycles": _service_number(cycles)})
    positive = {key: value for key, value in terms.items() if value > 0}
    limiter = max(positive, key=positive.get) if positive else None
    return {
        "schema": "merlin.perf.resource_service_floor.v1",
        "scope": scope,
        "floor_cycles": math.ceil(positive[limiter]) if limiter else None,
        "limiter": limiter,
        "status": "partial" if positive and missing else "derived" if positive else "unknown",
        "resource_demands": totals,
        "resource_provenance": provenance,
        "event_service_floor_cycles": {key: _service_number(value) for key, value in durations.items()},
        "completion_chains": chains,
        "terms": {key: _service_number(value) for key, value in terms.items()},
        "unresolved": missing,
        "convergence_eligible": False,
        "authority": "none; original owner must establish compulsory effects, rates, scope and measurements",
        "composition": "max of necessary resource totals and explicitly proved serial completion paths",
    }
