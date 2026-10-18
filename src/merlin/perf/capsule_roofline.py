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
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA",
    "compute_floor",
    "compute_peak_from_facts",
    "machine_bounds",
    "position",
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
