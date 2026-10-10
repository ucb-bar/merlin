"""A per-workload ROOFLINE: the fewest cycles any schedule of declared work can take on this machine.

    machine = machine_bounds(target, facts=facts, elaboration=fir)   # array geometry + memory path
    doc = roofline(contractions, read_bytes, write_bytes, machine)   # {"roofline_cycles", "limiter", ...}
    position(doc, cycles)                                            # measured / roofline, or None

Both floors are DERIVED, never configured, and both are about the MACHINE and the WORKLOAD -- not
about any implementation -- so the same bound applies to every program that computes the workload:

* **Compute.** A contraction ``M x K x N`` on an ``R x C`` array is cut into stationary blocks of at
  most ``R`` deep by ``C`` wide; every streamed row passes every block once, one row per cycle, and a
  block enters the array one row per cycle, which at best overlaps the previous block's stream. So a
  block costs at least ``max(streamed rows, block depth)`` cycles and the floor is the sum over
  blocks, minimised over which operand is held (``C = A B`` or ``C^T = B^T A^T``).
* **Movement.** Every byte the workload declares as read must come in over the accelerator's memory
  path and every byte it declares as written must go out. Read and write are separate channels, so
  the floor is ``max(read bytes / read width, write bytes / write width)``. The widths come from the
  elaborated circuit itself (:func:`merlin.targetgen.rtl.introspect.memory_path`), never from a
  configuration name.

The roofline is the larger floor, and ``limiter`` names which one it is. Omissions (fill/drain delay,
fixed invocation cost) only LOWER the bound, so it stays a floor, and they are recorded. A term whose
input is UNKNOWN is dropped and named in ``unresolved``; a workload with neither term has no roofline,
never a zero one. A measurement below the roofline REFUTES it (``status: refuted``): a bound a real
program beats means an input it was derived from does not describe the machine.

Nothing here names a target, an instruction, an opcode or a size.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA",
    "compute_floor",
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


def compute_floor(m: int, k: int, n: int, *, array_rows: int, array_cols: int) -> dict[str, Any]:
    """The array's floor for ``M x K x N`` (see the module docstring), minimised over orientation."""

    def oriented(streamed: int, held: int) -> tuple[int, int]:
        depths = [min(array_rows, k - start) for start in range(0, k, array_rows)]
        blocks_wide = _ceil(held, array_cols)
        cycles = blocks_wide * sum(max(streamed, depth) for depth in depths)
        computes = _ceil(streamed, array_rows) * len(depths) * blocks_wide
        return cycles, computes

    options = {"stream_m": oriented(m, n), "stream_n": oriented(n, m)}
    best = min(options, key=lambda o: options[o][0])
    return {
        "cycles": options[best][0],
        "min_computes": min(c for _, c in options.values()),
        "orientation": best,
        "options": {o: {"cycles": c, "computes": u} for o, (c, u) in options.items()},
    }


def machine_bounds(
    target: str,
    *,
    facts: Mapping[str, Any] | None = None,
    elaboration: str | Path | None = None,
) -> dict[str, Any]:
    """``{"array_rows", "array_cols", "read_bytes_per_cycle", "write_bytes_per_cycle", "basis",
    "unresolved"}`` for ``target`` -- the geometry from its RTL facts, the memory path from the exact
    elaboration the timing engine was built from. Every value is derived or None with a reason."""
    from merlin.perf.decompose import is_unknown  # noqa: PLC0415
    from merlin.perf.derived_bound import machine_from_facts  # noqa: PLC0415

    unresolved: dict[str, str] = {}
    basis: dict[str, str] = {}
    machine = machine_from_facts(target, facts=facts, measure_fill=False)
    rows, cols = machine.array_rows, machine.array_cols
    if is_unknown(rows) or is_unknown(cols):
        unresolved["compute"] = machine.refusals.get("array_rows") or "the array geometry is UNKNOWN"
        rows = cols = None
    else:
        basis["compute"] = machine.provenance.get("array_rows", "") or "facts.arrays"
    read_width = write_width = None
    if elaboration is None:
        unresolved["movement"] = "no elaborated circuit was named, so the memory path is not derivable"
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
            read_width = write_width = None
            unresolved["movement"] = str(found.get("reason") or "the memory path widths are not positive")
    return {
        "array_rows": None if rows is None else int(rows),
        "array_cols": None if cols is None else int(cols),
        "read_bytes_per_cycle": read_width,
        "write_bytes_per_cycle": write_width,
        "basis": basis,
        "unresolved": unresolved,
    }


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
    rows, cols = machine.get("array_rows"), machine.get("array_cols")
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
    elif rows and cols:
        cycles = 0
        for extents in contractions:
            m, k, n = (int(v) for v in extents)
            cycles += compute_floor(m, k, n, array_rows=int(rows), array_cols=int(cols))["cycles"]
        terms["compute"] = float(cycles)
        doc["compute_floor_cycles"] = cycles
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
