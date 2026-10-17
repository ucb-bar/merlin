"""A per-shape roofline whose stages stay separate: compute, movement and host.

:mod:`merlin.perf.derived_bound` answers "what rate could a legal tiling reach" with one
``max(compute, traffic)``. That is a valid floor, and it is the wrong thing to steer a Phase 2 search
with on its own, for three reasons this module keeps visible instead of folding away:

* **Padded versus useful arithmetic.** The array retires whole tiles. A shape whose extents are not
  tile multiples pays for the padding, so the compute stage reports useful MACs, padded MACs and both
  cycle counts. The attainable rate is quoted in USEFUL MACs per cycle.
* **Requested versus unique versus physical bytes.** *Unique* bytes are the compulsory operand and
  result volume. *Requested* bytes are what the least-traffic legal tiling must move, re-loads
  included. *Physical* bytes round every row segment of every request up to whole beats of the
  movement port, which is where non-aligned tails cost. Read and write are separate ports and are
  priced against their own peaks.
* **Serial stages are summed, never maxed.** Compute and movement may overlap only when the stores
  admit a second resident tile (``ping_pong_ways >= 2``, derived from the stores' banks); otherwise
  the accelerator stage is their sum. A host stage that feeds the accelerator (packing, im2col,
  command issue) runs before or between accelerator stages and is ADDED. Merlin-supplied host compute
  is outside the candidate's scope and is excluded from the floor altogether, by name.

Every unresolved input is listed. Dropping an unresolved serial term only lowers the floor, so the
floor stays a lower bound on cycles (``partial``) and says what it is missing. A measured count below
the floor refutes it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Any

from .decompose import is_unknown

__all__ = ["SCHEMA", "Contraction", "position", "staged_roofline"]

SCHEMA = "merlin.perf.staged_roofline.v1"


@dataclass(frozen=True)
class Contraction:
    """One contraction a capsule declares: ``M x K`` streamed against ``K x N`` held."""

    m: int
    k: int
    n: int
    #: Compulsory operand bytes when they are not the dense ``M*K`` / ``K*N`` matrices (a convolution
    #: reads its image, not its im2col matrix). ``None`` means dense.
    lhs_bytes: int | None = None
    rhs_bytes: int | None = None
    #: The weight is already resident from an earlier op of the same program (cross-op residency).
    rhs_resident: bool = False

    def __post_init__(self) -> None:
        for name in ("m", "k", "n"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"contraction extent {name} must be a positive int, got {value!r}")

    @property
    def macs(self) -> int:
        return self.m * self.k * self.n


def _round_up(value: int, quantum: int) -> int:
    return -(-int(value) // int(quantum)) * int(quantum)


def _segments(length: int, tile: int) -> list[int]:
    """Row-segment lengths (elements) of one row cut at ``tile``: full tiles plus the tail."""
    full, tail = divmod(int(length), int(tile))
    return [int(tile)] * full + ([tail] if tail else [])


def _physical(rows: int, length: int, tile: int, elem_bytes: int, beat: int, repeats: int) -> int:
    """Bytes the port moves for ``rows`` rows of ``length`` elements cut into ``tile`` segments."""
    per_row = sum(ceil(seg * elem_bytes / beat) * beat for seg in _segments(length, tile))
    return int(rows) * per_row * int(repeats)


def _compute(work: list[Contraction], machine) -> tuple[dict[str, Any], dict[str, str]]:
    unresolved: dict[str, str] = {}
    useful = sum(c.macs for c in work)
    out: dict[str, Any] = {"useful_macs": useful}
    peak, rows, cols = machine.peak_macs_per_cycle, machine.array_rows, machine.array_cols
    if is_unknown(peak) or is_unknown(rows) or is_unknown(cols):
        unresolved["compute"] = "the array geometry or its MAC idiom is not derived from the facts"
        out.update(padded_macs=None, useful_cycles=None, padded_cycles=None, padding_efficiency=None)
        return out, unresolved
    rows, cols, peak = int(rows), int(cols), int(peak)
    padded = sum(_round_up(c.m, rows) * _round_up(c.k, rows) * _round_up(c.n, cols) for c in work)
    out.update(
        peak_macs_per_cycle=peak,
        padded_macs=padded,
        useful_cycles=useful / peak,
        padded_cycles=padded / peak,
        padding_efficiency=useful / padded,
        basis=f"array {rows}x{cols} x {int(machine.muls_per_element)} mul(s); extents rounded to the array edge",
    )
    return out, unresolved


def _movement(
    work: list[Contraction], machine, ports: Mapping[str, Any], result_bytes: int
) -> tuple[dict[str, Any], dict[str, str]]:
    from .derived_bound import Gemm, select_tiling

    unresolved: dict[str, str] = {}
    op = machine.operand_bytes
    out: dict[str, Any] = {}
    if is_unknown(op):
        unresolved["movement"] = "the operand element width is not derived"
        return out, unresolved
    op = int(op)
    beats = {
        name: row.get("beat_bytes")
        for name, row in (ports.get("clients") or {}).items()
        if isinstance(row, Mapping) and row.get("beat_bytes")
    }
    by_direction: dict[str, int] = {}
    for row in (ports.get("clients") or {}).values():
        if isinstance(row, Mapping) and row.get("direction") and row.get("beat_bytes"):
            by_direction[row["direction"]] = max(by_direction.get(row["direction"], 0), int(row["beat_bytes"]))
    unique_read = unique_write = requested_read = requested_write = 0
    physical_read = physical_write = 0
    tilings = []
    for c in work:
        lhs = c.m * c.k * op if c.lhs_bytes is None else int(c.lhs_bytes)
        rhs = 0 if c.rhs_resident else (c.k * c.n * op if c.rhs_bytes is None else int(c.rhs_bytes))
        result = c.m * c.n * int(result_bytes)
        unique_read += lhs + rhs
        unique_write += result
        tiling = select_tiling(Gemm(c.m, c.n, c.k), machine, lhs_bytes=lhs, rhs_bytes=rhs, result_bytes=result)
        if isinstance(tiling, str):
            unresolved["movement"] = f"no legal tiling is derivable: {tiling}"
            return out, unresolved
        tilings.append(tiling.to_dict())
        requested_read += lhs * tiling.j_trips + rhs * tiling.i_trips
        requested_write += result
        read_beat, write_beat = by_direction.get("read"), by_direction.get("write")
        if read_beat and write_beat and c.lhs_bytes is None:
            physical_read += _physical(c.m, c.k, tiling.tile_k, op, read_beat, tiling.j_trips)
            if not c.rhs_resident and c.rhs_bytes is None:
                physical_read += _physical(c.k, c.n, tiling.tile_n, op, read_beat, tiling.i_trips)
            physical_write += _physical(c.m, c.n, tiling.tile_n, int(result_bytes), write_beat, 1)
        else:
            physical_read += lhs * tiling.j_trips + rhs * tiling.i_trips
            physical_write += result
    out.update(
        unique_bytes={"read": unique_read, "write": unique_write},
        requested_bytes={"read": requested_read, "write": requested_write},
        physical_bytes={"read": physical_read, "write": physical_write},
        tilings=tilings,
        port_beat_bytes=beats,
    )
    read_rate, write_rate = ports.get("read_bytes_per_cycle"), ports.get("write_bytes_per_cycle")
    if not read_rate or not write_rate:
        unresolved["movement_rate"] = "; ".join(
            f"{k}: {v}" for k, v in sorted((ports.get("unresolved") or {}).items())
        ) or "no movement port rate is derived"
        out["cycles"] = None
        return out, unresolved
    out["read_cycles"] = physical_read / float(read_rate)
    out["write_cycles"] = physical_write / float(write_rate)
    # Separate read and write ports run concurrently: the movement stage is the slower direction.
    out["cycles"] = max(out["read_cycles"], out["write_cycles"])
    out["rates_bytes_per_cycle"] = {"read": read_rate, "write": write_rate}
    return out, unresolved


def staged_roofline(
    work: list[Contraction],
    machine,
    ports: Mapping[str, Any],
    *,
    result_bytes: int,
    accelerator_host_cycles: float | None = None,
    merlin_host_operations: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The staged floor for ``work`` on ``machine`` with movement ``ports`` (:mod:`.movement_peak`).

    ``accelerator_host_cycles`` prices a host stage that feeds the accelerator (packing, im2col,
    command issue) when a model of it exists; ``None`` leaves it UNKNOWN and the floor partial.
    ``merlin_host_operations`` names the Merlin-supplied host compute this capsule's frozen placement
    keeps on the host: recorded, and excluded from the floor by construction.
    """
    if not work:
        raise ValueError("a staged roofline needs at least one declared contraction")
    if isinstance(result_bytes, bool) or not isinstance(result_bytes, int) or result_bytes < 1:
        raise ValueError("result_bytes must be the declared output element width in bytes")
    compute, unresolved = _compute(list(work), machine)
    movement, moved = _movement(list(work), machine, ports, result_bytes)
    unresolved.update(moved)
    ways = machine.ping_pong_ways
    overlap = (not is_unknown(ways)) and int(ways) >= 2
    host = {
        "accelerator_affecting_cycles": accelerator_host_cycles,
        "merlin_supplied_operations": dict(merlin_host_operations or {}),
        "merlin_supplied_scope": "excluded: compiled by Merlin's host package, never the candidate's work",
    }
    if accelerator_host_cycles is None:
        unresolved["accelerator_host_stage"] = (
            "no model prices the host stage that feeds the accelerator (packing, im2col, command issue); "
            "it is serial with the accelerator and the floor omits it"
        )
    compute_cycles = compute.get("padded_cycles")
    movement_cycles = movement.get("cycles")
    stages: list[tuple[str, float]] = []
    if compute_cycles is not None and movement_cycles is not None:
        accelerator = max(compute_cycles, movement_cycles) if overlap else compute_cycles + movement_cycles
        limiter = (
            ("compute" if compute_cycles >= movement_cycles else "movement")
            if overlap
            else "serial_compute_and_movement"
        )
        stages.append(("accelerator", accelerator))
    elif compute_cycles is not None:
        stages.append(("accelerator", compute_cycles))
        limiter = "compute"
    elif movement_cycles is not None:
        stages.append(("accelerator", movement_cycles))
        limiter = "movement"
    else:
        limiter = None
    if accelerator_host_cycles is not None:
        stages.append(("accelerator_affecting_host", float(accelerator_host_cycles)))
    floor = sum(value for _name, value in stages) if stages else None
    return {
        "schema": SCHEMA,
        "status": "derived" if floor is not None and not unresolved else ("partial" if floor is not None else "unknown"),
        "compute": compute,
        "movement": movement,
        "host": host,
        "composition": {
            "overlap_admissible": overlap,
            "basis": (
                f"the stores admit {ways} resident loop context(s), so compute and movement may overlap"
                if overlap
                else "the stores do not admit a second resident context, so compute and movement are serial"
            ),
            "serial_stages": [name for name, _value in stages],
        },
        "floor_cycles": floor,
        "limiter": limiter,
        "attainable_useful_macs_per_cycle": (compute["useful_macs"] / floor) if floor else None,
        "unresolved": unresolved,
    }


def position(document: Mapping[str, Any], cycles: Any) -> dict[str, Any]:
    """Where a measured count sits against the floor: ``attainment`` = floor / measured (1.0 is the floor)."""
    floor = document.get("floor_cycles")
    if isinstance(cycles, bool) or not isinstance(cycles, (int, float)) or cycles <= 0 or not floor:
        return {"status": "unknown", "attainment": None}
    if cycles < floor:
        return {
            "status": "refuted",
            "attainment": None,
            "reason": f"measured {cycles} cycles is below the derived floor {floor:.6g}",
        }
    return {"status": document.get("status"), "attainment": floor / float(cycles)}
