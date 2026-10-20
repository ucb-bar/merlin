"""Conservative staged floors with separate hypothetical schedule diagnostics.

V2 corrects V1's unsupported bank-count overlap, compulsory tile-padding and
requested-byte physical-traffic assumptions. Useful work / structural maximum
throughput is a floor. A selected tiling and beat-rounded row segments describe
one hypothetical schedule only: neither is a workload minimum. Movement needs
an explicit original-owner compulsory interface-crossing declaration.

Taking the maximum of valid known resource floors is safe with unknown overlap:
every execution must pay each term. It does not assert achievable overlap.
Serial addition needs independently proved shared service or completion edges;
use capsule_roofline.compose_service_floor for those explicit proofs. Unpriced
host/packing and unknown physical effects prevent convergence. Saved V1 evidence
retains its original interpretation and is never rewritten by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any

from .decompose import is_unknown

__all__ = ["SCHEMA", "Contraction", "position", "staged_roofline"]

SCHEMA = "merlin.perf.staged_roofline.v2"


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

        if (
            any(
                value is not None and (type(value) is not int or value < 0)
                for value in (self.lhs_bytes, self.rhs_bytes)
            )
            or type(self.rhs_resident) is not bool
        ):
            raise ValueError("explicit operand volumes and residency must be unambiguous")

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
    work: list[Contraction], machine, ports: Mapping[str, Any], result_bytes: int, compulsory_transfer_bytes
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
    beat_rounding_known = True
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
            beat_rounding_known = False
    out.update(
        unique_bytes={"read": unique_read, "write": unique_write},
        requested_bytes={"read": requested_read, "write": requested_write},
        physical_bytes={"read": None, "write": None},
        hypothetical_beat_rounded_bytes={
            "read": physical_read if beat_rounding_known else None,
            "write": physical_write if beat_rounding_known else None,
        },
        schedule_scope="one analytical tiling; no optimality or compulsory physical traffic proof",
        tilings=tilings,
        port_beat_bytes=beats,
    )
    rates = {name: ports.get(name + "_bytes_per_cycle") for name in ("read", "write")}
    out["rates_bytes_per_cycle"] = rates
    out["cycles"] = None
    unresolved["physical_traffic"] = "request geometry and port width do not prove physical transactions"
    if compulsory_transfer_bytes is None:
        unresolved["movement"] = "no original-owner compulsory interface-crossing declaration"
        return out, unresolved
    if (
        not isinstance(compulsory_transfer_bytes, Mapping)
        or set(compulsory_transfer_bytes) != {"read", "write", "provenance"}
        or not isinstance(compulsory_transfer_bytes["provenance"], str)
        or not compulsory_transfer_bytes["provenance"]
        or any(
            type(compulsory_transfer_bytes[name]) is not int or compulsory_transfer_bytes[name] < 0
            for name in ("read", "write")
        )
    ):
        raise ValueError("compulsory movement requires nonnegative bytes and explicit crossing provenance")
    out["compulsory_interface_bytes"] = dict(compulsory_transfer_bytes)
    terms = []
    for name, rate in rates.items():
        if type(rate) not in (int, float) or not isfinite(rate) or rate <= 0:
            unresolved[name + "_rate"] = "finite positive structural interface rate unavailable"
            continue
        cycles = compulsory_transfer_bytes[name] / rate
        out[name + "_cycles"] = cycles
        terms.append(cycles)
    out["cycles"] = max(terms) if terms else None
    return out, unresolved


def staged_roofline(
    work: list[Contraction],
    machine,
    ports: Mapping[str, Any],
    *,
    result_bytes: int,
    accelerator_host_cycles: float | None = None,
    merlin_host_operations: Mapping[str, int] | None = None,
    compulsory_transfer_bytes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Useful-work floor; explicit crossing declarations may add movement floors.

    ``accelerator_host_cycles`` is retained as an unqualified diagnostic estimate,
    never silently a serial lower bound. Host compute remains separately named.
    The original caller owns numerical/placement/crossing proofs; pure arithmetic
    and nonempty provenance strings issue no qualification.
    """
    if not work:
        raise ValueError("a staged roofline needs at least one declared contraction")
    if isinstance(result_bytes, bool) or not isinstance(result_bytes, int) or result_bytes < 1:
        raise ValueError("result_bytes must be the declared output element width in bytes")
    compute, unresolved = _compute(list(work), machine)
    movement, moved = _movement(list(work), machine, ports, result_bytes, compulsory_transfer_bytes)
    unresolved.update(moved)
    host = {
        "accelerator_affecting_cycles": accelerator_host_cycles,
        "cycles_interpretation": "unqualified estimate; not a compulsory serial service floor",
        "merlin_supplied_operations": dict(merlin_host_operations or {}),
        "merlin_supplied_scope": "reported separately; complete-call host demand still requires its owner",
    }
    unresolved["accelerator_host_stage"] = "complete-call packing, layout, issue and host service floors unavailable"
    unresolved["overlap"] = "banks and resident contexts establish neither overlap nor serialization"
    compute_cycles = compute.get("useful_cycles")
    movement_cycles = movement.get("cycles")
    terms = {
        name: value for name, value in (("compute", compute_cycles), ("movement", movement_cycles)) if value is not None
    }
    limiter = max(terms, key=terms.get) if terms else None
    floor = terms[limiter] if limiter else None
    return {
        "schema": SCHEMA,
        "status": "derived"
        if floor is not None and not unresolved
        else ("partial" if floor is not None else "unknown"),
        "compute": compute,
        "movement": movement,
        "host": host,
        "composition": {
            "overlap_admissible": None,
            "basis": "maximum of known necessary services; no overlap or serialization claim",
            "serial_stages": [],
            "bank_count_is_overlap_proof": False,
        },
        "convergence_eligible": False,
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
    if not isfinite(cycles):
        return {"status": "unknown", "attainment": None}
    if cycles < floor:
        return {
            "status": "refuted",
            "attainment": None,
            "reason": f"measured {cycles} cycles is below the derived floor {floor:.6g}",
        }
    return {"status": document.get("status"), "attainment": floor / float(cycles)}
