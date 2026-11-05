"""The memory-movement peak of an accelerator, read from its elaborated timing structure.

The roofline's movement term needs a rate. :mod:`merlin.perf.derived_bound` accepts a transfer
width from a datapath or interface block, and no target's facts published one, so the term stayed
UNKNOWN on every target. The facts extractor's ``timing_structure.memory_clients`` section does
publish the movement engine's own ports (``merlin.targetgen.rtl.timing_structure``): per memory
client, the beat width, the channel count, the in-flight transaction slots selected by the response
ID, and the largest request the size field can carry. This module turns those into:

* a **beat peak** per client: ``beat_bytes x beats_per_cycle``. A synchronous ready/valid channel
  retires at most one beat per cycle, so ``beats_per_cycle`` is the client's channel count;
* an **in-flight bound**: ``in_flight x max_request_bytes`` bytes may be outstanding, so by Little's
  law the sustained rate is at most that volume divided by the memory round trip. The round trip is
  NOT an RTL fact (the extractor lists it under ``fit_from_measurement``), so without a calibrated
  latency the in-flight limit is reported as the latency above which it binds, never as a rate.

Direction is read structurally, never from a client's name: a client whose RESPONSE carries data
reads memory, and one whose REQUEST carries data with a data-less response writes it.

Everything absent is UNKNOWN with a reason. A peak is an upper bound on the movement rate; it never
describes contention, translation misses or arbitration, which remain measured quantities.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["READ", "WRITE", "SCHEMA", "movement_peak"]

SCHEMA = "merlin.perf.movement_peak.v1"
READ = "read"
WRITE = "write"


def _positive(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _body(document: Mapping[str, Any]) -> Mapping[str, Any]:
    body = document.get("facts") if isinstance(document, Mapping) and "facts" in document else document
    return body if isinstance(body, Mapping) else {}


def _direction(client: Mapping[str, Any]) -> str | None:
    request = _positive(client.get("request_data_bits"))
    response = _positive(client.get("response_data_bits"))
    if response is not None:
        return READ
    if request is not None and client.get("response_data_bits") is None:
        return WRITE
    return None


def _client(name: str, client: Mapping[str, Any], latency: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "direction": _direction(client), "unresolved": {}}
    beat = _positive(client.get("beat_bytes"))
    channels = _positive(client.get("request_channels"))
    in_flight_block = client.get("in_flight")
    in_flight = (
        _positive(in_flight_block.get("value"))
        if isinstance(in_flight_block, Mapping) and in_flight_block.get("status") == "derived"
        else None
    )
    largest = _positive(client.get("max_request_bytes"))
    if row["direction"] is None:
        row["unresolved"]["direction"] = "neither the request nor the response states a data width"
    if beat is None or channels is None:
        row["unresolved"]["peak"] = "the client states no beat width or channel count"
        row["peak_bytes_per_cycle"] = None
    else:
        row["beat_bytes"] = beat
        row["beats_per_cycle"] = channels
        row["peak_bytes_per_cycle"] = beat * channels
    row["in_flight"] = in_flight
    row["max_request_bytes"] = largest
    if in_flight is None or largest is None:
        row["unresolved"]["in_flight"] = "the in-flight slots or the largest request size are not derived"
        row["in_flight_bytes"] = None
        row["binding_latency_cycles"] = None
    else:
        row["in_flight_bytes"] = in_flight * largest
        peak = row["peak_bytes_per_cycle"]
        # Above this round trip the outstanding-request window, not the beat width, limits the rate.
        row["binding_latency_cycles"] = (row["in_flight_bytes"] / peak) if peak else None
    peak = row["peak_bytes_per_cycle"]
    if peak is None:
        row["sustained_bytes_per_cycle"] = None
        row["sustained_basis"] = "no beat peak"
    elif latency is None:
        row["sustained_bytes_per_cycle"] = float(peak)
        row["sustained_basis"] = (
            "beat peak only: the memory round trip is a calibration fit, so the in-flight window is "
            "reported as the latency above which it binds, not applied"
        )
    elif row["in_flight_bytes"] is None:
        row["sustained_bytes_per_cycle"] = float(peak)
        row["sustained_basis"] = "beat peak; the in-flight window is not derived"
    else:
        windowed = row["in_flight_bytes"] / float(latency)
        row["sustained_bytes_per_cycle"] = min(float(peak), windowed)
        row["sustained_basis"] = (
            f"min(beat peak {peak}, in-flight {row['in_flight_bytes']} bytes / round trip {latency:g} cycles)"
        )
    return row


def movement_peak(document: Mapping[str, Any], *, round_trip_latency_cycles: float | None = None) -> dict[str, Any]:
    """Per-direction movement peaks from ``timing_structure.memory_clients``, or UNKNOWN with why.

    ``round_trip_latency_cycles`` is a fitted memory round trip (a calibration product). Supplied, the
    in-flight window caps each client's sustained rate; omitted, only the beat peak applies.
    """
    if round_trip_latency_cycles is not None and (
        isinstance(round_trip_latency_cycles, bool)
        or not isinstance(round_trip_latency_cycles, (int, float))
        or round_trip_latency_cycles <= 0
    ):
        raise ValueError("a round-trip latency must be a positive number of cycles")
    structure = _body(document).get("timing_structure")
    clients_block = structure.get("memory_clients") if isinstance(structure, Mapping) else None
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "unknown",
        "clients": {},
        "read_bytes_per_cycle": None,
        "write_bytes_per_cycle": None,
        "total_bytes_per_cycle": None,
        "round_trip_latency_cycles": round_trip_latency_cycles,
        "unresolved": {},
        "fit_from_measurement": list((structure or {}).get("fit_from_measurement") or [])
        if isinstance(structure, Mapping)
        else [],
    }
    if not isinstance(clients_block, Mapping) or clients_block.get("status") != "derived":
        out["unresolved"]["memory_clients"] = (
            "the facts carry no derived timing_structure.memory_clients section, so no movement port is known"
        )
        return out
    clients = clients_block.get("clients")
    if not isinstance(clients, Mapping) or not clients:
        out["unresolved"]["memory_clients"] = "the derived memory-client section names no client"
        return out
    totals: dict[str, float] = {}
    for name in sorted(clients):
        client = clients[name]
        if not isinstance(client, Mapping):
            out["unresolved"][str(name)] = "the client record is not a mapping"
            continue
        row = _client(str(name), client, round_trip_latency_cycles)
        out["clients"][str(name)] = row
        if row["direction"] is not None and row["sustained_bytes_per_cycle"] is not None:
            totals[row["direction"]] = totals.get(row["direction"], 0.0) + row["sustained_bytes_per_cycle"]
        else:
            out["unresolved"][str(name)] = "; ".join(f"{k}: {v}" for k, v in row["unresolved"].items())
    out["read_bytes_per_cycle"] = totals.get(READ)
    out["write_bytes_per_cycle"] = totals.get(WRITE)
    if READ in totals and WRITE in totals:
        # Separate read and write clients may run concurrently, so total traffic over their summed
        # rates is a lower bound on movement time: an upper bound on the achievable rate.
        out["total_bytes_per_cycle"] = totals[READ] + totals[WRITE]
        out["status"] = "derived"
    else:
        missing = sorted({READ, WRITE} - set(totals))
        out["unresolved"]["direction"] = f"no resolved client moves data in direction(s) {missing}"
    return out
