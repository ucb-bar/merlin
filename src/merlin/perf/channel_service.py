"""Compose explicit client-channel demands, without inferring a memory model.

The original adapter owns packetization, channel roles and resource ceilings.
This pure consumer cannot turn client envelopes into observed physical traffic,
completed requests, a workload objective floor or a runtime prediction.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from merlin.perf.capsule_roofline import compose_service_floor

CONTRACT_SCHEMA = "merlin.perf.channel_service_contract.v1"
SCHEMA = "merlin.perf.channel_service.v1"
_LIMIT = 100000
_U64 = 2**64


def _text(value):
    return isinstance(value, str) and 0 < len(value) <= 4096


def _unsigned(value, *, positive=False):
    return type(value) is int and int(positive) <= value < _U64


def _bounded_sum(previous, value):
    result = previous + value
    if not _unsigned(result):
        raise ValueError("channel demand exceeds unsigned report bounds")
    return result


def _contract(contract):
    if (
        not isinstance(contract, Mapping)
        or set(contract) != {"schema", "resources", "transfer_kinds", "unresolved", "provenance"}
        or contract["schema"] != CONTRACT_SCHEMA
        or not _text(contract["provenance"])
        or not isinstance(contract["unresolved"], Mapping)
        or len(contract["unresolved"]) > 1024
        or any(not _text(key) or not _text(value) for key, value in contract["unresolved"].items())
    ):
        raise ValueError("explicit bounded channel contract and provenance required")
    resources, kinds = contract["resources"], contract["transfer_kinds"]
    # Reuse the existing resource grammar, rate validation and UNKNOWN handling.
    compose_service_floor(resources, [], scope="emitted_schedule")
    if not isinstance(kinds, Mapping) or not 1 <= len(kinds) <= 1024:
        raise ValueError("channel transfer kinds require an explicit bounded inventory")
    for kind, row in kinds.items():
        if (
            not _text(kind)
            or not isinstance(row, Mapping)
            or set(row) != {"data_beat_bytes", "request", "response"}
            or not _unsigned(row["data_beat_bytes"], positive=True)
        ):
            raise ValueError("transfer kind needs an explicit data beat and channel roles")
        for role in ("request", "response"):
            channel = row[role]
            if (
                not isinstance(channel, Mapping)
                or set(channel) not in ({"resource", "units_per_packet"}, {"resource", "units_per_data_beat"})
                or not isinstance(channel["resource"], str)
                or channel["resource"] not in resources
            ):
                raise ValueError("channel role must name one resource and explicit demand law")
            factor = channel.get("units_per_packet", channel.get("units_per_data_beat"))
            if not _unsigned(factor, positive=True):
                raise ValueError("channel demand factor must be an explicit positive integer")
    return resources, kinds


def compose_transfer_service(contract, transfers):
    """Count source-derived client envelopes and compose a partial schedule floor.

    Each transfer supplies ``id``, ``kind``, ``packets`` and ``provenance``.
    Packets supply positive ``payload_bytes``, ``packet_bytes`` and ``data_beats``.
    The selected contract gives each kind's data-beat width and each channel's
    resource and units-per-packet OR units-per-data-beat. Full-beat envelopes
    must match that width exactly; this consumer never invents packet rounding.

    Resource identity determines sharing. Demands on the same resource add;
    different resources combine with max, without asserting achievable overlap.
    Schedule requests stay noncompulsory. No completion chain follows from a
    request/response role, a source ID count, or instruction order.
    """
    resources, kinds = _contract(contract)
    if not isinstance(transfers, (list, tuple)) or len(transfers) > _LIMIT:
        raise ValueError("channel transfer inventory exceeds explicit admission")
    events, totals, seen = [], {}, set()
    packets_seen = 0
    for transfer in transfers:
        if (
            not isinstance(transfer, Mapping)
            or set(transfer) != {"id", "kind", "packets", "provenance"}
            or not _text(transfer["id"])
            or transfer["id"] in seen
            or not _text(transfer["provenance"])
            or not isinstance(transfer["kind"], str)
            or transfer["kind"] not in kinds
            or not isinstance(transfer["packets"], (list, tuple))
            or len(transfer["packets"]) > _LIMIT - packets_seen
        ):
            raise ValueError("transfer requires unique identity, selected kind and complete bounded packets")
        seen.add(transfer["id"])
        kind, packets = kinds[transfer["kind"]], transfer["packets"]
        packets_seen += len(packets)
        counts = {"packets": len(packets), "payload_bytes": 0, "client_envelope_bytes": 0, "data_beats": 0}
        for packet in packets:
            if (
                not isinstance(packet, Mapping)
                or set(packet) != {"payload_bytes", "packet_bytes", "data_beats"}
                or any(not _unsigned(value, positive=True) for value in packet.values())
                or packet["payload_bytes"] > packet["packet_bytes"]
                or packet["packet_bytes"] != packet["data_beats"] * kind["data_beat_bytes"]
            ):
                raise ValueError("packet needs exact positive payload and complete selected-width data beats")
            for destination, source in (
                ("payload_bytes", "payload_bytes"),
                ("client_envelope_bytes", "packet_bytes"),
                ("data_beats", "data_beats"),
            ):
                counts[destination] = _bounded_sum(counts[destination], packet[source])
        aggregate = totals.setdefault(transfer["kind"], dict.fromkeys(counts, 0))
        for key, value in counts.items():
            aggregate[key] = _bounded_sum(aggregate[key], value)
        demand = {}
        for role in ("request", "response"):
            channel = kind[role]
            units = (
                counts["packets"] * channel["units_per_packet"]
                if "units_per_packet" in channel
                else counts["data_beats"] * channel["units_per_data_beat"]
            )
            resource = channel["resource"]
            demand[resource] = _bounded_sum(demand.get(resource, 0), units)
        events.append(
            {
                "id": transfer["id"],
                "demand": demand,
                "compulsory": False,
                "provenance": transfer["provenance"],
            }
        )
    unresolved = {
        **dict(contract["unresolved"]),
        "fixed_workload": "candidate packet requests are schedule demand, not invariant compulsory workload traffic",
        "completion_overlap": "channel handshake ceilings alone establish neither completion latency nor overlap",
        "physical_traffic": "client envelopes do not observe retries, translation, caches or downstream memory traffic",
    }
    floor = compose_service_floor(resources, events, scope="emitted_schedule", unresolved=unresolved)
    # compose_service_floor accepts individual u64 demands; enforce aggregate
    # reporting bounds here too, including aliases/shared request-response ports.
    if any(not _unsigned(value) for value in floor["resource_demands"].values()):
        raise ValueError("aggregate shared-channel demand exceeds unsigned report bounds")
    return {
        "schema": SCHEMA,
        "scope": "emitted_schedule",
        "resources": deepcopy(dict(resources)),
        "events": events,
        "completion_chains": [],
        "unresolved": unresolved,
        "transfer_totals": totals,
        "floor": floor,
        "provenance": contract["provenance"],
        "physical_traffic_bytes": None,
        "measured_cycles": None,
        "attainment": None,
        "convergence_eligible": False,
        "authority": "none; original adapter owns packet/effect/source qualification",
    }
