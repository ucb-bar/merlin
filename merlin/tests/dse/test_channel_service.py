"""Client handshakes compose without inventing downstream memory semantics."""

from copy import deepcopy

import pytest

from merlin.perf.channel_service import CONTRACT_SCHEMA, compose_transfer_service


def _contract(width):
    resources = {
        name: {"unit": "handshake", "capacity_per_cycle_upper": 1, "provenance": "checked independent client port"}
        for name in ("input.request", "input.response", "output.request", "output.response")
    }
    return {
        "schema": CONTRACT_SCHEMA,
        "resources": resources,
        "transfer_kinds": {
            "input": {
                "data_beat_bytes": width,
                "request": {"resource": "input.request", "units_per_packet": 1},
                "response": {"resource": "input.response", "units_per_data_beat": 1},
            },
            "output": {
                "data_beat_bytes": width,
                "request": {"resource": "output.request", "units_per_data_beat": 1},
                "response": {"resource": "output.response", "units_per_packet": 1},
            },
        },
        "unresolved": {"downstream": "shared downstream arbitration has no selected proof"},
        "provenance": "original adapter derives channels and packet role semantics",
    }


def _transfer(name, kind, *packets):
    return {
        "id": name,
        "kind": kind,
        "provenance": "complete original source envelope",
        "packets": [
            {"payload_bytes": payload, "packet_bytes": envelope, "data_beats": beats}
            for payload, envelope, beats in packets
        ],
    }


def test_two_geometries_keep_client_envelopes_separate_from_payload_and_physical_bytes():
    wide = compose_transfer_service(
        _contract(16),
        [
            _transfer("unaligned-tail", "input", (17, 32, 2), (1, 16, 1)),
            _transfer("readout", "output", (33, 64, 4)),
        ],
    )
    narrow = compose_transfer_service(
        _contract(8),
        [
            _transfer("unaligned-tail", "input", (17, 24, 3), (1, 8, 1)),
            _transfer("readout", "output", (33, 40, 5)),
        ],
    )
    assert wide["transfer_totals"]["input"] == {
        "packets": 2,
        "payload_bytes": 18,
        "client_envelope_bytes": 48,
        "data_beats": 3,
    }
    assert narrow["transfer_totals"]["input"]["payload_bytes"] == 18
    assert wide["floor"]["floor_cycles"] == 4
    assert narrow["floor"]["floor_cycles"] == 5
    assert wide["floor"]["resource_demands"] == {
        "input.request": 2,
        "input.response": 3,
        "output.request": 4,
        "output.response": 1,
    }
    assert wide["physical_traffic_bytes"] is None and wide["measured_cycles"] is None
    assert wide["attainment"] is None and wide["convergence_eligible"] is False
    assert wide["scope"] == "emitted_schedule" and wide["completion_chains"] == []
    assert all(event["compulsory"] is False for event in wide["events"])


def test_only_explicit_resource_identity_adds_cross_direction_service():
    contract = _contract(16)
    transfers = [_transfer("in", "input", (32, 32, 2)), _transfer("out", "output", (64, 64, 4))]
    assert compose_transfer_service(contract, transfers)["floor"]["floor_cycles"] == 4
    contract["transfer_kinds"]["output"]["request"]["resource"] = "input.response"
    result = compose_transfer_service(contract, transfers)
    assert result["floor"]["floor_cycles"] == 6
    assert result["floor"]["resource_demands"]["input.response"] == 6
    assert "downstream" in result["unresolved"]
    assert result["floor"]["status"] == "partial"


def test_shared_request_response_roles_add_without_inferring_completion_chain():
    contract = _contract(16)
    contract["transfer_kinds"]["input"]["response"]["resource"] = "input.request"
    result = compose_transfer_service(contract, [_transfer("in", "input", (32, 32, 2))])
    assert result["events"][0]["demand"] == {"input.request": 3}
    assert result["floor"]["floor_cycles"] == 3
    assert result["completion_chains"] == []


def test_unknown_service_rate_retains_known_demand_and_never_means_free_service():
    contract = _contract(16)
    contract["resources"]["output.request"]["capacity_per_cycle_upper"] = None
    result = compose_transfer_service(
        contract,
        [
            _transfer("in", "input", (32, 32, 2)),
            _transfer("out", "output", (64, 64, 4)),
        ],
    )
    assert result["floor"]["resource_demands"]["output.request"] == 4
    assert "resource:output.request" in result["floor"]["unresolved"]
    assert result["floor"]["floor_cycles"] == 2
    assert result["floor"]["status"] == "partial"


@pytest.mark.parametrize(
    "packet",
    [
        (17, 16, 1),
        (16, 32, 1),
        (0, 16, 1),
        (True, 16, 1),
        (16, 16, 0),
        (16, 16, None),
        (16, -16, 1),
        (16, 16, 2**64),
    ],
)
def test_missing_or_contradictory_packet_effects_refuse(packet):
    with pytest.raises(ValueError, match="packet needs"):
        compose_transfer_service(_contract(16), [_transfer("bad", "input", packet)])


def test_no_implicit_kind_or_packetization_and_duplicate_events_refuse():
    with pytest.raises(ValueError, match="selected kind"):
        compose_transfer_service(_contract(16), [_transfer("in", "undeclared", (16, 16, 1))])
    transfer = _transfer("in", "input", (16, 16, 1))
    with pytest.raises(ValueError, match="unique identity"):
        compose_transfer_service(_contract(16), [transfer, deepcopy(transfer)])
    contract = _contract(16)
    contract["transfer_kinds"]["input"]["request"]["units_per_data_beat"] = 1
    with pytest.raises(ValueError, match="demand law"):
        compose_transfer_service(contract, [transfer])


def test_aggregate_overflow_and_inventory_limits_refuse_before_publication():
    contract = _contract(1)
    transfer = _transfer("in", "input", (2**63, 2**63, 2**63), (2**63, 2**63, 2**63))
    with pytest.raises(ValueError, match="unsigned report bounds"):
        compose_transfer_service(contract, [transfer])
    with pytest.raises(ValueError, match="admission"):
        compose_transfer_service(contract, [None] * 100001)


def test_empty_complete_inventory_does_not_invent_zero_latency_or_mutate_inputs():
    contract = _contract(16)
    before = deepcopy(contract)
    result = compose_transfer_service(contract, [])
    assert result["floor"]["floor_cycles"] is None
    assert result["measured_cycles"] is None
    result["resources"]["input.request"]["capacity_per_cycle_upper"] = 99
    assert contract == before
