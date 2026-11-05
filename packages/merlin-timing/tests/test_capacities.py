"""RTL structural facts reach a device profile only through their declared role (synthetic data)."""

import hashlib
import json

import pytest
from merlin_timing import capacities as K
from merlin_timing import contracts as C
from merlin_timing.cli import main

PROFILE = {
    "schema": "synthetic.device.v1",
    "resources": {
        "unit.issue": {"capacity": None, "queue_capacity": None},
        "unit.transfer": {"capacity": None, "queue_capacity": None},
    },
    "target": {
        "hardware": {"width": 4, "reset_mode": None},
        "prices": {"move": {"latency_cycles": None}, "compute": {"latency_cycles": None}},
    },
}


def _fact(identity, role, applies_to, value, unit):
    return {
        "id": identity,
        "role": role,
        "applies_to": applies_to,
        "value": value,
        "unit": unit,
        "sources": [{"path": "/rtl/generated.fir", "sha256": C.digest(identity), "selector": "module.entries"}],
        "derivation": "counted instantiated entries in the selected elaboration",
    }


def _document(*facts, profile=PROFILE):
    return {
        "schema": K.SCHEMA,
        "device_profile_sha256": C.digest(profile),
        "elaboration": {"firrtl": C.digest("firrtl"), "dts": C.digest("dts")},
        "facts": list(facts),
    }


def test_facts_land_where_their_role_says_and_keep_their_receipt():
    document = _document(
        _fact("issue-lanes", "occupancy_lanes", "unit.issue", 1, "commands"),
        _fact("issue-queue", "start_queue_entries", "unit.issue", 2, "commands"),
        _fact("credits", "live_command_credits", "unit.issue", 8, "commands"),
        _fact("queue", "command_queue_entries", "unit.transfer", 2, "commands"),
        _fact("inflight", "outstanding_requests", "unit.transfer", 16, "requests"),
        _fact("burst", "max_transfer_bytes", "unit.transfer", 64, "bytes"),
        _fact("beat", "datapath_bytes_per_beat", "unit.transfer", 16, "bytes"),
        _fact("pipeline", "minimum_latency_cycles", "compute", 5, "cycles"),
        _fact("reset", "declared_hardware_fact", "reset_mode", 0, "enumeration"),
    )
    profile = K.apply(PROFILE, document, profile_sha256=C.digest(PROFILE))
    assert profile["resources"]["unit.issue"] == {"capacity": 1, "queue_capacity": 2}
    assert profile["resources"]["unit.transfer"] == {"capacity": None, "queue_capacity": None}
    assert profile["target"]["structure"] == {
        "live_command_credits": {"unit.issue": 8},
        "command_queue_entries": {"unit.transfer": 2},
        "outstanding_requests": {"unit.transfer": 16},
        "max_transfer_bytes": {"unit.transfer": 64},
        "datapath_bytes_per_beat": {"unit.transfer": 16},
    }
    assert profile["target"]["constraints"] == {"compute": {"minimum_latency_cycles": 5}}
    assert profile["target"]["hardware"]["reset_mode"] == 0
    assert all(fields["latency_cycles"] is None for fields in profile["target"]["prices"].values())
    receipt = profile["capacity_facts"]
    assert receipt["sha256"] == C.digest(document) and len(receipt["applied"]) == 9
    assert PROFILE["resources"]["unit.issue"]["capacity"] is None  # the input profile is not mutated
    # Only fully known resources become engine resources; the transfer unit stays unbound.
    assert K.model_resources(profile) == [{"name": "unit.issue", "capacity": 1, "queue_capacity": 2}]


@pytest.mark.parametrize(
    ("fact", "match"),
    [
        (_fact("x", "reservation_entries", "unit.issue", 8, "commands"), "unknown capacity role"),
        (_fact("x", "occupancy_lanes", "unit.issue", 8, "entries"), "measured in commands"),
        (_fact("x", "occupancy_lanes", "unit.issue", 0, "commands"), "positive integer"),
        (_fact("x", "occupancy_lanes", "unit.unknown", 1, "commands"), "declares no capacity"),
        (_fact("x", "minimum_latency_cycles", "unpriced", 3, "cycles"), "no price class"),
        (_fact("x", "declared_hardware_fact", "undeclared", 1, "count"), "no hardware fact"),
        (_fact("x", "declared_hardware_fact", "width", 8, "lanes"), "already 4"),
    ],
)
def test_misplaced_or_unsupported_facts_refuse(fact, match):
    with pytest.raises(C.ContractError, match=match):
        K.apply(PROFILE, _document(fact), profile_sha256=C.digest(PROFILE))


def test_facts_bind_to_the_exact_profile_and_cite_sources():
    document = _document(_fact("lanes", "occupancy_lanes", "unit.issue", 1, "commands"))
    with pytest.raises(C.ContractError, match="different device profile"):
        K.apply(PROFILE, document, profile_sha256=C.digest("another profile"))
    document["facts"][0]["sources"] = []
    with pytest.raises(C.ContractError, match="pinned sources"):
        K.facts_document(document)
    duplicate = _document(
        _fact("a", "occupancy_lanes", "unit.issue", 1, "commands"),
        _fact("b", "occupancy_lanes", "unit.issue", 2, "commands"),
    )
    with pytest.raises(C.ContractError, match="stated once"):
        K.facts_document(duplicate)


def test_cli_places_facts_into_a_new_profile_without_overwriting(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(PROFILE))
    digest = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    facts = _document(_fact("lanes", "start_queue_entries", "unit.transfer", 4, "commands"))
    facts["device_profile_sha256"] = digest
    facts_path = tmp_path / "facts.json"
    C.write(facts_path, facts)
    out = tmp_path / "loaded.json"
    args = ["apply-capacities", "--profile", str(profile_path), "--facts", str(facts_path), "--output", str(out)]
    assert main(args) == 0
    loaded = C.read(out)
    assert loaded["resources"]["unit.transfer"]["queue_capacity"] == 4
    assert loaded["capacity_facts"]["device_profile_sha256"] == digest
    assert main(args) == 2
