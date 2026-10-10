"""The header-free hardware-counter bracket: every encoding value comes from RTL facts or contract data."""

from __future__ import annotations

import pytest

from merlin.perf import hw_counters as hc
from merlin.targetgen.contract import counter_bracket as CB

FACTS = {
    "facts": {
        "interfaces": [
            {
                "name": "funct_decode_table",
                "legal_funct": [0, 1, 9],
                "width": 7,
                "names": {"0": "CFG", "1": "MOVE", "9": "COUNT_OP"},
                "custom_opcode": 11,
            }
        ]
    }
}
SPEC = {
    "command": "COUNT_OP",
    "config_fields": {
        "reset": {"lsb": 0, "width": 1},
        "snapshot_take": {"lsb": 1, "width": 1},
        "configure": {"lsb": 2, "width": 1},
        "slot": {"lsb": 4, "width": 2},
        "event": {"lsb": 8, "width": 4},
        "external": {"lsb": 20, "width": 1},
    },
    "disabled_event": "OFF",
    "events": {
        "internal": {"OFF": 0, "U_A_CYCLES": 1, "U_B_CYCLES": 2, "U_A_B_CYCLES": 3},
        "external": {"IN_BYTES_X": 1},
    },
    "slot_capacity": {"module": "Ctr", "state_families": ["cfg", "val"]},
}


def test_the_command_resolves_its_code_and_opcode_from_facts():
    command = CB.command_from_data(SPEC, FACTS)
    assert (command.opcode, command.funct) == (11, 9)
    assert command.word(configure=1, slot=2, event=3) == (1 << 2) | (2 << 4) | (3 << 8)


def test_an_undecoded_command_or_overlapping_fields_refuse():
    with pytest.raises(CB.CounterBracketError, match="not exactly one RTL-decoded funct"):
        CB.command_from_data(dict(SPEC, command="ABSENT"), FACTS)
    bad = dict(SPEC, config_fields=dict(SPEC["config_fields"], slot={"lsb": 0, "width": 2}))
    with pytest.raises(CB.CounterBracketError, match="overlap"):
        CB.command_from_data(bad, FACTS)
    with pytest.raises(CB.CounterBracketError, match="declares no"):
        CB.command_from_data(None, FACTS)
    with pytest.raises(CB.CounterBracketError, match="cannot carry"):
        CB.command_from_data(SPEC, FACTS).word(slot=4)


def test_render_configures_reads_pads_and_flags_external_events():
    command = CB.command_from_data(SPEC, FACTS)
    codes = {"OFF": 0, "U_A_CYCLES": 1, "U_B_CYCLES": 2, "U_A_B_CYCLES": 3, "IN_BYTES_X": 1}
    out = CB.render(
        command,
        names=["U_A_CYCLES", "IN_BYTES_X"],
        codes={**codes, "IN_BYTES_X": 5},
        slots=3,
        schema_sha256="a" * 64,
        external={"IN_BYTES_X"},
    )
    assert ".insn r 0xb, 7, 0x9, %0, %1, x0" in out["helper"]
    pro = "\n".join(out["prologue"])
    assert f"MERLIN_COUNTER_SCHEMA {'a' * 64}" in pro
    assert f"{(1 << 2) | (0 << 4) | (1 << 8):#x}ULL" in pro  # slot 0, internal event 1
    assert f"{(1 << 2) | (1 << 4) | (5 << 8) | (1 << 20):#x}ULL" in pro  # slot 1, external flag set
    assert "padding: disabled event" in pro  # slot 2
    epi = "\n".join(out["epilogue"])
    assert "MERLIN_HWCOUNTER U_A_CYCLES" in epi and "MERLIN_HWCOUNTER IN_BYTES_X" in epi
    with pytest.raises(CB.CounterBracketError, match="need slots"):
        CB.render(command, names=["U_A_CYCLES", "U_B_CYCLES"], codes=codes, slots=1, schema_sha256="a" * 64)


def test_a_declared_counter_table_derives_the_occupancy_partition_and_its_digest():
    declared = hc.declared_counter_set("probe", contract={"logical_harness": {"counter_bracket": SPEC}})
    assert declared["status"] == "derived" and declared["source"] == "contract"
    assert declared["external"] == ["IN_BYTES_X"]
    occupancy, codes = hc.occupancy_for_discovery(declared)
    assert set(occupancy.engines) == {"A", "B"} and occupancy.complete()
    assert codes["IN_BYTES_X"] == 1
    assert declared["header_sha256"] == hc.counter_set_digest(SPEC["events"])
    readings = {"U_A_CYCLES": 10, "U_B_CYCLES": 20, "U_A_B_CYCLES": 5}
    eta = hc.eta_from_counters(readings, occupancy, exclusivity=hc.DECLARED_BY_PRODUCER, measurement_cycles=100)
    assert eta["state"] == "measured" and eta["busy_cycles"] == {"A": 15, "B": 25}


def test_counters_are_off_unless_requested(monkeypatch):
    monkeypatch.delenv(CB.COUNTERS_ENV, raising=False)
    assert CB.counters_requested() is False
    monkeypatch.setenv(CB.COUNTERS_ENV, "1")
    monkeypatch.setenv(CB.UNIT_ENV, "bytes")
    assert CB.counters_requested() is True and CB.unit_requested() == "BYTES"
