"""Actual ordinary same-HW issuers expose conditional data without admission."""

import copy
import dataclasses
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import component_packing_sources as C
from merlin_experiments.phase0 import packing_intake as P
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake
from test_packing_memory_intake import selection

from merlin.targetgen.rtl.source_selection import produce_selection


@pytest.fixture(scope="module", params=["legacy", "modern"])
def native(request, tmp_path_factory):
    suffix = "_MODERN" if request.param == "modern" else ""
    keys = ("MERLIN_TEST_FIRTOOL" + suffix, "MERLIN_TEST_CIRCT_OPT" + suffix)
    if any(not os.environ.get(key) for key in keys):
        pytest.skip("packing intake controls require two explicitly selected coherent native CIRCT pairs")
    firtool, circt_opt = (Path(os.environ[key]).resolve(strict=True) for key in keys)
    root = tmp_path_factory.mktemp("packing-memory-" + request.param)
    source = root / "independent.fir"
    text = (
        "FIRRTL version 2.0.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/IndependentMemory.scala 1:1]\n"
        "    input clock : Clock\n    input index : UInt<2>\n    input consent : UInt<1>\n"
        "    input word : UInt<16>\n    output seen : UInt<16>\n"
    )
    for index in range(2):
        text += (
            f"    mem m{index} :\n      data-type => UInt<8>\n      depth => 3\n"
            "      read-latency => 0\n      write-latency => 1\n      reader => r\n      writer => w\n"
            "      read-under-write => undefined\n"
            f"    m{index}.r.addr <= index\n    m{index}.r.en <= consent\n    m{index}.r.clk <= clock\n"
            f"    m{index}.w.addr <= index\n    m{index}.w.en <= consent\n    m{index}.w.clk <= clock\n"
            f"    m{index}.w.data <= bits(word, {index * 8 + 7}, {index * 8})\n"
            f"    m{index}.w.mask <= consent\n"
        )
    source.write_text(text + "    seen <= cat(m1.r.data, m0.r.data)\n")
    bundle = produce_selection(
        target="test_unit",
        firrtl=source,
        generator="test_unit",
        config="IndependentMemoryPacking",
        core_root="Unit",
        firtool=firtool,
        output=root / "production",
    )
    descriptor = root / "descriptor.yaml"
    descriptor.write_text("target: test_unit\n")
    forbidden = (root / "excluded-answers",)
    hardware = issue_independent_hardware_intake(
        target="test_unit",
        descriptor=descriptor,
        source_bundle=bundle,
        forbidden_roots=forbidden,
        output=root / "hardware",
    )
    arguments = {"hardware": hardware, "circt_opt": circt_opt, "forbidden_roots": forbidden}
    old = P.issue_independent_packing_intake(**arguments, output=root / "packing-v1")
    new = P.issue_independent_packing_intake(**arguments, output=root / "packing-v2", memory_selection=selection())
    return old, new, root


def test_actual_same_source_v2_consumer_keeps_v1_partitions_and_every_memory_port(native):
    old, new, _ = native
    first, second = old.record(), new.record()
    assert first["schema"] == P.SCHEMA and second["schema"] == P.MEMORY_SCHEMA
    assert "memory_bindings" not in first
    assert first["facts"] == second["facts"] == second["memory_bindings"]["local_partitions"]
    assert old.hardware is new.hardware
    assert first["hardware_intake_sha256"] == second["hardware_intake_sha256"]
    facts = second["memory_bindings"]
    assert len(facts["ports"]) == 4
    writes = [row for row in facts["ports"] if row["data_status"] != "no_write_data_operand"]
    assert len(writes) == 2 and all(sum(piece["width"] for piece in row["intervals"]) == 8 for row in writes)
    assert {piece["source_low_bit"] for row in writes for piece in row["intervals"]} == {0, 8}
    assert all(memory["declaration"]["read_under_write"] == "undefined" for memory in facts["hierarchy"]["memories"])
    assert facts["packing_mapping_admission"] is facts["command_capacity_axis_or_temporal_admission"] is False
    assert facts["source_values_evaluated"] is False
    assert new.public_facts()["memory_bindings"] == facts
    assert first["unknowns"] == list(P._UNKNOWN) and second["unknowns"] == list(P._MEMORY_UNKNOWN)


def test_actual_v2_data_does_not_change_existing_conditional_source_or_missing_mapping_roster(native):
    old, new, _ = native
    rows = []
    for intake in (old, new):
        obligations, unknowns = [], []
        C.append_sources(
            facts=intake.record()["facts"],
            spec={"numerical_semantics": {"operand_dtype": "int8"}},
            movement_owners={"movement"},
            owner_links={"movement": []},
            program=lambda family: {},
            unknown=lambda kind, selector, reason: {"kind": kind, "selector": selector, "reason": reason},
            obligations=obligations,
            unknowns=unknowns,
        )
        rows.append((obligations, unknowns))
    assert rows[0] == rows[1]
    assert [row["kind"] for row in rows[1][1]] == ["packing_mapping", "packing_domain"]
    assert all(row["mandatory"] is True for row in rows[1][0])


@pytest.mark.parametrize(
    "change",
    [
        "missing_port",
        "false_as_zero",
        "interval",
        "unknown",
        "source_owner",
        "original_container",
        "budget",
        "new_facts_in_v1",
    ],
)
def test_actual_original_source_owner_types_budgets_and_complete_membership_cannot_be_substituted(native, change):
    old, new, _ = native
    record = copy.deepcopy(new.record())
    if change == "missing_port":
        record["memory_bindings"]["ports"].pop()
    elif change == "false_as_zero":
        record["memory_bindings"]["source_values_evaluated"] = 0
    elif change == "interval":
        next(row for row in record["memory_bindings"]["ports"] if row["intervals"])["intervals"][0][
            "source_low_bit"
        ] += 1
    elif change == "unknown":
        record["unknowns"].pop()
    elif change == "source_owner":
        record["hardware_intake_sha256"] = "0" * 64
    elif change == "original_container":
        next(row for row in record["source_pins"] if row["role"] == "original-core-hw")["path"] = record["invocation"]
    elif change == "budget":
        record["memory_binding_selection"]["binding_limits"]["operations"] = 1
    else:
        record = copy.deepcopy(old.record())
        record["memory_bindings"] = new.record()["memory_bindings"]
    with pytest.raises(ValueError):
        P.verify_record(record)


def test_actual_exported_record_cannot_recreate_live_same_hardware_authority(native):
    _, new, _ = native
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        dataclasses.replace(new).verify()
