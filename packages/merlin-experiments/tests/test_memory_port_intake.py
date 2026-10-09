"""Native minimal memory sources expose exact ports, never temporal or axis credit."""

import copy
import dataclasses
import json
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import memory_port_intake as M
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake

from merlin.common import invocation_record as I
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits, memory_port_observations
from merlin.targetgen.rtl.source_selection import produce_selection

LIMITS = MemoryPortLimits(32, 32, 64, 1024, 256, 32768, 64)


@pytest.fixture
def tools():
    names = ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_CIRCT_OPT")
    if any(not os.environ.get(name) for name in names):
        pytest.skip("memory port controls require an explicitly selected coherent native tool pair")
    return {name: Path(os.environ[name]).absolute() for name in names}


def _firrtl(kind):
    source = "FIRRTL version 2.0.0\ncircuit Unit :\n"
    if kind == "opaque":
        source += "  extmodule Other :\n    output signal : UInt<16>\n    defname = Other\n"
    source += (
        "  module Unit : @[generators/test_unit/src/MinimalMemory.scala 1:1]\n"
        "    input clock : Clock\n    input a : UInt<2>\n    input b : UInt<2>\n"
        "    input consent : UInt<1>\n    input mask0 : UInt<1>\n    input mask1 : UInt<1>\n"
        "    input arbitrary : UInt<16>\n    output observed : UInt<16>\n"
    )
    if kind == "state":
        source += "    reg word : UInt<16>, clock\n    word <= arbitrary\n"
    elif kind == "opaque":
        source += "    inst other of Other\n    node word = other.signal\n"
    else:
        source += "    node word = arbitrary\n"
    for index in range(2):
        source += (
            f"    mem lane{index} :\n      data-type => UInt<8>\n      depth => 3\n"
            "      read-latency => 1\n      write-latency => 1\n"
            "      reader => r\n      writer => w\n      read-under-write => undefined\n"
            f"    lane{index}.r.addr <= a\n    lane{index}.r.en <= UInt<1>(1)\n"
            f"    lane{index}.r.clk <= clock\n    lane{index}.w.addr <= b\n"
            f"    lane{index}.w.en <= consent\n    lane{index}.w.clk <= clock\n"
            f"    lane{index}.w.data <= bits(word, {index * 8 + 7}, {index * 8})\n"
            f"    lane{index}.w.mask <= mask{index}\n"
        )
    return source + "    observed <= cat(lane1.r.data, lane0.r.data)\n"


@pytest.fixture
def source(tools, tmp_path, request):
    kind = getattr(request, "param", "direct")
    firrtl = tmp_path / "source.fir"
    firrtl.write_text(_firrtl(kind))
    bundle = produce_selection(
        target="test_unit",
        firrtl=firrtl,
        generator="test_unit",
        config="DeclaredMemory",
        core_root="Unit",
        firtool=tools["MERLIN_TEST_FIRTOOL"],
        output=tmp_path / "original-production",
    )
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: test_unit\n")
    forbidden = (tmp_path / "excluded-answers",)
    hardware = issue_independent_hardware_intake(
        target="test_unit",
        descriptor=descriptor,
        source_bundle=bundle,
        forbidden_roots=forbidden,
        output=tmp_path / "live-hardware",
    )
    return hardware, forbidden


def _issue(source, tools, tmp_path, **changes):
    hardware, forbidden = source
    return M.issue_independent_memory_port_intake(
        hardware=hardware,
        circt_opt=tools["MERLIN_TEST_CIRCT_OPT"],
        source_bytes=1048576,
        limits=LIMITS,
        forbidden_roots=forbidden,
        output=tmp_path / "memory-intake",
        **changes,
    )


def test_native_memory_port_bindings_and_undefined_domains_are_exact(source, tools, tmp_path):
    intake = _issue(source, tools, tmp_path)
    record = intake.record()
    facts = record["facts"]
    assert facts["memory_contents_evaluated"] is False
    assert facts["command_tensor_allocation_or_temporal_admission"] is False
    assert set(facts["unknowns"]) >= {
        "allocation_capacity_and_physical_tails",
        "state_and_instance_transfer_semantics",
        "physical_alias_lifetime_order_and_completion",
        "initial_memory_contents_and_reachable_state",
    }
    unit = facts["modules"][0]
    nodes = {row["id"]: row for row in unit["expressions"]}
    assert len(unit["memories"]) == 2
    lanes = []
    for memory in unit["memories"]:
        assert (memory["depth"], memory["width"], memory["declared_storage_bits"]) == (3, 8, 24)
        assert (memory["read_latency"], memory["write_latency"], memory["read_under_write"]) == (1, 1, "undefined")
        assert memory["address_domain"] == {"minimum": 0, "maximum": 2, "out_of_range": "unestablished"}
        assert len(memory["ports"]) == 2
        read, write = memory["ports"]
        assert read["operation"] == "seq.firmem.read_port"
        assert write["operation"] == "seq.firmem.write_port"
        assert nodes[read["bindings"]["address"]]["name"] == "a"
        assert nodes[write["bindings"]["address"]]["name"] == "b"
        assert nodes[write["bindings"]["enable"]]["expression"] == "comb.and"
        interval = write["data_source_interval"]
        assert nodes[interval["root"]]["name"] == "arbitrary"
        lanes.append((interval["low_bit"], interval["width"]))
    assert sorted(lanes) == [(0, 8), (8, 8)]
    I.verify(Path(record["invocation"]))
    assert intake.hardware is source[0]
    assert not source[1][0].exists()


@pytest.mark.parametrize(
    "source,expected", [("state", "state_result"), ("opaque", "opaque_instance_result")], indirect=["source"]
)
def test_native_state_and_opaque_data_stay_symbolic(source, tools, tmp_path, expected):
    facts = _issue(source, tools, tmp_path).record()["facts"]
    unit = facts["modules"][0]
    nodes = {row["id"]: row for row in unit["expressions"]}
    for memory in unit["memories"]:
        write = next(row for row in memory["ports"] if "data" in row["bindings"])
        root = nodes[write["data_source_interval"]["root"]]
        assert root["kind"] == expected
        if expected == "opaque_instance_result":
            assert (root["instance"], root["module"], root["port"]) == ("other", "Other", "signal")
            assert root["original_callee_ports"] == [{"direction": "output", "name": "signal", "type": "i16"}]
    assert facts["command_tensor_allocation_or_temporal_admission"] is False


@pytest.mark.parametrize("field", ["depth", "read_latency", "read_under_write", "ports", "data_source_interval"])
def test_saved_memory_claims_cannot_substitute_original_native_source(source, tools, tmp_path, field):
    intake = _issue(source, tools, tmp_path)
    changed = copy.deepcopy(intake.record())
    memory = changed["facts"]["modules"][0]["memories"][0]
    if field == "ports":
        memory["ports"].pop()
    elif field == "data_source_interval":
        write = next(row for row in memory["ports"] if "data" in row["bindings"])
        write[field]["low_bit"] = 1
    else:
        memory[field] = "new" if field == "read_under_write" else 5
    with pytest.raises(RtlIntakeRefusal, match="complete original typed"):
        M.verify_record(changed)
    with pytest.raises(RtlIntakeRefusal, match="live same-HW"):
        dataclasses.replace(intake).verify()


def test_complete_memory_roster_costs_apply_before_any_content_evaluation(source, tools, tmp_path):
    hardware, forbidden = source
    with pytest.raises(ValueError, match="aggregate port budget"):
        M.issue_independent_memory_port_intake(
            hardware=hardware,
            circt_opt=tools["MERLIN_TEST_CIRCT_OPT"],
            source_bytes=1048576,
            limits=dataclasses.replace(LIMITS, ports=3),
            forbidden_roots=forbidden,
            output=tmp_path / "denied",
        )
    assert not (tmp_path / "denied/intake.json").exists()


def test_memory_source_size_budget_precedes_parser(source, tools, tmp_path, monkeypatch):
    hardware, forbidden = source

    def no_parse(*args, **kwargs):
        raise AssertionError("denied source reached native generic graph parser")

    monkeypatch.setattr(M, "parse_generic_hw", no_parse)
    with pytest.raises(RtlIntakeRefusal, match="preparse byte budget"):
        M.issue_independent_memory_port_intake(
            hardware=hardware,
            circt_opt=tools["MERLIN_TEST_CIRCT_OPT"],
            source_bytes=1,
            limits=LIMITS,
            forbidden_roots=forbidden,
            output=tmp_path / "denied-parse",
        )


def test_closed_memory_record_retains_original_unknown_and_limit_rosters(source, tools, tmp_path):
    record = _issue(source, tools, tmp_path).record()
    missing = copy.deepcopy(record)
    missing["facts"]["unknowns"].remove("physical_alias_lifetime_order_and_completion")
    with pytest.raises(RtlIntakeRefusal, match="complete original typed"):
        M.verify_record(missing)
    missing = json.loads(json.dumps(record))
    missing["limits"].pop("nodes")
    with pytest.raises(RtlIntakeRefusal, match="complete closed"):
        M.verify_record(missing)


@pytest.mark.parametrize("read_write", [False, True])
def test_native_seq_operand_roles_and_optional_defaults_are_preserved(tools, tmp_path, read_write):
    memory_type = "!seq.firmem<3 x 16, mask 2>"
    code = (
        "module {\n  hw.module @Unit(in %clock: !seq.clock, in %a: i2, in %word: i16, "
        "in %consent: i1, in %mask: i2, in %choice: i1, out result: i16) {\n"
        '    %memory = "seq.firmem"() {readLatency = 1 : i32, writeLatency = 1 : i32, '
        f"ruw = 0 : i32, wuw = 1 : i32}} : () -> {memory_type}\n"
    )
    if read_write:
        code += (
            '    %data = "seq.firmem.read_write_port"(%memory, %a, %clock, %consent, %word, %choice, %mask) '
            "{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1>} : "
            f"({memory_type}, i2, !seq.clock, i1, i16, i1, i2) -> i16\n"
        )
    else:
        code += (
            '    "seq.firmem.write_port"(%memory, %a, %clock, %word) '
            "{operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>} : "
            f"({memory_type}, i2, !seq.clock, i16) -> ()\n"
            f'    %data = "seq.firmem.read_port"(%memory, %a, %clock) : ({memory_type}, i2, !seq.clock) -> i16\n'
        )
    code += "    hw.output %data : i16\n  }\n}\n"
    original, generic = tmp_path / "ordinary-memory.mlir", tmp_path / "generic.mlir"
    original.write_text(code)
    I.run(
        [str(tools["MERLIN_TEST_CIRCT_OPT"]), "--mlir-print-op-generic", str(original), "-o", str(generic)],
        directory=tmp_path,
        stage="ordinary_native_seq_memory_schema_control",
        inputs=(original,),
        outputs=(generic,),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        check=True,
        timeout=30,
    )
    facts = memory_port_observations(parse_generic_hw(generic.read_text()), limits=LIMITS)
    unit = facts["modules"][0]
    memory = unit["memories"][0]
    assert memory["mask_width"] == 2
    nodes = {row["id"]: row for row in unit["expressions"]}
    write = next(port for port in memory["ports"] if "data" in port["bindings"])
    assert nodes[write["bindings"]["data"]]["name"] == "word"
    assert nodes[write["bindings"]["address"]]["name"] == "a"
    assert nodes[write["bindings"]["clock"]]["name"] == "clock"
    if read_write:
        assert nodes[write["bindings"]["enable"]]["name"] == "consent"
        assert nodes[write["bindings"]["mode"]]["name"] == "choice"
        assert nodes[write["bindings"]["mask"]]["name"] == "mask"
        assert write["implicit_enable"] is None and write["implicit_write_mask"] is None
    else:
        assert write["implicit_enable"] == "true" and write["implicit_write_mask"] == "all_bits_enabled"
    assert facts["memory_contents_evaluated"] is False
