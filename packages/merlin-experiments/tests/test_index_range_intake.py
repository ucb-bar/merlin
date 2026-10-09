"""Same-native-source range facets retain original ports and unknown premises."""

import copy
import dataclasses
import io
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import index_range_intake as R
from merlin_experiments.phase0.address_transition_intake import issue_independent_address_transition_intake
from merlin_experiments.phase0.hierarchical_memory_intake import issue_independent_hierarchical_memory_intake
from merlin_experiments.phase0.memory_port_intake import issue_independent_memory_port_intake
from merlin_experiments.phase0.rtl_intake import issue_independent_hardware_intake
from xdsl.printer import Printer

from merlin.common import invocation_record as I
from merlin.targetgen.rtl import hw_index_ranges as core
from merlin.targetgen.rtl.hw_address_transitions import AddressTransitionLimits
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits
from merlin.targetgen.rtl.hw_index_ranges import IndexRangeLimits
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
from merlin.targetgen.rtl.hw_observations import _name
from merlin.targetgen.rtl.source_selection import produce_selection

LOCAL = MemoryPortLimits(16, 16, 32, 2048, 64, 65536, 64)
HIERARCHY = HierarchyBindingLimits(16, 16, 16, 32, 256, 2048, 64, 65536, 16, 128)
TRANSITIONS = AddressTransitionLimits(32, 32, 128, 8192, 128, 2048, 64, 65536, 64)
LIMITS = IndexRangeLimits(32, 64, 64, 4096)


@pytest.fixture(params=["legacy", "modern"])
def sdk(request):
    suffix = "_MODERN" if request.param == "modern" else ""
    keys = ["MERLIN_TEST_FIRTOOL" + suffix, "MERLIN_TEST_CIRCT_OPT" + suffix]
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("index range controls require both explicitly selected coherent native SDK pairs")
    return tuple(Path(os.environ[key]).resolve(strict=True) for key in keys)


@pytest.fixture
def source(sdk, tmp_path, request):
    depth = getattr(request, "param", 4)
    width = max(1, (depth - 1).bit_length())
    fir = tmp_path / "minimal.fir"
    fir.write_text(
        "FIRRTL version 2.0.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/IndependentIndexRange.scala 1:1]\n"
        f"    input clock : Clock\n    input offered : UInt<{width}>\n"
        "    input enable : UInt<1>\n    input value : UInt<8>\n    output seen : UInt<8>\n"
        f"    reg stored : UInt<{width}>, clock\n    stored <= offered\n"
        f"    mem storage :\n      data-type => UInt<8>\n      depth => {depth}\n"
        "      read-latency => 0\n      write-latency => 1\n      reader => r\n      writer => w\n"
        "      read-under-write => undefined\n    storage.r.addr <= stored\n"
        "    storage.r.en <= enable\n    storage.r.clk <= clock\n    storage.w.addr <= stored\n"
        "    storage.w.en <= enable\n    storage.w.clk <= clock\n    storage.w.data <= value\n"
        "    storage.w.mask <= enable\n    seen <= storage.r.data\n"
    )
    selection = produce_selection(
        target="test_unit",
        firrtl=fir,
        generator="test_unit",
        config="IndependentIndexRange",
        core_root="Unit",
        firtool=sdk[0],
        output=tmp_path / "original",
    )
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: test_unit\n")
    forbidden = (tmp_path / "excluded-answers",)
    hardware = issue_independent_hardware_intake(
        target="test_unit",
        descriptor=descriptor,
        source_bundle=selection,
        forbidden_roots=forbidden,
        output=tmp_path / "hardware",
    )
    memory = issue_independent_memory_port_intake(
        hardware=hardware,
        circt_opt=sdk[1],
        source_bytes=1048576,
        limits=LOCAL,
        forbidden_roots=forbidden,
        output=tmp_path / "memory",
    )
    hierarchy = issue_independent_hierarchical_memory_intake(
        memory=memory, source_bytes=1048576, limits=HIERARCHY, forbidden_roots=forbidden, output=tmp_path / "hierarchy"
    )
    transitions = issue_independent_address_transition_intake(
        hierarchy=hierarchy,
        source_bytes=1048576,
        limits=TRANSITIONS,
        forbidden_roots=forbidden,
        output=tmp_path / "transitions",
    )
    return transitions, forbidden, depth


def _issue(source, tmp_path):
    return R.issue_independent_index_range_intake(
        transitions=source[0],
        source_bytes=1048576,
        limits=LIMITS,
        forbidden_roots=source[1],
        output=tmp_path / "ranges",
    )


@pytest.mark.parametrize("source", [4, 5, 6], indirect=True)
def test_complete_native_port_roster_and_conditional_domain_only(source, tmp_path):
    original = source[0].receipt_json
    issued = _issue(source, tmp_path)
    record = issued.record()
    assert source[0].receipt_json == original
    assert len(record["facts"]["addresses"]) == 2
    for row in record["facts"]["addresses"]:
        assert row["known_bit_domain"]["conditional_domain_contained"] is (source[2] == 4)
        assert row["known_bit_domain"]["address_definedness_proved"] is False
        assert row["original_range_obligation"]["proved"] is False
        assert row["known_bit_domain"]["declared_depth"] == source[2]
        assert "initialization_and_memory_history" in row["unknowns"]
    assert record["facts"]["state_events_evaluated"] is False
    assert record["facts"]["address_definedness_granted"] is False
    assert record["facts"]["command_capacity_axis_or_effect_admission"] is False
    with pytest.raises(ValueError, match="live"):
        copy.copy(issued).require_issued()


@pytest.mark.parametrize("field", ["membership", "conditional", "definedness", "depth", "source", "limits"])
def test_exact_original_membership_types_and_unknowns_replay(source, tmp_path, field):
    record = _issue(source, tmp_path).record()
    bad = copy.deepcopy(record)
    if field == "membership":
        bad["facts"]["addresses"].pop()
    elif field == "limits":
        bad["limits"]["addresses"] = True
    elif field == "source":
        bad["facts"]["addresses"][0]["original_address_source"]["type"] = "i3"
    elif field == "depth":
        bad["facts"]["addresses"][0]["known_bit_domain"]["declared_depth"] = 5
    else:
        key = "conditional_domain_contained" if field == "conditional" else "address_definedness_proved"
        bad["facts"]["addresses"][0]["known_bit_domain"][key] = 1 if field == "conditional" else 0
    with pytest.raises(ValueError):
        R.verify_record(bad)


@pytest.mark.parametrize("field", ["addresses", "proof_bits"])
def test_whole_proof_preflight_precedes_domain_expansion(source, monkeypatch, field):
    pin = next(pin for pin in source[0].hierarchy.memory.source_pins if pin.role == "generic-core-hw")
    parsed = parse_generic_hw(Path(pin.path).read_text(), reject_dense_literals=True)
    limits = dataclasses.replace(LIMITS, **{field: 1})

    def forbidden(*args, **kwargs):
        raise AssertionError("range expansion started before the complete roster budget")

    monkeypatch.setattr(core, "known_unsigned_index_domain", forbidden)
    with pytest.raises(ValueError, match="pre-expansion"):
        core.index_range_observations(
            parsed,
            root="Unit",
            local_limits=LOCAL,
            hierarchy_limits=HIERARCHY,
            transition_limits=TRANSITIONS,
            limits=limits,
        )


def test_native_address_width_mismatch_refuses_source_join(source, sdk, tmp_path):
    pin = next(pin for pin in source[0].hierarchy.memory.source_pins if pin.role == "generic-core-hw")
    parsed = parse_generic_hw(Path(pin.path).read_text(), reject_dense_literals=True)
    read = next(op for op in parsed.walk() if _name(op) == "seq.firmem.read_port")
    constant = next(
        op
        for op in parse_generic_hw('module { %a = "hw.constant"() {value=0:i3} : () -> i3 }').walk()
        if _name(op) == "hw.constant"
    )
    constant.detach()
    read.parent.insert_op_before(constant, read)
    read.operands = (read.operands[0], constant.results[0], *read.operands[2:])
    with pytest.raises(ValueError, match="type"):
        core.index_range_observations(
            parsed,
            root="Unit",
            local_limits=LOCAL,
            hierarchy_limits=HIERARCHY,
            transition_limits=TRANSITIONS,
            limits=LIMITS,
        )
    text = io.StringIO()
    Printer(stream=text, print_generic_format=True).print_op(parsed)
    path = tmp_path / "independent-malformed-width.mlir"
    path.write_text(text.getvalue() + "\n")
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    result = I.run(
        [str(sdk[1]), str(path), "--verify-each", "--mlir-print-op-generic"],
        directory=tmp_path / "native-width-refusal",
        cwd=tmp_path,
        stage="native_memory_index_width_refusal",
        inputs=(path, Path(__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and result.stderr
