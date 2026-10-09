"""Independent native crossed hierarchies bind every original state operand."""

import copy
import dataclasses
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import transition_connectivity_intake as C
from merlin_experiments.phase0.address_transition_intake import issue_independent_address_transition_intake
from merlin_experiments.phase0.hierarchical_memory_intake import issue_independent_hierarchical_memory_intake
from merlin_experiments.phase0.memory_port_intake import issue_independent_memory_port_intake
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake
from xdsl.dialects.builtin import ArrayAttr, IntegerType, SymbolRefAttr

from merlin.targetgen.rtl.hw_address_transitions import AddressTransitionLimits
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
from merlin.targetgen.rtl.hw_observations import _attribute, _name
from merlin.targetgen.rtl.hw_transition_connectivity import (
    TransitionConnectivityLimits,
    transition_operand_connectivity,
)
from merlin.targetgen.rtl.source_selection import produce_selection

LOCAL = MemoryPortLimits(32, 32, 64, 2048, 256, 65536, 64)
HIERARCHY = HierarchyBindingLimits(32, 32, 32, 64, 512, 2048, 256, 65536, 16, 128)
TRANSITIONS = AddressTransitionLimits(64, 64, 256, 8192, 256, 2048, 256, 65536, 64)
LIMITS = TransitionConnectivityLimits(32, 512, 256, 8192, 2048, 256, 65536, 128)


def _source(kind):
    text = "FIRRTL version 2.0.0\ncircuit Unit :\n"
    if kind in {"opaque", "parameterized"}:
        text += (
            "  extmodule Route :\n    input clock : Clock\n    input left : UInt<3>\n    input right : UInt<3>\n"
            "    output first : UInt<3>\n    output second : UInt<3>\n    defname = Route\n"
        )
        if kind == "parameterized":
            text += "    parameter arbitrary = 11\n"
    else:
        text += (
            "  module Route :\n    input clock : Clock\n    input left : UInt<3>\n    input right : UInt<3>\n"
            "    output first : UInt<3>\n    output second : UInt<3>\n"
        )
        if kind == "state":
            text += "    reg held : UInt<3>, clock\n    held <= right\n    first <= held\n"
        elif kind == "unsupported":
            text += "    first <= div(right, left)\n"
        elif kind == "memory":
            text += (
                "    mem source :\n      data-type => UInt<3>\n      depth => 5\n      read-latency => 1\n"
                "      write-latency => 1\n      reader => r\n      writer => w\n      read-under-write => undefined\n"
                "    source.r.addr <= right\n    source.r.en <= UInt<1>(1)\n    source.r.clk <= clock\n"
                "    source.w.addr <= left\n    source.w.en <= UInt<1>(1)\n    source.w.clk <= clock\n"
                "    source.w.data <= left\n    source.w.mask <= UInt<1>(1)\n"
                "    first <= source.r.data\n"
            )
        else:
            text += "    first <= right\n"
        text += "    second <= left\n"
    text += (
        "  module Leaf :\n    input clock : Clock\n    input cut : UInt<1>\n    input keep : UInt<1>\n"
        "    input offered : UInt<3>\n    input value : UInt<8>\n    output seen : UInt<8>\n"
    )
    if kind == "no_reset":
        text += "    reg stored : UInt<3>, clock\n"
    else:
        text += "    reg stored : UInt<3>, clock with :\n      reset => (cut, UInt<3>(2))\n"
    text += (
        "    stored <= mux(keep, stored, offered)\n"
        "    mem storage :\n      data-type => UInt<8>\n      depth => 5\n      read-latency => 1\n"
        "      write-latency => 1\n      reader => r\n      writer => w\n      read-under-write => undefined\n"
        "    storage.r.addr <= stored\n    storage.r.en <= keep\n    storage.r.clk <= clock\n"
        "    storage.w.addr <= stored\n    storage.w.en <= keep\n    storage.w.clk <= clock\n"
        "    storage.w.data <= value\n    storage.w.mask <= keep\n    seen <= storage.r.data\n"
        "  module Bridge :\n    input clock : Clock\n    input cut : UInt<1>\n    input keep : UInt<1>\n"
        "    input left : UInt<3>\n    input right : UInt<3>\n    input value : UInt<8>\n    output seen : UInt<8>\n"
        "    inst route of Route\n    route.clock <= clock\n    route.left <= left\n    route.right <= right\n"
    )
    for name, output in (("a", "first"), ("b", "second")):
        text += (
            f"    inst {name} of Leaf\n    {name}.clock <= clock\n    {name}.cut <= cut\n"
            f"    {name}.keep <= keep\n    {name}.offered <= route.{output}\n    {name}.value <= value\n"
        )
    text += (
        "    seen <= xor(a.seen, b.seen)\n"
        "  module Unit : @[generators/test_unit/src/IndependentOperandGraph.scala 1:1]\n"
        "    input clock : Clock\n    input clear : UInt<1>\n    input choose : UInt<1>\n"
        "    input p : UInt<3>\n    input q : UInt<3>\n    input value : UInt<8>\n    output seen : UInt<8>\n"
    )
    for name, left, right in (("first", "p", "q"), ("second", "q", "p")):
        text += (
            f"    inst {name} of Bridge\n    {name}.clock <= clock\n    {name}.cut <= clear\n"
            f"    {name}.keep <= choose\n    {name}.left <= {left}\n    {name}.right <= {right}\n"
            f"    {name}.value <= value\n"
        )
    return text + "    seen <= xor(first.seen, second.seen)\n"


@pytest.fixture
def source(tmp_path, request):
    if not all(os.environ.get(key) for key in ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_CIRCT_OPT")):
        pytest.skip("operand controls require an explicitly selected coherent native CIRCT pair")
    fir = tmp_path / "minimal.fir"
    fir.write_text(_source(getattr(request, "param", "direct")))
    bundle = produce_selection(
        target="test_unit",
        firrtl=fir,
        generator="test_unit",
        config="IndependentOperandGraph",
        core_root="Unit",
        firtool=Path(os.environ["MERLIN_TEST_FIRTOOL"]),
        output=tmp_path / "original",
    )
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: test_unit\n")
    forbidden = (tmp_path / "excluded-answers",)
    hardware = issue_independent_hardware_intake(
        target="test_unit",
        descriptor=descriptor,
        source_bundle=bundle,
        forbidden_roots=forbidden,
        output=tmp_path / "hardware",
    )
    memory = issue_independent_memory_port_intake(
        hardware=hardware,
        circt_opt=Path(os.environ["MERLIN_TEST_CIRCT_OPT"]),
        source_bytes=1048576,
        limits=LOCAL,
        forbidden_roots=forbidden,
        output=tmp_path / "memory",
    )
    hierarchy = issue_independent_hierarchical_memory_intake(
        memory=memory,
        source_bytes=1048576,
        limits=HIERARCHY,
        forbidden_roots=forbidden,
        output=tmp_path / "hierarchy",
    )
    transitions = issue_independent_address_transition_intake(
        hierarchy=hierarchy,
        source_bytes=1048576,
        limits=TRANSITIONS,
        forbidden_roots=forbidden,
        output=tmp_path / "transitions",
    )
    return transitions, forbidden


def _parsed(source):
    pin = next(pin for pin in source[0].hierarchy.memory.source_pins if pin.role == "generic-core-hw")
    return parse_generic_hw(Path(pin.path).read_text(), reject_dense_literals=True)


def _derive(parsed, limits=LIMITS):
    return transition_operand_connectivity(
        parsed,
        root="Unit",
        local_limits=LOCAL,
        hierarchy_limits=HIERARCHY,
        transition_limits=TRANSITIONS,
        limits=limits,
    )


def _issue(source, tmp_path):
    return C.issue_independent_transition_connectivity_intake(
        transitions=source[0],
        source_bytes=1048576,
        limits=LIMITS,
        forbidden_roots=source[1],
        output=tmp_path / "connectivity",
    )


def _roots(facts, identity):
    nodes = {row["id"]: row for row in facts["expressions"]}
    pending, seen, roots = [identity], set(), []
    while pending:
        index = pending.pop()
        if index in seen:
            continue
        seen.add(index)
        row = nodes[index]
        if row["kind"] in {"combinational", "instance_input_binding", "instance_output_binding"}:
            pending.extend(row.get("operands", []))
        else:
            roots.append(row)
    return roots


@pytest.mark.parametrize("source,slots", [("direct", 16), ("no_reset", 8)], indirect=["source"])
def test_native_all_original_slots_cross_exact_ports_with_repeated_callees(source, slots, tmp_path):
    before = source[0].record()
    facts = _issue(source, tmp_path).record()["facts"]
    assert facts["cost"]["operand_bindings"] == slots
    assert len(facts["operand_bindings"]) == slots
    frames = {row["id"]: row for row in facts["source_frames"]}
    actual = {}
    for binding in facts["operand_bindings"]:
        roots = _roots(facts, binding["expression"])
        ports = {row["port"] for row in roots if row["kind"] == "root_input"}
        if binding["primitive_role"] == "next":
            path = tuple(row["instance"] for row in frames[binding["frame"]]["path"])
            actual[path] = ports - {"choose"}
            assert "choose" in ports and any(row["kind"] == "state_result" for row in roots)
        elif binding["primitive_role"] == "clock":
            assert ports == {"clock"} and binding["type"] == "!seq.clock"
        elif binding["primitive_role"] == "reset":
            assert ports == {"clear"}
        else:
            assert not roots  # Exact declared constant reset value has no opaque roots.
    assert actual == {("first", "a"): {"q"}, ("first", "b"): {"p"}, ("second", "a"): {"p"}, ("second", "b"): {"q"}}
    assert source[0].record() == before
    assert facts["clock_events_evaluated"] is False and facts["command_axis_capacity_or_temporal_admission"] is False
    assert not source[1][0].exists()


@pytest.mark.parametrize(
    "source,kind",
    [
        ("state", "state_result"),
        ("memory", "memory_read_result"),
        ("opaque", "opaque_instance_result"),
        ("parameterized", "opaque_instance_result"),
        ("unsupported", "unsupported_result"),
    ],
    indirect=["source"],
)
def test_native_complete_slots_retain_state_memory_opaque_and_unsupported_stops(source, kind, tmp_path):
    facts = _issue(source, tmp_path).record()["facts"]
    assert facts["cost"]["operand_bindings"] == 16
    next_bindings = [row for row in facts["operand_bindings"] if row["primitive_role"] == "next"]
    first_leaf = [row for row in next_bindings if facts["source_frames"][row["frame"]]["path"][-1]["instance"] == "a"]
    assert all(kind in {root["kind"] for root in _roots(facts, row["expression"])} for row in first_leaf)
    if kind != "state_result":
        assert all(
            not {"p", "q"} & {root.get("port") for root in _roots(facts, row["expression"])} for row in first_leaf
        )
    assert facts["command_axis_capacity_or_temporal_admission"] is False


@pytest.mark.parametrize(
    "field", ["frames", "port_bindings", "operand_bindings", "traversal_steps", "nodes", "bit_work", "expression_depth"]
)
def test_explicit_complete_roster_and_connectivity_work_budgets_refuse(source, field):
    with pytest.raises(ValueError, match="budget"):
        _derive(_parsed(source), dataclasses.replace(LIMITS, **{field: 1}))


def test_scalar_width_has_an_independent_connectivity_budget(source):
    with pytest.raises(ValueError, match="bounded scalar type"):
        _derive(_parsed(source), dataclasses.replace(LIMITS, scalar_bits=1))


@pytest.mark.parametrize(
    "field,change",
    [(field, change) for field in ("argNames", "resultNames") for change in ("missing", "duplicate", "reordered")],
)
def test_exact_instance_membership_and_order_cannot_be_substituted(source, field, change):
    parsed = _parsed(source)
    op = next(op for op in parsed.walk() if _name(op) == "hw.instance" and len(op.results) == 2)
    names = list(_attribute(op, field).data)
    if change == "missing":
        names.pop()
    elif change == "duplicate":
        names[1] = names[0]
    else:
        names[0], names[1] = names[1], names[0]
    (op.properties if field in op.properties else op.attributes)[field] = ArrayAttr(names)
    with pytest.raises(ValueError):
        _derive(parsed)


@pytest.mark.parametrize("change", ["callee", "type"])
def test_original_callee_and_operand_type_correspondence_are_required(source, change):
    parsed = _parsed(source)
    op = next(op for op in parsed.walk() if _name(op) == "hw.instance" and len(op.results) == 2)
    if change == "callee":
        (op.properties if "moduleName" in op.properties else op.attributes)["moduleName"] = SymbolRefAttr("Missing")
    else:
        op.operands[0]._type = IntegerType(9)
    with pytest.raises(ValueError):
        _derive(parsed)


def test_saved_partial_roster_and_copied_object_cannot_mint_live_identity(source, tmp_path):
    intake = _issue(source, tmp_path)
    record = copy.deepcopy(intake.record())
    record["facts"]["operand_bindings"].pop()
    with pytest.raises(RtlIntakeRefusal, match="complete original"):
        C.verify_record(record)
    with pytest.raises(RtlIntakeRefusal, match="live"):
        copy.copy(intake).verify()


@pytest.mark.parametrize("change", ["boolean_as_integer", "integer_as_boolean", "integer_as_float"])
def test_json_type_substitutions_cannot_match_original_facts(source, tmp_path, change):
    record = _issue(source, tmp_path).record()
    if change == "boolean_as_integer":
        record["facts"]["clock_events_evaluated"] = 0
    elif change == "integer_as_boolean":
        record["facts"]["operand_bindings"][0]["operand_ordinal"] = False
    else:
        record["facts"]["operand_bindings"][0]["operand_ordinal"] = 0.0
    with pytest.raises(RtlIntakeRefusal, match="complete original"):
        C.verify_record(record)


def test_preparse_source_budget_precedes_parser(source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source admission must precede parser")

    monkeypatch.setattr(C, "parse_generic_hw", forbidden)
    with pytest.raises(RtlIntakeRefusal, match="preparse"):
        C.issue_independent_transition_connectivity_intake(
            transitions=source[0],
            source_bytes=1,
            limits=LIMITS,
            forbidden_roots=source[1],
            output=tmp_path / "denied",
        )
    assert not (tmp_path / "denied").exists()
