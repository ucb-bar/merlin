"""Independent native hierarchies prove connectivity, preserving semantic stops."""

import copy
import dataclasses
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import hierarchical_memory_intake as H
from merlin_experiments.phase0.memory_port_intake import issue_independent_memory_port_intake
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake
from xdsl.dialects.builtin import ArrayAttr, IntegerType, SymbolRefAttr

from merlin.common import invocation_record as I
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits, hierarchical_memory_bindings
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
from merlin.targetgen.rtl.hw_observations import _attribute, _name
from merlin.targetgen.rtl.source_selection import produce_selection

LOCAL = MemoryPortLimits(32, 32, 64, 2048, 256, 65536, 64)
LIMITS = HierarchyBindingLimits(32, 32, 32, 64, 512, 2048, 256, 65536, 16, 128)


def _source(kind):
    text = "FIRRTL version 2.0.0\ncircuit Unit :\n"
    if kind in {"opaque", "parameterized"}:
        text += (
            "  extmodule Route :\n    input clock : Clock\n    input l : UInt<16>\n    input r : UInt<16>\n"
            "    output first : UInt<16>\n    output second : UInt<16>\n    defname = Route\n"
        )
        if kind == "parameterized":
            text += "    parameter arbitrary = 7\n"
    else:
        text += (
            "  module Route :\n    input clock : Clock\n    input l : UInt<16>\n    input r : UInt<16>\n"
            "    output first : UInt<16>\n    output second : UInt<16>\n"
        )
        if kind == "state":
            text += "    reg held : UInt<16>, clock\n    held <= r\n    first <= held\n"
        elif kind == "unsupported":
            text += "    first <= div(r, l)\n"
        else:
            text += "    first <= r\n"
        text += "    second <= l\n"
    text += (
        "  module Leaf :\n    input clock : Clock\n    input index : UInt<2>\n"
        "    input consent : UInt<1>\n    input arbitrary : UInt<16>\n    output seen : UInt<16>\n"
    )
    for index in range(2):
        text += (
            f"    mem lane{index} :\n      data-type => UInt<8>\n      depth => 3\n"
            "      read-latency => 1\n      write-latency => 1\n      reader => r\n      writer => w\n"
            "      read-under-write => undefined\n"
            f"    lane{index}.r.addr <= index\n    lane{index}.r.en <= consent\n    lane{index}.r.clk <= clock\n"
            f"    lane{index}.w.addr <= index\n    lane{index}.w.en <= consent\n    lane{index}.w.clk <= clock\n"
            f"    lane{index}.w.data <= bits(arbitrary, {index * 8 + 7}, {index * 8})\n"
            f"    lane{index}.w.mask <= consent\n"
        )
    text += "    seen <= cat(lane1.r.data, lane0.r.data)\n"
    text += (
        "  module Bridge :\n    input clock : Clock\n    input index : UInt<2>\n    input consent : UInt<1>\n"
        "    input l : UInt<16>\n    input r : UInt<16>\n    output seen : UInt<16>\n"
        "    inst route of Route\n    route.clock <= clock\n    route.l <= l\n    route.r <= r\n"
    )
    for name, word in (("a", "route.first"), ("b", "a.seen" if kind == "memory" else "route.second")):
        text += (
            f"    inst {name} of Leaf\n    {name}.clock <= clock\n    {name}.index <= index\n"
            f"    {name}.consent <= consent\n    {name}.arbitrary <= {word}\n"
        )
    text += "    seen <= xor(a.seen, b.seen)\n"
    text += (
        "  module Unit : @[generators/test_unit/src/IndependentHierarchy.scala 1:1]\n"
        "    input clock : Clock\n    input index : UInt<2>\n    input consent : UInt<1>\n"
        "    input x : UInt<16>\n    input y : UInt<16>\n    output seen : UInt<16>\n"
    )
    for name, left, right in (("first", "x", "y"), ("second", "y", "x")):
        text += (
            f"    inst {name} of Bridge\n    {name}.clock <= clock\n    {name}.index <= index\n"
            f"    {name}.consent <= consent\n    {name}.l <= {left}\n    {name}.r <= {right}\n"
        )
    return text + "    seen <= xor(first.seen, second.seen)\n"


@pytest.fixture
def source(tmp_path, request):
    if not all(os.environ.get(key) for key in ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_CIRCT_OPT")):
        pytest.skip("hierarchical controls require an explicitly selected coherent native CIRCT pair")
    kind = getattr(request, "param", "direct")
    fir = tmp_path / "minimal.fir"
    fir.write_text(_source(kind))
    bundle = produce_selection(
        target="test_unit",
        firrtl=fir,
        generator="test_unit",
        config="IndependentHierarchy",
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
    return memory, forbidden


def _issue(source, tmp_path, limits=LIMITS):
    memory, forbidden = source
    return H.issue_independent_hierarchical_memory_intake(
        memory=memory, source_bytes=1048576, limits=limits, forbidden_roots=forbidden, output=tmp_path / "hierarchy"
    )


def _parsed(source):
    pin = next(pin for pin in source[0].source_pins if pin.role == "generic-core-hw")
    return parse_generic_hw(Path(pin.path).read_text(), reject_dense_literals=True)


def test_native_complete_occurrences_and_exact_cross_module_byte_roots(source, tmp_path):
    before = source[0].record()
    intake = _issue(source, tmp_path)
    facts = intake.record()["facts"]
    assert (facts["cost"]["occurrences"], facts["cost"]["memory_occurrences"], facts["cost"]["memory_ports"]) == (
        9,
        8,
        16,
    )
    nodes = {row["id"]: row for row in facts["expressions"]}
    frames = {row["id"]: row for row in facts["frames"]}
    actual = {}
    for memory in facts["memories"]:
        frame = frames[memory["frame"]]
        path = tuple(row["instance"] for row in frame["path"])
        write = next(row for row in memory["ports"] if row["operation"] == "seq.firmem.write_port")
        interval = write["data_source_interval"]
        root = nodes[interval["root"]]
        assert root["kind"] == "root_input" and root["frame"] == 0
        actual[path, interval["low_bit"]] = root["port"]
        assert memory["declaration"]["read_under_write"] == "undefined"
        assert memory["declaration"]["address_domain"]["out_of_range"] == "unestablished"
    assert actual == {
        ((outer, leaf), low): port
        for outer, leaf, port in (("first", "a", "y"), ("first", "b", "x"), ("second", "a", "x"), ("second", "b", "y"))
        for low in (0, 8)
    }
    assert {row["kind"] for row in nodes.values()} >= {
        "instance_input_binding",
        "instance_output_binding",
        "root_input",
    }
    assert source[0].record() == before
    assert facts["memory_contents_evaluated"] is False
    assert facts["command_capacity_axis_or_temporal_admission"] is False
    assert not source[1][0].exists()


@pytest.mark.parametrize(
    "source,kind",
    [
        ("state", "state_result"),
        ("opaque", "opaque_instance_result"),
        ("parameterized", "opaque_instance_result"),
        ("memory", "memory_read_result"),
        ("unsupported", "unsupported_result"),
    ],
    indirect=["source"],
)
def test_native_temporal_opaque_parameterized_and_unsupported_stops_remain(source, kind, tmp_path):
    facts = _issue(source, tmp_path).record()["facts"]
    nodes = {row["id"]: row for row in facts["expressions"]}
    roots = [
        nodes[port["data_source_interval"]["root"]]
        for memory in facts["memories"]
        for port in memory["ports"]
        if port["data_source_interval"] is not None
    ]
    assert kind in {row["kind"] for row in nodes.values()}
    if kind != "memory_read_result":
        assert kind in {row["kind"] for row in roots}
    assert facts["command_capacity_axis_or_temporal_admission"] is False
    assert "state_transfer_reachability_and_initialization" in facts["unknowns"]
    if any(row["stop"] == "parameterized_body" for row in facts["frames"]):
        assert any(row.get("stop") == "parameterized_body" for row in roots)


@pytest.mark.parametrize("field", ["occurrences", "memory_occurrences", "memory_ports", "port_bindings"])
def test_complete_hierarchy_budget_refuses_before_expansion(source, field, monkeypatch):
    from merlin.targetgen.rtl import hw_hierarchy_bindings as B

    def forbidden(*args, **kwargs):
        raise AssertionError("hierarchical expression work must follow complete roster admission")

    monkeypatch.setattr(B, "_expression", forbidden)
    with pytest.raises(ValueError, match="pre-expansion"):
        B.hierarchical_memory_bindings(
            _parsed(source), root="Unit", local_limits=LOCAL, limits=dataclasses.replace(LIMITS, **{field: 1})
        )


@pytest.mark.parametrize(
    "field,change",
    [(field, change) for field in ("argNames", "resultNames") for change in ("missing", "duplicate", "reordered")],
)
def test_original_instance_port_membership_and_order_cannot_be_substituted(source, field, change):
    parsed = _parsed(source)
    op = next(op for op in parsed.walk() if _name(op) == "hw.instance" and len(op.results) > 1)
    names = list(_attribute(op, field).data)
    if change == "missing":
        names.pop()
    elif change == "duplicate":
        names[1] = names[0]
    else:
        names[0], names[1] = names[1], names[0]
    owner = op.properties if field in op.properties else op.attributes
    owner[field] = ArrayAttr(names)
    with pytest.raises(ValueError, match="binding"):
        hierarchical_memory_bindings(parsed, root="Unit", local_limits=LOCAL, limits=LIMITS)


@pytest.mark.parametrize("change", ["type", "callee"])
def test_original_instance_type_and_callee_are_required(source, change):
    parsed = _parsed(source)
    op = next(op for op in parsed.walk() if _name(op) == "hw.instance" and len(op.results) > 1)
    if change == "type":
        op.results[0]._type = IntegerType(17)
    else:
        owner = op.properties if "moduleName" in op.properties else op.attributes
        owner["moduleName"] = SymbolRefAttr("Unavailable")
    with pytest.raises(ValueError, match="binding|callee"):
        hierarchical_memory_bindings(parsed, root="Unit", local_limits=LOCAL, limits=LIMITS)


def test_native_legacy_parameter_dictionary_stops_defined_body(source, tmp_path):
    original = next(pin for pin in source[0].source_pins if pin.role == "generic-core-hw")
    text = Path(original.path).read_text()
    spelling = 'instanceName = "route"'
    assert text.count(spelling) == 1
    selected = tmp_path / "legacy-parameter.mlir"
    line = next(line for line in text.splitlines() if spelling in line)
    if "}>" in line:
        # Unknown keys in modern inherent properties are discarded by native
        # parsing. A retained legacy dictionary belongs in discardable attrs.
        head, tail = line.split("}>", 1)
        replacement = head + "}> {oldParameters = {arbitrary = 7 : i32}}" + tail
        selected.write_text(text.replace(line, replacement))
    else:
        selected.write_text(text.replace(spelling, "oldParameters = {arbitrary = 7 : i32}, " + spelling))
    output = tmp_path / "native-legacy-parameter.mlir"
    I.run(
        [os.environ["MERLIN_TEST_CIRCT_OPT"], "--mlir-print-op-generic", str(selected), "-o", str(output)],
        directory=tmp_path,
        stage="native_original_legacy_parameter_structure",
        inputs=(selected,),
        outputs=(output,),
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        capture_output=True,
        check=True,
        timeout=30,
    )
    parsed = parse_generic_hw(output.read_text(), reject_dense_literals=True)
    route = next(
        op for op in parsed.walk() if _name(op) == "hw.instance" and _attribute(op, "instanceName").data == "route"
    )
    assert _attribute(route, "oldParameters") is not None
    facts = hierarchical_memory_bindings(parsed, root="Unit", local_limits=LOCAL, limits=LIMITS)
    roots = [row for row in facts["expressions"] if row["kind"] == "opaque_instance_result"]
    assert roots and all(row["stop"] == "parameterized_body" for row in roots)
    assert any(row["instance_legacy_parameters"] != "None" for row in facts["frames"])


@pytest.mark.parametrize("field", ["hierarchy_depth", "expression_depth", "nodes", "bit_work", "scalar_bits"])
def test_explicit_traversal_metadata_limits_refuse(source, field):
    with pytest.raises(ValueError, match="budget|width"):
        hierarchical_memory_bindings(
            _parsed(source), root="Unit", local_limits=LOCAL, limits=dataclasses.replace(LIMITS, **{field: 1})
        )


@pytest.mark.parametrize("change", ["binding", "interval", "unknown", "budget"])
def test_saved_hierarchy_claims_cannot_replace_original_source(source, tmp_path, change):
    intake = _issue(source, tmp_path)
    record = copy.deepcopy(intake.record())
    if change == "binding":
        record["facts"]["memories"][0]["ports"][0]["bindings"]["address"] += 1
    elif change == "interval":
        port = next(row for row in record["facts"]["memories"][0]["ports"] if row["data_source_interval"] is not None)
        port["data_source_interval"]["low_bit"] += 1
    elif change == "unknown":
        record["unknowns"].pop()
    else:
        record["limits"].pop("port_bindings")
    with pytest.raises(RtlIntakeRefusal):
        H.verify_record(record)
    with pytest.raises(RtlIntakeRefusal, match="live"):
        dataclasses.replace(intake).verify()


def test_explicit_preparse_budget_refuses_before_parser(source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source byte admission must precede parsing")

    monkeypatch.setattr(H, "parse_generic_hw", forbidden)
    with pytest.raises(RtlIntakeRefusal, match="preparse"):
        H.issue_independent_hierarchical_memory_intake(
            memory=source[0], source_bytes=1, limits=LIMITS, forbidden_roots=source[1], output=tmp_path / "denied"
        )
    assert not (tmp_path / "denied").exists()
