"""Opt-in array dependencies retain original slots and all temporal/source gaps."""

import copy
import io
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import transition_connectivity_intake as C
from merlin_experiments.phase0.address_transition_intake import issue_independent_address_transition_intake
from merlin_experiments.phase0.hierarchical_memory_intake import issue_independent_hierarchical_memory_intake
from merlin_experiments.phase0.memory_port_intake import issue_independent_memory_port_intake
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake
from xdsl.printer import Printer

from merlin.common import invocation_record as I
from merlin.common.jsonio import strict_json_equal
from merlin.targetgen.rtl import hw_transition_connectivity as core
from merlin.targetgen.rtl.hw_address_transitions import AddressTransitionLimits
from merlin.targetgen.rtl.hw_array_selection import ArraySelectionLimits
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
from merlin.targetgen.rtl.hw_observations import _name
from merlin.targetgen.rtl.hw_transition_connectivity import TransitionConnectivityLimits
from merlin.targetgen.rtl.source_selection import produce_selection

LOCAL = MemoryPortLimits(16, 16, 32, 2048, 256, 65536, 64)
HIERARCHY = HierarchyBindingLimits(16, 16, 16, 32, 256, 2048, 256, 65536, 16, 128)
TRANSITIONS = AddressTransitionLimits(32, 32, 128, 8192, 128, 2048, 256, 65536, 64)
LIMITS = TransitionConnectivityLimits(16, 256, 128, 8192, 2048, 256, 65536, 128)
ARRAYS = ArraySelectionLimits(128, 1024, 65536)


def _source(count):
    text = (
        "FIRRTL version 2.0.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/src/IndependentArrayGraph.scala 1:1]\n"
        "    input clock : Clock\n    input clear : UInt<1>\n    input index : UInt<2>\n"
        "    input value : UInt<8>\n    input enable : UInt<1>\n    output seen : UInt<8>\n"
    )
    for i in range(count):
        text += f"    input u{i} : UInt<3>\n"
    text += f"    wire lanes : UInt<3>[{count}]\n"
    for i in range(count):
        text += f"    reg held{i} : UInt<3>, clock\n    held{i} <= u{i}\n    lanes[{i}] <= held{i}\n"
    text += (
        "    reg address : UInt<3>, clock with :\n      reset => (clear, UInt<3>(0))\n"
        "    address <= lanes[index]\n"
        "    mem storage :\n      data-type => UInt<8>\n      depth => 5\n"
        "      read-latency => 1\n      write-latency => 1\n      reader => r\n      writer => w\n"
        "      read-under-write => undefined\n    storage.r.addr <= address\n"
        "    storage.r.en <= enable\n    storage.r.clk <= clock\n    storage.w.addr <= address\n"
        "    storage.w.en <= enable\n    storage.w.clk <= clock\n    storage.w.data <= value\n"
        "    storage.w.mask <= enable\n    seen <= storage.r.data\n"
    )
    return text


@pytest.fixture(params=["legacy", "modern"])
def sdk(request):
    suffix = "" if request.param == "legacy" else "_MODERN"
    keys = ["MERLIN_TEST_FIRTOOL" + suffix, "MERLIN_TEST_CIRCT_OPT" + suffix]
    if not all(os.environ.get(key) for key in keys):
        pytest.skip("array intake controls require both explicitly selected coherent native SDK pairs")
    return tuple(Path(os.environ[key]).resolve(strict=True) for key in keys)


@pytest.fixture
def source(sdk, tmp_path, request):
    count = getattr(request, "param", 4)
    fir = tmp_path / "minimal.fir"
    fir.write_text(_source(count))
    selection = produce_selection(
        target="test_unit",
        firrtl=fir,
        generator="test_unit",
        config="IndependentArrayGraph",
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
    return transitions, forbidden


def _issue(source, tmp_path, array_limits=ARRAYS):
    return C.issue_independent_transition_connectivity_intake(
        transitions=source[0],
        source_bytes=1048576,
        limits=LIMITS,
        forbidden_roots=source[1],
        output=tmp_path / "connectivity",
        array_limits=array_limits,
    )


def test_native_opt_in_exact_scalar_dependencies_keep_v1_and_all_source_slots(source, tmp_path):
    authority = _issue(source, tmp_path)
    record = authority.record()
    facts = record["facts"]
    assert record["schema"] == C.ARRAY_SCHEMA
    baseline = _issue(source, tmp_path / "old-mode", None).record()
    assert (
        baseline["schema"] == C.SCHEMA
        and "array_limits" not in baseline
        and "array_selection_cost" not in baseline["facts"]
    )
    keys = ("frame", "state_ordinal", "operand_ordinal", "primitive_role", "type")
    assert [{k: r[k] for k in keys} for r in facts["operand_bindings"]] == [
        {k: r[k] for k in keys} for r in baseline["facts"]["operand_bindings"]
    ]
    assert strict_json_equal(facts["source_frames"], baseline["facts"]["source_frames"])
    assert strict_json_equal(facts["original_root_ports"], baseline["facts"]["original_root_ports"])
    arrays = [row for row in facts["expressions"] if row.get("expression") == "hw.array_get"]
    assert len(arrays) == 1
    row = arrays[0]
    selection = row["array_selection"]
    nodes = {row["id"]: row for row in facts["expressions"]}
    assert selection["element_count"] == 4 and selection["runtime_index_to_creation_operand"] == [3, 2, 1, 0]
    assert selection["full_original_index_domain_defined"] is True
    assert nodes[row["operands"][0]]["kind"] == "root_input" and nodes[row["operands"][0]]["port"] == "index"
    assert all(nodes[key]["kind"] == "state_result" for key in row["operands"][1:])
    assert any(
        row.get("operation") == "hw.array_get" and row["kind"] == "unsupported_result"
        for row in baseline["facts"]["expressions"]
    )
    assert facts["clock_events_evaluated"] is False and facts["command_axis_capacity_or_temporal_admission"] is False
    assert all(row["range_obligation"]["proved"] is False for row in source[0].record()["facts"]["addresses"])
    with pytest.raises(RtlIntakeRefusal):
        copy.copy(authority).verify()


@pytest.mark.parametrize("source", [3], indirect=True)
def test_native_non_power_of_two_index_domain_retains_conditional_stop(source, sdk, tmp_path):
    # FIRRTL's three-element vector lowering fills an otherwise undefined fourth
    # slot by repeating a source element. Declare a separate minimal native HW
    # program with exactly three elements; never mint intake from edited receipts.
    pin = next(pin for pin in source[0].hierarchy.memory.source_pins if pin.role == "generic-core-hw")
    parsed = parse_generic_hw(Path(pin.path).read_text(), reject_dense_literals=True)
    create = next(op for op in parsed.walk() if _name(op) == "hw.array_create")
    assert len(create.operands) == 4 and create.operands[0] is create.operands[-1]
    create.operands = tuple(create.operands[1:])
    typ = create.results[0].type
    create.results[0]._type = type(typ)(typ.attr_name, typ.is_type, typ.is_opaque, "3xi3")
    text = io.StringIO()
    Printer(stream=text, print_generic_format=True).print_op(parsed)
    original = tmp_path / "independent-three.mlir"
    original.write_text(text.getvalue() + "\n")
    output = tmp_path / "independent-three.generic.mlir"
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    result = I.run(
        [str(sdk[1]), str(original), "--verify-each", "--mlir-print-op-generic", "-o", str(output)],
        directory=tmp_path / "native-three",
        stage="independent_three_element_array_source",
        inputs=(original, Path(__file__)),
        outputs=(output,),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    facts = core.transition_operand_connectivity(
        parse_generic_hw(output.read_text(), reject_dense_literals=True),
        root="Unit",
        local_limits=LOCAL,
        hierarchy_limits=HIERARCHY,
        transition_limits=TRANSITIONS,
        limits=LIMITS,
        array_limits=ARRAYS,
    )
    arrays = [row for row in facts["expressions"] if "array_selection" in row]
    assert len(arrays) == 1
    assert arrays[0]["kind"] == "unsupported_result" and "out-of-range" in arrays[0]["reason"]
    assert arrays[0]["array_selection"]["full_original_index_domain_defined"] is False
    assert arrays[0]["array_selection"]["defined_index_maximum"] == 2
    assert arrays[0]["array_selection"]["runtime_index_to_creation_operand"] == [2, 1, 0]
    for record in (tmp_path / "native-three").glob("invocations/*/invocation.json"):
        I.require_environment(record, environment=environment)


def test_whole_array_preflight_precedes_dependency_expansion(source, monkeypatch):
    pin = next(pin for pin in source[0].hierarchy.memory.source_pins if pin.role == "generic-core-hw")
    parsed = parse_generic_hw(Path(pin.path).read_text(), reject_dense_literals=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("aggregate expansion began before whole-roster preflight")

    monkeypatch.setattr(core, "typed_array_selection", forbidden)
    with pytest.raises(ValueError, match="pre-expansion"):
        core.transition_operand_connectivity(
            parsed,
            root="Unit",
            local_limits=LOCAL,
            hierarchy_limits=HIERARCHY,
            transition_limits=TRANSITIONS,
            limits=LIMITS,
            array_limits=ArraySelectionLimits(1, 1024, 65536),
        )


@pytest.mark.parametrize("field", ["schema", "array_limits", "creation_ordinal", "full_original_index_domain_defined"])
def test_new_mode_budgets_and_exact_original_bindings_replay(source, tmp_path, field):
    record = _issue(source, tmp_path).record()
    bad = copy.deepcopy(record)
    if field == "schema":
        bad["schema"] = []
    elif field == "array_limits":
        bad.pop("array_limits")
    else:
        row = next(row for row in bad["facts"]["expressions"] if "array_selection" in row)
        row["array_selection"][field] = False if field == "creation_ordinal" else 1
    with pytest.raises(ValueError):
        C.verify_record(bad)
