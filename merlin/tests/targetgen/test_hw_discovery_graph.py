"""Discovery retains exact original HW fields across native serialization eras."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import OpResult

from merlin.common import invocation_record as I
from merlin.targetgen.rtl import hw_graph
from merlin.targetgen.rtl.hw_observations import _name
from merlin.targetgen.rtl.source_selection import digest, selected_sources

SOURCE = """builtin.module {
  "hw.module"() ({
  ^bb0(%data:i8):
    %low = "comb.extract"(%data) {lowBit=0:i32} : (i8) -> i4
    %k = "hw.constant"() {value=17:i8} : () -> i8
    %eq = "comb.icmp"(%data,%k) {predicate=0:i64} : (i8,i8) -> i1
    "hw.output"(%low,%eq) : (i4,i1) -> ()
  }) {sym_name="Child",parameters=[],
      module_type=!hw.modty<input data : i8,output low : i4,output equal : i1>} : () -> ()
  "hw.module"() ({
  ^bb0(%data:i8):
    %low,%eq = "hw.instance"(%data) {instanceName="member",moduleName=@Child,
      argNames=["data"],resultNames=["low","equal"],parameters=[]} : (i8) -> (i4,i1)
    "hw.output"(%low,%eq) : (i4,i1) -> ()
  }) {sym_name="Parent",parameters=[],
      module_type=!hw.modty<input data : i8,output low : i4,output equal : i1>} : () -> ()
}"""
PORTS = (("input", "data", "i8"), ("output", "low", "i4"), ("output", "equal", "i1"))


def _module(storage):
    module = hw_graph.parse_generic_hw(SOURCE)
    for op in module.walk():
        for index, field in enumerate(sorted(key for key in op.attributes if key != "op_name__")):
            if storage == "properties" or storage == "mixed" and index % 2 == 0:
                op.properties[field] = op.attributes.pop(field)
    return module


@pytest.mark.parametrize("storage", ["attributes", "properties", "mixed"])
def test_projection_preserves_every_original_field_and_ssa_without_mutating_parser(storage):
    original = _module(storage)
    before = [(dict(op.attributes), dict(op.properties)) for op in original.walk()]
    projected, declared = hw_graph._discovery_projection(original)
    assert declared == {name: (name, PORTS) for name in ("Child", "Parent")}
    original_ops, projected_ops = set(original.walk()), set(projected.walk())
    for source, copy in zip(original.walk(), projected.walk(), strict=True):
        assert source is not copy
        assert {**source.attributes, **source.properties} == {**copy.attributes, **copy.properties}
        assert tuple(value.type for value in source.operands) == tuple(value.type for value in copy.operands)
        for value in copy.operands:
            if isinstance(value, OpResult):
                assert value.owner in projected_ops and value.owner not in original_ops
            else:
                assert value.owner.parent_op() in projected_ops
                assert value.owner.parent_op() not in original_ops
    assert before == [(dict(op.attributes), dict(op.properties)) for op in original.walk()]


@pytest.mark.parametrize("same", [True, False])
@pytest.mark.parametrize(
    "kind,field",
    [
        ("hw.module", "sym_name"),
        ("hw.module", "module_type"),
        ("hw.instance", "moduleName"),
        ("hw.instance", "resultNames"),
        ("hw.constant", "value"),
        ("comb.icmp", "predicate"),
        ("comb.extract", "lowBit"),
    ],
)
def test_duplicate_original_field_ownership_refuses_before_discovery(kind, field, same):
    original = _module("attributes")
    op = next(op for op in original.walk() if _name(op) == kind)
    op.properties[field] = op.attributes[field] if same else StringAttr("different")
    with pytest.raises(ValueError, match="ambiguous attribute/property ownership"):
        hw_graph._discovery_projection(original)


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("missing_symbol", "complete explicit symbol"),
        ("wrong_symbol_type", "complete explicit symbol"),
        ("duplicate_symbol", "duplicate module symbol"),
        ("missing_type", "lossless explicit module type"),
        ("duplicate_port", "input names"),
        ("wrong_input_type", "declared input types differ"),
        ("inout", "input names"),
    ],
)
def test_incomplete_original_module_identity_cannot_become_anonymous_or_drop_ports(defect, reason):
    original = _module("attributes")
    members = [op for op in original.walk() if _name(op) == "hw.module"]
    if defect == "missing_symbol":
        del members[0].attributes["sym_name"]
    elif defect == "wrong_symbol_type":
        members[0].attributes["sym_name"] = members[0].attributes["parameters"]
    elif defect == "duplicate_symbol":
        members[1].attributes["sym_name"] = members[0].attributes["sym_name"]
    elif defect == "missing_type":
        del members[0].attributes["module_type"]
    else:
        changed = {
            "duplicate_port": "input data : i8,input data : i8,output equal : i1",
            "wrong_input_type": "input data : i9,output low : i4,output equal : i1",
            "inout": "input data : i8,inout low : i4,output equal : i1",
        }[defect]
        original = hw_graph.parse_generic_hw(
            SOURCE.replace("input data : i8,output low : i4,output equal : i1", changed)
        )
    with pytest.raises(ValueError, match=reason):
        hw_graph._discovery_projection(original)


@pytest.fixture
def selected_discovery(monkeypatch):
    selected = os.environ.get("MERLIN_TEST_HWGRAPH_SOURCE")
    if not selected:
        pytest.skip("an explicitly selected public discovery source is required")
    source = Path(selected)
    if source.is_symlink() or not source.is_file():
        raise ValueError("selected discovery source must be an exact regular file")
    spec = importlib.util.spec_from_file_location("mlc.discover.irgraph", source.resolve())
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, source.resolve()


def _check_actual_graph(graph):
    assert tuple(graph.modules) == ("Child", "Parent")
    assert graph.top_module().name == "Parent"
    for member in graph.modules.values():
        assert tuple((p.direction, p.name, p.type_str) for p in member.ports) == PORTS
        assert tuple(p.width for p in member.ports) == (8, 4, 1)
    child = graph.modules["Child"]
    assert [graph.const_value(op) for op in graph.ops("hw.constant", child)] == [17]
    assert [graph.icmp_predicate(op) for op in graph.ops("comb.icmp", child)] == [0]
    assert [graph.extract_range(op) for op in graph.ops("comb.extract", child)] == [(0, 4)]


@pytest.mark.parametrize("storage", ["attributes", "properties", "mixed"])
def test_real_selected_discovery_preserves_modules_ports_and_discriminators(storage, selected_discovery):
    discovery, _ = selected_discovery
    projected, declared = hw_graph._discovery_projection(_module(storage))
    graph = discovery.HwGraph(projected)
    _check_actual_graph(graph)
    assert set(graph.modules) == set(declared)


def test_real_discovery_cannot_silently_drop_valid_but_unsupported_port_serialization(selected_discovery, tmp_path):
    _, _ = selected_discovery
    source = tmp_path / "compact.generic.mlir"
    source.write_text(SOURCE.replace(" : i", ":i"))
    selected = {
        "target": "independent",
        "_generic_hw_output": str(source),
        "_genericization": {
            "input": {"sha256": digest(source)},
            "output": {"sha256": digest(source)},
        },
    }
    with selected_sources(selected), pytest.raises(ValueError, match="differs from original module/port identity"):
        hw_graph.load_hw_graph(source, circt_opt=tmp_path / "unused")


@pytest.mark.parametrize("era,prefix", [("legacy", "MERLIN_TEST_"), ("current", "MERLIN_TEST_CURRENT_")])
def test_actual_native_hardware_reaches_ordinary_discovery_with_exact_names_ports(
    era, prefix, selected_discovery, tmp_path
):
    selections = [os.environ.get(prefix + name) for name in ("FIRTOOL", "CIRCT_OPT")]
    if not all(selections):
        pytest.skip("explicitly selected native firtool and circt-opt are required")
    _, discovery_source = selected_discovery
    firtool, circt_opt = (Path(tool).resolve(strict=True) for tool in selections)
    source, hardware, generic = (
        tmp_path / name for name in ("original.fir", "original.hw.mlir", "original.generic.mlir")
    )
    source.write_text("""FIRRTL version 3.3.0
circuit Parent :
  module Child :
    input data : UInt<8>
    output low : UInt<4>
    output equal : UInt<1>
    connect low, bits(data,3,0)
    connect equal, eq(data,UInt<8>(17))
  module Parent :
    input data : UInt<8>
    output low : UInt<4>
    output equal : UInt<1>
    inst member of Child
    connect member.data, data
    connect low, member.low
    connect equal, member.equal
""")
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    for stage, command, inputs, outputs in (
        ("lower", [str(firtool), str(source), "--ir-hw", "-o", str(hardware)], (source,), (hardware,)),
        (
            "generic",
            [str(circt_opt), str(hardware), "--mlir-print-op-generic", "-o", str(generic)],
            (hardware,),
            (generic,),
        ),
    ):
        I.run(
            command,
            directory=tmp_path / stage,
            stage="native_hwgraph_" + stage,
            inputs=(*inputs, Path(__file__)),
            outputs=outputs,
            dependencies=(Path(I.__file__), discovery_source, Path(hw_graph.__file__)),
            env=environment,
            check=True,
            capture_output=True,
            timeout=30,
        )
    original_bytes = hardware.read_bytes(), generic.read_bytes()
    selected = {
        "target": "independent",
        "_generic_hw_output": str(generic),
        "_genericization": {
            "input": {"sha256": digest(hardware)},
            "output": {"sha256": digest(generic)},
        },
    }
    product = tmp_path / "graph.json"
    with I.observe_call(
        tmp_path / "graph",
        stage="native_hwgraph_ordinary_loader",
        function=hw_graph.load_hw_graph,
        arguments={"path": str(hardware), "circt_opt": str(circt_opt)},
        inputs=(source, hardware, generic),
        outputs=(product,),
        dependencies=(discovery_source, Path(__file__)),
    ) as record:
        with selected_sources(selected):
            graph = hw_graph.load_hw_graph(hardware, circt_opt=circt_opt)
        product.write_text(
            json.dumps(
                {
                    "modules": {
                        name: [[p.direction, p.name, p.type_str] for p in m.ports] for name, m in graph.modules.items()
                    },
                    "top": graph.top_module().name,
                },
                sort_keys=True,
            )
            + "\n"
        )
        record.returned()
    _check_actual_graph(graph)
    assert original_bytes == (hardware.read_bytes(), generic.read_bytes())
    for record in tmp_path.rglob("invocation.json"):
        if I.verify(record)["kind"] == "subprocess":
            I.require_environment(record, environment=environment)
