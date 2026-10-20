"""CIRCT 1.75 prints some inherent attributes in the attribute dictionary; discovery must still see them."""

import pytest

from merlin.targetgen.rtl import hw_graph
from merlin.targetgen.rtl.hw_graph import parse_generic_hw

# The shape firtool-1.75.0 / circt-opt --mlir-print-op-generic emits for a chipyard elaboration:
# hw.module and hw.constant carry inherent attributes in `{...}`, comb.extract already in `<{...}>`.
_CIRCT_175 = """builtin.module {
  "hw.module"() ({
  ^bb0(%a: i8):
    %k = "hw.constant"() {value = 3 : i8} : () -> i8
    %lo = "comb.extract"(%a) <{lowBit = 2 : i32}> {sv.namehint = "lo"} : (i8) -> i4
    "hw.output"(%k) : (i8) -> ()
  }) {emit.fragments = [], module_type = !hw.modty<input a : i8, output y : i8>, parameters = [],
      sym_name = "Cell", sym_visibility = "private"} : () -> ()
}"""


def _op(root, name):
    return next(op for op in root.walk() if op.attributes.get("op_name__") and op.attributes["op_name__"].data == name)


def test_dictionary_printed_inherent_attributes_reach_discovery_on_a_clone():
    root = parse_generic_hw(_CIRCT_175)
    module = _op(root, "hw.module")
    assert "sym_name" not in module.properties  # the reader mlc's HwGraph uses would see `<anon>`

    projected, declared = hw_graph._discovery_projection(root)

    assert declared == {"Cell": ("Cell", (("input", "a", "i8"), ("output", "y", "i8")))}
    copy = _op(projected, "hw.module")
    assert str(copy.properties["sym_name"]).strip('"') == "Cell"
    assert str(copy.properties["module_type"]).startswith("!hw.modty<input a : i8")
    assert "value" in _op(projected, "hw.constant").properties
    assert "sym_name" not in module.properties  # the parsed original is never rewritten


def test_discovery_graph_names_modules_and_ports():
    irgraph = pytest.importorskip("mlc.discover.irgraph")
    projected, _ = hw_graph._discovery_projection(parse_generic_hw(_CIRCT_175))
    graph = irgraph.HwGraph(projected)
    assert list(graph.modules) == ["Cell"]
    assert [(p.name, p.direction, p.width) for p in graph.modules["Cell"].ports] == [
        ("a", "input", 8),
        ("y", "output", 8),
    ]
