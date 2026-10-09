"""Original HW fields have one owner across legacy and property serialization."""

from io import StringIO

import pytest
from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.printer import Printer

from merlin.targetgen.rtl.hw_combinational import EvaluationLimits, prepare_combinational_observation
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_instance_inputs import prepare_instance_input_observation
from merlin.targetgen.rtl.hw_observations import _name, input_observations
from merlin.targetgen.rtl.hw_packing import equal_partitions

LIMITS = EvaluationLimits(20000, 100, 64, 4, 10000)
SOURCE = '''builtin.module {
  "hw.module.extern"() {sym_name="Producer", parameters=[],
    module_type=!hw.modty<input inword:i12, output word:i12>} : () -> ()
  "hw.module.extern"() {sym_name="Consumer", parameters=[],
    module_type=!hw.modty<input low:i6, input selected:i1>} : () -> ()
  "hw.module"() ({
  ^bb0(%a:i12):
    %word = "hw.instance"(%a) {instanceName="source",moduleName=@Producer,
      argNames=["inword"],resultNames=["word"],parameters=[]} : (i12) -> i12
    %low = "comb.extract"(%word) {lowBit=0:i32} : (i12) -> i6
    %high = "comb.extract"(%word) {lowBit=6:i32} : (i12) -> i6
    %k = "hw.constant"() {value=17:i12} : () -> i12
    %selected = "comb.icmp"(%word,%k) {predicate=0:i64} : (i12,i12) -> i1
    "hw.instance"(%low,%selected) {instanceName="sink",moduleName=@Consumer,
      argNames=["low","selected"],resultNames=[],parameters=[]} : (i6,i1) -> ()
    "hw.output"() : () -> ()
  }) {sym_name="Parent",parameters=[],module_type=!hw.modty<input a:i12>} : () -> ()
  "hw.module"() ({
  ^bb0(%a:i12):
    %low = "comb.extract"(%a) {lowBit=0:i32} : (i12) -> i6
    %high = "comb.extract"(%a) {lowBit=6:i32} : (i12) -> i6
    %sum = "comb.add"(%low,%high) : (i6,i6) -> i6
    "hw.output"(%sum) : (i6) -> ()
  }) {sym_name="Pure",parameters=[],module_type=!hw.modty<input a:i12,output z:i6>} : () -> ()
}'''


def _print(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _source(storage):
    module = parse_generic_hw(SOURCE)
    for op in module.walk():
        fields = sorted(key for key in op.attributes if key != "op_name__")
        for index, key in enumerate(fields):
            if storage == "properties" or storage == "mixed" and index % 2 == 0:
                op.properties[key] = op.attributes.pop(key)
    return _print(module)


@pytest.mark.parametrize("storage", ["attributes", "properties", "mixed"])
def test_every_reader_preserves_original_fields_and_opaque_boundaries(storage):
    source = _source(storage)
    observations = input_observations(parse_generic_hw(source))
    parent = next(row for row in observations["modules"] if row["module"] == "Parent")
    assert parent["inputs"] == [{"name": "a", "width": 12}]
    assert parent["instance_output_observations"] == [{
        "scope": "opaque_instance_result", "instance": "source", "module": "Producer", "output": "word",
        "width": 12, "equality_constants": [17], "input_correspondence": "unknown", "instance_effects": "unknown",
    }]
    facts = equal_partitions(parse_generic_hw(source))
    assert facts["complete_resource_domain"] is False
    assert {(row["module"], row["root"]["kind"], row["slice_width"], row["slice_count"])
            for row in facts["partitions"]} == {
        ("Parent", "opaque_instance_output", 6, 2), ("Pure", "module_input", 6, 2),
    }
    pure = prepare_combinational_observation(source, module="Pure", limits=LIMITS)
    assert pure.evaluate(({"a": 0}, {"a": 17}, {"a": 4095})) == ({"z": 0}, {"z": 17}, {"z": 62})
    sink = prepare_instance_input_observation(source, module="Parent", instance="sink",
                                              ports=("low", "selected"), limits=LIMITS)
    assert len(sink.roots) == 1 and sink.roots[0].kind == "opaque_instance_result"
    root = sink.roots[0].port.name
    assert sink.evaluate(({root: 0}, {root: 17}, {root: 4095})) == (
        {"low": 0, "selected": 0}, {"low": 17, "selected": 1}, {"low": 63, "selected": 0},
    )


@pytest.mark.parametrize("field,kind", [
    ("sym_name", "hw.module"), ("module_type", "hw.module"),
    ("resultNames", "hw.instance"), ("moduleName", "hw.instance"), ("instanceName", "hw.instance"),
    ("value", "hw.constant"), ("predicate", "comb.icmp"),
])
@pytest.mark.parametrize("same", [True, False])
def test_input_observer_refuses_ambiguous_ownership_even_when_values_agree(field, kind, same):
    module = parse_generic_hw(SOURCE)
    op = next(op for op in module.walk() if _name(op) == kind and field in op.attributes)
    value = op.attributes[field]
    if not same:
        value = IntegerAttr(1, value.type) if isinstance(value, IntegerAttr) else StringAttr("different")
    op.properties[field] = value
    with pytest.raises(ValueError, match="ambiguous attribute/property ownership"):
        input_observations(module)


@pytest.mark.parametrize("reader", [input_observations, equal_partitions])
def test_input_type_mismatch_cannot_supply_field_or_packing_facts(reader):
    source = SOURCE.replace('module_type=!hw.modty<input a:i12>', 'module_type=!hw.modty<input a:i13>')
    with pytest.raises(ValueError, match="declared input types differ"):
        reader(parse_generic_hw(source))


@pytest.mark.parametrize("reader", ["complete", "instance"])
def test_bounded_readers_refuse_duplicate_selected_module_fields(reader):
    module = parse_generic_hw(SOURCE)
    op = next(op for op in module.walk() if _name(op) == "hw.module")
    op.properties["sym_name"] = op.attributes["sym_name"]
    with pytest.raises(ValueError, match="ambiguous attribute/property ownership"):
        if reader == "complete":
            prepare_combinational_observation(_print(module), module="Pure", limits=LIMITS)
        else:
            prepare_instance_input_observation(_print(module), module="Parent", instance="sink",
                                              ports=("low", "selected"), limits=LIMITS)
