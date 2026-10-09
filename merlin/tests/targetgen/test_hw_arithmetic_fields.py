"""Arithmetic relations retain the original module symbol's unique owner."""

import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.targetgen.rtl.hw_arithmetic import local_arithmetic
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_observations import _name

SOURCE = '''builtin.module {
  "hw.module"() ({
  ^bb0(%x:i8,%y:i8,%z:i20):
    %xs = "comb.extract"(%x) {lowBit=7:i32} : (i8) -> i1
    %ys = "comb.extract"(%y) {lowBit=7:i32} : (i8) -> i1
    %xh = "comb.replicate"(%xs) : (i1) -> i8
    %yh = "comb.replicate"(%ys) : (i1) -> i8
    %xe = "comb.concat"(%xh,%x) : (i8,i8) -> i16
    %ye = "comb.concat"(%yh,%y) : (i8,i8) -> i16
    %p = "comb.mul"(%xe,%ye) : (i16,i16) -> i16
    %ps = "comb.extract"(%p) {lowBit=15:i32} : (i16) -> i1
    %ph = "comb.replicate"(%ps) : (i1) -> i4
    %pe = "comb.concat"(%ph,%p) : (i4,i16) -> i20
    %q = "comb.add"(%pe,%z) : (i20,i20) -> i20
    "hw.output"(%q) : (i20) -> ()
  }) {sym_name="Independent",parameters=[],
      module_type=!hw.modty<input x:i8,input y:i8,input z:i20,output q:i20>} : () -> ()
}'''


def _module(storage, recognized=True):
    module = parse_generic_hw(SOURCE if recognized else SOURCE.replace('"hw.output"(%q)', '"hw.output"(%z)'))
    for op in module.walk():
        for index, key in enumerate(sorted(key for key in op.attributes if key != "op_name__")):
            if storage == "properties" or storage == "mixed" and index % 2 == 0:
                op.properties[key] = op.attributes.pop(key)
    return module


@pytest.mark.parametrize("storage", ["attributes", "properties", "mixed"])
def test_arithmetic_preserves_original_symbol_and_relation(storage):
    facts = local_arithmetic(_module(storage))
    assert facts["relations"] == [{
        "module": "Independent", "output_ordinal": 0,
        "law": "signed_multiply_add_modulo_result_width",
        "operands": [{"input_ordinal": 0, "width": 8}, {"input_ordinal": 1, "width": 8}],
        "addend": {"input_ordinal": 2, "width": 20}, "product_bits": 16, "result_bits": 20,
    }]
    assert facts["complete_arithmetic_domain"] is False
    assert facts["examined_modules"] == facts["examined_outputs"] == 1
    assert facts["unrecognized_outputs"] == 0


@pytest.mark.parametrize("recognized", [True, False])
@pytest.mark.parametrize("same", [True, False])
def test_ambiguous_symbol_refuses_even_for_unrecognized_outputs(recognized, same):
    module = _module("attributes", recognized)
    op = next(op for op in module.walk() if _name(op) == "hw.module")
    op.properties["sym_name"] = op.attributes["sym_name"] if same else StringAttr("Different")
    with pytest.raises(ValueError, match="ambiguous attribute/property ownership"):
        local_arithmetic(module)


@pytest.mark.parametrize("recognized", [True, False])
def test_missing_original_symbol_refuses_for_every_module(recognized):
    module = _module("properties", recognized)
    op = next(op for op in module.walk() if _name(op) == "hw.module")
    del op.properties["sym_name"]
    with pytest.raises(ValueError, match="no complete explicit symbol name"):
        local_arithmetic(module)
