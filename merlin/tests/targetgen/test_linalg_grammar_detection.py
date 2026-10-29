"""Grammar routing depends on parsed tensor functions, not optional provenance or case identity."""
import pytest

from merlin.targetgen.contract import linalg_iface
from merlin.targetgen.oot_starterkit import iface

_TENSOR_ADD = """builtin.module {
  func.func @forward(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %r = arith.addf %x, %y : tensor<2x3xf32>
    func.return %r : tensor<2x3xf32>
  }
}"""
_TENSOR_ABS = """builtin.module {
  func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %r = math.absf %x : tensor<2x3xf32>
    func.return %r : tensor<2x3xf32>
  }
}"""
_TENSOR_IDENTITY = """builtin.module {
  func.func @read_tensor(%x: tensor<5x7xi8>) -> tensor<5x7xi8> {
    func.return %x : tensor<5x7xi8>
  }
}"""


@pytest.mark.parametrize("source", [_TENSOR_ADD, _TENSOR_ABS, _TENSOR_IDENTITY])
def test_unannotated_tensor_functions_route_to_reader_without_rewriting(source):
    assert "prov.level" not in source
    assert linalg_iface.is_linalg_on_tensors(source)
    assert iface.is_linalg_on_tensors(source)
    parsed = iface.parse_linalg(source)
    assert parsed["level"] == "linalg-on-tensors"
    assert parsed["args"] and parsed["results"]


def test_legacy_explicit_level_remains_a_routing_hint():
    marked = _TENSOR_ADD.replace("builtin.module {", 'builtin.module attributes {prov.level = "linalg-on-tensors"} {')
    assert linalg_iface.is_linalg_on_tensors(marked)
    assert iface.parse_linalg(marked)["ops"]


@pytest.mark.parametrize("source", [
    "",
    "builtin.module { func.func @forward(",
    "builtin.module {}",
    "builtin.module { func.func @scalar(%x: i32) -> i32 { func.return %x : i32 } }",
    'builtin.module { "merlin_iface.tensor"() : () -> tensor<2x3xi8> }',
    'builtin.module { func.func @forward(%x: tensor<2x3xi8>) -> tensor<2x3xi8> {'
    ' %r = "merlin_iface.custom"(%x) : (tensor<2x3xi8>) -> tensor<2x3xi8>'
    ' func.return %r : tensor<2x3xi8> } }',
    "builtin.module {"
    " func.func @one(%x: tensor<2xi8>) -> tensor<2xi8> { func.return %x : tensor<2xi8> }"
    " func.func @two(%x: tensor<2xi8>) -> tensor<2xi8> { func.return %x : tensor<2xi8> } }",
])
def test_other_grammars_invalid_text_and_ambiguous_entries_do_not_match(source):
    assert not linalg_iface.is_linalg_on_tensors(source)
    assert not iface.is_linalg_on_tensors(source)
