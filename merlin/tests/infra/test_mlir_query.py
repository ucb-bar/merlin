"""Structured MLIR reads via xDSL (merlin.common.mlir_query) — replaces regex on IR text."""
from __future__ import annotations

import pytest

pytest.importorskip("xdsl")

from merlin.common import mlir_query as q

_MOD = """module {
  func.func @forward(%a: tensor<1x50x32xf32>, %b: tensor<32x64xi8>) -> tensor<1x50x64xf32> {
    %c = "quant_ext.dequantize_per_channel"(%b) {input_dtype = "i8", prov.op = "dequantize"} : (tensor<32x64xi8>) -> tensor<32x64xf32>
    %0 = linalg.matmul {prov.quantization = "int8_weight_only"} ins(%a, %c : tensor<1x50x32xf32>, tensor<32x64xf32>) outs(%a : tensor<1x50x32xf32>) -> tensor<1x50x64xf32>
    return %0 : tensor<1x50x64xf32>
  }
}"""


def test_forward_signature():
    inputs, results = q.forward_signature(_MOD)
    assert inputs == [([1, 50, 32], "f32"), ([32, 64], "i8")]
    assert results == [([1, 50, 64], "f32")]


def test_op_name_recovers_unregistered():
    mod = q.parse(_MOD)
    names = {q.op_name(op) for op in mod.walk()}
    assert "quant_ext.dequantize_per_channel" in names  # not "builtin.unregistered"
    assert "linalg.matmul" in names


def test_op_count_and_walk():
    mod = q.parse(_MOD)
    assert q.op_count(mod, "linalg.matmul") == 1
    assert q.op_count(mod, "quant_ext.dequantize_per_channel") == 1
    assert q.op_count(mod, "scf.for", "scf.while") == 0
    assert [q.op_name(op) for op in q.walk(mod, "linalg.matmul")] == ["linalg.matmul"]


def test_attrs_and_provenance():
    mod = q.parse(_MOD)
    deq = next(q.walk(mod, "quant_ext.dequantize_per_channel"))
    assert q.attr_str(deq, "input_dtype") == "i8"
    assert q.provenance(deq) == {"prov.op": "dequantize"}
    mm = next(q.walk(mod, "linalg.matmul"))
    assert q.provenance(mm) == {"prov.quantization": "int8_weight_only"}
    assert q.attr_str(mm, "absent") is None


def test_parse_accepts_module_str_and_path(tmp_path):
    mod = q.parse(_MOD)
    assert q.parse(mod) is mod                      # already-parsed passthrough (not a Path/str)
    p = tmp_path / "m.mlir"
    p.write_text(_MOD)
    assert q.op_count(q.parse(p), "linalg.matmul") == 1        # Path (3.13 Path.walk() must not fool it)
    assert q.op_count(q.parse(str(p)), "linalg.matmul") == 1   # str path


def test_forward_signature_missing_func():
    with pytest.raises(ValueError):
        q.forward_signature("module { }", func_name="nope")
