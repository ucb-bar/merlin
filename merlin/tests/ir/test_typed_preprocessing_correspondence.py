"""A source-transform map can support a narrow, hash-bound type check."""

import hashlib
import json

import pytest

from merlin.common.ir_audit import IrAudit
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.llvmlower.typed_preprocessing_correspondence import check_result_types

SOURCE = """module {
  func.func @forward(%x: f32) -> tensor<2x2xf32> {
    %c0 = arith.constant 0.0 : f32
    %t = tensor.splat %c0 : tensor<2x2xf32>
    func.return %t : tensor<2x2xf32>
  }
}"""

MULTI_SOURCE = """module {
  func.func private @external(%x: f32) -> f32
  func.func @first(%x: f32) -> f32 {
    %sum = arith.addf %x, %x : f32
    func.return %sum : f32
  }
  func.func @second(%x: f32) -> f32 {
    %two = arith.constant 2.0 : f32
    %product = arith.mulf %x, %two : f32
    func.return %product : f32
  }
}"""

MULTI_WITH_DEAD = """module {
  func.func @first(%x: f32) -> f32 {
    %dead = arith.constant 2.0 : f32
    %unused = arith.addf %dead, %x : f32
    %live = arith.addf %x, %x : f32
    func.return %live : f32
  }
  func.func @second(%x: f32) -> f32 {
    %live = arith.addf %x, %x : f32
    func.return %live : f32
  }
}"""

MULTI_WITH_QUANT = """module {
  func.func @first(%w: tensor<2x8xi8>, %s: tensor<f32>, %z: tensor<i32>) -> tensor<2x8xf32> {
    %r = "quant_ext.dequantize_per_tensor"(%w, %s, %z) <{input_dtype = "i8"}>
        : (tensor<2x8xi8>, tensor<f32>, tensor<i32>) -> tensor<2x8xf32>
    func.return %r : tensor<2x8xf32>
  }
  func.func @second(%w: tensor<2x8xi8>, %s: tensor<f32>, %z: tensor<i32>) -> tensor<2x8xf32> {
    %r = "quant_ext.dequantize_per_tensor"(%w, %s, %z) <{input_dtype = "i8"}>
        : (tensor<2x8xi8>, tensor<f32>, tensor<i32>) -> tensor<2x8xf32>
    func.return %r : tensor<2x8xf32>
  }
}"""


def _audit(tmp_path):
    with IrAudit(tmp_path, enabled="compact", producer="typed-correspondence-test", source=__file__) as audit:
        preprocessed, _ = preprocess_text(SOURCE, audit=audit)
    return preprocessed, audit.directory / "index.json"


def test_checked_result_types_are_explicitly_short_of_value_or_terminal_proof(tmp_path):
    preprocessed, index = _audit(tmp_path)
    result = check_result_types(SOURCE.encode(), preprocessed.encode(), index)
    assert result["source_operation_count"] == 2
    assert result["checked_result_count"] == 2
    assert result["claim"] == "preprocessing_result_types_only"
    assert result["value_preservation"] == "not_checked"
    assert result["terminal_lowering"] == "not_checked"


def test_checked_result_types_refuse_wrong_source_and_modified_receipt(tmp_path):
    preprocessed, index = _audit(tmp_path)
    with pytest.raises(ValueError, match="source digest"):
        check_result_types(SOURCE.replace("2x2", "3x2").encode(), preprocessed.encode(), index)

    map_path = index.parent / "source-transform-map.json"
    map_path.write_bytes(map_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt digest"):
        check_result_types(SOURCE.encode(), preprocessed.encode(), index)


def test_checked_result_types_reparse_and_reject_a_hash_consistent_wrong_mapping(tmp_path):
    preprocessed, index_path = _audit(tmp_path)
    index = json.loads(index_path.read_text())
    map_path = index_path.parent / "source-transform-map.json"
    receipt = json.loads(map_path.read_text())
    first, second = receipt["operations"]
    first["preprocessed_op_indices"], second["preprocessed_op_indices"] = (
        second["preprocessed_op_indices"], first["preprocessed_op_indices"]
    )
    first["result_map"][0]["preprocessed_op_index"] = first["preprocessed_op_indices"][0]
    second["result_map"][0]["preprocessed_op_index"] = second["preprocessed_op_indices"][0]
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    map_path.write_bytes(raw)
    index["accounting_receipts"][0]["sha256"] = digest
    index["source_transform_map"]["sha256"] = digest
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="result type"):
        check_result_types(SOURCE.encode(), preprocessed.encode(), index_path)


def test_multi_function_map_checks_each_body_and_external_signature(tmp_path):
    with IrAudit(tmp_path, enabled="compact", producer="multi-correspondence-test", source=__file__) as audit:
        preprocessed, _ = preprocess_text(MULTI_SOURCE, audit=audit)
    result = check_result_types(MULTI_SOURCE.encode(), preprocessed.encode(), audit.directory / "index.json")
    assert result["source_operation_count"] == 3
    assert result["checked_result_count"] == 3
    assert result["defined_function_count"] == 2
    assert result["value_preservation"] == "not_checked"


def test_multi_function_map_accounts_for_dead_pure_cone(tmp_path):
    with IrAudit(tmp_path, enabled="compact", producer="multi-correspondence-test", source=__file__) as audit:
        preprocessed, stats = preprocess_text(MULTI_WITH_DEAD, audit=audit)
    result = check_result_types(MULTI_WITH_DEAD.encode(), preprocessed.encode(), audit.directory / "index.json")
    assert stats["dead_tensor_ops_pruned"] == 2
    assert result["source_operation_count"] == 4
    assert result["checked_result_count"] == 2
    assert result["elided_source_operation_count"] == 2
    assert result["elided_source_result_count"] == 2


def test_multi_function_map_assigns_generated_quant_ops_to_each_source(tmp_path):
    with IrAudit(tmp_path, enabled="compact", producer="multi-correspondence-test", source=__file__) as audit:
        preprocessed, stats = preprocess_text(MULTI_WITH_QUANT, audit=audit)
    result = check_result_types(MULTI_WITH_QUANT.encode(), preprocessed.encode(), audit.directory / "index.json")
    receipt = json.loads((audit.directory / "source-transform-map.json").read_text())
    assert stats["quant_ext_lowered"] == 2
    assert [row["preprocessed_op_count"] for row in receipt["functions"]] == [2, 2]
    assert result["checked_result_count"] == 2
    assert result["defined_function_count"] == 2


def test_multi_function_map_refuses_hash_consistent_cross_function_swap(tmp_path):
    with IrAudit(tmp_path, enabled="compact", producer="multi-correspondence-test", source=__file__) as audit:
        preprocessed, _ = preprocess_text(MULTI_SOURCE, audit=audit)
    index_path = audit.directory / "index.json"
    index = json.loads(index_path.read_text())
    receipt_path = audit.directory / "source-transform-map.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["functions"][0], receipt["functions"][1] = receipt["functions"][1], receipt["functions"][0]
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    receipt_path.write_bytes(raw)
    index["accounting_receipts"][0]["sha256"] = digest
    index["source_transform_map"]["sha256"] = digest
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="function identity"):
        check_result_types(MULTI_SOURCE.encode(), preprocessed.encode(), index_path)


def test_multi_function_map_refuses_elision_of_live_source_result(tmp_path):
    with IrAudit(tmp_path, enabled="compact", producer="multi-correspondence-test", source=__file__) as audit:
        preprocessed, _ = preprocess_text(MULTI_SOURCE, audit=audit)
    index_path = audit.directory / "index.json"
    index = json.loads(index_path.read_text())
    receipt_path = audit.directory / "source-transform-map.json"
    receipt = json.loads(receipt_path.read_text())
    constant, product = receipt["functions"][1]["operations"][:2]
    product["preprocessed_op_indices"].extend(constant["preprocessed_op_indices"])
    constant["preprocessed_op_indices"] = []
    constant["result_map"] = []
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    receipt_path.write_bytes(raw)
    index["accounting_receipts"][0]["sha256"] = digest
    index["source_transform_map"]["sha256"] = digest
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="live source user"):
        check_result_types(MULTI_SOURCE.encode(), preprocessed.encode(), index_path)
