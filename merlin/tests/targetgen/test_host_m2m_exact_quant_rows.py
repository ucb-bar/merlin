"""Proposed exact-quantization host rows: admit only their typed signatures, never as-shipped."""

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "examples" / "gemmini" / "platform" / "host-capabilities-portable-v3.yaml"
NEW = {
    "host_m2m_exact_quantize_f32_i8",
    "host_m2m_exact_readout_i32_f32",
    "host_m2m_dequantize_per_tensor_f64_scale",
    "host_f32_layer_norm_two_result_moments",
}
Q = {"zero_point": 0, "quant_min": -128, "quant_max": 127}


def _selection(document):
    raw = yaml.safe_dump(document).encode()
    return {
        "int8_w8a8": {
            "capability_spec": document,
            "capability_spec_sha256": hashlib.sha256(raw).hexdigest(),
            "package_sha256": document["compiler"]["package_sha256"],
            "dtype_strategy": document["compiler"]["dtype_strategy"],
        }
    }


def _reviewed(document):
    document = copy.deepcopy(document)
    for row in document["operations"]:
        if row["id"] in NEW:
            row["status"] = "reviewed"
            row["numerical_contract"]["status"] = "reviewed"
            row["numerical_contract"]["evidence"].pop("review_status")
    return document


@pytest.fixture(scope="module")
def shipped():
    return yaml.safe_load(PROFILE.read_bytes())


def _admit(document, frontend, mlir, operands, results, rank=2, family="elementwise_map", quant=None):
    row = {"frontend_op": frontend, "mlir_operation": mlir}
    signature = {
        "family": family,
        "ordered_operand_dtypes": operands,
        "ordered_result_dtypes": results,
        "rank": rank,
    }
    if quant is not None:
        signature["quantization_parameters"] = quant
    return admit_host_operation(_selection(document), row, signature)


ADMIT = [
    ("m2m_q.quantize.default", "quant_ext.quantize_per_tensor", ["f32", "f64", "i64"], ["i8"], 2, Q),
    ("m2m_q.exact_readout.default", "linalg.generic", ["i32", "f32"], ["f32"], 2, None),
    ("quantized_decomposed.dequantize_per_tensor.default", "quant_ext.dequantize_per_tensor",
     ["i8", "f64", "i64"], ["f32"], 2, Q),
    ("aten.native_layer_norm.default", "linalg.generic", ["f32", "f32"], ["f32", "f32"], 2, None),
]


def test_profile_validates_and_new_rows_are_unreviewed(shipped):
    validate_host_capabilities(shipped)
    rows = {row["id"]: row for row in shipped["operations"]}
    assert NEW <= set(rows)
    for name in NEW:
        assert rows[name]["status"] == "unreviewed"
        assert rows[name]["numerical_contract"]["status"] == "unreviewed"
        assert "Recorded operator review" not in yaml.safe_dump(rows[name])
    assert "PROPOSED" in shipped["review"]["m2m_exact_quant_addendum"]


@pytest.mark.parametrize("frontend,mlir,operands,results,rank,quant", ADMIT)
def test_as_shipped_rows_do_not_admit(shipped, frontend, mlir, operands, results, rank, quant):
    assert _admit(shipped, frontend, mlir, operands, results, rank, quant=quant)["status"] != "admitted"


@pytest.mark.parametrize("frontend,mlir,operands,results,rank,quant", ADMIT)
def test_reviewed_rows_admit_the_observed_micro_model_signatures(shipped, frontend, mlir, operands, results, rank, quant):
    decision = _admit(_reviewed(shipped), frontend, mlir, operands, results, rank, quant=quant)
    assert decision["status"] == "admitted", decision["reason"]


REFUSE = [
    # m2m exact quantizer: reviewed f32-scale TorchAO form is a different contract; wrong storage/range/rank.
    ("m2m_q.quantize.default", "quant_ext.quantize_per_tensor", ["f32", "f32", "i64"], ["i8"], 2, Q),
    ("m2m_q.quantize.default", "quant_ext.quantize_per_tensor", ["bf16", "f64", "i64"], ["i8"], 2, Q),
    ("m2m_q.quantize.default", "quant_ext.quantize_per_tensor", ["f32", "f64", "i64"], ["ui8"], 2, Q),
    ("m2m_q.quantize.default", "quant_ext.quantize_per_tensor", ["f32", "f64", "i64"], ["i8"], 5, Q),
    ("m2m_q.quantize.default", "quant_ext.quantize_per_tensor", ["f32", "f64", "i64"], ["i8"], 2,
     {"zero_point": 3, "quant_min": -128, "quant_max": 127}),
    ("m2m_q.quantize.default", "quant_ext.quantize_per_tensor", ["f32", "f64", "i64"], ["i8"], 2,
     {"zero_point": 0, "quant_min": -127, "quant_max": 127}),
    # The new row must not widen the TorchAO quantizer identity to an f64 scale.
    ("quantized_decomposed.quantize_per_tensor.default", "quant_ext.quantize_per_tensor",
     ["f32", "f64", "i64"], ["i8"], 2, Q),
    # Readout: bias operand, requantized integer result, wider accumulator.
    ("m2m_q.exact_readout.default", "linalg.generic", ["i32", "f32", "f32"], ["f32"], 2, None),
    ("m2m_q.exact_readout.default", "linalg.generic", ["i32", "i8"], ["i8"], 2, None),
    ("m2m_q.exact_readout.default", "linalg.generic", ["i64", "f32"], ["f32"], 2, None),
    ("m2m_q.exact_readout.default", "linalg.generic", ["i32", "f32"], ["f32"], 1, None),
    # Dequantize: other result dtype, zero point, rank.
    ("quantized_decomposed.dequantize_per_tensor.default", "quant_ext.dequantize_per_tensor",
     ["i8", "f64", "i64"], ["bf16"], 2, Q),
    ("quantized_decomposed.dequantize_per_tensor.default", "quant_ext.dequantize_per_tensor",
     ["i8", "f64", "i64"], ["f32"], 2, {"zero_point": 5, "quant_min": -128, "quant_max": 127}),
    ("quantized_decomposed.dequantize_per_tensor.default", "quant_ext.dequantize_per_tensor",
     ["i8", "f64", "i64"], ["f32"], 5, Q),
    ("quantized_decomposed.dequantize_per_tensor.default", "quant_ext.dequantize_per_tensor",
     ["i32", "f64", "i64"], ["f32"], 2, Q),
    # Two-result layer-norm moments only.
    ("aten.native_layer_norm.default", "linalg.generic", ["f32", "f32"], ["f32", "f32", "f32"], 2, None),
    ("aten.native_layer_norm.default", "linalg.generic", ["bf16", "bf16"], ["bf16", "bf16"], 2, None),
    ("aten._softmax.default", "linalg.generic", ["f32", "f32"], ["f32", "f32"], 2, None),
]


@pytest.mark.parametrize("frontend,mlir,operands,results,rank,quant", REFUSE)
def test_reviewed_rows_still_refuse_other_variants(shipped, frontend, mlir, operands, results, rank, quant):
    decision = _admit(_reviewed(shipped), frontend, mlir, operands, results, rank, quant=quant)
    assert decision["status"] != "admitted"
