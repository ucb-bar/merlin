"""Rank-3/4 siblings of the signed-off m2m exact-quantization rows: proposed, so never counted as reviewed."""

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "examples" / "gemmini" / "platform" / "host-capabilities-portable-v3.yaml"
BASE = ("host_m2m_exact_quantize_f32_i8", "host_m2m_exact_readout_i32_f32", "host_m2m_dequantize_per_tensor_f64_scale")
PROPOSED = {name + "_rank3_4" for name in BASE}
Q = {"zero_point": 0, "quant_min": -128, "quant_max": 127}
CASES = [
    ("m2m_q.quantize.default", "quant_ext.quantize_per_tensor", ["f32", "f64", "i64"], ["i8"], Q),
    ("m2m_q.exact_readout.default", "linalg.generic", ["i32", "f32"], ["f32"], None),
    ("quantized_decomposed.dequantize_per_tensor.default", "quant_ext.dequantize_per_tensor",
     ["i8", "f64", "i64"], ["f32"], Q),
]


@pytest.fixture(scope="module")
def shipped():
    return yaml.safe_load(PROFILE.read_bytes())


def _admit(document, frontend, mlir, operands, results, rank, quant):
    raw = yaml.safe_dump(document).encode()
    selection = {
        "int8_w8a8": {
            "capability_spec": document,
            "capability_spec_sha256": hashlib.sha256(raw).hexdigest(),
            "package_sha256": document["compiler"]["package_sha256"],
            "dtype_strategy": document["compiler"]["dtype_strategy"],
        }
    }
    signature = {"family": "elementwise_map", "ordered_operand_dtypes": operands, "ordered_result_dtypes": results,
                 "rank": rank}
    if quant is not None:
        signature["quantization_parameters"] = quant
    if frontend == "m2m_q.exact_readout.default":
        signature.update(aliasing="disjoint_inputs_outputs", layout="row_major_contiguous", readout_dtype="f32")
    return admit_host_operation(selection, {"frontend_op": frontend, "mlir_operation": mlir}, signature)


def test_proposed_rows_are_unreviewed_siblings_of_the_signed_off_rows(shipped):
    validate_host_capabilities(shipped)
    rows = {row["id"]: row for row in shipped["operations"]}
    assert PROPOSED <= set(rows)
    for name in BASE:
        base, proposed = rows[name], rows[name + "_rank3_4"]
        assert base["status"] == "reviewed" and base["signature"]["ranks"] == [2]
        assert proposed["status"] == "unreviewed" and proposed["numerical_contract"]["status"] == "unreviewed"
        assert proposed["signature"]["ranks"] == [3, 4]
        assert "NOT REVIEWED" in proposed["numerical_contract"]["evidence"]["reviewed_by"]
        # Only the ranks, statuses and review note differ from the signed-off row.
        a, b = copy.deepcopy(base), copy.deepcopy(proposed)
        for row in (a, b):
            row.pop("id"), row.pop("status"), row["signature"].pop("ranks")
            row["numerical_contract"].pop("status"), row["numerical_contract"]["evidence"].pop("reviewed_by")
        assert a == b


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("rank", [3, 4])
def test_rank_3_4_is_not_a_reviewed_admission_until_an_operator_reviews_the_row(shipped, case, rank):
    verdict = _admit(shipped, *case[:4], rank, case[4])
    # program_admission._lanes counts a lane only for status admitted AND reviewed.
    assert not (verdict["status"] == "admitted" and verdict["reviewed"] is True)
    decisions = {d.get("declaration"): d for d in verdict["profiles"][0]["decisions"]}
    proposed = [name for name in decisions if name in PROPOSED]
    assert len(proposed) == 1 and decisions[proposed[0]]["status"] != "admitted"
    # Counterfactual: the same row, once an operator records review, is the reviewed admission.
    flipped = copy.deepcopy(shipped)
    for row in flipped["operations"]:
        if row["id"] == proposed[0]:
            row["status"] = "reviewed"
            row["numerical_contract"]["status"] = "reviewed"
    after = _admit(flipped, *case[:4], rank, case[4])
    assert after["status"] == "admitted" and after["reviewed"] is True


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_signed_off_rank_2_admission_is_unchanged(shipped, case):
    verdict = _admit(shipped, *case[:4], 2, case[4])
    assert verdict["status"] == "admitted" and verdict["reviewed"] is True


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_rank_5_and_wrong_dtypes_stay_refused(shipped, case):
    assert _admit(shipped, *case[:4], 5, case[4])["status"] != "admitted"
    operands = ["f16"] + list(case[2][1:])
    assert _admit(shipped, case[0], case[1], operands, case[3], 3, case[4])["status"] != "admitted"
