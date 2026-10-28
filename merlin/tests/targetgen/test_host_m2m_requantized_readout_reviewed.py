"""The requantized i32,i8 -> i8 exact-readout host row is operator-approved (chat 2026-10-28) and admits as reviewed."""

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "examples" / "gemmini" / "platform" / "host-capabilities-portable-v3.yaml"
ROW = "host_m2m_exact_readout_i32_i8_requantized_rank3_4"


@pytest.fixture(scope="module")
def shipped():
    return yaml.safe_load(PROFILE.read_bytes())


def _admit(document, operands, results, rank, readout="i8"):
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
                 "rank": rank, "aliasing": "disjoint_inputs_outputs", "layout": "row_major_contiguous",
                 "readout_dtype": readout}
    return admit_host_operation(
        selection, {"frontend_op": "m2m_q.exact_readout.default", "mlir_operation": "linalg.generic"}, signature)


def _demoted(shipped):
    flipped = copy.deepcopy(shipped)
    for row in flipped["operations"]:
        if row["id"] == ROW:
            row["status"] = "unreviewed"
            row["numerical_contract"]["status"] = "unreviewed"
    return flipped


def test_row_is_reviewed_by_the_operator(shipped):
    validate_host_capabilities(shipped)
    row = {r["id"]: r for r in shipped["operations"]}[ROW]
    assert row["status"] == "reviewed" and row["numerical_contract"]["status"] == "reviewed"
    assert "Loren Hung (approved in operator chat 2026-10-28)" in row["numerical_contract"]["evidence"]["reviewed_by"]
    assert row["signature"]["ranks"] == [3, 4] and row["signature"]["ordered_result_dtypes"] == ["i8"]


@pytest.mark.parametrize("rank", [3, 4])
def test_reviewed_row_is_a_reviewed_admission_and_demotion_would_withdraw_it(shipped, rank):
    after = _admit(shipped, ["i32", "i8"], ["i8"], rank)
    assert after["status"] == "admitted" and after["reviewed"] is True
    before = _admit(_demoted(shipped), ["i32", "i8"], ["i8"], rank)
    assert not (before["status"] == "admitted" and before["reviewed"] is True)


def test_other_forms_stay_refused_even_when_reviewed(shipped):
    flipped = shipped
    assert _admit(flipped, ["i32", "i8"], ["i8"], 2)["status"] != "admitted"
    assert _admit(flipped, ["i32", "i8"], ["i8"], 5)["status"] != "admitted"
    assert _admit(flipped, ["i32", "f32"], ["i8"], 3)["status"] != "admitted"
    assert _admit(flipped, ["i32", "i8"], ["i16"], 3, readout="i16")["status"] != "admitted"
