"""The value-and-index max_pool2d host row is a proposal: it must never count as a reviewed admission."""

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "examples" / "gemmini" / "platform" / "host-capabilities-portable-v3.yaml"
ROW = "host_f32_max_pool2d_with_i64_indices_rank4"


@pytest.fixture(scope="module")
def shipped():
    return yaml.safe_load(PROFILE.read_bytes())


def _admit(document, operands, results, rank):
    raw = yaml.safe_dump(document).encode()
    selection = {
        "int8_w8a8": {
            "capability_spec": document,
            "capability_spec_sha256": hashlib.sha256(raw).hexdigest(),
            "package_sha256": document["compiler"]["package_sha256"],
            "dtype_strategy": document["compiler"]["dtype_strategy"],
        }
    }
    signature = {"family": "reduction", "ordered_operand_dtypes": operands, "ordered_result_dtypes": results,
                 "rank": rank}
    return admit_host_operation(
        selection, {"frontend_op": "aten.max_pool2d_with_indices.default", "mlir_operation": "linalg.generic"},
        signature)


def _flipped(shipped):
    flipped = copy.deepcopy(shipped)
    for row in flipped["operations"]:
        if row["id"] == ROW:
            row["status"] = "reviewed"
            row["numerical_contract"]["status"] = "reviewed"
    return flipped


def test_row_is_proposed_and_marked_not_reviewed(shipped):
    validate_host_capabilities(shipped)
    rows = {r["id"]: r for r in shipped["operations"]}
    row = rows[ROW]
    assert row["status"] == "unreviewed" and row["numerical_contract"]["status"] == "unreviewed"
    assert "NOT REVIEWED" in row["numerical_contract"]["evidence"]["reviewed_by"]
    assert row["numerical_contract"]["semantics"]["equal_values"] == "first_maximum"
    # The reviewed values-only row is untouched.
    assert rows["host_f32_value_only_max_pool"]["status"] == "reviewed"


def test_unreviewed_row_is_never_a_reviewed_admission_and_review_would_make_it_one(shipped):
    before = _admit(shipped, ["f32", "i64"], ["f32", "i64"], 4)
    assert not (before["status"] == "admitted" and before["reviewed"] is True)
    after = _admit(_flipped(shipped), ["f32", "i64"], ["f32", "i64"], 4)
    assert after["status"] == "admitted" and after["reviewed"] is True


def test_other_forms_stay_refused_even_when_reviewed(shipped):
    flipped = _flipped(shipped)
    assert _admit(flipped, ["f32", "i64"], ["f32", "i64"], 3)["status"] != "admitted"
    assert _admit(flipped, ["f16", "i64"], ["f16", "i64"], 4)["status"] != "admitted"
    assert _admit(flipped, ["f32", "i64"], ["f32", "i32"], 4)["status"] != "admitted"
