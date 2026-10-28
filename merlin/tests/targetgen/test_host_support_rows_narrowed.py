"""The tensor.* support-op host rows admit only the ranks and dtypes observed in the contraction and reduce programs."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

PROFILE = Path(__file__).resolve().parents[3] / "examples/gemmini/platform/host-capabilities-portable-v3.yaml"
SUPPORT = {"tensor.empty", "tensor.splat", "tensor.extract", "tensor.from_elements", "tensor.collapse_shape",
           "tensor.expand_shape", "tensor.extract_slice", "tensor.insert_slice"}


@pytest.fixture(scope="module")
def doc():
    return yaml.safe_load(PROFILE.read_bytes())


def _rows(doc):
    return [r for r in doc["operations"] if r["id"].startswith("host_support_") and r["ops"][0] in SUPPORT]


def _admit(doc, mlir, operands, results, rank):
    raw = yaml.safe_dump(doc).encode()
    selection = {"int8_w8a8": {"capability_spec": doc, "capability_spec_sha256": hashlib.sha256(raw).hexdigest(),
                               "package_sha256": doc["compiler"]["package_sha256"],
                               "dtype_strategy": doc["compiler"]["dtype_strategy"]}}
    signature = {"family": "movement", "ordered_operand_dtypes": operands, "ordered_result_dtypes": results,
                 "rank": rank}
    return admit_host_operation(selection, {"frontend_op": "aten.view.default", "mlir_operation": mlir}, signature)


def test_profile_validates(doc):
    validate_host_capabilities(doc)


def test_every_support_row_declares_the_observed_ranks_and_keeps_its_obligations(doc):
    rows = _rows(doc)
    assert len(rows) == 38
    for row in rows:
        ranks = row["signature"]["ranks"]
        assert ranks and ranks == sorted(set(ranks)) and max(ranks) <= 7
        evidence = row["numerical_contract"]["evidence"]
        assert "narrowing" in evidence and "open_obligation" in evidence
        assert row["status"] == "reviewed"


def test_unobserved_dtypes_have_no_row(doc):
    ids = {r["id"] for r in _rows(doc)}
    assert "host_support_empty_i1" not in ids and "host_support_splat_i1" not in ids


def test_each_row_admits_its_observed_ranks_and_refuses_others(doc):
    for row in _rows(doc):
        sig = row["signature"]
        operands, results = sig["ordered_operand_dtypes"], sig["ordered_result_dtypes"]
        for rank in sig["ranks"]:
            assert _admit(doc, row["ops"][0], operands, results, rank)["status"] == "admitted", (row["id"], rank)
        for rank in (8, 9):
            assert _admit(doc, row["ops"][0], operands, results, rank)["status"] != "admitted", (row["id"], rank)


def test_other_dtypes_stay_refused(doc):
    assert _admit(doc, "tensor.empty", [], ["i1"], 2)["status"] != "admitted"
    assert _admit(doc, "tensor.splat", ["i1"], ["i1"], 0)["status"] != "admitted"
    assert _admit(doc, "tensor.collapse_shape", ["i32"], ["i32"], 1)["status"] != "admitted"
