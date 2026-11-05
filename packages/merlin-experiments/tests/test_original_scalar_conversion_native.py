"""Ordinary registered conversion of independent scalar calls; no admission."""

import copy
from dataclasses import replace
from pathlib import Path

import original_scalar_conversion_fixtures as F
import pytest

from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads

native_originals = F.native_originals


def test_actual_public_schema_factory_registered_importer_complete_products_and_stock_parser(native_originals):
    record = native_originals.record()
    assert len(record["members"]) == 18
    assert sum(row["state"] == "registered_source_checked" for row in record["members"]) == 15
    assert sum(row["state"] == "unavailable" for row in record["members"]) == 3
    assert {row["cohort"] for row in record["members"]} == {"functional_guard", "withheld_transfer"}
    assert {
        row["correspondence"]["coefficient_f32_bits"]
        for row in record["members"]
        if row["state"] == "registered_source_checked"
    } == {"3f800000", "400f1bbd", "80000000", "3dcccccd"}
    assert "original_numerical_policy_and_reference" in record["unavailable_requirements"]
    assert "registered_torch_tensor_literal_source_correspondence" in record["unavailable_requirements"]


def test_copied_live_object_and_modified_saved_receipt_cannot_issue_registered_conversion(native_originals):
    with pytest.raises(ValueError, match="actual live"):
        replace(native_originals).record()
    record = loads(native_originals.receipt_json)
    record["members"].pop()
    with pytest.raises(ValueError, match="actual live"):
        replace(native_originals, receipt_json=canonical_json(record)).record()


@pytest.mark.parametrize("product", ["source", "trace", "registry"])
def test_actual_observed_product_drift_cannot_be_accepted(native_originals, product):
    # Existing bytes are restored to preserve this independent fixture for the
    # next drift control; each attempted verification itself reopens them.
    record = loads(native_originals.receipt_json)
    row = next(row for row in record["members"] if row["state"] == "registered_source_checked")
    path = Path(row["products"][product]["path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError):
            native_originals.record()
    finally:
        path.write_bytes(original)


def test_selection_drift_cannot_change_original_members_or_native_bounds(native_originals):
    path = native_originals.selection
    original = path.read_bytes()
    try:
        selected = copy.deepcopy(loads(original))
        selected["budget"]["max_members"] = 17
        path.write_bytes(canonical_json(selected))
        with pytest.raises(ValueError):
            native_originals.record()
    finally:
        path.write_bytes(original)
