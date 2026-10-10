"""Cheap closed-version and scalar-kind checks before native transpose work."""

import copy

import pytest
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_reference_roster as R

from merlin.targetgen.original_transpose_reference import OriginalTransposeReferencePolicy


def record():
    return OriginalTransposeReferencePolicy(
        "aten.transpose.int", ("float32",), ("float32",), "rank_two_axis_swap", True, "exact_element_storage_bits"
    ).record()


def test_new_policy_is_explicit_and_cannot_enter_legacy_selection():
    original = record()
    assert P.policy(original, transpose=True).record() == original
    for schema in (P.SCHEMA, P.BATCH_SCHEMA, P.POINTWISE_SCHEMA):
        with pytest.raises(ValueError):
            P.policy(original, pointwise=schema == P.POINTWISE_SCHEMA)
    assert R.record_schema({"schema": P.TRANSPOSE_SCHEMA}) == R.TRANSPOSE_SCHEMA
    assert R.record_schema({"schema": P.POINTWISE_SCHEMA}) == R.POINTWISE_SCHEMA


@pytest.mark.parametrize(
    "key,value",
    [
        ("finite_only", 1),
        ("finite_only", 1.0),
        ("movement", True),
        ("operand_dtypes", "float32"),
        ("readout_dtypes", [True]),
        ("comparison", None),
    ],
)
def test_new_policy_refuses_scalar_and_storage_aliases_before_native_work(key, value):
    changed = record()
    changed[key] = value
    with pytest.raises(ValueError):
        P.policy(changed, transpose=True)


@pytest.mark.parametrize("field", ["operation", "movement", "finite_only", "comparison"])
def test_new_policy_never_defaults_missing_semantics(field):
    changed = record()
    changed.pop(field)
    with pytest.raises(ValueError):
        P.policy(changed, transpose=True)


def test_policy_declaration_is_copied_and_unsupported_storage_refuses():
    original = record()
    selected = P.policy(original, transpose=True)
    copied = copy.deepcopy(original)
    original["operand_dtypes"][0] = "int8"
    assert selected.record() == copied
    copied["operand_dtypes"] = copied["readout_dtypes"] = ["float16"]
    with pytest.raises(ValueError):
        P.policy(copied, transpose=True).verify()
