"""Concrete envelopes improve a sufficient proof without using cancellation."""

import pytest

from merlin.targetgen.operation_numerics import integer_matrix_partial_sum_bound

SEMANTICS = {"internal_arithmetic": {"mac_result_bits": 20, "signed_operand_bits": 8}}


def test_disjoint_operand_peaks_do_not_require_a_fictitious_large_product():
    result = integer_matrix_partial_sum_bound(
        SEMANTICS, lhs_shape=(1, 64), rhs_shape=(64, 1),
        lhs_values=[127] * 32 + [0] * 32, rhs_values=[0] * 32 + [127] * 32,
        initial_values=[17],
    )
    assert result["global_maximum_bound"] > result["signed_positive_limit"]
    assert result["status"] == "proven_safe" and result["bound"] == 17


def test_cancellation_cannot_hide_a_large_internal_sum():
    result = integer_matrix_partial_sum_bound(
        SEMANTICS, lhs_shape=(1, 64), rhs_shape=(64, 1),
        lhs_values=[127] * 64, rhs_values=[127] * 32 + [-127] * 32,
    )
    assert result["status"] == "may_overflow"
    assert result["bound"] == 64 * 127 * 127


def test_negative_minimum_and_nonzero_initial_values_are_included():
    result = integer_matrix_partial_sum_bound(
        SEMANTICS, lhs_shape=(1, 1), rhs_shape=(1, 1),
        lhs_values=[-128], rhs_values=[-128], initial_values=[-524287],
    )
    assert result["status"] == "may_overflow" and result["bound"] == 524287 + 16384


@pytest.mark.parametrize("mutation", ["length", "width", "float", "budget"])
def test_unknown_or_invalid_operand_proofs_never_approve(mutation):
    lhs, rhs = [1], [1]
    if mutation == "length":
        lhs = [1, 1]
    elif mutation == "width":
        rhs = [128]
    elif mutation == "float":
        lhs = [1.0]
    if mutation == "budget":
        result = integer_matrix_partial_sum_bound(
            SEMANTICS, lhs_shape=(1, 2), rhs_shape=(2, 1), lhs_values=[1, 1], rhs_values=[1, 1], max_products=1
        )
        assert result["status"] == "unknown"
    else:
        with pytest.raises(ValueError):
            integer_matrix_partial_sum_bound(
                SEMANTICS, lhs_shape=(1, 1), rhs_shape=(1, 1), lhs_values=lhs, rhs_values=rhs
            )
