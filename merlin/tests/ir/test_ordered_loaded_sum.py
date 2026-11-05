"""Public scalar domains; no model payload or observed operand ranges."""
import hashlib

import pytest

from merlin.llvmlower.quantized_affine_pair import (
    prove_loaded_sum,
    prove_ordered_loaded_sum,
    source_table,
)


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("source_scale", [1.0, 0.25, 2.0])
def test_original_scale_triple_can_prove_existing_bulk_loaded_sum(source_scale, relu):
    proof = prove_ordered_loaded_sum(
        source_scale, source_scale, source_scale,
        load_lhs_scale=1.0, load_rhs_scale=1.0, store_scale=1.0, relu=relu,
    )
    assert proof["schema"] == "merlin.ordered_loaded_operand_sum.v1"
    assert proof["source"]["output_scale"] == source_scale
    assert proof["checked_pairs"] == 65536
    assert proof["bound_lsb"] == proof["max_error_lsb"] == 0
    expected = source_table(source_scale, source_scale, source_scale, relu=relu)
    assert proof["source_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert proof["source_sha256"] == proof["proposed_sha256"]
    assert proof["target_execution"] == "not_established"


def test_fractional_load_rounding_cannot_replace_original_ordered_arithmetic():
    with pytest.raises(ValueError, match="source error bound"):
        prove_ordered_loaded_sum(
            0.125, 0.375, 0.5, load_lhs_scale=0.25,
            load_rhs_scale=0.75, store_scale=1.0,
        )


def test_source_scale_triple_changes_the_proof_instead_of_collapsing_ratios():
    # Load-rounds happen BEFORE summation; equal effective coefficients alone
    # are not a proof of the separately rounded original source expression.
    legacy = prove_loaded_sum(
        0.5, 0.5, load_lhs_scale=0.5, load_rhs_scale=0.5,
        store_scale=1.0, bound_lsb=1,
    )
    assert legacy["max_error_lsb"] == 1
    assert legacy["schema"] == "merlin.loaded_operand_sum.v1"
    with pytest.raises(ValueError, match="source error bound"):
        prove_ordered_loaded_sum(
            0.25, 0.25, 0.5, load_lhs_scale=0.5,
            load_rhs_scale=0.5, store_scale=1.0,
        )


@pytest.mark.parametrize("scale", [0.0, float("inf"), float("nan")])
def test_original_output_scale_must_be_positive_finite(scale):
    with pytest.raises(ValueError, match="binary32"):
        prove_ordered_loaded_sum(
            1.0, 1.0, scale, load_lhs_scale=1.0,
            load_rhs_scale=1.0, store_scale=1.0,
        )
