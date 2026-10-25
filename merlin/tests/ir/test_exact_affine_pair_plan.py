from dataclasses import replace

import numpy as np
import pytest

from merlin.llvmlower.quantized_affine_pair import (
    predictor_table,
    propose_exact_affine_pair,
    source_table,
)


def propose(scales=(0.5, 0.25, 1.0), **overrides):
    limits = dict(max_denominator=128, coefficient_max=3, max_terms=16, accumulator_abs_max=32767)
    limits.update(overrides)
    return propose_exact_affine_pair(*scales, **limits)


@pytest.mark.parametrize("relu", [False, True])
def test_actual_constant_terms_and_readout_equal_every_ordered_source_pair(relu):
    plan = propose(relu=relu)
    assert plan is not None
    proof = plan.revalidate()
    assert proof["checked_pairs"] == 65536
    assert proof["target_execution"] == "not_established"
    assert sum(w for axis, w in plan.terms if axis == 0) == plan.p
    assert sum(w for axis, w in plan.terms if axis == 1) == plan.q
    assert all(0 < w <= 3 for _, w in plan.terms)
    assert np.array_equal(
        source_table(*plan.source_scales, relu=relu), predictor_table(plan.p, plan.q, plan.scale, relu=relu)
    )


def test_resource_limits_refuse_instead_of_accepting_a_local_error_budget():
    assert propose(max_terms=1) is None
    assert propose(accumulator_abs_max=127) is None
    with pytest.raises(ValueError, match="finite"):
        propose(max_denominator=True)


@pytest.mark.parametrize(
    "mutation",
    [
        dict(source_scales=(0.5, 0.25, 0.875)),
        dict(terms=((0, 1), (1, 1))),
        dict(source_table_sha256="0" * 64),
        dict(accumulator_abs_max=127),
        dict(coefficient_max=0),
    ],
)
def test_source_order_constant_band_and_domain_digest_mutations_refuse(mutation):
    plan = propose()
    with pytest.raises(ValueError):
        replace(plan, **mutation).revalidate()


def test_repeated_operand_terms_are_nonzero_and_preserve_coefficient():
    plan = propose((0.75, 0.125, 1.0), coefficient_max=2)
    assert plan is not None
    assert len([term for term in plan.terms if term[0] == 0]) > 1
    assert plan.revalidate()["required_accumulator_abs"] <= 32767


def test_positive_small_lhs_skips_zero_numerator_initial_rational_proposal():
    plan = propose((0.125, 1.0, 1.0))
    assert plan is not None
    assert (plan.p, plan.q) == (1, 8)
    assert plan.revalidate()["checked_pairs"] == 65536
