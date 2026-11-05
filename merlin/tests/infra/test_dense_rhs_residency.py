"""Capacity arithmetic for full-RHS residency, independent of target encoding."""

import pytest

from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency


def plan(m=31, k=17, n=33, **kwargs):
    return plan_dense_rhs_residency(m, k, n, **(dict(block=16, operand_rows=16384,
        accumulator_rows=1024, rhs_base=256) | kwargs))


def test_padded_tiles_are_disjoint_from_lhs_and_cover_complete_rhs():
    selected = plan()
    assert selected.k_tiles == 2 and selected.n_tiles == 3
    assert selected.padded_k == 32 and selected.rhs_rows == 96
    assert selected.rhs_base >= selected.block
    assert selected.rhs_base + selected.rhs_rows <= selected.operand_rows


@pytest.mark.parametrize("change", [dict(m=0), dict(k=True), dict(n=-1), dict(rhs_base=8),
    dict(rhs_base=17), dict(accumulator_rows=15), dict(operand_rows=351),
    dict(max_static_rhs_tiles=5), dict(m=1 << 62)])
def test_resource_numeric_and_address_overflow_guards(change):
    dims = dict(m=31, k=17, n=33)
    dims.update({key: value for key, value in change.items() if key in dims})
    change = {key: value for key, value in change.items() if key not in dims}
    assert plan_dense_rhs_residency(**dims, **(dict(block=16, operand_rows=16384,
        accumulator_rows=1024, rhs_base=256, max_static_rhs_tiles=1024) | change)) is None


def test_exact_capacity_is_accepted_but_one_fewer_row_is_refused():
    assert plan(operand_rows=352) is not None
    assert plan(operand_rows=351) is None
