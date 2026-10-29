"""Bank-separated full-K panel reservations for aligned, tail and capacity cases."""

import pytest

from merlin.compile.scheduling.dense_resident_panel import plan_dense_resident_panel, plan_dense_stationary_panel


def plan(m, k, n, **changes):
    return plan_dense_resident_panel(m, k, n, **(dict(block=16, operand_rows=16384,
        bank_rows=4096, accumulator_rows=1024) | changes))


@pytest.mark.parametrize("shape", [(12544, 64, 256), (3136, 128, 512), (12544, 256, 64),
    (3136, 512, 128), (784, 256, 1024), (512, 512, 512), (17, 641, 33), (1, 1, 1)])
def test_exact_source_geometry_full_k_and_disjoint_bank_reservations(shape):
    selected = plan(*shape)
    assert selected is not None and selected.padded_k >= shape[1]
    assert selected.padded_k % 16 == selected.rhs_base % 4096 == 0
    assert selected.rhs_base >= selected.padded_k
    assert selected.rhs_base + selected.rhs_rows <= selected.operand_rows
    assert selected.columns > 0


def test_large_rhs_uses_column_panels_instead_of_spilling_or_losing_k():
    selected = plan(512, 512, 512)
    assert selected.columns == 384 and selected.padded_k == 512
    assert selected.rhs_base + selected.rhs_rows == selected.operand_rows


@pytest.mark.parametrize("shape", [(1, 8193, 33), (0, 16, 16), (1, True, 16), (1 << 62, 16, 16)])
def test_unsupported_capacity_or_geometry_declines(shape):
    assert plan(*shape) is None


def test_incomplete_bank_and_accumulator_reservations_decline():
    assert plan(17, 17, 17, bank_rows=4095) is None
    assert plan(17, 17, 17, accumulator_rows=15) is None


@pytest.mark.parametrize("shape", [(12544, 64, 256), (3136, 128, 512), (12544, 256, 64),
    (3136, 512, 128), (784, 256, 1024), (512, 512, 512), (17, 641, 33)])
def test_stationary_groups_have_disjoint_operands_and_live_accumulator_capacity(shape):
    selected = plan_dense_stationary_panel(*shape, block=16, operand_rows=16384,
                                           bank_rows=4096, accumulator_rows=1024)
    assert selected is not None
    assert selected.m_blocks * selected.padded_k <= selected.rhs_base
    assert selected.rhs_base + selected.rhs_rows <= selected.operand_rows
    assert selected.row_group <= 1024
    assert selected.m_blocks >= 1


def test_stationary_planning_does_not_sacrifice_extra_lhs_replays_for_larger_row_groups():
    selected = plan_dense_stationary_panel(512, 512, 512, block=16, operand_rows=16384,
                                           bank_rows=4096, accumulator_rows=1024)
    assert selected.m_blocks == 16 and selected.columns == 256


@pytest.mark.parametrize('shape', [(17, 17, 19), (3, 17, 1105), (512, 512, 512)])
def test_output_panel_reserves_distinct_full_reduction_column_blocks(shape):
    from merlin.compile.scheduling.dense_resident_panel import plan_dense_output_panel

    selected = plan_dense_output_panel(*shape, block=16, operand_rows=16384,
                                      bank_rows=4096, accumulator_rows=1023)
    assert selected is not None
    assert selected.columns <= 1008
    assert selected.rhs_base + selected.rhs_rows <= selected.operand_rows
    assert selected.padded_k >= shape[1]


@pytest.mark.parametrize('shape', [(12544, 64, 256), (512, 512, 512), (64, 64, 256),
                                   (65, 17, 257), (17, 641, 33), (3, 17, 1105)])
def test_joint_row_column_products_fit_all_live_storage(shape):
    from merlin.compile.scheduling.dense_resident_panel import plan_dense_joint_panel

    selected = plan_dense_joint_panel(*shape, block=16, operand_rows=16384,
                                     bank_rows=4096, accumulator_rows=1024)
    assert selected is not None
    assert selected.m_blocks * selected.padded_k <= selected.rhs_base
    assert selected.rhs_base + selected.rhs_rows <= selected.operand_rows
    assert selected.m_blocks * selected.n_blocks * selected.block <= 1024


def test_joint_plan_keeps_all_tall_columns_live_while_reusing_weights():
    from merlin.compile.scheduling.dense_resident_panel import plan_dense_joint_panel

    selected = plan_dense_joint_panel(12544, 64, 256, block=16, operand_rows=16384,
                                     bank_rows=4096, accumulator_rows=1024)
    assert selected.m_blocks == 4 and selected.n_blocks == 16
