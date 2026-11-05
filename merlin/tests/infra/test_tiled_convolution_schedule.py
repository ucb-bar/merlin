"""Bounded convolution reuse: execute actual rows against independent direct convolution."""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from merlin.compile.scheduling.block_schedule import (
    AXES,
    BANK_ALIGNED,
    CONTIGUOUS,
    LHS,
    WEIGHT,
    BlockScheduleError,
    Compute,
    ConvContraction,
    Knobs,
    Load,
    M,
    Preload,
    Store,
    check_residency,
    execute,
    schedule_convolution,
    schedule_tiled_convolution,
)
from merlin.compile.scheduling.derive import geometry_from_address_space
from merlin.targetgen.address_space import derive_address_space


def geometry(edge, *, operand_tiles=24, output_tiles=12):
    # Neutral memory names, with roles resolved by facts rather than spelling.
    facts = {
        "schema_version": "2.0",
        "inputs": {},
        "facts": {
            "arrays": [{"name": "array", "rows": edge, "cols": edge}],
            "datapaths": [
                {"name": "input", "dtype": "i8", "evidence": "a UInt<8>"},
                {"name": "accumulator", "dtype": "i32", "evidence": "b SInt<32>"},
            ],
            "memories": [
                {"name": "a", "bytes": operand_tiles * edge**2, "depth": operand_tiles * edge // 2},
                {"name": "b", "bytes": output_tiles * edge**2 * 4, "depth": output_tiles * edge},
            ],
        },
    }
    return geometry_from_address_space(derive_address_space("synthetic", facts=facts))


def direct(conv, lhs, weight):
    image = lhs.reshape(conv.batch, conv.in_h, conv.in_w, conv.ci).astype(np.int64)
    weight = weight.reshape(conv.kh, conv.kw, conv.ci, conv.co).astype(np.int64)
    top, left, bottom, right = conv.padding
    padded = np.pad(image, ((0, 0), (top, bottom), (left, right), (0, 0)))
    sh, sw = conv.stride
    dh, dw = conv.dilation
    # Output extents calculated independently of ConvContraction's properties.
    height = (padded.shape[1] - (conv.kh - 1) * dh - 1) // sh + 1
    width = (padded.shape[2] - (conv.kw - 1) * dw - 1) // sw + 1
    out = np.empty((conv.batch, height, width, conv.co), dtype=np.int64)
    for b in range(conv.batch):
        for y in range(height):
            for x in range(width):
                patch = padded[
                    b, y * sh : y * sh + (conv.kh - 1) * dh + 1 : dh, x * sw : x * sw + (conv.kw - 1) * dw + 1 : dw
                ]
                out[b, y, x] = np.tensordot(patch, weight, axes=((0, 1, 2), (0, 1, 2)))
    return out.reshape(-1, conv.co)


def operands(conv):
    rng = np.random.default_rng(591)
    return (
        rng.integers(-4, 5, (conv.batch * conv.in_h * conv.in_w, conv.ci), dtype=np.int8),
        rng.integers(-4, 5, (conv.kh * conv.kw * conv.ci, conv.co), dtype=np.int8),
    )


@pytest.mark.parametrize("edge", [2, 4, 8])
@pytest.mark.parametrize("order", list(permutations(AXES)))
@pytest.mark.parametrize("lookahead", [0, 1, None])
def test_actual_addresses_match_direct_convolution_across_panels(edge, order, lookahead):
    conv = ConvContraction(2, 7, 8, edge + 1, 3, 2, 3 * edge + 1, stride=(2, 1), padding=(2, 1, 1, 0), dilation=(2, 1))
    result = schedule_tiled_convolution(conv, geometry(edge), Knobs(loop_order=order, lookahead_steps=lookahead))
    lhs, rhs = operands(conv)
    np.testing.assert_array_equal(execute(result.schedule, lhs, rhs), direct(conv, lhs, rhs))
    kt = conv.kh * conv.kw * ((conv.ci + edge - 1) // edge)
    assert result.k_blocks < kt, "exercise continued accumulators across reused operand panels"
    assert any(op.weight_row is None for op in result.schedule.ops if isinstance(op, Preload)) == (order[-1] == M)
    assert (
        sum(op.rows * op.cols for op in result.schedule.ops if isinstance(op, Load))
        == result.requested_operand_elements
    )
    assert result.panels[0].first_operation == 0
    assert result.panels[-1].end_operation == len(result.schedule.ops)
    for before, after in zip(result.panels, result.panels[1:]):
        assert before.end_operation == after.first_operation
    for panel in result.panels:
        stores = [
            op for op in result.schedule.ops[panel.first_operation : panel.end_operation] if isinstance(op, Store)
        ]
        assert len(stores) == (panel.m_blocks * panel.n_blocks if panel.k_start + panel.k_blocks == kt else 0)


@pytest.mark.parametrize("order", list(permutations(AXES)))
@pytest.mark.parametrize("load_once", [False, True])
def test_realized_transfer_volume_and_capacity(order, load_once):
    conv = ConvContraction(1, 5, 6, 5, 3, 3, 13, padding=(1, 1, 1, 1))
    g = geometry(4)
    knobs = Knobs(loop_order=order, load_on_index_change=load_once, placement=BANK_ALIGNED)
    result = schedule_tiled_convolution(conv, g, knobs)
    lhs, rhs = operands(conv)
    np.testing.assert_array_equal(execute(result.schedule, lhs, rhs), direct(conv, lhs, rhs))
    assert (
        sum(op.rows * op.cols for op in result.schedule.ops if isinstance(op, Load))
        == result.requested_operand_elements
    )
    for op in result.schedule.ops:
        if isinstance(op, Load):
            assert 0 <= op.row < op.row + op.rows <= g.operand_rows
        if isinstance(op, (Preload, Store)):
            assert 0 <= op.accumulator_row < op.accumulator_row + op.rows <= g.accumulator_rows
    a, b = result.schedule.regions[LHS], result.schedule.regions[WEIGHT]
    assert a[0] + a[1] <= b[0]


def test_tiling_removes_whole_operand_capacity_refusal_and_reduces_real_requests():
    conv = ConvContraction(1, 8, 8, 9, 3, 3, 17, padding=(1, 1, 1, 1))
    g = geometry(4, operand_tiles=64, output_tiles=32)
    with pytest.raises(BlockScheduleError, match="tiled"):
        schedule_convolution(conv, g)
    result = schedule_tiled_convolution(conv, g)
    mt, nt, kt = 16, 5, 27
    assert result.m_blocks > 1 and result.n_blocks > 1
    assert result.schedule.count(Compute) == mt * nt * kt
    assert result.schedule.count(Store) == mt * nt
    assert result.schedule.count(Load) < 2 * mt * nt * kt // 2
    assert sum(op.weight_row is not None for op in result.schedule.ops if isinstance(op, Preload)) < mt * nt * kt
    lhs, rhs = operands(conv)
    np.testing.assert_array_equal(execute(result.schedule, lhs, rhs), direct(conv, lhs, rhs))


def small():
    conv = ConvContraction(1, 5, 6, 5, 3, 3, 13, padding=(1, 1, 1, 1))
    return conv, schedule_tiled_convolution(conv, geometry(4)).schedule


@pytest.mark.parametrize("fault", ["operand", "continuation", "accumulator", "store"])
def test_independent_checker_rejects_mutated_physical_effects(fault):
    _, schedule = small()
    ops = list(schedule.ops)
    if fault == "operand":
        i = next(i for i, op in enumerate(ops) if isinstance(op, Compute))
        ops[i] = replace(ops[i], input_row=schedule.geometry.operand_rows - 1)
    elif fault == "continuation":
        i = next(i for i, op in enumerate(ops) if isinstance(op, Preload) and op.accumulate)
        ops[i] = replace(ops[i], accumulate=False)
    elif fault == "accumulator":
        i = next(i for i, op in enumerate(ops) if isinstance(op, Store))
        ops[i] = replace(ops[i], accumulator_row=ops[i].accumulator_row + 1)
    else:
        i = next(i for i, op in enumerate(ops) if isinstance(op, Store))
        del ops[i]
    with pytest.raises(BlockScheduleError):
        check_residency(replace(schedule, ops=tuple(ops)))


def test_direct_reference_detects_consistent_but_wrong_gather():
    conv, schedule = small()
    ops = list(schedule.ops)
    i = next(
        i
        for i, op in enumerate(ops)
        if isinstance(op, Load) and op.gather and any(x not in (None, (0, 0)) for x in op.gather)
    )
    old = ops[i]
    wrong = tuple((0, 0) if x is not None else None for x in old.gather)
    for j, op in enumerate(ops):
        if isinstance(op, Load) and op.role == LHS and op.block == old.block:
            ops[j] = replace(op, gather=wrong)
        if isinstance(op, Compute) and op.input_block == old.block:
            ops[j] = replace(op, input_gather=wrong)
    mutated = replace(schedule, ops=tuple(ops))
    # Residency alone is not a source proof: execution must also match the source.
    check_residency(mutated)
    lhs, rhs = operands(conv)
    assert not np.array_equal(execute(mutated, lhs, rhs), direct(conv, lhs, rhs))


@pytest.mark.parametrize(
    "field, value",
    [("stride", (0, 1)), ("dilation", (-1, 1)), ("padding", (0, -1, 0, 0)), ("stride", (1,)), ("ci", True)],
)
def test_invalid_geometry_refuses_before_expansion(field, value):
    conv = replace(ConvContraction(1, 5, 5, 4, 3, 3, 4), **{field: value})
    with pytest.raises(BlockScheduleError):
        schedule_tiled_convolution(conv, geometry(4))


def test_budgets_and_missing_capacity_refuse():
    conv = ConvContraction(1, 5, 6, 5, 3, 3, 13)
    for settings in ({"max_operations": 10}, {"candidate_budget": 1}, {"candidate_budget": True}):
        with pytest.raises(BlockScheduleError, match="budget"):
            schedule_tiled_convolution(conv, geometry(4), **settings)
    with pytest.raises(BlockScheduleError, match="does not fit"):
        schedule_tiled_convolution(conv, geometry(4, operand_tiles=1))
    with pytest.raises(BlockScheduleError, match="separate"):
        schedule_tiled_convolution(conv, replace(geometry(4), separate_accumulator_space=False))
    with pytest.raises(BlockScheduleError, match="operand_bank_rows"):
        schedule_tiled_convolution(conv, replace(geometry(4), operand_bank_rows=None), Knobs(placement=BANK_ALIGNED))


def test_single_reduction_and_whole_panel_needs_no_continuation():
    conv = ConvContraction(1, 2, 2, 3, 1, 1, 3)
    result = schedule_tiled_convolution(conv, geometry(4), Knobs(placement=CONTIGUOUS))
    assert len(result.panels) == 1 and result.k_blocks == 1
    assert all(not op.accumulate for op in result.schedule.ops if isinstance(op, Preload))
    lhs, rhs = operands(conv)
    np.testing.assert_array_equal(execute(result.schedule, lhs, rhs), direct(conv, lhs, rhs))
