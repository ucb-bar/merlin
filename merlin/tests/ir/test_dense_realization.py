"""Contraction planning checked against an enumerated, target-free traversal."""

from dataclasses import replace

import pytest

from merlin.transforms.dense_planning import (
    DenseChoice,
    DenseEmission,
    DenseProblem,
    DenseResources,
    choose_dense_block,
    dense_blocks,
    verify_dense_emission,
)


def _traverse(problem, block):
    """Count real edge transfers and evaluate the retained-panel algorithm."""
    p = problem
    tm, tk, tn = p.tile_m, p.tile_k, p.tile_n
    lhs = [[(i + k * 3) % 7 - 3 for k in range(p.k)] for i in range(p.m)]
    rhs = [[(k * 2 + j) % 5 - 2 for j in range(p.n)] for k in range(p.k)]
    output = [[0] * p.n for _ in range(p.m)]
    visits = [[0] * p.n for _ in range(p.m)]
    loads = [0, 0, 0]
    stores = payload = 0
    for m in range(0, p.m, block.m * tm):
        ms = range(m, min(p.m, m + block.m * tm))
        for n in range(0, p.n, block.n * tn):
            ns = range(n, min(p.n, n + block.n * tn))
            for i in ms:
                for j in ns:
                    output[i][j] = j - 2 if p.bias else 0
            for i in range(m, ms.stop, tm):
                for j in range(n, ns.stop, tn):
                    stores += 1
                    loads[2] += int(p.bias)
            for k in range(0, p.k, block.k * tk):
                ks = range(k, min(p.k, k + block.k * tk))
                # One immutable panel per output block; no per-output reloads.
                a = {(i, r): lhs[i][r] for i in ms for r in ks}
                b = {(r, j): rhs[r][j] for r in ks for j in ns}
                payload += len(a) * p.lhs_bytes + len(b) * p.rhs_bytes
                loads[0] += len(range(m, ms.stop, tm)) * len(range(k, ks.stop, tk))
                loads[1] += len(range(n, ns.stop, tn)) * len(range(k, ks.stop, tk))
                for i in ms:
                    for j in ns:
                        for r in ks:
                            assert visits[i][j] == r  # Preserve increasing source K.
                            output[i][j] += a[i, r] * b[r, j]
                            visits[i][j] += 1
    expected = [
        [sum(a * b for a, b in zip(row, column, strict=True)) + (j - 2 if p.bias else 0)
         for j, column in enumerate(zip(*rhs, strict=True))]
        for row in lhs
    ]
    assert output == expected
    assert all(count == p.k for row in visits for count in row)
    return loads, stores, payload


@pytest.mark.parametrize("tile", [(2, 3, 4), (4, 2, 3), (3, 4, 2)])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_every_candidate_preserves_coverage_order_and_exact_traffic(tile, tail, bias):
    tm, tk, tn = tile
    p = DenseProblem(tm * 3 - tail, tk * 3 - tail, tn * 3 - tail, tm, tk, tn, 1, 2, 4, bias)
    r = DenseResources(2048, 2048, max_k_blocks=3,
                       partial_output_tiles=True, partial_output_blocks=True)
    candidates = dense_blocks(p, r)
    assert len(candidates) == 9
    for block in candidates:
        loads, stores, payload = _traverse(p, block)
        assert loads == [block.lhs_loads, block.rhs_loads, block.bias_loads]
        assert stores == block.output_tiles
        assert payload == block.requested_operand_bytes


def _selection():
    p = DenseProblem(12, 9, 12, 3, 3, 3, 1, 2, 4, True)
    r = DenseResources(1024, 1024)
    choice = choose_dense_block(p, r, static_budget=64, price=lambda block: 24)
    b = choice.block
    return choice, DenseEmission(20, b.lhs_loads, b.rhs_loads, b.bias_loads,
                                b.output_tiles, b.operand_bytes, b.accumulator_bytes)


def test_realization_accepts_a_conservative_command_bound():
    choice, emission = _selection()
    verify_dense_emission(choice, emission, static_budget=24)


@pytest.mark.parametrize("field", ["lhs_loads", "rhs_loads", "bias_loads", "output_tiles"])
def test_realization_requires_each_selected_transfer_role(field):
    choice, emission = _selection()
    with pytest.raises(ValueError, match=field):
        verify_dense_emission(choice, replace(emission, **{field: getattr(emission, field) + 1}), static_budget=64)


def test_equal_total_loads_do_not_hide_a_changed_operand_role():
    choice, emission = _selection()
    wrong = replace(emission, lhs_loads=emission.lhs_loads + 1, rhs_loads=emission.rhs_loads - 1)
    with pytest.raises(ValueError, match="lhs_loads"):
        verify_dense_emission(choice, wrong, static_budget=64)


@pytest.mark.parametrize("role", ["operand", "accumulator"])
@pytest.mark.parametrize("missing", [False, True])
def test_storage_overflow_or_missing_observation_refuses(role, missing):
    choice, emission = _selection()
    key = role + "_high_water"
    wrong = replace(emission, **{key: 0 if missing else getattr(emission, key) + 1})
    with pytest.raises(ValueError, match=role):
        verify_dense_emission(choice, wrong, static_budget=64)


def test_both_static_price_and_caller_budget_are_enforced():
    choice, emission = _selection()
    with pytest.raises(ValueError, match="command"):
        verify_dense_emission(choice, emission, static_budget=19)
    with pytest.raises(ValueError, match="command"):
        verify_dense_emission(DenseChoice(choice.block, 19, 1), emission, static_budget=64)


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_invalid_observations_refuse(value):
    _, emission = _selection()
    with pytest.raises(ValueError, match="nonnegative integers"):
        replace(emission, lhs_loads=value)


def test_exact_capacity_and_single_token_fallback():
    p = DenseProblem(4, 4, 4, 2, 2, 2, 1, 1, 4)
    r = DenseResources(16, 64)
    chosen = choose_dense_block(p, r, static_budget=20, price=lambda block: 8)
    assert chosen and chosen.block.accumulator_bytes == 64
    assert choose_dense_block(p, replace(r, operand_bytes=7), static_budget=20, price=lambda b: 8) is None
    assert choose_dense_block(replace(p, m=1, n=1), replace(r, partial_output_tiles=True),
                              static_budget=20, price=lambda b: 8) is None
