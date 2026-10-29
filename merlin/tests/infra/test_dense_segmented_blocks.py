"""Physical K bounds retain legal segmented M/N choices without enlarging stores."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER
from test_dense_segmented_pipeline import chosen, inputs

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_segmented_pipeline import _assignment, dense_segmented_blocks
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks


def geometry():
    p = DenseProblem(1, 32, 64, 4, 4, 4, 1, 1, 4)
    s = DenseLoopStorage(DenseMemoryView("lhs", "spad", 4, 0, 256),
        DenseMemoryView("rhs", "spad", 4, 256, 256),
        DenseMemoryView("acc", "acc", 16, 0, 64))
    r = DenseResources(2048, 1024, 1, 16, 8, partial_output_tiles=True, partial_output_blocks=True)
    return p, r, s


def test_smaller_physical_k_keeps_wide_n_that_post_filter_loses():
    p, r, s = geometry()
    old = next(b for b in dense_blocks(p, r) if b.n == 16)
    assert old.k == 7
    with pytest.raises(ValueError):
        _assignment(p, old, r, s)
    new = next(b for b in dense_segmented_blocks(p, r, s) if b.n == 16)
    assert new.k == 2
    lhs, rhs = _assignment(p, new, r, s)
    assert lhs.capacity == 32
    assert [v.capacity for v in rhs] == [128, 128]
    assert rhs[-1].base + rhs[-1].capacity == s.rhs.base + s.rhs.capacity
    assert r.operand_bytes == 2048 and s.rhs.capacity == 256


def test_byte_budget_alignment_and_lhs_custody_are_independent_limits():
    p, r, s = geometry()
    tight = replace(r, operand_bytes=128+2*64*4)
    assert next(b for b in dense_segmented_blocks(p, tight, s) if b.n == 16).k == 1
    assert not dense_segmented_blocks(p, r, replace(s, lhs=replace(s.lhs, capacity=31)))
    aligned = replace(s, rhs=replace(s.rhs, alignment=64))
    assert all((b.k*b.n*p.tile_k) % 64 == 0 for b in dense_segmented_blocks(p, r, aligned))
    alias = replace(s, rhs=replace(s.rhs, base=0))
    assert not dense_segmented_blocks(p, r, alias)
    assert not dense_segmented_blocks(p, r, replace(s, accumulator=None))


def test_selected_physical_choice_runs_existing_full_source_and_epoch_owner():
    p, r, s, _, admission = inputs((5, 14, 13), 2)
    r = replace(r, max_k_blocks=8)
    new = next(b for b in dense_segmented_blocks(p, r, s) if b.m == b.n == 2)
    assert new.k == 4  # original RHS rows, not the arbitrary max-K exploration bound
    result = chosen(p, r, s, new, admission)
    result.revalidate()
    result.materialize_scf().verify()
    assert result.realization["source"]["output_tiles"] == p.tiles[0]*p.tiles[2]
    assert result.realization["source"]["configuration_identity"] == ADAPTER.configuration_identity
