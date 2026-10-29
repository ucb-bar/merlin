"""Capacity-derived row panels consume one checked schedule and retain RHS loads."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER
from test_dense_array_retention import execute

from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_panel_pipeline import (
    DenseRetainedReductionAdmission,
    dense_m_panel_capacity,
    schedule_dense_m_panels,
)
from merlin.transforms.dense_planning import DenseProblem, DenseResources


def fixture(*, shape=(8, 6, 8), tile=(2, 3, 4), m_slots=2):
    problem = DenseProblem(*shape, *tile, 1, 1, 4)
    kt, nt = problem.tiles[1:]
    lhs = m_slots * kt * problem.tile_m
    rhs = kt * nt * problem.tile_k
    acc = m_slots * nt * problem.tile_m
    storage = DenseLoopStorage(
        DenseMemoryView("A", "operand-A", problem.tile_k, 0, lhs),
        DenseMemoryView("W", "operand-W", problem.tile_n, 100, rhs),
        DenseMemoryView("C", "partial-sums", problem.tile_n * 4, 0, acc),
    )
    resources = DenseResources(
        lhs * problem.tile_k + rhs * problem.tile_n,
        acc * problem.tile_n * 4,
        max_m_blocks=16,
        max_n_blocks=nt,
        max_k_blocks=kt,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    law = DenseRetainedReductionAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "neutral-original-ordered-reduction-v1",
        m_slots * kt * nt,
        (m_slots + nt) * kt,
        2,
        2,
    )
    return problem, resources, storage, law


def construct(p, r, s, law, **kwargs):
    return schedule_dense_m_panels(
        p,
        r,
        s,
        ADAPTER,
        law,
        n_tiles=p.tiles[2],
        name="multiple_rows",
        dtypes=("i8", "i8", "i32"),
        source_identity="a" * 64,
        external_buffers_disjoint=True,
        **kwargs,
    )


@pytest.mark.parametrize("shape,tile", [((8, 6, 8), (2, 3, 4)), ((18, 4, 10), (3, 2, 5))])
def test_two_geometries_derive_m_panels_and_reuse_weights_in_actual_counted_ir(shape, tile):
    p, r, s, law = fixture(shape=shape, tile=tile)
    capacity = dense_m_panel_capacity(p, r, s, n_tiles=p.tiles[2])
    assert capacity["maximum_m_tiles"] == 2
    assert capacity["maximum_m_rows"] == 2 * p.tile_m
    attempt = construct(p, r, s, law)
    result = attempt.result
    assert result is not None and attempt.feedback["counted_m_panels"] >= 2
    assert result.realization["source"]["rhs_loads"] == p.tiles[1] * p.tiles[2]
    assert result.realization["source"]["rhs_loads"] < (attempt.feedback["counted_m_panels"] * p.tiles[1] * p.tiles[2])
    assert "schedule.loop_transform" in result.realization["schedule_ir"]["text"]
    assert result.realization["schedule_ir"]["pipeline_kind"] == "retained_full_k"
    assert attempt.feedback["status"] == "checked_schedule_constructed"
    assert attempt.feedback["emitted"] == "not_observed"
    assert attempt.feedback["transfer_status"] == "not_established"
    execute(result.kernel, p)  # Independent arithmetic over every original projected command and output.
    result.materialize_scf().verify()
    result.revalidate()


def test_capacity_tail_and_law_limits_explain_decline_without_claiming_fallback_emission():
    p, r, s, law = fixture()
    no_law = construct(p, r, s, None)
    assert no_law.result is None
    assert "selected_read_only_operand_and_ordered_reduction_law" in no_law.feedback["extension_blockers"]
    assert no_law.feedback["emitted"] == "not_observed"
    too_wide = construct(p, r, s, law, m_tiles=3)
    assert too_wide.result is None and too_wide.feedback["status"] == "not_applied"
    tight = replace(s, accumulator=replace(s.accumulator, capacity=1))
    refused = construct(p, r, tight, law)
    assert refused.result is None
    assert "accumulator_private_rows" in refused.feedback["extension_blockers"]
    limited_law = replace(law, max_issued_compute_pairs=p.tiles[1] * p.tiles[2])
    attempt = construct(p, r, s, limited_law)
    assert attempt.result is not None and attempt.feedback["selected_m_tiles"] == 1
    assert attempt.feedback["extension_blockers"] == ["selected_issue_limit"]


def test_tail_panels_keep_complete_outputs_and_expose_their_extra_rhs_reload():
    p, r, s, law = fixture(shape=(10, 6, 8))
    attempt = construct(p, r, s, law)
    assert attempt.result is not None
    assert attempt.feedback["counted_m_panels"] == 3
    assert attempt.feedback["rhs_loads"] == 2 * p.tiles[1] * p.tiles[2]
    execute(attempt.result.kernel, p)


def test_failed_concrete_target_verification_never_falls_back_silently():
    p, r, s, law = fixture()
    with pytest.raises(ValueError, match="original|selected|configuration|identity|adapter"):
        construct(p, r, s, replace(law, hardware_identity="another-original-configuration"))


def test_selected_n_width_does_not_enumerate_unrelated_panel_choices():
    p, r, s, law = fixture(shape=(8, 6, 4096))
    r = replace(r, candidate_budget=8)
    s = replace(s, rhs=replace(s.rhs, capacity=12))
    attempt = schedule_dense_m_panels(p, r, s, ADAPTER, law, n_tiles=2,
        name="one_width", dtypes=("i8", "i8", "i32"), source_identity="b" * 64,
        external_buffers_disjoint=True)
    assert attempt.result is not None and attempt.feedback["selected_m_tiles"] == 2
    assert attempt.feedback["capacity"]["n_tiles"] == 2


def test_single_m_tile_does_not_claim_transfer_of_a_multi_panel_mechanism():
    p, r, s, law = fixture(shape=(1, 6, 8))
    attempt = construct(p, r, s, law)
    assert attempt.result is None and attempt.feedback["status"] == "not_applied"
    assert "multiple_m_tiles" in attempt.feedback["extension_blockers"]
