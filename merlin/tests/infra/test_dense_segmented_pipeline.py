"""Actual counted segmented operand commands, source cover and epoch hazards."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture

from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    DensePanelOverlapAdmission,
    DenseRetainedReductionAdmission,
    _serial_projection,
    verify_dense_panel_pipeline,
)
from merlin.compile.scheduling.dense_segmented_pipeline import (
    DenseSegmentedAdmission,
    build_dense_segmented_pipeline,
    schedule_dense_segmented_pipeline,
    verify_dense_segmented_pipeline,
)
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import Loop, instances
from merlin.transforms.dense_planning import dense_blocks


def inputs(shape=(5, 14, 13), segment=2):
    p, r, s = fixture(shape=shape)
    s = replace(s, lhs=replace(s.lhs, capacity=256), rhs=replace(s.rhs, capacity=48),
                accumulator=replace(s.accumulator, capacity=64))
    r = replace(r, operand_bytes=256 * s.lhs.unit_bytes + 48 * s.rhs.unit_bytes,
                accumulator_bytes=64 * s.accumulator.unit_bytes, max_m_blocks=2, max_n_blocks=2,
                max_k_blocks=segment)
    b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, segment, 2))
    overlap = DensePanelOverlapAdmission(ADAPTER.configuration_identity, ADAPTER.numerical_policy_identity,
                                        "synthetic-disjoint-epoch-law", 4096, 4096, 2, 2)
    reduction = DenseRetainedReductionAdmission(ADAPTER.configuration_identity, ADAPTER.numerical_policy_identity,
                                               "synthetic-ordered-commit-law", 4096, 4096, 2, 2)
    admission = DenseSegmentedAdmission(overlap, reduction, "synthetic-COMBINED-ordered-commit-and-disjoint-overlap")
    return p, r, s, b, admission


def chosen(p, r, s, b, a, prefetch=True, issue_order=None):
    plan = make_dense_schedule_plan(
        p, r, s, ADAPTER, source_identity="b" * 64, dtypes=("i8", "i8", "i32"), static_budget=128,
        block_tiles=(b.m, b.k, b.n), pipeline_kind="retained_lhs_segmented_k", pipeline_prefetch=prefetch,
        pipeline_issue_order=issue_order)
    return schedule_dense_segmented_pipeline(
        p, r, s, ADAPTER, a, name="segmented", dtypes=("i8", "i8", "i32"),
        schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)), source_identity="b" * 64,
        external_buffers_disjoint=True)


def lookahead_inputs(shape):
    p, r, s, _, admission = inputs(shape, 1)
    s = replace(s, rhs=replace(s.rhs, capacity=256))
    r = replace(r, operand_bytes=s.lhs.capacity * s.lhs.unit_bytes + s.rhs.capacity * s.rhs.unit_bytes,
                max_k_blocks=p.tiles[1])
    block = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, p.tiles[1], 2))
    return p, r, s, block, admission


@pytest.mark.parametrize("shape", [(5, 14, 9), (5, 14, 17), (5, 14, 29), (4, 6, 37), (4, 6, 41)])
def test_full_k_rhs_lookahead_has_actual_mixed_epochs_and_preserves_all_tail_products(shape):
    from merlin.compile.scheduling.dense_segmented_pipeline import OUTPUT_GROUP_PREFETCH

    p, r, s, b, admission = lookahead_inputs(shape)
    previous = chosen(p, r, s, b, admission)
    result = chosen(p, r, s, b, admission, issue_order=OUTPUT_GROUP_PREFETCH)
    assert previous.realization["overlapped_compute_pairs"] == 0
    assert result.realization["overlapped_compute_pairs"] > 0
    assert result.realization["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    for field in ("lhs_loads", "rhs_loads", "requested_load_bytes", "requested_store_bytes"):
        assert result.realization["source"][field] == previous.realization["source"][field]
    result.materialize_scf().verify()
    result.revalidate()


def test_full_k_rhs_lookahead_runs_the_independent_original_integer_executor(monkeypatch):
    import test_compact_dense as existing
    from merlin.compile.scheduling.dense_segmented_pipeline import OUTPUT_GROUP_PREFETCH

    original_fixture = existing.fixture

    def capacity_fixture(**kwargs):
        p, r, s = original_fixture(**kwargs)
        s = replace(s, lhs=replace(s.lhs, capacity=256), rhs=replace(s.rhs, capacity=256),
                    accumulator=replace(s.accumulator, capacity=64))
        r = replace(r, operand_bytes=256 * (s.lhs.unit_bytes + s.rhs.unit_bytes),
                    accumulator_bytes=64 * s.accumulator.unit_bytes,
                    max_m_blocks=2, max_n_blocks=2, max_k_blocks=p.tiles[1])
        return p, r, s

    def schedule(p, r, s, **kwargs):
        b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, p.tiles[1], 2))
        admission = lookahead_inputs((5, 14, 13))[-1]
        result = chosen(p, r, s, b, admission, issue_order=OUTPUT_GROUP_PREFETCH)
        return replace(result, kernel=_serial_projection(result.kernel), realization=result.realization["source"])

    monkeypatch.setattr(existing, "fixture", capacity_fixture)
    monkeypatch.setattr(existing, "schedule", schedule)
    existing.test_tails_reuse_and_independent_execution(("m", "n"), False)


def test_full_k_rhs_lookahead_rejects_missing_frontiers_and_current_slot_overwrite():
    from merlin.compile.scheduling.dense_segmented_pipeline import OUTPUT_GROUP_PREFETCH
    from merlin.compile.scheduling.tile_load_coalescing import _linear

    p, r, s, b, admission = lookahead_inputs((5, 14, 29))
    with pytest.raises(ValueError, match="prefetched epoch"):
        chosen(p, r, s, b, admission, prefetch=False, issue_order=OUTPUT_GROUP_PREFETCH)
    shorter = next(x for x in dense_blocks(p, replace(r, max_k_blocks=1)) if (x.m, x.n) == (2, 2))
    with pytest.raises(ValueError, match="full K"):
        chosen(p, r, s, shorter, admission, issue_order=OUTPUT_GROUP_PREFETCH)
    result = chosen(p, r, s, b, admission, issue_order=OUTPUT_GROUP_PREFETCH)
    second_slot = s.rhs.base + b.k * b.n * p.tile_k

    def overwrite(body):
        rewritten = []
        for statement in body:
            if type(statement) is Loop:
                statement = replace(statement, body=overwrite(statement.body))
            elif statement.instr == "weight_load":
                # Map every RHS write into slot0 while current slot0 readers
                # remain live. The original range/epoch owner must reject it.
                operation = ADAPTER.project(statement)
                if _linear(operation.row)[0] >= second_slot:
                    statement = ADAPTER.encode(replace(operation, row=operation.row + (s.rhs.base - second_slot)))
            rewritten.append(statement)
        return tuple(rewritten)

    with pytest.raises(ValueError, match="overlapping in-flight"):
        verify_dense_segmented_pipeline(replace(result.kernel, body=overwrite(result.kernel.body)),
            p, b, r, s, ADAPTER, admission, static_budget=128)


@pytest.mark.parametrize("shape,segment", [((5, 14, 13), 2), ((4, 27, 8), 2), ((4, 6, 8), 2), ((4, 7, 8), 2)])
def test_actual_segmented_source_tails_and_serial_control_have_identical_data_calls(shape, segment):
    p, r, s, b, a = inputs(shape, segment)
    actual = chosen(p, r, s, b, a)
    control = chosen(p, r, s, b, a, prefetch=False)
    from merlin.sched.ir import evaluate

    def calls(result):
        return [(c.instr, tuple((k, evaluate(v, env)) for k, v in c.args))
                for c, env in instances(result.kernel) if c.instr != DRAIN]
    assert calls(actual) == calls(control)
    assert actual.realization["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert actual.realization["source"]["lhs_loads"] == p.tiles[0] * p.tiles[1]
    assert actual.realization["source"]["rhs_loads"] == p.tiles[1] * p.tiles[2] * ((p.tiles[0] + b.m - 1) // b.m)
    if p.tiles[1] > segment:
        assert actual.realization["overlapped_compute_pairs"] > 0
        assert actual.realization["dynamic_epoch_drains"] < control.realization["dynamic_epoch_drains"]
    assert "retained_lhs_segmented_k" in actual.realization["schedule_ir"]["text"]
    actual.materialize_scf().verify()
    actual.revalidate()


def test_actual_scalar_execution_preserves_all_source_products_and_outputs(monkeypatch):
    import test_compact_dense as existing

    original_fixture = existing.fixture

    def capacity_fixture(**kwargs):
        p, r, s = original_fixture(**kwargs)
        s = replace(s, lhs=replace(s.lhs, capacity=256), rhs=replace(s.rhs, capacity=48),
                    accumulator=replace(s.accumulator, capacity=64))
        r = replace(r, operand_bytes=256 * s.lhs.unit_bytes + 48 * s.rhs.unit_bytes,
                    accumulator_bytes=64 * s.accumulator.unit_bytes, max_m_blocks=2, max_n_blocks=2, max_k_blocks=2)
        return p, r, s

    def segmented_schedule(p, r, s, **kwargs):
        b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, 2, 2))
        *_, admission = inputs()
        result = chosen(p, r, s, b, admission)
        return replace(result, kernel=_serial_projection(result.kernel), realization=result.realization["source"])

    monkeypatch.setattr(existing, "fixture", capacity_fixture)
    monkeypatch.setattr(existing, "schedule", segmented_schedule)
    existing.test_tails_reuse_and_independent_execution(("m", "n"), False)


def test_unproved_combination_alias_and_same_window_prefetch_refuse():
    p, r, s, b, a = inputs()
    kernel = build_dense_segmented_pipeline(p, b, r, s, ADAPTER, name="negative", dtypes=("i8", "i8", "i32"))
    for law in (a.overlap, a.reduction):
        with pytest.raises(ValueError, match="combined execution admission"):
            verify_dense_segmented_pipeline(kernel, p, b, r, s, ADAPTER, law, static_budget=128)
    with pytest.raises(ValueError, match="one primitive K"):
        verify_dense_panel_pipeline(kernel, p, b, r, s, ADAPTER, a.overlap, static_budget=128)
    shared = replace(s, rhs=replace(s.rhs, backing=s.lhs.backing, base=s.lhs.base))
    with pytest.raises(ValueError, match="resource|alias"):
        build_dense_segmented_pipeline(p, b, r, shared, ADAPTER, name="alias", dtypes=("i8", "i8", "i32"))
    with pytest.raises(ValueError, match="alias"):
        build_dense_segmented_pipeline(p, b, r, s, ADAPTER, name="same_slots", dtypes=("i8", "i8", "i32"),
                                       window_offsets=((0, 0), (0, 0)))
    serial = 0

    def walk(body):
        nonlocal serial
        result = []
        for stmt in body:
            if type(stmt) is Loop:
                result.append(replace(stmt, body=walk(stmt.body)))
            else:
                if stmt.instr == "weight_load":
                    serial += 1
                    if serial == 2:
                        stmt = replace(stmt, args=tuple((k, v + (-b.k * b.n * p.tile_k) if k == "dst" else v)
                                                        for k, v in stmt.args))
                result.append(stmt)
        return tuple(result)
    invalid = replace(kernel, body=walk(kernel.body))
    with pytest.raises(ValueError, match="overlapping in-flight"):
        verify_dense_segmented_pipeline(invalid, p, b, r, s, ADAPTER, a, static_budget=128)


def test_live_capacity_is_full_lhs_plus_two_rhs_segments_not_two_full_operands():
    p, r, s, b, _ = inputs((4, 27, 8), 2)
    # LHS holds all9 K tiles. Each RHS slot holds only2; the full9-tile
    # weight panel cannot fit this original assigned RHS view at all.
    s = replace(s, rhs=replace(s.rhs, capacity=2 * b.k * b.n * p.tile_k))
    kernel = build_dense_segmented_pipeline(p, b, r, s, ADAPTER, name="bounded", dtypes=("i8", "i8", "i32"))
    assert p.tiles[1] * b.n * p.tile_k > s.rhs.capacity
    short = replace(s, lhs=replace(s.lhs, capacity=b.m * p.tiles[1] * p.tile_m - 1))
    with pytest.raises(ValueError, match="escapes"):
        build_dense_segmented_pipeline(p, b, r, short, ADAPTER, name="small", dtypes=("i8", "i8", "i32"))
    assert sum(1 for _ in instances(kernel)) > 0


def test_existing_pure_store_grouping_composes_without_changing_mixed_epochs_or_capabilities():
    from merlin.compile.scheduling.dense_segmented_pipeline import batch_dense_segmented_readouts

    p, r, s, b, a = inputs()
    source = chosen(p, r, s, b, a)
    selected = replace(a, overlap=replace(a.overlap, max_issued_readouts=4,
                                         readout_configuration="ordered_snapshot_before_store_issue"),
                       reduction=replace(a.reduction, max_issued_readouts=4,
                                         readout_configuration="ordered_snapshot_before_store_issue"))
    actual = batch_dense_segmented_readouts(source, p, b, r, s, ADAPTER, selected)
    from merlin.sched.ir import evaluate

    def data(result):
        return [(c.instr, tuple((k, evaluate(v, env)) for k, v in c.args))
                for c, env in instances(result.kernel) if c.instr != DRAIN]
    assert data(actual) == data(source)
    assert actual.realization["readout_batching"]["removed_dynamic_drains"] > 0
    assert actual.realization["overlapped_compute_pairs"] == source.realization["overlapped_compute_pairs"]
    source_mixed = [e for e in source.realization["epochs"] if e["compute_pairs"] or e["future_or_initial_loads"]]
    final_mixed = [e for e in actual.realization["epochs"] if e["compute_pairs"] or e["future_or_initial_loads"]]
    assert source_mixed == final_mixed
    wrong = replace(selected, overlap=replace(selected.overlap, max_issued_loads=999))
    with pytest.raises(ValueError, match="operand or reduction capability"):
        batch_dense_segmented_readouts(source, p, b, r, s, ADAPTER, wrong)


def test_combined_law_enforces_the_smaller_component_readout_budget():
    from merlin.compile.scheduling.dense_panel_readout import _propose

    p, r, s, b, a = inputs()
    source = chosen(p, r, s, b, a)
    policy = "ordered_snapshot_before_store_issue"
    unequal = replace(a, overlap=replace(a.overlap, max_issued_readouts=4, readout_configuration=policy),
                      reduction=replace(a.reduction, max_issued_readouts=1, readout_configuration=policy))
    proposed, _ = _propose(source.kernel, ADAPTER, 4)
    with pytest.raises(ValueError, match="readout exceeds"):
        verify_dense_segmented_pipeline(proposed, p, b, r, s, ADAPTER, unequal, static_budget=128)


@pytest.mark.parametrize("shape", [(4, 12, 8), (3, 14, 7), (4, 27, 8)])
@pytest.mark.parametrize("issue_order", ["prefetch_rhs_between_columns", "prefetch_rhs_between_k_tiles"])
def test_typed_frontier_issue_reorders_real_calls_without_changing_cover_reductions_or_drains(shape, issue_order):
    from collections import Counter

    from merlin.sched.ir import evaluate

    p, r, s, b, a = inputs(shape, 2)
    original = chosen(p, r, s, b, a)
    interleaved = chosen(p, r, s, b, a, issue_order=issue_order)

    def epochs(result):
        groups, current = [], []
        for call, env in instances(result.kernel):
            if call.instr == DRAIN:
                groups.append(current)
                current = []
            else:
                current.append((call.instr, tuple((k, evaluate(v, env)) for k, v in call.args)))
        assert not current
        return groups
    before, after = epochs(original), epochs(interleaved)
    assert before != after
    assert Counter(c for group in before for c in group) == Counter(c for group in after for c in group)
    assert len(before) == len(after)
    assert interleaved.admission == original.admission
    assert interleaved.realization["source"]["output_tiles"] == original.realization["source"]["output_tiles"]
    assert (interleaved.realization["source"]["requested_load_bytes"]
            == original.realization["source"]["requested_load_bytes"])
    assert interleaved.realization["frontier_issue"]["regions"]
    for group in after:
        for index, (instruction, _args) in enumerate(group):
            if instruction == "preload":
                assert group[index + 1][0] == "compute"
    mixed_before = [g for g in before if any(c[0] == "compute" for c in g) and any(c[0] == "weight_load" for c in g)]
    mixed_after = [g for g in after if any(c[0] == "compute" for c in g) and any(c[0] == "weight_load" for c in g)]
    assert all(next(i for i, c in enumerate(a) if c[0] == "weight_load")
               < next(i for i, c in enumerate(b) if c[0] == "weight_load")
               for a, b in zip(mixed_after, mixed_before, strict=True))
    if p.tiles[1] % b.k:
        assert any(x["compute_only_suffix"] for x in interleaved.realization["frontier_issue"]["regions"])
    interleaved.materialize_scf().verify()
    from types import SimpleNamespace

    from merlin.compile.scheduling.dense_panel_pipeline import _fission_noninterfering_epoch_loads

    checking, changes = _fission_noninterfering_epoch_loads(interleaved.kernel, ADAPTER)
    assert changes
    assert interleaved.realization["source_phase_fission"]["checking_kernel_sha256"] == checking.digest()
    normalized = epochs(SimpleNamespace(kernel=checking))
    assert len(normalized) == len(after)
    for original_epoch, checking_epoch in zip(after, normalized, strict=True):
        for load_phase in (False, True):
            def phase(c):
                return (c[0] in ("lhs_load", "weight_load")) == load_phase
            assert list(filter(phase, original_epoch)) == list(filter(phase, checking_epoch))


@pytest.mark.parametrize("issue_order", ["prefetch_rhs_between_columns", "prefetch_rhs_between_k_tiles"])
def test_frontier_issue_scalar_outputs_still_match_original_source(monkeypatch, issue_order):
    import test_compact_dense as existing

    original_fixture = existing.fixture

    def capacity_fixture(**kwargs):
        # Keep all three scalar tails while isolating this issue transform
        # from the separate multi-output-block verifier scalability tests.
        kwargs["shape"] = (3, 14, 7)
        p, r, s = original_fixture(**kwargs)
        s = replace(s, lhs=replace(s.lhs, capacity=256), rhs=replace(s.rhs, capacity=48),
                    accumulator=replace(s.accumulator, capacity=64))
        r = replace(r, operand_bytes=256 * s.lhs.unit_bytes + 48 * s.rhs.unit_bytes,
                    accumulator_bytes=64 * s.accumulator.unit_bytes, max_m_blocks=2, max_n_blocks=2, max_k_blocks=2)
        return p, r, s

    def schedule(p, r, s, **kwargs):
        b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, 2, 2))
        *_, admission = inputs()
        result = chosen(p, r, s, b, admission, issue_order=issue_order)
        return replace(result, kernel=_serial_projection(result.kernel), realization=result.realization["source"])

    monkeypatch.setattr(existing, "fixture", capacity_fixture)
    monkeypatch.setattr(existing, "schedule", schedule)
    existing.test_tails_reuse_and_independent_execution(("m", "n"), False)


def test_frontier_issue_refuses_missing_epoch_and_bad_next_segment_and_keeps_actual_alias_check():
    from merlin.compile.scheduling.dense_segmented_pipeline import interleave_dense_segmented_frontier

    p, r, s, b, a = inputs()
    with pytest.raises(ValueError, match="explicit prefetched epoch"):
        chosen(p, r, s, b, a, prefetch=False, issue_order="prefetch_rhs_between_columns")
    control = build_dense_segmented_pipeline(p, b, r, s, ADAPTER, name="serial", dtypes=("i8", "i8", "i32"),
                                              prefetch=False)
    with pytest.raises(ValueError, match="at least one actual mixed"):
        interleave_dense_segmented_frontier(control, ADAPTER)
    source = build_dense_segmented_pipeline(p, b, r, s, ADAPTER, name="wrong", dtypes=("i8", "i8", "i32"))

    def rewrite(body, field):
        nonlocal count
        output = []
        for statement in body:
            if type(statement) is Loop:
                statement = replace(statement, body=rewrite(statement.body, field))
            elif statement.instr == "weight_load":
                count += 1
                if count == 2:
                    delta = -b.k if field == "a" else -b.k * b.n * p.tile_k
                    statement = replace(statement, args=tuple((key, value + delta if key == field else value)
                                                              for key, value in statement.args))
            output.append(statement)
        return tuple(output)
    count = 0
    wrong = replace(source, body=rewrite(source.body, "a"))
    with pytest.raises(ValueError, match="exact following K segment"):
        interleave_dense_segmented_frontier(wrong, ADAPTER)
    count = 0
    alias = replace(source, body=rewrite(source.body, "dst"))
    proposed, _ = interleave_dense_segmented_frontier(alias, ADAPTER)
    with pytest.raises(ValueError, match="overlapping in-flight"):
        verify_dense_segmented_pipeline(proposed, p, b, r, s, ADAPTER, a, static_budget=128)


def test_phase_fission_never_bypasses_actual_hazards_or_source_dependencies(monkeypatch):
    from test_compact_dense import mutate

    import merlin.compile.scheduling.dense_panel_pipeline as epochs
    from merlin.compile.scheduling.dense_segmented_pipeline import interleave_dense_segmented_frontier

    p, r, s, b, a = inputs((3, 14, 7), 2)
    source = build_dense_segmented_pipeline(p, b, r, s, ADAPTER, name="bad", dtypes=("i8", "i8", "i32"))
    interleaved, _ = interleave_dense_segmented_frontier(source, ADAPTER, issue_order="prefetch_rhs_between_k_tiles")
    corrupt = mutate(interleaved, "lhs_load", "src_col", 1)
    with pytest.raises(ValueError):
        verify_dense_segmented_pipeline(corrupt, p, b, r, s, ADAPTER, a, static_budget=128)
    actual = epochs._fission_noninterfering_epoch_loads
    checking, _ = actual(interleaved, ADAPTER)
    # An already separated representation is idempotent and contributes no
    # new phase movement; original command fields and complete domains stay.
    twice, changes = actual(checking, ADAPTER)
    assert twice == checking and not changes
    seen = 0

    def collide(body):
        nonlocal seen
        result = []
        for statement in body:
            if type(statement) is Loop:
                statement = replace(statement, body=collide(statement.body))
            elif statement.instr == "weight_load":
                seen += 1
                if seen == 2:
                    statement = replace(statement, args=tuple((k, v + (-b.k * b.n * p.tile_k) if k == "dst" else v)
                                                              for k, v in statement.args))
            result.append(statement)
        return tuple(result)
    alias = replace(interleaved, body=collide(interleaved.body))

    def forbidden(*args):
        raise AssertionError("unsafe epoch reached source phase fission")

    monkeypatch.setattr(epochs, "_fission_noninterfering_epoch_loads", forbidden)
    with pytest.raises(ValueError, match="overlapping in-flight"):
        verify_dense_segmented_pipeline(alias, p, b, r, s, ADAPTER, a, static_budget=128)
