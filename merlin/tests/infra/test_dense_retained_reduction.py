"""Full-K immutable operands, ordered accumulator commits and explicit drains."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture, mutate

from merlin.compile.scheduling.compact_dense import emit_compact_dense
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    DenseRetainedReductionAdmission,
    _serial_projection,
    complete_dense_retained_reduction,
    dense_retained_operand_hoists,
    schedule_dense_panel_pipeline,
    verify_dense_retained_reduction,
)
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.transforms.dense_planning import dense_blocks


def inputs(shape=(9, 14, 25), tile=(2, 3, 4)):
    p, r, s = fixture(shape=shape, tile=tile)
    s = replace(s, lhs=replace(s.lhs, capacity=128), rhs=replace(s.rhs, capacity=128))
    r = replace(
        r,
        operand_bytes=128 * (s.lhs.unit_bytes + s.rhs.unit_bytes),
        max_m_blocks=2,
        max_n_blocks=2,
        max_k_blocks=p.tiles[1],
    )
    b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, p.tiles[1], 2))
    a = DenseRetainedReductionAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "independent-neutral-ordered-commit-interpreter-v1",
        b.m * b.k * b.n,
        (b.m + b.n) * b.k,
        2,
        2,
    )
    return p, r, s, b, a


def chosen(p, r, s, b, a, *, role="lhs", order=("m", "n"), hoist=True, compute_order=None, compute_unroll=None):
    raw = emit_compact_dense(p, b, s, ADAPTER, name="retained", dtypes=("i8", "i8", "i32"), outer_order=order)
    transforms = dense_retained_operand_hoists(raw, ADAPTER, role=role) if hoist else ()
    if compute_order is not None:
        from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_compute_order
        from merlin.compile.scheduling.loop_schedule import propose_loop_transforms

        current = propose_loop_transforms(raw, transforms)
        transforms += dense_retained_compute_order(current, ADAPTER, order=compute_order)
    if compute_unroll is not None:
        from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_compute_unroll
        from merlin.compile.scheduling.loop_schedule import propose_loop_transforms

        current = propose_loop_transforms(raw, transforms)
        axis, factor = compute_unroll
        transforms += dense_retained_compute_unroll(current, ADAPTER, axis=axis, factor=factor)
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="d" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(b.m, b.k, b.n),
        outer_order=order,
        loop_transforms=transforms,
        pipeline_kind="retained_full_k",
        pipeline_prefetch=False,
    )
    return schedule_dense_panel_pipeline(
        p,
        r,
        s,
        ADAPTER,
        a,
        name="retained",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=parse_dense_schedule_plan(dense_schedule_text(plan)),
        source_identity="d" * 64,
        external_buffers_disjoint=True,
        static_budget=128,
    )


@pytest.mark.parametrize("order,role", [(("m", "n"), "lhs"), (("n", "m"), "weight")])
def test_typed_full_k_and_invariant_hoist_preserve_tails_source_and_actual_emission(order, role):
    p, r, s, b, a = inputs()
    original = chosen(p, r, s, b, a, order=order, role=role, hoist=False)
    result = chosen(p, r, s, b, a, order=order, role=role)
    assert result.kernel.digest() != original.kernel.digest()
    key = "lhs_loads" if role == "lhs" else "rhs_loads"
    assert result.realization["source"][key] < original.realization["source"][key]
    assert result.realization["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert all(not (e["compute_pairs"] and e["future_or_initial_loads"]) for e in result.realization["epochs"])
    assert max(e["compute_pairs"] for e in result.realization["epochs"]) == b.m * b.k * b.n
    assert result.realization["schedule_ir"]["pipeline_kind"] == "retained_full_k"
    assert "schedule.loop_transform" in result.realization["schedule_ir"]["text"]
    result.materialize_scf().verify()
    result.revalidate()


def test_actual_reduction_executes_exact_source_with_ordered_tails(monkeypatch):
    # Reuse an independent scalar executor over the actual projected operands;
    # it checks every multiply, all stores and ordered reduction arithmetic.
    import test_compact_dense as existing

    previous_fixture = existing.fixture

    def retained_fixture(**kwargs):
        p, r, s = previous_fixture(**kwargs)
        s = replace(s, lhs=replace(s.lhs, capacity=128), rhs=replace(s.rhs, capacity=128))
        r = replace(r, operand_bytes=128 * (s.lhs.unit_bytes + s.rhs.unit_bytes), max_k_blocks=p.tiles[1])
        return p, r, s

    def retained_schedule(p, r, s, **kwargs):
        b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, p.tiles[1], 2))
        a = DenseRetainedReductionAdmission(
            ADAPTER.configuration_identity,
            ADAPTER.numerical_policy_identity,
            "independent-neutral-ordered-commit-interpreter-v1",
            b.m * b.k * b.n,
            (b.m + b.n) * b.k,
            2,
            2,
        )
        result = chosen(p, r, s, b, a, role="weight", order=kwargs["outer_order"])
        return replace(result, kernel=_serial_projection(result.kernel), realization=result.realization["source"])

    monkeypatch.setattr(existing, "fixture", retained_fixture)
    monkeypatch.setattr(existing, "schedule", retained_schedule)
    existing.test_tails_reuse_and_independent_execution(("n", "m"), True)


@pytest.mark.parametrize(
    "instruction,operand,value,omit",
    [
        ("lhs_load", None, None, True),
        ("weight_load", "dst", 3, False),
        ("compute", "k", 1, False),
        ("preload", "continue", 0, False),
        ("store", None, None, True),
        (DRAIN, None, None, True),
    ],
)
def test_unknown_overwrite_reduction_observer_and_completion_refuse(instruction, operand, value, omit):
    p, r, s, b, a = inputs()
    good = chosen(p, r, s, b, a)
    bad = mutate(good.kernel, instruction, operand, value, omit=omit)
    with pytest.raises(ValueError):
        verify_dense_retained_reduction(bad, p, b, r, s, ADAPTER, a, static_budget=128)


def test_two_window_grant_cannot_authorize_ordered_accumulator_commits():
    from merlin.compile.scheduling.dense_panel_pipeline import DensePanelOverlapAdmission

    p, r, s, b, a = inputs()
    result = chosen(p, r, s, b, a)
    old = DensePanelOverlapAdmission(
        a.hardware_identity,
        a.numerical_policy_identity,
        a.qualification_identity,
        a.max_issued_compute_pairs,
        a.max_issued_loads,
        2,
        2,
    )
    with pytest.raises(ValueError, match="ordered-commit"):
        verify_dense_retained_reduction(result.kernel, p, b, r, s, ADAPTER, old, static_budget=128)
    with pytest.raises(ValueError, match="full K"):
        complete_dense_retained_reduction(result.kernel, p, replace(b, k=1), r, s, ADAPTER)
    with pytest.raises(ValueError, match="prefetch"):
        make_dense_schedule_plan(
            p,
            r,
            s,
            ADAPTER,
            source_identity="d" * 64,
            dtypes=("i8", "i8", "i32"),
            static_budget=128,
            block_tiles=(b.m, b.k, b.n),
            pipeline_kind="retained_full_k",
            pipeline_prefetch=True,
        )


def test_overlapping_stage_writes_refuse_even_when_serial_source_is_valid():
    from merlin.compile.scheduling.compact_verification import verify_compact_dense
    from merlin.sched.ir import Call, Loop

    p, r, s, b, a = inputs()
    good = chosen(p, r, s, b, a)
    changed = False

    def duplicate(body):
        nonlocal changed
        result = []
        for statement in body:
            if type(statement) is Loop:
                statement = replace(statement, body=duplicate(statement.body))
            result.append(statement)
            if type(statement) is Call and statement.instr == "lhs_load" and not changed:
                result.append(statement)
                changed = True
        return tuple(result)

    bad = replace(good.kernel, body=duplicate(good.kernel.body))
    verify_compact_dense(_serial_projection(bad), p, r, s, ADAPTER, static_budget=128)
    with pytest.raises(ValueError, match="overlapping in-flight"):
        verify_dense_retained_reduction(bad, p, b, r, s, ADAPTER, a, static_budget=128)


@pytest.mark.parametrize("batch", [False, True])
def test_final_wide_lhs_k_bands_and_rhs_n_bands_use_retained_hazard_checker(batch):
    from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
    from merlin.compile.scheduling.tile_load_coalescing import (
        TileLoadAddressLaw,
        TileLoadMapping,
        _linear,
        propose_tile_load_coalescing,
    )
    from merlin.sched.ir import Const

    p, r, s, b, a = inputs(shape=(8, 8, 8), tile=(2, 2, 2))
    original = chosen(p, r, s, b, a)
    if batch:
        from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts

        original = batch_dense_panel_readouts(
            original,
            p,
            b,
            r,
            s,
            ADAPTER,
            replace(
                a,
                max_issued_readouts=b.m * b.n,
                pending_readouts=1,
                readout_configuration="ordered_snapshot_before_store_issue",
            ),
        )
    laws = {
        role: TileLoadAddressLaw(2, 2, 8, 1, pitch, 2, "independent-test-physical-addresses")
        for role, pitch in (("lhs", p.k), ("weight", p.n))
    }

    def project(call):
        if call.instr not in ("lhs_load", "weight_load"):
            return None
        op = ADAPTER.project(call)
        view = s.lhs if op.role == "lhs" else s.rhs
        return TileLoadMapping(
            op.role,
            view.backing,
            op.dram_row * laws[op.role].source_pitch_bytes + op.dram_col,
            op.row,
            op.rows,
            op.cols,
            laws[op.role],
        )

    def encode(call, columns):
        return replace(call, args=tuple((k, Const(columns) if k == "cols" else v) for k, v in call.args))

    def band(call, mapping):
        origin, terms = _linear(mapping.source_byte)
        base, base_terms = _linear(project(call).source_byte)
        assert terms == base_terms
        columns = origin - base
        return replace(
            call,
            args=tuple(
                (
                    k,
                    mapping.local_row
                    if k == "dst"
                    else mapping.columns
                    if k == "cols"
                    else v + columns
                    if k == "src_col"
                    else v + columns // mapping.law.tile_columns
                    if k == "b"
                    else v,
                )
                for k, v in call.args
            ),
        )

    proposal, _ = propose_tile_load_coalescing(original.kernel, project=project, encode=encode, max_tiles=4)
    result = admit_panel_tile_loads(original, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
    projection = result.realization["tile_load_projection"]
    assert projection["physical_load_commands"] < projection["expanded_logical_tile_bands"]
    assert projection["expanded_logical_tile_bands"] == sum(
        original.realization["source"][key] for key in ("lhs_loads", "rhs_loads")
    )
    assert result.realization["schema"] == "merlin.dense_retained_reduction.v1"
    assert result.realization["dynamic_epoch_drains"] == original.realization["dynamic_epoch_drains"]
    result.materialize_scf().verify()
    result.revalidate()


def test_retained_readout_batches_keep_exact_calls_and_original_compute_law():
    from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts
    from merlin.sched.ir import evaluate, instances

    p, r, s, b, a = inputs()
    original = chosen(p, r, s, b, a)
    selected = replace(
        a,
        max_issued_readouts=b.m * b.n,
        pending_readouts=1,
        readout_configuration="ordered_snapshot_before_store_issue",
    )
    result = batch_dense_panel_readouts(original, p, b, r, s, ADAPTER, selected)

    def data(kernel):
        return [
            (c.instr, tuple((key, evaluate(value, env)) for key, value in c.args))
            for c, env in instances(kernel)
            if c.instr != DRAIN
        ]

    assert data(result.kernel) == data(original.kernel)
    assert result.realization["readout_batching"]["removed_dynamic_drains"] > 0
    assert result.realization["schedule_ir"] == original.realization["schedule_ir"]
    assert max(e["readout_commands"] for e in result.realization["epochs"]) == b.m * b.n
    with pytest.raises(ValueError, match="store completion budget"):
        verify_dense_retained_reduction(result.kernel, p, b, r, s, ADAPTER, a, static_budget=128)
    with pytest.raises(ValueError, match="execution premises"):
        batch_dense_panel_readouts(
            original, p, b, r, s, ADAPTER, replace(selected, max_issued_loads=a.max_issued_loads + 1)
        )
    with pytest.raises(ValueError, match="premises"):
        replace(a, max_issued_readouts=4)


@pytest.mark.parametrize("role", ["lhs", "weight"])
def test_residency_choice_transfer_estimates_equal_independently_checked_actual_tail_program(role):
    from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_choices

    p, r, s, _b, _a = inputs()
    choice = dense_retained_choices(p, r, s, roles=(role,))[0]
    b = choice.block
    a = DenseRetainedReductionAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "independent-neutral-ordered-commit-interpreter-v1",
        b.m * b.k * b.n,
        (b.m + b.n) * b.k,
        2,
        2,
    )
    result = chosen(p, r, s, b, a, role=choice.role, order=choice.outer_order)
    actual = result.realization["source"]
    assert (choice.lhs_loads, choice.rhs_loads, choice.requested_operand_bytes) == (
        actual["lhs_loads"],
        actual["rhs_loads"],
        actual["requested_load_bytes"],
    )


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((32, 2048, 2048), "lhs"),
        ((12544, 256, 64), "weight"),
        ((3136, 128, 512), "weight"),
    ],
)
def test_shape_and_original_separate_capacity_select_both_residency_families(shape, expected):
    from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
    from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_choices
    from merlin.transforms.dense_planning import DenseProblem, DenseResources

    p = DenseProblem(*shape, 16, 16, 16, 1, 1, 4)
    storage = DenseLoopStorage(
        DenseMemoryView("a", "inputs", 16, 0, 8192),
        DenseMemoryView("b", "inputs", 16, 8192, 8192),
        DenseMemoryView("c", "sums", 64, 0, 1024),
    )
    resources = DenseResources(
        262144,
        65536,
        min(p.tiles[0], 64),
        min(p.tiles[2], 64),
        p.tiles[1],
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    choices = dense_retained_choices(p, resources, storage)
    assert {choice.role for choice in choices} == {"lhs", "weight"}
    assert choices[0].role == expected
    assert all(storage.admits(p, choice.block, resources) and choice.block.k == p.tiles[1] for choice in choices)
    # An unrelated original capacity cannot be silently lent to the other role.
    tight = replace(storage, lhs=replace(storage.lhs, capacity=1))
    assert dense_retained_choices(p, resources, tight) == ()
    with pytest.raises(ValueError, match="distinct explicit"):
        dense_retained_choices(p, resources, storage, roles=("lhs", "lhs"))


@pytest.mark.parametrize("compute_order", [("k", "m", "n"), ("m", "n", "k"), ("n", "k", "m")])
def test_resident_compute_interchanges_change_real_commands_and_keep_exact_reductions(compute_order):
    from collections import Counter

    from merlin.sched.ir import evaluate, instances

    p, r, s, b, a = inputs()
    original = chosen(p, r, s, b, a)
    result = chosen(p, r, s, b, a, compute_order=compute_order)

    def data(kernel):
        return [
            (c.instr, tuple((key, evaluate(value, env)) for key, value in c.args))
            for c, env in instances(kernel)
            if c.instr != DRAIN
        ]

    assert Counter(data(result.kernel)) == Counter(data(original.kernel))
    diagnostics = {"kernel_sha256", "solver_queries", "reused_solver_queries", "source_proof_reused"}
    assert {k: v for k, v in result.realization["source"].items() if k not in diagnostics} == {
        k: v for k, v in original.realization["source"].items() if k not in diagnostics
    }
    assert result.realization["dynamic_epoch_drains"] == original.realization["dynamic_epoch_drains"]
    if compute_order != ("k", "m", "n"):
        assert data(result.kernel) != data(original.kernel)
        assert any(request["action"] == "reorder" for request in result.realization["schedule_ir"]["loop_transforms"])
    reductions, pending = {}, None
    for instruction, values in data(result.kernel):
        values = dict(values)
        if instruction == "preload":
            pending = values
        elif instruction == "compute":
            reductions.setdefault((values["m"], pending["n"]), []).append(values["k"])
            pending = None
    assert len(reductions) == p.tiles[0] * p.tiles[2]
    assert all(ks == list(range(p.tiles[1])) for ks in reductions.values())
    result.materialize_scf().verify()


def test_compute_interchange_uses_projected_axes_and_refuses_unknown_order():
    from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_compute_order

    p, _r, s, b, _a = inputs()
    raw = emit_compact_dense(p, b, s, ADAPTER, name="axes", dtypes=("i8", "i8", "i32"))
    with pytest.raises(ValueError, match="permutation"):
        dense_retained_compute_order(raw, ADAPTER, order=("m", "m", "k"))
    bad = mutate(raw, "compute", "k", 0)
    with pytest.raises(ValueError, match="source coordinates"):
        dense_retained_compute_order(bad, ADAPTER, order=("m", "n", "k"))


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("order", [("k", "m", "n"), ("m", "n", "k")])
def test_semantic_compute_unroll_preserves_exact_command_order_and_partial_reduction(factor, order):
    from merlin.sched.ir import evaluate, instances

    p, r, s, b, a = inputs(shape=(8, 14, 16))
    baseline = chosen(p, r, s, b, a, compute_order=order)
    actual = chosen(p, r, s, b, a, compute_order=order, compute_unroll=("k", factor))

    def commands(kernel):
        return [(c.instr, tuple((key, evaluate(value, env)) for key, value in c.args)) for c, env in instances(kernel)]

    assert actual.kernel.digest() != baseline.kernel.digest()
    assert commands(actual.kernel) == commands(baseline.kernel)
    assert actual.realization["dynamic_epoch_drains"] == baseline.realization["dynamic_epoch_drains"]
    assert any(t["action"] == "unroll" for t in actual.realization["schedule_ir"]["loop_transforms"])
    actual.materialize_scf().verify()


def test_semantic_compute_unroll_refuses_unknown_axes_excess_extent_and_static_growth():
    from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_compute_unroll
    from merlin.sched.primitives import NotApplicable

    p, _r, s, b, _a = inputs()
    raw = emit_compact_dense(p, b, s, ADAPTER, name="unroll", dtypes=("i8", "i8", "i32"))
    with pytest.raises(ValueError, match="source axis"):
        dense_retained_compute_unroll(raw, ADAPTER, axis="unknown", factor=2)
    with pytest.raises(ValueError, match="extent"):
        dense_retained_compute_unroll(raw, ADAPTER, axis="k", factor=64)
    with pytest.raises(NotApplicable, match="static operation budget"):
        dense_retained_compute_unroll(raw, ADAPTER, axis="k", factor=2, max_static_operations=1)
