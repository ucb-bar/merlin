"""Executed loop decisions, exact ordered tails and final dependence barriers."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture, schedule

from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import Kernel, Ptr, TensorArg, call, evaluate, instances, loop, select
from merlin.sched.ir.expr import Expr, Var
from merlin.sched.primitives import NotApplicable

SOURCE = "b" * 64


def trace(kernel):
    def concrete(value, env):
        if isinstance(value, Ptr):
            return replace(value, offset=concrete(value.offset, env))
        return evaluate(value, env) if isinstance(value, Expr) else value

    return [
        replace(
            op,
            **{
                field: tuple((key, concrete(value, env)) for key, value in getattr(op, field))
                for field in ("args", "sets", "assumes")
            },
        )
        for op, env in instances(kernel)
    ]


def base(extent=17):
    return Kernel(
        "ordered",
        (TensorArg("x", (extent,), "i8", "read"),),
        (
            loop(
                "i",
                extent,
                call(
                    "read",
                    src=Ptr("x", Var("i")),
                    n=select("i", extent - 1, 1, 2),
                    unit="selected-read-unit",
                    sets={"last": select("i", 0, 5, 6)},
                ),
            ),
        ),
    )


@pytest.mark.parametrize("action", ["split", "unroll"])
def test_exact_ordered_tail_predicates_pointer_configuration_and_unit_survive(action):
    original = base()
    actual = propose_loop_transforms(original, (LoopTransform(action, ("i",), factor=4),))
    assert actual.digest() != original.digest()
    assert trace(actual) == trace(original)
    assert "for io in 0..4" in actual.text()
    assert "for i in 0..1" in actual.text()
    assert ("for ii" in actual.text()) == (action == "split")


def test_split_then_unroll_composes_against_current_cursor_and_not_dynamic_domain(monkeypatch):
    from merlin.sched import primitives

    original = base(10**12 + 3)
    monkeypatch.setattr(primitives, "_instance_keys", lambda *_: pytest.fail("dynamic enumeration forbidden"))
    actual = propose_loop_transforms(
        original,
        (
            LoopTransform("split", ("i",), factor=4),
            LoopTransform("unroll", ("io", "ii"), factor=4),
        ),
    )
    assert "for io in 0..250000000000" in actual.text()
    assert "for i in 0..3" in actual.text()
    assert actual.text().count("read(") == 5


def test_fission_and_fusion_are_actual_proposals_with_preserved_per_index_values():
    original = Kernel(
        "pointwise",
        (),
        (
            loop(
                "i",
                5,
                call("a", offset=Var("i"), value=select("i", 4, 19, 7)),
                call("b", offset=Var("i"), value=select("i", 0, 11, 3)),
            ),
        ),
    )
    divided = propose_loop_transforms(original, (LoopTransform("fission", ("i",), cut=1),))
    fused = propose_loop_transforms(divided, (LoopTransform("fuse", ("i",)),))
    assert trace(divided) != trace(original)
    assert trace(fused) == trace(original)

    def output(program):
        result = {"a": [None] * 5, "b": [None] * 5}
        for op in trace(program):
            args = dict(op.args)
            result[op.instr][args["offset"]] = args["value"]
        return result

    assert output(original) == output(divided) == output(fused)


def proposal(p, r, s, transforms):
    return parse_dense_schedule_plan(
        dense_schedule_text(
            make_dense_schedule_plan(
                p,
                r,
                s,
                ADAPTER,
                source_identity=SOURCE,
                dtypes=("i8", "i8", "i32"),
                static_budget=80,
                block_tiles=(1, 1, 1),
                loop_transforms=transforms,
            )
        )
    )


@pytest.mark.parametrize(
    "transforms",
    [
        (LoopTransform("split", ("outer_m", "outer_n", "panel1"), factor=2),),
        (LoopTransform("unroll", ("outer_m", "outer_n", "panel1"), factor=2),),
        (LoopTransform("reorder", ("outer_m",)),),
    ],
)
def test_typed_decisions_feed_actual_dense_emission_and_independent_source_verifier(transforms):
    p, r, s = fixture()
    ordinary = schedule(p, r, s, schedule_plan=proposal(p, r, s, ()), source_identity=SOURCE)
    result = schedule(p, r, s, schedule_plan=proposal(p, r, s, transforms), source_identity=SOURCE)
    assert result.kernel.digest() != ordinary.kernel.digest()
    assert result.realization["schedule_ir"]["loop_transforms"]
    assert "schedule.loop_transform" in result.realization["schedule_ir"]["text"]
    assert result.realization["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert result.realization["executed_instruction_calls"] == ordinary.realization["executed_instruction_calls"]
    result.revalidate()


def test_semantically_unsafe_fission_is_refused_by_unchanged_dense_dependence_checker():
    p, r, s = fixture()
    # Loading every K panel before executing them destroys reusable operand rows.
    transforms = (LoopTransform("fission", ("outer_m", "outer_n", "panel1"), cut=2),)
    plan = proposal(p, r, s, transforms)
    with pytest.raises(ValueError):
        schedule(p, r, s, schedule_plan=plan, source_identity=SOURCE)


def test_hoist_moves_actual_invariant_statement_and_folds_singleton_index():
    original = Kernel(
        "reuse",
        (),
        (loop("i", 4, loop("once", 1, call("load", value=select("once", 0, 7, 9))), call("consume", index=Var("i"))),),
    )
    actual = propose_loop_transforms(
        original,
        (
            LoopTransform("hoist", ("i", "once"), index=0),
            LoopTransform("hoist", ("i",), index=0),
        ),
    )
    assert [c.instr for c in trace(actual)] == ["load", *["consume"] * 4]
    assert dict(trace(actual)[0].args)["value"] == 7
    assert [dict(c.args)["index"] for c in trace(actual)[1:]] == list(range(4))
    with pytest.raises(NotApplicable, match="crossed loop index"):
        propose_loop_transforms(original, (LoopTransform("hoist", ("i",), index=1),))
    with pytest.raises(NotApplicable, match="completion|staging|configuration"):
        propose_loop_transforms(base(), (LoopTransform("hoist", ("i",), index=0),))


def test_typed_hoist_reuses_invariant_full_k_operand_and_final_source_checks_it():
    from merlin.compile.scheduling.compact_dense import emit_compact_dense
    from merlin.compile.scheduling.compact_verification import verify_compact_dense
    from merlin.compile.scheduling.schedule_ir import read_dense_schedule_plan
    from merlin.transforms.dense_planning import dense_blocks

    p, r, s = fixture(shape=(5, 14, 25))
    s = replace(s, lhs=replace(s.lhs, capacity=128), rhs=replace(s.rhs, capacity=128))
    r = replace(
        r,
        operand_bytes=128 * (s.lhs.unit_bytes + s.rhs.unit_bytes),
        max_m_blocks=2,
        max_n_blocks=2,
        max_k_blocks=p.tiles[1],
    )
    b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, p.tiles[1], 2))
    transforms = (
        LoopTransform("hoist", ("outer_m", "outer_n", "panel1"), index=0),
        LoopTransform("hoist", ("outer_m", "outer_n"), index=0),
    )
    plan = parse_dense_schedule_plan(
        dense_schedule_text(
            make_dense_schedule_plan(
                p,
                r,
                s,
                ADAPTER,
                source_identity=SOURCE,
                dtypes=("i8", "i8", "i32"),
                static_budget=80,
                block_tiles=(b.m, b.k, b.n),
                loop_transforms=transforms,
            )
        )
    )
    chosen = read_dense_schedule_plan(
        plan, p, r, s, ADAPTER, source_identity=SOURCE, dtypes=("i8", "i8", "i32"), static_budget=80
    )
    assert chosen.loop_transforms == transforms
    raw = emit_compact_dense(p, b, s, ADAPTER, name="reuse", dtypes=("i8", "i8", "i32"))
    changed = propose_loop_transforms(raw, chosen.loop_transforms)
    before = verify_compact_dense(raw, p, r, s, ADAPTER, static_budget=80)
    after = verify_compact_dense(changed, p, r, s, ADAPTER, static_budget=80)
    assert after["lhs_loads"] < before["lhs_loads"]
    assert after["output_tiles"] == before["output_tiles"]
    assert after["rhs_loads"] == before["rhs_loads"]
    # An invariant compute may still change the source domain: proposal does
    # not grant dependence or product-cover legality.
    bad = propose_loop_transforms(raw, (LoopTransform("hoist", ("outer_m", "outer_n", "panel1"), index=2),))
    # Folding the singleton alone is legal; removing reduction iterations is not.
    with pytest.raises(NotApplicable, match="crossed loop index"):
        propose_loop_transforms(bad, (LoopTransform("hoist", ("outer_m", "outer_n", "tile6"), index=0),))


def test_requested_unsupported_combination_never_substitutes_another_transform():
    with pytest.raises(ValueError, match="ignored"):
        LoopTransform("reorder", ("i",), factor=2)
    with pytest.raises(NotApplicable, match="budget"):
        propose_loop_transforms(base(), (LoopTransform("unroll", ("i",), factor=4),), max_static_operations=4)
    with pytest.raises(NotApplicable, match="perfect"):
        propose_loop_transforms(base(), (LoopTransform("reorder", ("i",)),))
    token = Kernel("async", (), (loop("i", 8, call("launch", produces="done"), call("wait", awaits="done")),))
    with pytest.raises(NotApplicable, match="completion|ill-formed"):
        propose_loop_transforms(token, (LoopTransform("unroll", ("i",), factor=2),))


@pytest.mark.parametrize("action", ["split", "unroll"])
def test_transformed_dense_tail_and_reduction_execute_original_arithmetic(monkeypatch, action):
    import test_compact_dense as existing

    ordinary = existing.schedule

    def chosen(p, r, s, **kwargs):
        order = kwargs.pop("outer_order")
        plan = make_dense_schedule_plan(
            p,
            r,
            s,
            ADAPTER,
            source_identity=SOURCE,
            dtypes=("i8", "i8", "i32"),
            static_budget=80,
            block_tiles=(2, 1, 2),
            outer_order=order,
            loop_transforms=(LoopTransform(action, ("outer_n", "outer_m", "panel1"), factor=2),),
        )
        return ordinary(p, r, s, schedule_plan=plan, source_identity=SOURCE, **kwargs)

    monkeypatch.setattr(existing, "schedule", chosen)
    existing.test_tails_reuse_and_independent_execution(("n", "m"), True)


def test_named_stripes_compose_across_repeated_scope_names_and_keep_ordered_tail():
    from merlin.compile.scheduling.loop_schedule import striped_fission_transforms

    original = Kernel(
        "scoped",
        (),
        tuple(
            loop(scope, 1, loop("i", 9, call("a", value=Var("i")), call("b", value=select("i", 8, 99, 7))))
            for scope in ("left", "right")
        ),
    )
    transforms = striped_fission_transforms(original, (("left", "i"), ("right", "i")), factor=4, cut=1)
    assert len(transforms) == 4 and all(request.names for request in transforms)
    names = [name for request in transforms for name in request.names]
    assert len(set(names)) == len(names)
    actual = propose_loop_transforms(original, transforms)
    expected = []
    for _scope in range(2):
        for start in (0, 4):
            expected.extend(("a", i) for i in range(start, start + 4))
            expected.extend(("b", 7) for i in range(start, start + 4))
        expected.extend((("a", 8), ("b", 99)))
    assert [(call.instr, dict(call.args)["value"]) for call in trace(actual)] == expected


def test_explicit_names_roundtrip_and_collisions_never_silently_rename():
    transforms = (
        LoopTransform("split", ("outer_m", "outer_n", "panel1"), factor=2, names=("reduction_stripe", "within_stripe")),
    )
    p, r, s = fixture()
    result = schedule(p, r, s, schedule_plan=proposal(p, r, s, transforms), source_identity=SOURCE)
    assert "reduction_stripe" in result.kernel.text()
    assert 'names = ["reduction_stripe", "within_stripe"]' in result.realization["schedule_ir"]["text"]
    with pytest.raises(ValueError, match="result names"):
        LoopTransform("split", ("i",), factor=4, names=("same", "same"))
    with pytest.raises(Exception, match="already a loop variable|collides"):
        propose_loop_transforms(base(), (LoopTransform("split", ("i",), factor=4, names=("i", "inner")),))
