"""Schedule IR choices must alter real operands and retain independent checks."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture, schedule
from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IntegerAttr, StringAttr, i64

from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import evaluate, instances
from merlin.xdsl_dialects.schedule import BindOp, HandleType

SOURCE = "a" * 64


def proposal(p, r, s, **kwargs):
    return make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity=SOURCE,
        dtypes=("i8", "i8", "i32"),
        static_budget=80,
        block_tiles=(1, 1, 1),
        **kwargs,
    )


def decision(plan):
    return list(plan.body.block.first_op.body.block.ops)[1]


def calls(result):
    return [(call.instr, tuple((k, evaluate(v, env)) for k, v in call.args)) for call, env in instances(result.kernel)]


def test_text_roundtrip_and_edited_knobs_drive_actual_checked_emission():
    p, r, s = fixture()
    plan = parse_dense_schedule_plan(dense_schedule_text(proposal(p, r, s)))
    baseline = schedule(p, r, s, schedule_plan=plan, source_identity=SOURCE)
    op = decision(plan)
    # A compatible edit requires no opaque backend schedule reconstruction.
    op.attributes["outer_order"] = ArrayAttr([StringAttr("n"), StringAttr("m")])
    op.attributes["row_bases"] = DenseArrayBase.from_list(i64, [3, 4, 2])
    edited = schedule(p, r, s, schedule_plan=plan, source_identity=SOURCE)
    assert calls(baseline) != calls(edited)
    assert baseline.kernel.digest() != edited.kernel.digest()
    assert edited.choice.cost_unit == "explicit_schedule_ir"
    assert edited.realization["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert edited.realization["machine_emission"] == "not_established"
    assert "schedule.block_contraction" in edited.realization["schedule_ir"]["text"]
    assert edited.realization["schedule_ir"]["source_identity"] == SOURCE
    first_load = next(args for name, args in calls(edited) if name == "lhs_load")
    assert dict(first_load)["dst"] == 3
    edited.revalidate()
    # The result owns checked emitted IR; later proposal edits are not consumed.
    op.attributes["row_bases"] = DenseArrayBase.from_list(i64, [500, 500, 500])
    edited.revalidate()
    with pytest.raises(ValueError, match="escapes"):
        schedule(p, r, s, schedule_plan=plan, source_identity=SOURCE)


def test_selected_block_is_consumed_instead_of_researched():
    p, r, s = fixture()
    small = schedule(p, r, s, schedule_plan=proposal(p, r, s), source_identity=SOURCE)
    large_plan = make_dense_schedule_plan(
        p, r, s, ADAPTER, source_identity=SOURCE, dtypes=("i8", "i8", "i32"), static_budget=80, block_tiles=(2, 2, 2)
    )
    large = schedule(p, r, s, schedule_plan=large_plan, source_identity=SOURCE)
    assert (small.choice.block.m, small.choice.block.k, small.choice.block.n) == (1, 1, 1)
    assert (large.choice.block.m, large.choice.block.k, large.choice.block.n) == (2, 2, 2)
    assert large.realization["lhs_loads"] < small.realization["lhs_loads"]
    assert large.realization["rhs_loads"] < small.realization["rhs_loads"]
    assert small.kernel.digest() != large.kernel.digest()


@pytest.mark.parametrize("change", ["source", "effects", "capacity", "storage", "bias"])
def test_original_binding_cannot_be_overridden(change):
    p, r, s = fixture()
    plan = proposal(p, r, s)
    source = SOURCE
    adapter = ADAPTER
    if change == "source":
        source = "b" * 64
    elif change == "effects":
        adapter = replace(ADAPTER, numerical_policy_identity="different-rounding")
    elif change == "capacity":
        r = replace(r, operand_bytes=r.operand_bytes + 1)
    elif change == "storage":
        s = replace(s, lhs=replace(s.lhs, backing="different-allocation"))
    else:
        p = replace(p, bias=True)
    from merlin.compile.scheduling.compact_dense import schedule_compact_dense

    with pytest.raises(ValueError, match="binding"):
        schedule_compact_dense(
            p,
            r,
            s,
            adapter,
            name="dense",
            dtypes=("i8", "i8", "i32"),
            static_budget=80,
            external_buffers_disjoint=True,
            schedule_plan=plan,
            source_identity=source,
        )


def test_forged_counts_and_unsupported_decisions_are_refused():
    p, r, s = fixture()
    plan = proposal(p, r, s)
    decision(plan).attributes["static_commands"] = IntegerAttr(0, i64)
    with pytest.raises(ValueError, match="command count"):
        schedule(p, r, s, schedule_plan=plan, source_identity=SOURCE)
    plan = proposal(p, r, s)
    plan.body.block.first_op.body.block.add_op(
        BindOp.create(properties={"target": StringAttr("unsupported-extra-decision")}, result_types=[HandleType()])
    )
    with pytest.raises(ValueError, match="bind then"):
        schedule(p, r, s, schedule_plan=plan, source_identity=SOURCE)


def test_source_entry_supplies_original_identity_and_preserves_source_custody(tmp_path):
    from test_dense_source import compile_source, source
    from test_selected_dense_contracts import inputs

    from merlin.transforms.certificates import ir_digest

    model = inputs(tmp_path)
    selected, p, s, adapter = model
    module, op = source((p.m, p.k, p.n), selected.dense.dtypes)
    r, bound = selected.dense.bind(
        p,
        s,
        adapter,
        dtypes=selected.dense.dtypes,
        hardware=selected.hardware_identity,
        policy=selected.numerical_policy,
    )
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        bound,
        source_identity=ir_digest(module),
        dtypes=selected.dense.dtypes,
        static_budget=80,
        block_tiles=(1, 1, 1),
    )
    result = compile_source(model, module, op, schedule_plan=plan)
    assert result.compiled.realization["schedule_ir"]["source_identity"] == ir_digest(module)
    result.revalidate()
    module.attributes["changed"] = StringAttr("different-original-source")
    with pytest.raises(ValueError, match="source"):
        result.revalidate()
    with pytest.raises(ValueError, match="binding"):
        compile_source(model, module, op, schedule_plan=plan)


def test_prospective_export_is_separate_and_has_closed_dialect_dependencies(tmp_path):
    import os
    import subprocess
    import sys

    from merlin.compile.scheduling.export import export_shared_scheduling
    from merlin.compile.scheduling.library_sources import library_modules

    assert "merlin.compile.scheduling.schedule_ir" not in library_modules(11)
    assert library_modules(12)[: len(library_modules(11))] == library_modules(11)
    exported = tmp_path / "export"
    manifest = export_shared_scheduling(exported, version=12)
    assert manifest["inventory_version"] == 12
    assert "merlin.compile.scheduling.schedule_ir" in manifest["public_modules"]
    env = dict(os.environ, PYTHONPATH=str(exported))
    subprocess.run(
        [sys.executable, "-c", "from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("loop_order", [("m", "n"), ("n", "m")])
def test_edited_plan_executes_original_arithmetic_with_tails(monkeypatch, loop_order):
    import test_compact_dense as existing

    ordinary_schedule = existing.schedule

    def use_plan(p, r, s, **kwargs):
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
            row_bases=(s.lhs.base + 1, s.rhs.base + 1, s.accumulator.base + 1),
        )
        return ordinary_schedule(p, r, s, schedule_plan=plan, source_identity=SOURCE, **kwargs)

    monkeypatch.setattr(existing, "schedule", use_plan)
    # Existing independent instruction interpreter compares every output with
    # numpy matmul and checks actual transfer bytes/counts, including tails.
    existing.test_tails_reuse_and_independent_execution(loop_order, True)


def test_plan_is_no_authority_for_corrupted_actual_effects():
    from merlin.compile.scheduling.compact_dense import schedule_compact_dense
    from merlin.sched.ir import Const

    p, r, s = fixture()
    plan = proposal(p, r, s)

    def corrupt(operation):
        call = ADAPTER.encode(operation)
        if call.instr == "lhs_load":
            return replace(call, args=tuple((k, Const(500) if k == "dst" else v) for k, v in call.args))
        return call

    adapter = replace(ADAPTER, encode=corrupt)
    with pytest.raises(ValueError):
        schedule_compact_dense(
            p,
            r,
            s,
            adapter,
            name="dense",
            dtypes=("i8", "i8", "i32"),
            static_budget=80,
            external_buffers_disjoint=True,
            schedule_plan=plan,
            source_identity=SOURCE,
        )


def test_two_orders_on_same_original_source_change_commands_and_preserve_mac_cover(tmp_path):
    from test_dense_source import compile_source, source
    from test_selected_dense_contracts import inputs

    from merlin.transforms.certificates import ir_digest

    model = inputs(tmp_path)
    selected, p, s, adapter = model
    module, op = source((p.m, p.k, p.n), selected.dense.dtypes)
    r, bound = selected.dense.bind(
        p,
        s,
        adapter,
        dtypes=selected.dense.dtypes,
        hardware=selected.hardware_identity,
        policy=selected.numerical_policy,
    )
    results = []
    for order in (("m", "n"), ("n", "m")):
        plan = make_dense_schedule_plan(
            p,
            r,
            s,
            bound,
            source_identity=ir_digest(module),
            dtypes=selected.dense.dtypes,
            static_budget=80,
            block_tiles=(1, 1, 1),
            outer_order=order,
        )
        results.append(compile_source(model, module, op, schedule_plan=plan).compiled)
    assert results[0].kernel.digest() != results[1].kernel.digest()
    assert calls(results[0]) != calls(results[1])
    for result in results:
        # Identical complete source work, no reassociation of the K reduction.
        preload = None
        macs = 0
        for name, arguments in calls(result):
            args = dict(arguments)
            if name == "preload":
                preload = args
            elif name == "compute":
                macs += args["rows"] * args["depth"] * preload["cols"]
        assert macs == p.m * p.k * p.n
        assert result.realization["output_tiles"] == p.tiles[0] * p.tiles[2]
        result.revalidate()
