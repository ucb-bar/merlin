"""Shared blocking decisions across distinct tile and storage representations."""

from dataclasses import replace

import pytest

from merlin.transforms.dense_planning import (
    DenseProblem,
    DenseResources,
    choose_dense_block,
    dense_blocks,
)


@pytest.mark.parametrize("tile", [(4, 8, 16), (16, 16, 16), (32, 4, 8)])
def test_reuse_and_exact_capacity_across_rectangular_tiles(tile):
    tm, tk, tn = tile
    problem = DenseProblem(tm * 4, tk * 5 - 1, tn * 4, tm, tk, tn, 1, 2, 4)
    resources = DenseResources(65536, tm * tn * 4 * 8)
    choice = choose_dense_block(problem, resources, static_budget=64, price=lambda b: 2 * b.m * b.n + b.m + b.n)
    assert choice is not None
    block = choice.block
    assert block.operand_bytes <= resources.operand_bytes
    assert block.accumulator_bytes <= resources.accumulator_bytes
    assert block.loads < 4 * 4 * 5 * 2
    assert block.requested_operand_bytes < problem.m * problem.k * 4 + problem.k * problem.n * 2 * 4


def test_capacity_and_emitter_support_are_independent_constraints():
    problem = DenseProblem(64, 69, 64, 16, 16, 16, 1, 1, 4)
    resources = DenseResources(256 * 6, 1024 * 8)
    candidates = dense_blocks(problem, resources)
    assert all(b.operand_bytes <= 1536 and b.accumulator_bytes <= 8192 for b in candidates)
    choice = choose_dense_block(problem, resources, static_budget=64, price=lambda b: 20, admits=lambda b: b.n == 1)
    assert choice is not None and choice.block.n == 1
    assert choose_dense_block(problem, resources, static_budget=64, price=lambda b: None) is None
    assert choose_dense_block(problem, resources, static_budget=0, price=lambda b: 1) is None


def test_tail_support_and_search_limits_are_explicit():
    problem = DenseProblem(17, 69, 35, 16, 16, 16, 1, 1, 4)
    resources = DenseResources(65536, 65536)
    assert dense_blocks(problem, resources) == ()
    choices = dense_blocks(problem, replace(resources, partial_output_tiles=True, partial_output_blocks=True))
    assert choices and all(b.output_tiles == 6 for b in choices)
    with pytest.raises(ValueError, match="candidate budget"):
        dense_blocks(
            replace(problem, m=512, n=512), replace(resources, max_m_blocks=32, max_n_blocks=32, candidate_budget=64)
        )


def test_target_price_is_checked_and_deterministic():
    problem = DenseProblem(64, 64, 64, 16, 16, 16, 1, 1, 4)
    resources = DenseResources(4096, 8192)
    with pytest.raises(ValueError, match="target price"):
        choose_dense_block(problem, resources, static_budget=64, price=lambda b: True)
    with pytest.raises(ValueError, match="nonnegative"):
        choose_dense_block(problem, resources, static_budget=True, price=lambda b: 1)
    first = choose_dense_block(problem, resources, static_budget=64, price=lambda b: 20)
    assert first == choose_dense_block(problem, resources, static_budget=64, price=lambda b: 20)


def test_large_shapes_do_not_expand_dynamic_work():
    problem = DenseProblem(1 << 30, 2048, 5632, 16, 16, 16, 1, 1, 4)
    calls = []
    choice = choose_dense_block(
        problem, DenseResources(65536, 65536), static_budget=64, price=lambda b: calls.append(b) or 32
    )
    assert choice is not None and len(calls) <= 64


def test_typed_decision_roundtrips_and_rejects_invented_storage():
    from xdsl.context import Context
    from xdsl.dialects import builtin
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import VerifyException

    from merlin.transforms.dense_ir import dense_schedule_op
    from merlin.xdsl_dialects.schedule import get_dialect

    problem = DenseProblem(64, 64, 64, 16, 16, 16, 1, 1, 4)
    resources = DenseResources(4096, 8192)
    choice = choose_dense_block(problem, resources, static_budget=64, price=lambda b: 20)
    decision = dense_schedule_op(problem, choice.block, resources, choice.static_commands, 64)
    module = builtin.ModuleOp([decision])
    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(get_dialect())
    parsed = Parser(context, str(module)).parse_module()
    parsed.verify()
    decision.attributes["storage_bytes"] = builtin.DenseArrayBase.from_list(builtin.i64, [1, 1])
    with pytest.raises(VerifyException, match="storage differs"):
        decision.verify()


def test_standalone_lifetime_export_is_opt_in_and_byte_exact(tmp_path):
    from pathlib import Path

    from merlin.compile.scheduling import block_schedule
    from merlin.compile.scheduling.export import export_dense_planner

    old = export_dense_planner(tmp_path / "old")
    assert old["schema"] == "merlin.dense_planner.v1"
    assert set(old["files_sha256"]) == {"common_dense_planning.py", "common_dense_ir.py"}
    new = export_dense_planner(tmp_path / "new", include_block_schedule=True)
    assert new["schema"] == "merlin.dense_planner.v2"
    assert (tmp_path / "new/common_block_schedule.py").read_bytes() == Path(block_schedule.__file__).read_bytes()
