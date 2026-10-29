"""Compact periodic proposals preserve complete source dots and private windows."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.periodic_dense import build_periodic_staged_dense, verify_periodic_staged_dense
from merlin.compile.scheduling.staged_dense import StagedDenseContract, StagedDenseProblem
from merlin.sched.ir import Loop, as_expr, select
from merlin.sched.ir.expr import free_vars
from merlin.sched.ir.kernel import concretize, instances


def contract(**changes):
    return replace(
        StagedDenseContract(
            "selected-facts",
            "selected-effects",
            "signed-i8-ordered-zero-i32",
            "explicit-test-fixture",
            4,
            16,
            4,
            8,
            32,
            32,
            "read_at_compute_consume",
            "data_commands_preserve_pending_descriptor",
            "raw_accumulator_identity",
            True,
            True,
            False,
        ),
        **changes,
    )


def execute(kernel, problem):
    """Independent scalar model of actual macro operands, only for small tests."""
    lhs = [[((row * 11 + k * 5) % 17) - 8 for k in range(problem.k)] for row in range(problem.m)]
    rhs = [[((k * 7 + col * 3) % 19) - 9 for col in range(problem.n)] for k in range(problem.k)]
    local = {}
    acc = {}
    output = {}
    for call, environment in instances(kernel):
        a = concretize(call, environment)
        if call.instr in ("load_lhs", "load_rhs"):
            source = lhs if call.instr == "load_lhs" else rhs
            for row in range(a["rows"]):
                for col in range(a["cols"]):
                    local[a["local_row"] + row, col] = source[a["source_row"] + row][a["source_col"] + col]
        elif call.instr == "paired_compute":
            for row in range(a["rows"]):
                for col in range(a["cols"]):
                    partial = sum(local[a["lhs_row"] + row, k] * local[a["rhs_row"] + k, col] for k in range(a["k"]))
                    key = (a["acc_row"] + row, col)
                    acc[key] = (acc[key] if a["continuation"] else 0) + partial
        elif call.instr == "store":
            for row in range(a["rows"]):
                for col in range(a["cols"]):
                    key = (a["output_row"] + row, a["output_col"] + col)
                    assert key not in output
                    output[key] = acc[a["acc_row"] + row, col]
        else:
            assert call.instr == "drain"
    expected = {
        (row, col): sum(lhs[row][k] * rhs[k][col] for k in range(problem.k))
        for row in range(problem.m)
        for col in range(problem.n)
    }
    assert output == expected


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 2), (5, 9, 7), (9, 12, 6), (5, 17, 7)])
@pytest.mark.parametrize("prefetch", [False, True])
def test_small_tails_and_period_boundaries_execute_exactly(shape, prefetch):
    problem = StagedDenseProblem(*shape)
    selected = contract()
    kernel = build_periodic_staged_dense(problem, selected, prefetch=prefetch)
    report = verify_periodic_staged_dense(kernel, problem, selected, prefetch=prefetch)
    assert report["dynamic_products"] == problem.m * problem.k * problem.n
    assert report["complete_outputs"] == problem.m * problem.n
    assert report["timing_authority"] == "NONE"
    execute(kernel, problem)


def mutate_call(kernel, predicate, change):
    changed = False

    def walk(body):
        nonlocal changed
        result = []
        for statement in body:
            if isinstance(statement, Loop):
                statement = replace(statement, body=walk(statement.body))
            elif not changed and predicate(statement):
                statement = change(statement)
                changed = True
            result.append(statement)
        return tuple(result)

    result = replace(kernel, body=walk(kernel.body))
    assert changed
    return result


def operand(call, name, value):
    return replace(call, args=tuple((key, as_expr(value) if key == name else original) for key, original in call.args))


@pytest.mark.parametrize("defect", ["source", "alias", "continuation", "hidden_effect", "drain", "output_extent"])
def test_adversarial_macro_or_ownership_change_refuses(defect):
    problem = StagedDenseProblem(5, 17, 7)
    selected = contract()
    kernel = build_periodic_staged_dense(problem, selected)
    if defect == "source":
        bad = mutate_call(kernel, lambda c: c.instr == "load_lhs", lambda c: operand(c, "source_col", 1))
    elif defect == "alias":
        bad = mutate_call(kernel, lambda c: c.instr == "load_rhs", lambda c: operand(c, "local_row", 0))
    elif defect == "continuation":
        bad = mutate_call(kernel, lambda c: c.instr == "paired_compute", lambda c: operand(c, "continuation", 1))
    elif defect == "hidden_effect":
        bad = mutate_call(kernel, lambda c: c.instr == "load_lhs", lambda c: replace(c, unit="unmodeled"))
    elif defect == "drain":
        bad = mutate_call(kernel, lambda c: c.instr == "drain", lambda c: replace(c, instr="not_a_drain"))
    else:
        bad = mutate_call(kernel, lambda c: c.instr == "store", lambda c: operand(c, "cols", selected.dim + 1))
    with pytest.raises(ValueError):
        verify_periodic_staged_dense(bad, problem, selected)


def test_distant_selector_mutation_cannot_hide_outside_a_sampled_prefix():
    problem = StagedDenseProblem(5, 40001, 7)
    selected = contract()
    kernel = build_periodic_staged_dense(problem, selected)

    def during_period(call):
        return call.instr == "load_lhs" and "pi" in free_vars(dict(call.args)["source_col"])

    bad = mutate_call(
        kernel, during_period, lambda c: operand(c, "source_col", dict(c.args)["source_col"] + select("pi", 123, 1, 0))
    )
    with pytest.raises(ValueError, match="source/lifetime invariant"):
        verify_periodic_staged_dense(bad, problem, selected)


@pytest.mark.parametrize("prefetch", [False, True])
def test_huge_shapes_retain_bounded_ir_and_complete_symbolic_check(prefetch):
    problem = StagedDenseProblem(1_000_003, 1_000_001, 1_000_005)
    selected = contract()
    kernel = build_periodic_staged_dense(problem, selected, prefetch=prefetch)
    assert len(kernel.text()) < 20_000
    report = verify_periodic_staged_dense(kernel, problem, selected, prefetch=prefetch)
    assert report["static_calls"] < 32
    assert report["dynamic_products"] == problem.m * problem.k * problem.n


@pytest.mark.parametrize("premise", ["operand_rows", "accumulator_rows", "external_buffers_disjoint"])
def test_missing_capacity_or_ownership_refuses(premise):
    problem = StagedDenseProblem(5, 17, 7)
    changes = {premise: False if premise == "external_buffers_disjoint" else 1}
    selected = contract(**changes)
    kernel = build_periodic_staged_dense(problem, selected)
    with pytest.raises(ValueError, match="two disjoint operand-pair windows"):
        verify_periodic_staged_dense(kernel, problem, selected)


def test_prospective_inventory_keeps_all_frozen_versions_unchanged():
    from merlin.compile.scheduling import library_sources as owners

    assert owners.library_modules() == owners.LIBRARY_MODULES_V6
    for version in (6, 7, 8):
        assert owners.library_modules(version) == getattr(owners, f"LIBRARY_MODULES_V{version}")
        assert "merlin.compile.scheduling.periodic_dense" not in owners.library_modules(version)
    assert owners.library_modules(9) == (*owners.LIBRARY_MODULES_V8, "merlin.compile.scheduling.periodic_dense")
