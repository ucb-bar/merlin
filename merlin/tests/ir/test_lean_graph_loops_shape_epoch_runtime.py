"""Finite four-buffer observations, separate from kernel and native fidelity.

The unchanged shape interpreter covers source/normalized, then the unchanged
memory interpreter covers normalized/reused/segmented. The normalized buffer
and nonzero observations are common to both checks. This is empirical evidence,
not an assumed semantic bridge in the Lean theorem.
"""

from __future__ import annotations

import json

import pytest

from merlin.tests.ir.test_lean_graph_loops_shape import parse
from merlin.tests.ir.test_lean_graph_loops_shape_epoch import case_paths, mutate, project
from merlin.tests.ir.test_lean_graph_loops_shape_runtime import array
from merlin.verify.lean_graph_loops_memory_scf_runtime import execute as memory_execute
from merlin.verify.lean_graph_loops_shape_epoch_cases import CASES
from merlin.verify.lean_graph_loops_shape_runtime import execute as shape_execute


@pytest.mark.parametrize("case", CASES[:-1], ids=lambda case: case.name)
@pytest.mark.parametrize("seeds", ((17, 29, 91), (-5, 9, 3), (0, 0, 7)))
def test_finite_complete_observations(case, seeds, tmp_path):
    paths = case_paths(case)

    def arguments():
        return (*seeds[:2], array((1,), case.element_bits, seeds[2], case.index_bits))

    shape = [
        shape_execute(parse(path.read_text()), arguments(), function="kernel", index_bits=case.index_bits)
        for path in paths[:2]
    ]
    memory = [memory_execute(parse(path.read_text()), arguments(), index_bits=case.index_bits) for path in paths[1:]]
    (tmp_path / "FINITE.json").write_text(json.dumps(dict(shape=shape, memory=memory), indent=2) + "\n")
    assert shape[0] == shape[1]
    expected = 2 * seeds[2] + 4 * seeds[0] + 2 * seeds[1]
    caller = seeds[2] + 2 * seeds[0] + 2 * seeds[1]
    assert shape[0]["outputs"] == [expected, case.rows]
    assert shape[0]["input_memories"][0]["cells"] == [caller]
    assert all(list(row["result"]) == [expected, case.rows] and row["caller"] == [caller] for row in memory)
    assert memory[0]["trace"] == memory[1]["trace"] == memory[2]["trace"]
    assert memory[0]["allocation_count"] == 2
    assert memory[1]["allocation_count"] == memory[2]["allocation_count"] == 1
    assert all(row["all_private_reads_initialized"] and row["all_private_frees_valid"] for row in memory)
    assert not any(row["actual_native_execution"] for row in shape + memory)


def test_valid_wrong_operand_has_nonzero_counterexample(tmp_path):
    case = CASES[0]
    modules = [parse(path.read_text()) for path in case_paths(case)]
    before = memory_execute(modules[3], (17, 29, array((1,), 32, 91, 64)), index_bits=64)
    mutate(modules[3], "wrong_scalar_operand")
    after = memory_execute(modules[3], (17, 29, array((1,), 32, 91, 64)), index_bits=64)
    (tmp_path / "COUNTEREXAMPLE.json").write_text(json.dumps(dict(before=before, after=after), indent=2) + "\n")
    assert before["result"] == (308, 2) and before["caller"] == [183]
    assert after["result"] == (332, 2) and after["caller"] == [195]
    with pytest.raises(ValueError):
        project(modules, tmp_path / "refusal")


def test_private_free_mutation_is_not_a_successful_equivalent_run():
    case = CASES[0]
    module = parse(case_paths(case)[3].read_text())
    mutate(module, "wrong_free")
    with pytest.raises(ValueError, match="invalid/double private free"):
        memory_execute(module, (17, 29, array((1,), 32, 91, 64)), index_bits=64)


def test_large_symbolic_case_is_not_finitely_unrolled():
    case = CASES[-1]
    with pytest.raises(ValueError, match="finite allocation outside"):
        shape_execute(
            parse(case_paths(case)[0].read_text()), (17, 29, array((1,), 32, 91, 64)), function="kernel", index_bits=64
        )
