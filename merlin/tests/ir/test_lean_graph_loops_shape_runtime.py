"""Independent finite xDSL observations and symbolic whole-case Lean witnesses."""

from __future__ import annotations

import json
import os
import subprocess
from math import prod
from pathlib import Path

import pytest
from xdsl.dialects import builtin
from xdsl.interpreters.memref import xtype_for_el_type
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr

from merlin.tests.ir.test_lean_graph_loops_shape import check, context_data, mutate, parse
from merlin.verify.lean_graph_loops_memory_scf import audit
from merlin.verify.lean_graph_loops_shape_cases import CASES, Case
from merlin.verify.lean_graph_loops_shape_model import SHAPE_ROOTS
from merlin.verify.lean_graph_loops_shape_runtime import execute
from merlin.verify.lean_graph_loops_shape_witness import WITNESS_ROOTS, witness_obligation


@pytest.fixture(scope="module")
def context():
    return context_data()


FINITE_CASES = tuple(case for case in CASES if case.name != "compact_huge") + (
    Case("callee_dimensions", (3, 5), 32, 1, callee=True),
)


def array(shape, bits, value, index_bits):
    return ShapedArray(
        TypedPtr.new([value] * prod(shape), xtype=xtype_for_el_type(builtin.IntegerType(bits), index_bits)), list(shape)
    )


def arguments(case, *, alias=False):
    left = array(case.shape, case.bits, 17, case.index_bits)
    right = left if alias else array(case.shape, case.bits, 29, case.index_bits)
    return left, right, array((1,), case.bits, 91, case.index_bits)


@pytest.mark.parametrize("case", FINITE_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("alias", (False, True), ids=("distinct", "same_input_allocation"))
def test_finite_whole_observations(context, tmp_path, case, alias):
    _, paths = context
    before, after, _ = paths[case.name]
    results = [
        execute(parse(path.read_text()), arguments(case, alias=alias), function="probe", index_bits=case.index_bits)
        for path in (before, after)
    ]
    (tmp_path / "FINITE.json").write_text(json.dumps(results, indent=2) + "\n")
    assert results[0] == results[1]
    expected = [case.shape[case.axis], case.shape[(case.axis + 1) % len(case.shape)], 91 if 0 in case.shape else 17]
    assert results[0]["outputs"] == expected
    assert results[0]["input_memories"][-1]["cells"] == [expected[-1]]
    assert not results[0]["actual_native_execution"]


def test_finite_memory_chain(context, tmp_path):
    _, paths = context
    before, after, _ = paths["shape_memory_chain"]
    results = [
        execute(parse(path.read_text()), (17, 29, array((1,), 32, 91, 64)), function="kernel", index_bits=64)
        for path in (before, after)
    ]
    (tmp_path / "FINITE.json").write_text(json.dumps(results, indent=2) + "\n")
    assert results[0] == results[1]
    assert results[0]["input_memories"][0]["cells"] != [91]
    assert sum(row["operation"] == "memref.alloc" and row["phase"] == "after" for row in results[0]["trace"]) == 2


def test_runtime_descriptor_mismatch_refuses(context):
    _, paths = context
    module = parse(paths["rank_two"][0].read_text())
    args = (array((3, 2), 32, 17, 64), array((2, 3), 32, 29, 64), array((1,), 32, 91, 64))
    with pytest.raises(ValueError, match="runtime descriptor violates static source type"):
        execute(module, args, function="probe", index_bits=64)


def test_zero_view_read_is_not_inhabited(context):
    from xdsl.dialects import memref

    _, paths = context
    module = parse(paths["zero_metadata_only"][0].read_text())
    # Construct a compiler-valid rank-two load from the empty view.
    function = module.body.block.first_op
    load = next(op for op in function.walk() if isinstance(op, memref.LoadOp))
    zero = function.body.block.first_op.result
    replacement = memref.LoadOp.get(function.body.block.args[0], [zero, zero])
    load.results[0].replace_all_uses_with(replacement.results[0])
    function.body.block.insert_op_before(replacement, load)
    function.body.block.erase_op(load)
    module.verify()
    case = next(case for case in CASES if case.name == "zero_metadata_only")
    with pytest.raises(ValueError, match="out-of-bounds"):
        execute(module, arguments(case), function="probe", index_bits=64)


@pytest.mark.parametrize("mutation", ("wrong_buffer", "wrong_dimension"))
def test_valid_mutation_has_nonzero_observable_counterexample(context, tmp_path, mutation):
    _, paths = context
    before, after, _ = paths["rank_two"]
    case = next(case for case in CASES if case.name == "rank_two")
    source, candidate = parse(before.read_text()), parse(after.read_text())
    mutate(candidate, mutation)
    results = [execute(module, arguments(case), function="probe", index_bits=64) for module in (source, candidate)]
    assert results[0]["outputs"] != results[1]["outputs"]
    (tmp_path / "COUNTEREXAMPLE.json").write_text(json.dumps(results, indent=2) + "\n")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_actual_inhabited_lean_witness(context, tmp_path, case):
    selection, paths = context
    before, after, _ = paths[case.name]
    checked = check(context, tmp_path / "fresh_actual", name=case.name)
    assert checked["lean_complete_shape_checked"]
    projection = json.loads(Path(checked["model"]["path"]).read_bytes())
    query = tmp_path / "Witness.lean"
    query.write_text(witness_obligation(projection))
    bindings = selection.dependencies()
    env = dict(
        os.environ,
        LEAN_PATH=str(selection.proof.library),
        LEAN_SYSROOT=str(selection.proof.executable.parent.parent),
        LEAN_NUM_THREADS="2",
    )
    command = [str(selection.proof.executable), str(query)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=180)
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert selection.dependencies() == bindings
    assert result.returncode == 0, result.stdout + result.stderr
    roots = audit(result.stdout, SHAPE_ROOTS + WITNESS_ROOTS)
    (tmp_path / "WITNESS.json").write_text(
        json.dumps(
            dict(
                returncode=result.returncode,
                roots=roots,
                source=str(before),
                candidate=str(after),
                qualified=False,
                native_conformance_proved=False,
            ),
            indent=2,
        )
        + "\n"
    )
    assert len(roots) == 41
