from dataclasses import replace

from corefixtures.compact_dense import ADAPTER as FRESH_ADAPTER
from corefixtures.compact_dense import encode as fresh_encode
from corefixtures.compact_dense import project as fresh_project
from corefixtures.dense_retained_reduction import chosen, inputs

from merlin.compiler.kernel_ir.ir import Call, concretize, instances
from merlin.compiler.schedule.blocks import Compute, Load, Preload, Store
from merlin.compiler.schedule.dense_array_retention import (
    DenseArrayRetentionAdmission,
    propose_dense_array_retention,
    verify_dense_array_retention,
)
from merlin.compiler.schedule.dense_panel_pipeline import DRAIN


def encode(op):
    if type(op) is Preload and op.weight_row is None:
        assert op.weight_block is None
        return Call(
            "keep_descriptor",
            tuple(
                zip(
                    ("acc", "continue", "rows", "cols", "depth"),
                    (op.accumulator_row, op.accumulate, op.rows, op.cols, op.weight_rows),
                )
            ),
        )
    if type(op) is Compute and (not op.fresh_weights):
        return replace(fresh_encode(op), instr="stay")
    return fresh_encode(op)


def project(call):
    a = dict(call.args)
    if call.instr == "keep_descriptor":
        return Preload(None, None, a["acc"], a["continue"], a["rows"], a["cols"], a["depth"])
    if call.instr == "stay":
        return replace(fresh_project(replace(call, instr="compute")), fresh_weights=False)
    return fresh_project(call)


ADAPTER = replace(FRESH_ADAPTER, encode=encode, project=project)
LAW = DenseArrayRetentionAdmission(
    ADAPTER.configuration_identity, ADAPTER.numerical_policy_identity, "synthetic-array-capture-stay-output-only-v1"
)


def setup(shape=(9, 14, 25), tile=(2, 3, 4)):
    p, r, s, b, a = inputs(shape, tile)
    original = chosen(p, r, s, b, a, compute_order=("k", "n", "m"))
    return (p, r, s, b, a, original)


def execute(kernel, problem):
    lhs = [[(m * problem.k + k) % 7 - 3 for k in range(problem.k)] for m in range(problem.m)]
    rhs = [[(k * problem.n + n) % 5 - 2 for n in range(problem.n)] for k in range(problem.k)]
    memory, accumulators, output = ({}, {}, {})
    pending = array = None
    for call, env in instances(kernel):
        if call.instr == DRAIN:
            assert pending is None
            continue
        op = project(replace(call, args=concretize(call, env)))
        if type(op) is Load:
            source = lhs if op.role == "lhs" else rhs
            columns = problem.tile_k if op.role == "lhs" else problem.tile_n
            stride = problem.tile_m if op.role == "lhs" else problem.tile_k
            for r in range(op.rows):
                for c in range(op.cols):
                    memory[op.role, op.row + r + c // columns * stride, c % columns] = source[op.dram_row + r][
                        op.dram_col + c
                    ]
        elif type(op) is Preload:
            assert pending is None
            pending = op
        elif type(op) is Compute:
            assert pending is not None
            if op.fresh_weights:
                array = [
                    [memory["weight", pending.weight_row + k, n] for n in range(pending.cols)]
                    for k in range(pending.weight_rows)
                ]
            assert array is not None and len(array) == op.cols
            for r in range(op.rows):
                for n in range(pending.cols):
                    key = (pending.accumulator_row + r, n)
                    value = accumulators[key] if pending.accumulate else 0
                    for k in range(op.cols):
                        value += memory["lhs", op.input_row + r, k] * array[k][n]
                    accumulators[key] = value
            pending = None
        elif type(op) is Store:
            for r in range(op.rows):
                for n in range(op.cols):
                    key = (op.dram_row + r, op.dram_col + n)
                    assert key not in output
                    output[key] = accumulators[op.accumulator_row + r, n]
    expected = {
        (m, n): sum((lhs[m][k] * rhs[k][n] for k in range(problem.k)))
        for m in range(problem.m)
        for n in range(problem.n)
    }
    assert output == expected
    return output


def test_actual_counted_first_and_remaining_preserve_tails_products_and_completion():
    p, r, s, b, a, original = setup()
    proposal = propose_dense_array_retention(original.kernel, ADAPTER, LAW)
    report = verify_dense_array_retention(proposal, p, b, r, s, ADAPTER, a, static_budget=256)
    assert proposal.checking_kernel == original.kernel != proposal.kernel
    assert report["retained_pairs"] > 0 and report["scratchpad_rhs_retained"]
    assert execute(proposal.kernel, p) == execute(original.kernel, p)
    assert sum((call.instr == DRAIN for call, _ in instances(proposal.kernel))) == sum(
        (call.instr == DRAIN for call, _ in instances(original.kernel))
    )
    from corefixtures.scf_commands import record

    from merlin.compiler.schedule.scf_commands import materialize_scf_commands

    actual = materialize_scf_commands(proposal.kernel)
    actual.verify()
    calls = record(actual)
    assert any((name == "stay" for name, _ in calls))
    assert sum((name == "stay" for name, _ in calls)) == report["retained_pairs"]
    assert not any((name == "stay" for name, _ in record(materialize_scf_commands(original.kernel))))
