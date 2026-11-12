from dataclasses import replace

from merlin.compiler.kernel_ir.ir import Call, Loop, as_expr
from merlin.compiler.schedule.blocks import Compute, Load, Preload, Store
from merlin.compiler.schedule.compact_dense import (
    DenseLoopAdapter,
    DenseLoopStorage,
    DenseMemoryView,
)
from merlin.transforms.dense_planning import DenseProblem, DenseResources


def encode(op):
    if isinstance(op, Load):
        names = ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step")
        values = (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step)
        instruction = op.role + "_load"
    elif isinstance(op, Preload):
        names = ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth")
        values = (op.weight_row, *op.weight_block, op.accumulator_row, op.accumulate, op.rows, op.cols, op.weight_rows)
        instruction = "preload"
    elif isinstance(op, Compute):
        names = ("lhs", "m", "k", "rows", "depth")
        values = (op.input_row, *op.input_block, op.rows, op.cols)
        instruction = "compute"
    else:
        assert isinstance(op, Store)
        names = ("row", "col", "rows", "cols", "acc")
        values = (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row)
        instruction = "store"
    return Call(instruction, tuple(zip(names, map(as_expr, values))))


def project(call):
    a = dict(call.args)
    if call.instr in ("lhs_load", "weight_load"):
        return Load(
            call.instr.removesuffix("_load"),
            (a["a"], a["b"]),
            a["src_row"],
            a["src_col"],
            a["rows"],
            a["cols"],
            a["dst"],
            a["step"],
        )
    if call.instr == "preload":
        return Preload(a["weight"], (a["k"], a["n"]), a["acc"], a["continue"], a["rows"], a["cols"], a["depth"])
    if call.instr == "compute":
        return Compute(a["lhs"], (a["m"], a["k"]), a["rows"], a["depth"], True)
    if call.instr == "store":
        return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
    raise ValueError("unknown instruction")


ADAPTER = DenseLoopAdapter(encode, project, "neutral-test-config-v1", "source-ordered-test-policy-v1", True, True, 64)


def fixture(*, shape=(5, 14, 13), tile=(2, 3, 4), widths=(1, 1, 4), padded=False):
    p = DenseProblem(*shape, *tile, *widths)
    lhs_width = p.tile_k * p.lhs_bytes + (4 if padded else 0)
    rhs_width = p.tile_n * p.rhs_bytes + (6 if padded else 0)
    acc_width = p.tile_n * p.accumulator_bytes + (8 if padded else 0)
    s = DenseLoopStorage(
        DenseMemoryView("left", "left_sram", lhs_width, 2, 12),
        DenseMemoryView("right", "right_sram", rhs_width, 3, 12),
        DenseMemoryView("sums", "acc_sram", acc_width, 1, 12),
    )
    r = DenseResources(
        12 * (lhs_width + rhs_width), 12 * acc_width, 3, 3, 2, partial_output_tiles=True, partial_output_blocks=True
    )
    return (p, r, s)


def mutate(kernel, instruction, operand=None, value=None, *, omit=False):
    changed = False

    def walk(body):
        nonlocal changed
        out = []
        for statement in body:
            if isinstance(statement, Loop):
                out.append(replace(statement, body=walk(statement.body)))
            elif not changed and statement.instr == instruction:
                changed = True
                if not omit:
                    out.append(
                        replace(
                            statement,
                            args=tuple(((k, as_expr(value) if k == operand else v) for k, v in statement.args)),
                        )
                    )
            else:
                out.append(statement)
        return tuple(out)

    result = replace(kernel, body=walk(kernel.body))
    assert changed
    return result
