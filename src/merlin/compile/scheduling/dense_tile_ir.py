"""Logical tile calls for the existing checked dense loop IR.

These are synchronous logical operations, not machine instructions. An OOT
lowerer can expand a call into load/wait/push/compute/readout sequences. It must
check that realization separately; the shared proof covers tile coverage,
storage and reduction order only. No hardware qualification is implied.
"""

from merlin.sched.ir import Call, as_expr

from .block_schedule import Compute, Load, Preload, Store


_FIELDS = {
    "dense.load_lhs": ("block0", "block1", "src_row", "src_col", "rows", "cols", "dst", "step"),
    "dense.load_rhs": ("block0", "block1", "src_row", "src_col", "rows", "cols", "dst", "step"),
    "dense.preload": ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth"),
    "dense.compute": ("lhs", "m", "k", "rows", "depth", "preloaded"),
    "dense.store": ("row", "col", "rows", "cols", "acc"),
}


def encode_dense_tile(op):
    """Retain every effect operand in a regular Kernel Call."""
    if type(op) is Load and op.role in ("lhs", "weight") and op.gather is None:
        name = "dense.load_lhs" if op.role == "lhs" else "dense.load_rhs"
        values = (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step)
    elif type(op) is Preload:
        name = "dense.preload"
        values = (op.weight_row, *op.weight_block, op.accumulator_row, op.accumulate,
                  op.rows, op.cols, op.weight_rows)
    elif type(op) is Compute and op.input_gather is None:
        name = "dense.compute"
        values = (op.input_row, *op.input_block, op.rows, op.cols, int(op.fresh_weights))
    elif type(op) is Store:
        name = "dense.store"
        values = (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row)
    else:
        raise ValueError("logical dense tiles admit loads, preload, compute and raw accumulator store only")
    return Call(name, tuple(zip(_FIELDS[name], (as_expr(int(v) if type(v) is bool else v) for v in values))))


def project_dense_tile(call):
    """Decode actual call arguments, rejecting unknown fields and effects."""
    if (type(call) is not Call or call.instr not in _FIELDS or
            tuple(k for k, _ in call.args) != _FIELDS[call.instr] or
            call.unit is not None or call.produces is not None or call.awaits or call.stages or
            call.sets or call.assumes):
        raise ValueError("expected a closed synchronous logical dense tile call")
    a = dict(call.args)
    if call.instr in ("dense.load_lhs", "dense.load_rhs"):
        return Load("lhs" if call.instr == "dense.load_lhs" else "weight", (a["block0"], a["block1"]),
                    a["src_row"], a["src_col"], a["rows"], a["cols"], a["dst"], a["step"])
    if call.instr == "dense.preload":
        return Preload(a["weight"], (a["k"], a["n"]), a["acc"], a["continue"], a["rows"], a["cols"], a["depth"])
    if call.instr == "dense.compute":
        from merlin.sched.ir import Const

        if a["preloaded"] != Const(1):
            raise ValueError("logical dense compute requires explicit preload")
        return Compute(a["lhs"], (a["m"], a["k"]), a["rows"], a["depth"], True)
    return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
