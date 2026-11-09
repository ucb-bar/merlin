"""Execute named scheduled addresses as a neutral host reference, without target inference."""

from __future__ import annotations

from typing import (
    Any,
)

from .model import (
    LHS,
    WEIGHT,
    BlockSchedule,
    BlockScheduleError,
    Compute,
    ConvContraction,
    InitializeAccumulator,
    Load,
    Preload,
    Store,
)
from .residency import (
    check_residency,
)


def execute(schedule: BlockSchedule, lhs: Any, weight: Any, *, bias: Any = None) -> Any:
    """Run ``schedule`` on host arrays exactly as an in-order device would, and return the output.

    The schedule's own oracle: every row a load writes, every block a preload makes resident, every
    accumulate and every drain is simulated at the addresses the schedule names -- nothing is recomputed
    from the contraction. So a schedule that addresses the wrong rows, drains the wrong accumulator block
    or reuses a stale resident block produces a wrong ``out``, which a direct ``lhs @ weight`` exposes in
    milliseconds, before any backend or simulator is involved.
    """
    import numpy as np

    c = schedule.contraction
    lhs, weight = np.asarray(lhs), np.asarray(weight)
    if isinstance(c, ConvContraction):
        want_lhs, want_weight, out_shape = (
            (c.batch * c.in_h * c.in_w, c.ci),
            (c.kh * c.kw * c.ci, c.co),
            (c.pixels, c.co),
        )
    else:
        want_lhs, want_weight, out_shape = (c.m, c.k), (c.k, c.n), (c.m, c.n)
    if lhs.shape != want_lhs or weight.shape != want_weight:
        raise BlockScheduleError(
            f"operands {lhs.shape} x {weight.shape} are not the scheduled {want_lhs} x {want_weight}"
        )
    if schedule.column_bias is None:
        if bias is not None:
            raise BlockScheduleError("undeclared bias operand")
    else:
        check_residency(schedule)
        bias = np.asarray(bias)
        if bias.shape != (out_shape[1],) or bias.dtype != np.dtype(f"int{schedule.column_bias.bits}"):
            raise BlockScheduleError("bias values differ from exact source columns/accumulator width")
        if not all(np.issubdtype(x.dtype, np.signedinteger) for x in (lhs, weight)):
            raise BlockScheduleError("modular bias execution requires signed integer operands")
    dram = {LHS: lhs, WEIGHT: weight}
    rows_on_chip: dict[int, Any] = {}
    accumulator: dict[int, Any] = {}
    out = np.zeros(
        out_shape, dtype=np.result_type(lhs, weight, np.int64) if schedule.column_bias is None else bias.dtype
    )
    resident = None
    target_row, accumulate = None, False

    def _row(row: int, what: str) -> Any:
        if row not in rows_on_chip:
            raise BlockScheduleError(f"{what} reads on-chip row {row}, which no load ever wrote")
        return rows_on_chip[row]

    def _accumulated(row: int, what: str) -> Any:
        if row not in accumulator:
            raise BlockScheduleError(f"{what} reads accumulator block {row}, which no compute ever wrote")
        return accumulator[row]

    for op in schedule.ops:
        if isinstance(op, InitializeAccumulator):
            accumulator[op.row] = np.broadcast_to(bias[op.dram_col : op.dram_col + op.cols], (op.rows, op.cols)).copy()
        elif isinstance(op, Load):
            source = dram[op.role]
            if op.gather is not None:
                for offset, entry in enumerate(op.gather):
                    rows_on_chip[op.row + offset] = (
                        np.zeros(op.cols, dtype=source.dtype)
                        if entry is None
                        else source[entry[0], entry[1] : entry[1] + op.cols].copy()
                    )
                continue
            block = source[op.dram_row : op.dram_row + op.rows, op.dram_col : op.dram_col + op.cols]
            for offset in range(op.rows):
                rows_on_chip[op.row + offset] = block[offset].copy()
        elif isinstance(op, Preload):
            if op.weight_row is not None:
                resident = np.stack(
                    [_row(op.weight_row + r, "a preload")[: op.cols] for r in range(op.weight_rows or op.rows)]
                )
            target_row, accumulate = op.accumulator_row, op.accumulate
        elif isinstance(op, Compute):
            if resident is None:
                raise BlockScheduleError("a compute ran with no resident block")
            streamed = np.stack([_row(op.input_row + r, "a compute")[: op.cols] for r in range(op.rows)])
            product = streamed.astype(out.dtype) @ resident.astype(out.dtype)
            if accumulate:
                accumulator[target_row] = _accumulated(target_row, "an accumulating compute") + product
            else:
                accumulator[target_row] = product
        elif isinstance(op, Store):
            if op.readout is not None:
                raise BlockScheduleError("block execution has no qualified implementation of the selected readout")
            out[op.dram_row : op.dram_row + op.rows, op.dram_col : op.dram_col + op.cols] = _accumulated(
                op.accumulator_row, "a store"
            )[: op.rows, : op.cols]
    return out
