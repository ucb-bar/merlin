"""Retain complete contraction outputs and plan explicitly typed spatial readout drains."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    replace,
)
from typing import (
    Any,
)

from .execute import (
    execute,
)
from .model import (
    BlockSchedule,
    BlockScheduleError,
    Compute,
    ConvContraction,
    InitializeAccumulator,
    Op,
    Preload,
    Store,
)
from .residency import (
    check_residency,
)


@dataclass(frozen=True)
class SpatialMaxReadout:
    """Source spatial max after an explicitly bound signed scalar readout.

    Padding is in the post-readout integer domain. This declaration supplies no
    physical pooling implementation or completion semantics.
    """

    input_hw: tuple[int, int]
    window: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)  # top,left,bottom,right
    pad_value: int | None = None
    accumulator_bits: int = 32
    output_bits: int = 8

    def __post_init__(self):
        for values, length, minimum in (
            (self.input_hw, 2, 1),
            (self.window, 2, 1),
            (self.stride, 2, 1),
            (self.padding, 4, 0),
        ):
            if (
                type(values) is not tuple
                or len(values) != length
                or any(type(v) is not int or not minimum <= v < 2**31 for v in values)
            ):
                raise BlockScheduleError("spatial readout requires explicit bounded integer geometry")
        if any(type(v) is not int or v not in (8, 16, 32, 64) for v in (self.accumulator_bits, self.output_bits)):
            raise BlockScheduleError("spatial readout requires explicit signed integer widths")
        if self.pad_value is not None and (
            type(self.pad_value) is not int
            or not -(1 << (self.output_bits - 1)) <= self.pad_value < (1 << (self.output_bits - 1))
        ):
            raise BlockScheduleError("padding must be in the post-readout signed domain")
        if any(self.padding) and self.pad_value is None:
            raise BlockScheduleError("padded spatial readout requires an explicit numerical padding value")
        if min(self.output_hw) <= 0:
            raise BlockScheduleError("spatial readout has no output positions")

    @property
    def output_hw(self):
        h, w = self.input_hw
        kh, kw = self.window
        sh, sw = self.stride
        pt, pl, pb, pr = self.padding
        return ((h + pt + pb - kh) // sh + 1, (w + pl + pr - kw) // sw + 1)


@dataclass(frozen=True)
class SpatialStore:
    batch: int
    first_column: int
    columns: int
    accumulator_base: int
    output_row: int
    # Each output pixel names its complete ordered input-window rows. None is
    # the explicit source padding value; all columns share this spatial map.
    window_rows: tuple[tuple[int | None, ...], ...]
    readout: str


@dataclass(frozen=True)
class SpatialReadoutSchedule:
    retained: BlockSchedule
    prefix: tuple[Op, ...]
    stores: tuple[SpatialStore, ...]
    source: SpatialMaxReadout
    readout: str
    first_row: int
    max_window_reads: int


def retain_accumulator_outputs(schedule: BlockSchedule, *, first_row: int = 0) -> BlockSchedule:
    """Preserve every output until the final drain, in column-block-major rows.

    This finite relayout consumes the checked original schedule, preserves every
    movement/product and rechecks actual accumulator ownership after moving the
    stores. It refuses when the complete retained output does not fit.
    """
    check_residency(schedule)
    if type(first_row) is not int or first_row < 0:
        raise BlockScheduleError("retained accumulator base must be nonnegative")
    c = schedule.contraction
    d = schedule.geometry.block
    m = c.batch * c.out_h * c.out_w if isinstance(c, ConvContraction) else c.m
    n = c.co if isinstance(c, ConvContraction) else c.n
    padded_m = ((m + d - 1) // d) * d
    nt = (n + d - 1) // d
    if first_row + padded_m * nt > schedule.geometry.accumulator_rows:
        raise BlockScheduleError("complete spatial source exceeds retained accumulator capacity")

    def row(tile):
        return first_row + tile[1] * padded_m + tile[0] * d

    prefix = []
    stores = []
    pending = None
    weight = None
    for op in schedule.ops:
        if isinstance(op, InitializeAccumulator):
            prefix.append(replace(op, row=row(op.block)))
        elif isinstance(op, Preload):
            if op.weight_row is not None:
                weight = op.weight_block
            pending = (len(prefix), op, weight)
            prefix.append(op)
        elif isinstance(op, Compute):
            index, preload, current_weight = pending
            prefix[index] = replace(preload, accumulator_row=row((op.input_block[0], current_weight[1])))
            prefix.append(op)
            pending = None
        elif isinstance(op, Store):
            stores.append(replace(op, accumulator_row=row((op.dram_row // d, op.dram_col // d))))
        else:
            prefix.append(op)
    changed = replace(schedule, ops=tuple(prefix + stores))
    check_residency(changed)
    return changed


def _spatial_stores(schedule, source, readout, first_row, max_window_reads):
    if type(source) is not SpatialMaxReadout or type(readout) is not str or not readout:
        raise BlockScheduleError("spatial readout requires its exact typed source and scalar identity")
    if type(max_window_reads) is not int or max_window_reads <= 0:
        raise BlockScheduleError("spatial readout needs an explicit finite window budget")
    c = schedule.contraction
    d = schedule.geometry.block
    m = c.batch * c.out_h * c.out_w if isinstance(c, ConvContraction) else c.m
    n = c.co if isinstance(c, ConvContraction) else c.n
    h, w = source.input_hw
    oh, ow = source.output_hw
    kh, kw = source.window
    sh, sw = source.stride
    pt, pl, _, _ = source.padding
    if m % (h * w):
        raise BlockScheduleError("spatial planes must cover all contraction rows exactly")
    batches = m // (h * w)
    padded_m = ((m + d - 1) // d) * d
    nt = (n + d - 1) // d
    reads = batches * oh * ow * kh * kw * nt
    if reads > max_window_reads:
        raise BlockScheduleError(f"spatial readout exceeds explicit finite window budget: {reads} > {max_window_reads}")
    result = []
    for b in range(batches):
        for j in range(nt):
            base = first_row + j * padded_m + b * h * w
            windows = tuple(
                tuple(
                    base + (y * sh + ky - pt) * w + (x * sw + kx - pl)
                    if 0 <= y * sh + ky - pt < h and 0 <= x * sw + kx - pl < w
                    else None
                    for ky in range(kh)
                    for kx in range(kw)
                )
                for y in range(oh)
                for x in range(ow)
            )
            result.append(SpatialStore(b, j * d, min(d, n - j * d), base, b * oh * ow, windows, readout))
    return tuple(result)


def plan_spatial_max_readout(
    schedule: BlockSchedule,
    source: SpatialMaxReadout,
    readout: str,
    *,
    first_row: int = 0,
    max_window_reads: int = 65536,
) -> SpatialReadoutSchedule:
    retained = retain_accumulator_outputs(schedule, first_row=first_row)
    prefix = tuple(op for op in retained.ops if not isinstance(op, Store))
    stores = _spatial_stores(retained, source, readout, first_row, max_window_reads)
    result = SpatialReadoutSchedule(retained, prefix, stores, source, readout, first_row, max_window_reads)
    check_spatial_max_readout(result, source=source, readout=readout)
    return result


def check_spatial_max_readout(result: SpatialReadoutSchedule, *, source: SpatialMaxReadout, readout: str) -> None:
    """Check all retained source products, final lifetimes and spatial reads/writes.

    These are compiler-side conditional checks. Scalar numerical equivalence,
    target readout semantics and actual emitted effects remain separate checks.
    """
    if type(result) is not SpatialReadoutSchedule or result.source != source or result.readout != readout:
        raise BlockScheduleError("spatial readout source or scalar contract changed")
    check_residency(result.retained)
    expected = retain_accumulator_outputs(result.retained, first_row=result.first_row)
    if result.retained != expected or result.prefix != tuple(op for op in expected.ops if not isinstance(op, Store)):
        raise BlockScheduleError("spatial readout must consume the complete retained contraction before draining")
    if any(op.readout is not None for op in expected.ops if isinstance(op, Store)):
        raise BlockScheduleError("spatial source must retain raw accumulators; scalar readout belongs to its drain")
    if result.stores != _spatial_stores(expected, source, readout, result.first_row, result.max_window_reads):
        raise BlockScheduleError("spatial readout coverage, layout, padding or scalar identity changed")
    if expected.column_bias is not None and expected.column_bias.bits != source.accumulator_bits:
        raise BlockScheduleError("spatial accumulator width differs from its modular bias seed")


def execute_spatial_max_readout(
    result: SpatialReadoutSchedule, lhs: Any, weight: Any, scalar_readout: Any, *, bias: Any = None
) -> Any:
    """Neutral numerical execution; it is not a target readout implementation."""
    import numpy as np

    check_spatial_max_readout(result, source=result.source, readout=result.readout)
    source = result.source
    raw = execute(result.retained, lhs, weight, bias=bias).astype(f"int{source.accumulator_bits}")
    h, w = source.input_hw
    oh, ow = source.output_hw
    m, n = raw.shape
    d = result.retained.geometry.block
    padded_m = ((m + d - 1) // d) * d
    output = np.zeros((m // (h * w) * oh * ow, n), dtype=f"int{source.output_bits}")
    low, high = -(1 << (source.output_bits - 1)), (1 << (source.output_bits - 1)) - 1
    for store in result.stores:
        for pixel, window in enumerate(store.window_rows):
            for column in range(store.columns):
                values = []
                for address in window:
                    if address is None:
                        value = source.pad_value
                    else:
                        row = address - result.first_row - (store.first_column // d) * padded_m
                        value = scalar_readout(int(raw[row, store.first_column + column]))
                    if type(value) is not int or not low <= value <= high:
                        raise BlockScheduleError("scalar readout result differs from declared signed output domain")
                    values.append(value)
                output[store.output_row + pixel, store.first_column + column] = max(values)
    return output
