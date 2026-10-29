"""Bounded source-exact retention of quantized spatial rows across readouts.

This is a counted storage schedule, not a new arithmetic or asynchronous law.
A completed Q1 i8 row is reloaded unchanged into a disjoint caller-owned i32
row window. Original clipped border copies are retained too; native padding
is not required. Consecutive max readouts retain overlapping rows until an explicit
linear epoch ends. Original Q1 stores, Q2 conversion and signed-min padding
remain separate. Target operand decoding and completion are adapter obligations.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.rolling_spatial import RollingPackedSpatialPlan


@dataclass(frozen=True)
class SpatialReadoutCache:
    source: RollingPackedSpatialPlan
    producer: AddressWindow
    cache: AddressWindow
    row_bytes: int
    output_rows: int

    def __post_init__(self):
        if type(self.source) is not RollingPackedSpatialPlan:
            raise ValueError("readout cache requires the complete original rolling source")
        self.source.__post_init__()
        p, s = self.source.packed.source.producer, self.source.packed.source.spatial
        for w in (self.producer, self.cache):
            if type(w) is not AddressWindow:
                raise ValueError("readout cache requires explicit caller-owned windows")
            w.__post_init__()
        if (
            type(self.row_bytes) is not int
            or self.row_bytes != p.primitive * 4
            or type(self.output_rows) is not int
            or not 2 <= self.output_rows <= s.output_hw[0]
            or s.window != (3, 3)
            or s.stride != (2, 2)
            or s.padding != (1, 1, 1, 1)
            or s.pad_value != -128
            or self.source.packed.source.readout_mode != "staged_readouts"
            or self.producer.space != self.cache.space
            or self.producer.start != 0
            or self.producer.size % self.row_bytes
            or self.cache.start % self.row_bytes
            or self.cache.size % self.row_bytes
            or self.producer.start + self.producer.size > self.cache.start
            or self.cache.start + self.cache.size > p.accumulator_rows * self.row_bytes
            or self.required_rows * self.row_bytes > self.cache.size
        ):
            raise ValueError(
                "readout retention needs disjoint assigned complete-row capacity and supported source geometry"
            )
        # This route keeps the old clipped horizontal halo. Its original proof
        # requires producer ReLU; invalid horizontal values then cannot win max.
        if any(not c.producer.relu for c in self.source.packed.source.readouts):
            raise ValueError("clipped horizontal halo requires the original producer ReLU")

    @property
    def span(self):
        return (self.output_rows - 1) * 2 + 3

    @property
    def width(self):
        return (self.source.packed.source.output_tile_width - 1) * 2 + 3

    @property
    def planes(self):
        p = self.source.packed.source.producer
        return (p.output_channels + p.column_tile - 1) // p.column_tile

    @property
    def required_rows(self):
        return self.span * self.width * self.planes

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def report(self):
        self.__post_init__()
        source = self.source.packed.source
        p, s = source.producer, source.spatial
        height, _ = s.input_hw
        oh, ow = s.output_hw
        # Each epoch stages the exact union of original logical halo rows.
        # Border slots keep the ORIGINAL clipped-row copies; no new native
        # padding capability is assumed. One epoch overlap row is reloaded. Source shape,
        # not target timing, determines these exact logical byte counts.
        epochs = (oh + self.output_rows - 1) // self.output_rows
        staged = 2 * oh + 1 + max(0, epochs - 1)
        whole, tail = divmod(ow, source.output_tile_width)
        halo_columns = whole * self.width + ((tail - 1) * 2 + 3 if tail else 0)
        factor = p.window.input_shape[0] * halo_columns * p.output_channels
        return dict(
            schema="merlin.spatial_readout_cache.v1",
            identity=self.identity,
            original_source_identity=source.identity,
            producer_window=asdict(self.producer),
            cache_window=asdict(self.cache),
            cache_rows=self.required_rows,
            output_rows_per_epoch=self.output_rows,
            original_reload_bytes=oh * 3 * factor,
            retained_reload_bytes=staged * factor,
            q1_store_bytes=height * factor,
            arithmetic="original Q1 and Q2 unchanged",
            completion="every actual reload and pooled store completes before subsequent use",
            target_qualification=False,
        )


def propose_spatial_readout_cache(source, producer, cache, *, row_bytes):
    """Choose the largest bounded linear consumer epoch; no capacity guess."""
    if (
        type(source) is not RollingPackedSpatialPlan
        or type(cache) is not AddressWindow
        or type(producer) is not AddressWindow
    ):
        raise ValueError("explicit source and caller-owned readout capacity required")
    source.__post_init__()
    producer.__post_init__()
    cache.__post_init__()
    p, s = source.packed.source.producer, source.packed.source.spatial
    if (
        s.window != (3, 3)
        or s.stride != (2, 2)
        or s.padding != (1, 1, 1, 1)
        or s.pad_value != -128
        or any(not c.producer.relu for c in source.packed.source.readouts)
    ):
        return None
    if type(row_bytes) is not int or row_bytes <= 0:
        raise ValueError("positive exact physical row width required")
    width = (source.packed.source.output_tile_width - 1) * 2 + 3
    planes = (p.output_channels + p.column_tile - 1) // p.column_tile
    span = cache.size // row_bytes // (width * planes)
    rows = min(s.output_hw[0], (span - 3) // 2 + 1)
    if rows < 2:
        return None
    return SpatialReadoutCache(source, producer, cache, row_bytes, rows)


def emit_cached_spatial_max(plan, builder, lower, *, image, pool_y, output_x, output_pixels, synchronous_completion):
    """Emit actual frontier reloads and original pooled stores in counted control.

    ``lower`` must bind reload_intermediate to the selected exact i8->i32 load,
    store_spatial_max to original Q2+max, and complete each before returning.
    Dense producer work MUST stay in plan.producer; this independent assignment
    is deliberately explicit rather than inferred from a semantic label.
    """
    if type(plan) is not SpatialReadoutCache or synchronous_completion is not True:
        raise ValueError("readout cache needs checked plan and actual synchronous binding")
    plan.__post_init__()
    source = plan.source.packed.source
    p, s = source.producer, source.spatial
    height, _ = s.input_hw
    oh, ow = s.output_hw
    if type(output_pixels) is not int or not 0 < output_pixels <= source.output_tile_width:
        raise ValueError("actual stripe exceeds the checked readout footprint")
    op = builder.op
    width = (output_pixels - 1) * 2 + 3
    epoch_index = op("udiv", pool_y, plan.output_rows)
    epoch_first = op("mul", epoch_index, plan.output_rows)
    epoch_start = op("sub", op("mul", epoch_first, 2), 1)
    first = builder.cmp("eq", pool_y, epoch_first)
    original_first = op("sub", op("mul", pool_y, 2), 1)

    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            builder.loop(0, whole, 1, lambda i: body(op("mul", i, tile), tile))
        if tail:
            body(whole * tile, tail)

    def channels(column, count):
        plane = op("udiv", column, p.column_tile)
        base = op("add", plan.cache.start // plan.row_bytes, op("mul", plane, plan.span * width))

        def row(ky):
            original_y = op("add", original_first, ky)
            clipped = builder.select(builder.cmp("slt", original_y, 0), 0, original_y)
            clipped = builder.select(builder.cmp("slt", clipped, height), clipped, height - 1)
            needed = builder.cmp(
                "slt", 0, op("add", builder.select(first, 1, 0), builder.select(builder.cmp("slt", 0, ky), 1, 0))
            )

            def stage():
                def chunk(offset, rows):
                    lower(
                        "reload_intermediate",
                        dict(
                            accumulator_row=op(
                                "add", base, op("add", op("mul", op("sub", original_y, epoch_start), width), offset)
                            ),
                            scratch_offset=op(
                                "add",
                                plan.source.packed.workspace.start,
                                op(
                                    "add",
                                    op(
                                        "mul",
                                        op("add", op("mul", op("urem", clipped, 3), width), offset),
                                        p.output_channels,
                                    ),
                                    column,
                                ),
                            ),
                            scratch_row_stride=p.output_channels,
                            rows=rows,
                            columns=count,
                            stripe_rows=3 * width,
                            output_column=column,
                            readout_tile=plane,
                        ),
                    )

                split(width, p.row_tile, chunk)

            builder.branch(needed, stage, lambda: None)

        builder.loop(0, 3, 1, row)

        # Keep the old complete 3-row rectangle and clipped vertical halo.
        # The selected native pool declaration need not authorize padding.
        lower(
            "store_spatial_max",
            dict(
                output_row=op("add", op("mul", op("add", op("mul", image, oh), pool_y), ow), output_x),
                output_column=column,
                columns=count,
                output_pixels=output_pixels,
                accumulator_row=op("add", base, op("mul", op("sub", original_first, epoch_start), width)),
                local_height=3,
                local_width=width,
                window_height=3,
                window_width=3,
                stride_width=2,
                upad=0,
                readout_tile=plane,
            ),
        )

    split(p.output_channels, p.column_tile, channels)
