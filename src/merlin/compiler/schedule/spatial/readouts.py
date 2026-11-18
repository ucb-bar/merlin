"""Exact rounded readout composition for spatial maximum proposals.

The two source quantizers remain distinct. A replacement is admitted only when
the complete signed-i32 transition partition agrees, including saturation and
both rounding boundaries. This is numerical evidence, never target admission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from merlin.compiler.lower.quant.readout import derive, evaluate
from merlin.compiler.lower.quant.requantization import quantized
from merlin.compiler.schedule.pipeline.program_limits import AddressWindow

if TYPE_CHECKING:
    from merlin.compiler.schedule.spatial.producer import PackedSpatialProducerPlan
from merlin.compiler.schedule.spatial.rolling_window import (
    emit_rolling_window,
    materialize_rolling_window,
    propose_rolling_window,
)

# ---- spatial_readout ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundedReadout:
    scales: tuple[float, ...]
    relu: bool = False

    def __post_init__(self):
        if type(self.scales) is not tuple or type(self.relu) is not bool:
            raise ValueError("readout requires ordered scale constants and explicit activation")
        proof = derive(self.scales, relu=self.relu)
        object.__setattr__(self, "scales", tuple(proof["source_scales"]))


class ReadoutCompositionUnavailable(ValueError):
    """The proposed replacement changes an exact source value."""

    def __init__(self, witness):
        self.witness = witness
        super().__init__(
            f"rounded readout composition differs at accumulator {witness['accumulator']}: "
            f"source={witness['source']}, replacement={witness['replacement']}"
        )


@dataclass(frozen=True)
class SpatialReadoutChain:
    # None means the source producer really exposes raw i32 accumulators.
    producer: RoundedReadout | None
    pool: RoundedReadout
    replacement: RoundedReadout | None

    def __post_init__(self):
        if (
            self.producer is not None
            and type(self.producer) is not RoundedReadout
            or type(self.pool) is not RoundedReadout
            or self.replacement is not None
            and type(self.replacement) is not RoundedReadout
        ):
            raise ValueError("spatial readout requires exact typed source and proposed replacement stages")
        for stage in (self.producer, self.pool, self.replacement):
            if stage is not None:
                stage.__post_init__()

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def prove_spatial_readout(chain: SpatialReadoutChain):
    """Prove both quantizers against one proposed hardware readout on all i32.

    The first quantizer and replacement are constant between their complete
    monotone i8 transition points. Applying the second quantizer to that first
    i8 cannot add a transition elsewhere. Checking the union and immediate
    predecessors therefore covers every signed-i32 input, without samples.
    A positive readout is monotone, so exact maximum can commute with it; the
    caller still owns padding, source matching, observers and target semantics.
    """
    if type(chain) is not SpatialReadoutChain:
        raise ValueError("exact spatial readout chain required")
    chain.__post_init__()
    if chain.replacement is None:
        raise ValueError("single-readout proof requires an explicit proposed replacement")
    source_stage = chain.producer or chain.pool
    source = derive(source_stage.scales, relu=source_stage.relu)
    replacement = derive(chain.replacement.scales, relu=chain.replacement.relu)
    low, high = -(1 << 31), (1 << 31) - 1
    points = {low, high}
    for proof in (source, replacement):
        for transition in proof["thresholds"]:
            points.update(x for x in (transition - 1, transition) if low <= x <= high)
    for value in sorted(points):
        first = evaluate(value, source)
        expected = quantized(first, chain.pool.scales, chain.pool.relu) if chain.producer is not None else first
        actual = evaluate(value, replacement)
        if expected != actual:
            raise ReadoutCompositionUnavailable(
                dict(
                    accumulator=value,
                    intermediate=first,
                    source=expected,
                    replacement=actual,
                    chain_sha256=chain.identity,
                )
            )
    second_identity = all(quantized(value, chain.pool.scales, chain.pool.relu) == value for value in range(-128, 128))
    return dict(
        schema="merlin.spatial_readout_composition.v1",
        chain_sha256=chain.identity,
        domain=[low, high],
        checked_partition_endpoints=len(points),
        exact=True,
        producer_raw=chain.producer is None,
        second_readout_identity=second_identity,
        source_scales=None if chain.producer is None else list(chain.producer.scales),
        pool_scales=list(chain.pool.scales),
        replacement_scales=list(chain.replacement.scales),
        proof="complete union of source and replacement monotone transition partitions",
        scope="numerical equivalence only; no source, placement, ABI or execution authority",
    )


# ---- rolling_spatial ---------------------------------------------------------------------------
# Bounded rolling input rows and Q1 rows feeding the original spatial max.
#
# Original readouts, source observers, dense products and target completion retain
# their separate contracts. No full input/output image or quantization collapse
# is introduced by this counted transport and consumer order.


@dataclass(frozen=True)
class RollingPackedSpatialPlan:
    """Place circular input rows after the existing private Q1 ring and panel.

    The original packed workspace must explicitly cover the additional cache;
    caller-owned host capacity and target accumulator capacity are independent.
    Q1 ring row width is bounded by the original selected output stripe.
    """

    packed: PackedSpatialProducerPlan

    def __post_init__(self):
        from merlin.compiler.schedule.spatial.producer import PackedSpatialProducerPlan

        if type(self.packed) is not PackedSpatialProducerPlan:
            raise ValueError("rolling spatial requires its complete original packed source plan")
        self.packed.__post_init__()
        if self.required_bytes > self.packed.workspace.size:
            raise ValueError("rolling spatial rows exceed the assigned private workspace capacity")
        self.rolling.__post_init__()

    @property
    def cache_start(self):
        p = self.packed
        alignment = p.panel_alignment
        end = p.workspace.start + p.required_bytes
        return (end + alignment - 1) // alignment * alignment

    @property
    def cache_bytes(self):
        p = self.packed
        w = p.source.producer.window
        pitch = (w.input_shape[2] * w.input_shape[3] + p.panel_alignment - 1) // p.panel_alignment * p.panel_alignment
        return ((w.kernel[0] - 1) * w.dilation[0] + 1) * pitch

    @property
    def required_bytes(self):
        return self.cache_start + self.cache_bytes - self.packed.workspace.start

    @property
    def rolling(self):
        p = self.packed
        return propose_rolling_window(
            p.source.producer.window,
            AddressWindow(p.workspace.space, self.cache_start, self.cache_bytes),
            source_identity=p.source.identity,
            alignment=p.panel_alignment,
        )


@dataclass(frozen=True)
class RollingSpatialChoice:
    """Inspectable source/resource costs, never a target latency prediction."""

    plan: RollingPackedSpatialPlan
    dense_invocations: int
    logical_macs: int
    primitive_mac_extent: int
    staged_input_bytes: int


def propose_rolling_spatial_choices(source, workspace, *, panel_rows_limit, output_columns_limit, alignment):
    """Choose from capacity-derived stripe families without changing source math.

    Output columns are bounded by the original accumulator assignment. Each
    family uses its largest legal complete-K panel. Tail and alignment costs
    are explicit; no budget increase, target ID, latency guess or fallback is
    introduced. A caller can rank these alternatives with its selected public
    resource/roofline model or retain all for ordinary compiler experiments.
    """
    from merlin.compiler.schedule.spatial.producer import PackedSpatialProducerPlan, SpatialProducerPlan

    if type(source) is not SpatialProducerPlan or type(workspace) is not AddressWindow:
        raise ValueError("rolling choices require original source and assigned host workspace")
    source.__post_init__()
    workspace.__post_init__()
    if (
        any(type(v) is not int or not 0 < v < 1 << 63 for v in (panel_rows_limit, output_columns_limit, alignment))
        or alignment & (alignment - 1)
        or workspace.start % alignment
    ):
        raise ValueError("rolling choices require positive explicit limits and aligned assigned storage")
    p, s = source.producer, source.spatial
    w = p.window
    batch, _, input_width, channels = w.input_shape
    height, _ = w.output_spatial
    _, output_width = s.output_hw
    reduction = w.kernel[0] * w.kernel[1] * channels
    cache_rows = (w.kernel[0] - 1) * w.dilation[0] + 1
    pitch = (input_width * channels + alignment - 1) // alignment * alignment
    cache_bytes = cache_rows * pitch
    # The cache start must be aligned. Rounding down the last possible start
    # gives an exact closed capacity bound for every complete input panel.
    last_panel_end = (workspace.start + workspace.size - cache_bytes) // alignment * alignment
    choices = []
    for columns in range(1, min(source.output_tile_width, output_columns_limit) + 1):
        selected = replace(source, output_tile_width=columns)
        panel_start = (
            (workspace.start + selected.temporary_rows * p.output_channels + alignment - 1) // alignment * alignment
        )
        rows = min(panel_rows_limit, selected.temporary_rows, (last_panel_end - panel_start) // reduction)
        if rows < 1:
            continue
        packed = PackedSpatialProducerPlan(selected, rows, workspace, alignment)
        rolling = RollingPackedSpatialPlan(packed)
        packed = replace(packed, workspace=replace(workspace, size=rolling.required_bytes))
        rolling = RollingPackedSpatialPlan(packed)
        whole, tail = divmod(output_width, columns)
        invocations = logical_rows = primitive_rows = stripes = 0
        for count, pixels in ((whole, columns), (int(bool(tail)), tail)):
            if not count:
                continue
            width = (pixels - 1) * s.stride[1] + s.window[1]
            panels, remainder = divmod(width, rows)
            invocations += count * (panels + bool(remainder))
            logical_rows += count * width
            primitive_rows += count * (
                panels * ((rows + p.primitive - 1) // p.primitive * p.primitive)
                + ((remainder + p.primitive - 1) // p.primitive * p.primitive)
            )
            stripes += count
        extent = (
            (reduction + p.primitive - 1)
            // p.primitive
            * p.primitive
            * ((p.output_channels + p.primitive - 1) // p.primitive * p.primitive)
        )
        stages = cache_rows + (height - 1) * min(cache_rows, w.stride[0])
        choices.append(
            RollingSpatialChoice(
                rolling,
                batch * height * invocations,
                batch * height * logical_rows * reduction * p.output_channels,
                batch * height * primitive_rows * extent,
                batch * stripes * stages * rolling.rolling.row_bytes,
            )
        )
    if not choices:
        raise ValueError("no complete original dense panel and rolling rows fit assigned host storage")
    return tuple(choices)


def emit_rolling_spatial_producer(plan, builder, lower, *, synchronous_completion, pool_consumer=None):
    """Consume actual rolling SCF and emit complete original conv/Q1/Q2/pool.

    ``lower`` binds typed begin/stage/await and consumer_issued/complete, plus
    pack_cached_input_panel/produce_packed_panel/reload_intermediate/
    store_spatial_max. Every dense producer and spatial store must complete
    synchronously; complete must finish *all* remaining uses before frontier
    transport can overwrite either input or Q1 rows. An optional pool_consumer
    replaces only that readout, with the original image/pool_y/stripe indices;
    its own source, storage and completion obligations remain explicit. No
    target law is inferred.
    """
    if type(plan) is not RollingPackedSpatialPlan or synchronous_completion is not True:
        raise ValueError("rolling spatial requires original source plan and synchronous effect binding")
    plan.__post_init__()
    if not callable(lower) or any(
        not callable(getattr(builder, n, None)) for n in ("op", "cmp", "select", "branch", "loop")
    ):
        raise ValueError("rolling spatial requires ordinary counted control and actual effect callbacks")
    if pool_consumer is not None and not callable(pool_consumer):
        raise ValueError("explicit spatial consumer callback must be callable")
    packed = plan.packed
    source = packed.source
    p, s = source.producer, source.spatial
    height, _ = s.input_hw
    oh, ow = s.output_hw
    kh, kw = s.window
    sh, sw = s.stride
    pt, _, _, _ = s.padding
    op = builder.op
    receipts = []

    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            builder.loop(0, whole, 1, lambda i: body(op("mul", i, tile), tile))
        if tail:
            body(whole * tile, tail)

    def stripe(output_x, output_columns):
        width = (output_columns - 1) * sw + kw

        def consumer(image, y):
            # Every original producer row is computed for this one horizontal
            # stripe. Border X duplication follows the unchanged min-pad halo
            # contract; no producer-Y duplication or full Q1 image is required.
            def panel(first, rows):
                fields = dict(
                    image=image,
                    producer_y=y,
                    output_x=output_x,
                    output_pixels=output_columns,
                    first_halo_column=first,
                    rows=rows,
                    reduction=packed.reduction,
                    columns=p.output_channels,
                    panel_offset=packed.panel_offset,
                    stripe_offset=op(
                        "add",
                        packed.workspace.start,
                        op("mul", op("add", op("mul", op("urem", y, kh), width), first), p.output_channels),
                    ),
                    stripe_row_stride=p.output_channels,
                )
                lower("pack_cached_input_panel", dict(fields))
                lower("produce_packed_panel", dict(fields))

            split(width, packed.panel_rows, panel)

            def pool(pool_y):
                if pool_consumer is not None:
                    pool_consumer(image=image, pool_y=pool_y, output_x=output_x, output_pixels=output_columns)
                    return

                def channels(column, count):
                    def row(ky):
                        original_y = op("add", op("sub", op("mul", pool_y, sh), pt), ky)
                        clipped = builder.select(builder.cmp("slt", original_y, 0), 0, original_y)
                        clipped = builder.select(builder.cmp("slt", clipped, height), clipped, height - 1)

                        def reload(first, rows):
                            lower(
                                "reload_intermediate",
                                dict(
                                    accumulator_row=op(
                                        "add", p.accumulator_row, op("add", op("mul", ky, width), first)
                                    ),
                                    scratch_offset=op(
                                        "add",
                                        packed.workspace.start,
                                        op(
                                            "add",
                                            op(
                                                "mul",
                                                op("add", op("mul", op("urem", clipped, kh), width), first),
                                                p.output_channels,
                                            ),
                                            column,
                                        ),
                                    ),
                                    scratch_row_stride=p.output_channels,
                                    rows=rows,
                                    columns=count,
                                    stripe_rows=kh * width,
                                    output_column=column,
                                    readout_tile=op("udiv", column, p.column_tile),
                                ),
                            )

                        split(width, p.row_tile, reload)

                    builder.loop(0, kh, 1, row)
                    lower(
                        "store_spatial_max",
                        dict(
                            output_row=op("add", op("mul", op("add", op("mul", image, oh), pool_y), ow), output_x),
                            output_column=column,
                            columns=count,
                            output_pixels=output_columns,
                            accumulator_row=p.accumulator_row,
                            local_height=kh,
                            local_width=width,
                            window_height=kh,
                            window_width=kw,
                            stride_width=sw,
                            readout_tile=op("udiv", column, p.column_tile),
                        ),
                    )

                split(p.output_channels, p.column_tile, channels)

            def ordinary_row():
                numerator = op("sub", op("add", y, pt), kh - 1)
                ready = op(
                    "and", builder.cmp("ult", numerator, oh * sh), builder.cmp("eq", op("urem", numerator, sh), 0), "i1"
                )
                builder.branch(ready, lambda: pool(op("udiv", numerator, sh)), lambda: None)

            # Only the final source row can satisfy multiple bottom-clamped
            # windows. This exact static interval covers each remaining pool
            # output once, including a last window whose end is exactly H-1.
            first_final = max(0, (height - 1 + pt - (kh - 1) + sh - 1) // sh)
            builder.branch(
                builder.cmp("eq", y, height - 1),
                lambda: builder.loop(first_final, oh, 1, pool) if first_final < oh else None,
                ordinary_row,
            )

        def effect(kind, fields):
            if kind == "consume":
                consumer(fields["image"], fields["output_y"])
                return lower("consumer_issued", fields)
            return lower(kind, fields)

        receipts.append(emit_rolling_window(materialize_rolling_window(plan.rolling), plan.rolling, builder, effect))

    split(ow, source.output_tile_width, stripe)
    plan.__post_init__()
    return dict(
        schema="merlin.rolling_spatial_producer_emission.v1",
        original_plan_sha256=source.identity,
        packed_plan_sha256=packed.identity,
        original_readout_chains=[asdict(c) for c in source.readouts],
        private_workspace_bytes=plan.required_bytes,
        private_q1_ring_bytes=packed.stripe_bytes,
        private_input_row_cache_bytes=plan.cache_bytes,
        full_intermediate_image_bytes=0,
        output_elements=p.window.input_shape[0] * oh * ow * p.output_channels,
        rolling_ir=receipts,
        completion=(
            "all original Q1 stores and Q2/pool readouts before explicit consumer complete; no asynchronous grant"
        ),
        target_qualification=False,
    )


# ---- spatial_readout_cache ---------------------------------------------------------------------
# Bounded source-exact retention of quantized spatial rows across readouts.
#
# This is a counted storage schedule, not a new arithmetic or asynchronous law.
# A completed Q1 i8 row is reloaded unchanged into a disjoint caller-owned i32
# row window. Original clipped border copies are retained too; native padding
# is not required. Consecutive max readouts retain overlapping rows until an explicit
# linear epoch ends. Original Q1 stores, Q2 conversion and signed-min padding
# remain separate. Target operand decoding and completion are adapter obligations.


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
