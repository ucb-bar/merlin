"""Bounded rolling input rows and Q1 rows feeding the original spatial max.

Original readouts, source observers, dense products and target completion retain
their separate contracts. No full input/output image or quantization collapse
is introduced by this counted transport and consumer order.
"""

from dataclasses import asdict, dataclass, replace

from .program_limits import AddressWindow
from .rolling_window import emit_rolling_window, materialize_rolling_window, propose_rolling_window
from .spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan


@dataclass(frozen=True)
class RollingPackedSpatialPlan:
    """Place circular input rows after the existing private Q1 ring and panel.

    The original packed workspace must explicitly cover the additional cache;
    caller-owned host capacity and target accumulator capacity are independent.
    Q1 ring row width is bounded by the original selected output stripe.
    """

    packed: PackedSpatialProducerPlan

    def __post_init__(self):
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
