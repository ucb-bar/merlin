"""Bound live spatial/output-channel tiles around a complete resident RHS panel."""

from dataclasses import dataclass

from .affine_gather_runs import ConvolutionGatherRuns, ConvolutionResidentPanel


class ConvolutionGroupContinuation:
    """Compiler continuation for one initialized original consumer.

    Each advance emits complete reduction panels, including every N/M pair.
    The cursor checks source coverage during compilation; generated code uses
    ordinary counted loops and the original target commands and completions.
    """

    def __init__(self, runs, advance, finish):
        if type(runs) is not int or not 0 < runs < 1 << 63 or not callable(advance) or not callable(finish):
            raise ValueError("consumer continuation requires its original positive run count and real emitters")
        self.runs, self._advance, self._finish = runs, advance, finish
        self.position, self.finished = 0, False

    def advance(self, first, count, *, after_run=None):
        """Emit complete original K/N/M runs, then an optional typed frontier.

        ``after_run(run, reduction_row, depth)`` is invoked after every run's
        complete fresh/stay traversal. Its target effects need their own law;
        callback return does not establish hardware completion.
        """
        if (self.finished or type(first) is not int or type(count) is not int
                or first != self.position or not 0 < count <= self.runs - first):
            raise ValueError("consumer continuation skips, repeats or exceeds original reduction runs")
        if after_run is None:
            self._advance(first, first + count)
        elif callable(after_run):
            self._advance(first, first + count, after_run)
        else:
            raise ValueError("consumer frontier requires an actual after-run emitter")
        self.position += count
        return self

    def finish(self):
        if self.finished or self.position != self.runs:
            raise ValueError("consumer continuation requires every original reduction before readout")
        result = self._finish()
        self.finished = True
        return result


@dataclass(frozen=True)
class ConvolutionOutputGroup:
    panel: ConvolutionResidentPanel
    row_blocks: int
    output_rows: int = 1

    def __post_init__(self):
        if type(self.panel) is not ConvolutionResidentPanel:
            raise ValueError("original complete resident panel required")
        self.panel.__post_init__()
        if (
            type(self.row_blocks) is not int
            or self.row_blocks <= 0
            or type(self.output_rows) is not int
            or not 0 < self.output_rows <= self.row_blocks
            or self.row_blocks * self.panel.tile > self.panel.rhs_base
            or self.row_blocks * self.panel.accumulator_rows > self.panel.accumulator_capacity_rows
        ):
            raise ValueError("live spatial/output group exceeds its disjoint operand or accumulator windows")

    def lhs_row(self, row_block):
        if type(row_block) is not int or not 0 <= row_block < self.row_blocks:
            raise ValueError("spatial tile exceeds its live output group")
        return row_block * self.panel.tile

    def accumulator_row(self, row_block, column_block, *, active_column_blocks=None):
        active = self.panel.column_blocks if active_column_blocks is None else active_column_blocks
        if (
            type(active) is not int
            or not 0 < active <= self.panel.column_blocks
            or type(column_block) is not int
            or not 0 <= column_block < active
        ):
            raise ValueError("column tile exceeds its live output group")
        self.lhs_row(row_block)
        return (row_block * active + column_block) * self.panel.tile


@dataclass(frozen=True)
class ConvolutionRowFragmentGroup(ConvolutionOutputGroup):
    """Packed ACC tiles whose direct ring reads split at source-row boundaries.

    This alternative stages no additional input bytes. A physical ACC tile may
    have several compute fragments, each retaining the original per-output K
    order. The original readout drains its complete contiguous output interval.
    """
    source_identity: str = ""
    gather: ConvolutionGatherRuns | None = None

    def __post_init__(self):
        super().__post_init__()
        if type(self.gather) is not ConvolutionGatherRuns or not self.source_identity:
            raise ValueError("row fragments require the original affine gather and source")
        self.gather.__post_init__()
        if (self.gather.max_rows != self.panel.tile
                or self.row_blocks != (self.output_rows*self.gather.output_spatial[2]+self.panel.tile-1)//self.panel.tile):
            raise ValueError("row fragments change their complete packed ACC footprint")

    def tiles(self, count):
        if type(count) is not int or not 0 < count <= self.output_rows:
            raise ValueError("row fragments exceed their original source-row horizon")
        pixels=count*self.gather.output_spatial[2]
        return tuple((first,min(self.panel.tile,pixels-first))for first in range(0,pixels,self.panel.tile))


def plan_rolling_convolution_row_fragments(schedule, rhs, input_rows, *, max_row_blocks=16, max_output_rows=1):
    """Explicit direct-ring alternative; choose only a capacity-safe row band.

    Reuse the existing original affine gather to split packed logical tiles.
    More fragments can cost more than saved readouts, so this is an executable
    alternative with no performance policy or automatic promotion. The initial
    family retains primitive-sized original row tiles and whole source bands.
    """
    from .convolution_exploration import ConvolutionRhsPhases
    if type(rhs) is ConvolutionRhsPhases:
        rhs.verify_current()
        groups=tuple(plan_rolling_convolution_row_fragments(schedule,phase,input_rows,
            max_row_blocks=max_row_blocks,max_output_rows=max_output_rows)for phase in rhs.phases)
        return groups if any(g is not None for g in groups)else None
    ordinary=plan_rolling_convolution_output_group(schedule,rhs,input_rows,
        max_row_blocks=max_row_blocks,max_output_rows=1)
    if ordinary is None:
        return None
    p=schedule.source.plan;w=p.window
    if p.row_tile != p.primitive:
        return None
    # Validate both caller bounds through the ordinary planner before using
    # the same actual SP/ACC allocation with a different logical tile map.
    plan_rolling_convolution_output_group(schedule,rhs,input_rows,
        max_row_blocks=max_row_blocks,max_output_rows=max_output_rows)
    oh,ow=w.output_spatial
    span=(w.kernel[0]-1)*w.dilation[0]+1
    horizon=min(max_output_rows,oh,1+(input_rows.slots-span)//w.stride[0])
    panel=ordinary.panel
    capacity=min(max_row_blocks,panel.rhs_base//panel.tile,
                 panel.accumulator_capacity_rows//panel.accumulator_rows)
    legal=[r for r in range(1,horizon+1)if r<=(r*ow+p.primitive-1)//p.primitive<=capacity]
    if not legal:
        return None
    rows=max(legal);blocks=(rows*ow+p.primitive-1)//p.primitive
    b,h,width,ci=w.input_shape;pt,pl,pb,pr=w.padding
    gather=ConvolutionGatherRuns((b,h+pt+pb,width+pl+pr,ci),(b,oh,ow),
        w.stride,w.dilation,w.kernel,p.primitive)
    return ConvolutionRowFragmentGroup(panel,blocks,rows,schedule.source.identity,gather)


def plan_convolution_output_group(panel, pixels, *, max_row_blocks=16):
    """Select a bounded M group without spilling full-K live accumulators.

    The static unroll bound is explicit and limits code expansion; capacity
    chooses the remaining extent. Profitability and target weight reuse remain
    separate choices. Each K epoch loads all LHS tiles before immutable reads;
    the caller must complete every read before the next epoch overwrites them.
    """
    if type(panel) is not ConvolutionResidentPanel:
        raise ValueError("original complete resident panel required")
    panel.__post_init__()
    if any(type(value) is not int or not 0 < value < 1 << 63 for value in (pixels, max_row_blocks)):
        raise ValueError("positive bounded spatial extent and static unroll limit required")
    blocks = min(
        (pixels + panel.tile - 1) // panel.tile,
        panel.rhs_base // panel.tile,
        panel.accumulator_capacity_rows // panel.accumulator_rows,
        max_row_blocks,
    )
    return ConvolutionOutputGroup(panel, blocks)


def plan_rolling_convolution_output_group(schedule, rhs, input_rows, *, max_row_blocks=16, max_output_rows=1):
    """Bind the existing group planner to exact invocation-resident operands.

    Decline nonprimitive row/channel partitions, missing rolling input, or a
    single spatial tile. A complete phase owner returns an aligned tuple of
    phase-local groups (or None for an unchanged phase); direct partial-phase
    execution remains refused by the invocation owner. No rows or source cells are added.
    The caller still selects independent fresh/stay and completion admissions.

    ``max_output_rows=1`` preserves the original one-row realization. Larger
    caller bounds admit only whole source rows fitting both private ACC and
    the existing ring: kernel_span + (rows-1)*stride_y <= actual slots. Pass
    this returned group to emit_convolution_schedule; supplying origins alone
    does not establish the transport horizon.
    """
    from .convolution_exploration import ConvolutionRhsPhases, ConvolutionRhsResidency, ConvolutionSchedule
    from .window_transfer import ConvolutionResidentInputRows

    if type(schedule) is not ConvolutionSchedule:
        raise ValueError("spatial grouping needs an original convolution schedule")
    schedule.verify_current()
    if type(max_output_rows) is not int or not 0 < max_output_rows <= 16:
        raise ValueError("spatial grouping needs an explicit output-row bound 1..16")
    if type(max_row_blocks) is not int or not 0 < max_row_blocks <= 16:
        raise ValueError("spatial grouping needs a bounded original unroll limit 1..16")
    if type(rhs) is ConvolutionRhsPhases:
        rhs.verify_current()
        if rhs.schedule.identity != schedule.identity or rhs.input_rows != input_rows:
            raise ValueError("spatial phases need their exact original input and source")
        groups = tuple(plan_rolling_convolution_output_group(schedule, phase, input_rows,
            max_row_blocks=max_row_blocks, max_output_rows=max_output_rows) for phase in rhs.phases)
        return groups if any(group is not None for group in groups) else None
    p = schedule.source.plan
    if (
        type(rhs) is not ConvolutionRhsResidency
        or type(input_rows) is not ConvolutionResidentInputRows
        or not schedule.choice.load_issue
        or schedule.choice.compute_pairs <= 1
        or not p.column_tile == p.channel_tile == p.primitive
        or not 0 < p.row_tile <= p.primitive
    ):
        return None
    rhs.verify_current()
    input_rows.revalidate()
    if (
        rhs.schedule.identity != schedule.identity
        or input_rows.window != p.window
        or input_rows.source_identity != schedule.source.identity
        or not input_rows.disjoint_from(rhs.layout.rhs_base,rhs.layout.rhs_rows)
        or input_rows.base < 0
    ):
        raise ValueError("spatial group differs from original disjoint rolling input/RHS")
    chunks = (p.window.input_shape[-1] + p.channel_tile - 1) // p.channel_tile
    count = len(schedule.source.column_intervals if rhs.column_intervals is None else rhs.column_intervals)
    panel = ConvolutionResidentPanel(
        p.primitive,
        chunks,
        p.window.kernel[0] * p.window.kernel[1] * chunks,
        count,
        rhs.layout.rhs_base,
        rhs.layout.rhs_rows,
        count * p.primitive,
        rhs.layout.operand_rows,
        p.accumulator_rows - p.accumulator_row,
    )
    oh, ow = p.window.output_spatial
    per_row = (ow + p.row_tile - 1) // p.row_tile
    # All original source rows in the delayed-read horizon must have distinct
    # modulo slots. No extra cache capacity or transport effect is inferred.
    span = (p.window.kernel[0] - 1) * p.window.dilation[0] + 1
    ring_rows = 1 + (input_rows.slots - span) // p.window.stride[0]
    capacity = min(max_row_blocks, panel.rhs_base // panel.tile,
                   panel.accumulator_capacity_rows // panel.accumulator_rows)
    rows = min(max_output_rows, oh, ring_rows, capacity // per_row)
    if rows > 1:
        return ConvolutionOutputGroup(panel, rows * per_row, rows)
    group = plan_convolution_output_group(panel, ow, max_row_blocks=max_row_blocks)
    return group if group.row_blocks > 1 else None


def emit_rolling_convolution_output_group(
    schedule,
    group,
    builder,
    issue,
    complete,
    *,
    output_row,
    row_counts,
    output_tiles,
    input_rows,
    array_law,
    readout_law,
    tile_origins=None,
    ready_readout=False,
    readout_transfer_law=None,
    defer_consumer=False,
    pending_input=None,
):
    """K,N,M original products with one fresh RHS then checked array stays.

    M tiles use distinct ACC cells; multirow origins require a live ring union.
    The rolling owner keeps the entire required source rows immutable until the
    real final drain. Every K/N pair retains its immediate PRELOAD/COMPUTE ABI;
    only explicit ``fresh_weights`` changes for later M. No encoder is supplied.
    ``defer_consumer`` initializes once and returns a checked compiler
    continuation. Each advance emits complete original reduction panels in
    order; finish emits the original readout and real group drain. The caller
    separately admits intervening effects and retains all input/ACC storage.
    A separately admitted readout block map may coalesce adjacent stores with
    identical numerical state. The actual group ACC slots, not the ungrouped
    source allocation, determine adjacency; all last-use drains remain real.
    """
    from dataclasses import asdict

    from .dense_array_retention import DenseArrayRetentionAdmission
    from .window_transfer import (
        ConvolutionOutputTile,
        ConvolutionReadoutEpochLaw,
        ConvolutionResidentInputRows,
        ConvolutionReadoutTransferLaw,
        _validate_convolution_load_issue,
        plan_convolution_readout_transfers,
    )

    schedule.verify_current()
    if type(ready_readout) is not bool:
        raise ValueError("ready readout requires an explicit Boolean")
    if type(defer_consumer) is not bool:
        raise ValueError("consumer continuation selection requires an explicit Boolean")
    if type(group) not in (ConvolutionOutputGroup,ConvolutionRowFragmentGroup):
        raise ValueError("spatial output grouping requires its checked original planner")
    group.__post_init__()
    p = schedule.source.plan
    epoch = schedule.epoch_law
    if (
        not schedule.choice.load_issue
        or schedule.choice.compute_pairs <= 1
        or type(input_rows) is not ConvolutionResidentInputRows
        or input_rows.source_identity != schedule.source.identity
        or input_rows.window != p.window
        or not input_rows.disjoint_from(group.panel.rhs_base,group.panel.rhs_rows)
        or group.panel.tile != p.primitive
        or group.panel.accumulator_capacity_rows != p.accumulator_rows - p.accumulator_row
        or not p.column_tile == p.channel_tile == p.primitive
        or not 0 < p.row_tile <= p.primitive
        or schedule.retention_law is None
        or not schedule.retention_law.independent_output_rows
        or not schedule.retention_law.no_intermediate_observers
        or type(array_law) is not DenseArrayRetentionAdmission
        or type(readout_law) is not ConvolutionReadoutEpochLaw
        or type(row_counts) is not tuple
        or not row_counts
        or len(row_counts) > group.row_blocks
        or type(output_tiles) is not tuple
        or not output_tiles
        or any(type(r) is not int or not 0 < r <= p.row_tile for r in row_counts)
        or (group.output_rows == 1 and any(r != p.row_tile for r in row_counts[:-1]))
    ):
        raise ValueError("spatial group needs original rows, independent array and readout premises")
    input_rows.revalidate()
    ow = p.window.output_spatial[1]
    fragments=type(group)is ConvolutionRowFragmentGroup
    if fragments:
        count,partial=divmod(sum(row_counts),ow)
        span=(p.window.kernel[0]-1)*p.window.dilation[0]+1
        w=p.window;b,h,width,ci=w.input_shape;pt,pl,pb,pr=w.padding
        expected_gather=ConvolutionGatherRuns((b,h+pt+pb,width+pl+pr,ci),
            (b,*w.output_spatial),w.stride,w.dilation,w.kernel,p.primitive)
        if (group.source_identity != schedule.source.identity or partial
                or group.gather != expected_gather or p.row_tile != p.primitive
                or tuple(zip(tile_origins or (),row_counts)) != group.tiles(count)
                or span+(group.output_rows-1)*p.window.stride[0]>input_rows.slots):
            raise ValueError("row fragments require complete original bands and live ring rows")
        origins=tile_origins
    elif group.output_rows == 1:
        if tile_origins is not None or sum(row_counts) > ow:
            raise ValueError("one-row spatial group requires original contiguous tiles")
        origins = tuple(m * p.row_tile for m in range(len(row_counts)))
    else:
        per_row = (ow + p.row_tile - 1) // p.row_tile
        rows, partial = divmod(len(row_counts), per_row)
        expected = tuple((y * ow + x, min(p.row_tile, ow - x))
                         for y in range(rows) for x in range(0, ow, p.row_tile))
        span = (p.window.kernel[0] - 1) * p.window.dilation[0] + 1
        if (partial or not 0 < rows <= group.output_rows
            or type(tile_origins) is not tuple
            or tuple(zip(tile_origins, row_counts)) != expected
            or span + (group.output_rows - 1) * p.window.stride[0] > input_rows.slots):
            raise ValueError("multirow spatial group needs exact source origins and live ring rows")
        origins = tile_origins
    array_law.__post_init__()
    if (
        array_law.hardware_identity != epoch.hardware_identity
        or array_law.numerical_policy_identity != epoch.numerical_policy_identity
        or readout_law.hardware_identity != epoch.hardware_identity
        or readout_law.numerical_policy_identity != epoch.numerical_policy_identity
        or len(row_counts) * len(output_tiles) > readout_law.max_issued_readouts
    ):
        raise ValueError("spatial group differs from selected array/readout scope or issued budget")
    epoch.require(schedule.choice.compute_pairs)
    if readout_transfer_law is not None and (
        type(readout_transfer_law) is not ConvolutionReadoutTransferLaw
        or readout_transfer_law.hardware_identity != epoch.hardware_identity
        or readout_transfer_law.numerical_policy_identity != epoch.numerical_policy_identity
    ):
        raise ValueError("spatial wide readout requires the original selected block mapping")
    transfer_identity = None if readout_transfer_law is None else readout_transfer_law.identity
    before = schedule.identity, asdict(group), asdict(array_law), asdict(readout_law), input_rows.record()
    _validate_convolution_load_issue(
        p, builder, issue, complete, p.row_tile, schedule.issue_law, len(row_counts) * len(output_tiles) * int(p.bias)
    )
    outputs = []
    readouts = []
    coordinates = []
    for m, rows in enumerate(row_counts):
        tiles = tuple(
            ConvolutionOutputTile(
                t.output_column,
                t.columns,
                p.accumulator_row + group.accumulator_row(m, n, active_column_blocks=len(output_tiles)),
            )
            for n, t in enumerate(output_tiles)
        )
        readout_law.require(p, rows, tiles)
        # Use the actual per-M physical ACC slots, including primitive-sized
        # slots for short source rows. The ungrouped source map may differ.
        transfers = (tuple((tile,) for tile in tiles) if readout_transfer_law is None else
                     plan_convolution_readout_transfers(p, tiles, readout_transfer_law, rows=rows))
        row = builder.op("add", output_row, origins[m])
        oh, ow = p.window.output_spatial
        # Decode this source tile once, before K/N compute loops. The same
        # coordinates feed every original tap and stationary-weight consumer.
        if fragments:
            coordinates.append(tuple((used,length,
                (builder.op("urem",builder.op("udiv",builder.op("add",row,used),ow),oh),
                 builder.op("urem",builder.op("add",row,used),ow)))
                for _,used,length in group.gather.fragments(origins[m],rows)))
        else:
            coordinates.append(((0,rows,(builder.op("urem",builder.op("udiv",row,ow),oh),
                                         builder.op("urem",row,ow))),))
        outputs.append(
            tuple(
                dict(
                    output_row=row,
                    rows=rows,
                    output_column=t.output_column,
                    columns=t.columns,
                    accumulator_row=t.accumulator_row,
                )
                for t in tiles
            )
        )
        readouts.append(tuple(dict(output_row=row, rows=rows,
                                   output_column=transfer[0].output_column,
                                   accumulator_row=transfer[0].accumulator_row,
                                   columns=sum(t.columns for t in transfer))
                              for transfer in transfers))
    if pending_input is not None:
        from .convolution_input_first_use import PendingConvolutionInput
        if type(pending_input) is not PendingConvolutionInput:
            raise ValueError("first-use input requires its original typed pending owner")
        pending_input.start(schedule, group, input_rows, origins, row_counts, builder, issue, columns=len(output_tiles))
    if p.bias:
        for row in outputs:
            for out in row:
                issue("initialize_bias", out)
    ci = p.window.input_shape[-1]
    chunks = (ci + p.channel_tile - 1) // p.channel_tile
    kh, kw = p.window.kernel
    source_image = (builder.op("udiv",output_row,p.window.output_spatial[0]*p.window.output_spatial[1])
                    if getattr(input_rows,"image_windows",1)>1 else None)
    fragment_count=sum(map(len,coordinates))
    total = kh * kw * chunks * len(output_tiles) * fragment_count

    def panel(ky, kx, c, depth, after_run=None):
        reduction = builder.op(
            "add", builder.op("mul", builder.op("add", builder.op("mul", ky, kw), kx), ci), c
        )
        first = builder.op("and", builder.cmp("eq", ky, 0), builder.cmp("eq", kx, 0), "i1")
        first = builder.op("and", first, builder.cmp("eq", c, 0), "i1")
        chunk = builder.op(
            "add",
            builder.op("mul", builder.op("add", builder.op("mul", ky, kw), kx), chunks),
            builder.op("udiv", c, p.channel_tile),
        )
        if ready_readout:
            # This is the actual original last-K producer, not an
            # estimated readiness time or the end of an unrolled body.
            last = builder.op("and", builder.cmp("eq", ky, kh - 1),
                              builder.cmp("eq", kx, kw - 1), "i1")
            last = builder.op("and", last, builder.cmp("eq", c, ci - depth), "i1")
        for n, tile in enumerate(output_tiles):
            fragment_index=0
            for m, row in enumerate(outputs):
                for f,(used,length,coords)in enumerate(coordinates[m]):
                    original_out=row[n]
                    out=(dict(original_out,output_row=builder.op("add",original_out["output_row"],used),
                        accumulator_row=original_out["accumulator_row"]+used,rows=length)if fragments else original_out)
                    if pending_input is not None and f == 0:
                        pending_input.before_reader(builder, issue, ky, kx, c, m, n=n)
                    issue(
                        "load_weight",
                        dict(
                            reduction_row=reduction,
                            output_column=tile.output_column,
                            local_row=p.rhs_row,
                            rows=depth,
                            columns=tile.columns,
                        ),
                    )
                    fields = dict(
                        out,
                        lhs_row=input_rows.emit_row_coordinates(builder, *coords, ky, kx, c,
                            **({"image":source_image} if source_image is not None else {})),
                        rhs_row=p.rhs_row,
                        depth=depth,
                        first_reduction=first,
                        bias=p.bias,
                        fresh_weights=m == 0 and f == 0,
                    )
                    if input_rows.row_step != 1:
                        fields["lhs_row_step"] = input_rows.row_step
                    issue("compute", fields)
                    if ready_readout and f==len(coordinates[m])-1 and (readout_transfer_law is None or n == len(output_tiles) - 1):
                        # Separate RAW/result/config-capture/progress law
                        # orders this store. All later producers write
                        # independently checked, disjoint ACC cells.
                        stores = (original_out,) if readout_transfer_law is None else readouts[m]
                        def publish():
                            for store in stores:
                                issue("store", store)
                        builder.branch(last, publish, lambda: None)
                    pair = builder.op(
                        "add",
                        builder.op("mul", chunk, len(output_tiles) * fragment_count),
                        n * fragment_count + fragment_index + 1,
                    )
                    boundary = builder.cmp("eq", builder.op("urem", pair, schedule.choice.compute_pairs), 0)
                    builder.branch(
                        boundary,
                        lambda: builder.branch(
                            builder.cmp("eq", pair, total), lambda: None, lambda: complete("compute", fields)
                        ),
                        lambda: None,
                    )
                    fragment_index+=1
        if after_run is not None:
            after_run(chunk, reduction, depth)


    def kernel(ky):
        def column(kx):
            whole, tail = divmod(ci, p.channel_tile)
            if whole:
                builder.loop(0, whole, 1, lambda i: panel(ky, kx, builder.op("mul", i, p.channel_tile), p.channel_tile))
            if tail:
                panel(ky, kx, whole * p.channel_tile, tail)

        builder.loop(0, kw, 1, column)

    def interval(first, stop, after_run=None):
        # Peel only panel tails. Full channel panels stay in real counted loops;
        # every panel preserves the original N/M traversal and pair fences.
        while first < stop:
            tap, channel = divmod(first, chunks)
            end = min(stop, (tap + 1) * chunks)
            ky, kx = divmod(tap, kw)
            full_stop = min(end - tap * chunks, ci // p.channel_tile)
            if channel < full_stop:
                builder.loop(channel, full_stop, 1,
                    lambda i: panel(ky, kx, builder.op("mul", i, p.channel_tile), p.channel_tile, after_run))
            if end - tap * chunks > ci // p.channel_tile:
                panel(ky, kx, (ci // p.channel_tile) * p.channel_tile, ci % p.channel_tile, after_run)
            first = end

    def finish():
        if not ready_readout:
            for row in readouts:
                for out in row:
                    issue("store", out)
        completion_fields = dict(rows=sum(row_counts), output_row=output_row)
        if pending_input is None:
            complete("spatial_group_last_use", completion_fields)
        else:
            pending_input.finish(complete, completion_fields)
        if (before != (schedule.identity, asdict(group), asdict(array_law), asdict(readout_law), input_rows.record())
            or transfer_identity != (None if readout_transfer_law is None else readout_transfer_law.identity)):
            raise ValueError("spatial group source, private layout or selected laws changed during emission")
        return dict(
            row_counts=row_counts,
            original_columns=[t.output_column for t in output_tiles],
            accumulator_cells="(spatial_slot*active_column_slots+column_slot)*primitive+row; lane=column",
            fresh_weights="first spatial slot of each original K/column; later slots stay",
            completion="real final group drain before ACC reuse or rolling input last-use",
        )

    if defer_consumer:
        return ConvolutionGroupContinuation(kh * kw * chunks, interval, finish)
    builder.loop(0, kh, 1, kernel)
    return finish()
