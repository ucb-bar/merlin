"""Realize the selected original convolution schedule and checked effects."""

from __future__ import annotations

from dataclasses import asdict

from merlin.compiler.schedule.affine_gather_runs import ConvolutionGatherRuns
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsResidency as ConvolutionRhsResidency
from merlin.compiler.schedule.convolution_choices import ConvolutionSchedule as ConvolutionSchedule
from merlin.compiler.schedule.convolution_choices import _law_digest as _law_digest
from merlin.compiler.schedule.window_transfer import (
    ConvolutionOutputTile,
    ConvolutionResidentInputRows,
    emit_convolution_column_block,
    emit_convolution_compute_epoch_column_block,
    emit_convolution_load_issue_column_block,
    emit_convolution_load_issue_tile,
    emit_convolution_tile,
    plan_convolution_readout_transfers,
)

from .convolution_admission import (
    _phase_blocks,
    _verify_batch_input_prefetch,
    _verify_grouped_transport,
    _verify_input_first_use,
    _verify_input_prefetch,
    _verify_readout_law,
    _verify_readout_transfers,
    _verify_spatial_output_group,
)
from .convolution_rhs import _prepare_convolution_rhs


def _emit_convolution_schedule(
    schedule,
    builder,
    lower,
    *,
    complete=None,
    synchronous_completion=False,
    rhs=None,
    spatial_iteration="rows",
    input_rows=None,
    readout_law=None,
    readout_transfer_law=None,
    output_group=None,
    array_law=None,
    ready_readout=False,
    grouped_transport=False,
    rhs_frontier=None,
    rhs_frontier_law=None,
    rhs_prefetched=False,
    input_prefetch_law=None,
    batch_input_prefetch_law=None,
    input_first_use_law=None,
    reused_input=None,
):
    """Emit the complete counted original invocation, consuming actual lifetimes.

    Scalar callbacks own actual selected instruction encodings and readouts.
    Synchronous mode retains the original completion assertion. Issued mode needs
    a real complete(kind, fields) callback. The default drains after every
    compute/store; the stronger opt-in epoch law permits runtime-bounded compute
    groups and retains full completion after every original store. An optional
    separate ``readout_law`` orders the final producer and original stores by
    selected dependencies, with a real full drain at each column block's end.
    No host payload packing, alias permission or implicit event is added. Explicit ``linear``
    spatial iteration coalesces valid unpadded row/image tails through exact
    affine gather fragments; default ``rows`` preserves original realization.
    """
    if type(schedule) is not ConvolutionSchedule:
        raise ValueError("convolution emission requires its original checked schedule")
    schedule.verify_current()
    if (
        not callable(lower)
        or any(not callable(getattr(builder, n, None)) for n in ("op", "cmp", "loop", "branch"))
        or (schedule.choice.load_issue and not callable(complete))
        or (not schedule.choice.load_issue and synchronous_completion is not True)
    ):
        raise ValueError("convolution emission lacks original target callbacks or real completion")
    if spatial_iteration not in ("rows", "linear"):
        raise ValueError("convolution spatial iteration must be rows or linear")
    if type(ready_readout) is not bool or ready_readout and output_group is None:
        raise ValueError("ready readout requires an explicit Boolean and checked spatial group")
    p = schedule.source.plan
    blocks = _phase_blocks(schedule, None if rhs is None else rhs.column_intervals)
    _verify_readout_law(schedule, readout_law, blocks)
    _verify_readout_transfers(schedule, readout_transfer_law, readout_law, blocks)
    readout_identity = _law_digest(readout_law)
    transfer_identity = _law_digest(readout_transfer_law)
    batch = p.window.input_shape[0]
    oh, ow = p.window.output_spatial
    gather_runs = None
    if spatial_iteration == "linear":
        if any(p.window.padding) or not callable(getattr(builder, "select", None)):
            raise ValueError("linear convolution iteration requires a valid unpadded window and select builder")
        gather_runs = ConvolutionGatherRuns(
            p.window.input_shape, (batch, oh, ow), p.window.stride, p.window.dilation, p.window.kernel, p.row_tile
        )
    if input_rows is not None:
        if (
            type(input_rows) is not ConvolutionResidentInputRows
            or input_rows.window != p.window
            or input_rows.source_identity != schedule.source.identity
            or spatial_iteration != "rows"
            or input_rows.end > p.operand_rows
            or not callable(complete)
        ):
            raise ValueError("resident input requires exact source, original row domains and real completion")
        input_rows.revalidate()
        if schedule.choice.load_issue and input_rows.initial_load_commands > schedule.issue_law.load_issue_budget:
            raise ValueError("resident input transport exceeds the selected issued load budget")
        if rhs is not None and not input_rows.disjoint_from(rhs.layout.rhs_base, rhs.layout.rhs_rows):
            raise ValueError("resident input rows overlap the assigned complete RHS")
    group_identity, array_identity = _verify_spatial_output_group(
        schedule,
        builder,
        lower,
        complete,
        rhs=rhs,
        input_rows=input_rows,
        spatial_iteration=spatial_iteration,
        readout_law=readout_law,
        readout_transfer_law=readout_transfer_law,
        output_group=output_group,
        array_law=array_law,
        blocks=blocks,
    )
    if grouped_transport:
        _verify_grouped_transport(schedule, input_rows, output_group)
    _verify_input_prefetch(schedule, input_rows, output_group, blocks, grouped_transport, input_prefetch_law)
    _verify_input_first_use(schedule, input_rows, output_group, blocks, grouped_transport, input_first_use_law)
    first_use_identity = _law_digest(input_first_use_law)
    _verify_batch_input_prefetch(
        schedule, input_rows, output_group, blocks, grouped_transport, batch_input_prefetch_law
    )
    prefetch_identity = _law_digest(input_prefetch_law)
    batch_prefetch_identity = _law_digest(batch_input_prefetch_law)
    before = schedule.identity
    rhs_identity = None
    rhs_issue = lower

    def rhs_check():
        return None

    if rhs is not None:
        if type(rhs) is not ConvolutionRhsResidency or rhs.schedule.identity != before or not callable(complete):
            raise ValueError("convolution RHS realization needs its original schedule and real preparation completion")
        rhs.verify_current()
        rhs_identity = rhs.identity
        lower, rhs_check = _prepare_convolution_rhs(rhs, builder, lower, complete, already_prepared=rhs_prefetched)

    def tile_body(row, rows):
        for block in blocks:
            args = dict(output_row=row, rows=rows)
            if input_rows is not None:
                args["input_rows"] = input_rows
            if gather_runs is not None:
                args["gather_runs"] = gather_runs
            if schedule.choice.compute_pairs > 1:
                emit_convolution_compute_epoch_column_block(
                    p,
                    builder,
                    lower,
                    complete,
                    **args,
                    output_tiles=block,
                    law=schedule.retention_law,
                    issue_law=schedule.issue_law,
                    epoch_law=schedule.epoch_law,
                    compute_pairs=schedule.choice.compute_pairs,
                    readout_law=readout_law,
                    readout_transfer_law=readout_transfer_law,
                )
            elif len(block) == 1:
                t = block[0]
                args.update(output_column=t.output_column, columns=t.columns, accumulator_row=t.accumulator_row)
                if schedule.choice.load_issue:
                    emit_convolution_load_issue_tile(p, builder, lower, complete, **args, law=schedule.issue_law)
                else:
                    emit_convolution_tile(p, builder, lower, **args, synchronous_completion=True)
            elif schedule.choice.load_issue:
                emit_convolution_load_issue_column_block(
                    p,
                    builder,
                    lower,
                    complete,
                    **args,
                    output_tiles=block,
                    law=schedule.retention_law,
                    issue_law=schedule.issue_law,
                )
            else:
                emit_convolution_column_block(
                    p,
                    builder,
                    lower,
                    **args,
                    output_tiles=block,
                    law=schedule.retention_law,
                    synchronous_completion=True,
                )

    rolling = None
    if input_rows is not None:
        from merlin.compiler.schedule.rolling_window import emit_rolling_window, materialize_rolling_window

        ring = input_rows.rolling_plan()

        def transport(kind, fields):
            # The handle is the real caller-assigned private cache base, not an
            # asynchronous ticket. Every ready/last-use edge emits a full drain.
            if kind == "stage_row":
                input_rows.emit_stage(builder, fields, lower)
            elif kind == "await_transport":
                complete("input_ready", dict(local_row=input_rows.base))
            elif kind == "advance_consumer":
                return fields["consumer"].advance(
                    fields["first_run"], fields["run_count"], after_run=fields.get("after_run")
                )
            elif kind == "begin_rhs_prefetch":
                if rhs_frontier is None or fields["rhs_row"] != rhs_frontier.rhs_base:
                    raise ValueError("RHS frontier changes the checked original phase allocation")
                return rhs_frontier.rhs_base
            elif kind == "stage_rhs":
                if rhs_frontier is None or fields["transfer"] != rhs_frontier.rhs_base:
                    raise ValueError("RHS frontier requires the actual assigned next-phase rows")
                rhs_issue(
                    "load_weight",
                    {k: fields[k] for k in ("reduction_row", "output_column", "local_row", "rows", "columns")},
                )
            elif kind == "finish_consumer":
                fields["consumer"].finish()
            elif kind in ("consume", "consume_group", "start_consumer"):
                split = kind == "start_consumer"
                if split and (output_group is None or len(blocks) != 1):
                    raise ValueError("split rolling consumer needs one checked live output-column block")

                def column(x, rows):
                    row = builder.op(
                        "add",
                        builder.op(
                            "mul", builder.op("add", builder.op("mul", fields["image"], oh), fields["output_y"]), ow
                        ),
                        x,
                    )
                    tile_body(row, rows)

                if output_group is None:
                    whole, tail = divmod(ow, p.row_tile)
                    builder.loop(0, whole, 1, lambda x: column(builder.op("mul", x, p.row_tile), p.row_tile))
                    if tail:
                        column(whole * p.row_tile, tail)
                elif output_group.output_rows > 1 or kind in ("consume_group", "start_consumer"):
                    from merlin.compiler.schedule.convolution_output_groups import (
                        ConvolutionRowFragmentGroup,
                        emit_rolling_convolution_output_group,
                    )

                    # Preserve original transport and every real ready/last-use
                    # drain. Delay products until this bounded group is staged;
                    # the source-row union is injective in the selected ring.
                    def rows_group(count):
                        first_y = (
                            fields["output_y"]
                            if kind in ("consume_group", "start_consumer")
                            else builder.op("sub", fields["output_y"], count - 1)
                        )
                        row = builder.op("mul", builder.op("add", builder.op("mul", fields["image"], oh), first_y), ow)
                        tiles = (
                            output_group.tiles(count)
                            if type(output_group) is ConvolutionRowFragmentGroup
                            else tuple(
                                (y * ow + x, min(p.row_tile, ow - x))
                                for y in range(count)
                                for x in range(0, ow, p.row_tile)
                            )
                        )
                        for block in blocks:
                            consumer = emit_rolling_convolution_output_group(
                                schedule,
                                output_group,
                                builder,
                                lower,
                                complete,
                                output_row=row,
                                row_counts=tuple(t[1] for t in tiles),
                                tile_origins=(
                                    None
                                    if output_group.output_rows == 1
                                    and type(output_group) is not ConvolutionRowFragmentGroup
                                    else tuple(t[0] for t in tiles)
                                ),
                                output_tiles=block,
                                input_rows=input_rows,
                                array_law=array_law,
                                readout_law=readout_law,
                                ready_readout=ready_readout,
                                readout_transfer_law=readout_transfer_law,
                                defer_consumer=split,
                            )
                            if split:
                                if consumer.runs != fields["consumer_runs"]:
                                    raise ValueError("split rolling consumer differs from original reduction extent")
                                return consumer

                    if kind in ("consume_group", "start_consumer"):
                        consumer = rows_group(fields["output_rows"])
                        if split:
                            return consumer
                        return input_rows.base
                    rows = output_group.output_rows
                    tail = oh % rows
                    boundary = builder.cmp("eq", builder.op("urem", builder.op("add", fields["output_y"], 1), rows), 0)

                    def last_rows():
                        if tail:
                            builder.branch(
                                builder.cmp("eq", fields["output_y"], oh - 1), lambda: rows_group(tail), lambda: None
                            )

                    builder.branch(boundary, lambda: rows_group(rows), last_rows)
                else:
                    from merlin.compiler.schedule.convolution_output_groups import emit_rolling_convolution_output_group

                    def grouped(x, counts):
                        row = builder.op(
                            "add",
                            builder.op(
                                "mul", builder.op("add", builder.op("mul", fields["image"], oh), fields["output_y"]), ow
                            ),
                            x,
                        )
                        for block in blocks:
                            emit_rolling_convolution_output_group(
                                schedule,
                                output_group,
                                builder,
                                lower,
                                complete,
                                output_row=row,
                                row_counts=counts,
                                output_tiles=block,
                                input_rows=input_rows,
                                array_law=array_law,
                                readout_law=readout_law,
                                ready_readout=ready_readout,
                                readout_transfer_law=readout_transfer_law,
                            )

                    width = p.row_tile * output_group.row_blocks
                    whole, tail = divmod(ow, width)
                    if whole:
                        builder.loop(
                            0,
                            whole,
                            1,
                            lambda x: grouped(builder.op("mul", x, width), (p.row_tile,) * output_group.row_blocks),
                        )
                    if tail:
                        full, last = divmod(tail, p.row_tile)
                        grouped(whole * width, (p.row_tile,) * full + ((last,) if last else ()))
            elif kind == "complete":
                complete("input_last_use", dict(local_row=input_rows.base))
            elif kind != "begin_transport":
                raise ValueError("unknown original rolling input effect")
            return input_rows.base

        if reused_input is not None:
            from merlin.compiler.schedule.convolution_input_reuse import InvariantConvolutionInput

            if type(reused_input) is not InvariantConvolutionInput:
                raise ValueError("completed input reuse needs its original invocation lifetime owner")
            rolling = reused_input.emit_reused_group(
                rhs,
                output_group,
                builder,
                lower,
                complete,
                blocks=blocks,
                array_law=array_law,
                readout_law=readout_law,
                readout_transfer_law=readout_transfer_law,
                ready_readout=ready_readout,
                rhs_frontier=rhs_frontier,
                rhs_frontier_law=rhs_frontier_law,
                rhs_issue=rhs_issue,
            )
        elif input_first_use_law is not None:
            from merlin.compiler.schedule.convolution_input_first_use import emit_pending_input_images

            rolling = emit_pending_input_images(
                schedule,
                output_group,
                input_rows,
                input_first_use_law,
                builder,
                lower,
                complete,
                array_law=array_law,
                readout_law=readout_law,
                readout_transfer_law=readout_transfer_law,
                ready_readout=ready_readout,
                output_tiles=blocks[0],
                rhs_frontier=rhs_frontier,
                rhs_issue=rhs_issue,
            )
        elif rhs_frontier_law is not None or input_prefetch_law is not None or batch_input_prefetch_law is not None:
            from merlin.compiler.schedule.rolling_lookahead import (
                RollingBatchFrontier,
                RollingLookaheadAdmission,
                emit_lookahead_rolling_window,
                materialize_lookahead_rolling_window,
            )

            p = schedule.source.plan
            runs = (
                p.window.kernel[0]
                * p.window.kernel[1]
                * ((p.window.input_shape[-1] + p.channel_tile - 1) // p.channel_tile)
            )
            selected_frontier = batch_input_prefetch_law or rhs_frontier_law
            admission = (
                input_prefetch_law
                if input_prefetch_law is not None
                else RollingLookaheadAdmission(
                    selected_frontier.hardware_identity,
                    selected_frontier.qualification_identity,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                )
            )
            future_input = input_prefetch_law is not None
            batch_frontier = (
                None
                if batch_input_prefetch_law is None
                else RollingBatchFrontier(schedule.source.identity, p.channel_tile)
            )
            module = materialize_lookahead_rolling_window(
                ring,
                output_rows=output_group.output_rows,
                consumer_runs=runs,
                rhs_frontier=rhs_frontier,
                future_input=future_input,
                batch_frontier=batch_frontier,
            )
            rolling = emit_lookahead_rolling_window(
                module,
                ring,
                builder,
                transport,
                output_rows=output_group.output_rows,
                consumer_runs=runs,
                rhs_frontier=rhs_frontier,
                future_input=future_input,
                admission=admission,
                rhs_frontier_admission=rhs_frontier_law,
                batch_frontier=batch_frontier,
                batch_frontier_admission=batch_input_prefetch_law,
            )
        elif grouped_transport:
            from merlin.compiler.schedule.rolling_window import (
                emit_grouped_rolling_window,
                materialize_grouped_rolling_window,
            )

            rolling = emit_grouped_rolling_window(
                materialize_grouped_rolling_window(ring, output_rows=output_group.output_rows),
                ring,
                builder,
                transport,
                output_rows=output_group.output_rows,
            )
        else:
            rolling = emit_rolling_window(materialize_rolling_window(ring), ring, builder, transport)
        input_rows.revalidate()
    elif spatial_iteration == "linear":
        whole, tail = divmod(batch * oh * ow, p.row_tile)
        builder.loop(0, whole, 1, lambda index: tile_body(builder.op("mul", index, p.row_tile), p.row_tile))
        if tail:
            tile_body(whole * p.row_tile, tail)
    else:

        def image_body(image):
            def y_body(y):
                def x_body(x, rows):
                    row = builder.op(
                        "add", builder.op("mul", builder.op("add", builder.op("mul", image, oh), y), ow), x
                    )
                    tile_body(row, rows)

                whole, tail = divmod(ow, p.row_tile)
                builder.loop(0, whole, 1, lambda x: x_body(builder.op("mul", x, p.row_tile), p.row_tile))
                if tail:
                    x_body(whole * p.row_tile, tail)

            builder.loop(0, oh, 1, y_body)

        builder.loop(0, batch, 1, image_body)
    rhs_check()
    if (
        schedule.identity != before
        or _law_digest(readout_law) != readout_identity
        or _law_digest(readout_transfer_law) != transfer_identity
        or _law_digest(output_group) != group_identity
        or _law_digest(array_law) != array_identity
        or _law_digest(input_prefetch_law) != prefetch_identity
        or _law_digest(input_first_use_law) != first_use_identity
        or _law_digest(batch_input_prefetch_law) != batch_prefetch_identity
    ):
        raise ValueError("original convolution schedule changed during target realization")
    result = schedule.summary()
    if input_first_use_law is not None:
        result = dict(result, input_first_use_admission=asdict(input_first_use_law))
    if input_prefetch_law is not None:
        result = dict(result, input_prefetch_admission=asdict(input_prefetch_law))
    if batch_input_prefetch_law is not None:
        result = dict(result, batch_input_prefetch_admission=asdict(batch_input_prefetch_law))
    if output_group is not None:
        result = dict(
            result,
            spatial_output_group=dict(
                layout=asdict(output_group),
                group_sha256=group_identity,
                array_law_sha256=array_identity,
                iteration=(
                    "one output row; K,N,M; original row and channel tails"
                    if output_group.output_rows == 1
                    else "source-row groups; K,N,M; exact per-row tails and ring union"
                ),
                completion=(
                    "actual input RAW and real final group/last-use drains"
                    if input_first_use_law is not None
                    else "real final group drain and unchanged ring input-ready/last-use drains"
                ),
                selected_by="original source row extent and explicitly owned RHS/SP/ACC capacity",
            ),
        )
        if ready_readout:
            result["spatial_output_group"]["ready_readout"] = dict(
                applied=True,
                frontier="immediately after each exact last original K producer",
                ordering="selected ACC RAW/result visibility and ordered scalar configuration capture",
                storage="original disjoint ACC cells; no new window or reuse before final group drain",
            )
        if output_group.output_rows > 1:
            result["spatial_output_group"]["ring_read_horizon"] = dict(
                output_rows=output_group.output_rows,
                source_row_span=(p.window.kernel[0] - 1) * p.window.dilation[0]
                + 1
                + (output_group.output_rows - 1) * p.window.stride[0],
                slots=input_rows.slots,
                completion=(
                    "actual target RAW before each input reader; real group drain before image reuse"
                    if input_first_use_law is not None
                    else "group transports complete before products; group drain before the next source-row stage"
                    if grouped_transport
                    else "all original transport drains; group drain before the next source-row stage"
                ),
            )
        from merlin.compiler.schedule.convolution_output_groups import ConvolutionRowFragmentGroup

        if type(output_group) is ConvolutionRowFragmentGroup:
            tiles = output_group.tiles(output_group.output_rows)
            result["spatial_output_group"]["row_fragments"] = dict(
                mapping="existing affine gather; direct ring reads split at original output-row boundaries",
                physical_ACC_tiles=len(tiles),
                compute_fragments_per_K_N=sum(len(output_group.gather.fragments(first, rows)) for first, rows in tiles),
                extra_input_bytes=0,
                readout="one original contiguous interval per packed ACC tile",
                selection="explicit executable alternative; no inferred performance win",
            )
    if readout_law is not None:
        result = dict(
            result,
            readout_epoch=asdict(readout_law),
            completion=(
                "full drain after each checked spatial group before input/ACC/RHS reuse"
                if output_group is not None
                else "full drain after each original output block before input/ACC/RHS reuse"
            ),
        )
    if readout_transfer_law is not None:
        transfer_blocks = (
            blocks
            if output_group is None
            else tuple(
                tuple(
                    ConvolutionOutputTile(
                        t.output_column,
                        t.columns,
                        p.accumulator_row + output_group.accumulator_row(0, n, active_column_blocks=len(block)),
                    )
                    for n, t in enumerate(block)
                )
                for block in blocks
            )
        )
        result = dict(
            result,
            readout_transfers=dict(
                law_sha256=transfer_identity,
                blocks=[
                    [
                        [
                            dict(output_column=t.output_column, columns=t.columns, accumulator_row=t.accumulator_row)
                            for t in group
                        ]
                        for group in plan_convolution_readout_transfers(p, block, readout_transfer_law, rows=p.row_tile)
                    ]
                    for block in transfer_blocks
                ],
                original_stores_per_spatial_tile=sum(map(len, blocks)),
                physical_stores_per_spatial_tile=sum(
                    len(plan_convolution_readout_transfers(p, block, readout_transfer_law, rows=p.row_tile))
                    for block in transfer_blocks
                ),
                completion=(
                    "original real final block drain, before input/ACC/RHS ownership reuse"
                    if output_group is None
                    else "original real final spatial group drain, before input/ACC/RHS ownership reuse"
                ),
            ),
        )
    if input_rows is not None:
        result = dict(result, resident_input_rows=input_rows.record(), rolling_transport=rolling)
        if reused_input is not None:
            result["spatial_output_group"]["completion"] = (
                "completed invariant input view; unchanged real group/input-last-use/phase drains"
            )
            if "ring_read_horizon" in result["spatial_output_group"]:
                result["spatial_output_group"]["ring_read_horizon"]["completion"] = (
                    "same completed immutable full input union; no phase input overwrite"
                )
    if spatial_iteration == "linear":
        result = dict(
            result,
            spatial_iteration="linear",
            spatial_tiles=(batch * oh * ow + p.row_tile - 1) // p.row_tile,
            affine_gather="exact valid-window runs split at output row and image boundaries",
        )
    if rhs is not None:
        if rhs.identity != rhs_identity:
            raise ValueError("original convolution RHS placement changed during realization")
        if rhs.column_intervals is not None:
            result = dict(
                result,
                column_blocks=[[asdict(t) for t in block] for block in blocks],
                partial_original_column_intervals=rhs.column_intervals,
                whole_invocation=False,
            )
        return dict(result, rhs_residency=rhs.summary(), realization_sha256=rhs_identity)
    return result


# Preserve the original public lookup and pickle identities.
for _definition in (_emit_convolution_schedule,):
    _definition.__module__ = "merlin.compiler.schedule.convolution_exploration"
del _definition
