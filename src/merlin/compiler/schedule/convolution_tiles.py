"""Emit original counted convolution tiles through checked transfer callbacks."""

from merlin.compiler.schedule.affine_gather_runs import ConvolutionGatherRuns as ConvolutionGatherRuns
from merlin.compiler.schedule.affine_gather_runs import emit_convolution_gather_runs as emit_convolution_gather_runs

from .convolution_input_rows import ConvolutionResidentInputRows as ConvolutionResidentInputRows
from .window_geometry import CountedConvolutionPlan as CountedConvolutionPlan


def emit_counted_convolution(plan: CountedConvolutionPlan, builder, lower, *, synchronous_completion: bool):
    """Emit the checked counted plan through actual target operation callbacks.

    ``builder`` supplies integer op/cmp, loop and branch, retaining counted loops
    in an IR implementation. ``lower(kind, fields)`` implements each effect and
    its synchronous completion. Interior rows use one affine input DMA; boundary
    tiles use checked one-row transfers or explicit zero-source transfers. No
    input payload is copied or read by this shared host control emitter.
    """
    if type(plan) is not CountedConvolutionPlan or synchronous_completion is not True:
        raise ValueError("counted convolution requires an explicit synchronous target realization")
    plan.__post_init__()
    identity = plan.identity
    if any(not callable(getattr(builder, n, None)) for n in ("op", "cmp", "loop", "branch")) or not callable(lower):
        raise ValueError("counted convolution needs an index/control builder and target callback")
    window = plan.window
    batch, _, _, _ = window.input_shape
    oh, ow = window.output_spatial

    def op(name, a, b):
        return builder.op(name, a, b)

    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            builder.loop(0, whole, 1, lambda i: body(op("mul", i, tile), tile))
        if tail:
            body(whole * tile, tail)

    def batch_body(image):
        def y_body(y):
            def x_body(x, rows):
                row = op("add", op("mul", op("add", op("mul", image, oh), y), ow), x)

                def n_body(column, columns):
                    emit_convolution_tile(
                        plan,
                        builder,
                        lower,
                        output_row=row,
                        output_column=column,
                        rows=rows,
                        columns=columns,
                        accumulator_row=plan.accumulator_row,
                        synchronous_completion=True,
                    )

                split(plan.output_channels, plan.column_tile, n_body)

            split(ow, plan.row_tile, x_body)

        builder.loop(0, oh, 1, y_body)

    builder.loop(0, batch, 1, batch_body)
    plan.__post_init__()
    if plan.identity != identity:
        raise ValueError("counted convolution source/resources changed during emission")
    return dict(
        plan_sha256=identity,
        **plan.coverage,
        scope="emitted source partitions; target effects and execution qualification remain separate",
    )


def emit_convolution_tile(
    plan,
    builder,
    lower,
    *,
    output_row,
    output_column,
    rows,
    columns,
    accumulator_row,
    synchronous_completion,
    gather_runs=None,
    input_rows=None,
):
    """Emit one original convolution rectangle into caller-assigned private rows.

    This is the same source-ordered reduction and checked input DMA predicate as
    the whole-image emitter. The caller proves the global row/column rectangle,
    complete invocation coverage, final observations and assigned-row lifetime.
    No output copy is implicit: the final ``store`` effect remains caller-owned.
    """
    if type(plan) is not CountedConvolutionPlan or synchronous_completion is not True:
        raise ValueError("convolution tile requires an exact plan and synchronous realization")
    return _emit_convolution_tile_effects(
        plan,
        builder,
        lower,
        output_row=output_row,
        output_column=output_column,
        rows=rows,
        columns=columns,
        accumulator_row=accumulator_row,
        gather_runs=gather_runs,
        input_rows=input_rows,
    )


def _emit_convolution_tile_effects(
    plan,
    builder,
    lower,
    *,
    output_row,
    output_column,
    rows,
    columns,
    accumulator_row,
    output_tiles=None,
    gather_runs=None,
    input_rows=None,
):
    """Canonical source partition; completion is owned by the public wrapper."""
    plan.__post_init__()
    if any(type(x) is not int or x <= 0 for x in (rows, columns)) or rows > plan.row_tile or columns > plan.column_tile:
        raise ValueError("convolution rectangle exceeds its declared primitive tile")
    window = plan.window
    _, _, _, ci = window.input_shape
    kh, kw = window.kernel
    if input_rows is not None:
        if (
            type(input_rows) is not ConvolutionResidentInputRows
            or input_rows.window != window
            or input_rows.channel_tile != plan.channel_tile
            or gather_runs is not None
            or input_rows.end > plan.operand_rows
            or not input_rows.disjoint_from(plan.rhs_row, plan.channel_tile)
        ):
            raise ValueError("resident input layout differs from original source or overlaps its RHS tile")
        input_rows.revalidate()
    if gather_runs is not None:
        expected = ConvolutionGatherRuns(
            window.input_shape,
            (window.input_shape[0], *window.output_spatial),
            window.stride,
            window.dilation,
            window.kernel,
            plan.row_tile,
        )
        if (
            type(gather_runs) is not ConvolutionGatherRuns
            or gather_runs != expected
            or any(window.padding)
            or not callable(getattr(builder, "select", None))
        ):
            raise ValueError("linear convolution gathers require the exact original valid window and select builder")

    def op(name, a, b):
        return builder.op(name, a, b)

    common = dict(
        output_row=output_row, output_column=output_column, rows=rows, columns=columns, accumulator_row=accumulator_row
    )
    outputs = (
        (common,)
        if output_tiles is None
        else tuple(
            dict(
                output_row=output_row,
                rows=rows,
                output_column=t.output_column,
                columns=t.columns,
                accumulator_row=t.accumulator_row,
            )
            for t in output_tiles
        )
    )
    if plan.bias:
        for output in outputs:
            lower("initialize_bias", output)

    def ky_body(ky):
        def kx_body(kx):
            def channel_body(channel, depth):
                def transfer(first, stride, local, count, zero):
                    lower(
                        "load_input",
                        dict(
                            first_element=first,
                            row_stride_elements=stride,
                            local_row=local,
                            rows=count,
                            columns=depth,
                            zero=zero,
                        ),
                    )

                def interior():
                    transfer(run["first_element"], run["row_stride_elements"], plan.lhs_row, rows, False)

                def boundary():
                    def one(index):
                        item = window.emit_row_run(builder, op("add", output_row, index), 1, ky, kx, channel, depth)
                        local = op("add", plan.lhs_row, index)
                        builder.branch(
                            item["predicate"],
                            lambda: transfer(item["first_element"], item["row_stride_elements"], local, 1, False),
                            lambda: transfer(0, 0, local, 1, True),
                        )

                    builder.loop(0, rows, 1, one)

                lhs_row = plan.lhs_row
                if input_rows is not None:
                    lhs_row = input_rows.emit_row(builder, output_row, ky, kx, channel)
                elif gather_runs is None:
                    run = window.emit_row_run(builder, output_row, rows, ky, kx, channel, depth)
                    builder.branch(run["predicate"], interior, boundary)
                else:
                    _, _, width, channels = window.input_shape
                    dy, dx = window.dilation
                    tap = op(
                        "add", op("add", op("mul", ky, dy * width * channels), op("mul", kx, dx * channels)), channel
                    )
                    for base, local, count in emit_convolution_gather_runs(gather_runs, builder, output_row, rows):

                        def fragment(base=base, local=local, count=count):
                            transfer(
                                op("add", base, tap),
                                gather_runs.source_row_stride,
                                op("add", plan.lhs_row, local),
                                count,
                                False,
                            )

                        builder.branch(builder.cmp("ult", 0, count), fragment, lambda: None)
                reduction = op("add", op("mul", op("add", op("mul", ky, kw), kx), ci), channel)
                first = builder.op("and", builder.cmp("eq", ky, 0), builder.cmp("eq", kx, 0), "i1")
                first = builder.op("and", first, builder.cmp("eq", channel, 0), "i1")
                for output in outputs:
                    lower(
                        "load_weight",
                        dict(
                            reduction_row=reduction,
                            output_column=output["output_column"],
                            local_row=plan.rhs_row,
                            rows=depth,
                            columns=output["columns"],
                        ),
                    )
                    compute = dict(
                        output,
                        lhs_row=lhs_row,
                        rhs_row=plan.rhs_row,
                        depth=depth,
                        first_reduction=first,
                        bias=plan.bias,
                    )
                    if input_rows is not None and input_rows.row_step != 1:
                        compute["lhs_row_step"] = input_rows.row_step
                    lower("compute", compute)

            whole, tail = divmod(ci, plan.channel_tile)
            if whole:
                builder.loop(
                    0, whole, 1, lambda index: channel_body(op("mul", index, plan.channel_tile), plan.channel_tile)
                )
            if tail:
                channel_body(whole * plan.channel_tile, tail)

        builder.loop(0, kw, 1, kx_body)

    builder.loop(0, kh, 1, ky_body)
    for output in outputs:
        lower("store", output)


# Preserve the original public lookup and pickle identities.
for _definition in (
    emit_counted_convolution,
    emit_convolution_tile,
    _emit_convolution_tile_effects,
):
    _definition.__module__ = "merlin.compiler.schedule.window_transfer"
del _definition
