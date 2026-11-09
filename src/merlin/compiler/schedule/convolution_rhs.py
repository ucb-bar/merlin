"""Prepare and complete original convolution RHS panels."""

from __future__ import annotations

from merlin.compiler.schedule.convolution_choices import ConvolutionRhsResidency as ConvolutionRhsResidency
from merlin.compiler.schedule.convolution_choices import _rhs_intervals as _rhs_intervals


def _prepare_convolution_rhs(rhs, builder, lower, complete, *, already_prepared=False):
    p = rhs.schedule.source.plan
    ci = p.window.input_shape[-1]
    kh, kw = p.window.kernel
    intervals = _rhs_intervals(rhs.schedule, rhs.column_intervals)
    chunks = (ci + p.channel_tile - 1) // p.channel_tile
    columns = {column: n for n, (column, _) in enumerate(intervals)}

    def address(kernel, chunk, column):
        if type(column) is int:
            slot = columns[column]
        else:
            # Counted consumers may carry the original column loop as SSA.
            # Division is an exact slot map only for complete regular tiles
            # (with an independent final tail), never arbitrary scalar exits.
            start = intervals[0][0]
            if any(
                c != start + n * p.column_tile or (width != p.column_tile and n != len(intervals) - 1)
                for n, (c, width) in enumerate(intervals)
            ):
                raise ValueError("counted RHS columns require original regular tile boundaries")
            slot = builder.op("udiv", builder.op("sub", column, start), p.column_tile)
        index = builder.op("add", builder.op("mul", kernel, chunks), chunk)
        index = builder.op("add", builder.op("mul", index, len(intervals)), slot)
        return builder.op("add", rhs.layout.rhs_base, builder.op("mul", index, p.channel_tile))

    def kernel_body(kernel):
        def channel_body(chunk, channel, depth):
            reduction = builder.op("add", builder.op("mul", kernel, ci), channel)
            for column, width in intervals:
                lower(
                    "load_weight",
                    dict(
                        reduction_row=reduction,
                        output_column=column,
                        local_row=address(kernel, chunk, column),
                        rows=depth,
                        columns=width,
                    ),
                )

        whole, tail = divmod(ci, p.channel_tile)
        if whole:
            builder.loop(
                0, whole, 1, lambda chunk: channel_body(chunk, builder.op("mul", chunk, p.channel_tile), p.channel_tile)
            )
        if tail:
            channel_body(whole, whole * p.channel_tile, tail)

    if not already_prepared:
        builder.loop(0, kh * kw, 1, kernel_body)
    complete("weight_preparation", dict(panels=rhs.layout.k_tiles * rhs.layout.n_tiles, local_rows=rhs.layout.rhs_rows))
    pending = None

    def consume(kind, fields):
        nonlocal pending
        if kind == "load_weight":
            if pending is not None:
                raise ValueError("resident convolution RHS lost original load/compute pairing")
            pending = fields
        elif kind == "compute":
            if pending is None or pending["output_column"] != fields["output_column"]:
                raise ValueError("resident convolution RHS lacks original consumed weight coordinates")
            reduction = pending["reduction_row"]
            local = address(
                builder.op("udiv", reduction, ci),
                builder.op("udiv", builder.op("urem", reduction, ci), p.channel_tile),
                fields["output_column"],
            )
            lower(kind, dict(fields, rhs_row=local))
            pending = None
        else:
            if pending is not None:
                raise ValueError("resident convolution RHS has intervening original effects")
            lower(kind, fields)

    def check():
        if pending is not None:
            raise ValueError("resident convolution RHS leaves an unmatched original compute")

    return consume, check


def emit_convolution_rhs_preparation(rhs, builder, lower, *, complete):
    """Prepare real checked RHS tiles for an existing original consumer.

    Return (consumer_lower, finish). Route every original consumer effect
    through consumer_lower, then call finish after its real final drain. This
    consumes the existing effect law and exact physical layout; it grants no
    extra RHS lifetime, source-product coverage, completion or target authority.
    """
    if (
        type(rhs) is not ConvolutionRhsResidency
        or not callable(lower)
        or not callable(complete)
        or any(not callable(getattr(builder, n, None)) for n in ("op", "loop"))
    ):
        raise ValueError("RHS preparation requires its checked owner and actual emit/completion callbacks")
    rhs.verify_current()
    identity = rhs.identity
    consume, check = _prepare_convolution_rhs(rhs, builder, lower, complete)

    def finish():
        check()
        rhs.verify_current()
        if rhs.identity != identity:
            raise ValueError("prepared convolution RHS changed during its original consumer lifetime")

    return consume, finish


# Preserve the original public lookup and pickle identities.
for _definition in (
    _prepare_convolution_rhs,
    emit_convolution_rhs_preparation,
):
    _definition.__module__ = "merlin.compiler.schedule.convolution_exploration"
del _definition
