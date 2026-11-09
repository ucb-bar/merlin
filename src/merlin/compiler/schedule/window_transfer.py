"""Prove a convolution window's rows form one in-bounds affine DMA rectangle.

This target-neutral index calculation supplies only a predicate, first logical
element and row stride. The caller retains its scalar gather for every rejected
run and owns the physical transfer, element width and target qualification."""

import hashlib as hashlib
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from math import prod as prod

from merlin.compiler.schedule.affine_gather_runs import ConvolutionGatherRuns as ConvolutionGatherRuns
from merlin.compiler.schedule.affine_gather_runs import emit_convolution_gather_runs as emit_convolution_gather_runs

from .convolution_input_rows import ConvolutionResidentInputRows as ConvolutionResidentInputRows
from .convolution_tiles import _emit_convolution_tile_effects as _emit_convolution_tile_effects
from .convolution_tiles import emit_convolution_tile as emit_convolution_tile
from .convolution_tiles import emit_counted_convolution as emit_counted_convolution
from .convolution_transfer_laws import (
    ConvolutionComputeEpochLaw as ConvolutionComputeEpochLaw,
)
from .convolution_transfer_laws import (
    ConvolutionInputRetentionLaw as ConvolutionInputRetentionLaw,
)
from .convolution_transfer_laws import (
    ConvolutionLoadIssueLaw as ConvolutionLoadIssueLaw,
)
from .convolution_transfer_laws import (
    ConvolutionOutputTile as ConvolutionOutputTile,
)
from .convolution_transfer_laws import (
    ConvolutionReadoutEpochLaw as ConvolutionReadoutEpochLaw,
)
from .convolution_transfer_laws import (
    ConvolutionReadoutTransferLaw as ConvolutionReadoutTransferLaw,
)
from .convolution_transfer_laws import (
    _validate_convolution_load_issue as _validate_convolution_load_issue,
)
from .convolution_transfer_laws import (
    plan_convolution_readout_transfers as plan_convolution_readout_transfers,
)
from .window_geometry import CountedConvolutionPlan as CountedConvolutionPlan
from .window_geometry import Window2D as Window2D


def emit_convolution_load_issue_tile(
    plan,
    builder,
    issue,
    complete,
    *,
    output_row,
    output_column,
    rows,
    columns,
    accumulator_row,
    law,
    gather_runs=None,
    input_rows=None,
):
    """Issue one source tile with real full completion at each compute/store.

    This distinct entry does not claim synchronous completion for individual
    loads or weaken the existing synchronous emitter. Input border rows and the
    weight transfer may remain pending together; selected range dependencies
    must delay the actual paired descriptor/compute until their operands are
    ready. No next reduction, output reuse or host observation crosses the full
    completion callback. The original source reduction order and effects stay
    in the canonical shared implementation.
    """
    before = _validate_convolution_load_issue(
        plan, builder, issue, complete, rows, law, int(plan.bias) if type(plan) is CountedConvolutionPlan else 0
    )

    def lower(kind, fields):
        issue(kind, fields)
        if kind in ("compute", "store"):
            complete(kind, fields)

    _emit_convolution_tile_effects(
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
    plan.__post_init__()
    law.__post_init__()
    if before != (plan.identity, asdict(law)):
        raise ValueError("original convolution issue source or target premises changed during emission")
    return dict(
        schema="merlin.convolution_load_issue.v1",
        plan_sha256=plan.identity,
        law=asdict(law),
        completion="full device drain after each paired compute and each store",
        source_order="unchanged kernel_y/kernel_x/input_channel and original loads/stores",
        execution_qualification="independent original target obligation",
        timing="UNKNOWN",
    )


def emit_convolution_column_block(
    plan,
    builder,
    lower,
    *,
    output_row,
    rows,
    output_tiles,
    law,
    synchronous_completion,
    gather_runs=None,
    input_rows=None,
):
    """Reuse canonical window input across independent private output tiles.

    The caller retains global invocation coverage. This entry only interleaves
    the listed contiguous output intervals; each consumes the unchanged ordered
    kh/kw/channel reduction and has its own nonoverlapping ACC lifetime through
    final store. No callback completion, host preparation or input copy is
    inferred. There is no implicit asynchronous execution grant.
    """
    if synchronous_completion is not True:
        raise ValueError("convolution retention requires original plan/law and synchronous realization")
    return _emit_convolution_column_block(
        plan,
        builder,
        lower,
        output_row=output_row,
        rows=rows,
        output_tiles=output_tiles,
        law=law,
        gather_runs=gather_runs,
        input_rows=input_rows,
    )


def _emit_convolution_column_block(
    plan, builder, lower, *, output_row, rows, output_tiles, law, gather_runs=None, input_rows=None
):
    if type(plan) is not CountedConvolutionPlan or type(law) is not ConvolutionInputRetentionLaw:
        raise ValueError("convolution retention requires original plan/law and synchronous realization")
    plan.__post_init__()
    law.__post_init__()
    if (
        not all((law.input_preserved, law.independent_outputs, law.immutable_mapped_disjoint_buffers))
        or not callable(lower)
        or any(not callable(getattr(builder, n, None)) for n in ("op", "cmp", "loop", "branch"))
    ):
        raise ValueError("convolution retention lacks complete source/effect premises")
    if (
        type(output_tiles) is not tuple
        or not output_tiles
        or len(output_tiles) > law.max_live_tiles
        or type(rows) is not int
        or not 0 < rows <= plan.row_tile
    ):
        raise ValueError("convolution retention exceeds its live tile/row budget")
    previous_end = None
    spans = []
    for tile in output_tiles:
        if (
            type(tile) is not ConvolutionOutputTile
            or any(type(x) is not int for x in (tile.output_column, tile.columns, tile.accumulator_row))
            or not 0 < tile.columns <= plan.column_tile
            or tile.output_column < 0
            or tile.output_column + tile.columns > plan.output_channels
            or tile.accumulator_row < 0
            or tile.accumulator_row + rows > plan.accumulator_rows
            or (previous_end is not None and previous_end != tile.output_column)
        ):
            raise ValueError("convolution retention requires contiguous original output intervals and bounded ACC rows")
        span = (tile.accumulator_row, tile.accumulator_row + rows)
        if any(span[0] < end and start < span[1] for start, end in spans):
            raise ValueError("convolution retention has overlapping live ACC rows")
        spans.append(span)
        previous_end = tile.output_column + tile.columns
    before = plan.identity, asdict(law), tuple(asdict(t) for t in output_tiles)
    first = output_tiles[0]
    _emit_convolution_tile_effects(
        plan,
        builder,
        lower,
        output_row=output_row,
        output_column=first.output_column,
        rows=rows,
        columns=first.columns,
        accumulator_row=first.accumulator_row,
        output_tiles=output_tiles,
        gather_runs=gather_runs,
        input_rows=input_rows,
    )
    plan.__post_init__()
    law.__post_init__()
    if before != (plan.identity, asdict(law), tuple(asdict(t) for t in output_tiles)):
        raise ValueError("original convolution retention source/resources changed during emission")
    return dict(
        schema="merlin.convolution_column_retention.v1",
        plan_sha256=plan.identity,
        law=asdict(law),
        outputs=[asdict(t) for t in output_tiles],
        live_accumulator_rows=spans,
        input_transfer="once per original kh/kw/channel chunk for this output block",
        source_order="unchanged per-output kernel_y/kernel_x/input_channel",
        completion="synchronous target callbacks before private-row reuse",
        execution_qualification="independent original target obligation",
        timing="UNKNOWN",
    )


def emit_convolution_load_issue_column_block(
    plan, builder, issue, complete, *, output_row, rows, output_tiles, law, issue_law, gather_runs=None, input_rows=None
):
    """Compose original input retention and a bounded load/compute issue epoch.

    This separate entry preserves both parents' admission. The first epoch may
    contain one bias load per live output, the original boundary-row input loads
    and one RHS load. Actual selected range dependencies delay paired compute;
    full completion after each compute/store precedes RHS or output reuse. The
    next output tile may reuse LHS only under the independent retention law.
    """
    if type(plan) is not CountedConvolutionPlan or type(output_tiles) is not tuple or not output_tiles:
        raise ValueError("convolution load retention requires original plan and output tiles")
    before = _validate_convolution_load_issue(
        plan, builder, issue, complete, rows, issue_law, len(output_tiles) * int(plan.bias)
    )

    def lower(kind, fields):
        issue(kind, fields)
        if kind in ("compute", "store"):
            complete(kind, fields)

    result = _emit_convolution_column_block(
        plan,
        builder,
        lower,
        output_row=output_row,
        rows=rows,
        output_tiles=output_tiles,
        law=law,
        gather_runs=gather_runs,
        input_rows=input_rows,
    )
    plan.__post_init__()
    issue_law.__post_init__()
    if before != (plan.identity, asdict(issue_law)):
        raise ValueError("original convolution issue source or target premises changed during emission")
    return dict(
        result,
        schema="merlin.convolution_load_issue_column_retention.v1",
        issue_law=asdict(issue_law),
        completion="full device drain after each paired compute and each store",
    )


def emit_convolution_compute_epoch_column_block(
    plan,
    builder,
    issue,
    complete,
    *,
    output_row,
    rows,
    output_tiles,
    law,
    issue_law,
    epoch_law,
    compute_pairs,
    gather_runs=None,
    input_rows=None,
    readout_law=None,
    readout_transfer_law=None,
):
    """Emit runtime-bounded complete compute pairs, preserving original stores.

    The actual source K coordinates determine a one-based pair index. A real
    runtime branch drains on multiples of compute_pairs, across nested kh/kw/ci
    iterations and output columns. A Python emission counter is never used.
    By default every store has full completion. The separate optional readout
    law instead orders the final producer and original disjoint stores through
    actual dependencies, with a real full drain after the last store. All
    intermediate compute epoch drains remain. No private ownership crosses this
    block's final drain.
    """
    if (
        type(plan) is not CountedConvolutionPlan
        or type(output_tiles) is not tuple
        or not output_tiles
        or type(epoch_law) is not ConvolutionComputeEpochLaw
    ):
        raise ValueError("convolution compute epochs require original plan, output tiles and stronger source law")
    epoch_law.require(compute_pairs)
    before_issue = _validate_convolution_load_issue(
        plan, builder, issue, complete, rows, issue_law, len(output_tiles) * int(plan.bias)
    )
    if (
        epoch_law.hardware_identity != issue_law.hardware_identity
        or epoch_law.numerical_policy_identity != issue_law.numerical_policy_identity
    ):
        raise ValueError("convolution compute epoch and issue laws select different contracts")
    if readout_law is not None:
        if type(readout_law) is not ConvolutionReadoutEpochLaw:
            raise ValueError("convolution readout epoch requires its separate typed source law")
        readout_law.require(plan, rows, output_tiles)
        if (
            readout_law.hardware_identity != epoch_law.hardware_identity
            or readout_law.numerical_policy_identity != epoch_law.numerical_policy_identity
        ):
            raise ValueError("convolution readout and compute epochs select different contracts")
    before_readout = None if readout_law is None else asdict(readout_law)
    transfers = None
    transfer_identity = None
    if readout_transfer_law is not None:
        if (
            type(readout_transfer_law) is not ConvolutionReadoutTransferLaw
            or readout_law is None
            or readout_transfer_law.hardware_identity != epoch_law.hardware_identity
            or readout_transfer_law.numerical_policy_identity != epoch_law.numerical_policy_identity
        ):
            raise ValueError("wide readout requires its original readout epoch and same selected target")
        transfers = plan_convolution_readout_transfers(plan, output_tiles, readout_transfer_law, rows=rows)
        transfer_identity = readout_transfer_law.identity
    before_epoch = asdict(epoch_law)
    ci = plan.window.input_shape[-1]
    chunks = (ci + plan.channel_tile - 1) // plan.channel_tile
    positions = {t.output_column: n for n, t in enumerate(output_tiles) if type(t) is ConvolutionOutputTile}
    pending = None
    stores = 0
    held_stores = []
    transfer_index = 0
    total_pairs = plan.window.kernel[0] * plan.window.kernel[1] * chunks * len(output_tiles)

    def lower(kind, fields):
        nonlocal pending, stores, transfer_index
        if kind == "load_weight":
            if pending is not None:
                raise ValueError("convolution epoch changed the original weight/compute pairing")
            pending = fields["reduction_row"], fields["output_column"]
        if kind != "store" or transfers is None:
            issue(kind, fields)
        if kind == "compute":
            if pending is None or pending[1] != fields["output_column"]:
                raise ValueError("convolution epoch lost original reduction coordinates")
            reduction, column = pending
            kernel = builder.op("udiv", reduction, ci)
            channel = builder.op("urem", reduction, ci)
            chunk = builder.op("add", builder.op("mul", kernel, chunks), builder.op("udiv", channel, plan.channel_tile))
            pair = builder.op("add", builder.op("mul", chunk, len(output_tiles)), positions[column] + 1)
            boundary = builder.cmp("eq", builder.op("urem", pair, compute_pairs), 0)

            def drain_boundary():
                if readout_law is None:
                    complete(kind, fields)
                else:
                    # The pair index is actual runtime source K/column work,
                    # never a Python template counter or a queue-slot count.
                    builder.branch(builder.cmp("eq", pair, total_pairs), lambda: None, lambda: complete(kind, fields))

            builder.branch(boundary, drain_boundary, lambda: None)
            pending = None
        elif kind == "store":
            if pending is not None:
                raise ValueError("convolution epoch reaches a store with an unmatched descriptor")
            if stores >= len(output_tiles) or fields["output_column"] != output_tiles[stores].output_column:
                raise ValueError("convolution readout epoch changed original store order")
            stores += 1
            if transfers is not None:
                held_stores.append(dict(fields))
                group = transfers[transfer_index]
                if len(held_stores) == len(group):
                    first = held_stores[0]
                    if any(
                        f["rows"] != rows
                        or f["output_row"] is not first["output_row"]
                        and f["output_row"] != first["output_row"]
                        or (f["output_column"], f["columns"], f["accumulator_row"])
                        != (t.output_column, t.columns, t.accumulator_row)
                        for f, t in zip(held_stores, group, strict=True)
                    ):
                        raise ValueError("actual readout changed its checked original block mapping")
                    issue("store", dict(first, columns=sum(t.columns for t in group)))
                    held_stores.clear()
                    transfer_index += 1
            if readout_law is None or stores == len(output_tiles):
                complete(kind, fields)

    result = _emit_convolution_column_block(
        plan,
        builder,
        lower,
        output_row=output_row,
        rows=rows,
        output_tiles=output_tiles,
        law=law,
        gather_runs=gather_runs,
        input_rows=input_rows,
    )
    if (
        pending is not None
        or stores != len(output_tiles)
        or held_stores
        or transfers is not None
        and (transfer_index != len(transfers) or readout_transfer_law.identity != transfer_identity)
        or before_issue != (plan.identity, asdict(issue_law))
        or before_epoch != asdict(epoch_law)
        or before_readout != (None if readout_law is None else asdict(readout_law))
    ):
        raise ValueError("original convolution epoch source/target premises changed during emission")
    return dict(
        result,
        schema="merlin.convolution_compute_epoch.v1",
        issue_law=asdict(issue_law),
        epoch_law=asdict(epoch_law),
        compute_pairs=compute_pairs,
        epoch_index="((ky*kw+kx)*ceil(ci/channel_tile)+channel/channel_tile)*output_tiles+column_slot+1",
        readout_law=before_readout,
        **(
            {}
            if transfers is None
            else dict(
                readout_transfers=dict(
                    law_sha256=transfer_identity,
                    original_stores=len(output_tiles),
                    physical_stores=len(transfers),
                    groups=[
                        dict(
                            output_column=g[0].output_column,
                            columns=sum(t.columns for t in g),
                            accumulator_row=g[0].accumulator_row,
                            original_tiles=len(g),
                        )
                        for g in transfers
                    ],
                )
            )
        ),
        completion=(
            "real intermediate compute boundaries; producer/store RAW; full completion after last original store"
            if readout_law is not None
            else "real runtime modulo boundary after complete pairs; full completion after every store"
        ),
    )
