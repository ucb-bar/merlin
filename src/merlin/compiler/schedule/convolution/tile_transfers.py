"""Explicit convolution issue, retention and readout premises."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from merlin.compiler.schedule.convolution.input_rows import (
    ConvolutionGatherRuns,
    ConvolutionResidentInputRows,
    emit_convolution_gather_runs,
)

if TYPE_CHECKING:
    from merlin.compiler.schedule.convolution.windows import CountedConvolutionPlan

# ---- convolution_transfer_laws -----------------------------------------------------------------


@dataclass(frozen=True)
class ConvolutionLoadIssueLaw:
    """Selected original target premises for one bounded load/compute epoch.

    Load issue is not completion. The selected target must enforce actual local
    range dependencies and ordered configuration capture, and backpressure every
    finite queue. All source buffers stay mapped, immutable and disjoint from
    outputs. The completion callback must perform a real full-device drain after
    each paired compute and each store, before any private-row reuse. These
    caller-owned premises supply no target qualification by themselves.
    """

    hardware_identity: str
    numerical_policy_identity: str
    source_identity: str
    qualification_identity: str
    load_issue_budget: int
    local_range_dependencies: bool = False
    ordered_load_configuration: bool = False
    paired_descriptor_compute: bool = False
    finite_queue_backpressure: bool = False
    complete_device_drain: bool = False
    immutable_mapped_disjoint_buffers: bool = False

    def __post_init__(self):
        identities = (
            self.hardware_identity,
            self.numerical_policy_identity,
            self.source_identity,
            self.qualification_identity,
        )
        if (
            any(type(x) is not str or not x.strip() for x in identities)
            or type(self.load_issue_budget) is not int
            or not 0 < self.load_issue_budget < 2**31
            or any(
                type(x) is not bool
                for x in (
                    self.local_range_dependencies,
                    self.ordered_load_configuration,
                    self.paired_descriptor_compute,
                    self.finite_queue_backpressure,
                    self.complete_device_drain,
                    self.immutable_mapped_disjoint_buffers,
                )
            )
        ):
            raise ValueError("convolution load issue requires bounded typed original target premises")


def _validate_convolution_load_issue(plan, builder, issue, complete, rows, law, bias_loads):
    from merlin.compiler.schedule.convolution.windows import CountedConvolutionPlan

    if type(plan) is not CountedConvolutionPlan or type(law) is not ConvolutionLoadIssueLaw:
        raise ValueError("convolution load issue requires its original typed plan and law")
    plan.__post_init__()
    law.__post_init__()
    if (
        not all(
            (
                law.local_range_dependencies,
                law.ordered_load_configuration,
                law.paired_descriptor_compute,
                law.finite_queue_backpressure,
                law.complete_device_drain,
                law.immutable_mapped_disjoint_buffers,
            )
        )
        or not callable(issue)
        or not callable(complete)
        or any(not callable(getattr(builder, n, None)) for n in ("op", "cmp", "loop", "branch"))
    ):
        raise ValueError("convolution load issue lacks complete source dependencies or real completion")
    if type(rows) is not int or rows < 1 or rows + 1 + bias_loads > law.load_issue_budget:
        raise ValueError("convolution load issue exceeds its selected logical epoch budget")
    return plan.identity, asdict(law)


@dataclass(frozen=True)
class ConvolutionOutputTile:
    """One static original output-channel interval and its private live ACC rows."""

    output_column: int
    columns: int
    accumulator_row: int


@dataclass(frozen=True)
class ConvolutionReadoutTransferLaw:
    """Separate selected mapping for a multi-block scaled integer readout.

    Physical block ``b``, row ``r`` reads ACC base+b*block_rows+r and
    publishes original output columns b*tile_columns through the unchanged
    numerical state. This is not implied by a load map or readout epoch law.
    Raw accumulator, pooling and normalization readouts are not admitted here.
    State identities must come from the original source exits, not fitted
    coefficients. A final full drain still closes every original output.
    """

    hardware_identity: str
    numerical_policy_identity: str
    source_identity: str
    qualification_identity: str
    tile_columns: int
    block_rows: int
    max_columns: int
    column_states: tuple[tuple[int, int, str], ...]
    scaled_i8: bool = False
    exact_block_column_map: bool = False
    original_numerical_state: bool = False

    @property
    def identity(self):
        return hashlib.sha256(repr(asdict(self)).encode()).hexdigest()


def plan_convolution_readout_transfers(plan, output_tiles, law, *, rows):
    """Partition all original stores into exact adjacent physical block runs.

    No new ACC rows, columns or output bytes are introduced. A partial tile
    ends a run; different numerical states or nonmatching physical spacing
    retain separate stores. The consumer still owns source-qualified target
    encoding, ordinary output nonalias and producer/readout completion.
    """
    from merlin.compiler.schedule.convolution.windows import CountedConvolutionPlan

    if (
        type(plan) is not CountedConvolutionPlan
        or type(law) is not ConvolutionReadoutTransferLaw
        or any(
            type(v) is not str or not v.strip()
            for v in (
                law.hardware_identity,
                law.numerical_policy_identity,
                law.source_identity,
                law.qualification_identity,
            )
        )
        or any(type(v) is not int or not 0 < v < 2**31 for v in (law.tile_columns, law.block_rows, law.max_columns))
        or any(
            type(v) is not bool or not v
            for v in (law.scaled_i8, law.exact_block_column_map, law.original_numerical_state)
        )
        or law.source_identity != plan.identity
        or law.tile_columns != plan.column_tile
        or law.max_columns < law.tile_columns
        or type(rows) is not int
        or not 0 < rows <= min(law.block_rows, plan.row_tile)
        or type(output_tiles) is not tuple
        or not output_tiles
        or type(law.column_states) is not tuple
        or not 0 < len(law.column_states) <= 262144
    ):
        raise ValueError("wide convolution readout needs original source, block map and scaled-i8 premises")
    end = 0
    for state in law.column_states:
        if (
            type(state) is not tuple
            or len(state) != 3
            or type(state[0]) is not int
            or type(state[1]) is not int
            or state[0] != end
            or state[1] <= 0
            or type(state[2]) is not str
            or len(state[2]) != 64
            or any(c not in "0123456789abcdef" for c in state[2])
        ):
            raise ValueError("readout states must cover every original column exactly")
        end += state[1]
    if end != plan.output_channels:
        raise ValueError("readout state roster differs from original output columns")
    groups, states, spans, end = [], [], [], None
    for tile in output_tiles:
        if (
            type(tile) is not ConvolutionOutputTile
            or any(type(v) is not int for v in (tile.output_column, tile.columns, tile.accumulator_row))
            or not 0 < tile.columns <= law.tile_columns
            or tile.output_column < 0
            or tile.output_column + tile.columns > plan.output_channels
            or end is not None
            and tile.output_column != end
            or tile.accumulator_row < 0
            or tile.accumulator_row + rows > plan.accumulator_rows
            or any(tile.accumulator_row < b and a < tile.accumulator_row + rows for a, b in spans)
        ):
            raise ValueError("wide readout requires original contiguous outputs and disjoint private ACC rows")
        selected = {
            key
            for start, count, key in law.column_states
            if start < tile.output_column + tile.columns and tile.output_column < start + count
        }
        if len(selected) != 1:
            raise ValueError("original store crosses numerical readout states")
        state = selected.pop()
        if (
            groups
            and states[-1] == state
            and groups[-1][-1].columns == law.tile_columns
            and tile.accumulator_row == groups[-1][0].accumulator_row + len(groups[-1]) * law.block_rows
            and sum(t.columns for t in groups[-1]) + tile.columns <= law.max_columns
        ):
            groups[-1].append(tile)
        else:
            groups.append([tile])
            states.append(state)
        spans.append((tile.accumulator_row, tile.accumulator_row + rows))
        end = tile.output_column + tile.columns
    return tuple(tuple(group) for group in groups)


@dataclass(frozen=True)
class ConvolutionInputRetentionLaw:
    """Caller-bound target premises for reusing one loaded input across outputs.

    Every callback completes synchronously. RHS loads, compute and stores must
    preserve the input rows. External inputs are immutable and mapped; outputs
    are disjoint. Independent channel reductions may interleave, but each keeps
    its original K order and every original final observation. These premises
    do not confer execution qualification on a target. Cross-spatial grouping
    additionally requires explicit independent_output_rows and absence of
    intermediate observers; the original channel-only premise does not grant it.
    """

    hardware_identity: str
    numerical_policy_identity: str
    source_identity: str
    qualification_identity: str
    max_live_tiles: int
    input_preserved: bool = False
    independent_outputs: bool = False
    immutable_mapped_disjoint_buffers: bool = False
    independent_output_rows: bool = False
    no_intermediate_observers: bool = False

    def __post_init__(self):
        if (
            any(
                type(x) is not str or not x.strip()
                for x in (
                    self.hardware_identity,
                    self.numerical_policy_identity,
                    self.source_identity,
                    self.qualification_identity,
                )
            )
            or type(self.max_live_tiles) is not int
            or not 0 < self.max_live_tiles < 2**31
            or any(
                type(x) is not bool
                for x in (
                    self.input_preserved,
                    self.independent_outputs,
                    self.immutable_mapped_disjoint_buffers,
                    self.independent_output_rows,
                    self.no_intermediate_observers,
                )
            )
        ):
            raise ValueError("convolution retention requires typed bounded original target premises")


@dataclass(frozen=True)
class ConvolutionComputeEpochLaw:
    """Additional selected source law for operand reuse across issued computes.

    This is not inferred from a load-issue law or storage capacity. Actual local
    RAW/WAR/WAW checks, accumulator result visibility and operand consumption
    retirement must govern accepted commands. Configuration is captured in its
    original lane order, every software compute emits its descriptor/data pair,
    and finite queues backpressure without making that pair unable to progress.
    Full completion remains real at epoch boundaries and every original store.

    ``max_compute_pairs`` bounds compiler-issued work between full drains, not
    hardware entries. Optional ``pending_compute_pairs`` records the selected
    finite pending capacity. Issuing more work than that capacity relies on the
    independently established ready/valid pair-progress law, never extra slots
    or invented completion. Unknown capacity stays unknown for older bindings.
    """

    hardware_identity: str
    numerical_policy_identity: str
    source_identity: str
    qualification_identity: str
    max_compute_pairs: int
    local_read_after_write: bool = False
    local_write_after_read: bool = False
    local_write_after_write: bool = False
    accumulator_result_visibility: bool = False
    operand_consumption_retirement: bool = False
    ordered_configuration_capture: bool = False
    adjacent_descriptor_compute: bool = False
    finite_queue_pair_progress: bool = False
    complete_device_drain: bool = False
    exclusive_private_state: bool = False
    pending_compute_pairs: int | None = None

    def __post_init__(self):
        identities = (
            self.hardware_identity,
            self.numerical_policy_identity,
            self.source_identity,
            self.qualification_identity,
        )
        flags = (
            self.local_read_after_write,
            self.local_write_after_read,
            self.local_write_after_write,
            self.accumulator_result_visibility,
            self.operand_consumption_retirement,
            self.ordered_configuration_capture,
            self.adjacent_descriptor_compute,
            self.finite_queue_pair_progress,
            self.complete_device_drain,
            self.exclusive_private_state,
        )
        if (
            any(type(v) is not str or not v.strip() for v in identities)
            or type(self.max_compute_pairs) is not int
            or not 1 < self.max_compute_pairs < 2**31
            or self.pending_compute_pairs is not None
            and (type(self.pending_compute_pairs) is not int or not 0 < self.pending_compute_pairs < 2**31)
            or any(type(v) is not bool for v in flags)
        ):
            raise ValueError("convolution compute epochs require bounded typed original source premises")

    def require(self, pairs):
        self.__post_init__()
        if (
            type(pairs) is not int
            or not 1 < pairs <= self.max_compute_pairs
            or not all(
                (
                    self.local_read_after_write,
                    self.local_write_after_read,
                    self.local_write_after_write,
                    self.accumulator_result_visibility,
                    self.operand_consumption_retirement,
                    self.ordered_configuration_capture,
                    self.adjacent_descriptor_compute,
                    self.finite_queue_pair_progress,
                    self.complete_device_drain,
                    self.exclusive_private_state,
                )
            )
        ):
            raise ValueError("convolution compute epoch lacks complete dependency/progress premises or budget")


@dataclass(frozen=True)
class ConvolutionReadoutEpochLaw:
    """Selected producer/store dependencies for one private output block.

    Readout commands retain their original scalar configurations and addresses.
    Actual accumulator RAW/result visibility orders stores after the final
    producer; finite queues backpressure independently of the compiler's issued
    work bound. A real final drain precedes any private-row or output reuse.
    This does not permit overlap with the next output block or input transport.
    The checked spatial-group consumer may bind a larger private block only
    with separate independent-row/no-observer permissions, exact disjoint ACC
    cells and a sufficient issued-readout budget. Its final drain closes the
    entire group before any ACC or input-ring ownership reuse.
    Within that same group, its checked consumer can publish a cell after its
    exact last original K contribution while later producers use disjoint cells.
    RAW/result visibility and ordered configuration capture still apply; this
    grants neither a readiness latency nor an early overwrite/completion.
    """

    hardware_identity: str
    numerical_policy_identity: str
    source_identity: str
    qualification_identity: str
    max_issued_readouts: int
    pending_readouts: int
    local_read_after_write: bool = False
    accumulator_result_visibility: bool = False
    ordered_configuration_capture: bool = False
    finite_queue_progress: bool = False
    complete_device_drain: bool = False
    exclusive_private_state: bool = False

    def __post_init__(self):
        if (
            any(
                type(v) is not str or not v.strip()
                for v in (
                    self.hardware_identity,
                    self.numerical_policy_identity,
                    self.source_identity,
                    self.qualification_identity,
                )
            )
            or any(type(v) is not int or not 0 < v < 2**31 for v in (self.max_issued_readouts, self.pending_readouts))
            or any(
                type(v) is not bool
                for v in (
                    self.local_read_after_write,
                    self.accumulator_result_visibility,
                    self.ordered_configuration_capture,
                    self.finite_queue_progress,
                    self.complete_device_drain,
                    self.exclusive_private_state,
                )
            )
        ):
            raise ValueError("convolution readout epochs require typed bounded original source premises")

    def require(self, plan, rows, output_tiles):
        from merlin.compiler.schedule.convolution.windows import CountedConvolutionPlan

        self.__post_init__()
        if (
            not all(
                (
                    self.local_read_after_write,
                    self.accumulator_result_visibility,
                    self.ordered_configuration_capture,
                    self.finite_queue_progress,
                    self.complete_device_drain,
                    self.exclusive_private_state,
                )
            )
            or type(plan) is not CountedConvolutionPlan
            or type(rows) is not int
            or not 0 < rows <= plan.row_tile
            or type(output_tiles) is not tuple
            or not output_tiles
            or len(output_tiles) > self.max_issued_readouts
        ):
            raise ValueError("convolution readout epoch lacks dependency/progress premises or budget")
        plan.__post_init__()
        spans = []
        end = None
        for tile in output_tiles:
            if (
                type(tile) is not ConvolutionOutputTile
                or any(type(v) is not int for v in (tile.output_column, tile.columns, tile.accumulator_row))
                or not 0 < tile.columns <= plan.column_tile
                or tile.output_column < 0
                or tile.output_column + tile.columns > plan.output_channels
                or end is not None
                and tile.output_column != end
                or tile.accumulator_row < 0
                or tile.accumulator_row + rows > plan.accumulator_rows
                or any(tile.accumulator_row < hi and lo < tile.accumulator_row + rows for lo, hi in spans)
            ):
                raise ValueError("convolution readout epoch requires disjoint original output and ACC intervals")
            spans.append((tile.accumulator_row, tile.accumulator_row + rows))
            end = tile.output_column + tile.columns


# Preserve the original public lookup and pickle identities.
for _definition in (
    ConvolutionLoadIssueLaw,
    _validate_convolution_load_issue,
    ConvolutionOutputTile,
    ConvolutionReadoutTransferLaw,
    plan_convolution_readout_transfers,
    ConvolutionInputRetentionLaw,
    ConvolutionComputeEpochLaw,
    ConvolutionReadoutEpochLaw,
):
    _definition.__module__ = "merlin.compiler.schedule.convolution.windows"
del _definition


# ---- convolution_tiles -------------------------------------------------------------------------
# Emit original counted convolution tiles through checked transfer callbacks.


def emit_counted_convolution(plan: CountedConvolutionPlan, builder, lower, *, synchronous_completion: bool):
    """Emit the checked counted plan through actual target operation callbacks.

    ``builder`` supplies integer op/cmp, loop and branch, retaining counted loops
    in an IR implementation. ``lower(kind, fields)`` implements each effect and
    its synchronous completion. Interior rows use one affine input DMA; boundary
    tiles use checked one-row transfers or explicit zero-source transfers. No
    input payload is copied or read by this shared host control emitter.
    """
    from merlin.compiler.schedule.convolution.windows import CountedConvolutionPlan

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
    from merlin.compiler.schedule.convolution.windows import CountedConvolutionPlan

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
    _definition.__module__ = "merlin.compiler.schedule.convolution.windows"
del _definition
