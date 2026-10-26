"""Prove a convolution window's rows form one in-bounds affine DMA rectangle.

This target-neutral index calculation supplies only a predicate, first logical
element and row stride. The caller retains its scalar gather for every rejected
run and owns the physical transfer, element width and target qualification.
"""

import hashlib
from dataclasses import asdict, dataclass
from math import prod


@dataclass(frozen=True)
class Window2D:
    input_shape: tuple[int, int, int, int]
    kernel: tuple[int, int]
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self):
        for values, count, zero in ((self.input_shape, 4, False), (self.kernel, 2, False),
                                    (self.stride, 2, False), (self.dilation, 2, False),
                                    (self.padding, 4, True)):
            if (type(values) is not tuple or len(values) != count
                    or any(type(x) is not int or x < (0 if zero else 1) or x >= 2**63 for x in values)):
                raise ValueError("window transfer requires explicit bounded integer geometry")
        batch, height, width, channels = self.input_shape
        kh, kw = self.kernel
        sh, sw = self.stride
        dh, dw = self.dilation
        top, left, bottom, right = self.padding
        oh = (height + top + bottom - dh*(kh-1) - 1)//sh + 1
        ow = (width + left + right - dw*(kw-1) - 1)//sw + 1
        if min(oh, ow) <= 0:
            raise ValueError("window transfer has no output rows")
        if max(prod(self.input_shape), batch*oh*ow, channels*sw,
               oh*sh + dh*(kh-1) + top, ow*sw + dw*(kw-1) + left) >= 2**63:
            raise ValueError("window transfer geometry exceeds signed 64-bit address arithmetic")

    @property
    def output_spatial(self):
        _, height, width, _ = self.input_shape
        kh, kw = self.kernel
        sh, sw = self.stride
        dh, dw = self.dilation
        top, left, bottom, right = self.padding
        return ((height+top+bottom-dh*(kh-1)-1)//sh+1,
                (width+left+right-dw*(kw-1)-1)//sw+1)

    def emit_row_run(self, builder, row, rows, ky, kx, channel, columns):
        """Emit an exact same-output-row run predicate through a 64-bit builder.

        ``rows`` and ``columns`` are positive static transfer extents. Other
        coordinates may be builder values. All runtime coordinates and the
        complete input rectangle are checked before a caller may use the run.
        No input-payload read or target instruction is emitted here.
        """
        if any(type(x) is not int or x <= 0 or x >= 2**63 for x in (rows, columns)):
            raise ValueError("window row run needs positive static transfer extents")
        batch, height, width, channels = self.input_shape
        kh, kw = self.kernel
        sh, sw = self.stride
        dh, dw = self.dilation
        top, left, _, _ = self.padding
        oh, ow = self.output_spatial
        if columns > channels:
            raise ValueError("window row run exceeds the input channel extent")
        if rows > ow:
            return {"predicate": False, "first_element": 0, "row_stride_elements": channels*sw}
        image = builder.op("udiv", row, oh*ow)
        y = builder.op("urem", builder.op("udiv", row, ow), oh)
        x = builder.op("urem", row, ow)
        iy = builder.op("add", builder.op("sub", builder.op("mul", y, sh), top), builder.op("mul", ky, dh))
        ix = builder.op("add", builder.op("sub", builder.op("mul", x, sw), left), builder.op("mul", kx, dw))
        last_x = builder.op("add", ix, (rows-1)*sw)
        checks = (
            builder.cmp("ult", row, batch*oh*ow),
            builder.cmp("ule", x, ow-rows),
            builder.cmp("ult", ky, kh), builder.cmp("ult", kx, kw),
            builder.cmp("ule", channel, channels-columns),
            builder.cmp("ult", iy, height), builder.cmp("ult", ix, width),
            builder.cmp("ult", last_x, width),
        )
        predicate = checks[0]
        for value in checks[1:]:
            predicate = builder.op("and", predicate, value, "i1")
        first = builder.op("add", builder.op("mul", image, height), iy)
        first = builder.op("add", builder.op("mul", first, width), ix)
        first = builder.op("add", builder.op("mul", first, channels), channel)
        return {"predicate": predicate, "first_element": first, "row_stride_elements": channels*sw}


@dataclass(frozen=True)
class CountedConvolutionPlan:
    """Synchronous source convolution with bounded affine tile families.

    The supplied private rows and primitive dimensions are caller declarations,
    not inferred target capabilities. Every output tile consumes the complete
    kh/kw/channel reduction before its single store; no asynchronous reuse is
    authorized. Full/tail regions are at most eight combinations, independent
    of the number of runtime tiles.
    """
    window: Window2D
    output_channels: int
    primitive: int
    row_tile: int
    column_tile: int
    channel_tile: int
    operand_rows: int
    accumulator_rows: int
    lhs_row: int
    rhs_row: int
    accumulator_row: int
    bias: bool = False

    def __post_init__(self):
        if type(self.window) is not Window2D:
            raise ValueError("counted convolution requires its original typed window")
        self.window.__post_init__()
        positive=(self.output_channels,self.primitive,self.row_tile,self.column_tile,
                  self.channel_tile,self.operand_rows,self.accumulator_rows)
        if any(type(v) is not int or not 0<v<2**63 for v in positive):
            raise ValueError("counted convolution requires bounded positive resources and extents")
        if any(type(v) is not int or v<0 for v in (self.lhs_row,self.rhs_row,self.accumulator_row)):
            raise ValueError("counted convolution requires nonnegative private row assignments")
        if type(self.bias) is not bool:
            raise ValueError("counted convolution bias source must be explicit")
        if max(self.row_tile,self.column_tile,self.channel_tile)>self.primitive:
            raise ValueError("convolution tile exceeds the declared primitive")
        if self.row_tile>self.window.output_spatial[1]:
            raise ValueError("counted input runs cannot cross an output-row boundary")
        if (self.lhs_row+self.row_tile>self.operand_rows
            or self.rhs_row+self.channel_tile>self.operand_rows
            or self.accumulator_row+self.row_tile>self.accumulator_rows):
            raise ValueError("counted convolution private rows exceed declared capacity")
        if not (self.lhs_row+self.row_tile<=self.rhs_row or self.rhs_row+self.channel_tile<=self.lhs_row):
            raise ValueError("counted convolution operand lifetimes require disjoint private rows")
        batch,h,w,ci=self.window.input_shape
        oh,ow=self.window.output_spatial
        reduction=prod(self.window.kernel)*ci
        if max(reduction*self.output_channels,4*batch*oh*ow*self.output_channels,
               batch*oh*ow*self.output_channels*reduction)>=2**63:
            raise ValueError("counted convolution address/product arithmetic exceeds signed i64")

    @property
    def identity(self):
        return hashlib.sha256(repr(self).encode()).hexdigest()

    @property
    def coverage(self):
        """Arithmetic partitions, not enumeration of shape-sized schedules."""
        batch,_,_,ci=self.window.input_shape
        oh,ow=self.window.output_spatial
        axes={"output_x":(ow,self.row_tile),"output_channel":(self.output_channels,self.column_tile),
              "reduction_channel":(ci,self.channel_tile)}
        partitions={name:{"extent":extent,"tile":tile,"whole":extent//tile,"tail":extent%tile}
                    for name,(extent,tile)in axes.items()}
        combinations=prod(int(extent//tile>0)+int(extent%tile>0)for extent,tile in axes.values())
        return {"partitions":partitions,"static_tile_families":combinations,
                "output_elements":batch*oh*ow*self.output_channels,
                "reduction_products":batch*oh*ow*self.output_channels*prod(self.window.kernel)*ci,
                "reduction_order":["kernel_y","kernel_x","input_channel"],
                "completion":"synchronous target callbacks before private-row reuse",
                "host_payload_operations":0}


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
    identity=plan.identity
    if any(not callable(getattr(builder,n,None))for n in ("op","cmp","loop","branch"))or not callable(lower):
        raise ValueError("counted convolution needs an index/control builder and target callback")
    window=plan.window
    batch,_,_,_=window.input_shape
    oh,ow=window.output_spatial
    def op(name,a,b):return builder.op(name,a,b)
    def split(extent,tile,body):
        whole,tail=divmod(extent,tile)
        if whole:
            builder.loop(0,whole,1,lambda i:body(op("mul",i,tile),tile))
        if tail:
            body(whole*tile,tail)
    def batch_body(image):
        def y_body(y):
            def x_body(x,rows):
                row=op("add",op("mul",op("add",op("mul",image,oh),y),ow),x)
                def n_body(column,columns):
                    emit_convolution_tile(plan, builder, lower, output_row=row, output_column=column,
                        rows=rows, columns=columns, accumulator_row=plan.accumulator_row,
                        synchronous_completion=True)
                split(plan.output_channels,plan.column_tile,n_body)
            split(ow,plan.row_tile,x_body)
        builder.loop(0,oh,1,y_body)
    builder.loop(0,batch,1,batch_body)
    plan.__post_init__()
    if plan.identity!=identity:
        raise ValueError("counted convolution source/resources changed during emission")
    return dict(plan_sha256=identity,**plan.coverage,
                scope="emitted source partitions; target effects and execution qualification remain separate")


def emit_convolution_tile(plan, builder, lower, *, output_row, output_column,
                          rows, columns, accumulator_row, synchronous_completion):
    """Emit one original convolution rectangle into caller-assigned private rows.

    This is the same source-ordered reduction and checked input DMA predicate as
    the whole-image emitter. The caller proves the global row/column rectangle,
    complete invocation coverage, final observations and assigned-row lifetime.
    No output copy is implicit: the final ``store`` effect remains caller-owned.
    """
    if type(plan) is not CountedConvolutionPlan or synchronous_completion is not True:
        raise ValueError("convolution tile requires an exact plan and synchronous realization")
    return _emit_convolution_tile_effects(plan, builder, lower, output_row=output_row,
        output_column=output_column, rows=rows, columns=columns, accumulator_row=accumulator_row)


def _emit_convolution_tile_effects(plan, builder, lower, *, output_row, output_column,
                                    rows, columns, accumulator_row, output_tiles=None):
    """Canonical source partition; completion is owned by the public wrapper."""
    plan.__post_init__()
    if (any(type(x) is not int or x <= 0 for x in (rows, columns))
            or rows > plan.row_tile or columns > plan.column_tile):
        raise ValueError("convolution rectangle exceeds its declared primitive tile")
    window = plan.window
    _, _, _, ci = window.input_shape
    kh, kw = window.kernel

    def op(name, a, b):
        return builder.op(name, a, b)

    common = dict(output_row=output_row, output_column=output_column,
                  rows=rows, columns=columns, accumulator_row=accumulator_row)
    outputs = (common,) if output_tiles is None else tuple(
        dict(output_row=output_row, rows=rows, output_column=t.output_column,
             columns=t.columns, accumulator_row=t.accumulator_row) for t in output_tiles)
    if plan.bias:
        for output in outputs:
            lower("initialize_bias", output)

    def ky_body(ky):
        def kx_body(kx):
            def channel_body(channel, depth):
                run = window.emit_row_run(builder, output_row, rows, ky, kx, channel, depth)

                def transfer(first, stride, local, count, zero):
                    lower("load_input", dict(first_element=first, row_stride_elements=stride,
                        local_row=local, rows=count, columns=depth, zero=zero))

                def interior():
                    transfer(run["first_element"], run["row_stride_elements"], plan.lhs_row, rows, False)

                def boundary():
                    def one(index):
                        item = window.emit_row_run(builder, op("add", output_row, index), 1,
                                                   ky, kx, channel, depth)
                        local = op("add", plan.lhs_row, index)
                        builder.branch(item["predicate"],
                            lambda: transfer(item["first_element"], item["row_stride_elements"], local, 1, False),
                            lambda: transfer(0, 0, local, 1, True))
                    builder.loop(0, rows, 1, one)

                builder.branch(run["predicate"], interior, boundary)
                reduction = op("add", op("mul", op("add", op("mul", ky, kw), kx), ci), channel)
                first = builder.op("and", builder.cmp("eq", ky, 0), builder.cmp("eq", kx, 0), "i1")
                first = builder.op("and", first, builder.cmp("eq", channel, 0), "i1")
                for output in outputs:
                    lower("load_weight", dict(reduction_row=reduction,
                        output_column=output["output_column"], local_row=plan.rhs_row,
                        rows=depth, columns=output["columns"]))
                    lower("compute", dict(output, lhs_row=plan.lhs_row, rhs_row=plan.rhs_row,
                                          depth=depth, first_reduction=first, bias=plan.bias))
            whole, tail = divmod(ci, plan.channel_tile)
            if whole:
                builder.loop(0, whole, 1, lambda index:
                             channel_body(op("mul", index, plan.channel_tile), plan.channel_tile))
            if tail:
                channel_body(whole * plan.channel_tile, tail)
        builder.loop(0, kw, 1, kx_body)

    builder.loop(0, kh, 1, ky_body)
    for output in outputs:
        lower("store", output)


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


def emit_convolution_load_issue_tile(
    plan, builder, issue, complete, *, output_row, output_column, rows, columns, accumulator_row, law
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


@dataclass(frozen=True)
class ConvolutionOutputTile:
    """One static original output-channel interval and its private live ACC rows."""

    output_column: int
    columns: int
    accumulator_row: int


@dataclass(frozen=True)
class ConvolutionInputRetentionLaw:
    """Caller-bound target premises for reusing one loaded input across outputs.

    Every callback completes synchronously. RHS loads, compute and stores must
    preserve the input rows. External inputs are immutable and mapped; outputs
    are disjoint. Independent channel reductions may interleave, but each keeps
    its original K order and every original final observation. These premises
    do not confer execution qualification on a target.
    """

    hardware_identity: str
    numerical_policy_identity: str
    source_identity: str
    qualification_identity: str
    max_live_tiles: int
    input_preserved: bool = False
    independent_outputs: bool = False
    immutable_mapped_disjoint_buffers: bool = False

    def __post_init__(self):
        if (any(type(x) is not str or not x.strip() for x in (
                self.hardware_identity, self.numerical_policy_identity,
                self.source_identity, self.qualification_identity))
            or type(self.max_live_tiles) is not int or not 0 < self.max_live_tiles < 2**31
            or any(type(x) is not bool for x in (self.input_preserved,
                self.independent_outputs, self.immutable_mapped_disjoint_buffers))):
            raise ValueError("convolution retention requires typed bounded original target premises")


def emit_convolution_column_block(plan, builder, lower, *, output_row, rows,
                                  output_tiles, law, synchronous_completion):
    """Reuse canonical window input across independent private output tiles.

    The caller retains global invocation coverage. This entry only interleaves
    the listed contiguous output intervals; each consumes the unchanged ordered
    kh/kw/channel reduction and has its own nonoverlapping ACC lifetime through
    final store. No callback completion, host preparation or input copy is
    inferred. There is no implicit asynchronous execution grant.
    """
    if synchronous_completion is not True:
        raise ValueError("convolution retention requires original plan/law and synchronous realization")
    return _emit_convolution_column_block(plan, builder, lower, output_row=output_row,
        rows=rows, output_tiles=output_tiles, law=law)


def _emit_convolution_column_block(plan, builder, lower, *, output_row, rows, output_tiles, law):
    if (type(plan) is not CountedConvolutionPlan
        or type(law) is not ConvolutionInputRetentionLaw):
        raise ValueError("convolution retention requires original plan/law and synchronous realization")
    plan.__post_init__()
    law.__post_init__()
    if (not all((law.input_preserved, law.independent_outputs,
                 law.immutable_mapped_disjoint_buffers))
        or not callable(lower)
        or any(not callable(getattr(builder, n, None)) for n in ("op", "cmp", "loop", "branch"))):
        raise ValueError("convolution retention lacks complete source/effect premises")
    if (type(output_tiles) is not tuple or not output_tiles
        or len(output_tiles) > law.max_live_tiles
        or type(rows) is not int or not 0 < rows <= plan.row_tile):
        raise ValueError("convolution retention exceeds its live tile/row budget")
    previous_end = None
    spans = []
    for tile in output_tiles:
        if (type(tile) is not ConvolutionOutputTile
            or any(type(x) is not int for x in (tile.output_column, tile.columns, tile.accumulator_row))
            or not 0 < tile.columns <= plan.column_tile
            or tile.output_column < 0 or tile.output_column + tile.columns > plan.output_channels
            or tile.accumulator_row < 0 or tile.accumulator_row + rows > plan.accumulator_rows
            or (previous_end is not None and previous_end != tile.output_column)):
            raise ValueError("convolution retention requires contiguous original output intervals and bounded ACC rows")
        span = (tile.accumulator_row, tile.accumulator_row + rows)
        if any(span[0] < end and start < span[1] for start, end in spans):
            raise ValueError("convolution retention has overlapping live ACC rows")
        spans.append(span)
        previous_end = tile.output_column + tile.columns
    before = plan.identity, asdict(law), tuple(asdict(t) for t in output_tiles)
    first = output_tiles[0]
    _emit_convolution_tile_effects(plan, builder, lower, output_row=output_row,
        output_column=first.output_column, rows=rows, columns=first.columns,
        accumulator_row=first.accumulator_row, output_tiles=output_tiles)
    plan.__post_init__()
    law.__post_init__()
    if before != (plan.identity, asdict(law), tuple(asdict(t) for t in output_tiles)):
        raise ValueError("original convolution retention source/resources changed during emission")
    return dict(schema="merlin.convolution_column_retention.v1", plan_sha256=plan.identity,
        law=asdict(law), outputs=[asdict(t) for t in output_tiles], live_accumulator_rows=spans,
        input_transfer="once per original kh/kw/channel chunk for this output block",
        source_order="unchanged per-output kernel_y/kernel_x/input_channel",
        completion="synchronous target callbacks before private-row reuse",
        execution_qualification="independent original target obligation", timing="UNKNOWN")


def emit_convolution_load_issue_column_block(
    plan, builder, issue, complete, *, output_row, rows, output_tiles, law, issue_law
):
    """Compose original input retention and a bounded load/compute issue epoch.

    This separate entry preserves both parents' admission. The first epoch may
    contain one bias load per live output, the original boundary-row input loads
    and one RHS load. Actual selected range dependencies delay paired compute;
    full completion after each compute/store precedes RHS or output reuse. The
    next output tile may reuse LHS only under the independent retention law.
    """
    if (type(plan) is not CountedConvolutionPlan or type(output_tiles) is not tuple
        or not output_tiles):
        raise ValueError("convolution load retention requires original plan and output tiles")
    before = _validate_convolution_load_issue(
        plan, builder, issue, complete, rows, issue_law, len(output_tiles)*int(plan.bias)
    )

    def lower(kind, fields):
        issue(kind, fields)
        if kind in ("compute", "store"):
            complete(kind, fields)

    result = _emit_convolution_column_block(plan, builder, lower, output_row=output_row,
        rows=rows, output_tiles=output_tiles, law=law)
    plan.__post_init__()
    issue_law.__post_init__()
    if before != (plan.identity, asdict(issue_law)):
        raise ValueError("original convolution issue source or target premises changed during emission")
    return dict(result, schema="merlin.convolution_load_issue_column_retention.v1",
        issue_law=asdict(issue_law),
        completion="full device drain after each paired compute and each store")


@dataclass(frozen=True)
class ConvolutionComputeEpochLaw:
    """Additional selected source law for operand reuse across issued computes.

    This is not inferred from a load-issue law or storage capacity. Actual local
    RAW/WAR/WAW checks, accumulator result visibility and operand consumption
    retirement must govern accepted commands. Configuration is captured in its
    original lane order, every software compute emits its descriptor/data pair,
    and finite queues backpressure without making that pair unable to progress.
    Full completion remains real at epoch boundaries and every original store.
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

    def __post_init__(self):
        identities = (self.hardware_identity, self.numerical_policy_identity,
                      self.source_identity, self.qualification_identity)
        flags = (self.local_read_after_write, self.local_write_after_read,
                 self.local_write_after_write, self.accumulator_result_visibility,
                 self.operand_consumption_retirement, self.ordered_configuration_capture,
                 self.adjacent_descriptor_compute, self.finite_queue_pair_progress,
                 self.complete_device_drain, self.exclusive_private_state)
        if (any(type(v) is not str or not v.strip() for v in identities)
            or type(self.max_compute_pairs) is not int or not 1 < self.max_compute_pairs < 2**31
            or any(type(v) is not bool for v in flags)):
            raise ValueError("convolution compute epochs require bounded typed original source premises")

    def require(self, pairs):
        self.__post_init__()
        if (type(pairs) is not int or not 1 < pairs <= self.max_compute_pairs
            or not all((self.local_read_after_write, self.local_write_after_read,
                        self.local_write_after_write, self.accumulator_result_visibility,
                        self.operand_consumption_retirement, self.ordered_configuration_capture,
                        self.adjacent_descriptor_compute, self.finite_queue_pair_progress,
                        self.complete_device_drain, self.exclusive_private_state))):
            raise ValueError("convolution compute epoch lacks complete dependency/progress premises or budget")


def emit_convolution_compute_epoch_column_block(
    plan, builder, issue, complete, *, output_row, rows, output_tiles, law, issue_law, epoch_law, compute_pairs
):
    """Emit runtime-bounded complete compute pairs, preserving original stores.

    The actual source K coordinates determine a one-based pair index. A real
    runtime branch drains on multiples of compute_pairs, across nested kh/kw/ci
    iterations and output columns. A Python emission counter is never used.
    Every store is followed by full completion, closing any partial final epoch
    before ACC/RHS/input ownership changes outside this block. The stronger
    original target law must protect all intermediate physical reuse.
    """
    if (type(plan) is not CountedConvolutionPlan or type(output_tiles) is not tuple
        or not output_tiles or type(epoch_law) is not ConvolutionComputeEpochLaw):
        raise ValueError("convolution compute epochs require original plan, output tiles and stronger source law")
    epoch_law.require(compute_pairs)
    before_issue = _validate_convolution_load_issue(
        plan, builder, issue, complete, rows, issue_law, len(output_tiles)*int(plan.bias)
    )
    if (epoch_law.hardware_identity != issue_law.hardware_identity
        or epoch_law.numerical_policy_identity != issue_law.numerical_policy_identity):
        raise ValueError("convolution compute epoch and issue laws select different contracts")
    before_epoch = asdict(epoch_law)
    ci = plan.window.input_shape[-1]
    chunks = (ci+plan.channel_tile-1)//plan.channel_tile
    positions = {t.output_column: n for n, t in enumerate(output_tiles)
                 if type(t) is ConvolutionOutputTile}
    pending = None

    def lower(kind, fields):
        nonlocal pending
        if kind == "load_weight":
            if pending is not None:
                raise ValueError("convolution epoch changed the original weight/compute pairing")
            pending = fields["reduction_row"], fields["output_column"]
        issue(kind, fields)
        if kind == "compute":
            if pending is None or pending[1] != fields["output_column"]:
                raise ValueError("convolution epoch lost original reduction coordinates")
            reduction, column = pending
            kernel = builder.op("udiv", reduction, ci)
            channel = builder.op("urem", reduction, ci)
            chunk = builder.op("add", builder.op("mul", kernel, chunks),
                               builder.op("udiv", channel, plan.channel_tile))
            pair = builder.op("add", builder.op("mul", chunk, len(output_tiles)), positions[column]+1)
            boundary = builder.cmp("eq", builder.op("urem", pair, compute_pairs), 0)
            builder.branch(boundary, lambda: complete(kind, fields), lambda: None)
            pending = None
        elif kind == "store":
            if pending is not None:
                raise ValueError("convolution epoch reaches a store with an unmatched descriptor")
            complete(kind, fields)

    result = _emit_convolution_column_block(plan, builder, lower, output_row=output_row,
        rows=rows, output_tiles=output_tiles, law=law)
    if (pending is not None or before_issue != (plan.identity, asdict(issue_law))
        or before_epoch != asdict(epoch_law)):
        raise ValueError("original convolution epoch source/target premises changed during emission")
    return dict(result, schema="merlin.convolution_compute_epoch.v1", issue_law=asdict(issue_law),
        epoch_law=asdict(epoch_law), compute_pairs=compute_pairs,
        epoch_index="((ky*kw+kx)*ceil(ci/channel_tile)+channel/channel_tile)*output_tiles+column_slot+1",
        completion="real runtime modulo boundary after complete pairs; full completion after every store")
