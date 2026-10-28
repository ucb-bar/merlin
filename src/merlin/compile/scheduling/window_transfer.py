"""Prove a convolution window's rows form one in-bounds affine DMA rectangle.

This target-neutral index calculation supplies only a predicate, first logical
element and row stride. The caller retains its scalar gather for every rejected
run and owns the physical transfer, element width and target qualification.
"""

import hashlib
from dataclasses import asdict, dataclass, field
from math import prod

from .affine_gather_runs import ConvolutionGatherRuns, emit_convolution_gather_runs


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
class ConvolutionResidentInputRows:
    """Injective logical NHWC row cache into caller-owned tiled operand rows.

    A source word x*CI+c maps to row (c//tile)*row_pitch+left+x and
    lane c%tile within its ring slot. Unused row-pitch holes are never read.
    Explicit padded bindings stage the original integer-zero halo. Rolling-window
    ownership proves the original row version; this map supplies no transfer,
    instruction or completion authority. Horizontal stride1 keeps each compute
    input row contiguous. Other strides require an explicit compute_row_step
    binding; the target must independently implement this source-cell sampling.
    Optional stage_channels bounds actual transfer width without changing the
    source CI stride or physical channel-block map; no target limit is inferred.
    """
    window: Window2D
    source_identity: str
    channel_tile: int
    base: int
    row_pitch: int
    slots: int
    private_first: int
    private_rows: int
    stage_rows: int
    halo_value: int | None = None
    stage_channels: int | None = None
    compute_row_step: int | None = None
    _source_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, '_source_sha256', hashlib.sha256(repr(self).encode()).hexdigest())

    def _validate(self):
        if type(self.window) is not Window2D:
            raise ValueError("resident input rows require the exact original window")
        self.window.__post_init__()
        if (type(self.source_identity) is not str or len(self.source_identity) != 64
            or any(c not in '0123456789abcdef' for c in self.source_identity)):
            raise ValueError("resident input rows require source-bound windows")
        if (self.compute_row_step is None and self.window.stride[1] != 1
            or self.compute_row_step is not None and (
                type(self.compute_row_step) is not int or self.compute_row_step != self.window.stride[1])):
            raise ValueError("resident horizontal stride requires an exact explicit compute row-step binding")
        # CountedConvolutionPlan defines the original signed-integer padding
        # as zero. A different affine quantized zero point is not that source.
        if (any(self.window.padding) and (type(self.halo_value) is not int or self.halo_value != 0)
            or self.halo_value is not None and (type(self.halo_value) is not int or self.halo_value != 0)):
            raise ValueError("padded resident rows require explicit original integer-zero halo value")
        if (any(type(v) is not int or v < 0 for v in (self.base, self.private_first))
            or any(type(v) is not int or not 0 < v < 1 << 63 for v in
                   (self.channel_tile, self.row_pitch, self.slots, self.private_rows, self.stage_rows))
            or self.row_pitch < self.padded_width or self.stage_rows > self.row_pitch
            or self.slots < (self.window.kernel[0]-1)*self.window.dilation[0]+1
            or self.base < self.private_first or self.end > self.private_first+self.private_rows
            or self.end*self.channel_tile >= 1 << 63):
            raise ValueError("resident input row layout exceeds its explicit private footprint or transport bounds")
        if (self.stage_channels is not None and (type(self.stage_channels) is not int
            or not 0 < self.stage_channels < 1 << 63 or self.stage_channels % self.channel_tile)):
            raise ValueError("resident channel staging requires an explicit positive whole-channel-block limit")

    @property
    def padded_width(self):
        return self.window.input_shape[2]+self.window.padding[1]+self.window.padding[3]

    @property
    def row_step(self):
        """Physical rows between consecutive output samples, not a target encoding."""
        return 1 if self.compute_row_step is None else self.compute_row_step

    @property
    def channel_blocks(self):
        return (self.window.input_shape[-1]+self.channel_tile-1)//self.channel_tile

    @property
    def rows_per_source_row(self):
        return self.row_pitch*self.channel_blocks

    @property
    def end(self):
        return self.base+self.slots*self.rows_per_source_row

    @property
    def initial_load_commands(self):
        return self.stage_command_bound((self.window.kernel[0]-1)*self.window.dilation[0]+1)

    def stage_command_bound(self, source_rows):
        """Bound emitted row/halo/channel transfers, independently of queue slots."""
        if type(source_rows) is not int or not 0 < source_rows <= self.slots:
            raise ValueError("resident staging bound needs actual private source rows")
        _, left, _, right = self.window.padding
        def chunks(count):
            return (count+self.stage_rows-1)//self.stage_rows
        per_row = max(chunks(left)+chunks(self.window.input_shape[2])+chunks(right), chunks(self.padded_width))
        ci = self.window.input_shape[-1]
        channel_chunks = 1 if self.stage_channels is None else (ci+self.stage_channels-1)//self.stage_channels
        return per_row*source_rows*channel_chunks

    def revalidate(self):
        self._validate()
        if hashlib.sha256(repr(self).encode()).hexdigest() != self._source_sha256:
            raise ValueError("original resident input layout changed")

    def rolling_plan(self):
        from .program_limits import AddressWindow
        from .rolling_window import RollingWindowPlan
        self.revalidate()
        oh, ow = self.window.output_spatial
        pitch = self.rows_per_source_row*self.channel_tile
        return RollingWindowPlan(self.window, self.source_identity, 0, ow, 0, oh,
            AddressWindow('logical_resident_input', 0, self.slots*pitch), self.slots, pitch)

    def physical_cell(self, cache_byte):
        """Exact static map used by independent cell/layout checks."""
        self.revalidate()
        pitch = self.rows_per_source_row*self.channel_tile
        ci = self.window.input_shape[-1]
        if type(cache_byte) is not int or not 0 <= cache_byte < self.slots*pitch:
            raise ValueError("resident input cache byte outside original footprint")
        slot, within = divmod(cache_byte, pitch)
        if within >= self.window.input_shape[2]*ci:
            raise ValueError("resident input padding hole has no source word")
        x, c = divmod(within, ci)
        return (self.base+slot*self.rows_per_source_row+(c//self.channel_tile)*self.row_pitch
                + self.window.padding[1]+x, c%self.channel_tile)

    def emit_row(self, builder, output_row, ky, kx, channel):
        oh, ow = self.window.output_spatial
        y = builder.op('urem', builder.op('udiv', output_row, ow), oh)
        x = builder.op('urem', output_row, ow)
        return self.emit_row_coordinates(builder, y, x, ky, kx, channel)

    def emit_row_coordinates(self, builder, output_y, output_x, ky, kx, channel):
        """Map original coordinates directly, without decoding a logical byte.

        The checked source consumer owns the output/tap domain and matching
        consume/complete boundary. Coordinates may be hoisted out of reduction
        and output-channel loops. Padded rows use the existing staged zero halo;
        this neither changes source samples nor grants early slot reuse.
        """
        self.revalidate()
        # Physical columns already contain the original left halo, cancelling
        # the source's -left. Ring slots use unbiased y even for zero halo rows.
        slot = builder.op('urem', builder.op('add',
            builder.op('mul', output_y, self.window.stride[0]),
            builder.op('mul', ky, self.window.dilation[0])), self.slots)
        sample_x = output_x if self.window.stride[1] == 1 else builder.op(
            'mul', output_x, self.window.stride[1])
        column = builder.op('add', sample_x, builder.op('mul', kx, self.window.dilation[1]))
        chunk = builder.op('udiv', channel, self.channel_tile)
        return builder.op('add', self.base, builder.op('add', builder.op('mul', slot, self.rows_per_source_row),
            builder.op('add', builder.op('mul', chunk, self.row_pitch), column)))

    def emit_stage(self, builder, fields, lower):
        """Bind the checked logical stage row through its exact tiled cell map."""
        self.revalidate()
        width, ci = self.window.input_shape[2:]
        slot = builder.op('udiv', fields['cache_byte'], self.rows_per_source_row*self.channel_tile)
        local = builder.op('add', self.base, builder.op('mul', slot, self.rows_per_source_row))
        if self.stage_channels is not None and self.stage_channels < ci:
            original_lower = lower
            def split_channels(kind, args):
                def part(channel, depth):
                    value = dict(args, columns=depth, local_row=builder.op('add', args['local_row'],
                        builder.op('mul', builder.op('udiv', channel, self.channel_tile), self.row_pitch)))
                    if kind == 'stage_input_rows':
                        value['first_element'] = builder.op('add', args['first_element'], channel)
                    original_lower(kind, value)
                whole, tail = divmod(ci, self.stage_channels)
                builder.loop(0, whole, 1, lambda index:
                    part(builder.op('mul', index, self.stage_channels), self.stage_channels))
                if tail:
                    part(whole*self.stage_channels, tail)
            lower = split_channels
        if not any(self.window.padding):
            def part(x, count):
                lower('stage_input_rows', dict(first_element=builder.op('add', fields['source_byte'],
                    builder.op('mul', x, ci)), local_row=builder.op('add', local, x), rows=count, columns=ci,
                    row_stride_elements=ci, block_stride_rows=self.row_pitch))
            def valid():
                whole, tail = divmod(width, self.stage_rows)
                builder.loop(0, whole, 1,
                    lambda index: part(builder.op('mul', index, self.stage_rows), self.stage_rows))
                if tail:
                    part(whole*self.stage_rows, tail)
            builder.branch(fields['valid'], valid, lambda: None)
            return
        left, right = self.window.padding[1], self.window.padding[3]

        def pieces(count, emit):
            whole, tail = divmod(count, self.stage_rows)
            if whole:
                builder.loop(0, whole, 1,
                    lambda index: emit(builder.op('mul', index, self.stage_rows), self.stage_rows))
            if tail:
                emit(whole*self.stage_rows, tail)

        def fill(first, count):
            if not count:
                return
            def part(x, rows):
                lower('fill_input_rows', dict(value=self.halo_value,
                    local_row=builder.op('add', local, builder.op('add', first, x)), rows=rows, columns=ci,
                    block_stride_rows=self.row_pitch))
            pieces(count, part)

        def valid():
            fill(0, left)
            def part(x, count):
                lower('stage_input_rows', dict(first_element=builder.op('add', fields['source_byte'],
                    builder.op('mul', x, ci)), local_row=builder.op('add', local, builder.op('add', left, x)),
                    rows=count, columns=ci, row_stride_elements=ci, block_stride_rows=self.row_pitch))
            pieces(width, part)
            fill(left+width, right)

        if any(self.window.padding):
            builder.branch(fields['valid'], valid, lambda: fill(0, self.padded_width))
        else:
            builder.branch(fields['valid'], valid, lambda: None)

    def record(self):
        self.revalidate()
        return dict(schema='merlin.convolution_resident_input_rows.v1', source_sha256=self._source_sha256,
            layout=asdict(self), private_row_interval=[self.base, self.end],
            compute_read_row_step=self.row_step,
            mapping=('slot*rows_per_source_row + (channel//tile)*row_pitch + left+x; lane=channel%tile; '
                'halo uses original padded coordinates and explicit original integer-zero fill'
                + ('; compute source_x=output_x*stride_x+kx*dilation_x-left; '
                   'sample r reads base_compute_row+r*stride_x' if self.row_step != 1 else '')),
            qualification=('exact source-cell and lifetime map; target transfer, row-step and '
                'full completion remain separate'))


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
                          rows, columns, accumulator_row, synchronous_completion, gather_runs=None, input_rows=None):
    """Emit one original convolution rectangle into caller-assigned private rows.

    This is the same source-ordered reduction and checked input DMA predicate as
    the whole-image emitter. The caller proves the global row/column rectangle,
    complete invocation coverage, final observations and assigned-row lifetime.
    No output copy is implicit: the final ``store`` effect remains caller-owned.
    """
    if type(plan) is not CountedConvolutionPlan or synchronous_completion is not True:
        raise ValueError("convolution tile requires an exact plan and synchronous realization")
    return _emit_convolution_tile_effects(plan, builder, lower, output_row=output_row,
        output_column=output_column, rows=rows, columns=columns, accumulator_row=accumulator_row,
        gather_runs=gather_runs, input_rows=input_rows)


def _emit_convolution_tile_effects(plan, builder, lower, *, output_row, output_column,
                                    rows, columns, accumulator_row, output_tiles=None, gather_runs=None,
                                    input_rows=None):
    """Canonical source partition; completion is owned by the public wrapper."""
    plan.__post_init__()
    if (any(type(x) is not int or x <= 0 for x in (rows, columns))
            or rows > plan.row_tile or columns > plan.column_tile):
        raise ValueError("convolution rectangle exceeds its declared primitive tile")
    window = plan.window
    _, _, _, ci = window.input_shape
    kh, kw = window.kernel
    if input_rows is not None:
        if (type(input_rows) is not ConvolutionResidentInputRows or input_rows.window != window
            or input_rows.channel_tile != plan.channel_tile or gather_runs is not None
            or input_rows.end > plan.operand_rows
            or not (input_rows.end <= plan.rhs_row or input_rows.base >= plan.rhs_row+plan.channel_tile)):
            raise ValueError("resident input layout differs from original source or overlaps its RHS tile")
        input_rows.revalidate()
    if gather_runs is not None:
        expected = ConvolutionGatherRuns(window.input_shape,
            (window.input_shape[0], *window.output_spatial), window.stride,
            window.dilation, window.kernel, plan.row_tile)
        if (type(gather_runs) is not ConvolutionGatherRuns or gather_runs != expected
            or any(window.padding) or not callable(getattr(builder, "select", None))):
            raise ValueError("linear convolution gathers require the exact original valid window and select builder")

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

                lhs_row = plan.lhs_row
                if input_rows is not None:
                    lhs_row = input_rows.emit_row(builder, output_row, ky, kx, channel)
                elif gather_runs is None:
                    run = window.emit_row_run(builder, output_row, rows, ky, kx, channel, depth)
                    builder.branch(run["predicate"], interior, boundary)
                else:
                    _, _, width, channels = window.input_shape
                    dy, dx = window.dilation
                    tap = op("add", op("add", op("mul", ky, dy*width*channels),
                                            op("mul", kx, dx*channels)), channel)
                    for base, local, count in emit_convolution_gather_runs(gather_runs, builder, output_row, rows):
                        def fragment(base=base, local=local, count=count):
                            transfer(op("add", base, tap), gather_runs.source_row_stride,
                                     op("add", plan.lhs_row, local), count, False)
                        builder.branch(builder.cmp("ult", 0, count), fragment, lambda: None)
                reduction = op("add", op("mul", op("add", op("mul", ky, kw), kx), ci), channel)
                first = builder.op("and", builder.cmp("eq", ky, 0), builder.cmp("eq", kx, 0), "i1")
                first = builder.op("and", first, builder.cmp("eq", channel, 0), "i1")
                for output in outputs:
                    lower("load_weight", dict(reduction_row=reduction,
                        output_column=output["output_column"], local_row=plan.rhs_row,
                        rows=depth, columns=output["columns"]))
                    compute = dict(output, lhs_row=lhs_row, rhs_row=plan.rhs_row,
                                   depth=depth, first_reduction=first, bias=plan.bias)
                    if input_rows is not None and input_rows.row_step != 1:
                        compute["lhs_row_step"] = input_rows.row_step
                    lower("compute", compute)
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
    plan, builder, issue, complete, *, output_row, output_column, rows, columns, accumulator_row, law,
    gather_runs=None, input_rows=None
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
        gather_runs=gather_runs, input_rows=input_rows,
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
    if (type(plan) is not CountedConvolutionPlan or type(law) is not ConvolutionReadoutTransferLaw
        or any(type(v) is not str or not v.strip() for v in (law.hardware_identity,
            law.numerical_policy_identity, law.source_identity, law.qualification_identity))
        or any(type(v) is not int or not 0 < v < 2**31
               for v in (law.tile_columns, law.block_rows, law.max_columns))
        or any(type(v) is not bool or not v for v in
               (law.scaled_i8, law.exact_block_column_map, law.original_numerical_state))
        or law.source_identity != plan.identity or law.tile_columns != plan.column_tile
        or law.max_columns < law.tile_columns or type(rows) is not int
        or not 0 < rows <= min(law.block_rows, plan.row_tile)
        or type(output_tiles) is not tuple or not output_tiles
        or type(law.column_states) is not tuple or not 0 < len(law.column_states) <= 262144):
        raise ValueError("wide convolution readout needs original source, block map and scaled-i8 premises")
    end = 0
    for state in law.column_states:
        if (type(state) is not tuple or len(state) != 3 or type(state[0]) is not int
            or type(state[1]) is not int or state[0] != end or state[1] <= 0
            or type(state[2]) is not str or len(state[2]) != 64
            or any(c not in "0123456789abcdef" for c in state[2])):
            raise ValueError("readout states must cover every original column exactly")
        end += state[1]
    if end != plan.output_channels:
        raise ValueError("readout state roster differs from original output columns")
    groups, states, spans, end = [], [], [], None
    for tile in output_tiles:
        if (type(tile) is not ConvolutionOutputTile
            or any(type(v) is not int for v in (tile.output_column, tile.columns, tile.accumulator_row))
            or not 0 < tile.columns <= law.tile_columns or tile.output_column < 0
            or tile.output_column + tile.columns > plan.output_channels
            or end is not None and tile.output_column != end
            or tile.accumulator_row < 0 or tile.accumulator_row + rows > plan.accumulator_rows
            or any(tile.accumulator_row < b and a < tile.accumulator_row + rows for a, b in spans)):
            raise ValueError("wide readout requires original contiguous outputs and disjoint private ACC rows")
        selected = {key for start, count, key in law.column_states
                    if start < tile.output_column + tile.columns and tile.output_column < start + count}
        if len(selected) != 1:
            raise ValueError("original store crosses numerical readout states")
        state = selected.pop()
        if (groups and states[-1] == state and groups[-1][-1].columns == law.tile_columns
            and tile.accumulator_row == groups[-1][0].accumulator_row + len(groups[-1]) * law.block_rows
            and sum(t.columns for t in groups[-1]) + tile.columns <= law.max_columns):
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
        if (any(type(x) is not str or not x.strip() for x in (
                self.hardware_identity, self.numerical_policy_identity,
                self.source_identity, self.qualification_identity))
            or type(self.max_live_tiles) is not int or not 0 < self.max_live_tiles < 2**31
            or any(type(x) is not bool for x in (self.input_preserved,
                self.independent_outputs, self.immutable_mapped_disjoint_buffers,
                self.independent_output_rows, self.no_intermediate_observers))):
            raise ValueError("convolution retention requires typed bounded original target premises")


def emit_convolution_column_block(plan, builder, lower, *, output_row, rows,
                                  output_tiles, law, synchronous_completion, gather_runs=None, input_rows=None):
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
        rows=rows, output_tiles=output_tiles, law=law, gather_runs=gather_runs, input_rows=input_rows)


def _emit_convolution_column_block(plan, builder, lower, *, output_row, rows, output_tiles, law,
                                   gather_runs=None, input_rows=None):
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
        accumulator_row=first.accumulator_row, output_tiles=output_tiles,
        gather_runs=gather_runs, input_rows=input_rows)
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
    plan, builder, issue, complete, *, output_row, rows, output_tiles, law, issue_law, gather_runs=None, input_rows=None
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
        rows=rows, output_tiles=output_tiles, law=law, gather_runs=gather_runs, input_rows=input_rows)
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
        identities = (self.hardware_identity, self.numerical_policy_identity,
                      self.source_identity, self.qualification_identity)
        flags = (self.local_read_after_write, self.local_write_after_read,
                 self.local_write_after_write, self.accumulator_result_visibility,
                 self.operand_consumption_retirement, self.ordered_configuration_capture,
                 self.adjacent_descriptor_compute, self.finite_queue_pair_progress,
                 self.complete_device_drain, self.exclusive_private_state)
        if (any(type(v) is not str or not v.strip() for v in identities)
            or type(self.max_compute_pairs) is not int or not 1 < self.max_compute_pairs < 2**31
            or self.pending_compute_pairs is not None and (
                type(self.pending_compute_pairs) is not int or not 0 < self.pending_compute_pairs < 2**31)
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
        if (any(type(v) is not str or not v.strip() for v in (
                self.hardware_identity, self.numerical_policy_identity,
                self.source_identity, self.qualification_identity))
            or any(type(v) is not int or not 0 < v < 2**31
                   for v in (self.max_issued_readouts, self.pending_readouts))
            or any(type(v) is not bool for v in (
                self.local_read_after_write, self.accumulator_result_visibility,
                self.ordered_configuration_capture, self.finite_queue_progress,
                self.complete_device_drain, self.exclusive_private_state))):
            raise ValueError("convolution readout epochs require typed bounded original source premises")

    def require(self, plan, rows, output_tiles):
        self.__post_init__()
        if (not all((self.local_read_after_write, self.accumulator_result_visibility,
                     self.ordered_configuration_capture, self.finite_queue_progress,
                     self.complete_device_drain, self.exclusive_private_state))
            or type(plan) is not CountedConvolutionPlan
            or type(rows) is not int or not 0 < rows <= plan.row_tile
            or type(output_tiles) is not tuple or not output_tiles
            or len(output_tiles) > self.max_issued_readouts):
            raise ValueError("convolution readout epoch lacks dependency/progress premises or budget")
        plan.__post_init__()
        spans = []
        end = None
        for tile in output_tiles:
            if (type(tile) is not ConvolutionOutputTile
                or any(type(v) is not int for v in (
                    tile.output_column, tile.columns, tile.accumulator_row))
                or not 0 < tile.columns <= plan.column_tile
                or tile.output_column < 0 or tile.output_column+tile.columns > plan.output_channels
                or end is not None and tile.output_column != end
                or tile.accumulator_row < 0 or tile.accumulator_row+rows > plan.accumulator_rows
                or any(tile.accumulator_row < hi and lo < tile.accumulator_row+rows for lo, hi in spans)):
                raise ValueError("convolution readout epoch requires disjoint original output and ACC intervals")
            spans.append((tile.accumulator_row, tile.accumulator_row+rows))
            end = tile.output_column+tile.columns


def emit_convolution_compute_epoch_column_block(
    plan, builder, issue, complete, *, output_row, rows, output_tiles, law, issue_law, epoch_law, compute_pairs,
    gather_runs=None, input_rows=None, readout_law=None, readout_transfer_law=None
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
    if readout_law is not None:
        if type(readout_law) is not ConvolutionReadoutEpochLaw:
            raise ValueError("convolution readout epoch requires its separate typed source law")
        readout_law.require(plan, rows, output_tiles)
        if (readout_law.hardware_identity != epoch_law.hardware_identity
            or readout_law.numerical_policy_identity != epoch_law.numerical_policy_identity):
            raise ValueError("convolution readout and compute epochs select different contracts")
    before_readout = None if readout_law is None else asdict(readout_law)
    transfers = None
    transfer_identity = None
    if readout_transfer_law is not None:
        if (type(readout_transfer_law) is not ConvolutionReadoutTransferLaw or readout_law is None
            or readout_transfer_law.hardware_identity != epoch_law.hardware_identity
            or readout_transfer_law.numerical_policy_identity != epoch_law.numerical_policy_identity):
            raise ValueError("wide readout requires its original readout epoch and same selected target")
        transfers = plan_convolution_readout_transfers(plan, output_tiles, readout_transfer_law, rows=rows)
        transfer_identity = readout_transfer_law.identity
    before_epoch = asdict(epoch_law)
    ci = plan.window.input_shape[-1]
    chunks = (ci+plan.channel_tile-1)//plan.channel_tile
    positions = {t.output_column: n for n, t in enumerate(output_tiles)
                 if type(t) is ConvolutionOutputTile}
    pending = None
    stores = 0
    held_stores = []
    transfer_index = 0
    total_pairs = plan.window.kernel[0]*plan.window.kernel[1]*chunks*len(output_tiles)

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
            chunk = builder.op("add", builder.op("mul", kernel, chunks),
                               builder.op("udiv", channel, plan.channel_tile))
            pair = builder.op("add", builder.op("mul", chunk, len(output_tiles)), positions[column]+1)
            boundary = builder.cmp("eq", builder.op("urem", pair, compute_pairs), 0)
            def drain_boundary():
                if readout_law is None:
                    complete(kind, fields)
                else:
                    # The pair index is actual runtime source K/column work,
                    # never a Python template counter or a queue-slot count.
                    builder.branch(builder.cmp("eq", pair, total_pairs),
                        lambda: None, lambda: complete(kind, fields))
            builder.branch(boundary, drain_boundary, lambda: None)
            pending = None
        elif kind == "store":
            if pending is not None:
                raise ValueError("convolution epoch reaches a store with an unmatched descriptor")
            if (stores >= len(output_tiles)
                or fields["output_column"] != output_tiles[stores].output_column):
                raise ValueError("convolution readout epoch changed original store order")
            stores += 1
            if transfers is not None:
                held_stores.append(dict(fields))
                group = transfers[transfer_index]
                if len(held_stores) == len(group):
                    first = held_stores[0]
                    if any(f["rows"] != rows or f["output_row"] is not first["output_row"]
                           and f["output_row"] != first["output_row"]
                           or (f["output_column"], f["columns"], f["accumulator_row"])
                           != (t.output_column, t.columns, t.accumulator_row)
                           for f, t in zip(held_stores, group, strict=True)):
                        raise ValueError("actual readout changed its checked original block mapping")
                    issue("store", dict(first, columns=sum(t.columns for t in group)))
                    held_stores.clear()
                    transfer_index += 1
            if readout_law is None or stores == len(output_tiles):
                complete(kind, fields)

    result = _emit_convolution_column_block(plan, builder, lower, output_row=output_row,
        rows=rows, output_tiles=output_tiles, law=law, gather_runs=gather_runs, input_rows=input_rows)
    if (pending is not None or stores != len(output_tiles) or held_stores
        or transfers is not None and (transfer_index != len(transfers)
            or readout_transfer_law.identity != transfer_identity)
        or before_issue != (plan.identity, asdict(issue_law))
        or before_epoch != asdict(epoch_law)
        or before_readout != (None if readout_law is None else asdict(readout_law))):
        raise ValueError("original convolution epoch source/target premises changed during emission")
    return dict(result, schema="merlin.convolution_compute_epoch.v1", issue_law=asdict(issue_law),
        epoch_law=asdict(epoch_law), compute_pairs=compute_pairs,
        epoch_index="((ky*kw+kx)*ceil(ci/channel_tile)+channel/channel_tile)*output_tiles+column_slot+1",
        readout_law=before_readout,
        **({} if transfers is None else dict(readout_transfers=dict(law_sha256=transfer_identity,
            original_stores=len(output_tiles), physical_stores=len(transfers),
            groups=[dict(output_column=g[0].output_column, columns=sum(t.columns for t in g),
                         accumulator_row=g[0].accumulator_row, original_tiles=len(g)) for g in transfers]))),
        completion=("real intermediate compute boundaries; producer/store RAW; "
                    "full completion after last original store"
                    if readout_law is not None else
                    "real runtime modulo boundary after complete pairs; full completion after every store"))
