"""Prove a convolution window's rows form one in-bounds affine DMA rectangle.

This target-neutral index calculation supplies only a predicate, first logical
element and row stride. The caller retains its scalar gather for every rejected
run and owns the physical transfer, element width and target qualification.
"""

import hashlib
from dataclasses import dataclass
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
    if plan.bias:
        lower("initialize_bias", common)

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
                lower("load_weight", dict(reduction_row=reduction, output_column=output_column,
                    local_row=plan.rhs_row, rows=depth, columns=columns))
                first = builder.op("and", builder.cmp("eq", ky, 0), builder.cmp("eq", kx, 0), "i1")
                first = builder.op("and", first, builder.cmp("eq", channel, 0), "i1")
                lower("compute", dict(common, lhs_row=plan.lhs_row, rhs_row=plan.rhs_row,
                                      depth=depth, first_reduction=first, bias=plan.bias))
            whole, tail = divmod(ci, plan.channel_tile)
            if whole:
                builder.loop(0, whole, 1, lambda index:
                             channel_body(op("mul", index, plan.channel_tile), plan.channel_tile))
            if tail:
                channel_body(whole * plan.channel_tile, tail)
        builder.loop(0, kw, 1, kx_body)
    builder.loop(0, kh, 1, ky_body)
    lower("store", common)
