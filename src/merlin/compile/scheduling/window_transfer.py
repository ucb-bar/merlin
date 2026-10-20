"""Prove a convolution window's rows form one in-bounds affine DMA rectangle.

This target-neutral index calculation supplies only a predicate, first logical
element and row stride. The caller retains its scalar gather for every rejected
run and owns the physical transfer, element width and target qualification.
"""

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
