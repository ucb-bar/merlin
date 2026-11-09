"""Original convolution window geometry and counted coverage."""

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
        for values, count, zero in (
            (self.input_shape, 4, False),
            (self.kernel, 2, False),
            (self.stride, 2, False),
            (self.dilation, 2, False),
            (self.padding, 4, True),
        ):
            if (
                type(values) is not tuple
                or len(values) != count
                or any(type(x) is not int or x < (0 if zero else 1) or x >= 2**63 for x in values)
            ):
                raise ValueError("window transfer requires explicit bounded integer geometry")
        batch, height, width, channels = self.input_shape
        kh, kw = self.kernel
        sh, sw = self.stride
        dh, dw = self.dilation
        top, left, bottom, right = self.padding
        oh = (height + top + bottom - dh * (kh - 1) - 1) // sh + 1
        ow = (width + left + right - dw * (kw - 1) - 1) // sw + 1
        if min(oh, ow) <= 0:
            raise ValueError("window transfer has no output rows")
        if (
            max(
                prod(self.input_shape),
                batch * oh * ow,
                channels * sw,
                oh * sh + dh * (kh - 1) + top,
                ow * sw + dw * (kw - 1) + left,
            )
            >= 2**63
        ):
            raise ValueError("window transfer geometry exceeds signed 64-bit address arithmetic")

    @property
    def output_spatial(self):
        _, height, width, _ = self.input_shape
        kh, kw = self.kernel
        sh, sw = self.stride
        dh, dw = self.dilation
        top, left, bottom, right = self.padding
        return (
            (height + top + bottom - dh * (kh - 1) - 1) // sh + 1,
            (width + left + right - dw * (kw - 1) - 1) // sw + 1,
        )

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
            return {"predicate": False, "first_element": 0, "row_stride_elements": channels * sw}
        image = builder.op("udiv", row, oh * ow)
        y = builder.op("urem", builder.op("udiv", row, ow), oh)
        x = builder.op("urem", row, ow)
        iy = builder.op("add", builder.op("sub", builder.op("mul", y, sh), top), builder.op("mul", ky, dh))
        ix = builder.op("add", builder.op("sub", builder.op("mul", x, sw), left), builder.op("mul", kx, dw))
        last_x = builder.op("add", ix, (rows - 1) * sw)
        checks = (
            builder.cmp("ult", row, batch * oh * ow),
            builder.cmp("ule", x, ow - rows),
            builder.cmp("ult", ky, kh),
            builder.cmp("ult", kx, kw),
            builder.cmp("ule", channel, channels - columns),
            builder.cmp("ult", iy, height),
            builder.cmp("ult", ix, width),
            builder.cmp("ult", last_x, width),
        )
        predicate = checks[0]
        for value in checks[1:]:
            predicate = builder.op("and", predicate, value, "i1")
        first = builder.op("add", builder.op("mul", image, height), iy)
        first = builder.op("add", builder.op("mul", first, width), ix)
        first = builder.op("add", builder.op("mul", first, channels), channel)
        return {"predicate": predicate, "first_element": first, "row_stride_elements": channels * sw}


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
        positive = (
            self.output_channels,
            self.primitive,
            self.row_tile,
            self.column_tile,
            self.channel_tile,
            self.operand_rows,
            self.accumulator_rows,
        )
        if any(type(v) is not int or not 0 < v < 2**63 for v in positive):
            raise ValueError("counted convolution requires bounded positive resources and extents")
        if any(type(v) is not int or v < 0 for v in (self.lhs_row, self.rhs_row, self.accumulator_row)):
            raise ValueError("counted convolution requires nonnegative private row assignments")
        if type(self.bias) is not bool:
            raise ValueError("counted convolution bias source must be explicit")
        if max(self.row_tile, self.column_tile, self.channel_tile) > self.primitive:
            raise ValueError("convolution tile exceeds the declared primitive")
        if self.row_tile > self.window.output_spatial[1]:
            raise ValueError("counted input runs cannot cross an output-row boundary")
        if (
            self.lhs_row + self.row_tile > self.operand_rows
            or self.rhs_row + self.channel_tile > self.operand_rows
            or self.accumulator_row + self.row_tile > self.accumulator_rows
        ):
            raise ValueError("counted convolution private rows exceed declared capacity")
        if not (self.lhs_row + self.row_tile <= self.rhs_row or self.rhs_row + self.channel_tile <= self.lhs_row):
            raise ValueError("counted convolution operand lifetimes require disjoint private rows")
        batch, h, w, ci = self.window.input_shape
        oh, ow = self.window.output_spatial
        reduction = prod(self.window.kernel) * ci
        if (
            max(
                reduction * self.output_channels,
                4 * batch * oh * ow * self.output_channels,
                batch * oh * ow * self.output_channels * reduction,
            )
            >= 2**63
        ):
            raise ValueError("counted convolution address/product arithmetic exceeds signed i64")

    @property
    def identity(self):
        return hashlib.sha256(repr(self).encode()).hexdigest()

    @property
    def coverage(self):
        """Arithmetic partitions, not enumeration of shape-sized schedules."""
        batch, _, _, ci = self.window.input_shape
        oh, ow = self.window.output_spatial
        axes = {
            "output_x": (ow, self.row_tile),
            "output_channel": (self.output_channels, self.column_tile),
            "reduction_channel": (ci, self.channel_tile),
        }
        partitions = {
            name: {"extent": extent, "tile": tile, "whole": extent // tile, "tail": extent % tile}
            for name, (extent, tile) in axes.items()
        }
        combinations = prod(int(extent // tile > 0) + int(extent % tile > 0) for extent, tile in axes.values())
        return {
            "partitions": partitions,
            "static_tile_families": combinations,
            "output_elements": batch * oh * ow * self.output_channels,
            "reduction_products": batch * oh * ow * self.output_channels * prod(self.window.kernel) * ci,
            "reduction_order": ["kernel_y", "kernel_x", "input_channel"],
            "completion": "synchronous target callbacks before private-row reuse",
            "host_payload_operations": 0,
        }


# Preserve the original public lookup and pickle identities.
for _definition in (
    Window2D,
    CountedConvolutionPlan,
):
    _definition.__module__ = "merlin.compiler.schedule.window_transfer"
del _definition
