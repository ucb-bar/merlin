"""Exact constant-stride runs for flattened valid convolution windows.

The plan splits at output-row boundaries. Each fragment therefore has one
constant input-row stride and one contiguous destination-row interval. It does
not select a device, command encoding, capacity or profitability policy.
"""

from dataclasses import dataclass
from math import prod

from .block_schedule import Geometry


@dataclass(frozen=True)
class ConvolutionGatherRuns:
    input_shape: tuple[int, int, int, int]
    output_spatial: tuple[int, int, int]
    stride: tuple[int, int]
    dilation: tuple[int, int]
    kernel: tuple[int, int]
    max_rows: int

    def __post_init__(self):
        for values, rank in ((self.input_shape, 4), (self.output_spatial, 3),
                             (self.stride, 2), (self.dilation, 2), (self.kernel, 2)):
            if (type(values) is not tuple or len(values) != rank
                    or any(type(v) is not int or not 0 < v < 1 << 63 for v in values)):
                raise ValueError("positive exact affine gather extents required")
        if type(self.max_rows) is not int or not 0 < self.max_rows < 1 << 63:
            raise ValueError("positive selected row capacity required")
        b, h, w, _ = self.input_shape
        kh, kw = self.kernel
        sh, sw = self.stride
        dh, dw = self.dilation
        expected = (b, (h - dh * (kh - 1) - 1) // sh + 1, (w - dw * (kw - 1) - 1) // sw + 1)
        if self.output_spatial != expected or prod(self.input_shape) >= 1 << 63:
            raise ValueError("affine gather requires the complete exact valid window and signed index bounds")
        ci = self.input_shape[3]
        if max(sw * ci, dh * w * ci, dw * ci) >= 1 << 63:
            raise ValueError("affine gather byte pitches exceed signed index bounds")

    @property
    def source_row_stride(self):
        return self.stride[1] * self.input_shape[3]

    def max_fragments(self, count):
        if type(count) is not int or not 0 < count <= self.max_rows:
            raise ValueError("gather row count exceeds its selected capacity")
        width = self.output_spatial[2]
        return min(count, 1 + (count + width - 2) // width)

    def fragments(self, first, count):
        """Exact static reference, including image, spatial and row tails."""
        self.max_fragments(count)
        if type(first) is not int or first < 0 or first + count > prod(self.output_spatial):
            raise ValueError("gather fragment exceeds the complete output spatial extent")
        _, h, w, ci = self.input_shape
        _, oh, ow = self.output_spatial
        sh, sw = self.stride
        result, used = [], 0
        while used < count:
            position = first + used
            image, spatial = divmod(position, oh * ow)
            y, x = divmod(spatial, ow)
            length = min(count - used, ow - x)
            offset = ((image * h + y * sh) * w + x * sw) * ci
            result.append((offset, used, length))
            used += length
        return tuple(result)


def emit_convolution_gather_runs(plan, builder, first, count, *, max_count=None):
    """Emit target-neutral integer index SSA using supplied arithmetic builders.

    Builders supply integer ``op``, ``cmp`` and ``select`` only. Zero-length
    inactive fragments may remain when a dynamic start needs fewer than the
    worst-case fragment count; command consumers must guard those lengths.
    Dynamic counts require an explicit static ``max_count`` from the caller's
    checked iteration domain. The caller proves ``0 <= count <= max_count``
    and ``first + count`` within the complete spatial extent; this index emitter
    does not establish those premises or grant a transfer's physical effects.
    """
    if type(plan) is not ConvolutionGatherRuns:
        raise ValueError("exact original affine gather plan required")
    bound = count if max_count is None else max_count
    fragments = plan.max_fragments(bound)
    if type(count) is int and not 0 < count <= bound:
        raise ValueError("gather row count exceeds its explicit bounded domain")
    _, h, w, ci = plan.input_shape
    _, oh, ow = plan.output_spatial
    sh, sw = plan.stride
    result, used = [], 0
    for _ in range(fragments):
        position = builder.op("add", first, used)
        image = builder.op("udiv", position, oh * ow)
        y = builder.op("urem", builder.op("udiv", position, ow), oh)
        x = builder.op("urem", position, ow)
        remaining = builder.op("sub", count, used)
        boundary = builder.op("sub", ow, x)
        length = builder.select(builder.cmp("ult", remaining, boundary), remaining, boundary)
        offset = builder.op("add", builder.op("mul", image, h * w * ci),
                            builder.op("add", builder.op("mul", y, sh * w * ci), builder.op("mul", x, sw * ci)))
        result.append((offset, used, length))
        used = builder.op("add", used, length)
    return tuple(result)


@dataclass(frozen=True)
class ConvolutionResidentPanel:
    """One complete-reduction RHS panel in a caller-owned private row epoch."""

    tile: int
    channel_tiles: int
    reduction_tiles: int
    column_blocks: int
    rhs_base: int
    rhs_rows: int
    accumulator_rows: int
    operand_capacity_rows: int
    accumulator_capacity_rows: int

    def __post_init__(self):
        if (any(type(v) is not int or not 0 < v < 1 << 63 for v in (
                self.tile, self.channel_tiles, self.reduction_tiles, self.column_blocks,
                self.rhs_base, self.rhs_rows, self.accumulator_rows,
                self.operand_capacity_rows, self.accumulator_capacity_rows))
                or self.reduction_tiles % self.channel_tiles
                or self.rhs_rows != self.reduction_tiles * self.column_blocks * self.tile
                or self.accumulator_rows != self.column_blocks * self.tile
                or self.rhs_base < self.tile or self.rhs_base % self.tile
                or self.rhs_base + self.rhs_rows > self.operand_capacity_rows
                or self.accumulator_rows > self.accumulator_capacity_rows):
            raise ValueError("resident convolution needs disjoint complete bounded tile windows")

    def rhs_row(self, tap, channel_tile, column_block, *, active_column_blocks=None):
        active = self.column_blocks if active_column_blocks is None else active_column_blocks
        if (any(type(v) is not int or v < 0 for v in (tap, channel_tile, column_block))
                or type(active) is not int or not 0 < active <= self.column_blocks
                or tap * self.channel_tiles + channel_tile >= self.reduction_tiles
                or channel_tile >= self.channel_tiles or column_block >= active):
            raise ValueError("resident convolution tile leaves its original panel")
        return self.rhs_base + ((tap * self.channel_tiles + channel_tile) * active + column_block) * self.tile


def plan_convolution_resident_panel(weight_shape, geometry, *, private_rows_owned=False, max_column_blocks=None):
    """Maximize complete-K column residency within explicit separate stores.

    One activation tile occupies rows [0,tile). RHS tiles have an injective
    reduction-major layout at the opposite end. Accumulator tiles remain live
    together until complete reductions store them. No operand/accumulator spill,
    hardware overlap or measured profitability is inferred.
    """
    if type(geometry) is not Geometry or private_rows_owned is not True:
        raise ValueError("resident convolution requires an explicit private resource epoch")
    if (type(weight_shape) is not tuple or len(weight_shape) != 4
            or any(type(v) is not int or not 0 < v < 1 << 63 for v in weight_shape)
            or prod(weight_shape) >= 1 << 63
            or any(type(v) is not int or not 0 < v < 1 << 63
                   for v in (geometry.block, geometry.operand_rows, geometry.accumulator_rows))
            or geometry.separate_accumulator_space is not True):
        raise ValueError("resident convolution requires positive geometry and separate accumulator rows")
    if max_column_blocks is not None and (type(max_column_blocks) is not int or max_column_blocks <= 0):
        raise ValueError("positive explicit resident column bound required")
    kh, kw, ci, co = weight_shape
    tile = geometry.block
    channel_tiles = (ci + tile - 1) // tile
    reduction_tiles = kh * kw * channel_tiles
    blocks = min((co + tile - 1) // tile, (geometry.operand_rows - tile) // (reduction_tiles * tile),
                 geometry.accumulator_rows // tile)
    if max_column_blocks is not None:
        blocks = min(blocks, max_column_blocks)
    if blocks <= 0:
        return None
    rhs_rows = reduction_tiles * blocks * tile
    rhs_base = ((geometry.operand_rows - rhs_rows) // tile) * tile
    if rhs_base < tile or rhs_base + rhs_rows > geometry.operand_rows or blocks * tile > geometry.accumulator_rows:
        raise ValueError("resident convolution layout exceeds its supplied resources")
    return ConvolutionResidentPanel(tile, channel_tiles, reduction_tiles, blocks, rhs_base, rhs_rows, blocks * tile,
                                    geometry.operand_rows, geometry.accumulator_rows)


def emit_convolution_matrix_gather(plan, builder, first, count, *, reduction_block, reduction_tile,
                                   max_count=None):
    """Map one checked logical im2col tile to exact original NHWC fragments.

    The logical matrix is (B*OH*OW, KH*KW*CI), in original ky/kx/channel
    order. A reduction tile must divide CI, so it cannot cross a source tap.
    The caller supplies the checked matrix iteration domain and owns original
    dtypes, all arithmetic/readout semantics, nonalias and physical row bounds.
    Each returned tuple is (source element, destination row, row count); issue
    consumers guard zero counts and retain their original completion contract.
    No host payload copy, target command or storage permission is introduced.
    """
    if type(plan) is not ConvolutionGatherRuns:
        raise ValueError("matrix gathers require the exact original valid window")
    plan.__post_init__()
    ci = plan.input_shape[-1]
    if type(reduction_tile) is not int or reduction_tile <= 0 or ci % reduction_tile:
        raise ValueError("matrix gather reduction tile must divide original input channels")
    _, _, width, _ = plan.input_shape
    _, kw = plan.kernel
    dh, dw = plan.dilation
    chunks = ci // reduction_tile
    if type(reduction_block) is int and not 0 <= reduction_block < plan.kernel[0]*kw*chunks:
        raise ValueError("matrix gather reduction block exceeds the original complete K domain")
    tap = builder.op("udiv", reduction_block, chunks)
    ky = builder.op("udiv", tap, kw)
    kx = builder.op("urem", tap, kw)
    channel = builder.op("mul", builder.op("urem", reduction_block, chunks), reduction_tile)
    offset = builder.op("add", builder.op("add", builder.op("mul", ky, dh*width*ci),
                             builder.op("mul", kx, dw*ci)), channel)
    return tuple((builder.op("add", base, offset), row, rows)
                 for base, row, rows in emit_convolution_gather_runs(
                     plan, builder, first, count, max_count=max_count))
