"""Private NHWC halo rows with source-congruent byte-copy addresses.

Logical padding and coordinates remain unchanged. Only private row pitch and
bounded leading slack change; no source alignment or out-of-range read is used.
"""

from dataclasses import dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import i1, i8, i64
from xdsl.ir import SSAValue

from merlin.compile.scheduling.window_transfer import Window2D

from .convolution_packing import _WindowBuilder
from .host_builder import PTR, FnBuilder


@dataclass(frozen=True)
class PitchedNHWCHalo:
    """An exact private byte layout for a complete original unit-window halo."""

    window: Window2D
    copy_alignment: int

    def __post_init__(self):
        if type(self.window) is not Window2D:
            raise ValueError("pitched halo needs the original typed source window")
        self.window.__post_init__()
        if self.window.kernel != (1, 1) or self.window.stride != (1, 1):
            raise ValueError("pitched halo needs complete unit-stride unit windows")
        a = self.copy_alignment
        if type(a) is not int or a <= 0 or a > 1 << 20 or a & (a - 1):
            raise ValueError("copy alignment must be a bounded positive power of two")
        if max(self.private_bytes, self.logical_bytes) >= 1 << 63:
            raise ValueError("pitched halo exceeds bounded signed-i64 byte coordinates")

    @property
    def source_row_bytes(self):
        return self.window.input_shape[2] * self.window.input_shape[3]

    @property
    def logical_row_bytes(self):
        return self.window.output_spatial[1] * self.window.input_shape[3]

    @property
    def row_pitch_bytes(self):
        return self.logical_row_bytes + (self.source_row_bytes - self.logical_row_bytes) % self.copy_alignment

    @property
    def image_pitch_bytes(self):
        return self.window.output_spatial[0] * self.row_pitch_bytes + self.copy_alignment - 1

    @property
    def private_bytes(self):
        return self.window.input_shape[0] * self.image_pitch_bytes

    @property
    def logical_image_bytes(self):
        return self.window.output_spatial[0] * self.logical_row_bytes

    @property
    def logical_bytes(self):
        return self.window.input_shape[0] * self.logical_image_bytes

    @property
    def payload_offset(self):
        pt, pl, _, _ = self.window.padding
        return pt * self.row_pitch_bytes + pl * self.window.input_shape[3]

    def physical_offset(self, source_address, allocation_address, logical_offset):
        """Concrete coordinate relation, including arbitrary original alignment."""
        self.__post_init__()
        if type(logical_offset) is not int or not 0 <= logical_offset < self.logical_bytes:
            raise ValueError("logical halo byte is outside the complete original domain")
        image, inner = divmod(logical_offset, self.logical_image_bytes)
        row, column = divmod(inner, self.logical_row_bytes)
        source_image_bytes = self.window.input_shape[1] * self.source_row_bytes
        base = image * self.image_pitch_bytes
        leading = (source_address + image * source_image_bytes - allocation_address - base
                   - self.payload_offset) & (self.copy_alignment - 1)
        return base + leading + row * self.row_pitch_bytes + column

    def summary(self):
        self.__post_init__()
        return dict(schema="merlin.pitched_nhwc_halo.v1",
                    window=dict(input_shape=self.window.input_shape, padding=self.window.padding),
                    copy_alignment=self.copy_alignment, logical_bytes=self.logical_bytes,
                    private_bytes=self.private_bytes, row_pitch_bytes=self.row_pitch_bytes,
                    image_pitch_bytes=self.image_pitch_bytes,
                    leading_slack_bytes=self.copy_alignment - 1,
                    source_alignment_assumed=False, source_overread_bytes=0)


def _image_base(builder, layout, source, allocation, image):
    ops = _WindowBuilder(builder)
    source_image_bytes = layout.window.input_shape[1] * layout.source_row_bytes
    source_offset = ops.op("mul", image, source_image_bytes)
    allocation_offset = ops.op("mul", image, layout.image_pitch_bytes)
    source_address = builder.add(llvm.PtrToIntOp(source, i64)).output
    allocation_address = builder.add(llvm.PtrToIntOp(allocation, i64)).output
    delta = ops.op("sub", ops.op("add", source_address, source_offset),
                   ops.op("add", allocation_address, allocation_offset))
    leading = ops.op("and", ops.op("sub", delta, layout.payload_offset), layout.copy_alignment - 1)
    return ops, source_offset, ops.op("add", allocation_offset, leading)


def emit_pitched_halo_address(builder, layout, source, allocation, logical_offset):
    """Map one already source-proved logical byte start into private storage.

    A multirow consumer must separately retain the original within-row run
    proof and source pitch; this helper grants no transfer/effect permission.
    """
    if type(builder) is not FnBuilder or type(layout) is not PitchedNHWCHalo:
        raise ValueError("pitched halo address needs the original builder and typed layout")
    layout.__post_init__()
    if any(not isinstance(v, SSAValue) or v.type != PTR for v in (source, allocation)):
        raise ValueError("pitched halo needs original byte-addressable pointers")
    if not isinstance(logical_offset, SSAValue) or logical_offset.type != i64:
        raise ValueError("pitched halo address requires an original i64 byte coordinate")
    ops = _WindowBuilder(builder)
    image = ops.op("udiv", logical_offset, layout.logical_image_bytes)
    inner = ops.op("urem", logical_offset, layout.logical_image_bytes)
    row = ops.op("udiv", inner, layout.logical_row_bytes)
    column = ops.op("urem", inner, layout.logical_row_bytes)
    ops, _, base = _image_base(builder, layout, source, allocation, image)
    offset = ops.op("add", base, ops.op("add", ops.op("mul", row, layout.row_pitch_bytes), column))
    return builder.gep(allocation, offset, i8)


def emit_pitched_nhwc_halo(builder, layout, source, allocation, *,
                           source_destination_disjoint, maximum_private_bytes):
    """Copy exact source rows and zero halo; caller owns capacity and lifetime."""
    if type(builder) is not FnBuilder or type(layout) is not PitchedNHWCHalo:
        raise ValueError("pitched halo needs the original builder and typed layout")
    layout.__post_init__()
    if source_destination_disjoint is not True:
        raise ValueError("pitched halo needs disjoint original source and private storage")
    if (type(maximum_private_bytes) is not int or maximum_private_bytes < layout.private_bytes
            or maximum_private_bytes >= 1 << 63):
        raise ValueError("pitched halo exceeds caller-selected private capacity")
    if any(not isinstance(v, SSAValue) or v.type != PTR for v in (source, allocation)):
        raise ValueError("pitched halo needs original byte-addressable pointers")
    b, h, _, c = layout.window.input_shape
    pt, pl, pb, _ = layout.window.padding

    def image(image):
        ops, source_base, destination_base = _image_base(builder, layout, source, allocation, image)

        def fill(offset, count):
            if count:
                builder.add(llvm.CallIntrinsicOp("llvm.memset.p0.i64", [
                    builder.gep(allocation, ops.value(offset), i8), builder.const(0, i8),
                    builder.const(count), builder.const(0, i1)], []))

        fill(destination_base, pt * layout.row_pitch_bytes)

        def row(y):
            dst = ops.op("add", destination_base, ops.op("mul", ops.op("add", y, pt), layout.row_pitch_bytes))
            src = ops.op("add", source_base, ops.op("mul", y, layout.source_row_bytes))
            fill(dst, pl * c)
            builder.add(llvm.CallIntrinsicOp("llvm.memcpy.p0.p0.i64", [
                builder.gep(allocation, ops.op("add", dst, pl * c), i8),
                builder.gep(source, src, i8), builder.const(layout.source_row_bytes), builder.const(0, i1)], []))
            fill(ops.op("add", dst, pl * c + layout.source_row_bytes),
                 layout.row_pitch_bytes - pl * c - layout.source_row_bytes)

        builder.for_loop(0, h, 1, row, retain_loop=True)
        fill(ops.op("add", destination_base, (pt + h) * layout.row_pitch_bytes), pb * layout.row_pitch_bytes)

    builder.for_loop(0, b, 1, image, retain_loop=True)
    return dict(schema="merlin.pitched_nhwc_halo_preparation.v1", private_bytes=layout.private_bytes,
                row_pitch_bytes=layout.row_pitch_bytes, image_pitch_bytes=layout.image_pitch_bytes,
                logical_bytes=layout.logical_bytes, copy_alignment=layout.copy_alignment,
                source_overread_bytes=0, source_alignment_assumed=False, device_arithmetic=False,
                initialization="exact complete original source rows and integer-zero halo before every consumer",
                unobserved_slack="at most alignment-1 bytes per image; never read by mapped logical coordinates")
