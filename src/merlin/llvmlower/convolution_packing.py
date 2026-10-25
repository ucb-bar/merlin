"""Bounded NHWC window transport into a caller-owned row-major matrix panel.

Only input bytes and zero padding move here. The caller selects panel capacity,
proves disjoint storage, and owns the device contraction and its completion.
"""

from dataclasses import dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, i1, i8, i64
from xdsl.ir import Block, SSAValue

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.llvmlower.host_builder import PTR, FnBuilder


class _WindowBuilder:
    def __init__(self, builder):
        self.builder = builder

    def value(self, value, dtype=i64):
        return self.builder.const(value, dtype) if type(value) is int else value

    def op(self, name, lhs, rhs, dtype="i64"):
        typ = i1 if dtype == "i1" else i64
        operation = {
            "add": llvm.AddOp,
            "sub": llvm.SubOp,
            "mul": llvm.MulOp,
            "udiv": llvm.UDivOp,
            "urem": llvm.URemOp,
            "and": llvm.AndOp,
        }[name]
        return self.builder.add(operation(self.value(lhs, typ), self.value(rhs, typ))).results[0]

    def cmp(self, predicate, lhs, rhs):
        flag = {
            "ult": llvm.ICmpPredicateFlag.ULT,
            "ule": llvm.ICmpPredicateFlag.ULE,
            "slt": llvm.ICmpPredicateFlag.SLT,
        }[predicate]
        return self.builder.add(llvm.ICmpOp(self.value(lhs), self.value(rhs), IntegerAttr(flag.to_int(), i64))).results[
            0
        ]

    def select(self, condition, lhs, rhs):
        return self.builder.add(llvm.SelectOp(condition, self.value(lhs), self.value(rhs))).results[0]


@dataclass(frozen=True)
class ClippedSpatialHaloRows:
    """The original pool's rectangular producer rows, in local row-major order.

    Clamping duplicates border producer values. This is legal only when the
    source observer proof separately admits removing minimum-valued pool pads
    after the original readout. The mapping does not collapse either readout.
    """

    spatial: SpatialMaxReadout
    output_columns: int

    def __post_init__(self):
        if type(self.spatial) is not SpatialMaxReadout:
            raise ValueError("halo packing requires the original typed pool geometry")
        self.spatial.__post_init__()
        s = self.spatial
        if type(self.output_columns) is not int or not 0 < self.output_columns <= s.output_hw[1]:
            raise ValueError("halo packing requires a bounded original output stripe")
        if any(s.padding) and s.pad_value != -128:
            raise ValueError("clipped halos require minimum-valued signed-i8 pool padding")
        kh, kw = s.window
        sh, sw = s.stride
        pt, pl, _, _ = s.padding
        h, w = s.input_hw
        oh, ow = s.output_hw
        if kh <= pt or kw <= pl or (oh - 1) * sh - pt >= h or (ow - 1) * sw - pl >= w:
            raise ValueError("clipped halos cannot replace completely padded pool windows")
        if max(self.rows, oh * sh + kh, ow * sw + kw) >= 1 << 63:
            raise ValueError("halo packing exceeds signed-i64 coordinate arithmetic")

    @property
    def width(self):
        return (self.output_columns - 1) * self.spatial.stride[1] + self.spatial.window[1]

    @property
    def rows(self):
        return self.spatial.window[0] * self.width


def emit_nhwc_window_panel(
    builder, window, source, destination, row_start, rows, *, source_destination_disjoint, maximum_panel_bytes
):
    """Emit counted byte-copy loops for ``[rows, kh*kw*channels]`` signed-i8 input.

    ``row_start`` is an unsigned logical output-row offset (i64). Rows beyond the
    window domain produce zeros and never read source memory. Real convolution
    callers must select only original output rows. Destination has at least the
    declared panel bytes; source holds the complete NHWC input. No allocation,
    target operation, reduction, quantization or implicit asynchronous lifetime.
    """
    if type(builder) is not FnBuilder or type(window) is not Window2D:
        raise ValueError("typed LLVM builder and explicit window geometry required")
    window.__post_init__()
    if source_destination_disjoint is not True:
        raise ValueError("window packing requires disjoint input and output storage")
    if any(not isinstance(p, SSAValue) or p.type != PTR for p in (source, destination)):
        raise ValueError("window packing requires byte-addressable pointers")
    if not isinstance(row_start, SSAValue) or row_start.type != i64:
        raise ValueError("window packing requires an i64 output-row start")
    if (
        type(rows) is not int
        or rows <= 0
        or type(maximum_panel_bytes) is not int
        or maximum_panel_bytes <= 0
        or maximum_panel_bytes >= 1 << 63
    ):
        raise ValueError("window packing requires a positive bounded panel budget")
    kh, kw = window.kernel
    channels = window.input_shape[-1]
    reduction = kh * kw * channels
    panel_bytes = rows * reduction
    if panel_bytes > maximum_panel_bytes:
        raise ValueError("window panel exceeds caller-selected packing capacity")
    ops = _WindowBuilder(builder)
    oh, ow = window.output_spatial
    start_valid = ops.cmp("ult", row_start, window.input_shape[0] * oh * ow)

    def row_mapping(panel_row):
        return ops.op("add", row_start, panel_row), start_valid

    _emit_window_bytes(builder, window, source, destination, rows, row_mapping)
    return _panel_receipt(rows, reduction, panel_bytes)


def _emit_window_bytes(builder, window, source, destination, rows, row_mapping):
    """Closed byte-copy implementation shared by the two declared row maps."""
    ops = _WindowBuilder(builder)
    kh, kw = window.kernel
    channels = window.input_shape[-1]
    reduction = kh * kw * channels

    def row_body(panel_row):
        row, row_valid = row_mapping(panel_row)

        def ky_body(ky):
            def kx_body(kx):
                run = window.emit_row_run(ops, row, 1, ky, kx, 0, channels)
                packed = ops.op(
                    "add",
                    ops.op("mul", panel_row, reduction),
                    ops.op("mul", ops.op("add", ops.op("mul", ky, kw), kx), channels),
                )
                valid, padding, after = Block(), Block(), Block()
                for block in (valid, padding, after):
                    builder.region.add_block(block)
                builder.add(llvm.CondBrOp(ops.op("and", row_valid, run["predicate"], "i1"), valid, [], padding, []))
                builder.blk = valid

                def copy(channel):
                    src = builder.gep(source, ops.op("add", run["first_element"], channel), i8)
                    dst = builder.gep(destination, ops.op("add", packed, channel), i8)
                    value = builder.add(llvm.LoadOp(src, i8)).results[0]
                    builder.add(llvm.StoreOp(value, dst))

                builder.for_loop(0, channels, 1, copy, retain_loop=True)
                builder.add(llvm.BrOp(after))
                builder.blk = padding

                def zero(channel):
                    dst = builder.gep(destination, ops.op("add", packed, channel), i8)
                    builder.add(llvm.StoreOp(builder.const(0, i8), dst))

                builder.for_loop(0, channels, 1, zero, retain_loop=True)
                builder.add(llvm.BrOp(after))
                builder.blk = after

            builder.for_loop(0, kw, 1, kx_body, retain_loop=True)

        builder.for_loop(0, kh, 1, ky_body, retain_loop=True)

    builder.for_loop(0, rows, 1, row_body, retain_loop=True)


def _panel_receipt(rows, reduction, panel_bytes):
    return {
        "kind": "host_input_transport",
        "panel_shape": [rows, reduction],
        "panel_bytes": panel_bytes,
        "input_layout": "nhwc",
        "reduction_order": "kh,kw,channel",
        "zero_padding": True,
        "device_arithmetic": False,
        "storage_completion": "synchronous host copies; caller owns device consumer lifetime",
    }


def emit_nhwc_halo_panel(
    builder,
    window,
    halo,
    source,
    destination,
    image,
    output_y,
    output_x,
    row_start,
    rows,
    *,
    source_destination_disjoint,
    maximum_panel_bytes,
):
    """Pack a bounded subset of the original pool halo into ``[rows, K]``.

    Runtime coordinates are unsigned domain-checked before any source load.
    K is exactly kh/kw/channel order, including original convolution zero pads.
    Rows crossing a halo scanline retain their correct source coordinates.
    Out-of-domain calls and rows produce zeros. No allocation or device effect
    occurs; the caller must bind the source observer and private storage proofs.
    """
    if type(builder) is not FnBuilder or type(window) is not Window2D or type(halo) is not ClippedSpatialHaloRows:
        raise ValueError("halo packing requires typed builder, source window and pool row mapping")
    window.__post_init__()
    halo.__post_init__()
    if window.output_spatial != halo.spatial.input_hw:
        raise ValueError("halo pool input differs from the original producer output")
    if source_destination_disjoint is not True:
        raise ValueError("halo packing requires disjoint input and output storage")
    if any(not isinstance(p, SSAValue) or p.type != PTR for p in (source, destination)):
        raise ValueError("halo packing requires byte-addressable pointers")
    if any(not isinstance(v, SSAValue) or v.type != i64 for v in (image, output_y, output_x, row_start)):
        raise ValueError("halo packing coordinates require i64")
    reduction = window.kernel[0] * window.kernel[1] * window.input_shape[-1]
    if (
        type(rows) is not int
        or not 0 < rows < 1 << 63
        or type(maximum_panel_bytes) is not int
        or not 0 < maximum_panel_bytes < 1 << 63
        or rows * reduction > maximum_panel_bytes
    ):
        raise ValueError("halo panel exceeds caller-selected packing capacity")
    ops = _WindowBuilder(builder)
    s = halo.spatial
    h, w = s.input_hw
    oh, ow = s.output_hw

    def both(lhs, rhs):
        return ops.op("and", lhs, rhs, "i1")

    valid = both(ops.cmp("ult", image, window.input_shape[0]), ops.cmp("ult", output_y, oh))
    valid = both(valid, ops.cmp("ule", output_x, ow - halo.output_columns))
    valid = both(valid, ops.cmp("ult", row_start, halo.rows))

    def clip(value, extent):
        nonnegative = ops.select(ops.cmp("slt", value, 0), 0, value)
        return ops.select(ops.cmp("ult", nonnegative, extent), nonnegative, extent - 1)

    def row_mapping(panel_row):
        # The subtraction predicate prevents row_start+panel_row wrapping back
        # into the domain even when the runtime start is UINT64_MAX.
        row_valid = both(valid, ops.cmp("ult", panel_row, ops.op("sub", halo.rows, row_start)))
        local = ops.op("add", row_start, panel_row)
        y = ops.op(
            "add", ops.op("sub", ops.op("mul", output_y, s.stride[0]), s.padding[0]), ops.op("udiv", local, halo.width)
        )
        x = ops.op(
            "add", ops.op("sub", ops.op("mul", output_x, s.stride[1]), s.padding[1]), ops.op("urem", local, halo.width)
        )
        row = ops.op("add", ops.op("mul", ops.op("add", ops.op("mul", image, h), clip(y, h)), w), clip(x, w))
        return row, row_valid

    _emit_window_bytes(builder, window, source, destination, rows, row_mapping)
    return dict(
        _panel_receipt(rows, reduction, rows * reduction),
        row_mapping="clipped_original_pool_halo",
        halo_shape=[s.window[0], halo.width],
        output_columns=halo.output_columns,
        source_observer_qualification="caller-owned; not inferred",
    )
