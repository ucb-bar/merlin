"""Compact spatial-max gather plans, independent of an instruction encoding.

The adapter supplies a scalar readout followed by a local maximum operation.
This planner supplies source indices, channel tails and complete output order.
It establishes no physical mapping, completion semantics or runtime evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.compiler.kernel_ir.ir import Call, Const, Kernel, Loop, TensorArg, Var, select
from merlin.compiler.schedule.blocks import SpatialMaxReadout

# ---- spatial_gather ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GatheredSpatialMax:
    source: SpatialMaxReadout
    kernel: Kernel
    batches: int
    channels: int
    channel_block: int

    @property
    def temporary_rows(self):
        return self.source.window[0] * self.source.window[1]

    @property
    def dynamic_calls(self):
        oh, ow = self.source.output_hw
        blocks = (self.channels + self.channel_block - 1) // self.channel_block
        return self.batches * oh * ow * blocks * (self.temporary_rows + 1)


def _axis_index(var, extent, output_extent, window_offset, stride, padding):
    """Affine interior with only the bounded boundary iterations peeled by select."""
    raw = Var(var) * stride + (window_offset - padding)
    low_stop = min(output_extent, max(0, (padding - window_offset + stride - 1) // stride))
    high_start = max(0, min(output_extent, (extent + padding - window_offset + stride - 1) // stride))
    result = raw
    for index in range(low_stop):
        result = select(var, index, 0, result)
    for index in range(high_start, output_extent):
        result = select(var, index, extent - 1, result)
    return result


def plan_gathered_spatial_max(source, *, batches, channels, channel_block, max_window_elements=256):
    """Gather one complete window at a time without expanding image-sized loops.

    For minimum-valued padding, clamping indices duplicates valid border values.
    Every admitted window contains a valid cell: all valid entries remain, and
    duplicates cannot change max. The adapter must apply the same source scalar
    readout to each gathered entry *before* the local maximum. No monotonicity
    assumption about that scalar readout is needed.

    Padding in another numerical domain, or a completely padded window, refuses.
    The logical scratch rows are private until the synchronous local store ends;
    the physical adapter must enforce this lifetime and its own capacity limits.
    """
    if type(source) is not SpatialMaxReadout:
        raise ValueError("spatial gather requires an explicit source readout")
    if any(type(n) is not int or n <= 0 for n in (batches, channels, channel_block, max_window_elements)):
        raise ValueError("spatial gather requires positive integer extents and limits")
    if source.output_bits != 8:
        raise ValueError("this gather ABI requires signed i8 source and output tensors")
    h, w = source.input_hw
    kh, kw = source.window
    sh, sw = source.stride
    pt, pl, _, _ = source.padding
    oh, ow = source.output_hw
    if kh * kw > max_window_elements:
        raise ValueError("spatial gather exceeds the explicit local-window budget")
    if any(source.padding) and source.pad_value != -128:
        raise ValueError("border duplication requires minimum-valued signed i8 padding")
    if kh <= pt or kw <= pl or (oh - 1) * sh - pt >= h or (ow - 1) * sw - pl >= w:
        raise ValueError("spatial gather cannot replace a completely padded window")
    if max(batches * h * w * channels, batches * oh * ow * channels) >= 1 << 63:
        raise ValueError("spatial gather byte offsets exceed signed i64")
    blocks = (channels + channel_block - 1) // channel_block
    column = Var("channel") * channel_block
    columns = select("channel", blocks - 1, (channels - 1) % channel_block + 1, channel_block)
    body = []
    for ky in range(kh):
        iy = _axis_index("y", h, oh, ky, sh, pt)
        for kx in range(kw):
            ix = _axis_index("x", w, ow, kx, sw, pl)
            source_row = Var("batch") * (h * w) + iy * w + ix
            body.append(
                Call(
                    "gather_window_row",
                    (
                        ("source_row", source_row),
                        ("first_column", column),
                        ("columns", columns),
                        ("window_row", Const(ky * kw + kx)),
                    ),
                )
            )
    output_row = Var("batch") * (oh * ow) + Var("y") * ow + Var("x")
    body.append(
        Call(
            "store_window_max",
            (
                ("output_row", output_row),
                ("first_column", column),
                ("columns", columns),
            ),
        )
    )
    kernel = Kernel(
        "gathered_spatial_max",
        (
            TensorArg("input", (batches * h * w, channels), "i8", "read"),
            TensorArg("output", (batches * oh * ow, channels), "i8", "write"),
        ),
        (Loop("batch", batches, (Loop("y", oh, (Loop("x", ow, (Loop("channel", blocks, tuple(body)),)),)),)),),
    )
    return GatheredSpatialMax(source, kernel, batches, channels, channel_block)


# ---- spatial_max_stripes -----------------------------------------------------------------------
# Reuse complete source row strips across spatial maximum readouts.
#
# This changes staging granularity, not scalar arithmetic or completion laws.
# A backend must bind every staged row and the full strip readout to its selected
# addressing/pooling semantics, and complete that readout before strip reuse.


def _horizontal(width, stripe_width, left_padding):
    left = min(left_padding, stripe_width)
    middle = max(0, min(width, stripe_width - left))
    return left, middle, stripe_width - left - middle


def _kernel(source, batches, channels, channel_block, transfer_rows):
    h, w = source.input_hw
    kh, kw = source.window
    sh, sw = source.stride
    oh, ow = source.output_hw
    pt, pl, _, _ = source.padding
    width = (ow - 1) * sw + kw
    left, middle, right = _horizontal(w, width, pl)
    blocks = (channels + channel_block - 1) // channel_block
    column = Var("channel") * channel_block
    columns = select("channel", blocks - 1, (channels - 1) % channel_block + 1, channel_block)
    body = []
    for ky in range(kh):
        y = _axis_index("output_y", h, oh, ky, sh, pt)
        base = Var("batch") * (h * w) + y * w

        def load(source_x, local_x, rows):
            return Call(
                "stage_spatial_rows",
                (
                    ("source_row", base + source_x),
                    ("local_row", Const(ky * width) + local_x),
                    ("rows", rows),
                    ("first_column", column),
                    ("columns", columns),
                ),
            )

        if left:
            name = f"left_{ky}"
            body.append(Loop(name, left, (load(Const(0), Var(name), Const(1)),)))
        full, tail = divmod(middle, transfer_rows)
        if full:
            name = f"middle_{ky}"
            body.append(
                Loop(
                    name,
                    full,
                    (load(Var(name) * transfer_rows, Var(name) * transfer_rows + left, Const(transfer_rows)),),
                )
            )
        if tail:
            body.append(load(Const(full * transfer_rows), Const(left + full * transfer_rows), Const(tail)))
        if right:
            name = f"right_{ky}"
            body.append(Loop(name, right, (load(Const(w - 1), Var(name) + left + middle, Const(1)),)))
    body.append(
        Call(
            "read_spatial_strip",
            (
                ("output_row", Var("batch") * (oh * ow) + Var("output_y") * ow),
                ("first_column", column),
                ("columns", columns),
            ),
        )
    )
    return Kernel(
        "striped_spatial_max",
        (
            TensorArg("input", (batches * h * w, channels), "i8", "read"),
            TensorArg("output", (batches * oh * ow, channels), "i8", "write"),
        ),
        (Loop("batch", batches, (Loop("output_y", oh, (Loop("channel", blocks, tuple(body)),)),)),),
    )


@dataclass(frozen=True)
class StripedSpatialMax:
    source: SpatialMaxReadout
    batches: int
    channels: int
    channel_block: int
    transfer_rows: int
    capacity_rows: int
    kernel: Kernel

    @property
    def stripe_hw(self):
        return self.source.window[0], (self.source.output_hw[1] - 1) * self.source.stride[1] + self.source.window[1]

    @property
    def temporary_rows(self):
        height, width = self.stripe_hw
        return height * width

    @property
    def dynamic_calls(self):
        left, middle, right = _horizontal(self.source.input_hw[1], self.stripe_hw[1], self.source.padding[1])
        loads = self.source.window[0] * (left + (middle + self.transfer_rows - 1) // self.transfer_rows + right)
        return (
            self.batches
            * self.source.output_hw[0]
            * ((self.channels + self.channel_block - 1) // self.channel_block)
            * (loads + 1)
        )

    def revalidate(self):
        """Check this actual counted program and its original capacity/source join.

        Only the canonical strip producer is admitted here. Any transformed
        program needs its own complete effects/source checker. Caller-provided
        capacity is a bound, not an ownership or physical address-space grant.
        """
        original = plan_striped_spatial_max(
            self.source,
            batches=self.batches,
            channels=self.channels,
            channel_block=self.channel_block,
            transfer_rows=self.transfer_rows,
            capacity_rows=self.capacity_rows,
        )
        if self.kernel != original.kernel:
            raise ValueError("actual spatial strip program differs from complete original source staging/readout")
        return {
            "schema": "merlin.striped_spatial_max.v1",
            "kernel_sha256": self.kernel.digest(),
            "stripe_hw": list(self.stripe_hw),
            "output_hw": [1, self.source.output_hw[1]],
            "temporary_rows": self.temporary_rows,
            "dynamic_calls": self.dynamic_calls,
            "scalar_readout": "unchanged caller-bound readout before signed maximum",
            "padding": "minimum signed i8; duplicate clamped valid source cells",
            "completion_required": "complete every staging call and strip readout before reuse",
            "physical_mapping_verified": False,
            "target_execution": False,
            "timing": None,
        }


def plan_striped_spatial_max(source, *, batches, channels, channel_block, transfer_rows, capacity_rows):
    """Stage one window-height strip, then read all horizontal windows once.

    Horizontal halo cells duplicate border source values. Every original window
    has a valid source cell (checked by the existing gather planner), so minimum
    post-readout padding and these duplicates have the same maximum, including
    for arbitrary caller-bound scalar readouts. No CPU tensor math is proposed.
    """
    # Reuse original complete-window/minimum-padding/source-extent validation.
    plan_gathered_spatial_max(source, batches=batches, channels=channels, channel_block=channel_block)
    if any(type(v) is not int or v <= 0 for v in (transfer_rows, capacity_rows)):
        raise ValueError("spatial strip needs positive explicit transfer and private capacity row bounds")
    height = source.window[0]
    width = (source.output_hw[1] - 1) * source.stride[1] + source.window[1]
    if height * width > capacity_rows:
        raise ValueError("complete spatial strip exceeds selected private row capacity")
    kernel = _kernel(source, batches, channels, channel_block, transfer_rows)
    return StripedSpatialMax(source, batches, channels, channel_block, transfer_rows, capacity_rows, kernel)
