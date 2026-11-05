"""Compact spatial-max gather plans, independent of an instruction encoding.

The adapter supplies a scalar readout followed by a local maximum operation.
This planner supplies source indices, channel tails and complete output order.
It establishes no physical mapping, completion semantics or runtime evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, select


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


def plan_gathered_spatial_max(source, *, batches, channels, channel_block,
                             max_window_elements=256):
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
        raise ValueError('spatial gather requires an explicit source readout')
    if any(type(n) is not int or n <= 0 for n in (batches, channels, channel_block, max_window_elements)):
        raise ValueError('spatial gather requires positive integer extents and limits')
    if source.output_bits != 8:
        raise ValueError('this gather ABI requires signed i8 source and output tensors')
    h, w = source.input_hw
    kh, kw = source.window
    sh, sw = source.stride
    pt, pl, _, _ = source.padding
    oh, ow = source.output_hw
    if kh * kw > max_window_elements:
        raise ValueError('spatial gather exceeds the explicit local-window budget')
    if any(source.padding) and source.pad_value != -128:
        raise ValueError('border duplication requires minimum-valued signed i8 padding')
    if kh <= pt or kw <= pl or (oh - 1) * sh - pt >= h or (ow - 1) * sw - pl >= w:
        raise ValueError('spatial gather cannot replace a completely padded window')
    if max(batches * h * w * channels, batches * oh * ow * channels) >= 1 << 63:
        raise ValueError('spatial gather byte offsets exceed signed i64')
    blocks = (channels + channel_block - 1) // channel_block
    column = Var('channel') * channel_block
    columns = select('channel', blocks - 1, (channels - 1) % channel_block + 1, channel_block)
    body = []
    for ky in range(kh):
        iy = _axis_index('y', h, oh, ky, sh, pt)
        for kx in range(kw):
            ix = _axis_index('x', w, ow, kx, sw, pl)
            source_row = Var('batch') * (h * w) + iy * w + ix
            body.append(Call('gather_window_row', (
                ('source_row', source_row), ('first_column', column),
                ('columns', columns), ('window_row', Const(ky * kw + kx)),
            )))
    output_row = Var('batch') * (oh * ow) + Var('y') * ow + Var('x')
    body.append(Call('store_window_max', (
        ('output_row', output_row), ('first_column', column), ('columns', columns),
    )))
    kernel = Kernel('gathered_spatial_max', (
        TensorArg('input', (batches * h * w, channels), 'i8', 'read'),
        TensorArg('output', (batches * oh * ow, channels), 'i8', 'write'),
    ), (Loop('batch', batches, (Loop('y', oh, (Loop('x', ow, (
        Loop('channel', blocks, tuple(body)),
    )),)),)),))
    return GatheredSpatialMax(source, kernel, batches, channels, channel_block)
