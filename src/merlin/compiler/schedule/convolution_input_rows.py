"""Source-bound resident convolution input-row placement."""

import hashlib
from dataclasses import asdict, dataclass, field

from .window_geometry import Window2D


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
    Explicit image_windows partition the same private footprint into complete
    original image-row ranges. The original image coordinate selects a window;
    source lifetimes and target transfer/overlap permissions remain separate.
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
    image_windows: int = 1
    channel_segments: tuple[tuple[int, int, int], ...] | None = None
    _source_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_source_sha256", hashlib.sha256(repr(self).encode()).hexdigest())

    def _validate(self):
        if type(self.window) is not Window2D:
            raise ValueError("resident input rows require the exact original window")
        self.window.__post_init__()
        if (
            type(self.slots) is not int
            or type(self.image_windows) is not int
            or not 0 < self.image_windows < 1 << 63
            or self.slots % self.image_windows
        ):
            raise ValueError("resident image windows need distinct equal private row ranges")
        if (
            type(self.source_identity) is not str
            or len(self.source_identity) != 64
            or any(c not in "0123456789abcdef" for c in self.source_identity)
        ):
            raise ValueError("resident input rows require source-bound windows")
        if (
            self.compute_row_step is None
            and self.window.stride[1] != 1
            or self.compute_row_step is not None
            and (type(self.compute_row_step) is not int or self.compute_row_step != self.window.stride[1])
        ):
            raise ValueError("resident horizontal stride requires an exact explicit compute row-step binding")
        # CountedConvolutionPlan defines the original signed-integer padding
        # as zero. A different affine quantized zero point is not that source.
        if (
            any(self.window.padding)
            and (type(self.halo_value) is not int or self.halo_value != 0)
            or self.halo_value is not None
            and (type(self.halo_value) is not int or self.halo_value != 0)
        ):
            raise ValueError("padded resident rows require explicit original integer-zero halo value")
        if (
            any(type(v) is not int or v < 0 for v in (self.base, self.private_first))
            or any(
                type(v) is not int or not 0 < v < 1 << 63
                for v in (self.channel_tile, self.row_pitch, self.slots, self.private_rows, self.stage_rows)
            )
            or self.row_pitch < self.padded_width
            or self.stage_rows > self.row_pitch
            or self.slots // self.image_windows < (self.window.kernel[0] - 1) * self.window.dilation[0] + 1
            or self.base < self.private_first
            or self.end > self.private_first + self.private_rows
            or self.end * self.channel_tile >= 1 << 63
        ):
            raise ValueError("resident input row layout exceeds its explicit private footprint or transport bounds")
        if self.stage_channels is not None and (
            type(self.stage_channels) is not int
            or not 0 < self.stage_channels < 1 << 63
            or self.stage_channels % self.channel_tile
        ):
            raise ValueError("resident channel staging requires an explicit positive whole-channel-block limit")
        if self.channel_segments is not None:
            ci = self.window.input_shape[-1]
            if (
                type(self.channel_segments) is not tuple
                or not self.channel_segments
                or any(
                    type(s) is not tuple or len(s) != 3 or any(type(v) is not int for v in s)
                    for s in self.channel_segments
                )
            ):
                raise ValueError("input channel segments require exact static original channel intervals and row bases")
            first = 0
            for channel, count, base in self.channel_segments:
                if (
                    channel != first
                    or count <= 0
                    or channel % self.channel_tile
                    or channel + count > ci
                    or channel + count < ci
                    and count % self.channel_tile
                    or channel
                    and channel % (self.stage_channels or ci)
                    or base < self.private_first
                ):
                    raise ValueError("input channel segments change original channel coverage or transfer boundaries")
                first += count
            windows = self.physical_windows
            if (
                first != ci
                or self.channel_segments[0][2] != self.base
                or any(a < d and c < b for i, (a, b) in enumerate(windows) for c, d in windows[i + 1 :])
            ):
                raise ValueError("input channel segments require complete original coverage and disjoint physical rows")
        if self.image_windows > 1 and self.slots // self.image_windows < (
            (self.window.kernel[0] - 1) * self.window.dilation[0]
            + 1
            + (self.window.output_spatial[0] - 1) * self.window.stride[0]
        ):
            raise ValueError("resident image windows must each own their complete original source rows")

    @property
    def padded_width(self):
        return self.window.input_shape[2] + self.window.padding[1] + self.window.padding[3]

    @property
    def row_step(self):
        """Physical rows between consecutive output samples, not a target encoding."""
        return 1 if self.compute_row_step is None else self.compute_row_step

    @property
    def channel_blocks(self):
        return (self.window.input_shape[-1] + self.channel_tile - 1) // self.channel_tile

    @property
    def rows_per_source_row(self):
        return self.row_pitch * self.channel_blocks

    @property
    def physical_windows(self):
        """Actual disjoint row intervals; holes between segments are unowned."""
        if self.channel_segments is None:
            return ((self.base, self.base + self.slots * self.rows_per_source_row),)
        if (
            type(self.channel_segments) is not tuple
            or not self.channel_segments
            or any(
                type(s) is not tuple or len(s) != 3 or any(type(v) is not int for v in s) for s in self.channel_segments
            )
        ):
            raise ValueError("input channel segments require typed static interval triples")
        return tuple(
            (base, base + self.slots * ((count + self.channel_tile - 1) // self.channel_tile) * self.row_pitch)
            for channel, count, base in self.channel_segments
        )

    @property
    def end(self):
        return max(stop for first, stop in self.physical_windows)

    @property
    def allocated_rows(self):
        return sum(stop - first for first, stop in self.physical_windows)

    def disjoint_from(self, first, count):
        return all(stop <= first or base >= first + count for base, stop in self.physical_windows)

    def _emit_slot_channel(self, builder, slot, channel, column):
        if self.channel_segments is None:
            return builder.op(
                "add",
                self.base,
                builder.op(
                    "add",
                    builder.op("mul", slot, self.rows_per_source_row),
                    builder.op(
                        "add", builder.op("mul", builder.op("udiv", channel, self.channel_tile), self.row_pitch), column
                    ),
                ),
            )
        if len(self.channel_segments) > 1 and not callable(getattr(builder, "select", None)):
            raise ValueError("segmented input row map requires the original typed index select builder")

        def segment(channel_first, count, base):
            blocks = (count + self.channel_tile - 1) // self.channel_tile
            within = builder.op("udiv", builder.op("sub", channel, channel_first), self.channel_tile)
            return builder.op(
                "add",
                base,
                builder.op(
                    "add",
                    builder.op("mul", slot, blocks * self.row_pitch),
                    builder.op("add", builder.op("mul", within, self.row_pitch), column),
                ),
            )

        result = segment(*self.channel_segments[-1])
        for item in reversed(self.channel_segments[:-1]):
            result = builder.select(builder.cmp("ult", channel, item[0] + item[1]), segment(*item), result)
        return result

    @property
    def initial_load_commands(self):
        rows = (self.window.kernel[0] - 1) * self.window.dilation[0] + 1
        if self.image_windows > 1:
            rows += (self.window.output_spatial[0] - 1) * self.window.stride[0]
        return self.stage_command_bound(rows)

    def stage_command_bound(self, source_rows):
        """Bound emitted row/halo/channel transfers, independently of queue slots."""
        if type(source_rows) is not int or not 0 < source_rows <= self.slots:
            raise ValueError("resident staging bound needs actual private source rows")
        _, left, _, right = self.window.padding

        def chunks(count):
            return (count + self.stage_rows - 1) // self.stage_rows

        per_row = max(chunks(left) + chunks(self.window.input_shape[2]) + chunks(right), chunks(self.padded_width))
        ci = self.window.input_shape[-1]
        channel_chunks = 1 if self.stage_channels is None else (ci + self.stage_channels - 1) // self.stage_channels
        return per_row * source_rows * channel_chunks

    def revalidate(self):
        self._validate()
        if hashlib.sha256(repr(self).encode()).hexdigest() != self._source_sha256:
            raise ValueError("original resident input layout changed")

    def rolling_plan(self):
        from merlin.compiler.schedule.program_limits import AddressWindow
        from merlin.compiler.schedule.rolling_window import RollingWindowPlan

        self.revalidate()
        oh, ow = self.window.output_spatial
        pitch = self.rows_per_source_row * self.channel_tile
        return RollingWindowPlan(
            self.window,
            self.source_identity,
            0,
            ow,
            0,
            oh,
            AddressWindow("logical_resident_input", 0, self.slots * pitch),
            self.slots,
            pitch,
            image_windows=self.image_windows,
        )

    def physical_cell(self, cache_byte):
        """Exact static map used by independent cell/layout checks."""
        self.revalidate()
        pitch = self.rows_per_source_row * self.channel_tile
        ci = self.window.input_shape[-1]
        if type(cache_byte) is not int or not 0 <= cache_byte < self.slots * pitch:
            raise ValueError("resident input cache byte outside original footprint")
        slot, within = divmod(cache_byte, pitch)
        if within >= self.window.input_shape[2] * ci:
            raise ValueError("resident input padding hole has no source word")
        x, c = divmod(within, ci)
        if self.channel_segments is None:
            address = self.base + slot * self.rows_per_source_row + (c // self.channel_tile) * self.row_pitch
        else:
            first, count, base = next(s for s in self.channel_segments if s[0] <= c < s[0] + s[1])
            address = (
                base
                + slot * ((count + self.channel_tile - 1) // self.channel_tile) * self.row_pitch
                + ((c - first) // self.channel_tile) * self.row_pitch
            )
        return address + self.window.padding[1] + x, c % self.channel_tile

    def emit_row(self, builder, output_row, ky, kx, channel):
        oh, ow = self.window.output_spatial
        y = builder.op("urem", builder.op("udiv", output_row, ow), oh)
        x = builder.op("urem", output_row, ow)
        image = None if self.image_windows == 1 else builder.op("udiv", output_row, oh * ow)
        return self.emit_row_coordinates(builder, y, x, ky, kx, channel, image=image)

    def emit_row_coordinates(self, builder, output_y, output_x, ky, kx, channel, *, image=None):
        """Map original coordinates directly, without decoding a logical byte.

        The checked source consumer owns the output/tap domain and matching
        consume/complete boundary. Coordinates may be hoisted out of reduction
        and output-channel loops. Padded rows use the existing staged zero halo;
        this neither changes source samples nor grants early slot reuse.
        """
        self.revalidate()
        windows = getattr(self, "image_windows", 1)
        if windows > 1 and image is None:
            raise ValueError("resident image windows require their original source image")
        # Physical columns already contain the original left halo, cancelling
        # the source's -left. Ring slots use unbiased y even for zero halo rows.
        slot = builder.op(
            "urem",
            builder.op(
                "add",
                builder.op("mul", output_y, self.window.stride[0]),
                builder.op("mul", ky, self.window.dilation[0]),
            ),
            self.slots // windows,
        )
        if windows > 1:
            slot = builder.op("add", slot, builder.op("mul", builder.op("urem", image, windows), self.slots // windows))
        sample_x = output_x if self.window.stride[1] == 1 else builder.op("mul", output_x, self.window.stride[1])
        column = builder.op("add", sample_x, builder.op("mul", kx, self.window.dilation[1]))
        return self._emit_slot_channel(builder, slot, channel, column)

    def emit_stage(self, builder, fields, lower, *, channel_interval=None):
        """Bind an original stage row through its exact tiled cell map.

        A pending input owner may issue one original channel transfer at its
        first reader. Its static interval must equal one complete transfer of
        this row's existing stage_channels partition, including its real tail.
        Default emission retains every original channel transfer in order.
        """
        self.revalidate()
        width, ci = self.window.input_shape[2:]
        if channel_interval is not None:
            part = self.stage_channels or ci
            if (
                type(channel_interval) is not tuple
                or len(channel_interval) != 2
                or any(type(v) is not int for v in channel_interval)
                or not 0 <= channel_interval[0] < ci
                or channel_interval[0] % part
                or channel_interval[1] != min(part, ci - channel_interval[0])
            ):
                raise ValueError("input stage interval changes its original channel transfer partition")
        slot = builder.op("udiv", fields["cache_byte"], self.rows_per_source_row * self.channel_tile)
        local = builder.op("add", self.base, builder.op("mul", slot, self.rows_per_source_row))
        if channel_interval is not None or self.stage_channels is not None and self.stage_channels < ci:
            original_lower = lower

            def split_channels(kind, args):
                def part(channel, depth):
                    address = (
                        builder.op(
                            "add",
                            args["local_row"],
                            builder.op("mul", builder.op("udiv", channel, self.channel_tile), self.row_pitch),
                        )
                        if self.channel_segments is None
                        else self._emit_slot_channel(
                            builder, slot, channel, builder.op("sub", args["local_row"], local)
                        )
                    )
                    value = dict(args, columns=depth, local_row=address)
                    if kind == "stage_input_rows":
                        value["first_element"] = builder.op("add", args["first_element"], channel)
                    original_lower(kind, value)

                if channel_interval is not None:
                    part(*channel_interval)
                else:
                    whole, tail = divmod(ci, self.stage_channels)
                    builder.loop(
                        0,
                        whole,
                        1,
                        lambda index: part(builder.op("mul", index, self.stage_channels), self.stage_channels),
                    )
                    if tail:
                        part(whole * self.stage_channels, tail)

            lower = split_channels
        if not any(self.window.padding):

            def part(x, count):
                lower(
                    "stage_input_rows",
                    dict(
                        first_element=builder.op("add", fields["source_byte"], builder.op("mul", x, ci)),
                        local_row=builder.op("add", local, x),
                        rows=count,
                        columns=ci,
                        row_stride_elements=ci,
                        block_stride_rows=self.row_pitch,
                    ),
                )

            def valid():
                whole, tail = divmod(width, self.stage_rows)
                builder.loop(
                    0, whole, 1, lambda index: part(builder.op("mul", index, self.stage_rows), self.stage_rows)
                )
                if tail:
                    part(whole * self.stage_rows, tail)

            builder.branch(fields["valid"], valid, lambda: None)
            return
        left, right = self.window.padding[1], self.window.padding[3]

        def pieces(count, emit):
            whole, tail = divmod(count, self.stage_rows)
            if whole:
                builder.loop(
                    0, whole, 1, lambda index: emit(builder.op("mul", index, self.stage_rows), self.stage_rows)
                )
            if tail:
                emit(whole * self.stage_rows, tail)

        def fill(first, count):
            if not count:
                return

            def part(x, rows):
                lower(
                    "fill_input_rows",
                    dict(
                        value=self.halo_value,
                        local_row=builder.op("add", local, builder.op("add", first, x)),
                        rows=rows,
                        columns=ci,
                        block_stride_rows=self.row_pitch,
                    ),
                )

            pieces(count, part)

        def valid():
            fill(0, left)

            def part(x, count):
                lower(
                    "stage_input_rows",
                    dict(
                        first_element=builder.op("add", fields["source_byte"], builder.op("mul", x, ci)),
                        local_row=builder.op("add", local, builder.op("add", left, x)),
                        rows=count,
                        columns=ci,
                        row_stride_elements=ci,
                        block_stride_rows=self.row_pitch,
                    ),
                )

            pieces(width, part)
            fill(left + width, right)

        if any(self.window.padding):
            builder.branch(fields["valid"], valid, lambda: fill(0, self.padded_width))
        else:
            builder.branch(fields["valid"], valid, lambda: None)

    def record(self):
        self.revalidate()
        return dict(
            schema="merlin.convolution_resident_input_rows.v1",
            source_sha256=self._source_sha256,
            layout=asdict(self),
            private_row_interval=[min(first for first, stop in self.physical_windows), self.end],
            private_row_windows=self.physical_windows,
            allocated_rows=self.allocated_rows,
            compute_read_row_step=self.row_step,
            mapping=(
                (
                    "segment_base+slot*segment_channel_blocks*row_pitch+((channel-segment_first)//tile)*row_pitch+left+x; lane=channel%tile; "
                    if self.channel_segments is not None
                    else "slot*rows_per_source_row + (channel//tile)*row_pitch + left+x; lane=channel%tile; "
                )
                + "halo uses original padded coordinates and explicit original integer-zero fill"
                + (
                    "; slot=(image%image_windows)*(slots/image_windows)+unbiased_y%(slots/image_windows)"
                    if self.image_windows > 1
                    else ""
                )
                + (
                    "; compute source_x=output_x*stride_x+kx*dilation_x-left; "
                    "sample r reads base_compute_row+r*stride_x"
                    if self.row_step != 1
                    else ""
                )
            ),
            qualification=(
                "exact source-cell and lifetime map; target transfer, row-step and full completion remain separate"
            ),
        )


# Preserve the original public lookup and pickle identities.
for _definition in (ConvolutionResidentInputRows,):
    _definition.__module__ = "merlin.compiler.schedule.window_transfer"
del _definition
