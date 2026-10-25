"""Compact pure-convolution halo stripes feeding exact spatial maximum.

Only one halo rectangle and one channel tile are retained in private raw
accumulator rows. Overlapping pure products may be recomputed. No full image,
payload packing, accumulator copy primitive or ring-wrap pool layout is assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from math import prod

from .block_schedule import SpatialMaxReadout
from .program_limits import AddressWindow
from .spatial_readout import SpatialReadoutChain, prove_spatial_readout
from .window_transfer import CountedConvolutionPlan, emit_convolution_tile


@dataclass(frozen=True)
class SpatialProducerPlan:
    producer: CountedConvolutionPlan
    spatial: SpatialMaxReadout
    readouts: tuple[SpatialReadoutChain, ...]
    output_tile_width: int
    external_observers: tuple[str, ...] = ()
    readout_mode: str = "single_readout"
    intermediate: AddressWindow | None = None

    def __post_init__(self):
        if type(self.producer) is not CountedConvolutionPlan or type(self.spatial) is not SpatialMaxReadout:
            raise ValueError("spatial producer requires original typed convolution and pool geometry")
        self.producer.__post_init__()
        self.spatial.__post_init__()
        if type(self.external_observers) is not tuple or self.external_observers:
            raise ValueError("producer has external observers: retain the original observed source route")
        p, s = self.producer, self.spatial
        if (
            p.window.output_spatial != s.input_hw
            or s.accumulator_bits != 32
            or s.output_bits != 8
            or type(self.output_tile_width) is not int
            or not 0 < self.output_tile_width <= s.output_hw[1]
        ):
            raise ValueError("spatial stripe differs from complete original source geometry or signed ABI")
        blocks = (p.output_channels + p.column_tile - 1) // p.column_tile
        if (
            type(self.readouts) is not tuple
            or len(self.readouts) != blocks
            or any(type(chain) is not SpatialReadoutChain for chain in self.readouts)
        ):
            raise ValueError("one exact source readout chain per complete channel tile is required")
        if self.readout_mode == "single_readout":
            if self.intermediate is not None:
                raise ValueError("single readout does not consume intermediate byte storage")
            for chain in self.readouts:
                prove_spatial_readout(chain)
        elif self.readout_mode == "staged_readouts":
            if type(self.intermediate) is not AddressWindow:
                raise ValueError("staged readouts require explicit private intermediate byte storage")
            self.intermediate.__post_init__()
            if self.intermediate.size < self.temporary_rows * p.column_tile:
                raise ValueError("staged readout scratch does not cover the complete i8 halo stripe")
            if self.intermediate.start + self.intermediate.size >= 1 << 63:
                raise ValueError("staged readout scratch exceeds signed i64 indices")
            for chain in self.readouts:
                chain.__post_init__()
                if chain.producer is None:
                    raise ValueError("staged readouts retain an original rounded i8 producer boundary")
                if chain.replacement is not None:
                    raise ValueError("staged readouts do not consume a collapsed replacement")
        else:
            raise ValueError("unknown spatial readout realization")
        h, w = s.input_hw
        kh, kw = s.window
        sh, sw = s.stride
        pt, pl, _, _ = s.padding
        oh, ow = s.output_hw
        if any(s.padding) and s.pad_value != -128:
            raise ValueError("halo border duplication requires minimum-valued post-readout signed-i8 padding")
        if kh <= pt or kw <= pl or (oh - 1) * sh - pt >= h or (ow - 1) * sw - pl >= w:
            raise ValueError("spatial producer cannot replace completely padded windows")
        if p.accumulator_row + self.temporary_rows > p.accumulator_rows:
            raise ValueError("halo stripe exceeds assigned private accumulator capacity")
        if (
            max(
                self.temporary_rows * p.primitive * 4,
                p.window.input_shape[0] * oh * ow * p.output_channels,
                h + kh,
                w + kw,
                oh * sh + kh,
                ow * sw + kw,
            )
            >= 1 << 63
        ):
            raise ValueError("spatial producer index arithmetic exceeds signed i64")

    @property
    def temporary_rows(self):
        kh, kw = self.spatial.window
        return kh * ((self.output_tile_width - 1) * self.spatial.stride[1] + kw)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def coverage(self):
        p, s = self.producer, self.spatial
        oh, ow = s.output_hw
        return dict(
            output_elements=p.window.input_shape[0] * oh * ow * p.output_channels,
            output_x_partition=[ow // self.output_tile_width, ow % self.output_tile_width],
            channel_partition=[p.output_channels // p.column_tile, p.output_channels % p.column_tile],
            private_accumulator_rows=self.temporary_rows,
            private_accumulator_bytes=self.temporary_rows * p.primitive * 4,
            full_intermediate_image_bytes=0,
            private_intermediate_bytes=0 if self.intermediate is None else self.temporary_rows * p.column_tile,
            host_payload_operations=0,
            source_reduction_order=["kernel_y", "kernel_x", "input_channel"],
            overlap_policy="recompute pure producer halo; no observable intermediate writes",
            reuse="only after synchronous final spatial store completion",
        )


@dataclass(frozen=True)
class PackedSpatialProducerPlan:
    """A source-exact staged stripe with one reusable host input panel.

    The source plan retains the original observer and Q1/Q2 obligations. The
    new byte layout retains Q1 for *all* channels before the pool reloads it.
    Host capacity is explicitly caller-owned; it is not an accelerator fact.
    A dense callback must prove its original flattened kh/kw/channel products,
    bias and tiled Q1 readout separately, and complete before storage reuse.
    """

    source: SpatialProducerPlan
    panel_rows: int
    workspace: AddressWindow
    panel_alignment: int = 1

    def __post_init__(self):
        if type(self.source) is not SpatialProducerPlan or type(self.workspace) is not AddressWindow:
            raise ValueError("packed spatial producer requires original source and explicit private workspace")
        self.source.__post_init__()
        self.workspace.__post_init__()
        if self.source.readout_mode != "staged_readouts":
            raise ValueError("packed producer retains both original staged rounded readouts")
        if type(self.panel_rows) is not int or not 0 < self.panel_rows <= self.source.temporary_rows:
            raise ValueError("packed producer requires a positive bounded halo panel")
        if (
            type(self.panel_alignment) is not int
            or not 0 < self.panel_alignment < 1 << 63
            or self.panel_alignment & (self.panel_alignment - 1)
        ):
            raise ValueError("packed producer requires power-of-two panel alignment")
        if self.workspace.start % self.panel_alignment:
            raise ValueError("packed producer workspace start is not panel-aligned")
        if self.required_bytes > self.workspace.size or self.panel_offset + self.panel_bytes >= 1 << 63:
            raise ValueError("packed producer exceeds caller-owned private workspace capacity")

    @property
    def reduction(self):
        w = self.source.producer.window
        return prod(w.kernel) * w.input_shape[-1]

    @property
    def stripe_bytes(self):
        return self.source.temporary_rows * self.source.producer.output_channels

    @property
    def panel_offset(self):
        end = self.workspace.start + self.stripe_bytes
        return (end + self.panel_alignment - 1) // self.panel_alignment * self.panel_alignment

    @property
    def panel_bytes(self):
        return self.panel_rows * self.reduction

    @property
    def required_bytes(self):
        return self.panel_offset + self.panel_bytes - self.workspace.start

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def read_spatial_plan_record(record, *, kind, expected_identity):
    """Decode exact original plan data under an independently selected kind.

    A frozen old owner selects ``staged_gather``; a prospective owner must
    explicitly select ``staged_packed`` and its larger storage contract. The
    record cannot select its own authority. Its expected identity must come
    from the original frozen owner, not an author request. This decoder checks
    geometry/resources and custody only, never source or target qualification.
    """
    from .spatial_readout import RoundedReadout
    from .window_transfer import Window2D

    if kind not in ("staged_gather", "staged_packed") or type(kind) is not str:
        raise ValueError("original owner must explicitly select a declared spatial plan kind")
    if (
        type(expected_identity) is not str
        or len(expected_identity) != 64
        or any(c not in "0123456789abcdef" for c in expected_identity)
    ):
        raise ValueError("spatial plan requires the original owner's exact custody identity")

    def exact(value, cls):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("spatial plan record has missing or unknown " + cls.__name__ + " fields")
        return dict(value)

    def stage(value):
        if value is None:
            return None
        data = exact(value, RoundedReadout)
        if type(data["scales"]) is not list:
            raise ValueError("spatial readout record requires literal scale sequence")
        return RoundedReadout(tuple(data["scales"]), data["relu"])

    def source(value):
        data = exact(value, SpatialProducerPlan)
        producer = exact(data["producer"], CountedConvolutionPlan)
        window = exact(producer["window"], Window2D)
        if any(type(v) is not list for v in window.values()):
            raise ValueError("spatial source window record requires explicit coordinate sequences")
        producer["window"] = Window2D(**{k: tuple(v) for k, v in window.items()})
        spatial = exact(data["spatial"], SpatialMaxReadout)
        for name in ("input_hw", "window", "stride", "padding"):
            if type(spatial[name]) is not list:
                raise ValueError("spatial pool record requires explicit coordinate sequences")
            spatial[name] = tuple(spatial[name])
        if type(data["readouts"]) is not list or type(data["external_observers"]) is not list:
            raise ValueError("spatial source record requires complete readout and observer sequences")
        readouts = tuple(
            SpatialReadoutChain(**{k: stage(v) for k, v in exact(row, SpatialReadoutChain).items()})
            for row in data["readouts"]
        )
        data.update(
            producer=CountedConvolutionPlan(**producer),
            spatial=SpatialMaxReadout(**spatial),
            readouts=readouts,
            external_observers=tuple(data["external_observers"]),
            intermediate=(
                None if data["intermediate"] is None else AddressWindow(**exact(data["intermediate"], AddressWindow))
            ),
        )
        selected = SpatialProducerPlan(**data)
        if selected.readout_mode != "staged_readouts":
            raise ValueError("original spatial plan record must retain staged Q1/Q2 boundaries")
        return selected

    if kind == "staged_gather":
        selected = source(record)
    else:
        data = exact(record, PackedSpatialProducerPlan)
        data["source"] = source(data["source"])
        data["workspace"] = AddressWindow(**exact(data["workspace"], AddressWindow))
        selected = PackedSpatialProducerPlan(**data)
    if selected.identity != expected_identity:
        raise ValueError("spatial plan record differs from the original owner's exact identity")
    return selected


def emit_packed_spatial_producer(plan, builder, lower, *, synchronous_completion):
    """Issue host transport, closed dense helpers, then the original pool.

    This real counted-loop driver changes the callback command order: each
    complete K panel produces all N Q1 bytes; only after every panel completes
    can Q2/pool consume the stripe. Callbacks own target encoding, source-bound
    dense product/readout proof, and completion. This is not an async grant.
    """
    if type(plan) is not PackedSpatialProducerPlan or synchronous_completion is not True:
        raise ValueError("packed spatial producer requires a typed plan and synchronous callback contract")
    plan.__post_init__()
    if any(not callable(getattr(builder, name, None)) for name in ("op", "loop")) or not callable(lower):
        raise ValueError("packed spatial producer requires counted integer control and actual callbacks")
    identity = plan.identity
    original = plan.source
    p, s = original.producer, original.spatial
    oh, ow = s.output_hw

    def op(name, lhs, rhs):
        return builder.op(name, lhs, rhs)

    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            builder.loop(0, whole, 1, lambda index: body(op("mul", index, tile), tile))
        if tail:
            body(whole * tile, tail)

    def image_body(image):
        def y_body(y):
            def x_body(x, output_columns):
                width = (output_columns - 1) * s.stride[1] + s.window[1]
                halo_rows = s.window[0] * width

                def panel_body(first, rows):
                    fields = dict(
                        image=image,
                        output_y=y,
                        output_x=x,
                        output_pixels=output_columns,
                        first_halo_row=first,
                        rows=rows,
                        reduction=plan.reduction,
                        columns=p.output_channels,
                        panel_offset=plan.panel_offset,
                        stripe_offset=op("add", plan.workspace.start, op("mul", first, p.output_channels)),
                        stripe_row_stride=p.output_channels,
                    )
                    lower("pack_input_panel", dict(fields))
                    lower("produce_packed_panel", dict(fields))

                split(halo_rows, plan.panel_rows, panel_body)

                def channel_body(column, columns):
                    def reload(first, rows):
                        lower(
                            "reload_intermediate",
                            dict(
                                accumulator_row=op("add", p.accumulator_row, first),
                                scratch_offset=op(
                                    "add", plan.workspace.start, op("add", op("mul", first, p.output_channels), column)
                                ),
                                scratch_row_stride=p.output_channels,
                                rows=rows,
                                columns=columns,
                                stripe_rows=halo_rows,
                                output_column=column,
                                readout_tile=op("udiv", column, p.column_tile),
                            ),
                        )

                    split(halo_rows, p.row_tile, reload)
                    lower(
                        "store_spatial_max",
                        dict(
                            output_row=op("add", op("mul", op("add", op("mul", image, oh), y), ow), x),
                            output_column=column,
                            columns=columns,
                            output_pixels=output_columns,
                            accumulator_row=p.accumulator_row,
                            local_height=s.window[0],
                            local_width=width,
                            window_height=s.window[0],
                            window_width=s.window[1],
                            stride_width=s.stride[1],
                            readout_tile=op("udiv", column, p.column_tile),
                        ),
                    )

                split(p.output_channels, p.column_tile, channel_body)

            split(ow, original.output_tile_width, x_body)

        builder.loop(0, oh, 1, y_body)

    builder.loop(0, p.window.input_shape[0], 1, image_body)
    plan.__post_init__()
    if identity != plan.identity:
        raise ValueError("packed spatial source or private storage changed during emission")
    return dict(
        schema="merlin.packed_spatial_producer_emission.v1",
        plan_sha256=identity,
        original_plan_sha256=original.identity,
        original_readout_chains=[asdict(c) for c in original.readouts],
        output_elements=p.window.input_shape[0] * oh * ow * p.output_channels,
        host_panel_shape=[plan.panel_rows, plan.reduction],
        private_panel_bytes=plan.panel_bytes,
        private_q1_stripe_bytes=plan.stripe_bytes,
        private_workspace_bytes=plan.required_bytes,
        source_reduction_order=["kernel_y", "kernel_x", "input_channel"],
        completion="synchronous dense helper before panel reuse; Q1 stripe before Q2/pool; pool before reuse",
        source_and_target_qualification="separate original-source and target callback obligations",
    )


def emit_spatial_producer(plan, builder, lower, *, synchronous_completion):
    """Emit actual counted index/control IR and every original product callback.

    The physical adapter must bind dynamic accumulator addresses, exact readout
    tile metadata, local pool rectangle, complete pointer ABI and synchronous
    effects. The source matcher separately proves that the producer is pure,
    its inputs immutable, and no eliminated intermediate has another observer.
    Failure of any source or physical obligation retains the original route.
    """
    if type(plan) is not SpatialProducerPlan or synchronous_completion is not True:
        raise ValueError("spatial producer requires a checked plan and synchronous target realization")
    plan.__post_init__()
    identity = plan.identity
    if any(not callable(getattr(builder, name, None)) for name in ("op", "cmp", "select", "loop", "branch")):
        raise ValueError("spatial producer requires an integer index/control builder")
    if not callable(lower):
        raise ValueError("spatial producer requires actual target callbacks")
    p, s = plan.producer, plan.spatial
    batches = p.window.input_shape[0]
    h, w = s.input_hw
    kh, kw = s.window
    sh, sw = s.stride
    pt, pl, _, _ = s.padding
    oh, ow = s.output_hw

    def op(name, a, b):
        return builder.op(name, a, b)

    def clip(value, extent):
        # Signed-negative and signed-positive boundaries are distinct. ULT
        # alone would classify negative indices as an upper-border value.
        positive = builder.select(builder.cmp("slt", value, 0), 0, value)
        return builder.select(builder.cmp("ult", positive, extent), positive, extent - 1)

    def split(extent, tile, body):
        whole, tail = divmod(extent, tile)
        if whole:
            builder.loop(0, whole, 1, lambda index: body(op("mul", index, tile), tile))
        if tail:
            body(whole * tile, tail)

    def image_body(image):
        def output_y_body(y):
            def output_x_body(x, output_columns):
                stripe_width = (output_columns - 1) * sw + kw
                base_x = op("sub", op("mul", x, sw), pl)
                base_y = op("sub", op("mul", y, sh), pt)

                def channel_body(column, columns):
                    def stripe_y_body(ky):
                        iy = clip(op("add", base_y, ky), h)

                        def stripe_x_body(local_x, rows):
                            raw_x = op("add", base_x, local_x)
                            last_x = op("add", raw_x, rows - 1)
                            valid = builder.op("and", builder.cmp("ult", raw_x, w), builder.cmp("ult", last_x, w), "i1")

                            def produce(ix, count, local):
                                global_row = op("add", op("mul", op("add", op("mul", image, h), iy), w), ix)
                                accumulator = op(
                                    "add", p.accumulator_row, op("add", op("mul", ky, stripe_width), local)
                                )

                                def effect(kind, fields):
                                    lower("finish_raw_tile" if kind == "store" else kind, fields)

                                emit_convolution_tile(
                                    p,
                                    builder,
                                    effect,
                                    output_row=global_row,
                                    output_column=column,
                                    rows=count,
                                    columns=columns,
                                    accumulator_row=accumulator,
                                    synchronous_completion=True,
                                )

                            def interior():
                                produce(raw_x, rows, local_x)

                            def boundary():
                                builder.loop(
                                    0,
                                    rows,
                                    1,
                                    lambda index: produce(
                                        clip(op("add", raw_x, index), w), 1, op("add", local_x, index)
                                    ),
                                )

                            builder.branch(valid, interior, boundary)

                        split(stripe_width, p.row_tile, stripe_x_body)

                    builder.loop(0, kh, 1, stripe_y_body)
                    if plan.readout_mode == "staged_readouts":
                        # The first complete loop publishes every Q1 byte before
                        # a second loop overwrites any raw accumulator with i8.
                        # Each physical callback must complete synchronously.
                        def stage_row(ky, kind):
                            def stage_x(local_x, rows):
                                local = op("add", op("mul", ky, stripe_width), local_x)
                                lower(
                                    kind,
                                    dict(
                                        accumulator_row=op("add", p.accumulator_row, local),
                                        scratch_offset=op(
                                            "add", plan.intermediate.start, op("mul", local, p.column_tile)
                                        ),
                                        scratch_row_stride=p.column_tile,
                                        rows=rows,
                                        columns=columns,
                                        stripe_rows=kh * stripe_width,
                                        output_column=column,
                                        readout_tile=op("udiv", column, p.column_tile),
                                    ),
                                )

                            split(stripe_width, p.row_tile, stage_x)

                        builder.loop(0, kh, 1, lambda ky: stage_row(ky, "store_intermediate"))
                        builder.loop(0, kh, 1, lambda ky: stage_row(ky, "reload_intermediate"))
                    output_row = op("add", op("mul", op("add", op("mul", image, oh), y), ow), x)
                    lower(
                        "store_spatial_max",
                        dict(
                            output_row=output_row,
                            output_column=column,
                            columns=columns,
                            output_pixels=output_columns,
                            accumulator_row=p.accumulator_row,
                            local_height=kh,
                            local_width=stripe_width,
                            window_height=kh,
                            window_width=kw,
                            stride_width=sw,
                            readout_tile=op("udiv", column, p.column_tile),
                        ),
                    )

                split(p.output_channels, p.column_tile, channel_body)

            split(ow, plan.output_tile_width, output_x_body)

        builder.loop(0, oh, 1, output_y_body)

    builder.loop(0, batches, 1, image_body)
    plan.__post_init__()
    if plan.identity != identity:
        raise ValueError("spatial producer source or private resources changed during emission")
    return dict(
        schema="merlin.spatial_producer_emission.v1",
        plan_sha256=identity,
        readout_proofs=(
            [prove_spatial_readout(chain) for chain in plan.readouts] if plan.readout_mode == "single_readout" else []
        ),
        readout_mode=plan.readout_mode,
        retained_readout_chains=[asdict(chain) for chain in plan.readouts],
        **plan.coverage,
        scope="actual compact index/product emission; source binding and target execution remain separate",
    )
