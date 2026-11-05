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
class SpatialProducerBand:
    """An explicit staged Q1/Q2 band within the original assigned stores.

    Neighboring bands may recompute pure halo products. This describes exact
    storage/work costs, without selecting a profitable height or granting a
    target multirow pool command. Every original observer/quantizer remains.
    """
    source: SpatialProducerPlan
    output_rows: int

    def __post_init__(self):
        if type(self.source) is not SpatialProducerPlan:
            raise ValueError("spatial band requires the complete original producer source")
        self.source.__post_init__()
        p,s=self.source.producer,self.source.spatial
        if (type(self.output_rows) is not int or not 0<self.output_rows<=s.output_hw[0]
                or self.source.readout_mode!='staged_readouts'
                or p.accumulator_row+self.temporary_rows>p.accumulator_rows
                or self.source.intermediate.size<self.temporary_rows*p.column_tile):
            raise ValueError("spatial band needs both original readouts and complete assigned ACC/Q1 capacity")

    @property
    def height(self):
        return (self.output_rows-1)*self.source.spatial.stride[0]+self.source.spatial.window[0]

    @property
    def width(self):
        return (self.source.output_tile_width-1)*self.source.spatial.stride[1]+self.source.spatial.window[1]

    @property
    def temporary_rows(self):
        return self.height*self.width

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def report(self):
        self.__post_init__()
        p,s=self.source.producer,self.source.spatial
        oh,ow=s.output_hw;kh,kw=s.window;sh,sw=s.stride
        whole,tail=divmod(oh,self.output_rows)
        producer_y=whole*self.height+((tail-1)*sh+kh if tail else 0)
        wide,last=divmod(ow,self.source.output_tile_width)
        producer_x=wide*self.width+((last-1)*sw+kw if last else 0)
        pixels=p.window.input_shape[0]*producer_y*producer_x
        return dict(schema='merlin.spatial_producer_band.v1',band_sha256=self.identity,
            original_plan_sha256=self.source.identity,output_rows=self.output_rows,
            producer_height=self.height,producer_width=self.width,
            private_accumulator_rows=self.temporary_rows,private_intermediate_bytes=self.temporary_rows*p.column_tile,
            producer_halo_pixels=pixels,unique_producer_pixels=p.window.input_shape[0]*prod(s.input_hw),
            logical_macs=pixels*prod(p.window.kernel)*p.window.input_shape[-1]*p.output_channels,
            overlap_policy='bounded pure halo recomputation between bands; no external intermediate observers',
            arithmetic='original Q1 store/reload then original Q2 and signed spatial maximum',
            completion='complete every source effect and full pooled rectangle before ACC/Q1 reuse',
            target_qualification=False,timing=None)


def spatial_producer_band_choices(source, *, max_output_rows):
    """Enumerate complete legal band heights from actual ACC and Q1 capacity."""
    if type(source) is not SpatialProducerPlan:
        raise ValueError("spatial bands require the original typed source")
    source.__post_init__()
    if type(max_output_rows) is not int or max_output_rows<=0 or source.readout_mode!='staged_readouts':
        raise ValueError("spatial bands need an explicit positive bound and both original readouts")
    p,s=source.producer,source.spatial
    width=(source.output_tile_width-1)*s.stride[1]+s.window[1]
    rows=min(p.accumulator_rows-p.accumulator_row,source.intermediate.size//p.column_tile)
    height=rows//width
    bound=min(max_output_rows,s.output_hw[0],1+(height-s.window[0])//s.stride[0])
    return tuple(SpatialProducerBand(source,n)for n in range(1,bound+1))


@dataclass(frozen=True)
class SpatialProducerBandAdmission:
    """Selected rectangular pool addressing/tracking and completed reuse law."""
    source_identity: str
    hardware_identity: str
    qualification_identity: str
    maximum_output_rows: int
    maximum_output_columns: int
    maximum_source_rows: int
    maximum_source_columns: int
    maximum_pool_row_requests: int
    original_q1_q2: bool = False
    exclusive_accumulator_and_scratch: bool = False
    rectangular_pool_mapping: bool = False
    complete_before_reuse: bool = False

    def require(self,band):
        band.__post_init__()
        if (self.source_identity!=band.source.identity
                or any(type(v)is not str or not v.strip()for v in
                       (self.source_identity,self.hardware_identity,self.qualification_identity))
                or any(type(v)is not int or not 0<v<1<<63 for v in
                       (self.maximum_output_rows,self.maximum_output_columns,self.maximum_source_rows,
                        self.maximum_source_columns,self.maximum_pool_row_requests))
                or any(type(v)is not bool or not v for v in
                       (self.original_q1_q2,self.exclusive_accumulator_and_scratch,
                        self.rectangular_pool_mapping,self.complete_before_reuse))
                or band.output_rows>self.maximum_output_rows
                or band.source.output_tile_width>self.maximum_output_columns
                or band.height>self.maximum_source_rows or band.width>self.maximum_source_columns
                or band.output_rows*band.source.output_tile_width*prod(band.source.spatial.window)>self.maximum_pool_row_requests):
            raise ValueError("spatial band needs selected complete rectangular pool mapping, tracking and Q1/Q2 lifetime admission")


@dataclass(frozen=True)
class SpatialProducerBandAlternative:
    """Checked band and independent original RHS storage choice."""
    band: SpatialProducerBand
    rhs: object = None

    def verify_current(self):
        if type(self.band)is not SpatialProducerBand:
            raise ValueError("spatial alternative requires its complete original band")
        self.band.__post_init__()
        if self.rhs is not None:
            from .convolution_exploration import ConvolutionRhsResidency
            p=self.band.source.producer;rhs=self.rhs
            expected=tuple((n,min(p.column_tile,p.output_channels-n))for n in range(0,p.output_channels,p.column_tile))
            if (type(rhs)is not ConvolutionRhsResidency or rhs.column_intervals is not None
                    or rhs.schedule.source.plan!=p or rhs.schedule.source.source_identity!=self.band.source.identity
                    or rhs.schedule.source.column_intervals!=expected):
                raise ValueError("spatial band RHS differs from its complete original source")
            rhs.verify_current()

    @property
    def identity(self):
        self.verify_current()
        value=(self.band.identity,None if self.rhs is None else self.rhs.identity)
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def spatial_producer_pass_space(source, *, output_row_limits, rhs_choices=(None,),
                               verify_source,realize,max_candidates=8):
    """Register independent band/RHS planning before actual source emission.

    Existing prerequisite ordering and emitted-byte dedup remain authoritative.
    ``realize(alternative)`` must consume both checked owners and return an
    EmittedProgram bound to alternative.identity. Neither counts nor these
    choices supply a cycle score; original source/target review remains separate.
    """
    from merlin.transforms.pass_exploration import EmittedProgram,PassFamily,generate_candidates
    from merlin.transforms.pipeline_policy import RegisteredPass
    from merlin.transforms.pipeline_state import PassSpec
    if (type(source)is not SpatialProducerPlan or not callable(verify_source)or not callable(realize)
            or type(output_row_limits)is not tuple or not 1<=len(output_row_limits)<=16
            or len(set(output_row_limits))!=len(output_row_limits)
            or type(rhs_choices)is not tuple or not 1<=len(rhs_choices)<=16):
        raise ValueError("spatial exploration requires bounded original band/storage choices and real callbacks")
    source.__post_init__();identity=source.identity
    for rows in output_row_limits:
        for rhs in rhs_choices:SpatialProducerBandAlternative(SpatialProducerBand(source,rows),rhs).verify_current()
    rhs_identities=tuple(None if rhs is None else rhs.identity for rhs in rhs_choices)
    if len(set(rhs_identities))!=len(rhs_identities):
        raise ValueError("spatial exploration requires distinct checked RHS choices")
    def reopen(current):
        if type(current)is not SpatialProducerPlan or current.identity!=identity:
            raise ValueError("spatial exploration original source changed")
        if verify_source(current)is not None or current.identity!=identity:
            raise ValueError("spatial exploration source callback did not preserve/verify the original")
        if tuple(None if rhs is None else rhs.identity for rhs in rhs_choices)!=rhs_identities:
            raise ValueError("spatial exploration RHS ownership changed")
    def band(options):
        def action(session):
            reopen(session.module)
            return {'spatial_band':SpatialProducerBand(session.module,options['output_rows'])}
        return RegisteredPass(PassSpec('spatial_producer_band',0,produces=('spatial_band',),
            mutates_ir=False,ir_level='spatial_producer_source'),action)
    def storage(options):
        def action(session):
            reopen(session.module)
            return {'spatial_rhs':rhs_choices[options['rhs']]}
        return RegisteredPass(PassSpec('spatial_producer_rhs',0,produces=('spatial_rhs',),
            mutates_ir=False,ir_level='spatial_producer_source'),action)
    def emit(_):
        def action(session):
            reopen(session.module)
            value=SpatialProducerBandAlternative(session.analysis('spatial_band'),session.analysis('spatial_rhs'))
            selected=value.identity;result=realize(value)
            reopen(session.module)
            if type(result)is not EmittedProgram or result.schedule_sha256!=selected or value.identity!=selected:
                raise ValueError("spatial program is not bound to its actual band/RHS owners")
            return {'spatial_producer_program':result}
        return RegisteredPass(PassSpec('realize_spatial_producer',1,requires=('spatial_band','spatial_rhs'),
            produces=('spatial_producer_program',),mutates_ir=False,ir_level='spatial_producer_source'),action)
    return generate_candidates((PassFamily('band',band,knobs=(('output_rows',output_row_limits),)),
        PassFamily('rhs',storage,knobs=(('rhs',tuple(range(len(rhs_choices)))),)),PassFamily('emit',emit)),
        output='spatial_producer_program',max_candidates=max_candidates)


def explore_spatial_producer(source,space,*,context,verify_source,feedback=None):
    """Use the existing explorer, actual emitted bytes and optional real cycles."""
    import copy
    from merlin.transforms.pass_exploration import explore_candidates
    from merlin.transforms.pipeline_policy import PolicyFacts
    def verify(current):
        if type(current)is not SpatialProducerPlan:
            raise ValueError("spatial exploration source type changed")
        current.__post_init__()
        if verify_source(current)is not None:
            raise ValueError("spatial exploration source did not verify")
    def facts(session):
        p=session.module.producer
        return PolicyFacts(session.stamp(),shapes=(p.window.input_shape,
            (*p.window.kernel,p.window.input_shape[-1],p.output_channels)),
            operations=frozenset({'convolution','spatial_max'}),
            resources=(('operand_rows',p.operand_rows),('accumulator_rows',p.accumulator_rows)))
    return explore_candidates(source,space,facts,output='spatial_producer_program',
        emission=lambda result:result.artifact,feedback=feedback,context=context,
        ir_level='spatial_producer_source',clone=copy.deepcopy,
        fingerprint=lambda current:current.identity,verify_ir=verify)


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
    return _emit_spatial_producer(plan,builder,lower,synchronous_completion=synchronous_completion)


def emit_spatial_producer_band(band, builder, lower, *, synchronous_completion, admission,
                              rhs=None,rhs_complete=None):
    """Emit a real bounded producer union, Q1 reload and multirow pool readout.

    The adapter must independently admit actual ``output_rows``, stride,
    rectangle addressing, command tracking and full completion. All assigned
    rows and the original source/readout identity are checked before effects.
    The ordinary connected route uses ``band.source`` for its unchanged full
    logical ABI and caller-owned scratch; this choice supplies no host effects.
    An optional complete original RHS owner composes the existing preparation
    wrapper with this consumer. Its separate immutable storage law and actual
    preparation completion remain required. No RHS phase is inferred.
    """
    if type(band) is not SpatialProducerBand:
        raise ValueError("spatial band emission requires its complete typed choice")
    if type(admission)is not SpatialProducerBandAdmission:
        raise ValueError("spatial band requires its separately selected target admission")
    if (synchronous_completion is not True or not callable(lower)
            or any(not callable(getattr(builder,name,None))
                   for name in ('op','cmp','select','loop','branch'))):
        raise ValueError("spatial band requires an actual synchronous target realization")
    admission.require(band);target_identity=asdict(admission)
    band.__post_init__();identity=band.identity
    finish_rhs=None
    if rhs is not None:
        from .convolution_exploration import emit_convolution_rhs_preparation
        SpatialProducerBandAlternative(band,rhs).verify_current()
        if (not callable(rhs_complete)
                or rhs.law.hardware_identity!=admission.hardware_identity):
            raise ValueError("spatial band RHS preparation differs from the complete original source or selected target")
        rhs.verify_current()
        lower,finish_rhs=emit_convolution_rhs_preparation(rhs,builder,lower,complete=rhs_complete)
    result=_emit_spatial_producer(band.source,builder,lower,
        synchronous_completion=synchronous_completion,output_rows=band.output_rows)
    band.__post_init__()
    if band.identity!=identity:
        raise ValueError("spatial source band changed during actual emission")
    admission.require(band)
    if asdict(admission)!=target_identity:
        raise ValueError("spatial band target admission changed during actual emission")
    if finish_rhs is not None:finish_rhs()
    return dict(result,**band.report(),admission=target_identity,
                rhs_preparation=None if rhs is None else rhs.summary())


def _emit_spatial_producer(plan,builder,lower,*,synchronous_completion,output_rows=1):
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
        def output_y_body(y,rows=1):
            def output_x_body(x, output_columns):
                stripe_width = (output_columns - 1) * sw + kw
                stripe_height = (rows-1)*sh+kh
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

                    builder.loop(0, stripe_height, 1, stripe_y_body)
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
                                        stripe_rows=stripe_height * stripe_width,
                                        output_column=column,
                                        readout_tile=op("udiv", column, p.column_tile),
                                    ),
                                )

                            split(stripe_width, p.row_tile, stage_x)

                        builder.loop(0, stripe_height, 1, lambda ky: stage_row(ky, "store_intermediate"))
                        builder.loop(0, stripe_height, 1, lambda ky: stage_row(ky, "reload_intermediate"))
                    output_row = op("add", op("mul", op("add", op("mul", image, oh), y), ow), x)
                    lower(
                        "store_spatial_max",
                        dict(
                            output_row=output_row,
                            output_column=column,
                            columns=columns,
                            output_pixels=output_columns,
                            accumulator_row=p.accumulator_row,
                            local_height=stripe_height,
                            local_width=stripe_width,
                            window_height=kh,
                            window_width=kw,
                            stride_width=sw,
                            readout_tile=op("udiv", column, p.column_tile),
                            **(dict(output_rows=rows,stride_height=sh)if output_rows>1 else {}),
                        ),
                    )

                split(p.output_channels, p.column_tile, channel_body)

            split(ow, plan.output_tile_width, output_x_body)

        if output_rows==1:
            builder.loop(0, oh, 1, output_y_body)
        else:
            split(oh,output_rows,output_y_body)

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
