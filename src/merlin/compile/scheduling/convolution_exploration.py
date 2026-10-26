"""Bounded source convolution choices through the ordinary pass registry.

This module supplies no target facts, numerical recipe, completion mechanism or
preferred schedule. Targets reopen original source, declare private resources
and effects, and realize the resulting counted schedule as actual program bytes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from merlin.transforms.pass_exploration import (
    EmittedProgram,
    PassFamily,
    explore_candidates,
    generate_candidates,
)
from merlin.transforms.pipeline_policy import PolicyFacts, RegisteredPass
from merlin.transforms.pipeline_state import PassSpec

from .dense_rhs_residency import DenseRhsResidency, plan_dense_rhs_residency
from .window_transfer import (
    ConvolutionComputeEpochLaw,
    ConvolutionInputRetentionLaw,
    ConvolutionLoadIssueLaw,
    ConvolutionOutputTile,
    CountedConvolutionPlan,
    emit_convolution_column_block,
    emit_convolution_compute_epoch_column_block,
    emit_convolution_load_issue_column_block,
    emit_convolution_load_issue_tile,
    emit_convolution_tile,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _law_digest(law):
    return _digest(asdict(law)) if law is not None else None


@dataclass(frozen=True)
class ConvolutionExplorationSource:
    """Original source partitions; scale/epilogue ownership stays with target.

    Intervals must cover all original output columns, without gaps or overlap,
    and must already respect the original scalar readout boundaries. Source
    identity is provenance, not proof. The registry's mandatory source callback
    must independently reopen and compare the selected original source.
    """
    plan: CountedConvolutionPlan
    column_intervals: tuple[tuple[int, int], ...]
    source_identity: str

    def __post_init__(self):
        if (type(self.plan) is not CountedConvolutionPlan
            or type(self.source_identity) is not str or not self.source_identity.strip()
            or type(self.column_intervals) is not tuple or not self.column_intervals):
            raise ValueError("convolution exploration requires original typed source and intervals")
        self.plan.__post_init__()
        end = 0
        for interval in self.column_intervals:
            if (type(interval) is not tuple or len(interval) != 2
                or any(type(v) is not int for v in interval)
                or interval[0] != end or not 0 < interval[1] <= self.plan.column_tile):
                raise ValueError("convolution source intervals must be bounded and contiguous")
            end += interval[1]
        if end != self.plan.output_channels:
            raise ValueError("convolution source intervals do not cover every original output column")

    @property
    def identity(self):
        self.__post_init__()
        return _digest(asdict(self))


@dataclass(frozen=True)
class ConvolutionScheduleChoice:
    """An explicit compiler choice; None uses only declared private capacity."""
    live_tiles: int | None = 1
    load_issue: bool = False
    compute_pairs: int = 1

    def __post_init__(self):
        if (self.live_tiles is not None and
            (type(self.live_tiles) is not int or not 0 < self.live_tiles < 2**31)
            or type(self.load_issue) is not bool or type(self.compute_pairs) is not int
            or not 0 < self.compute_pairs < 2**31):
            raise ValueError("convolution choices need a bounded tile limit and explicit issue mode")


def _column_blocks(source, choice):
    source.__post_init__()
    choice.__post_init__()
    p = source.plan
    capacity = (p.accumulator_rows-p.accumulator_row)//p.row_tile
    slots = capacity if choice.live_tiles is None else min(capacity, choice.live_tiles)
    return tuple(tuple(ConvolutionOutputTile(column, width, p.accumulator_row+i*p.row_tile)
                       for i, (column, width) in enumerate(source.column_intervals[start:start+slots]))
                 for start in range(0, len(source.column_intervals), slots))


@dataclass(frozen=True)
class ConvolutionSchedule:
    """An immutable counted lifetime schedule, checked before each emission."""
    source: ConvolutionExplorationSource
    choice: ConvolutionScheduleChoice
    column_blocks: tuple[tuple[ConvolutionOutputTile, ...], ...]
    retention_law: ConvolutionInputRetentionLaw | None = None
    issue_law: ConvolutionLoadIssueLaw | None = None
    epoch_law: ConvolutionComputeEpochLaw | None = None
    _source_sha256: str = field(init=False, repr=False)
    _retention_sha256: str | None = field(init=False, repr=False)
    _issue_sha256: str | None = field(init=False, repr=False)
    _epoch_sha256: str | None = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_source_sha256", self.source.identity)
        object.__setattr__(self, "_retention_sha256", _law_digest(self.retention_law))
        object.__setattr__(self, "_issue_sha256", _law_digest(self.issue_law))
        object.__setattr__(self, "_epoch_sha256", _law_digest(self.epoch_law))

    def _validate(self):
        if (type(self.source) is not ConvolutionExplorationSource
            or type(self.choice) is not ConvolutionScheduleChoice
            or self.column_blocks != _column_blocks(self.source, self.choice)):
            raise ValueError("convolution lifetime schedule differs from its original source/resources")
        count = max(map(len, self.column_blocks))
        r, i = self.retention_law, self.issue_law
        if r is not None:
            if type(r) is not ConvolutionInputRetentionLaw:
                raise ValueError("convolution retention law must be typed")
            r.__post_init__()
        if i is not None:
            if type(i) is not ConvolutionLoadIssueLaw:
                raise ValueError("convolution issue law must be typed")
            i.__post_init__()
        if (count > 1 or self.choice.compute_pairs > 1) and (r is None or count > r.max_live_tiles
                         or not all((r.input_preserved, r.independent_outputs,
                                     r.immutable_mapped_disjoint_buffers))):
            raise ValueError("convolution schedule lacks original input-retention premises")
        if self.choice.load_issue:
            if i is None or not all((i.local_range_dependencies, i.ordered_load_configuration,
                i.paired_descriptor_compute, i.finite_queue_backpressure,
                i.complete_device_drain, i.immutable_mapped_disjoint_buffers)):
                raise ValueError("convolution schedule lacks original bounded-issue premises")
            if self.source.plan.row_tile+1+count*int(self.source.plan.bias) > i.load_issue_budget:
                raise ValueError("convolution schedule exceeds its original first-epoch load budget")
            if (count > 1 or self.choice.compute_pairs > 1) and (r.hardware_identity != i.hardware_identity
                             or r.numerical_policy_identity != i.numerical_policy_identity):
                raise ValueError("convolution retention and issue laws select different contracts")

        if self.epoch_law is not None:
            if type(self.epoch_law) is not ConvolutionComputeEpochLaw:
                raise ValueError("convolution compute epoch law must be typed")
            self.epoch_law.__post_init__()
        if self.choice.compute_pairs > 1:
            if not self.choice.load_issue or self.epoch_law is None:
                raise ValueError("convolution compute epochs need delayed issue and a stronger original source law")
            self.epoch_law.require(self.choice.compute_pairs)
            if (self.epoch_law.hardware_identity != i.hardware_identity
                or self.epoch_law.numerical_policy_identity != i.numerical_policy_identity):
                raise ValueError("convolution compute epoch and issue laws select different contracts")

    def verify_current(self):
        self._validate()
        if (self.source.identity != self._source_sha256
            or _law_digest(self.retention_law) != self._retention_sha256
            or _law_digest(self.issue_law) != self._issue_sha256
            or _law_digest(self.epoch_law) != self._epoch_sha256):
            raise ValueError("original convolution source or target laws changed after planning")

    def summary(self):
        self.verify_current()
        return dict(schema="merlin.convolution_schedule.v1", source_sha256=self._source_sha256,
                    plan_sha256=self.source.plan.identity, choice=asdict(self.choice),
                    column_blocks=[[asdict(tile) for tile in block] for block in self.column_blocks],
                    retention_law_sha256=self._retention_sha256, issue_law_sha256=self._issue_sha256,
                    epoch_law_sha256=self._epoch_sha256,
                    qualification="source/effect premises remain independently caller-owned")

    @property
    def identity(self):
        return _digest(self.summary())


def plan_convolution_schedule(source, choice=ConvolutionScheduleChoice(), *,
                              retention_law=None, issue_law=None, epoch_law=None):
    """Allocate actual live ACC intervals, clamping choices to private capacity.

    A one-tile synchronous fallback needs neither new effect law. Capacity alone
    cannot authorize input retention or delayed load completion. No performance
    choice or global winner is selected here.
    """
    if type(source) is not ConvolutionExplorationSource or type(choice) is not ConvolutionScheduleChoice:
        raise ValueError("convolution planning needs original typed source and choice")
    return ConvolutionSchedule(source, choice, _column_blocks(source, choice), retention_law, issue_law, epoch_law)


@dataclass(frozen=True)
class ConvolutionRhsResidencyLaw:
    """Selected transfer-only RHS recipe and invocation-private preservation.

    Source weights must remain mapped and immutable. Each original weight
    transfer/config callback has no observable effect except filling its given
    local rows. Other input, compute and readout callbacks preserve those rows.
    The caller establishes entry quiescence and implements a real full drain
    after preparation. This grants no cross-invocation residency.
    """
    hardware_identity: str
    numerical_policy_identity: str
    source_identity: str
    qualification_identity: str
    weight_transfer_only: bool = False
    immutable_mapped_disjoint_buffers: bool = False
    other_effects_preserve_rhs: bool = False
    exclusive_private_rows: bool = False
    entry_quiescence_and_complete_preparation: bool = False

    def __post_init__(self):
        if (any(type(v) is not str or not v.strip() for v in (
                self.hardware_identity, self.numerical_policy_identity,
                self.source_identity, self.qualification_identity))
            or any(type(v) is not bool for v in (self.weight_transfer_only,
                self.immutable_mapped_disjoint_buffers, self.other_effects_preserve_rhs,
                self.exclusive_private_rows, self.entry_quiescence_and_complete_preparation))):
            raise ValueError("convolution RHS residency requires typed original source premises")

    def require(self):
        self.__post_init__()
        if not all((self.weight_transfer_only, self.immutable_mapped_disjoint_buffers,
                    self.other_effects_preserve_rhs, self.exclusive_private_rows,
                    self.entry_quiescence_and_complete_preparation)):
            raise ValueError("convolution RHS residency lacks source preservation/completion premises")


@dataclass(frozen=True)
class ConvolutionRhsResidency:
    """Checked physical panel slots; original channel tails stay separate."""
    schedule: ConvolutionSchedule
    layout: DenseRhsResidency
    law: ConvolutionRhsResidencyLaw
    assigned_rhs_base: int | None = None
    _schedule_sha256: str = field(init=False, repr=False)
    _law_sha256: str = field(init=False, repr=False)
    _layout_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        if (type(self.schedule) is not ConvolutionSchedule or type(self.layout) is not DenseRhsResidency
            or type(self.law) is not ConvolutionRhsResidencyLaw):
            raise ValueError("convolution RHS residency requires original checked schedule and capacity plan")
        self.schedule.verify_current()
        self.law.require()
        if self.layout != _rhs_capacity(self.schedule, self.layout.operand_rows, self.assigned_rhs_base):
            raise ValueError("convolution RHS placement differs from original private capacity/source geometry")
        for law in (self.schedule.retention_law, self.schedule.issue_law, self.schedule.epoch_law):
            if law is not None and (law.hardware_identity != self.law.hardware_identity
                or law.numerical_policy_identity != self.law.numerical_policy_identity):
                raise ValueError("convolution RHS and execution laws select different contracts")
        object.__setattr__(self, "_schedule_sha256", self.schedule.identity)
        object.__setattr__(self, "_law_sha256", _law_digest(self.law))
        object.__setattr__(self, "_layout_sha256", _digest((self.assigned_rhs_base, asdict(self.layout))))

    def verify_current(self):
        self.schedule.verify_current()
        self.law.require()
        if (self.schedule.identity != self._schedule_sha256 or _law_digest(self.law) != self._law_sha256
            or _digest((self.assigned_rhs_base, asdict(self.layout))) != self._layout_sha256
            or self.layout != _rhs_capacity(self.schedule, self.layout.operand_rows, self.assigned_rhs_base)):
            raise ValueError("original convolution RHS source/law/private placement changed after planning")

    def summary(self):
        self.verify_current()
        return dict(schema="merlin.convolution_rhs_residency.v1", schedule_sha256=self._schedule_sha256,
            law_sha256=self._law_sha256, physical_panel_layout=asdict(self.layout),
            assigned_rhs_base=self.assigned_rhs_base,
            lifetime="one invocation; preparation fully completes before first consumer",
            reduction="original per-kernel channel chunks, including each independent tail")

    @property
    def identity(self):
        return _digest(self.summary())


def _rhs_capacity(schedule, operand_rows, rhs_base=None):
    p = schedule.source.plan
    if (type(operand_rows) is not int
        or not max(p.lhs_row+p.row_tile, p.rhs_row+p.channel_tile) <= operand_rows <= p.operand_rows):
        raise ValueError("convolution RHS rows must be explicitly owned within original operand capacity")
    kh, kw = p.window.kernel
    ci = p.window.input_shape[-1]
    chunks = kh*kw*((ci+p.channel_tile-1)//p.channel_tile)
    # Reuse the pure dense *capacity* planner for a physical tile-slot array.
    # These padded slot extents are not a flattened convolution or source shape:
    # CI tails at each kernel position and source scale intervals remain intact.
    base = (max(p.rhs_row, p.lhs_row+p.row_tile)+p.channel_tile-1)//p.channel_tile*p.channel_tile
    if rhs_base is not None:
        if (type(rhs_base) is not int or rhs_base < p.lhs_row+p.row_tile or rhs_base % p.channel_tile
            or rhs_base >= operand_rows):
            raise ValueError("assigned convolution RHS base must be aligned and disjoint within private rows")
        base = rhs_base
    return plan_dense_rhs_residency(1, chunks*p.channel_tile,
        len(schedule.source.column_intervals)*p.channel_tile,
        block=p.channel_tile, operand_rows=operand_rows, accumulator_rows=p.accumulator_rows,
        rhs_base=base, max_static_rhs_tiles=operand_rows//p.channel_tile)


def plan_convolution_rhs_residency(schedule, *, operand_rows, law, rhs_base=None):
    """Return a checked invocation-local placement, or None if it does not fit.

    Missing source/ownership premises refuse; insufficient declared capacity
    alone declines without changing the original schedule. Physical placement
    reuses the existing dense RHS capacity owner, not an assumed target layout.
    Optional rhs_base is a caller-assigned row offset, for example from selected
    physical bank intervals. It grants neither bank geometry nor parallelism.
    """
    if type(schedule) is not ConvolutionSchedule or type(law) is not ConvolutionRhsResidencyLaw:
        raise ValueError("convolution RHS preparation needs original schedule and separate source law")
    schedule.verify_current()
    law.require()
    layout = _rhs_capacity(schedule, operand_rows, rhs_base)
    return None if layout is None else ConvolutionRhsResidency(schedule, layout, law, rhs_base)


def _prepare_convolution_rhs(rhs, builder, lower, complete):
    p = rhs.schedule.source.plan
    ci = p.window.input_shape[-1]
    kh, kw = p.window.kernel
    intervals = rhs.schedule.source.column_intervals
    chunks = (ci+p.channel_tile-1)//p.channel_tile
    columns = {column: n for n, (column, _) in enumerate(intervals)}

    def address(kernel, chunk, column):
        index = builder.op("add", builder.op("mul", kernel, chunks), chunk)
        index = builder.op("add", builder.op("mul", index, len(intervals)), columns[column])
        return builder.op("add", rhs.layout.rhs_base, builder.op("mul", index, p.channel_tile))

    def kernel_body(kernel):
        def channel_body(chunk, channel, depth):
            reduction = builder.op("add", builder.op("mul", kernel, ci), channel)
            for column, width in intervals:
                lower("load_weight", dict(reduction_row=reduction, output_column=column,
                    local_row=address(kernel, chunk, column), rows=depth, columns=width))
        whole, tail = divmod(ci, p.channel_tile)
        if whole:
            builder.loop(0, whole, 1,
                lambda chunk: channel_body(chunk, builder.op("mul", chunk, p.channel_tile), p.channel_tile))
        if tail:
            channel_body(whole, whole*p.channel_tile, tail)

    builder.loop(0, kh*kw, 1, kernel_body)
    complete("weight_preparation", dict(panels=rhs.layout.k_tiles*rhs.layout.n_tiles,
                                       local_rows=rhs.layout.rhs_rows))
    pending = None

    def consume(kind, fields):
        nonlocal pending
        if kind == "load_weight":
            if pending is not None:
                raise ValueError("resident convolution RHS lost original load/compute pairing")
            pending = fields
        elif kind == "compute":
            if pending is None or pending["output_column"] != fields["output_column"]:
                raise ValueError("resident convolution RHS lacks original consumed weight coordinates")
            reduction = pending["reduction_row"]
            local = address(builder.op("udiv", reduction, ci),
                builder.op("udiv", builder.op("urem", reduction, ci), p.channel_tile),
                fields["output_column"])
            lower(kind, dict(fields, rhs_row=local))
            pending = None
        else:
            if pending is not None:
                raise ValueError("resident convolution RHS has intervening original effects")
            lower(kind, fields)

    def check():
        if pending is not None:
            raise ValueError("resident convolution RHS leaves an unmatched original compute")

    return consume, check


def emit_convolution_schedule(schedule, builder, lower, *, complete=None, synchronous_completion=False, rhs=None):
    """Emit the complete counted original invocation, consuming actual lifetimes.

    Scalar callbacks own actual selected instruction encodings and readouts.
    Synchronous mode retains the original completion assertion. Issued mode needs
    a real complete(kind, fields) callback. The default drains after every
    compute/store; the stronger opt-in epoch law permits runtime-bounded compute
    groups and retains full completion after every original store. No host payload
    packing, alias permission or implicit event is added.
    """
    if type(schedule) is not ConvolutionSchedule:
        raise ValueError("convolution emission requires its original checked schedule")
    schedule.verify_current()
    if (not callable(lower) or any(not callable(getattr(builder, n, None))
        for n in ("op", "cmp", "loop", "branch"))
        or (schedule.choice.load_issue and not callable(complete))
        or (not schedule.choice.load_issue and synchronous_completion is not True)):
        raise ValueError("convolution emission lacks original target callbacks or real completion")
    before = schedule.identity
    rhs_identity = None
    def rhs_check():
        return None
    if rhs is not None:
        if type(rhs) is not ConvolutionRhsResidency or rhs.schedule.identity != before or not callable(complete):
            raise ValueError("convolution RHS realization needs its original schedule and real preparation completion")
        rhs.verify_current()
        rhs_identity = rhs.identity
        lower, rhs_check = _prepare_convolution_rhs(rhs, builder, lower, complete)
    p = schedule.source.plan
    batch, _, _, _ = p.window.input_shape
    oh, ow = p.window.output_spatial

    def image_body(image):
        def y_body(y):
            def x_body(x, rows):
                row = builder.op("add", builder.op("mul", builder.op("add",
                    builder.op("mul", image, oh), y), ow), x)
                for block in schedule.column_blocks:
                    args = dict(output_row=row, rows=rows)
                    if schedule.choice.compute_pairs > 1:
                        emit_convolution_compute_epoch_column_block(p, builder, lower, complete, **args,
                            output_tiles=block, law=schedule.retention_law, issue_law=schedule.issue_law,
                            epoch_law=schedule.epoch_law, compute_pairs=schedule.choice.compute_pairs)
                    elif len(block) == 1:
                        t = block[0]
                        args.update(output_column=t.output_column, columns=t.columns,
                                    accumulator_row=t.accumulator_row)
                        if schedule.choice.load_issue:
                            emit_convolution_load_issue_tile(p, builder, lower, complete,
                                **args, law=schedule.issue_law)
                        else:
                            emit_convolution_tile(p, builder, lower, **args, synchronous_completion=True)
                    elif schedule.choice.load_issue:
                        emit_convolution_load_issue_column_block(p, builder, lower, complete, **args,
                            output_tiles=block, law=schedule.retention_law, issue_law=schedule.issue_law)
                    else:
                        emit_convolution_column_block(p, builder, lower, **args, output_tiles=block,
                            law=schedule.retention_law, synchronous_completion=True)
            whole, tail = divmod(ow, p.row_tile)
            builder.loop(0, whole, 1, lambda x: x_body(builder.op("mul", x, p.row_tile), p.row_tile))
            if tail:
                x_body(whole*p.row_tile, tail)
        builder.loop(0, oh, 1, y_body)
    builder.loop(0, batch, 1, image_body)
    rhs_check()
    if schedule.identity != before:
        raise ValueError("original convolution schedule changed during target realization")
    if rhs is not None:
        if rhs.identity != rhs_identity:
            raise ValueError("original convolution RHS placement changed during realization")
        return dict(schedule.summary(), rhs_residency=rhs.summary(), realization_sha256=rhs_identity)
    return schedule.summary()


def convolution_pass_space(source, *, verify_source: Callable, realize: Callable,
                           retention_law=None, issue_law=None, epoch_law=None,
                           live_tile_limits=(1, None), issue_modes=(False, True),
                           compute_pair_limits=(1,), rhs_modes=(False,), rhs_law=None,
                           private_operand_rows=None, rhs_base=None, max_candidates=8):
    """Register independent source/lifetime/issue/realization actions.

    The returned ordinary ExplorationSpace exposes its registry and bindings for
    extension/reordering under existing prerequisites. ``realize(schedule)`` must
    return actual EmittedProgram bytes bound to schedule.identity, typically by
    calling emit_convolution_schedule in the target's normal module builder.
    ``verify_source(source)`` must reopen original source/effects/resources and
    return None; naming identities or registry capabilities cannot replace it.
    Opting into RHS choices additionally calls ``realize(schedule, rhs=plan)``;
    the callback must consume that checked preparation in actual emission and
    bind EmittedProgram to plan.identity, or schedule.identity for a fallback.
    """
    if (type(source) is not ConvolutionExplorationSource or not callable(verify_source)
        or not callable(realize) or type(live_tile_limits) is not tuple or not live_tile_limits
        or len(live_tile_limits) > 8 or len(set(live_tile_limits)) != len(live_tile_limits)
        or type(issue_modes) is not tuple or not issue_modes
        or len(set(issue_modes)) != len(issue_modes) or any(type(v) is not bool for v in issue_modes)
        or type(compute_pair_limits) is not tuple or not compute_pair_limits or len(compute_pair_limits) > 8
        or len(set(compute_pair_limits)) != len(compute_pair_limits)
        or any(type(v) is not int or not 0 < v < 2**31 for v in compute_pair_limits)
        or type(rhs_modes) is not tuple or not rhs_modes or len(set(rhs_modes)) != len(rhs_modes)
        or any(type(v) is not bool for v in rhs_modes)):
        raise ValueError("convolution registry requires bounded choices and original target callbacks")
    for limit in live_tile_limits:
        ConvolutionScheduleChoice(limit)
    for law, cls in ((retention_law, ConvolutionInputRetentionLaw), (issue_law, ConvolutionLoadIssueLaw),
                     (epoch_law, ConvolutionComputeEpochLaw), (rhs_law, ConvolutionRhsResidencyLaw)):
        if law is not None:
            if type(law) is not cls:
                raise ValueError("convolution registry requires typed original effect laws")
            law.__post_init__()
    def law_ids():
        return (_law_digest(retention_law), _law_digest(issue_law), _law_digest(epoch_law), _law_digest(rhs_law))

    law_identities = law_ids()
    source_identity = source.identity
    level = "counted_convolution_source"

    def reopen(current):
        if (type(current) is not ConvolutionExplorationSource or current.identity != source_identity
            or law_identities != law_ids()):
            raise ValueError("convolution registry source or target laws changed")
        if (verify_source(current) is not None or current.identity != source_identity
            or law_identities != law_ids()):
            raise ValueError("original convolution source callback did not validate unchanged input")

    def entry(name, stage, requires, produces, action):
        return RegisteredPass(PassSpec(name, stage, requires=requires, produces=produces,
                                      mutates_ir=False, ir_level=level), action)

    def original(_):
        def action(session):
            reopen(session.module)
            return {"convolution_source": session.module}
        return entry("original_window_partitions", 0, (), ("convolution_source",), action)

    def columns(options):
        def action(session):
            current = session.analysis("convolution_source")
            reopen(current)
            limit = None if options["live_tiles"] == "capacity" else options["live_tiles"]
            value = plan_convolution_schedule(current, ConvolutionScheduleChoice(limit),
                                             retention_law=retention_law)
            return {"convolution_columns": value}
        return entry("allocate_column_lifetimes", 1, ("convolution_source",), ("convolution_columns",), action)

    def issue(options):
        def action(session):
            previous = session.analysis("convolution_columns")
            previous.verify_current()
            value = plan_convolution_schedule(previous.source,
                ConvolutionScheduleChoice(previous.choice.live_tiles, options["enabled"]),
                retention_law=retention_law, issue_law=issue_law)
            return {"convolution_schedule": value}
        return entry("select_bounded_load_epochs", 2, ("convolution_columns",), ("convolution_schedule",), action)

    def epochs(options):
        def action(session):
            previous = session.analysis("convolution_schedule")
            previous.verify_current()
            value = plan_convolution_schedule(previous.source,
                ConvolutionScheduleChoice(previous.choice.live_tiles, previous.choice.load_issue, options["pairs"]),
                retention_law=retention_law, issue_law=issue_law, epoch_law=epoch_law)
            return {"convolution_epoch_schedule": value}
        return entry("group_nested_compute_epochs", 3, ("convolution_schedule",),
                     ("convolution_epoch_schedule",), action)

    extended = compute_pair_limits != (1,)
    schedule_analysis = "convolution_epoch_schedule" if extended else "convolution_schedule"
    prepared = rhs_modes != (False,)

    def residency(options):
        def action(session):
            schedule = session.analysis(schedule_analysis)
            schedule.verify_current()
            rhs = plan_convolution_rhs_residency(schedule, operand_rows=private_operand_rows,
                law=rhs_law, rhs_base=rhs_base) if options["enabled"] else None
            return {"convolution_preparation": (schedule, rhs)}
        return entry("prepare_invocation_rhs", 4 if extended else 3, (schedule_analysis,),
                     ("convolution_preparation",), action)

    def emit(_):
        def action(session):
            if prepared:
                schedule, rhs = session.analysis("convolution_preparation")
            else:
                schedule, rhs = session.analysis(schedule_analysis), None
            reopen(session.module)
            before = rhs.identity if rhs is not None else schedule.identity
            result = realize(schedule, rhs=rhs) if prepared else realize(schedule)
            reopen(session.module)
            if (type(result) is not EmittedProgram or result.schedule_sha256 != before
                or (rhs.identity if rhs is not None else schedule.identity) != before):
                raise ValueError("target program is not bound to the original realized convolution schedule")
            return {"convolution_program": result}
        return entry("realize_original_convolution", 3+int(extended)+int(prepared),
                     ("convolution_preparation" if prepared else schedule_analysis,),
                     ("convolution_program",), action)

    families = (PassFamily("source", original), PassFamily("columns", columns,
        knobs=(("live_tiles", tuple("capacity" if v is None else v for v in live_tile_limits)),)),
        PassFamily("issue", issue, knobs=(("enabled", issue_modes),)))
    if extended:
        families += (PassFamily("compute_epochs", epochs, knobs=(("pairs", compute_pair_limits),)),)
    if prepared:
        families += (PassFamily("rhs_preparation", residency, knobs=(("enabled", rhs_modes),)),)
    return generate_candidates((*families, PassFamily("emit", emit)),
                               output="convolution_program", max_candidates=max_candidates)


def explore_convolution_schedules(source, space, *, context, feedback=None):
    """Execute registered choices and deduplicate actual emitted bytes.

    Feedback is optional and identity-bound by the existing ordinary explorer.
    It supplies no default score, simulator, ranking or performance promotion.
    Targets may instead consume ``space.registry`` with the ordinary manager.
    """
    def verify(current):
        if type(current) is not ConvolutionExplorationSource:
            raise ValueError("convolution exploration source type changed")
        current.__post_init__()

    def facts(session):
        p = session.module.plan
        return PolicyFacts(session.stamp(), shapes=(p.window.input_shape,
            (*p.window.kernel, p.window.input_shape[-1], p.output_channels)),
            operations=frozenset({"convolution"}),
            resources=(("operand_rows", p.operand_rows), ("accumulator_rows", p.accumulator_rows)))

    return explore_candidates(source, space, facts, output="convolution_program",
        emission=lambda result: result.artifact, feedback=feedback, context=context,
        ir_level="counted_convolution_source", clone=copy.deepcopy,
        fingerprint=lambda current: current.identity, verify_ir=verify)
