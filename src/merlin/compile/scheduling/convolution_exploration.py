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

from .affine_gather_runs import ConvolutionGatherRuns
from .dense_rhs_residency import DenseRhsResidency, plan_dense_rhs_residency
from .window_transfer import (
    ConvolutionComputeEpochLaw,
    ConvolutionInputRetentionLaw,
    ConvolutionLoadIssueLaw,
    ConvolutionOutputTile,
    ConvolutionReadoutEpochLaw,
    ConvolutionReadoutTransferLaw,
    ConvolutionResidentInputRows,
    CountedConvolutionPlan,
    emit_convolution_column_block,
    emit_convolution_compute_epoch_column_block,
    emit_convolution_load_issue_column_block,
    emit_convolution_load_issue_tile,
    emit_convolution_tile,
    plan_convolution_readout_transfers,
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
    column_intervals: tuple[tuple[int, int], ...] | None = None
    _schedule_sha256: str = field(init=False, repr=False)
    _law_sha256: str = field(init=False, repr=False)
    _layout_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        if (type(self.schedule) is not ConvolutionSchedule or type(self.layout) is not DenseRhsResidency
            or type(self.law) is not ConvolutionRhsResidencyLaw):
            raise ValueError("convolution RHS residency requires original checked schedule and capacity plan")
        self.schedule.verify_current()
        self.law.require()
        if self.layout != _rhs_capacity(self.schedule, self.layout.operand_rows,
            self.assigned_rhs_base, self.column_intervals):
            raise ValueError("convolution RHS placement differs from original private capacity/source geometry")
        for law in (self.schedule.retention_law, self.schedule.issue_law, self.schedule.epoch_law):
            if law is not None and (law.hardware_identity != self.law.hardware_identity
                or law.numerical_policy_identity != self.law.numerical_policy_identity):
                raise ValueError("convolution RHS and execution laws select different contracts")
        object.__setattr__(self, "_schedule_sha256", self.schedule.identity)
        object.__setattr__(self, "_law_sha256", _law_digest(self.law))
        object.__setattr__(self, "_layout_sha256", self._layout_identity())

    def verify_current(self):
        self.schedule.verify_current()
        self.law.require()
        if (self.schedule.identity != self._schedule_sha256 or _law_digest(self.law) != self._law_sha256
            or self._layout_identity() != self._layout_sha256
            or self.layout != _rhs_capacity(self.schedule, self.layout.operand_rows,
                self.assigned_rhs_base, self.column_intervals)):
            raise ValueError("original convolution RHS source/law/private placement changed after planning")

    def _layout_identity(self):
        original = (self.assigned_rhs_base, asdict(self.layout))
        return _digest(original if self.column_intervals is None else (*original, self.column_intervals))

    def summary(self):
        self.verify_current()
        result = dict(schema="merlin.convolution_rhs_residency.v1", schedule_sha256=self._schedule_sha256,
            law_sha256=self._law_sha256, physical_panel_layout=asdict(self.layout),
            assigned_rhs_base=self.assigned_rhs_base,
            lifetime="one invocation; preparation fully completes before first consumer",
            reduction="original per-kernel channel chunks, including each independent tail")
        if self.column_intervals is not None:
            result["column_intervals"] = self.column_intervals
        return result

    @property
    def identity(self):
        return _digest(self.summary())


def _rhs_intervals(schedule, intervals):
    original = schedule.source.column_intervals
    if intervals is None:
        return original
    if (type(intervals) is not tuple or not intervals or any(type(i) is not tuple or len(i) != 2
        or any(type(v) is not int for v in i) for i in intervals)):
        raise ValueError("RHS phase needs exact original column intervals")
    try:
        first = original.index(intervals[0])
    except ValueError as exc:
        raise ValueError("RHS phase changes original column boundaries") from exc
    if original[first:first+len(intervals)] != intervals:
        raise ValueError("RHS phase must be one ordered original column slice")
    return intervals


def _rhs_capacity(schedule, operand_rows, rhs_base=None, column_intervals=None):
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
        len(_rhs_intervals(schedule, column_intervals))*p.channel_tile,
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


@dataclass(frozen=True)
class ConvolutionRhsPhaseLaw:
    """Separate source/effect permission to reorder independent output columns.

    All original scalar readout boundaries survive. The caller must establish
    that input buffers stay immutable, original output columns are disjoint and
    no observer can see intermediate store order. Actual full completion before
    replacement is emitted by the phase owner, never inferred from capacity.
    """
    source_identity: str
    qualification_identity: str
    independent_output_columns: bool = False
    immutable_inputs: bool = False
    no_intermediate_observers: bool = False
    full_completion_before_reprepare: bool = False

    def require(self, source):
        if (type(source) is not ConvolutionExplorationSource or self.source_identity != source.identity
            or type(self.qualification_identity) is not str or not self.qualification_identity.strip()
            or any(type(v) is not bool or not v for v in (self.independent_output_columns,
                self.immutable_inputs, self.no_intermediate_observers, self.full_completion_before_reprepare))):
            raise ValueError("RHS phases lack original independent-output/observer/completion premises")


def _phase_input(schedule, input_rows, operand_rows, rhs_base):
    if input_rows is not None:
        if type(input_rows) is not ConvolutionResidentInputRows:
            raise ValueError("RHS phases require the exact original resident input footprint")
        input_rows.revalidate()
        if (input_rows.source_identity != schedule.source.identity
            or input_rows.window != schedule.source.plan.window
            or input_rows.channel_tile != schedule.source.plan.channel_tile or input_rows.end > operand_rows):
            raise ValueError("RHS phases require the exact original resident input footprint")
        input_rows.revalidate()
        if rhs_base < input_rows.end:
            raise ValueError("RHS phases overlap the explicit resident input footprint")


@dataclass(frozen=True)
class ConvolutionRhsPhases:
    """Complete original N partition with bounded full-K private RHS phases."""
    schedule: ConvolutionSchedule
    phases: tuple[ConvolutionRhsResidency, ...]
    law: ConvolutionRhsPhaseLaw
    input_rows: ConvolutionResidentInputRows | None = None
    max_column_blocks: int | None = None
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_identity", self._current_identity())

    def _validate(self):
        if (type(self.schedule) is not ConvolutionSchedule or type(self.law) is not ConvolutionRhsPhaseLaw
            or type(self.phases) is not tuple or len(self.phases) < 2
            or any(type(p) is not ConvolutionRhsResidency or p.column_intervals is None for p in self.phases)):
            raise ValueError("RHS phases require complete typed original placements")
        if self.max_column_blocks is not None and (
            type(self.max_column_blocks) is not int or not 0 < self.max_column_blocks < 1 << 63
        ):
            raise ValueError("RHS phase column bound must be an explicit positive integer")
        self.schedule.verify_current()
        self.law.require(self.schedule.source)
        first = self.phases[0]
        for phase in self.phases:
            phase.verify_current()
            if (phase.schedule.identity != self.schedule.identity or phase.law != first.law
                or phase.layout.operand_rows != first.layout.operand_rows
                or phase.assigned_rhs_base != first.assigned_rhs_base):
                raise ValueError("RHS phases change original source, law or assigned capacity")
        _phase_input(self.schedule, self.input_rows, first.layout.operand_rows, first.layout.rhs_base)
        original = self.schedule.source.column_intervals
        capacity = (first.layout.operand_rows-first.layout.rhs_base)//(first.layout.k_tiles*first.layout.block)
        if self.max_column_blocks is not None:
            capacity = min(capacity, self.max_column_blocks)
        expected = tuple(original[i:i+capacity] for i in range(0, len(original), capacity))
        if tuple(p.column_intervals for p in self.phases) != expected:
            raise ValueError("RHS phases must cover every original column once using the bounded legal capacity")

    def _current_identity(self):
        original = (self.schedule.identity, [p.summary() for p in self.phases], asdict(self.law),
            None if self.input_rows is None else self.input_rows.record())
        return _digest(original if self.max_column_blocks is None else (*original, self.max_column_blocks))

    def verify_current(self):
        self._validate()
        if self._current_identity() != self._identity:
            raise ValueError("original RHS phase source, effects or input footprint changed")

    @property
    def identity(self):
        self.verify_current()
        return self._identity

    def summary(self):
        self.verify_current()
        return dict(schema="merlin.convolution_rhs_phases.v1", schedule_sha256=self.schedule.identity,
            realization_sha256=self._identity, phases=[p.summary() for p in self.phases],
            input_footprint=None if self.input_rows is None else self.input_rows.record(),
            **({} if self.max_column_blocks is None else {"max_column_blocks": self.max_column_blocks}),
            coverage="each original output column exactly once; complete original K and scalar exits",
            completion="real full phase-last-use drain before any RHS replacement")


def plan_convolution_rhs_phases(schedule, *, operand_rows, law, phase_law=None, rhs_base=None, input_rows=None,
                                max_column_blocks=None):
    """Keep a full-fit incumbent, else derive maximal complete-K N phases.

    Only the explicitly owned row interval is used. With resident input, the
    default RHS base follows its actual footprint, aligned to a channel block.
    A caller-selected base may reserve additional rows/banks. If even one source
    interval does not fit, return None for the unchanged ordinary weight path.
    Capacity never grants independent-output or observer/effect permission.
    An optional positive column-block bound retains legal narrower phases,
    allowing more original spatial accumulators to remain live. It does not
    reduce the declared hardware capacity or select a performance winner.
    """
    if type(schedule) is not ConvolutionSchedule or type(law) is not ConvolutionRhsResidencyLaw:
        raise ValueError("RHS phases need original schedule and RHS preservation law")
    schedule.verify_current()
    law.require()
    if max_column_blocks is not None and (
        type(max_column_blocks) is not int or not 0 < max_column_blocks < 1 << 63
    ):
        raise ValueError("RHS phase column bound must be an explicit positive integer")
    p = schedule.source.plan
    if rhs_base is None and input_rows is not None:
        if type(input_rows) is not ConvolutionResidentInputRows:
            raise ValueError("RHS phases require the exact original resident input footprint")
        input_rows.revalidate()
        rhs_base = (input_rows.end+p.channel_tile-1)//p.channel_tile*p.channel_tile
    full = plan_convolution_rhs_residency(schedule, operand_rows=operand_rows, law=law, rhs_base=rhs_base)
    base = (max(p.rhs_row, p.lhs_row+p.row_tile)+p.channel_tile-1)//p.channel_tile*p.channel_tile
    base = base if rhs_base is None else rhs_base
    _phase_input(schedule, input_rows, operand_rows, base)
    if full is not None and (max_column_blocks is None or len(schedule.source.column_intervals) <= max_column_blocks):
        return full
    original = schedule.source.column_intervals
    single = _rhs_capacity(schedule, operand_rows, base, original[:1])
    if single is None:
        return None
    if type(phase_law) is not ConvolutionRhsPhaseLaw:
        raise ValueError("RHS phases need separate original independent-output premises")
    phase_law.require(schedule.source)
    count = (operand_rows-base)//single.rhs_rows
    if max_column_blocks is not None:
        count = min(count, max_column_blocks)
    phases = tuple(ConvolutionRhsResidency(schedule, _rhs_capacity(schedule, operand_rows, base, subset),
        law, base, subset) for subset in (original[i:i+count] for i in range(0, len(original), count)))
    return ConvolutionRhsPhases(schedule, phases, phase_law, input_rows, max_column_blocks)


def _prepare_convolution_rhs(rhs, builder, lower, complete, *, already_prepared=False):
    p = rhs.schedule.source.plan
    ci = p.window.input_shape[-1]
    kh, kw = p.window.kernel
    intervals = _rhs_intervals(rhs.schedule, rhs.column_intervals)
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

    if not already_prepared:
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


def emit_convolution_schedule(schedule, builder, lower, *, complete=None, synchronous_completion=False, rhs=None,
                              spatial_iteration="rows", input_rows=None, readout_law=None, readout_transfer_law=None,
                              output_group=None, array_law=None, ready_readout=False, grouped_transport=False,
                              rhs_phase_prefetch_law=None):
    """Emit the checked complete invocation, optionally in original N phases.

    Full-fit/default schedules retain the same emitter. A phased owner must
    cover all original columns, preserve scalar exits, bind the same resident
    input and establish independent-output/observer premises. A real full drain
    ends every phase before its rows may be consumed by the next phase.

    For phased RHS, ``output_group`` is the exact phase-aligned tuple returned
    by plan_rolling_convolution_output_group, with None retaining an ordinary
    phase. The same original array/readout admissions apply independently to
    every selected group. All groups are checked before any preparation.

    ``rhs_phase_prefetch_law`` separately admits exact scratchpad WAR/RAW
    dependencies through real completion. Only the final source batch/spatial
    group replaces dead K prefixes with the next original RHS phase. Its N
    footprint may not grow. Input transport is serialized, all original full
    drains remain, and no second accumulator or RHS allocation is introduced.
    """
    if type(ready_readout) is not bool or ready_readout and output_group is None:
        raise ValueError("ready readout requires an explicit Boolean and checked spatial group")
    if type(grouped_transport) is not bool:
        raise ValueError("grouped transport requires an explicit Boolean")
    if rhs_phase_prefetch_law is not None:
        from .rolling_lookahead import RollingRhsFrontierAdmission
        if (type(rhs_phase_prefetch_law) is not RollingRhsFrontierAdmission or type(rhs) is not ConvolutionRhsPhases
                or not grouped_transport or output_group is None or schedule.epoch_law is None):
            raise ValueError("RHS frontier needs its separate target law, complete phase owner and grouped consumer")
        rhs_phase_prefetch_law.require()
        if (rhs_phase_prefetch_law.hardware_identity != schedule.epoch_law.hardware_identity
                or rhs_phase_prefetch_law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity):
            raise ValueError("RHS frontier differs from the selected original target/numerical scope")
    if type(rhs) is ConvolutionRhsPhases:
        rhs.verify_current()
        if (type(schedule) is not ConvolutionSchedule or rhs.schedule.identity != schedule.identity
            or not callable(complete) or rhs.input_rows != input_rows):
            raise ValueError("RHS phase emission needs original input, source and real completion")
        identity = rhs.identity
        readout_identity = _law_digest(readout_law)
        transfer_identity = _law_digest(readout_transfer_law)
        if output_group is None:
            if array_law is not None:
                raise ValueError("phased array retention needs its complete original group tuple")
            groups = (None,) * len(rhs.phases)
        elif (type(output_group) is not tuple or len(output_group) != len(rhs.phases)
              or not any(group is not None for group in output_group)):
            raise ValueError("phased spatial groups must align with every original RHS phase")
        else:
            groups = output_group
        for phase, group in zip(rhs.phases, groups, strict=True):
            _verify_readout_law(schedule, readout_law, _phase_blocks(schedule, phase.column_intervals))
            _verify_readout_transfers(schedule, readout_transfer_law, readout_law,
                                      _phase_blocks(schedule, phase.column_intervals))
            _verify_spatial_output_group(schedule, builder, lower, complete, rhs=phase, input_rows=input_rows,
                spatial_iteration=spatial_iteration, readout_law=readout_law,
                readout_transfer_law=readout_transfer_law, output_group=group,
                array_law=array_law if group is not None else None,
                blocks=_phase_blocks(schedule, phase.column_intervals))
            if grouped_transport and group is not None:
                _verify_grouped_transport(schedule, input_rows, group)
        group_identity = tuple(_law_digest(group) for group in groups)
        array_identity = _law_digest(array_law)
        frontier_identity=_law_digest(rhs_phase_prefetch_law)
        frontiers=[]
        for index, (phase, group) in enumerate(zip(rhs.phases, groups, strict=True)):
            frontier=None
            if rhs_phase_prefetch_law is not None:
                from .rolling_lookahead import RollingRhsFrontier
                if group is None:
                    raise ValueError("RHS frontier requires every phase's checked original consumer")
                if index+1<len(rhs.phases):
                    next_phase=rhs.phases[index+1]
                    frontier=RollingRhsFrontier(schedule.source.identity,phase.layout.rhs_base,
                        schedule.source.plan.channel_tile,schedule.source.plan.output_channels,
                        len(phase.column_intervals),next_phase.column_intervals)
                    frontier.require(input_rows.rolling_plan(),phase.layout.k_tiles)
            frontiers.append(frontier)
        results = []
        for index, (phase, group,frontier) in enumerate(zip(rhs.phases,groups,frontiers,strict=True)):
            results.append(_emit_convolution_schedule(schedule, builder, lower, complete=complete,
                synchronous_completion=synchronous_completion, rhs=phase, spatial_iteration=spatial_iteration,
                input_rows=input_rows, readout_law=readout_law, readout_transfer_law=readout_transfer_law,
                output_group=group, array_law=array_law if group is not None else None,
                ready_readout=ready_readout and group is not None,
                grouped_transport=grouped_transport and group is not None,
                rhs_frontier=frontier,rhs_frontier_law=rhs_phase_prefetch_law,
                rhs_prefetched=rhs_phase_prefetch_law is not None and index>0))
            complete("rhs_phase_last_use", dict(phase=index, local_row=phase.layout.rhs_base,
                local_rows=phase.layout.rhs_rows))
        rhs.verify_current()
        if (rhs.identity != identity or _law_digest(readout_law) != readout_identity
            or _law_digest(readout_transfer_law) != transfer_identity
            or tuple(_law_digest(group) for group in groups) != group_identity
            or _law_digest(array_law) != array_identity
            or _law_digest(rhs_phase_prefetch_law) != frontier_identity):
            raise ValueError("original RHS phases changed during realization")
        phase_report=rhs.summary()
        if rhs_phase_prefetch_law is not None:
            rhs_phase_prefetch_law.require()
            phase_report=dict(phase_report,frontier_admission=asdict(rhs_phase_prefetch_law),
                completion="dead K spans replaced after selected WAR dependencies; all original group, preparation and phase drains retained")
        return dict(schedule.summary(), rhs_phases=phase_report, phase_emissions=results,
            realization_sha256=identity)
    if rhs is not None:
        if type(rhs) is not ConvolutionRhsResidency:
            raise ValueError("convolution RHS needs its original checked placement")
        if rhs.column_intervals is not None:
            raise ValueError("partial RHS placement needs its complete original phase owner")
    return _emit_convolution_schedule(schedule, builder, lower, complete=complete,
        synchronous_completion=synchronous_completion, rhs=rhs, spatial_iteration=spatial_iteration,
        input_rows=input_rows, readout_law=readout_law, readout_transfer_law=readout_transfer_law,
        output_group=output_group, array_law=array_law, ready_readout=ready_readout,
        grouped_transport=grouped_transport)


def _phase_blocks(schedule, intervals):
    if intervals is None:
        return schedule.column_blocks
    columns = {column for column, _ in _rhs_intervals(schedule, intervals)}
    return tuple(part for block in schedule.column_blocks
        if (part := tuple(tile for tile in block if tile.output_column in columns)))


def _verify_readout_law(schedule, law, blocks):
    if law is None:
        return
    if (type(law) is not ConvolutionReadoutEpochLaw
        or not schedule.choice.load_issue or schedule.choice.compute_pairs <= 1):
        raise ValueError("convolution readout epochs require a typed law and issued compute epochs")
    for block in blocks:
        law.require(schedule.source.plan, schedule.source.plan.row_tile, block)
    if (law.hardware_identity != schedule.epoch_law.hardware_identity
        or law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity):
        raise ValueError("convolution readout and compute epochs select different contracts")


def _verify_readout_transfers(schedule, law, readout_law, blocks):
    if law is None:
        return
    if (type(law) is not ConvolutionReadoutTransferLaw or readout_law is None
        or law.hardware_identity != readout_law.hardware_identity
        or law.numerical_policy_identity != readout_law.numerical_policy_identity):
        raise ValueError("wide readout needs its separate map and original producer/readout epoch")
    for block in blocks:
        plan_convolution_readout_transfers(schedule.source.plan, block, law, rows=schedule.source.plan.row_tile)


def _verify_spatial_output_group(schedule, builder, lower, complete, *, rhs, input_rows,
                                 spatial_iteration, readout_law, readout_transfer_law,
                                 output_group, array_law, blocks):
    """Check active original column ownership and budgets before target effects."""
    p = schedule.source.plan
    group_identity = None
    array_identity = None
    if output_group is not None or array_law is not None:
        from .convolution_output_groups import ConvolutionOutputGroup, plan_rolling_convolution_output_group
        from .dense_array_retention import DenseArrayRetentionAdmission
        from .window_transfer import _validate_convolution_load_issue
        if (type(array_law) is not DenseArrayRetentionAdmission or type(output_group) is not ConvolutionOutputGroup
            or spatial_iteration != "rows" or readout_law is None):
            raise ValueError("spatial grouping needs separate array/readout premises")
        array_law.__post_init__()
        if (schedule.retention_law is None or not schedule.retention_law.independent_output_rows
            or not schedule.retention_law.no_intermediate_observers):
            raise ValueError("spatial grouping needs original independent-row and observer permissions")
        expected = plan_rolling_convolution_output_group(schedule, rhs, input_rows,
            max_row_blocks=output_group.row_blocks, max_output_rows=output_group.output_rows)
        if expected is None or output_group != expected:
            raise ValueError("spatial output group differs from original rolling/RHS ownership")
        if (array_law.hardware_identity != schedule.epoch_law.hardware_identity
            or array_law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity):
            raise ValueError("spatial array retention selects a different original contract")
        for block in blocks:
            _validate_convolution_load_issue(p, builder, lower, complete, p.row_tile,
                schedule.issue_law, output_group.row_blocks*len(block)*int(p.bias))
            if output_group.row_blocks*len(block) > readout_law.max_issued_readouts:
                raise ValueError("spatial group exceeds selected original readout work budget")
            for m in range(output_group.row_blocks):
                tiles = tuple(ConvolutionOutputTile(t.output_column,t.columns,
                    p.accumulator_row+output_group.accumulator_row(m,n,active_column_blocks=len(block)))
                    for n,t in enumerate(block))
                readout_law.require(p, p.row_tile, tiles)
                if readout_transfer_law is not None:
                    plan_convolution_readout_transfers(p, tiles, readout_transfer_law, rows=p.row_tile)
        group_identity, array_identity = _law_digest(output_group), _law_digest(array_law)
    return group_identity, array_identity


def _verify_grouped_transport(schedule, input_rows, output_group):
    from .rolling_window import materialize_grouped_rolling_window, verify_grouped_rolling_window
    if (output_group is None or input_rows is None or schedule.issue_law is None
        or not schedule.choice.load_issue):
        raise ValueError("grouped transport needs checked spatial ownership and bounded target issue")
    ring = input_rows.rolling_plan()
    module = materialize_grouped_rolling_window(ring, output_rows=output_group.output_rows)
    report = verify_grouped_rolling_window(module, ring, output_rows=output_group.output_rows)
    commands = input_rows.stage_command_bound(report["max_staged_rows_per_epoch"])
    if commands > schedule.issue_law.load_issue_budget:
        raise ValueError("grouped transport exceeds selected actual transfer issue bound")
    return commands


def _emit_convolution_schedule(schedule, builder, lower, *, complete=None, synchronous_completion=False, rhs=None,
                               spatial_iteration="rows", input_rows=None, readout_law=None, readout_transfer_law=None,
                               output_group=None, array_law=None, ready_readout=False, grouped_transport=False,
                               rhs_frontier=None,rhs_frontier_law=None,rhs_prefetched=False):
    """Emit the complete counted original invocation, consuming actual lifetimes.

    Scalar callbacks own actual selected instruction encodings and readouts.
    Synchronous mode retains the original completion assertion. Issued mode needs
    a real complete(kind, fields) callback. The default drains after every
    compute/store; the stronger opt-in epoch law permits runtime-bounded compute
    groups and retains full completion after every original store. An optional
    separate ``readout_law`` orders the final producer and original stores by
    selected dependencies, with a real full drain at each column block's end.
    No host payload packing, alias permission or implicit event is added. Explicit ``linear``
    spatial iteration coalesces valid unpadded row/image tails through exact
    affine gather fragments; default ``rows`` preserves original realization.
    """
    if type(schedule) is not ConvolutionSchedule:
        raise ValueError("convolution emission requires its original checked schedule")
    schedule.verify_current()
    if (not callable(lower) or any(not callable(getattr(builder, n, None))
        for n in ("op", "cmp", "loop", "branch"))
        or (schedule.choice.load_issue and not callable(complete))
        or (not schedule.choice.load_issue and synchronous_completion is not True)):
        raise ValueError("convolution emission lacks original target callbacks or real completion")
    if spatial_iteration not in ("rows", "linear"):
        raise ValueError("convolution spatial iteration must be rows or linear")
    if type(ready_readout) is not bool or ready_readout and output_group is None:
        raise ValueError("ready readout requires an explicit Boolean and checked spatial group")
    p = schedule.source.plan
    blocks = _phase_blocks(schedule, None if rhs is None else rhs.column_intervals)
    _verify_readout_law(schedule, readout_law, blocks)
    _verify_readout_transfers(schedule, readout_transfer_law, readout_law, blocks)
    readout_identity = _law_digest(readout_law)
    transfer_identity = _law_digest(readout_transfer_law)
    batch = p.window.input_shape[0]
    oh, ow = p.window.output_spatial
    gather_runs = None
    if spatial_iteration == "linear":
        if any(p.window.padding) or not callable(getattr(builder, "select", None)):
            raise ValueError("linear convolution iteration requires a valid unpadded window and select builder")
        gather_runs = ConvolutionGatherRuns(p.window.input_shape, (batch, oh, ow),
            p.window.stride, p.window.dilation, p.window.kernel, p.row_tile)
    if input_rows is not None:
        if (type(input_rows) is not ConvolutionResidentInputRows or input_rows.window != p.window
            or input_rows.source_identity != schedule.source.identity or spatial_iteration != "rows"
            or input_rows.end > p.operand_rows or not callable(complete)):
            raise ValueError("resident input requires exact source, original row domains and real completion")
        input_rows.revalidate()
        if (schedule.choice.load_issue and input_rows.initial_load_commands > schedule.issue_law.load_issue_budget):
            raise ValueError("resident input transport exceeds the selected issued load budget")
        if rhs is not None and not (input_rows.end <= rhs.layout.rhs_base
                or input_rows.base >= rhs.layout.rhs_base+rhs.layout.rhs_rows):
            raise ValueError("resident input rows overlap the assigned complete RHS")
    group_identity, array_identity = _verify_spatial_output_group(schedule, builder, lower, complete,
        rhs=rhs, input_rows=input_rows, spatial_iteration=spatial_iteration, readout_law=readout_law,
        readout_transfer_law=readout_transfer_law, output_group=output_group, array_law=array_law, blocks=blocks)
    if grouped_transport:
        _verify_grouped_transport(schedule, input_rows, output_group)
    before = schedule.identity
    rhs_identity = None
    rhs_issue=lower
    def rhs_check():
        return None
    if rhs is not None:
        if type(rhs) is not ConvolutionRhsResidency or rhs.schedule.identity != before or not callable(complete):
            raise ValueError("convolution RHS realization needs its original schedule and real preparation completion")
        rhs.verify_current()
        rhs_identity = rhs.identity
        lower, rhs_check = _prepare_convolution_rhs(rhs, builder, lower, complete,already_prepared=rhs_prefetched)
    def tile_body(row, rows):
        for block in blocks:
            args = dict(output_row=row, rows=rows)
            if input_rows is not None:
                args["input_rows"] = input_rows
            if gather_runs is not None:
                args["gather_runs"] = gather_runs
            if schedule.choice.compute_pairs > 1:
                emit_convolution_compute_epoch_column_block(p, builder, lower, complete, **args,
                    output_tiles=block, law=schedule.retention_law, issue_law=schedule.issue_law,
                    epoch_law=schedule.epoch_law, compute_pairs=schedule.choice.compute_pairs,
                    readout_law=readout_law, readout_transfer_law=readout_transfer_law)
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

    rolling = None
    if input_rows is not None:
        from .rolling_window import emit_rolling_window, materialize_rolling_window
        ring = input_rows.rolling_plan()
        def transport(kind, fields):
            # The handle is the real caller-assigned private cache base, not an
            # asynchronous ticket. Every ready/last-use edge emits a full drain.
            if kind == "stage_row":
                input_rows.emit_stage(builder, fields, lower)
            elif kind == "await_transport":
                complete("input_ready", dict(local_row=input_rows.base))
            elif kind == "advance_consumer":
                return fields["consumer"].advance(fields["first_run"], fields["run_count"],
                                                   after_run=fields.get("after_run"))
            elif kind == "begin_rhs_prefetch":
                if rhs_frontier is None or fields["rhs_row"] != rhs_frontier.rhs_base:
                    raise ValueError("RHS frontier changes the checked original phase allocation")
                return rhs_frontier.rhs_base
            elif kind == "stage_rhs":
                if rhs_frontier is None or fields["transfer"] != rhs_frontier.rhs_base:
                    raise ValueError("RHS frontier requires the actual assigned next-phase rows")
                rhs_issue("load_weight",{k:fields[k]for k in
                    ("reduction_row","output_column","local_row","rows","columns")})
            elif kind == "finish_consumer":
                fields["consumer"].finish()
            elif kind in ("consume", "consume_group", "start_consumer"):
                split = kind == "start_consumer"
                if split and (output_group is None or len(blocks) != 1):
                    raise ValueError("split rolling consumer needs one checked live output-column block")
                def column(x, rows):
                    row = builder.op("add", builder.op("mul", builder.op("add",
                        builder.op("mul", fields["image"], oh), fields["output_y"]), ow), x)
                    tile_body(row, rows)
                if output_group is None:
                    whole, tail = divmod(ow, p.row_tile)
                    builder.loop(0, whole, 1, lambda x: column(builder.op("mul", x, p.row_tile), p.row_tile))
                    if tail:
                        column(whole*p.row_tile, tail)
                elif output_group.output_rows > 1 or kind in ("consume_group", "start_consumer"):
                    from .convolution_output_groups import emit_rolling_convolution_output_group
                    # Preserve original transport and every real ready/last-use
                    # drain. Delay products until this bounded group is staged;
                    # the source-row union is injective in the selected ring.
                    def rows_group(count):
                        first_y = (fields["output_y"] if kind in ("consume_group", "start_consumer") else
                                   builder.op("sub", fields["output_y"], count - 1))
                        row = builder.op("mul", builder.op("add",
                            builder.op("mul", fields["image"], oh), first_y), ow)
                        tiles = tuple((y * ow + x, min(p.row_tile, ow - x))
                                      for y in range(count) for x in range(0, ow, p.row_tile))
                        for block in blocks:
                            consumer = emit_rolling_convolution_output_group(schedule, output_group, builder, lower, complete,
                                output_row=row, row_counts=tuple(t[1] for t in tiles),
                                tile_origins=tuple(t[0] for t in tiles), output_tiles=block,
                                input_rows=input_rows, array_law=array_law, readout_law=readout_law,
                                ready_readout=ready_readout, readout_transfer_law=readout_transfer_law,
                                defer_consumer=split)
                            if split:
                                if consumer.runs != fields["consumer_runs"]:
                                    raise ValueError("split rolling consumer differs from original reduction extent")
                                return consumer
                    if kind in ("consume_group", "start_consumer"):
                        consumer = rows_group(fields["output_rows"])
                        if split:
                            return consumer
                        return input_rows.base
                    rows = output_group.output_rows
                    tail = oh % rows
                    boundary = builder.cmp("eq", builder.op("urem",
                        builder.op("add", fields["output_y"], 1), rows), 0)
                    def last_rows():
                        if tail:
                            builder.branch(builder.cmp("eq", fields["output_y"], oh - 1),
                                           lambda: rows_group(tail), lambda: None)
                    builder.branch(boundary, lambda: rows_group(rows), last_rows)
                else:
                    from .convolution_output_groups import emit_rolling_convolution_output_group
                    def grouped(x, counts):
                        row = builder.op("add", builder.op("mul", builder.op("add",
                            builder.op("mul", fields["image"], oh), fields["output_y"]), ow), x)
                        for block in blocks:
                            emit_rolling_convolution_output_group(schedule, output_group, builder, lower, complete,
                                output_row=row, row_counts=counts, output_tiles=block, input_rows=input_rows,
                                array_law=array_law, readout_law=readout_law, ready_readout=ready_readout,
                                readout_transfer_law=readout_transfer_law)
                    width = p.row_tile*output_group.row_blocks
                    whole, tail = divmod(ow, width)
                    if whole:
                        builder.loop(0,whole,1,lambda x:grouped(builder.op("mul",x,width),
                            (p.row_tile,)*output_group.row_blocks))
                    if tail:
                        full, last = divmod(tail,p.row_tile)
                        grouped(whole*width,(p.row_tile,)*full+((last,)if last else()))
            elif kind == "complete":
                complete("input_last_use", dict(local_row=input_rows.base))
            elif kind != "begin_transport":
                raise ValueError("unknown original rolling input effect")
            return input_rows.base
        if rhs_frontier_law is not None:
            from .rolling_lookahead import (RollingLookaheadAdmission,materialize_lookahead_rolling_window,
                                            emit_lookahead_rolling_window)
            p=schedule.source.plan
            runs=p.window.kernel[0]*p.window.kernel[1]*((p.window.input_shape[-1]+p.channel_tile-1)//p.channel_tile)
            admission=RollingLookaheadAdmission(rhs_frontier_law.hardware_identity,
                rhs_frontier_law.qualification_identity,True,True,True,True,True,True)
            module=materialize_lookahead_rolling_window(ring,output_rows=output_group.output_rows,
                consumer_runs=runs,rhs_frontier=rhs_frontier,future_input=False)
            rolling=emit_lookahead_rolling_window(module,ring,builder,transport,output_rows=output_group.output_rows,
                consumer_runs=runs,rhs_frontier=rhs_frontier,future_input=False,admission=admission,
                rhs_frontier_admission=rhs_frontier_law)
        elif grouped_transport:
            from .rolling_window import materialize_grouped_rolling_window, emit_grouped_rolling_window
            rolling = emit_grouped_rolling_window(materialize_grouped_rolling_window(
                ring, output_rows=output_group.output_rows), ring, builder, transport,
                output_rows=output_group.output_rows)
        else:
            rolling = emit_rolling_window(materialize_rolling_window(ring), ring, builder, transport)
        input_rows.revalidate()
    elif spatial_iteration == "linear":
        whole, tail = divmod(batch*oh*ow, p.row_tile)
        builder.loop(0, whole, 1, lambda index:
                     tile_body(builder.op("mul", index, p.row_tile), p.row_tile))
        if tail:
            tile_body(whole*p.row_tile, tail)
    else:
        def image_body(image):
            def y_body(y):
                def x_body(x, rows):
                    row = builder.op("add", builder.op("mul", builder.op("add",
                        builder.op("mul", image, oh), y), ow), x)
                    tile_body(row, rows)
                whole, tail = divmod(ow, p.row_tile)
                builder.loop(0, whole, 1, lambda x: x_body(builder.op("mul", x, p.row_tile), p.row_tile))
                if tail:
                    x_body(whole*p.row_tile, tail)
            builder.loop(0, oh, 1, y_body)
        builder.loop(0, batch, 1, image_body)
    rhs_check()
    if (schedule.identity != before or _law_digest(readout_law) != readout_identity
        or _law_digest(readout_transfer_law) != transfer_identity
        or _law_digest(output_group) != group_identity or _law_digest(array_law) != array_identity):
        raise ValueError("original convolution schedule changed during target realization")
    result = schedule.summary()
    if output_group is not None:
        result = dict(result, spatial_output_group=dict(layout=asdict(output_group),
            group_sha256=group_identity, array_law_sha256=array_identity,
            iteration=("one output row; K,N,M; original row and channel tails"
                       if output_group.output_rows == 1 else
                       "source-row groups; K,N,M; exact per-row tails and ring union"),
            completion="real final group drain and unchanged ring input-ready/last-use drains",
            selected_by="original source row extent and explicitly owned RHS/SP/ACC capacity"))
        if ready_readout:
            result["spatial_output_group"]["ready_readout"] = dict(
                applied=True, frontier="immediately after each exact last original K producer",
                ordering="selected ACC RAW/result visibility and ordered scalar configuration capture",
                storage="original disjoint ACC cells; no new window or reuse before final group drain")
        if output_group.output_rows > 1:
            result["spatial_output_group"]["ring_read_horizon"] = dict(
                output_rows=output_group.output_rows,
                source_row_span=(p.window.kernel[0]-1)*p.window.dilation[0]+1
                                +(output_group.output_rows-1)*p.window.stride[0],
                slots=input_rows.slots,
                completion=("group transports complete before products; group drain before the next source-row stage"
                            if grouped_transport else
                            "all original transport drains; group drain before the next source-row stage"))
    if readout_law is not None:
        result = dict(result, readout_epoch=asdict(readout_law),
            completion=("full drain after each checked spatial group before input/ACC/RHS reuse"
                        if output_group is not None else
                        "full drain after each original output block before input/ACC/RHS reuse"))
    if readout_transfer_law is not None:
        transfer_blocks = blocks if output_group is None else tuple(
            tuple(ConvolutionOutputTile(t.output_column, t.columns,
                  p.accumulator_row + output_group.accumulator_row(0, n, active_column_blocks=len(block)))
                  for n, t in enumerate(block)) for block in blocks)
        result = dict(result, readout_transfers=dict(law_sha256=transfer_identity,
            blocks=[[[dict(output_column=t.output_column, columns=t.columns, accumulator_row=t.accumulator_row)
                       for t in group] for group in plan_convolution_readout_transfers(
                           p, block, readout_transfer_law, rows=p.row_tile)] for block in transfer_blocks],
            original_stores_per_spatial_tile=sum(map(len, blocks)),
            physical_stores_per_spatial_tile=sum(len(plan_convolution_readout_transfers(
                p, block, readout_transfer_law, rows=p.row_tile)) for block in transfer_blocks),
            completion=("original real final block drain, before input/ACC/RHS ownership reuse"
                        if output_group is None else
                        "original real final spatial group drain, before input/ACC/RHS ownership reuse")))
    if input_rows is not None:
        result = dict(result, resident_input_rows=input_rows.record(), rolling_transport=rolling)
    if spatial_iteration == "linear":
        result = dict(result, spatial_iteration="linear",
            spatial_tiles=(batch*oh*ow+p.row_tile-1)//p.row_tile,
            affine_gather="exact valid-window runs split at output row and image boundaries")
    if rhs is not None:
        if rhs.identity != rhs_identity:
            raise ValueError("original convolution RHS placement changed during realization")
        if rhs.column_intervals is not None:
            result = dict(result, column_blocks=[[asdict(t) for t in block] for block in blocks],
                partial_original_column_intervals=rhs.column_intervals, whole_invocation=False)
        return dict(result, rhs_residency=rhs.summary(), realization_sha256=rhs_identity)
    return result



@dataclass(frozen=True)
class ConvolutionStorageAlternative:
    """Joint private input/RHS/output ownership, without a profitability score.

    Input layouts and target effect laws are supplied by their original owner.
    Existing planners retain authority over source coverage, disjoint storage,
    complete reduction phases and the rolling read horizon. No target address,
    extra capacity or independent-output permission is inferred here.
    """
    schedule: ConvolutionSchedule
    input_rows: ConvolutionResidentInputRows
    rhs: ConvolutionRhsResidency | ConvolutionRhsPhases | None
    output_group: object
    max_row_blocks: int
    max_output_rows: int
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, "_identity", _digest(self._record()))

    def _validate(self):
        from .convolution_output_groups import plan_rolling_convolution_output_group
        if (type(self.schedule) is not ConvolutionSchedule
                or type(self.input_rows) is not ConvolutionResidentInputRows
                or any(type(v) is not int or not 1 <= v <= 16
                       for v in (self.max_row_blocks, self.max_output_rows))):
            raise ValueError("joint convolution storage needs original typed ownership and bounded groups")
        self.schedule.verify_current()
        self.input_rows.revalidate()
        p = self.schedule.source.plan
        if (self.input_rows.source_identity != self.schedule.source.identity
                or self.input_rows.window != p.window or self.input_rows.channel_tile != p.channel_tile
                or self.input_rows.end > p.operand_rows):
            raise ValueError("joint convolution storage changes original input or private capacity")
        if self.rhs is not None:
            if type(self.rhs) not in (ConvolutionRhsResidency, ConvolutionRhsPhases):
                raise ValueError("joint convolution storage needs its original RHS owner")
            self.rhs.verify_current()
            if self.rhs.schedule.identity != self.schedule.identity:
                raise ValueError("joint convolution storage changes original RHS schedule")
            if type(self.rhs) is ConvolutionRhsPhases and self.rhs.input_rows != self.input_rows:
                raise ValueError("joint convolution phases change original input footprint")
            phases = self.rhs.phases if type(self.rhs) is ConvolutionRhsPhases else (self.rhs,)
            for phase in phases:
                _phase_input(self.schedule, self.input_rows, phase.layout.operand_rows, phase.layout.rhs_base)
        expected = plan_rolling_convolution_output_group(self.schedule, self.rhs, self.input_rows,
            max_row_blocks=self.max_row_blocks, max_output_rows=self.max_output_rows)
        if type(self.output_group) is not type(expected) or self.output_group != expected:
            raise ValueError("joint convolution output ownership differs from its original checked plan")

    def _record(self):
        groups = self.output_group if type(self.output_group) is tuple else (self.output_group,)
        return dict(schema="merlin.convolution_joint_storage.v1", schedule_sha256=self.schedule.identity,
            input_rows=self.input_rows.record(), rhs=None if self.rhs is None else self.rhs.summary(),
            output_groups=[None if group is None else asdict(group) for group in groups],
            bounds=dict(max_row_blocks=self.max_row_blocks, max_output_rows=self.max_output_rows))

    def verify_current(self):
        self._validate()
        if _digest(self._record()) != self._identity:
            raise ValueError("joint convolution source, storage or effects changed after planning")

    @property
    def identity(self):
        self.verify_current()
        return self._identity

    def summary(self):
        self.verify_current()
        phases = self.rhs.phases if type(self.rhs) is ConvolutionRhsPhases else (self.rhs,)
        return dict(self._record(), realization_sha256=self._identity,
            structural_effects=dict(input_private_rows=self.input_rows.end-self.input_rows.base,
                rhs_private_rows=[None if phase is None else phase.layout.rhs_rows for phase in phases],
                complete_input_traversals=len(phases),
                performance_score=None, basis="source and declared storage, not elapsed cycles"))


def plan_convolution_storage_alternative(schedule, input_rows, *, operand_rows, rhs_law,
                                         rhs_phase_law=None, max_row_blocks=16, max_output_rows=1,
                                         rhs_column_blocks=None):
    """Plan RHS phases AFTER selecting one exact original input footprint.

    Call this for several independently admitted row maps rather than first
    maximizing input residency and treating the remaining RHS capacity as fixed.
    All legal alternatives survive for actual emission/cycle comparison. A
    capacity fallback remains ordinary streaming; it is not a measured loser.
    """
    from .convolution_output_groups import plan_rolling_convolution_output_group
    rhs = plan_convolution_rhs_phases(schedule, operand_rows=operand_rows, law=rhs_law,
        phase_law=rhs_phase_law, input_rows=input_rows, max_column_blocks=rhs_column_blocks)
    group = plan_rolling_convolution_output_group(schedule, rhs, input_rows,
        max_row_blocks=max_row_blocks, max_output_rows=max_output_rows)
    return ConvolutionStorageAlternative(schedule, input_rows, rhs, group, max_row_blocks, max_output_rows)


def convolution_storage_pass_space(schedule, *, input_rows_choices, operand_rows, rhs_law,
                                   verify_source: Callable, realize: Callable, rhs_phase_law=None,
                                   row_block_limits=(16,), output_row_limits=(1,), max_candidates=8,
                                   proposal_cache=None, proposal_key=None, rhs_column_limits=(None,)):
    """Bounded registry choices jointly covering input, RHS phases and M groups.

    Targets supply exact original row maps and reopen their source/effect laws.
    ``realize(alternative)`` must consume all checked owners through ordinary
    emission and return EmittedProgram bound to alternative.identity. Existing
    explore_convolution_schedules deduplicates actual bytes; its optional cycle
    provider, never these structural summaries, determines performance rankings.
    """
    if (type(schedule) is not ConvolutionSchedule or not callable(verify_source) or not callable(realize)
            or type(input_rows_choices) is not tuple or not 1 <= len(input_rows_choices) <= 16
            or any(type(rows) is not ConvolutionResidentInputRows for rows in input_rows_choices)
            or type(rhs_column_limits) is not tuple or not 1 <= len(rhs_column_limits) <= 16
            or len(set(rhs_column_limits)) != len(rhs_column_limits)
            or any(value is not None and (type(value) is not int or not 0 < value < 1 << 63)
                   for value in rhs_column_limits)
            or any(type(values) is not tuple or not 1 <= len(values) <= 16
                   or len(set(values)) != len(values)
                   or any(type(value) is not int or not 1 <= value <= 16 for value in values)
                   for values in (row_block_limits, output_row_limits))):
        raise ValueError("joint convolution exploration requires bounded original layouts and group choices")
    schedule.verify_current()
    # Construct once now: invalid original maps/laws refuse before ANY target
    # callback, instead of being mistaken for ordinary unavailable performance.
    for rows in input_rows_choices:
        plan_convolution_storage_alternative(schedule, rows, operand_rows=operand_rows,
            rhs_law=rhs_law, rhs_phase_law=rhs_phase_law,
            max_row_blocks=row_block_limits[0], max_output_rows=output_row_limits[0],
            rhs_column_blocks=rhs_column_limits[0])
    identities = tuple(_digest(rows.record()) for rows in input_rows_choices)
    if len(set(identities)) != len(identities):
        raise ValueError("joint convolution exploration requires distinct original input layouts")
    original = (schedule.identity, identities, _law_digest(rhs_law), _law_digest(rhs_phase_law))

    def reopen(source):
        current = (schedule.identity, tuple(_digest(rows.record()) for rows in input_rows_choices),
                   _law_digest(rhs_law), _law_digest(rhs_phase_law))
        if type(source) is not ConvolutionExplorationSource or source.identity != schedule.source.identity or current != original:
            raise ValueError("joint convolution exploration source, layouts or laws changed")
        if verify_source(source) is not None:
            raise ValueError("joint convolution original-source callback did not validate input")
        if (schedule.identity, tuple(_digest(rows.record()) for rows in input_rows_choices),
                _law_digest(rhs_law), _law_digest(rhs_phase_law)) != original:
            raise ValueError("joint convolution original-source callback changed ownership")

    def storage(options):
        def action(session):
            reopen(session.module)
            value = plan_convolution_storage_alternative(schedule, input_rows_choices[options["input_layout"]],
                operand_rows=operand_rows, rhs_law=rhs_law, rhs_phase_law=rhs_phase_law,
                max_row_blocks=options["row_blocks"], max_output_rows=options["output_rows"],
                rhs_column_blocks=(None if options["rhs_column_blocks"] == "capacity" else
                                   options["rhs_column_blocks"]))
            return {"convolution_joint_storage": value}
        return RegisteredPass(PassSpec("joint_input_rhs_output_storage", 0,
            produces=("convolution_joint_storage",), mutates_ir=False, ir_level="counted_convolution_source"), action)

    def emit(_):
        def action(session):
            reopen(session.module)
            value = session.analysis("convolution_joint_storage")
            identity = value.identity
            result = realize(value)
            reopen(session.module)
            if type(result) is not EmittedProgram or result.schedule_sha256 != identity or value.identity != identity:
                raise ValueError("target program is not bound to its actual joint convolution storage")
            return {"convolution_program": result}
        return RegisteredPass(PassSpec("realize_joint_convolution_storage", 1,
            requires=("convolution_joint_storage",), produces=("convolution_program",),
            mutates_ir=False, ir_level="counted_convolution_source"), action)

    return generate_candidates((PassFamily("joint_storage", storage,
        knobs=(("input_layout", tuple(range(len(input_rows_choices)))),
               ("row_blocks", row_block_limits), ("output_rows", output_row_limits),
               ("rhs_column_blocks", tuple("capacity" if n is None else n for n in rhs_column_limits)))),
        PassFamily("emit", emit)), output="convolution_program", max_candidates=max_candidates,
        proposal_cache=proposal_cache, proposal_key=proposal_key)


def convolution_pass_space(source, *, verify_source: Callable, realize: Callable,
                           retention_law=None, issue_law=None, epoch_law=None,
                           live_tile_limits=(1, None), issue_modes=(False, True),
                           compute_pair_limits=(1,), rhs_modes=(False,), rhs_law=None,
                           private_operand_rows=None, rhs_base=None, max_candidates=8,
                           rhs_phase_law=None, input_rows=None):
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
    Optional rhs_modes includes the literal "phased" to request the same
    capacity-derived complete-N phase owner; its separate source/effect law and
    original input footprint remain explicit, not inferred by the registry.
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
        or any(type(v) is not bool and v != "phased" for v in rhs_modes)):
        raise ValueError("convolution registry requires bounded choices and original target callbacks")
    for limit in live_tile_limits:
        ConvolutionScheduleChoice(limit)
    for law, cls in ((retention_law, ConvolutionInputRetentionLaw), (issue_law, ConvolutionLoadIssueLaw),
                     (epoch_law, ConvolutionComputeEpochLaw), (rhs_law, ConvolutionRhsResidencyLaw)):
        if law is not None:
            if type(law) is not cls:
                raise ValueError("convolution registry requires typed original effect laws")
            law.__post_init__()
    if rhs_phase_law is not None:
        if type(rhs_phase_law) is not ConvolutionRhsPhaseLaw:
            raise ValueError("convolution registry requires a typed original RHS phase law")
        rhs_phase_law.require(source)
    if input_rows is not None:
        if type(input_rows) is not ConvolutionResidentInputRows:
            raise ValueError("convolution registry requires the original resident input footprint")
        input_rows.revalidate()
        if input_rows.source_identity != source.identity or input_rows.window != source.plan.window:
            raise ValueError("convolution registry resident input differs from original source")
    def law_ids():
        return (_law_digest(retention_law), _law_digest(issue_law), _law_digest(epoch_law), _law_digest(rhs_law),
                _law_digest(rhs_phase_law), None if input_rows is None else _digest(input_rows.record()))

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
            if options["enabled"] == "phased":
                rhs = plan_convolution_rhs_phases(schedule, operand_rows=private_operand_rows,
                    law=rhs_law, phase_law=rhs_phase_law, rhs_base=rhs_base, input_rows=input_rows)
            else:
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
