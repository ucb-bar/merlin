"""Typed convolution alternatives and explicit storage-capacity conditions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


from .dense_rhs_residency import DenseRhsResidency, plan_dense_rhs_residency
from .library_sources import admission_point
from .window_transfer import (
    ConvolutionComputeEpochLaw,
    ConvolutionInputRetentionLaw,
    ConvolutionLoadIssueLaw,
    ConvolutionOutputTile,
    ConvolutionResidentInputRows,
    CountedConvolutionPlan,
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
        if (
            type(self.plan) is not CountedConvolutionPlan
            or type(self.source_identity) is not str
            or not self.source_identity.strip()
            or type(self.column_intervals) is not tuple
            or not self.column_intervals
        ):
            raise ValueError("convolution exploration requires original typed source and intervals")
        self.plan.__post_init__()
        end = 0
        for interval in self.column_intervals:
            if (
                type(interval) is not tuple
                or len(interval) != 2
                or any(type(v) is not int for v in interval)
                or interval[0] != end
                or not 0 < interval[1] <= self.plan.column_tile
            ):
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
        if (
            self.live_tiles is not None
            and (type(self.live_tiles) is not int or not 0 < self.live_tiles < 2**31)
            or type(self.load_issue) is not bool
            or type(self.compute_pairs) is not int
            or not 0 < self.compute_pairs < 2**31
        ):
            raise ValueError("convolution choices need a bounded tile limit and explicit issue mode")


def _column_blocks(source, choice):
    source.__post_init__()
    choice.__post_init__()
    p = source.plan
    capacity = (p.accumulator_rows - p.accumulator_row) // p.row_tile
    slots = capacity if choice.live_tiles is None else min(capacity, choice.live_tiles)
    return tuple(
        tuple(
            ConvolutionOutputTile(column, width, p.accumulator_row + i * p.row_tile)
            for i, (column, width) in enumerate(source.column_intervals[start : start + slots])
        )
        for start in range(0, len(source.column_intervals), slots)
    )


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
        if (
            type(self.source) is not ConvolutionExplorationSource
            or type(self.choice) is not ConvolutionScheduleChoice
            or self.column_blocks != _column_blocks(self.source, self.choice)
        ):
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
        if (count > 1 or self.choice.compute_pairs > 1) and (
            r is None
            or count > r.max_live_tiles
            or not all((r.input_preserved, r.independent_outputs, r.immutable_mapped_disjoint_buffers))
        ):
            raise ValueError("convolution schedule lacks original input-retention premises")
        if self.choice.load_issue:
            if i is None or not all(
                (
                    i.local_range_dependencies,
                    i.ordered_load_configuration,
                    i.paired_descriptor_compute,
                    i.finite_queue_backpressure,
                    i.complete_device_drain,
                    i.immutable_mapped_disjoint_buffers,
                )
            ):
                raise ValueError("convolution schedule lacks original bounded-issue premises")
            if self.source.plan.row_tile + 1 + count * int(self.source.plan.bias) > i.load_issue_budget:
                raise ValueError("convolution schedule exceeds its original first-epoch load budget")
            if (count > 1 or self.choice.compute_pairs > 1) and (
                r.hardware_identity != i.hardware_identity or r.numerical_policy_identity != i.numerical_policy_identity
            ):
                raise ValueError("convolution retention and issue laws select different contracts")

        if self.epoch_law is not None:
            if type(self.epoch_law) is not ConvolutionComputeEpochLaw:
                raise ValueError("convolution compute epoch law must be typed")
            self.epoch_law.__post_init__()
        if self.choice.compute_pairs > 1:
            if not self.choice.load_issue or self.epoch_law is None:
                raise ValueError("convolution compute epochs need delayed issue and a stronger original source law")
            self.epoch_law.require(self.choice.compute_pairs)
            if (
                self.epoch_law.hardware_identity != i.hardware_identity
                or self.epoch_law.numerical_policy_identity != i.numerical_policy_identity
            ):
                raise ValueError("convolution compute epoch and issue laws select different contracts")

    def verify_current(self):
        self._validate()
        if (
            self.source.identity != self._source_sha256
            or _law_digest(self.retention_law) != self._retention_sha256
            or _law_digest(self.issue_law) != self._issue_sha256
            or _law_digest(self.epoch_law) != self._epoch_sha256
        ):
            raise ValueError("original convolution source or target laws changed after planning")

    def summary(self):
        self.verify_current()
        return dict(
            schema="merlin.convolution_schedule.v1",
            source_sha256=self._source_sha256,
            plan_sha256=self.source.plan.identity,
            choice=asdict(self.choice),
            column_blocks=[[asdict(tile) for tile in block] for block in self.column_blocks],
            retention_law_sha256=self._retention_sha256,
            issue_law_sha256=self._issue_sha256,
            epoch_law_sha256=self._epoch_sha256,
            qualification="source/effect premises remain independently caller-owned",
        )

    @property
    def identity(self):
        return _digest(self.summary())


def plan_convolution_schedule(
    source, choice=ConvolutionScheduleChoice(), *, retention_law=None, issue_law=None, epoch_law=None
):
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
        if any(
            type(v) is not str or not v.strip()
            for v in (
                self.hardware_identity,
                self.numerical_policy_identity,
                self.source_identity,
                self.qualification_identity,
            )
        ) or any(
            type(v) is not bool
            for v in (
                self.weight_transfer_only,
                self.immutable_mapped_disjoint_buffers,
                self.other_effects_preserve_rhs,
                self.exclusive_private_rows,
                self.entry_quiescence_and_complete_preparation,
            )
        ):
            raise ValueError("convolution RHS residency requires typed original source premises")

    def require(self):
        self.__post_init__()
        if not all(
            (
                self.weight_transfer_only,
                self.immutable_mapped_disjoint_buffers,
                self.other_effects_preserve_rhs,
                self.exclusive_private_rows,
                self.entry_quiescence_and_complete_preparation,
            )
        ):
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
        if (
            type(self.schedule) is not ConvolutionSchedule
            or type(self.layout) is not DenseRhsResidency
            or type(self.law) is not ConvolutionRhsResidencyLaw
        ):
            raise ValueError("convolution RHS residency requires original checked schedule and capacity plan")
        self.schedule.verify_current()
        self.law.require()
        if self.layout != _rhs_capacity(
            self.schedule, self.layout.operand_rows, self.assigned_rhs_base, self.column_intervals
        ):
            raise ValueError("convolution RHS placement differs from original private capacity/source geometry")
        for law in (self.schedule.retention_law, self.schedule.issue_law, self.schedule.epoch_law):
            if law is not None and (
                law.hardware_identity != self.law.hardware_identity
                or law.numerical_policy_identity != self.law.numerical_policy_identity
            ):
                raise ValueError("convolution RHS and execution laws select different contracts")
        object.__setattr__(self, "_schedule_sha256", self.schedule.identity)
        object.__setattr__(self, "_law_sha256", _law_digest(self.law))
        object.__setattr__(self, "_layout_sha256", self._layout_identity())

    def verify_current(self):
        self.schedule.verify_current()
        self.law.require()
        if (
            self.schedule.identity != self._schedule_sha256
            or _law_digest(self.law) != self._law_sha256
            or self._layout_identity() != self._layout_sha256
            or self.layout
            != _rhs_capacity(self.schedule, self.layout.operand_rows, self.assigned_rhs_base, self.column_intervals)
        ):
            raise ValueError("original convolution RHS source/law/private placement changed after planning")

    def _layout_identity(self):
        original = (self.assigned_rhs_base, asdict(self.layout))
        return _digest(original if self.column_intervals is None else (*original, self.column_intervals))

    def summary(self):
        self.verify_current()
        result = dict(
            schema="merlin.convolution_rhs_residency.v1",
            schedule_sha256=self._schedule_sha256,
            law_sha256=self._law_sha256,
            physical_panel_layout=asdict(self.layout),
            assigned_rhs_base=self.assigned_rhs_base,
            lifetime="one invocation; preparation fully completes before first consumer",
            reduction="original per-kernel channel chunks, including each independent tail",
        )
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
    if (
        type(intervals) is not tuple
        or not intervals
        or any(type(i) is not tuple or len(i) != 2 or any(type(v) is not int for v in i) for i in intervals)
    ):
        raise ValueError("RHS phase needs exact original column intervals")
    try:
        first = original.index(intervals[0])
    except ValueError as exc:
        raise ValueError("RHS phase changes original column boundaries") from exc
    if original[first : first + len(intervals)] != intervals:
        raise ValueError("RHS phase must be one ordered original column slice")
    return intervals


def _rhs_capacity(schedule, operand_rows, rhs_base=None, column_intervals=None):
    p = schedule.source.plan
    if (
        type(operand_rows) is not int
        or not max(p.lhs_row + p.row_tile, p.rhs_row + p.channel_tile) <= operand_rows <= p.operand_rows
    ):
        raise ValueError("convolution RHS rows must be explicitly owned within original operand capacity")
    kh, kw = p.window.kernel
    ci = p.window.input_shape[-1]
    chunks = kh * kw * ((ci + p.channel_tile - 1) // p.channel_tile)
    # Reuse the pure dense *capacity* planner for a physical tile-slot array.
    # These padded slot extents are not a flattened convolution or source shape:
    # CI tails at each kernel position and source scale intervals remain intact.
    base = (max(p.rhs_row, p.lhs_row + p.row_tile) + p.channel_tile - 1) // p.channel_tile * p.channel_tile
    if rhs_base is not None:
        if (
            type(rhs_base) is not int
            or rhs_base < p.lhs_row + p.row_tile
            or rhs_base % p.channel_tile
            or rhs_base >= operand_rows
        ):
            raise ValueError("assigned convolution RHS base must be aligned and disjoint within private rows")
        base = rhs_base
    return plan_dense_rhs_residency(
        1,
        chunks * p.channel_tile,
        len(_rhs_intervals(schedule, column_intervals)) * p.channel_tile,
        block=p.channel_tile,
        operand_rows=operand_rows,
        accumulator_rows=p.accumulator_rows,
        rhs_base=base,
        max_static_rhs_tiles=operand_rows // p.channel_tile,
    )


@admission_point(
    "merlin.compile.scheduling.convolution_exploration.plan_convolution_rhs_residency",
    fallback_route="caller-selected original route; ordinary weight path",
    none_refuses="RHS residency does not fit the selected operand capacity (returned None)",
)
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
        if (
            type(source) is not ConvolutionExplorationSource
            or self.source_identity != source.identity
            or type(self.qualification_identity) is not str
            or not self.qualification_identity.strip()
            or any(
                type(v) is not bool or not v
                for v in (
                    self.independent_output_columns,
                    self.immutable_inputs,
                    self.no_intermediate_observers,
                    self.full_completion_before_reprepare,
                )
            )
        ):
            raise ValueError("RHS phases lack original independent-output/observer/completion premises")


def _phase_input(schedule, input_rows, operand_rows, rhs_base):
    """Bind input union to full source SP and RHS to its borrowed row prefix.

    A source input segment may lie above the separately owned RHS prefix.
    The complete physical source capacity still bounds every input segment.
    """
    if input_rows is not None:
        if type(input_rows) is not ConvolutionResidentInputRows:
            raise ValueError("RHS phases require the exact original resident input footprint")
        input_rows.revalidate()
        if (
            input_rows.source_identity != schedule.source.identity
            or input_rows.window != schedule.source.plan.window
            or input_rows.channel_tile != schedule.source.plan.channel_tile
            or input_rows.end > schedule.source.plan.operand_rows
        ):
            raise ValueError("RHS phases require the exact original resident input footprint")
        input_rows.revalidate()
        if not input_rows.disjoint_from(rhs_base, operand_rows - rhs_base):
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
        if (
            type(self.schedule) is not ConvolutionSchedule
            or type(self.law) is not ConvolutionRhsPhaseLaw
            or type(self.phases) is not tuple
            or len(self.phases) < 2
            or any(type(p) is not ConvolutionRhsResidency or p.column_intervals is None for p in self.phases)
        ):
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
            if (
                phase.schedule.identity != self.schedule.identity
                or phase.law != first.law
                or phase.layout.operand_rows != first.layout.operand_rows
                or phase.assigned_rhs_base != first.assigned_rhs_base
            ):
                raise ValueError("RHS phases change original source, law or assigned capacity")
        _phase_input(self.schedule, self.input_rows, first.layout.operand_rows, first.layout.rhs_base)
        original = self.schedule.source.column_intervals
        capacity = (first.layout.operand_rows - first.layout.rhs_base) // (first.layout.k_tiles * first.layout.block)
        if self.max_column_blocks is not None:
            capacity = min(capacity, self.max_column_blocks)
        expected = tuple(original[i : i + capacity] for i in range(0, len(original), capacity))
        if tuple(p.column_intervals for p in self.phases) != expected:
            raise ValueError("RHS phases must cover every original column once using the bounded legal capacity")

    def _current_identity(self):
        original = (
            self.schedule.identity,
            [p.summary() for p in self.phases],
            asdict(self.law),
            None if self.input_rows is None else self.input_rows.record(),
        )
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
        return dict(
            schema="merlin.convolution_rhs_phases.v1",
            schedule_sha256=self.schedule.identity,
            realization_sha256=self._identity,
            phases=[p.summary() for p in self.phases],
            input_footprint=None if self.input_rows is None else self.input_rows.record(),
            **({} if self.max_column_blocks is None else {"max_column_blocks": self.max_column_blocks}),
            coverage="each original output column exactly once; complete original K and scalar exits",
            completion="real full phase-last-use drain before any RHS replacement",
        )


@admission_point(
    "merlin.compile.scheduling.convolution_exploration.plan_convolution_rhs_phases",
    fallback_route="caller-selected original route; unchanged ordinary weight path",
    none_refuses="RHS phase interval does not fit the selected operand capacity (returned None)",
)
def plan_convolution_rhs_phases(
    schedule, *, operand_rows, law, phase_law=None, rhs_base=None, input_rows=None, max_column_blocks=None
):
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
    if max_column_blocks is not None and (type(max_column_blocks) is not int or not 0 < max_column_blocks < 1 << 63):
        raise ValueError("RHS phase column bound must be an explicit positive integer")
    p = schedule.source.plan
    if rhs_base is None and input_rows is not None:
        if type(input_rows) is not ConvolutionResidentInputRows:
            raise ValueError("RHS phases require the exact original resident input footprint")
        input_rows.revalidate()
        rhs_base = (input_rows.end + p.channel_tile - 1) // p.channel_tile * p.channel_tile
    full = plan_convolution_rhs_residency(schedule, operand_rows=operand_rows, law=law, rhs_base=rhs_base)
    base = (max(p.rhs_row, p.lhs_row + p.row_tile) + p.channel_tile - 1) // p.channel_tile * p.channel_tile
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
    count = (operand_rows - base) // single.rhs_rows
    if max_column_blocks is not None:
        count = min(count, max_column_blocks)
    phases = tuple(
        ConvolutionRhsResidency(schedule, _rhs_capacity(schedule, operand_rows, base, subset), law, base, subset)
        for subset in (original[i : i + count] for i in range(0, len(original), count))
    )
    return ConvolutionRhsPhases(schedule, phases, phase_law, input_rows, max_column_blocks)
