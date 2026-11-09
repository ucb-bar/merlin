"""Enumerate joint original convolution input, RHS and output storage choices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from merlin.compiler.schedule.convolution_choices import ConvolutionExplorationSource as ConvolutionExplorationSource
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsPhases as ConvolutionRhsPhases
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsResidency as ConvolutionRhsResidency
from merlin.compiler.schedule.convolution_choices import ConvolutionSchedule as ConvolutionSchedule
from merlin.compiler.schedule.convolution_choices import _digest as _digest
from merlin.compiler.schedule.convolution_choices import _law_digest as _law_digest
from merlin.compiler.schedule.convolution_choices import _phase_input as _phase_input
from merlin.compiler.schedule.convolution_choices import plan_convolution_rhs_phases as plan_convolution_rhs_phases
from merlin.compiler.schedule.window_transfer import ConvolutionResidentInputRows
from merlin.transforms.pass_exploration import EmittedProgram, PassFamily, generate_candidates
from merlin.transforms.pipeline_policy import RegisteredPass
from merlin.transforms.pipeline_state import PassSpec


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
        from merlin.compiler.schedule.convolution_output_groups import plan_rolling_convolution_output_group

        if (
            type(self.schedule) is not ConvolutionSchedule
            or type(self.input_rows) is not ConvolutionResidentInputRows
            or any(type(v) is not int or not 1 <= v <= 16 for v in (self.max_row_blocks, self.max_output_rows))
        ):
            raise ValueError("joint convolution storage needs original typed ownership and bounded groups")
        self.schedule.verify_current()
        self.input_rows.revalidate()
        p = self.schedule.source.plan
        if (
            self.input_rows.source_identity != self.schedule.source.identity
            or self.input_rows.window != p.window
            or self.input_rows.channel_tile != p.channel_tile
            or self.input_rows.end > p.operand_rows
        ):
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
        expected = plan_rolling_convolution_output_group(
            self.schedule,
            self.rhs,
            self.input_rows,
            max_row_blocks=self.max_row_blocks,
            max_output_rows=self.max_output_rows,
        )
        if type(self.output_group) is not type(expected) or self.output_group != expected:
            raise ValueError("joint convolution output ownership differs from its original checked plan")

    def _record(self):
        groups = self.output_group if type(self.output_group) is tuple else (self.output_group,)
        return dict(
            schema="merlin.convolution_joint_storage.v1",
            schedule_sha256=self.schedule.identity,
            input_rows=self.input_rows.record(),
            rhs=None if self.rhs is None else self.rhs.summary(),
            output_groups=[None if group is None else asdict(group) for group in groups],
            bounds=dict(max_row_blocks=self.max_row_blocks, max_output_rows=self.max_output_rows),
        )

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
        return dict(
            self._record(),
            realization_sha256=self._identity,
            structural_effects=dict(
                input_private_rows=self.input_rows.allocated_rows,
                rhs_private_rows=[None if phase is None else phase.layout.rhs_rows for phase in phases],
                complete_input_traversals=len(phases),
                performance_score=None,
                basis="source and declared storage, not elapsed cycles",
            ),
        )


def plan_convolution_storage_alternative(
    schedule,
    input_rows,
    *,
    operand_rows,
    rhs_law,
    rhs_phase_law=None,
    max_row_blocks=16,
    max_output_rows=1,
    rhs_column_blocks=None,
    rhs_base=None,
):
    """Plan RHS phases AFTER selecting one exact original input footprint.

    Call this for several independently admitted row maps rather than first
    maximizing input residency and treating the remaining RHS capacity as fixed.
    Explicit rhs_base and operand_rows can bound an owned RHS gap between
    original input segments; the source plan still bounds full physical SP.
    All legal alternatives survive for actual emission/cycle comparison. A
    capacity fallback remains ordinary streaming; it is not a measured loser.
    """
    from merlin.compiler.schedule.convolution_output_groups import plan_rolling_convolution_output_group

    rhs = plan_convolution_rhs_phases(
        schedule,
        operand_rows=operand_rows,
        law=rhs_law,
        phase_law=rhs_phase_law,
        input_rows=input_rows,
        max_column_blocks=rhs_column_blocks,
        rhs_base=rhs_base,
    )
    group = plan_rolling_convolution_output_group(
        schedule, rhs, input_rows, max_row_blocks=max_row_blocks, max_output_rows=max_output_rows
    )
    return ConvolutionStorageAlternative(schedule, input_rows, rhs, group, max_row_blocks, max_output_rows)


def convolution_storage_pass_space(
    schedule,
    *,
    input_rows_choices,
    operand_rows,
    rhs_law,
    verify_source: Callable,
    realize: Callable,
    rhs_phase_law=None,
    row_block_limits=(16,),
    output_row_limits=(1,),
    max_candidates=8,
    proposal_cache=None,
    proposal_key=None,
    rhs_column_limits=(None,),
):
    """Bounded registry choices jointly covering input, RHS phases and M groups.

    Targets supply exact original row maps and reopen their source/effect laws.
    ``realize(alternative)`` must consume all checked owners through ordinary
    emission and return EmittedProgram bound to alternative.identity. Existing
    explore_convolution_schedules deduplicates actual bytes; its optional cycle
    provider, never these structural summaries, determines performance rankings.
    """
    if (
        type(schedule) is not ConvolutionSchedule
        or not callable(verify_source)
        or not callable(realize)
        or type(input_rows_choices) is not tuple
        or not 1 <= len(input_rows_choices) <= 16
        or any(type(rows) is not ConvolutionResidentInputRows for rows in input_rows_choices)
        or type(rhs_column_limits) is not tuple
        or not 1 <= len(rhs_column_limits) <= 16
        or len(set(rhs_column_limits)) != len(rhs_column_limits)
        or any(value is not None and (type(value) is not int or not 0 < value < 1 << 63) for value in rhs_column_limits)
        or any(
            type(values) is not tuple
            or not 1 <= len(values) <= 16
            or len(set(values)) != len(values)
            or any(type(value) is not int or not 1 <= value <= 16 for value in values)
            for values in (row_block_limits, output_row_limits)
        )
    ):
        raise ValueError("joint convolution exploration requires bounded original layouts and group choices")
    schedule.verify_current()
    # Construct once now: invalid original maps/laws refuse before ANY target
    # callback, instead of being mistaken for ordinary unavailable performance.
    for rows in input_rows_choices:
        plan_convolution_storage_alternative(
            schedule,
            rows,
            operand_rows=operand_rows,
            rhs_law=rhs_law,
            rhs_phase_law=rhs_phase_law,
            max_row_blocks=row_block_limits[0],
            max_output_rows=output_row_limits[0],
            rhs_column_blocks=rhs_column_limits[0],
        )
    identities = tuple(_digest(rows.record()) for rows in input_rows_choices)
    if len(set(identities)) != len(identities):
        raise ValueError("joint convolution exploration requires distinct original input layouts")
    original = (schedule.identity, identities, _law_digest(rhs_law), _law_digest(rhs_phase_law))

    def reopen(source):
        current = (
            schedule.identity,
            tuple(_digest(rows.record()) for rows in input_rows_choices),
            _law_digest(rhs_law),
            _law_digest(rhs_phase_law),
        )
        if (
            type(source) is not ConvolutionExplorationSource
            or source.identity != schedule.source.identity
            or current != original
        ):
            raise ValueError("joint convolution exploration source, layouts or laws changed")
        if verify_source(source) is not None:
            raise ValueError("joint convolution original-source callback did not validate input")
        if (
            schedule.identity,
            tuple(_digest(rows.record()) for rows in input_rows_choices),
            _law_digest(rhs_law),
            _law_digest(rhs_phase_law),
        ) != original:
            raise ValueError("joint convolution original-source callback changed ownership")

    def storage(options):
        def action(session):
            reopen(session.module)
            value = plan_convolution_storage_alternative(
                schedule,
                input_rows_choices[options["input_layout"]],
                operand_rows=operand_rows,
                rhs_law=rhs_law,
                rhs_phase_law=rhs_phase_law,
                max_row_blocks=options["row_blocks"],
                max_output_rows=options["output_rows"],
                rhs_column_blocks=(
                    None if options["rhs_column_blocks"] == "capacity" else options["rhs_column_blocks"]
                ),
            )
            return {"convolution_joint_storage": value}

        return RegisteredPass(
            PassSpec(
                "joint_input_rhs_output_storage",
                0,
                produces=("convolution_joint_storage",),
                mutates_ir=False,
                ir_level="counted_convolution_source",
            ),
            action,
        )

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

        return RegisteredPass(
            PassSpec(
                "realize_joint_convolution_storage",
                1,
                requires=("convolution_joint_storage",),
                produces=("convolution_program",),
                mutates_ir=False,
                ir_level="counted_convolution_source",
            ),
            action,
        )

    return generate_candidates(
        (
            PassFamily(
                "joint_storage",
                storage,
                knobs=(
                    ("input_layout", tuple(range(len(input_rows_choices)))),
                    ("row_blocks", row_block_limits),
                    ("output_rows", output_row_limits),
                    ("rhs_column_blocks", tuple("capacity" if n is None else n for n in rhs_column_limits)),
                ),
            ),
            PassFamily("emit", emit),
        ),
        output="convolution_program",
        max_candidates=max_candidates,
        proposal_cache=proposal_cache,
        proposal_key=proposal_key,
    )


# Preserve the original public lookup and pickle identities.
for _definition in (
    ConvolutionStorageAlternative,
    plan_convolution_storage_alternative,
    convolution_storage_pass_space,
):
    _definition.__module__ = "merlin.compiler.schedule.convolution_exploration"
del _definition
