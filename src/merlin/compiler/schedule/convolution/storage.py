"""Enumerate joint original convolution input, RHS and output storage choices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from merlin.compiler.schedule.convolution.choices import (
    ConvolutionExplorationSource,
    ConvolutionRhsPhases,
    ConvolutionRhsResidency,
    ConvolutionSchedule,
    _digest,
    _law_digest,
    _phase_input,
    _rhs_intervals,
    plan_convolution_rhs_phases,
)
from merlin.compiler.schedule.convolution.windows import (
    ConvolutionOutputTile,
    ConvolutionReadoutEpochLaw,
    ConvolutionReadoutTransferLaw,
    ConvolutionResidentInputRows,
    plan_convolution_readout_transfers,
)
from merlin.compiler.transforms.exploration.branch_policy import RegisteredPass
from merlin.compiler.transforms.exploration.pass_orders import EmittedProgram, PassFamily, generate_candidates
from merlin.compiler.transforms.pipeline_state import PassSpec

# ---- convolution_storage -----------------------------------------------------------------------


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
        from merlin.compiler.schedule.convolution.output_groups import plan_rolling_convolution_output_group

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
    from merlin.compiler.schedule.convolution.output_groups import plan_rolling_convolution_output_group

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
    _definition.__module__ = "merlin.compiler.schedule.convolution.exploration"
del _definition


# ---- convolution_admission ---------------------------------------------------------------------
# Validate selected convolution transport and output-group obligations.


def _verify_input_first_use(schedule, rows, group, blocks, grouped, law):
    if law is None:
        return
    from merlin.compiler.schedule.convolution.input_residency import PendingConvolutionInput

    if not grouped or len(blocks) != 1:
        raise ValueError("first-use input needs one complete original grouped consumer")
    PendingConvolutionInput(schedule, group, rows, law, image=0)


def _verify_batch_input_prefetch(schedule, rows, group, blocks, grouped, law):
    if law is None:
        return
    from merlin.compiler.schedule.spatial.rolling_lookahead import (
        RollingBatchFrontier,
        RollingBatchFrontierAdmission,
    )

    if (
        type(law) is not RollingBatchFrontierAdmission
        or not grouped
        or group is None
        or rows is None
        or schedule.epoch_law is None
        or len(blocks) != 1
    ):
        raise ValueError("batch input frontier requires its complete original consumer and target law")
    law.require()
    if (
        law.hardware_identity != schedule.epoch_law.hardware_identity
        or law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
    ):
        raise ValueError("batch input frontier differs from the original target/numerical scope")
    p = schedule.source.plan
    runs = p.window.kernel[0] * p.window.kernel[1] * ((p.window.input_shape[-1] + p.channel_tile - 1) // p.channel_tile)
    RollingBatchFrontier(schedule.source.identity, p.channel_tile).stages(rows.rolling_plan(), group.output_rows, runs)


def _verify_input_prefetch(schedule, rows, group, blocks, grouped, law):
    if law is None:
        return
    from merlin.compiler.schedule.spatial.rolling_lookahead import RollingLookaheadAdmission, _validate

    if (
        type(law) is not RollingLookaheadAdmission
        or not grouped
        or group is None
        or rows is None
        or schedule.epoch_law is None
        or len(blocks) != 1
    ):
        raise ValueError("input prefetch needs a complete grouped K-run consumer and selected target law")
    law.require()
    if not law.retained_consumer_continuations or law.hardware_identity != schedule.epoch_law.hardware_identity:
        raise ValueError("input prefetch lacks the original continuation/target admission")
    _validate(rows.rolling_plan(), group.output_rows)


def _phase_blocks(schedule, intervals):
    if intervals is None:
        return schedule.column_blocks
    columns = {column for column, _ in _rhs_intervals(schedule, intervals)}
    return tuple(
        part
        for block in schedule.column_blocks
        if (part := tuple(tile for tile in block if tile.output_column in columns))
    )


def _verify_readout_law(schedule, law, blocks):
    if law is None:
        return
    if (
        type(law) is not ConvolutionReadoutEpochLaw
        or not schedule.choice.load_issue
        or schedule.choice.compute_pairs <= 1
    ):
        raise ValueError("convolution readout epochs require a typed law and issued compute epochs")
    for block in blocks:
        law.require(schedule.source.plan, schedule.source.plan.row_tile, block)
    if (
        law.hardware_identity != schedule.epoch_law.hardware_identity
        or law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
    ):
        raise ValueError("convolution readout and compute epochs select different contracts")


def _verify_readout_transfers(schedule, law, readout_law, blocks):
    if law is None:
        return
    if (
        type(law) is not ConvolutionReadoutTransferLaw
        or readout_law is None
        or law.hardware_identity != readout_law.hardware_identity
        or law.numerical_policy_identity != readout_law.numerical_policy_identity
    ):
        raise ValueError("wide readout needs its separate map and original producer/readout epoch")
    for block in blocks:
        plan_convolution_readout_transfers(schedule.source.plan, block, law, rows=schedule.source.plan.row_tile)


def _verify_spatial_output_group(
    schedule,
    builder,
    lower,
    complete,
    *,
    rhs,
    input_rows,
    spatial_iteration,
    readout_law,
    readout_transfer_law,
    output_group,
    array_law,
    blocks,
):
    """Check active original column ownership and budgets before target effects."""
    p = schedule.source.plan
    group_identity = None
    array_identity = None
    if output_group is not None or array_law is not None:
        from merlin.compiler.schedule.convolution.output_groups import (
            ConvolutionOutputGroup,
            ConvolutionRowFragmentGroup,
            plan_rolling_convolution_output_group,
            plan_rolling_convolution_row_fragments,
        )
        from merlin.compiler.schedule.convolution.windows import _validate_convolution_load_issue
        from merlin.compiler.schedule.panels.array_retention import DenseArrayRetentionAdmission

        if (
            type(array_law) is not DenseArrayRetentionAdmission
            or type(output_group) not in (ConvolutionOutputGroup, ConvolutionRowFragmentGroup)
            or spatial_iteration != "rows"
            or readout_law is None
        ):
            raise ValueError("spatial grouping needs separate array/readout premises")
        array_law.__post_init__()
        if (
            schedule.retention_law is None
            or not schedule.retention_law.independent_output_rows
            or not schedule.retention_law.no_intermediate_observers
        ):
            raise ValueError("spatial grouping needs original independent-row and observer permissions")
        planner = (
            plan_rolling_convolution_row_fragments
            if type(output_group) is ConvolutionRowFragmentGroup
            else plan_rolling_convolution_output_group
        )
        expected = planner(
            schedule, rhs, input_rows, max_row_blocks=output_group.row_blocks, max_output_rows=output_group.output_rows
        )
        if expected is None or output_group != expected:
            raise ValueError("spatial output group differs from original rolling/RHS ownership")
        if (
            array_law.hardware_identity != schedule.epoch_law.hardware_identity
            or array_law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
        ):
            raise ValueError("spatial array retention selects a different original contract")
        for block in blocks:
            _validate_convolution_load_issue(
                p,
                builder,
                lower,
                complete,
                p.row_tile,
                schedule.issue_law,
                output_group.row_blocks * len(block) * int(p.bias),
            )
            if output_group.row_blocks * len(block) > readout_law.max_issued_readouts:
                raise ValueError("spatial group exceeds selected original readout work budget")
            for m in range(output_group.row_blocks):
                tiles = tuple(
                    ConvolutionOutputTile(
                        t.output_column,
                        t.columns,
                        p.accumulator_row + output_group.accumulator_row(m, n, active_column_blocks=len(block)),
                    )
                    for n, t in enumerate(block)
                )
                readout_law.require(p, p.row_tile, tiles)
                if readout_transfer_law is not None:
                    plan_convolution_readout_transfers(p, tiles, readout_transfer_law, rows=p.row_tile)
        group_identity, array_identity = _law_digest(output_group), _law_digest(array_law)
    return group_identity, array_identity


def _verify_grouped_transport(schedule, input_rows, output_group):
    from merlin.compiler.schedule.spatial.rolling_window import (
        materialize_grouped_rolling_window,
        verify_grouped_rolling_window,
    )

    if output_group is None or input_rows is None or schedule.issue_law is None or not schedule.choice.load_issue:
        raise ValueError("grouped transport needs checked spatial ownership and bounded target issue")
    ring = input_rows.rolling_plan()
    module = materialize_grouped_rolling_window(ring, output_rows=output_group.output_rows)
    report = verify_grouped_rolling_window(module, ring, output_rows=output_group.output_rows)
    commands = input_rows.stage_command_bound(report["max_staged_rows_per_epoch"])
    if commands > schedule.issue_law.load_issue_budget:
        raise ValueError("grouped transport exceeds selected actual transfer issue bound")
    return commands


# Preserve the original public lookup and pickle identities.
for _definition in (
    _verify_input_first_use,
    _verify_batch_input_prefetch,
    _verify_input_prefetch,
    _phase_blocks,
    _verify_readout_law,
    _verify_readout_transfers,
    _verify_spatial_output_group,
    _verify_grouped_transport,
):
    _definition.__module__ = "merlin.compiler.schedule.convolution.exploration"
del _definition
