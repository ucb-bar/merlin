"""Bounded source convolution choices through the ordinary pass registry.

This module supplies no target facts, numerical recipe, completion mechanism or
preferred schedule. Targets reopen original source, declare private resources
and effects, and realize the resulting counted schedule as actual program bytes."""

from __future__ import annotations

import copy as copy
from collections.abc import Callable as Callable
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from dataclasses import field as field

from merlin.compiler.schedule.affine_gather_runs import ConvolutionGatherRuns as ConvolutionGatherRuns
from merlin.compiler.schedule.convolution_choices import ConvolutionExplorationSource as ConvolutionExplorationSource
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsPhaseLaw as ConvolutionRhsPhaseLaw
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsPhases as ConvolutionRhsPhases
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsResidency as ConvolutionRhsResidency
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsResidencyLaw as ConvolutionRhsResidencyLaw
from merlin.compiler.schedule.convolution_choices import ConvolutionSchedule as ConvolutionSchedule
from merlin.compiler.schedule.convolution_choices import ConvolutionScheduleChoice as ConvolutionScheduleChoice
from merlin.compiler.schedule.convolution_choices import _column_blocks as _column_blocks
from merlin.compiler.schedule.convolution_choices import _digest as _digest
from merlin.compiler.schedule.convolution_choices import _law_digest as _law_digest
from merlin.compiler.schedule.convolution_choices import _phase_input as _phase_input
from merlin.compiler.schedule.convolution_choices import _rhs_capacity as _rhs_capacity
from merlin.compiler.schedule.convolution_choices import _rhs_intervals as _rhs_intervals
from merlin.compiler.schedule.convolution_choices import plan_convolution_rhs_phases as plan_convolution_rhs_phases
from merlin.compiler.schedule.convolution_choices import (
    plan_convolution_rhs_residency as plan_convolution_rhs_residency,
)
from merlin.compiler.schedule.convolution_choices import plan_convolution_schedule as plan_convolution_schedule
from merlin.compiler.schedule.window_transfer import ConvolutionComputeEpochLaw as ConvolutionComputeEpochLaw
from merlin.compiler.schedule.window_transfer import ConvolutionInputRetentionLaw as ConvolutionInputRetentionLaw
from merlin.compiler.schedule.window_transfer import ConvolutionLoadIssueLaw as ConvolutionLoadIssueLaw
from merlin.compiler.schedule.window_transfer import ConvolutionOutputTile as ConvolutionOutputTile
from merlin.compiler.schedule.window_transfer import ConvolutionReadoutEpochLaw as ConvolutionReadoutEpochLaw
from merlin.compiler.schedule.window_transfer import ConvolutionReadoutTransferLaw as ConvolutionReadoutTransferLaw
from merlin.compiler.schedule.window_transfer import ConvolutionResidentInputRows as ConvolutionResidentInputRows
from merlin.compiler.schedule.window_transfer import emit_convolution_column_block as emit_convolution_column_block
from merlin.compiler.schedule.window_transfer import (
    emit_convolution_compute_epoch_column_block as emit_convolution_compute_epoch_column_block,
)
from merlin.compiler.schedule.window_transfer import (
    emit_convolution_load_issue_column_block as emit_convolution_load_issue_column_block,
)
from merlin.compiler.schedule.window_transfer import (
    emit_convolution_load_issue_tile as emit_convolution_load_issue_tile,
)
from merlin.compiler.schedule.window_transfer import emit_convolution_tile as emit_convolution_tile
from merlin.compiler.schedule.window_transfer import (
    plan_convolution_readout_transfers as plan_convolution_readout_transfers,
)
from merlin.compiler.transforms.pass_exploration import EmittedProgram as EmittedProgram
from merlin.compiler.transforms.pass_exploration import PassFamily as PassFamily
from merlin.compiler.transforms.pass_exploration import explore_candidates as explore_candidates
from merlin.compiler.transforms.pass_exploration import generate_candidates as generate_candidates
from merlin.compiler.transforms.pipeline_policy import PolicyFacts as PolicyFacts
from merlin.compiler.transforms.pipeline_policy import RegisteredPass as RegisteredPass
from merlin.compiler.transforms.pipeline_state import PassSpec as PassSpec

from .convolution_admission import (
    _phase_blocks as _phase_blocks,
)
from .convolution_admission import (
    _verify_batch_input_prefetch as _verify_batch_input_prefetch,
)
from .convolution_admission import (
    _verify_grouped_transport as _verify_grouped_transport,
)
from .convolution_admission import (
    _verify_input_first_use as _verify_input_first_use,
)
from .convolution_admission import (
    _verify_input_prefetch as _verify_input_prefetch,
)
from .convolution_admission import (
    _verify_readout_law as _verify_readout_law,
)
from .convolution_admission import (
    _verify_readout_transfers as _verify_readout_transfers,
)
from .convolution_admission import (
    _verify_spatial_output_group as _verify_spatial_output_group,
)
from .convolution_candidates import (
    convolution_pass_space as convolution_pass_space,
)
from .convolution_candidates import (
    explore_convolution_schedules as explore_convolution_schedules,
)
from .convolution_realization import _emit_convolution_schedule as _emit_convolution_schedule
from .convolution_rhs import (
    _prepare_convolution_rhs as _prepare_convolution_rhs,
)
from .convolution_rhs import (
    emit_convolution_rhs_preparation as emit_convolution_rhs_preparation,
)
from .convolution_storage import ConvolutionStorageAlternative as ConvolutionStorageAlternative
from .convolution_storage import convolution_storage_pass_space as convolution_storage_pass_space
from .convolution_storage import plan_convolution_storage_alternative as plan_convolution_storage_alternative


def emit_convolution_schedule(
    schedule,
    builder,
    lower,
    *,
    complete=None,
    synchronous_completion=False,
    rhs=None,
    spatial_iteration="rows",
    input_rows=None,
    readout_law=None,
    readout_transfer_law=None,
    output_group=None,
    array_law=None,
    ready_readout=False,
    grouped_transport=False,
    rhs_phase_prefetch_law=None,
    input_prefetch_law=None,
    batch_input_prefetch_law=None,
    input_first_use_law=None,
    invariant_input_law=None,
):
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
    ``input_prefetch_law`` instead selects the existing counted K-run consumer
    with future input strips in separately owned ring slots. Both the complete
    two-group source horizon and selected continuation effects must be admitted.
    These two prefetch modes are separate alternatives; neither is a default.
    ``batch_input_prefetch_law`` selects a complete-image consumer whose source
    proves every input row's final K run. Next-image rows replace those dead
    spans under separately admitted target WAR/RAW effects; all drains remain.
    ``input_first_use_law`` instead owns pending full-image input: original DMA
    precedes each row's first K/N/M reader under actual target RAW ordering.
    It grants no input-ready token; the real final consumer drain closes DMA.
    ``invariant_input_law`` instead retains a completed full-image input union
    across original N phases. Every original group/input-last-use/phase drain
    remains; only redundant input transport in later phases is omitted.
    It may compose with the independent RHS dead-prefix frontier when every
    actual future RHS write remains disjoint from the retained input union.
    """
    if type(ready_readout) is not bool or ready_readout and output_group is None:
        raise ValueError("ready readout requires an explicit Boolean and checked spatial group")
    if type(grouped_transport) is not bool:
        raise ValueError("grouped transport requires an explicit Boolean")
    if input_prefetch_law is not None and rhs_phase_prefetch_law is not None:
        raise ValueError("input and RHS prefetch require separate selected alternatives")
    if input_prefetch_law is not None and batch_input_prefetch_law is not None:
        raise ValueError("spatial and batch input frontiers are separately selected alternatives")
    if input_first_use_law is not None and (batch_input_prefetch_law is not None or input_prefetch_law is not None):
        raise ValueError("first-use pending input requires a separate input transport selection")
    if invariant_input_law is not None and (
        type(rhs) is not ConvolutionRhsPhases
        or not grouped_transport
        or any(law is not None for law in (input_first_use_law, input_prefetch_law, batch_input_prefetch_law))
    ):
        raise ValueError("invariant input reuse needs a separate complete grouped phase traversal")
    if rhs_phase_prefetch_law is not None:
        from merlin.compiler.schedule.rolling_lookahead import RollingRhsFrontierAdmission

        if (
            type(rhs_phase_prefetch_law) is not RollingRhsFrontierAdmission
            or type(rhs) is not ConvolutionRhsPhases
            or not grouped_transport
            or output_group is None
            or schedule.epoch_law is None
        ):
            raise ValueError("RHS frontier needs its separate target law, complete phase owner and grouped consumer")
        rhs_phase_prefetch_law.require()
        if (
            rhs_phase_prefetch_law.hardware_identity != schedule.epoch_law.hardware_identity
            or rhs_phase_prefetch_law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
        ):
            raise ValueError("RHS frontier differs from the selected original target/numerical scope")
    if type(rhs) is ConvolutionRhsPhases:
        rhs.verify_current()
        if (
            type(schedule) is not ConvolutionSchedule
            or rhs.schedule.identity != schedule.identity
            or not callable(complete)
            or rhs.input_rows != input_rows
        ):
            raise ValueError("RHS phase emission needs original input, source and real completion")
        identity = rhs.identity
        readout_identity = _law_digest(readout_law)
        transfer_identity = _law_digest(readout_transfer_law)
        if output_group is None:
            if array_law is not None:
                raise ValueError("phased array retention needs its complete original group tuple")
            groups = (None,) * len(rhs.phases)
        elif (
            type(output_group) is not tuple
            or len(output_group) != len(rhs.phases)
            or not any(group is not None for group in output_group)
        ):
            raise ValueError("phased spatial groups must align with every original RHS phase")
        else:
            groups = output_group
        for phase, group in zip(rhs.phases, groups, strict=True):
            _verify_readout_law(schedule, readout_law, _phase_blocks(schedule, phase.column_intervals))
            _verify_readout_transfers(
                schedule, readout_transfer_law, readout_law, _phase_blocks(schedule, phase.column_intervals)
            )
            _verify_spatial_output_group(
                schedule,
                builder,
                lower,
                complete,
                rhs=phase,
                input_rows=input_rows,
                spatial_iteration=spatial_iteration,
                readout_law=readout_law,
                readout_transfer_law=readout_transfer_law,
                output_group=group,
                array_law=array_law if group is not None else None,
                blocks=_phase_blocks(schedule, phase.column_intervals),
            )
            if grouped_transport and group is not None:
                _verify_grouped_transport(schedule, input_rows, group)
            _verify_input_prefetch(
                schedule,
                input_rows,
                group,
                _phase_blocks(schedule, phase.column_intervals),
                grouped_transport,
                input_prefetch_law,
            )
            _verify_input_first_use(
                schedule,
                input_rows,
                group,
                _phase_blocks(schedule, phase.column_intervals),
                grouped_transport,
                input_first_use_law,
            )
            _verify_batch_input_prefetch(
                schedule,
                input_rows,
                group,
                _phase_blocks(schedule, phase.column_intervals),
                grouped_transport,
                batch_input_prefetch_law,
            )
        group_identity = tuple(_law_digest(group) for group in groups)
        array_identity = _law_digest(array_law)
        frontier_identity = _law_digest(rhs_phase_prefetch_law)
        input_prefetch_identity = _law_digest(input_prefetch_law)
        first_use_identity = _law_digest(input_first_use_law)
        batch_prefetch_identity = _law_digest(batch_input_prefetch_law)
        input_reuse = None
        if invariant_input_law is not None:
            from merlin.compiler.schedule.convolution_input_reuse import InvariantConvolutionInput

            input_reuse = InvariantConvolutionInput(schedule, rhs, input_rows, groups, invariant_input_law)
        frontiers = []
        for index, (phase, group) in enumerate(zip(rhs.phases, groups, strict=True)):
            frontier = None
            if rhs_phase_prefetch_law is not None:
                from merlin.compiler.schedule.rolling_lookahead import RollingRhsFrontier

                if group is None:
                    raise ValueError("RHS frontier requires every phase's checked original consumer")
                if index + 1 < len(rhs.phases):
                    next_phase = rhs.phases[index + 1]
                    frontier = RollingRhsFrontier(
                        schedule.source.identity,
                        phase.layout.rhs_base,
                        schedule.source.plan.channel_tile,
                        schedule.source.plan.output_channels,
                        len(phase.column_intervals),
                        next_phase.column_intervals,
                    )
                    frontier.require(input_rows.rolling_plan(), phase.layout.k_tiles)
                if input_reuse is not None:
                    input_reuse.check_rhs_frontier(index, frontier, rhs_phase_prefetch_law)
                    if len(_phase_blocks(schedule, phase.column_intervals)) != 1:
                        raise ValueError("retained input RHS frontier requires one complete phase column consumer")
            frontiers.append(frontier)
        results = []
        for index, (phase, group, frontier) in enumerate(zip(rhs.phases, groups, frontiers, strict=True)):
            results.append(
                _emit_convolution_schedule(
                    schedule,
                    builder,
                    lower,
                    complete=complete,
                    synchronous_completion=synchronous_completion,
                    rhs=phase,
                    spatial_iteration=spatial_iteration,
                    input_rows=input_rows,
                    readout_law=readout_law,
                    readout_transfer_law=readout_transfer_law,
                    output_group=group,
                    array_law=array_law if group is not None else None,
                    ready_readout=ready_readout and group is not None,
                    grouped_transport=grouped_transport and group is not None,
                    rhs_frontier=frontier,
                    rhs_frontier_law=rhs_phase_prefetch_law,
                    rhs_prefetched=rhs_phase_prefetch_law is not None and index > 0,
                    input_prefetch_law=input_prefetch_law,
                    batch_input_prefetch_law=batch_input_prefetch_law,
                    input_first_use_law=input_first_use_law,
                    reused_input=input_reuse if index > 0 else None,
                )
            )
            phase_completion = dict(phase=index, local_row=phase.layout.rhs_base, local_rows=phase.layout.rhs_rows)
            if input_reuse is None:
                complete("rhs_phase_last_use", phase_completion)
            else:
                input_reuse.complete_phase(index, complete, phase_completion)
        rhs.verify_current()
        if (
            rhs.identity != identity
            or _law_digest(readout_law) != readout_identity
            or _law_digest(readout_transfer_law) != transfer_identity
            or tuple(_law_digest(group) for group in groups) != group_identity
            or _law_digest(array_law) != array_identity
            or _law_digest(rhs_phase_prefetch_law) != frontier_identity
            or _law_digest(input_prefetch_law) != input_prefetch_identity
            or _law_digest(input_first_use_law) != first_use_identity
            or _law_digest(batch_input_prefetch_law) != batch_prefetch_identity
        ):
            raise ValueError("original RHS phases changed during realization")
        phase_report = rhs.summary()
        if input_reuse is not None:
            phase_report = dict(phase_report, invariant_input=input_reuse.summary())
        if rhs_phase_prefetch_law is not None:
            rhs_phase_prefetch_law.require()
            phase_report = dict(
                phase_report,
                frontier_admission=asdict(rhs_phase_prefetch_law),
                completion="dead K spans replaced after selected WAR dependencies; all original group, preparation and phase drains retained",
            )
        return dict(schedule.summary(), rhs_phases=phase_report, phase_emissions=results, realization_sha256=identity)
    if rhs is not None:
        if type(rhs) is not ConvolutionRhsResidency:
            raise ValueError("convolution RHS needs its original checked placement")
        if rhs.column_intervals is not None:
            raise ValueError("partial RHS placement needs its complete original phase owner")
    return _emit_convolution_schedule(
        schedule,
        builder,
        lower,
        complete=complete,
        synchronous_completion=synchronous_completion,
        rhs=rhs,
        spatial_iteration=spatial_iteration,
        input_rows=input_rows,
        readout_law=readout_law,
        readout_transfer_law=readout_transfer_law,
        output_group=output_group,
        array_law=array_law,
        ready_readout=ready_readout,
        grouped_transport=grouped_transport,
        input_prefetch_law=input_prefetch_law,
        batch_input_prefetch_law=batch_input_prefetch_law,
        input_first_use_law=input_first_use_law,
    )
