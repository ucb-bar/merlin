"""Enumerate bounded original convolution storage and registered pass choices."""

from __future__ import annotations

import copy
import copy as copy
from collections.abc import Callable
from collections.abc import Callable as Callable
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from dataclasses import field as field

from merlin.compiler.schedule.convolution.choices import (
    ConvolutionExplorationSource,
    ConvolutionRhsPhaseLaw,
    ConvolutionRhsPhases,
    ConvolutionRhsResidency,
    ConvolutionRhsResidencyLaw,
    ConvolutionSchedule,
    ConvolutionScheduleChoice,
    _column_blocks,
    _digest,
    _law_digest,
    _phase_input,
    _prepare_convolution_rhs,
    _rhs_capacity,
    _rhs_intervals,
    emit_convolution_rhs_preparation,
    plan_convolution_rhs_phases,
    plan_convolution_rhs_residency,
    plan_convolution_schedule,
)
from merlin.compiler.schedule.convolution.input_rows import ConvolutionGatherRuns
from merlin.compiler.schedule.convolution.realization import _emit_convolution_schedule
from merlin.compiler.schedule.convolution.storage import (
    ConvolutionStorageAlternative,
    _phase_blocks,
    _verify_batch_input_prefetch,
    _verify_grouped_transport,
    _verify_input_first_use,
    _verify_input_prefetch,
    _verify_readout_law,
    _verify_readout_transfers,
    _verify_spatial_output_group,
    convolution_storage_pass_space,
    plan_convolution_storage_alternative,
)
from merlin.compiler.schedule.convolution.windows import (
    ConvolutionComputeEpochLaw,
    ConvolutionInputRetentionLaw,
    ConvolutionLoadIssueLaw,
    ConvolutionOutputTile,
    ConvolutionReadoutEpochLaw,
    ConvolutionReadoutTransferLaw,
    ConvolutionResidentInputRows,
    emit_convolution_column_block,
    emit_convolution_compute_epoch_column_block,
    emit_convolution_load_issue_column_block,
    emit_convolution_load_issue_tile,
    emit_convolution_tile,
    plan_convolution_readout_transfers,
)
from merlin.compiler.transforms.exploration.branch_policy import PolicyFacts, RegisteredPass
from merlin.compiler.transforms.exploration.pass_orders import (
    EmittedProgram,
    PassFamily,
    explore_candidates,
    generate_candidates,
)
from merlin.compiler.transforms.pipeline_state import PassSpec
from merlin.compiler.transforms.pipeline_state import PassSpec as PassSpec

# ---- convolution_candidates --------------------------------------------------------------------


def convolution_pass_space(
    source,
    *,
    verify_source: Callable,
    realize: Callable,
    retention_law=None,
    issue_law=None,
    epoch_law=None,
    live_tile_limits=(1, None),
    issue_modes=(False, True),
    compute_pair_limits=(1,),
    rhs_modes=(False,),
    rhs_law=None,
    private_operand_rows=None,
    rhs_base=None,
    max_candidates=8,
    rhs_phase_law=None,
    input_rows=None,
):
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
    if (
        type(source) is not ConvolutionExplorationSource
        or not callable(verify_source)
        or not callable(realize)
        or type(live_tile_limits) is not tuple
        or not live_tile_limits
        or len(live_tile_limits) > 8
        or len(set(live_tile_limits)) != len(live_tile_limits)
        or type(issue_modes) is not tuple
        or not issue_modes
        or len(set(issue_modes)) != len(issue_modes)
        or any(type(v) is not bool for v in issue_modes)
        or type(compute_pair_limits) is not tuple
        or not compute_pair_limits
        or len(compute_pair_limits) > 8
        or len(set(compute_pair_limits)) != len(compute_pair_limits)
        or any(type(v) is not int or not 0 < v < 2**31 for v in compute_pair_limits)
        or type(rhs_modes) is not tuple
        or not rhs_modes
        or len(set(rhs_modes)) != len(rhs_modes)
        or any(type(v) is not bool and v != "phased" for v in rhs_modes)
    ):
        raise ValueError("convolution registry requires bounded choices and original target callbacks")
    for limit in live_tile_limits:
        ConvolutionScheduleChoice(limit)
    for law, cls in (
        (retention_law, ConvolutionInputRetentionLaw),
        (issue_law, ConvolutionLoadIssueLaw),
        (epoch_law, ConvolutionComputeEpochLaw),
        (rhs_law, ConvolutionRhsResidencyLaw),
    ):
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
        return (
            _law_digest(retention_law),
            _law_digest(issue_law),
            _law_digest(epoch_law),
            _law_digest(rhs_law),
            _law_digest(rhs_phase_law),
            None if input_rows is None else _digest(input_rows.record()),
        )

    law_identities = law_ids()
    source_identity = source.identity
    level = "counted_convolution_source"

    def reopen(current):
        if (
            type(current) is not ConvolutionExplorationSource
            or current.identity != source_identity
            or law_identities != law_ids()
        ):
            raise ValueError("convolution registry source or target laws changed")
        if verify_source(current) is not None or current.identity != source_identity or law_identities != law_ids():
            raise ValueError("original convolution source callback did not validate unchanged input")

    def entry(name, stage, requires, produces, action):
        return RegisteredPass(
            PassSpec(name, stage, requires=requires, produces=produces, mutates_ir=False, ir_level=level), action
        )

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
            value = plan_convolution_schedule(current, ConvolutionScheduleChoice(limit), retention_law=retention_law)
            return {"convolution_columns": value}

        return entry("allocate_column_lifetimes", 1, ("convolution_source",), ("convolution_columns",), action)

    def issue(options):
        def action(session):
            previous = session.analysis("convolution_columns")
            previous.verify_current()
            value = plan_convolution_schedule(
                previous.source,
                ConvolutionScheduleChoice(previous.choice.live_tiles, options["enabled"]),
                retention_law=retention_law,
                issue_law=issue_law,
            )
            return {"convolution_schedule": value}

        return entry("select_bounded_load_epochs", 2, ("convolution_columns",), ("convolution_schedule",), action)

    def epochs(options):
        def action(session):
            previous = session.analysis("convolution_schedule")
            previous.verify_current()
            value = plan_convolution_schedule(
                previous.source,
                ConvolutionScheduleChoice(previous.choice.live_tiles, previous.choice.load_issue, options["pairs"]),
                retention_law=retention_law,
                issue_law=issue_law,
                epoch_law=epoch_law,
            )
            return {"convolution_epoch_schedule": value}

        return entry(
            "group_nested_compute_epochs", 3, ("convolution_schedule",), ("convolution_epoch_schedule",), action
        )

    extended = compute_pair_limits != (1,)
    schedule_analysis = "convolution_epoch_schedule" if extended else "convolution_schedule"
    prepared = rhs_modes != (False,)

    def residency(options):
        def action(session):
            schedule = session.analysis(schedule_analysis)
            schedule.verify_current()
            if options["enabled"] == "phased":
                rhs = plan_convolution_rhs_phases(
                    schedule,
                    operand_rows=private_operand_rows,
                    law=rhs_law,
                    phase_law=rhs_phase_law,
                    rhs_base=rhs_base,
                    input_rows=input_rows,
                )
            else:
                rhs = (
                    plan_convolution_rhs_residency(
                        schedule, operand_rows=private_operand_rows, law=rhs_law, rhs_base=rhs_base
                    )
                    if options["enabled"]
                    else None
                )
            return {"convolution_preparation": (schedule, rhs)}

        return entry(
            "prepare_invocation_rhs", 4 if extended else 3, (schedule_analysis,), ("convolution_preparation",), action
        )

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
            if (
                type(result) is not EmittedProgram
                or result.schedule_sha256 != before
                or (rhs.identity if rhs is not None else schedule.identity) != before
            ):
                raise ValueError("target program is not bound to the original realized convolution schedule")
            return {"convolution_program": result}

        return entry(
            "realize_original_convolution",
            3 + int(extended) + int(prepared),
            ("convolution_preparation" if prepared else schedule_analysis,),
            ("convolution_program",),
            action,
        )

    families = (
        PassFamily("source", original),
        PassFamily(
            "columns", columns, knobs=(("live_tiles", tuple("capacity" if v is None else v for v in live_tile_limits)),)
        ),
        PassFamily("issue", issue, knobs=(("enabled", issue_modes),)),
    )
    if extended:
        families += (PassFamily("compute_epochs", epochs, knobs=(("pairs", compute_pair_limits),)),)
    if prepared:
        families += (PassFamily("rhs_preparation", residency, knobs=(("enabled", rhs_modes),)),)
    return generate_candidates(
        (*families, PassFamily("emit", emit)), output="convolution_program", max_candidates=max_candidates
    )


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
        return PolicyFacts(
            session.stamp(),
            shapes=(p.window.input_shape, (*p.window.kernel, p.window.input_shape[-1], p.output_channels)),
            operations=frozenset({"convolution"}),
            resources=(("operand_rows", p.operand_rows), ("accumulator_rows", p.accumulator_rows)),
        )

    return explore_candidates(
        source,
        space,
        facts,
        output="convolution_program",
        emission=lambda result: result.artifact,
        feedback=feedback,
        context=context,
        ir_level="counted_convolution_source",
        clone=copy.deepcopy,
        fingerprint=lambda current: current.identity,
        verify_ir=verify,
    )


# Preserve the original public lookup and pickle identities.
for _definition in (convolution_pass_space, explore_convolution_schedules):
    _definition.__module__ = "merlin.compiler.schedule.convolution.exploration"
del _definition


# ---- convolution_exploration -------------------------------------------------------------------
# Bounded source convolution choices through the ordinary pass registry.
#
# This module supplies no target facts, numerical recipe, completion mechanism or
# preferred schedule. Targets reopen original source, declare private resources
# and effects, and realize the resulting counted schedule as actual program bytes.


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
        from merlin.compiler.schedule.spatial.rolling_lookahead import RollingRhsFrontierAdmission

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
            from merlin.compiler.schedule.convolution.input_residency import InvariantConvolutionInput

            input_reuse = InvariantConvolutionInput(schedule, rhs, input_rows, groups, invariant_input_law)
        frontiers = []
        for index, (phase, group) in enumerate(zip(rhs.phases, groups, strict=True)):
            frontier = None
            if rhs_phase_prefetch_law is not None:
                from merlin.compiler.schedule.spatial.rolling_lookahead import RollingRhsFrontier

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
