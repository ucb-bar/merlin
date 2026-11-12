"""Enumerate bounded original convolution storage and registered pass choices."""

from __future__ import annotations

import copy
from collections.abc import Callable

from merlin.compiler.schedule.convolution_choices import ConvolutionExplorationSource as ConvolutionExplorationSource
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsPhaseLaw as ConvolutionRhsPhaseLaw
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsPhases as ConvolutionRhsPhases
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsResidency as ConvolutionRhsResidency
from merlin.compiler.schedule.convolution_choices import ConvolutionRhsResidencyLaw as ConvolutionRhsResidencyLaw
from merlin.compiler.schedule.convolution_choices import ConvolutionSchedule as ConvolutionSchedule
from merlin.compiler.schedule.convolution_choices import ConvolutionScheduleChoice as ConvolutionScheduleChoice
from merlin.compiler.schedule.convolution_choices import _digest as _digest
from merlin.compiler.schedule.convolution_choices import _law_digest as _law_digest
from merlin.compiler.schedule.convolution_choices import _phase_input as _phase_input
from merlin.compiler.schedule.convolution_choices import plan_convolution_rhs_phases as plan_convolution_rhs_phases
from merlin.compiler.schedule.convolution_choices import (
    plan_convolution_rhs_residency as plan_convolution_rhs_residency,
)
from merlin.compiler.schedule.convolution_choices import plan_convolution_schedule as plan_convolution_schedule
from merlin.compiler.schedule.window_transfer import (
    ConvolutionComputeEpochLaw,
    ConvolutionInputRetentionLaw,
    ConvolutionLoadIssueLaw,
    ConvolutionResidentInputRows,
)
from merlin.compiler.transforms.pass_exploration import (
    EmittedProgram,
    PassFamily,
    explore_candidates,
    generate_candidates,
)
from merlin.compiler.transforms.pipeline_policy import PolicyFacts, RegisteredPass
from merlin.compiler.transforms.pipeline_state import PassSpec


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
    _definition.__module__ = "merlin.compiler.schedule.convolution_exploration"
del _definition
