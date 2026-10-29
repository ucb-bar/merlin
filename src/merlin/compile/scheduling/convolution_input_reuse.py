"""Retain a completed, immutable full input union across original N phases.

The first phase uses the ordinary counted rolling transport. Only its real
full phase completion establishes a reusable input view. Later phases issue
the same original products/readouts against that view; all group, input-last
and phase drains remain. No pending transfer is called ready and no target
completion callback is suppressed.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConvolutionInputReuseAdmission:
    hardware_identity: str
    numerical_policy_identity: str
    source_identity: str
    qualification_identity: str
    immutable_source_input: bool = False
    exclusive_private_input_rows: bool = False
    other_effects_preserve_input_rows: bool = False
    no_intermediate_observers: bool = False
    real_preparation_and_phase_completion: bool = False

    def require(self):
        values = asdict(self)
        if any(type(v) is not str or not v.strip() for k, v in values.items() if k.endswith("identity")) or any(
            type(v) is not bool or not v for k, v in values.items() if not k.endswith("identity")
        ):
            raise ValueError("invariant input reuse needs original immutable storage/effect and real completion admission")


def select_convolution_input_reuse(schedule, rhs, rows, groups, *, admission):
    """Select the existing reuse consumer only for its exact source domain.

    A complete admission is still supplied by the target/input owner. Ordinary
    valid schedules outside the full-image phased domain retain their current
    consumer. Malformed selections and stale sources do not become fallbacks.
    No model name, shape list, performance observation or target size is used.
    """
    from .convolution_exploration import ConvolutionRhsPhases
    from .convolution_output_groups import ConvolutionOutputGroup
    from .window_transfer import ConvolutionResidentInputRows

    schedule.verify_current()
    if type(admission) is not ConvolutionInputReuseAdmission or schedule.epoch_law is None:
        raise ValueError("input reuse selection requires its original target and numerical admission")
    admission.require()
    if (admission.hardware_identity != schedule.epoch_law.hardware_identity
            or admission.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
            or admission.source_identity != schedule.source.identity):
        raise ValueError("input reuse selection differs from the original source/target/numerics")
    if type(rhs) is not ConvolutionRhsPhases or rows is None or groups is None:
        return None
    rhs.verify_current()
    if type(rows) is not ConvolutionResidentInputRows or type(groups) is not tuple or len(groups) != len(rhs.phases):
        raise ValueError("input reuse selection requires exact aligned original row and phase owners")
    rows.revalidate()
    if (rhs.schedule.identity != schedule.identity or rhs.input_rows != rows
            or rows.source_identity != schedule.source.identity or rows.window != schedule.source.plan.window):
        raise ValueError("input reuse selection changed its original source or phase input owner")
    p = schedule.source.plan
    w = p.window
    oh, ow = w.output_spatial
    span = (oh - 1) * w.stride[0] + (w.kernel[0] - 1) * w.dilation[0] + 1
    if (len(rhs.phases) < 2 or w.input_shape[0] != 1 or getattr(rows, "image_windows", 1) != 1
            or span > rows.slots or any(group is None for group in groups)):
        return None
    if any(type(group) is not ConvolutionOutputGroup or group.output_rows != oh
           or group.row_blocks != oh * ((ow + p.row_tile - 1) // p.row_tile) for group in groups):
        return None
    InvariantConvolutionInput(schedule, rhs, rows, groups, admission)
    return admission


class InvariantConvolutionInput:
    """One invocation's exact input view, established by its first full phase."""

    def __init__(self, schedule, rhs, rows, groups, admission):
        from .convolution_exploration import ConvolutionRhsPhases
        from .convolution_output_groups import ConvolutionOutputGroup
        from .window_transfer import ConvolutionResidentInputRows

        schedule.verify_current()
        if (type(rhs) is not ConvolutionRhsPhases or type(rows) is not ConvolutionResidentInputRows
                or type(groups) is not tuple or type(admission) is not ConvolutionInputReuseAdmission
                or schedule.epoch_law is None):
            raise ValueError("invariant input reuse requires original phased source, complete groups and selected admission")
        rhs.verify_current()
        rows.revalidate()
        admission.require()
        p = schedule.source.plan
        w = p.window
        oh, ow = w.output_spatial
        span = (oh - 1) * w.stride[0] + (w.kernel[0] - 1) * w.dilation[0] + 1
        per_row = (ow + p.row_tile - 1) // p.row_tile
        if (rhs.schedule.identity != schedule.identity or rhs.input_rows != rows or len(rhs.phases) < 2
                or admission.hardware_identity != schedule.epoch_law.hardware_identity
                or admission.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
                or admission.source_identity != schedule.source.identity
                or rows.source_identity != schedule.source.identity or rows.window != w
                or w.input_shape[0] != 1 or getattr(rows, "image_windows", 1) != 1
                or span > rows.slots or len(groups) != len(rhs.phases)):
            raise ValueError("input reuse requires one complete injective original image under the same phase/source map")
        for phase, group in zip(rhs.phases, groups, strict=True):
            if (type(group) is not ConvolutionOutputGroup or group.output_rows != oh
                    or group.row_blocks != oh * per_row
                    or not rows.disjoint_from(phase.layout.rhs_base, phase.layout.rhs_rows)):
                raise ValueError("every input reuse phase must cover the same complete group with disjoint RHS writes")
            group.__post_init__()
        self.schedule, self.rhs, self.rows, self.groups, self.admission = schedule, rhs, rows, groups, admission
        self.identity = schedule.identity, rhs.identity, rows.record(), tuple(map(asdict, groups)), asdict(admission)
        self.next_phase = 0
        self.source_row_span = span

    def check(self):
        self.schedule.verify_current()
        self.rhs.verify_current()
        self.rows.revalidate()
        self.admission.require()
        if self.identity != (self.schedule.identity, self.rhs.identity, self.rows.record(),
                             tuple(map(asdict, self.groups)), asdict(self.admission)):
            raise ValueError("invariant input source, physical view or admitted preserving effects changed")

    def emit_reused_group(self, phase, group, builder, issue, complete, *, blocks, array_law,
                          readout_law, readout_transfer_law, ready_readout,
                          rhs_frontier=None, rhs_frontier_law=None, rhs_issue=None):
        from .convolution_exploration import _phase_blocks
        from .convolution_output_groups import emit_rolling_convolution_output_group

        self.check()
        if (not 0 < self.next_phase < len(self.rhs.phases) or phase != self.rhs.phases[self.next_phase]
                or group != self.groups[self.next_phase] or not callable(complete)
                or blocks != _phase_blocks(self.schedule, phase.column_intervals)):
            raise ValueError("input reuse view is unavailable before the preceding real complete phase")
        p = self.schedule.source.plan
        oh, ow = p.window.output_spatial
        tiles = tuple((y * ow + x, min(p.row_tile, ow - x))
                      for y in range(oh) for x in range(0, ow, p.row_tile))
        self.check_rhs_frontier(self.next_phase, rhs_frontier, rhs_frontier_law)
        if rhs_frontier is not None and (len(blocks) != 1 or not callable(rhs_issue)):
            raise ValueError("reused input RHS frontier needs one complete column consumer and its real transfer emitter")
        for block in blocks:
            consumer = emit_rolling_convolution_output_group(self.schedule, group, builder, issue, complete,
                output_row=0, row_counts=tuple(t[1] for t in tiles), tile_origins=tuple(t[0] for t in tiles),
                output_tiles=block, input_rows=self.rows, array_law=array_law, readout_law=readout_law,
                readout_transfer_law=readout_transfer_law, ready_readout=ready_readout,
                defer_consumer=rhs_frontier is not None)
            if rhs_frontier is not None:
                def after_run(run, reduction_row, depth):
                    # Each callback follows the complete original K/N/M run.
                    # The target's admitted WAR/RAW dependencies protect these
                    # actual writes; the original group/phase drains complete
                    # them before the next phase can consume the same rows.
                    for n, (column, columns) in enumerate(rhs_frontier.next_columns):
                        local = builder.op("add", rhs_frontier.rhs_base, builder.op("mul",
                            builder.op("add", builder.op("mul", run, len(rhs_frontier.next_columns)), n),
                            rhs_frontier.tile))
                        rhs_issue("load_weight", dict(reduction_row=reduction_row, output_column=column,
                            local_row=local, rows=depth, columns=columns))
                consumer.advance(0, consumer.runs, after_run=after_run)
                consumer.finish()
        complete("input_last_use", dict(local_row=self.rows.base))
        self.check()
        return dict(schema="merlin.invariant_convolution_input_phase.v1", phase=self.next_phase,
            source_row_union=self.source_row_span, slots=self.rows.slots, dynamic_stage_rows=0,
            dynamic_stage_bytes=0, prepared_by="ordinary first-phase counted transport and real phase completion",
            mapping="same immutable source cells and injective physical rows across all original N phases",
            rhs_dead_prefix_transfers=(0 if rhs_frontier is None else
                p.window.kernel[0] * p.window.kernel[1] * self.rows.channel_blocks * len(rhs_frontier.next_columns)),
            completion="unchanged real group, input-last-use and phase drains; retained input has no pending transport")

    def check_rhs_frontier(self, index, frontier, admission):
        """Reopen the independent future-RHS map before any target emission.

        Input reuse admits only preserving effects. The separate frontier law
        owns target WAR/RAW, configuration, queue progress and real completion;
        its entire assigned write span must stay outside the retained input.
        """
        from .rolling_lookahead import RollingRhsFrontier, RollingRhsFrontierAdmission

        self.check()
        if frontier is None and admission is None:
            return
        if (type(index) is not int or not 0 <= index < len(self.rhs.phases)
                or type(admission) is not RollingRhsFrontierAdmission):
            raise ValueError("retained input and RHS frontier need independent original target admission")
        admission.require()
        epoch = self.schedule.epoch_law
        if (admission.hardware_identity != epoch.hardware_identity
                or admission.numerical_policy_identity != epoch.numerical_policy_identity):
            raise ValueError("retained input RHS frontier changed the original target or numerical scope")
        if index + 1 == len(self.rhs.phases):
            if frontier is not None:
                raise ValueError("final original phase has no future RHS source")
            return
        phase, following = self.rhs.phases[index:index + 2]
        p = self.schedule.source.plan
        runs = p.window.kernel[0] * p.window.kernel[1] * self.rows.channel_blocks
        if (type(frontier) is not RollingRhsFrontier or frontier.rhs_base != phase.layout.rhs_base
                or frontier.current_column_blocks != len(phase.column_intervals)
                or frontier.next_columns != following.column_intervals
                or following.layout.rhs_base != phase.layout.rhs_base
                or not self.rows.disjoint_from(frontier.rhs_base,
                    runs * len(frontier.next_columns) * frontier.tile)):
            raise ValueError("future RHS writes change original phase coverage or overlap the retained input")
        frontier.require(self.rows.rolling_plan(), runs)

    def complete_phase(self, index, complete, fields):
        self.check()
        if type(index) is not int or index != self.next_phase or not callable(complete):
            raise ValueError("input reuse requires each original phase's real completion in source order")
        phase = self.rhs.phases[index]
        if fields != dict(phase=index, local_row=phase.layout.rhs_base, local_rows=phase.layout.rhs_rows):
            raise ValueError("input reuse phase completion changed its original private RHS allocation")
        complete("rhs_phase_last_use", fields)
        self.next_phase += 1
        self.check()

    def summary(self):
        self.check()
        if self.next_phase != len(self.rhs.phases):
            raise ValueError("invariant input lifetime has not reached actual final phase completion")
        return dict(schema="merlin.invariant_convolution_input.v1", admission=asdict(self.admission),
            source_row_union=self.source_row_span, original_input_traversals=len(self.rhs.phases),
            prepared_input_traversals=1, reused_input_traversals=len(self.rhs.phases)-1,
            completion="ordinary first preparation and every original real group/input-last-use/phase drain retained")
