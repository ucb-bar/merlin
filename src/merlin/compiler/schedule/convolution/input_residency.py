"""Retain a completed, immutable full input union across original N phases.

The first phase uses the ordinary counted rolling transport. Only its real
full phase completion establishes a reusable input view. Later phases issue
the same original products/readouts against that view; all group, input-last
and phase drains remain. No pending transfer is called ready and no target
completion callback is suppressed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# ---- convolution_input_reuse -------------------------------------------------------------------


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
            raise ValueError(
                "invariant input reuse needs original immutable storage/effect and real completion admission"
            )


def select_convolution_input_reuse(schedule, rhs, rows, groups, *, admission):
    """Select the existing reuse consumer only for its exact source domain.

    A complete admission is still supplied by the target/input owner. Ordinary
    valid schedules outside the full-image phased domain retain their current
    consumer. Malformed selections and stale sources do not become fallbacks.
    No model name, shape list, performance observation or target size is used.
    """
    from merlin.compiler.schedule.convolution.exploration import ConvolutionRhsPhases
    from merlin.compiler.schedule.convolution.output_groups import ConvolutionOutputGroup
    from merlin.compiler.schedule.convolution.windows import ConvolutionResidentInputRows

    schedule.verify_current()
    if type(admission) is not ConvolutionInputReuseAdmission or schedule.epoch_law is None:
        raise ValueError("input reuse selection requires its original target and numerical admission")
    admission.require()
    if (
        admission.hardware_identity != schedule.epoch_law.hardware_identity
        or admission.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
        or admission.source_identity != schedule.source.identity
    ):
        raise ValueError("input reuse selection differs from the original source/target/numerics")
    if type(rhs) is not ConvolutionRhsPhases or rows is None or groups is None:
        return None
    rhs.verify_current()
    if type(rows) is not ConvolutionResidentInputRows or type(groups) is not tuple or len(groups) != len(rhs.phases):
        raise ValueError("input reuse selection requires exact aligned original row and phase owners")
    rows.revalidate()
    if (
        rhs.schedule.identity != schedule.identity
        or rhs.input_rows != rows
        or rows.source_identity != schedule.source.identity
        or rows.window != schedule.source.plan.window
    ):
        raise ValueError("input reuse selection changed its original source or phase input owner")
    p = schedule.source.plan
    w = p.window
    oh, ow = w.output_spatial
    span = (oh - 1) * w.stride[0] + (w.kernel[0] - 1) * w.dilation[0] + 1
    if (
        len(rhs.phases) < 2
        or w.input_shape[0] != 1
        or getattr(rows, "image_windows", 1) != 1
        or span > rows.slots
        or any(group is None for group in groups)
    ):
        return None
    if any(
        type(group) is not ConvolutionOutputGroup
        or group.output_rows != oh
        or group.row_blocks != oh * ((ow + p.row_tile - 1) // p.row_tile)
        for group in groups
    ):
        return None
    InvariantConvolutionInput(schedule, rhs, rows, groups, admission)
    return admission


class InvariantConvolutionInput:
    """One invocation's exact input view, established by its first full phase."""

    def __init__(self, schedule, rhs, rows, groups, admission):
        from merlin.compiler.schedule.convolution.exploration import ConvolutionRhsPhases
        from merlin.compiler.schedule.convolution.output_groups import ConvolutionOutputGroup
        from merlin.compiler.schedule.convolution.windows import ConvolutionResidentInputRows

        schedule.verify_current()
        if (
            type(rhs) is not ConvolutionRhsPhases
            or type(rows) is not ConvolutionResidentInputRows
            or type(groups) is not tuple
            or type(admission) is not ConvolutionInputReuseAdmission
            or schedule.epoch_law is None
        ):
            raise ValueError(
                "invariant input reuse requires original phased source, complete groups and selected admission"
            )
        rhs.verify_current()
        rows.revalidate()
        admission.require()
        p = schedule.source.plan
        w = p.window
        oh, ow = w.output_spatial
        span = (oh - 1) * w.stride[0] + (w.kernel[0] - 1) * w.dilation[0] + 1
        per_row = (ow + p.row_tile - 1) // p.row_tile
        if (
            rhs.schedule.identity != schedule.identity
            or rhs.input_rows != rows
            or len(rhs.phases) < 2
            or admission.hardware_identity != schedule.epoch_law.hardware_identity
            or admission.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
            or admission.source_identity != schedule.source.identity
            or rows.source_identity != schedule.source.identity
            or rows.window != w
            or w.input_shape[0] != 1
            or getattr(rows, "image_windows", 1) != 1
            or span > rows.slots
            or len(groups) != len(rhs.phases)
        ):
            raise ValueError(
                "input reuse requires one complete injective original image under the same phase/source map"
            )
        for phase, group in zip(rhs.phases, groups, strict=True):
            if (
                type(group) is not ConvolutionOutputGroup
                or group.output_rows != oh
                or group.row_blocks != oh * per_row
                or not rows.disjoint_from(phase.layout.rhs_base, phase.layout.rhs_rows)
            ):
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
        if self.identity != (
            self.schedule.identity,
            self.rhs.identity,
            self.rows.record(),
            tuple(map(asdict, self.groups)),
            asdict(self.admission),
        ):
            raise ValueError("invariant input source, physical view or admitted preserving effects changed")

    def emit_reused_group(
        self,
        phase,
        group,
        builder,
        issue,
        complete,
        *,
        blocks,
        array_law,
        readout_law,
        readout_transfer_law,
        ready_readout,
        rhs_frontier=None,
        rhs_frontier_law=None,
        rhs_issue=None,
    ):
        from merlin.compiler.schedule.convolution.exploration import _phase_blocks
        from merlin.compiler.schedule.convolution.output_groups import emit_rolling_convolution_output_group

        self.check()
        if (
            not 0 < self.next_phase < len(self.rhs.phases)
            or phase != self.rhs.phases[self.next_phase]
            or group != self.groups[self.next_phase]
            or not callable(complete)
            or blocks != _phase_blocks(self.schedule, phase.column_intervals)
        ):
            raise ValueError("input reuse view is unavailable before the preceding real complete phase")
        p = self.schedule.source.plan
        oh, ow = p.window.output_spatial
        tiles = tuple((y * ow + x, min(p.row_tile, ow - x)) for y in range(oh) for x in range(0, ow, p.row_tile))
        self.check_rhs_frontier(self.next_phase, rhs_frontier, rhs_frontier_law)
        if rhs_frontier is not None and (len(blocks) != 1 or not callable(rhs_issue)):
            raise ValueError(
                "reused input RHS frontier needs one complete column consumer and its real transfer emitter"
            )
        for block in blocks:
            consumer = emit_rolling_convolution_output_group(
                self.schedule,
                group,
                builder,
                issue,
                complete,
                output_row=0,
                row_counts=tuple(t[1] for t in tiles),
                tile_origins=tuple(t[0] for t in tiles),
                output_tiles=block,
                input_rows=self.rows,
                array_law=array_law,
                readout_law=readout_law,
                readout_transfer_law=readout_transfer_law,
                ready_readout=ready_readout,
                defer_consumer=rhs_frontier is not None,
            )
            if rhs_frontier is not None:

                def after_run(run, reduction_row, depth):
                    # Each callback follows the complete original K/N/M run.
                    # The target's admitted WAR/RAW dependencies protect these
                    # actual writes; the original group/phase drains complete
                    # them before the next phase can consume the same rows.
                    for n, (column, columns) in enumerate(rhs_frontier.next_columns):
                        local = builder.op(
                            "add",
                            rhs_frontier.rhs_base,
                            builder.op(
                                "mul",
                                builder.op("add", builder.op("mul", run, len(rhs_frontier.next_columns)), n),
                                rhs_frontier.tile,
                            ),
                        )
                        rhs_issue(
                            "load_weight",
                            dict(
                                reduction_row=reduction_row,
                                output_column=column,
                                local_row=local,
                                rows=depth,
                                columns=columns,
                            ),
                        )

                consumer.advance(0, consumer.runs, after_run=after_run)
                consumer.finish()
        complete("input_last_use", dict(local_row=self.rows.base))
        self.check()
        return dict(
            schema="merlin.invariant_convolution_input_phase.v1",
            phase=self.next_phase,
            source_row_union=self.source_row_span,
            slots=self.rows.slots,
            dynamic_stage_rows=0,
            dynamic_stage_bytes=0,
            prepared_by="ordinary first-phase counted transport and real phase completion",
            mapping="same immutable source cells and injective physical rows across all original N phases",
            rhs_dead_prefix_transfers=(
                0
                if rhs_frontier is None
                else p.window.kernel[0] * p.window.kernel[1] * self.rows.channel_blocks * len(rhs_frontier.next_columns)
            ),
            completion="unchanged real group, input-last-use and phase drains; retained input has no pending transport",
        )

    def check_rhs_frontier(self, index, frontier, admission):
        """Reopen the independent future-RHS map before any target emission.

        Input reuse admits only preserving effects. The separate frontier law
        owns target WAR/RAW, configuration, queue progress and real completion;
        its entire assigned write span must stay outside the retained input.
        """
        from merlin.compiler.schedule.spatial.rolling_lookahead import (
            RollingRhsFrontier,
            RollingRhsFrontierAdmission,
        )

        self.check()
        if frontier is None and admission is None:
            return
        if (
            type(index) is not int
            or not 0 <= index < len(self.rhs.phases)
            or type(admission) is not RollingRhsFrontierAdmission
        ):
            raise ValueError("retained input and RHS frontier need independent original target admission")
        admission.require()
        epoch = self.schedule.epoch_law
        if (
            admission.hardware_identity != epoch.hardware_identity
            or admission.numerical_policy_identity != epoch.numerical_policy_identity
        ):
            raise ValueError("retained input RHS frontier changed the original target or numerical scope")
        if index + 1 == len(self.rhs.phases):
            if frontier is not None:
                raise ValueError("final original phase has no future RHS source")
            return
        phase, following = self.rhs.phases[index : index + 2]
        p = self.schedule.source.plan
        runs = p.window.kernel[0] * p.window.kernel[1] * self.rows.channel_blocks
        if (
            type(frontier) is not RollingRhsFrontier
            or frontier.rhs_base != phase.layout.rhs_base
            or frontier.current_column_blocks != len(phase.column_intervals)
            or frontier.next_columns != following.column_intervals
            or following.layout.rhs_base != phase.layout.rhs_base
            or not self.rows.disjoint_from(frontier.rhs_base, runs * len(frontier.next_columns) * frontier.tile)
        ):
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
        return dict(
            schema="merlin.invariant_convolution_input.v1",
            admission=asdict(self.admission),
            source_row_union=self.source_row_span,
            original_input_traversals=len(self.rhs.phases),
            prepared_input_traversals=1,
            reused_input_traversals=len(self.rhs.phases) - 1,
            completion="ordinary first preparation and every original real group/input-last-use/phase drain retained",
        )


# ---- convolution_input_first_use ---------------------------------------------------------------
# Source-derived input DMA at the first actual K/N/M reader.
#
# A pending owner grants no transport-ready value. The selected target's real
# scratchpad RAW dependencies order each reader after its original row writes.
# Only the original full consumer drain closes the pending input lifetime.


@dataclass(frozen=True)
class ConvolutionInputLead:
    """Separately selected source/resource lead; never a completion promise.

    Service estimates choose issue positions only. Correctness continues to
    depend on the actual target RAW/progress/full-drain admission. Prefix mode
    preserves original row/channel transfer order. Bounded mode works backward
    from actual first readers at admitted beat bandwidth, startup and finite
    load/reader capacity; no private schedule positions or output data enter.
    """

    policy: str
    hardware_identity: str
    qualification_identity: str
    load_entries: int
    reader_entries: int
    request_bytes: int
    beat_bytes: int
    startup_cycles: int
    compute_cycles_per_row: int = 1
    bank_rows: int | None = None
    bank_count: int | None = None

    def require(self, hardware_identity):
        if (
            self.policy not in ("prefix", "bounded", "bank_advance")
            or self.hardware_identity != hardware_identity
            or any(type(v) is not str or not v.strip() for v in (self.hardware_identity, self.qualification_identity))
            or any(
                type(v) is not int or not 0 < v < 1 << 31
                for v in (
                    self.load_entries,
                    self.reader_entries,
                    self.request_bytes,
                    self.beat_bytes,
                    self.startup_cycles,
                    self.compute_cycles_per_row,
                )
            )
            or self.request_bytes % self.beat_bytes
        ):
            raise ValueError("input lead requires its selected original finite resources and service scope")
        if self.policy == "bank_advance" and any(
            type(v) is not int or not 0 < v < 1 << 31 for v in (self.bank_rows, self.bank_count)
        ):
            raise ValueError("bank advance requires its original selected row-bank geometry")

    def service(self, rows, columns, compute_rows):
        """Conservative packet alignment; estimates guide issue, never wait."""
        ceil = lambda a, b: (a + b - 1) // b
        # Each strided source row may cross an extra original request boundary.
        packets = rows * ceil(columns + self.request_bytes - 1, self.request_bytes)
        beats = ceil(rows * columns, self.beat_bytes)
        latency = beats + ceil(packets, self.reader_entries) * self.startup_cycles
        spacing = max(beats, ceil(latency, self.load_entries))
        pair_cycles = compute_rows * self.compute_cycles_per_row
        return ceil(latency, pair_cycles), ceil(spacing, pair_cycles)


@dataclass(frozen=True)
class ConvolutionInputFirstUseAdmission:
    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    scratchpad_raw_dependencies: bool = False
    ordered_configuration: bool = False
    finite_queue_progress: bool = False
    immutable_input: bool = False
    dma_preserves_stationary_array: bool = False
    full_consumer_transport_completion: bool = False
    lead: ConvolutionInputLead | None = None

    def require(self):
        values = asdict(self)
        if any(
            type(values[k]) is not str or not values[k].strip()
            for k in ("hardware_identity", "numerical_policy_identity", "qualification_identity")
        ) or any(
            type(values[k]) is not bool or not values[k] for k in values if not k.endswith("identity") and k != "lead"
        ):
            raise ValueError(
                "first-use input needs actual RAW, configuration, progress, array retention and full completion admission"
            )
        if self.lead is not None:
            if type(self.lead) is not ConvolutionInputLead:
                raise ValueError("input lead needs its typed selected source/resource policy")
            self.lead.require(self.hardware_identity)


class PendingConvolutionInput:
    """One original image's source rows, pending until a real consumer drain.

    First use is the lexicographically first original (KY, output Y), with
    KX=N=0 and the first CI of each existing channel transfer. Every original
    transfer is issued once, including unchanged stride or dilation holes. This initial domain retains one complete output row per M
    tile and one injective full-image ring; it allocates no additional storage.
    """

    def __init__(self, schedule, group, rows, admission, *, image):
        from merlin.compiler.schedule.convolution.output_groups import ConvolutionOutputGroup
        from merlin.compiler.schedule.convolution.windows import ConvolutionResidentInputRows

        schedule.verify_current()
        if (
            type(group) is not ConvolutionOutputGroup
            or type(rows) is not ConvolutionResidentInputRows
            or type(admission) is not ConvolutionInputFirstUseAdmission
            or schedule.epoch_law is None
        ):
            raise ValueError("pending input needs original source, complete group, ring and target law")
        group.__post_init__()
        rows.revalidate()
        admission.require()
        p = schedule.source.plan
        w = p.window
        ci = w.input_shape[-1]
        oh, ow = w.output_spatial
        live = (oh - 1) * w.stride[0] + (w.kernel[0] - 1) * w.dilation[0] + 1
        if (
            admission.hardware_identity != schedule.epoch_law.hardware_identity
            or admission.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
            or rows.source_identity != schedule.source.identity
            or rows.window != w
            or group.output_rows != oh
            or group.row_blocks != oh
            or ow > p.row_tile
            or getattr(rows, "image_windows", 1) != 1
            or live > rows.slots
            or not rows.disjoint_from(group.panel.rhs_base, group.panel.rhs_rows)
        ):
            raise ValueError(
                "pending input requires the original complete image and injective single ring under its selected target"
            )
        self.schedule, self.group, self.rows, self.admission = schedule, group, rows, admission
        self.image = image
        self.identity = schedule.identity, asdict(group), rows.record(), asdict(admission)
        self.first_uses = tuple(
            (
                row,
                min(
                    (
                        (ky, y)
                        for ky in range(w.kernel[0])
                        for y in range(oh)
                        if y * w.stride[0] + ky * w.dilation[0] == row
                    ),
                    default=None,
                ),
            )
            for row in range(live)
        )
        part = rows.stage_channels or ci
        if part % p.channel_tile and part < ci:
            raise ValueError("pending input transfer starts must match original counted channel readers")
        self.channel_intervals = tuple((c, min(part, ci - c)) for c in range(0, ci, part))
        self.releases = self._plan_releases()
        self.started, self.closed = False, False
        self.visited = set()

    def _plan_releases(self):
        p = self.schedule.source.plan
        w = p.window
        chunks = (w.input_shape[-1] + p.channel_tile - 1) // p.channel_tile
        width = self.group.row_blocks * self.group.panel.column_blocks
        transfers = sorted(
            ((first[0] * w.kernel[1] * chunks + interval[0] // p.channel_tile) * width + first[1], row, interval)
            for row, first in self.first_uses
            if first is not None
            for interval in self.channel_intervals
        )
        lead = self.admission.lead
        if lead is None:
            return tuple((first, first, row, interval) for first, row, interval in transfers)
        if lead.policy == "prefix":
            return tuple((0, first, row, interval) for first, row, interval in transfers)
        # All input is immutable and the complete ring is injective. Earliest
        # release is invocation start; latest is this transfer's actual first
        # reader. Backward spacing reserves original reader bandwidth and bounds
        # live load service using the selected finite entry counts. Negative
        # positions form a real backpressured prologue, never fabricated tickets.
        bank_first = {}
        banks = {}
        if lead.policy == "bank_advance":
            if any(w.padding):
                raise ValueError("bank advance initially requires original unpadded source rows")
            for first, row, interval in transfers:
                selected = set()
                for c in range(interval[0], interval[0] + interval[1], p.channel_tile):
                    for x in range(w.input_shape[2]):
                        address, _ = self.rows.physical_cell(
                            row * self.rows.rows_per_source_row * p.channel_tile + x * w.input_shape[-1] + c
                        )
                        bank = address // lead.bank_rows
                        if not 0 <= bank < lead.bank_count:
                            raise ValueError("input bank map leaves its selected physical row space")
                        selected.add(bank)
                banks[row, interval] = selected
                for bank in selected:
                    bank_first[bank] = min(first, bank_first.get(bank, first))
            # Same-bank deadlines may precede original KY first-use order.
            # Order independent immutable DMA by those bank deadlines before
            # spacing, so late first-bank rows do not prefill the next bank.
            transfers.sort(key=lambda item: (min(bank_first[b] for b in banks[item[1], item[2]]), item[0]))
        result = []
        next_release = None
        source_width = w.input_shape[2]
        for first, row, interval in reversed(transfers):
            latency, spacing = lead.service(source_width, interval[1], w.output_spatial[1])
            deadline = min(bank_first[b] for b in banks[row, interval]) if lead.policy == "bank_advance" else first
            release = deadline - latency
            if next_release is not None:
                release = min(release, next_release - spacing)
            result.append((max(0, release), first, row, interval))
            next_release = release
        return tuple(reversed(result))

    def _check(self):
        self.schedule.verify_current()
        self.rows.revalidate()
        self.admission.require()
        if self.identity != (self.schedule.identity, asdict(self.group), self.rows.record(), asdict(self.admission)):
            raise ValueError("pending input source, allocation or target admission changed")

    def _stage(self, builder, issue, row, channel_interval=None):
        w = self.rows.window
        h, width, ci = w.input_shape[1:]
        source_y = builder.op("sub", row, w.padding[0])
        source = builder.op(
            "add", builder.op("mul", builder.op("add", builder.op("mul", self.image, h), source_y), width * ci), 0
        )
        self.rows.emit_stage(
            builder,
            dict(
                image=self.image,
                input_row=source_y,
                source_byte=source,
                cache_byte=builder.op(
                    "mul",
                    builder.op("urem", row, self.rows.slots),
                    self.rows.rows_per_source_row * self.rows.channel_tile,
                ),
                valid=builder.cmp("ult", source_y, h),
                row_bytes=width * ci,
            ),
            issue,
            channel_interval=channel_interval,
        )

    def start(self, schedule, group, rows, origins, counts, builder, issue, *, columns=None):
        self._check()
        oh, ow = rows.window.output_spatial
        if (
            self.started
            or self.closed
            or schedule is not self.schedule
            or group != self.group
            or rows != self.rows
            or tuple(origins) != tuple(y * ow for y in range(oh))
            or counts != (ow,) * oh
            or columns is not None
            and columns != group.panel.column_blocks
        ):
            raise ValueError("pending input must bind its exact complete original M traversal once")
        self.started = True
        if self.admission.lead is not None and self.admission.lead.policy == "prefix":
            builder.loop(0, len(self.first_uses), 1, lambda row: self._stage(builder, issue, row))
        else:
            for row, first in self.first_uses:
                if first is None:
                    self._stage(builder, issue, row)
        if self.admission.lead is not None and self.admission.lead.policy in ("bounded", "bank_advance"):
            for release, first, row, interval in self.releases:
                if release == 0:
                    self._stage(builder, issue, row, interval)

    def before_reader(self, builder, issue, ky, kx, channel, m, *, n=0):
        if not self.started or self.closed or type(m) is not int or not 0 <= m < self.group.row_blocks:
            raise ValueError("pending input reader is outside its original live M traversal")
        self.visited.add(m)
        p = self.schedule.source.plan
        chunks = (p.window.input_shape[-1] + p.channel_tile - 1) // p.channel_tile
        columns = self.group.panel.column_blocks
        if type(n) is not int or not 0 <= n < columns:
            raise ValueError("pending input reader changes its original N traversal")
        lead = self.admission.lead
        for release, first, row, interval in self.releases:
            if lead is not None and (lead.policy == "prefix" or release == 0):
                continue
            run, inner = divmod(release, columns * self.group.row_blocks)
            rn, rm = divmod(inner, self.group.row_blocks)
            tap, rc = divmod(run, chunks)
            rky, rkx = divmod(tap, p.window.kernel[1])
            if rn != n or rm != m or type(ky) is int and ky != rky or type(kx) is int and kx != rkx:
                continue
            condition = builder.op("and", builder.cmp("eq", ky, rky), builder.cmp("eq", kx, rkx), "i1")
            condition = builder.op("and", condition, builder.cmp("eq", channel, rc * p.channel_tile), "i1")
            builder.branch(
                condition, lambda row=row, interval=interval: self._stage(builder, issue, row, interval), lambda: None
            )

    def finish(self, complete, fields):
        self._check()
        if (
            not self.started
            or self.closed
            or self.visited != set(range(self.group.row_blocks))
            or not callable(complete)
        ):
            raise ValueError("pending input requires every original reader and a real final completion")
        complete("spatial_group_last_use", fields)
        self.closed = True

    def record(self):
        self._check()
        return dict(
            source_identity=self.schedule.source.identity,
            first_uses=[dict(source_row=row, first_KY_M=first) for row, first in self.first_uses],
            original_channel_transfers=self.channel_intervals,
            source_release_pairs=[
                dict(release_pair=a, first_reader_pair=b, source_row=c, channel_interval=d)
                for a, b, c, d in self.releases
            ],
            lead=None if self.admission.lead is None else asdict(self.admission.lead),
            storage=self.rows.record(),
            readiness="actual target scratchpad RAW; no ready token or early drain",
            completion="real original spatial_group_last_use drain completes compute and all input DMA",
        )


def emit_pending_input_images(
    schedule,
    group,
    rows,
    admission,
    builder,
    issue,
    complete,
    *,
    array_law,
    readout_law,
    readout_transfer_law,
    ready_readout,
    output_tiles,
    rhs_frontier=None,
    rhs_issue=None,
):
    """Ordinary counted image and original K/N/M consumer, with pending input."""
    from merlin.compiler.schedule.convolution.output_groups import emit_rolling_convolution_output_group

    p = schedule.source.plan
    batches, _, _, ci = p.window.input_shape
    oh, ow = p.window.output_spatial
    chunks = (ci + p.channel_tile - 1) // p.channel_tile
    runs = p.window.kernel[0] * p.window.kernel[1] * chunks
    # Validate all source and selected effect premises before emitting any DMA.
    checked = PendingConvolutionInput(schedule, group, rows, admission, image=0)
    if rhs_frontier is not None:
        rhs_frontier.require(rows.rolling_plan(), runs)
        if not callable(rhs_issue):
            raise ValueError("pending input retains the actual original RHS stage emitter")

    def image_body(image):
        pending = PendingConvolutionInput(schedule, group, rows, admission, image=image)
        consumer = emit_rolling_convolution_output_group(
            schedule,
            group,
            builder,
            issue,
            complete,
            output_row=builder.op("mul", image, oh * ow),
            row_counts=(ow,) * oh,
            tile_origins=tuple(y * ow for y in range(oh)),
            output_tiles=output_tiles,
            input_rows=rows,
            array_law=array_law,
            readout_law=readout_law,
            readout_transfer_law=readout_transfer_law,
            ready_readout=ready_readout,
            defer_consumer=True,
            pending_input=pending,
        )

        def after_run(run, reduction, depth):
            def stage():
                for n, (column, columns) in enumerate(rhs_frontier.next_columns):
                    local = builder.op(
                        "add",
                        rhs_frontier.rhs_base,
                        builder.op(
                            "mul",
                            builder.op("add", builder.op("mul", run, len(rhs_frontier.next_columns)), n),
                            rhs_frontier.tile,
                        ),
                    )
                    rhs_issue(
                        "load_weight",
                        dict(
                            reduction_row=reduction, output_column=column, local_row=local, rows=depth, columns=columns
                        ),
                    )

            builder.branch(builder.cmp("eq", image, batches - 1), stage, lambda: None)

        consumer.advance(0, runs, after_run=None if rhs_frontier is None else after_run)
        consumer.finish()
        if not pending.closed:
            raise ValueError("pending input escaped without real full consumer completion")
        complete("input_last_use", dict(local_row=rows.base))

    builder.loop(0, batches, 1, image_body)
    return dict(
        schema="merlin.convolution_pending_input.v1",
        **checked.record(),
        original_input_rows_per_image=len(checked.first_uses),
        original_K_runs=runs,
    )
