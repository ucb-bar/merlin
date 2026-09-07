"""Fast analytical projection for a repeated producer/consumer pipeline.

Cycle-accurate simulation is used to calibrate the duration of each stage on reduced witnesses.  A
full layer supplies a static repetition count. This module evaluates it analytically for a
stationary linear pipeline whose stages occupy distinct resources. Exactness is conditional on
that model, not proof that a reduced measurement predicts the full layer's cache/queue behavior.

The distinct-resource condition is explicit and load-bearing.  A target whose load and store stages
share a port has an arbitration problem, not this tandem-pipeline problem, and must supply a target
adapter/event schedule that models that arbitration.  We refuse instead of granting overlap.
Explicit buffer slots model ownership from producer start through consumer completion. The finite
buffer recurrence is evaluated by max-plus matrix powering, never by replaying full-size tiles.
Omitting buffers selects an explicitly labeled unlimited-buffer idealization, not a capacity proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from .global_planner import Bound, OccupancySummary


@dataclass(frozen=True)
class PipelineStage:
    id: str
    resource: str
    kind: str
    cycles_per_item: CycleInterval
    moved_bytes_per_item: float = 0.0
    commands_per_item: int = 0
    encoding_transition: bool = False
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.resource.strip() or not self.kind.strip():
            raise ValueError("a pipeline stage must name its id, resource, and kind")
        if not self.cycles_per_item.resolved:
            raise ValueError(f"pipeline stage {self.id!r} has unresolved cycles")
        if not all(math.isfinite(value) for value in
                   (self.cycles_per_item.lo, self.cycles_per_item.hi,
                    self.moved_bytes_per_item)):
            raise ValueError("pipeline costs and movement must be finite")
        if self.moved_bytes_per_item < 0 or self.commands_per_item < 0:
            raise ValueError("pipeline movement and command counts must be non-negative")


@dataclass(frozen=True)
class PipelineBuffer:
    """Independent ring slots on one stage edge, held through consumer completion.

    An adapter must derive slots from allocation/lifetime facts, not infer double buffering from
    independent engines. Shared pools, early release, and in-place aliases need a different model.
    ``bytes_per_slot`` is optional: unknown footprint never becomes zero.
    """

    producer: str
    consumer: str
    slots: int
    provenance: str
    bytes_per_slot: int | None = None

    def __post_init__(self) -> None:
        if not self.producer.strip() or not self.consumer.strip() or not self.provenance.strip():
            raise ValueError("pipeline buffers must name their endpoints and evidence")
        if isinstance(self.slots, bool) or not isinstance(self.slots, int) or self.slots < 1:
            raise ValueError("pipeline buffer slots must be a positive integer")
        if self.bytes_per_slot is not None and (
                isinstance(self.bytes_per_slot, bool) or
                not isinstance(self.bytes_per_slot, int) or self.bytes_per_slot < 0):
            raise ValueError("buffer footprint must be non-negative integer bytes or unknown")

    def to_dict(self) -> dict[str, Any]:
        return {"producer": self.producer, "consumer": self.consumer, "slots": self.slots,
                "bytes_per_slot": self.bytes_per_slot,
                "allocation_bytes": (None if self.bytes_per_slot is None else
                                     self.slots * self.bytes_per_slot),
                "provenance": self.provenance}


@dataclass(frozen=True)
class PipelineProjectionPolicy:
    #: Exact interval intersection is linear in ``repetitions * stages``.  Total latency and busy
    #: time stay exact above this cap; only the diagnostic compute/movement intersection is UNKNOWN.
    max_overlap_intervals: int = 1_000_000
    max_recurrence_states: int = 64

    def __post_init__(self) -> None:
        if self.max_overlap_intervals <= 0:
            raise ValueError("max_overlap_intervals must be positive")
        if self.max_recurrence_states <= 0:
            raise ValueError("max_recurrence_states must be positive")


@dataclass(frozen=True)
class PipelineProjection:
    repetitions: int
    cycles: CycleInterval
    resource_floor: Bound
    occupancy: OccupancySummary
    fill_cycles: float
    initiation_interval: float | None
    buffers: tuple[PipelineBuffer, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repetitions": self.repetitions,
            "cycles": self.cycles.to_dict(),
            "resource_floor": self.resource_floor.to_dict(),
            "occupancy": self.occupancy.to_dict(),
            "fill_cycles": self.fill_cycles,
            "initiation_interval": self.initiation_interval,
            "cycle_basis": "model_estimate",
            "buffering": ("unlimited_buffer_idealization" if self.buffers is None else
                          "independent_slots_held_until_consumer_completion"),
            "buffers": None if self.buffers is None else [b.to_dict() for b in self.buffers],
            "scope": "stationary stage-cost model; full-size cache/queue equivalence unproven",
        }


def _endpoint(stages: tuple[PipelineStage, ...], name: str) -> list[float]:
    return [float(getattr(stage.cycles_per_item, name)) for stage in stages]


def _total(durations: list[float], repetitions: int) -> float:
    if repetitions == 0 or not durations:
        return 0.0
    return sum(durations) + (repetitions - 1) * max(durations)


def _buffered_total(durations: list[float], repetitions: int, slots: list[int],
                    max_states: int) -> float:
    """Power the max-plus completion recurrence in O(states**3 * log(repetitions)).

    C[i,j] = p[j] + max(C[i,j-1], C[i-1,j], C[i-slots[j],j+1]).
    The last term reserves an output slot before producer service starts; omit it at the sink.
    Historical completions before item zero are zero (initially empty, available buffers).
    """
    if not repetitions or not durations:
        return 0.0
    # Each stage retains its previous completion and as many older completions as its producer
    # needs for slot reuse. Slots beyond the requested item count cannot block this projection.
    widths = [1, *(min(value, repetitions) for value in slots)]
    size = sum(widths)
    if size > max_states:
        raise ValueError(f"finite-buffer recurrence needs {size} states, exceeds cap {max_states}; "
                         "refuse rather than discard buffer constraints")
    offsets: list[int] = []
    for width in widths:
        offsets.append(sum(widths[:len(offsets)]))
    absent = float("-inf")
    matrix = [[absent] * size for _ in range(size)]
    for j, duration in enumerate(durations):
        row = matrix[offsets[j]]
        row[offsets[j]] = duration  # same resource, preceding item
        if j:
            previous = matrix[offsets[j - 1]]
            for k, weight in enumerate(previous):
                row[k] = max(row[k], weight + duration)
        if j < len(slots) and slots[j] < repetitions:
            k = offsets[j + 1] + slots[j] - 1
            row[k] = max(row[k], duration)
        for history in range(1, widths[j]):
            matrix[offsets[j] + history][offsets[j] + history - 1] = 0.0

    state = [0.0] * size
    power = repetitions
    while power:
        if power & 1:
            state = [max(weight + value for weight, value in zip(row, state))
                     for row in matrix]
        power >>= 1
        if power:
            squared = [[absent] * size for _ in range(size)]
            for i, row in enumerate(matrix):
                for k, left in enumerate(row):
                    if left == absent:
                        continue
                    for j, right in enumerate(matrix[k]):
                        if right != absent:
                            squared[i][j] = max(squared[i][j], left + right)
            matrix = squared
    return state[offsets[-1]]


def _intervals(durations: list[float], stage_index: int,
               repetitions: int) -> list[tuple[float, float]]:
    """Exact service intervals for one stage in a deterministic tandem queue."""
    prefix = durations[:stage_index + 1]
    duration = durations[stage_index]
    prefix_sum = sum(prefix)
    prefix_bottleneck = max(prefix)
    return [
        (prefix_sum + item * prefix_bottleneck - duration,
         prefix_sum + item * prefix_bottleneck)
        for item in range(repetitions)
    ]


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if not merged or lo >= merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return [(lo, hi) for lo, hi in merged]


def _intersection(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    left, right = _union(a), _union(b)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def project_pipeline(stages: tuple[PipelineStage, ...], repetitions: int, *,
                     policy: PipelineProjectionPolicy | None = None,
                     buffers: tuple[PipelineBuffer, ...] | None = None) -> PipelineProjection:
    """Project a stationary model from short-witness stage costs and static repetitions.

    For service times ``p[j]`` in a deterministic tandem line, completion of item ``i`` at stage
    ``j`` is ``sum(p[:j+1]) + i * max(p[:j+1])``.  At the final stage this gives the familiar exact
    fill plus steady-state law ``sum(p) + (N-1)*max(p)``—including compulsory fill/drain instead of
    hiding it in a fitted throughput. This closed form assumes unbounded intermediate storage.
    Explicit buffers instead use a logarithmic-time finite-slot completion recurrence. Neither
    form licenses extrapolating a probe outside its measured mechanism/validity domain.
    """
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 0:
        raise ValueError("pipeline repetitions must be a non-negative integer")
    ids = [stage.id for stage in stages]
    resources = [stage.resource for stage in stages]
    if len(ids) != len(set(ids)):
        raise ValueError("pipeline stage ids must be unique")
    if len(resources) != len(set(resources)):
        raise ValueError(
            "linear pipeline projection requires distinct stage resources; shared-resource "
            "arbitration needs an explicit target event schedule")

    pol = policy or PipelineProjectionPolicy()
    if buffers is not None:
        edges = [(stage.id, following.id) for stage, following in zip(stages, stages[1:])]
        if [(buffer.producer, buffer.consumer) for buffer in buffers] != edges:
            raise ValueError("explicit buffers must cover every adjacent pipeline edge in order")
    lo_durations, hi_durations = _endpoint(stages, "lo"), _endpoint(stages, "hi")
    if buffers is None:
        lo, hi = _total(lo_durations, repetitions), _total(hi_durations, repetitions)
    else:
        slots = [buffer.slots for buffer in buffers]
        lo = _buffered_total(lo_durations, repetitions, slots, pol.max_recurrence_states)
        hi = _buffered_total(hi_durations, repetitions, slots, pol.max_recurrence_states)
    busy = tuple(sorted(
        (stage.resource, float(stage.cycles_per_item.hi) * repetitions) for stage in stages))
    floor = max((cycles * repetitions for cycles in lo_durations), default=0.0)
    compute_resources = tuple(sorted(
        stage.resource for stage in stages if stage.kind == "compute"))
    movement_resources = tuple(sorted(
        stage.resource for stage in stages if stage.kind in ("movement", "encoding")))
    missing: list[str] = []
    overlap: float | None
    available: float | None
    movement_active: float | None = None
    if buffers is not None and repetitions:
        overlap = available = None
        missing.append("finite-slot wall-clock overlap is UNKNOWN; completion and resource busy "
                       "time use the bounded recurrence, without expanding full-model events")
    elif repetitions * len(stages) <= pol.max_overlap_intervals:
        compute_intervals: list[tuple[float, float]] = []
        movement_intervals: list[tuple[float, float]] = []
        for index, stage in enumerate(stages):
            target = (compute_intervals if stage.kind == "compute" else movement_intervals
                      if stage.kind in ("movement", "encoding") else None)
            if target is not None:
                target.extend(_intervals(hi_durations, index, repetitions))
        compute_union, movement_union = _union(compute_intervals), _union(movement_intervals)
        overlap = _intersection(compute_union, movement_union)
        compute_active = sum(end - start for start, end in compute_union)
        movement_active = sum(end - start for start, end in movement_union)
        available = min(compute_active, movement_active)
    else:
        overlap = available = None
        missing.append(
            f"exact compute/movement intersection exceeds the diagnostic interval cap "
            f"{pol.max_overlap_intervals}; latency and per-resource occupancy remain exact")

    provenance = tuple(dict.fromkeys(
        item for stage in stages
        for item in (stage.provenance, *stage.cycles_per_item.provenance) if item))
    if buffers is not None:
        provenance = tuple(dict.fromkeys((*provenance, *(b.provenance for b in buffers))))
    occupancy = OccupancySummary(
        total_cycles=hi,
        busy_cycles=busy,
        compute_resources=compute_resources,
        movement_resources=movement_resources,
        movement_elapsed_cycles=movement_active,
        overlap_cycles=overlap,
        overlap_available_cycles=available,
        idle_cycles=0.0 if hi > 0 else 0.0,
        critical_path_cycles=hi,
        movement_bytes=sum(stage.moved_bytes_per_item for stage in stages
                           if stage.kind in ("movement", "encoding")) * repetitions,
        movement_commands=sum(stage.commands_per_item for stage in stages
                              if stage.kind in ("movement", "encoding")) * repetitions,
        encoding_transitions=sum(stage.encoding_transition for stage in stages) * repetitions,
        provenance=provenance,
        missing=tuple(missing),
    )
    interval = CycleInterval(lo, hi, provenance=provenance)
    return PipelineProjection(
        repetitions, interval, Bound(floor, provenance=provenance), occupancy,
        fill_cycles=sum(hi_durations) if repetitions else 0.0,
        initiation_interval=(max(hi_durations, default=0.0) if buffers is None else None),
        buffers=buffers)
