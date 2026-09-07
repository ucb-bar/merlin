"""Resource timeline for data movement, compute occupancy, and latency hiding.

An adapter lowers a selected global plan to events.  Dependencies say when values become available;
``resource`` and ``serial_group`` say which events cannot occupy hardware together.  Nothing here
infers overlap from a unit name.  Independent events on independent declared resources may overlap,
while a non-overlapping target puts them in one serial group or supplies the required dependency.

This is an analytical schedule over exact full-model event counts.  Reduced simulations calibrate
the event durations; they are not replayed once per full-size tile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .global_planner import OccupancySummary


@dataclass(frozen=True)
class ActivityEvent:
    id: str
    resource: str
    kind: str
    cycles: float
    depends_on: tuple[str, ...] = ()
    serial_group: str = ""
    movement_bytes: float = 0.0
    movement_commands: int = 0
    encoding_transition: bool = False
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.resource.strip() or not self.kind.strip():
            raise ValueError("an activity event must name its id, resource, and kind")
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, (int, float)):
            raise TypeError("activity cycles must be numeric")
        if self.cycles < 0 or self.movement_bytes < 0 or self.movement_commands < 0:
            raise ValueError("activity cycles, bytes, and command counts must be non-negative")
        if self.id in self.depends_on:
            raise ValueError(f"activity event {self.id!r} depends on itself")


@dataclass(frozen=True)
class ScheduledEvent:
    event: ActivityEvent
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event.id,
            "resource": self.event.resource,
            "kind": self.event.kind,
            "start": self.start,
            "end": self.end,
            "cycles": self.event.cycles,
            "depends_on": list(self.event.depends_on),
            "serial_group": self.event.serial_group,
        }


def _merged_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if not merged or lo >= merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return [(lo, hi) for lo, hi in merged]


def _union_length(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(hi - lo for lo, hi in _merged_intervals(intervals))


def _intersection_length(a: Sequence[tuple[float, float]],
                         b: Sequence[tuple[float, float]]) -> float:
    # Several independent engines of one kind can occupy the same wall-clock
    # interval. Count that interval once, not once per participating engine.
    left = _merged_intervals(a)
    right = _merged_intervals(b)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if hi > lo:
            total += hi - lo
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


@dataclass(frozen=True)
class ActivityTimeline:
    events: tuple[ScheduledEvent, ...]
    total_cycles: float
    critical_path_cycles: float

    def occupancy(self, *, compute_kinds: Sequence[str] = ("compute",),
                  movement_kinds: Sequence[str] = ("movement", "encoding"),
                  provenance: Sequence[str] = ()) -> OccupancySummary:
        compute_set, movement_set = set(compute_kinds), set(movement_kinds)
        busy: dict[str, float] = {}
        compute_resources: set[str] = set()
        movement_resources: set[str] = set()
        compute_intervals: list[tuple[float, float]] = []
        movement_intervals: list[tuple[float, float]] = []
        all_intervals: list[tuple[float, float]] = []
        movement_bytes = 0.0
        movement_commands = 0
        encoding_transitions = 0
        for scheduled in self.events:
            event = scheduled.event
            busy[event.resource] = busy.get(event.resource, 0.0) + event.cycles
            interval = (scheduled.start, scheduled.end)
            all_intervals.append(interval)
            if event.kind in compute_set:
                compute_resources.add(event.resource)
                compute_intervals.append(interval)
            if event.kind in movement_set:
                movement_resources.add(event.resource)
                movement_intervals.append(interval)
                movement_bytes += event.movement_bytes
                movement_commands += event.movement_commands
            encoding_transitions += int(event.encoding_transition)
        compute_busy = _union_length(compute_intervals)
        movement_busy = _union_length(movement_intervals)
        overlap = _intersection_length(compute_intervals, movement_intervals)
        active = _union_length(all_intervals)
        return OccupancySummary(
            total_cycles=self.total_cycles,
            busy_cycles=tuple(sorted(busy.items())),
            compute_resources=tuple(sorted(compute_resources)),
            movement_resources=tuple(sorted(movement_resources)),
            movement_elapsed_cycles=movement_busy,
            overlap_cycles=overlap,
            overlap_available_cycles=min(compute_busy, movement_busy),
            idle_cycles=max(0.0, self.total_cycles - active),
            critical_path_cycles=self.critical_path_cycles,
            movement_bytes=movement_bytes,
            movement_commands=movement_commands,
            encoding_transitions=encoding_transitions,
            provenance=tuple(provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"total_cycles": self.total_cycles,
                "critical_path_cycles": self.critical_path_cycles,
                "events": [event.to_dict() for event in self.events]}


def schedule_activity(events: Sequence[ActivityEvent]) -> ActivityTimeline:
    """ASAP-schedule topologically ordered events under declared resource constraints.

    The caller supplies a topological order.  A dependency that has not appeared is refused rather
    than treated as ready at cycle zero.  Events sharing a resource serialize; events on different
    resources serialize only when they share a non-empty ``serial_group``.
    """

    ids = [event.id for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError("activity event ids must be unique")
    end_by_id: dict[str, float] = {}
    path_by_id: dict[str, float] = {}
    resource_free: dict[str, float] = {}
    group_free: dict[str, float] = {}
    scheduled: list[ScheduledEvent] = []
    for event in events:
        absent = [dep for dep in event.depends_on if dep not in end_by_id]
        if absent:
            raise ValueError(
                f"activity event {event.id!r} has non-topological or absent dependencies {absent}")
        ready = max((end_by_id[dep] for dep in event.depends_on), default=0.0)
        start = max(ready, resource_free.get(event.resource, 0.0))
        if event.serial_group:
            start = max(start, group_free.get(event.serial_group, 0.0))
        end = start + float(event.cycles)
        scheduled.append(ScheduledEvent(event, start, end))
        end_by_id[event.id] = end
        resource_free[event.resource] = end
        if event.serial_group:
            group_free[event.serial_group] = end
        path_by_id[event.id] = float(event.cycles) + max(
            (path_by_id[dep] for dep in event.depends_on), default=0.0)
    total = max(end_by_id.values(), default=0.0)
    critical = max(path_by_id.values(), default=0.0)
    return ActivityTimeline(tuple(scheduled), total, critical)
