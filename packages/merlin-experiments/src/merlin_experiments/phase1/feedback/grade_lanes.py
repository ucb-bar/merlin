"""Duration lanes of background grades: a long case drains without holding the next grade.

A background grade owner (:mod:`.dual_snapshots`) used to count as an owner until its slowest admitted
case finished, and at most two or three owners run. Measured on a live run (fresh6, 2026-10-27): the
grades started at 01:43 and 01:45 finished their small cases by 01:45 and 01:50, then held both owner
slots until 02:45 for whole-model and large scalability cases; two more started at 01:52 and 02:08 as
third owners while slots were free, and nothing else started until 02:43. The author's edits went
ungraded for 35 minutes at a time while the configured cadence was two minutes.

Each running case of a grade is in the *short* or the *long* lane. It is long when its estimated seconds
(the scheduler's measured costs, :func:`merlin.targetgen.capsule_scheduling.measured_costs`) exceed the
grade cadence, or once it has already run longer than the cadence: a mispriced case is reclassified by
its own elapsed time, and a case with no price yet starts short and is priced by this run's result. The
boundary is the cadence the grader was configured with, not a constant: a case that would outlast one
grade interval is the one that would hold the next grade. Prices count build and simulation seconds
only, while a running case also waits for CPU and emission, so elapsed time is the stronger signal: the
large cases of that run were priced 40-120 s and held their grade 2.5-23 minutes.

Once a grade has closed admission and every case it still runs is long, the grade is in its long lane:

* it no longer counts as a grade owner, so the next grade starts on the cadence (:mod:`.dual_snapshots`);
* its CPU leases drop to :data:`~merlin.common.cpu_pool.PREPARATION`, so free CPUs go to every newer
  grade's cases first (a class orders waiters; it never preempts a held slot);
* a newer grade withholds a case that another grade is running as long (:func:`withheld`), so one long
  case never runs twice at once and a stream of edits never restarts it; started work is never
  cancelled, so it always finishes and its row is published as soon as it lands
  (:func:`.completed_grade.record_rows`), labelled with the source digest it graded.

Both lanes of one grade share its single snapshot copy and package build: a lane is a phase of the same
owner, never a second grade. Outside a background owner (no :func:`scope`) nothing here applies.

Edits never close a grade's admission, so a newer and an older grade may admit at the same time. Their
work is deduplicated, never cancelled: an older grade skips a case once a newer grade (a later snapshot
token) has started or finished it (:data:`NEWER`), because that result is redundant, not because its own
would be obsolete. The newer grade never skips a case for an older one's sake (except a long run still
draining, above), and its leases outrank the older grade's (:mod:`.dual_snapshots`).
"""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar

_SCOPE: ContextVar[tuple[float, InFlight] | None] = ContextVar("merlin_grade_lane_scope", default=None)
_TRACKERS: weakref.WeakSet = weakref.WeakSet()
_TRACKERS_LOCK = threading.Lock()
WITHHELD = "long run of this test case still draining in an older grade"
NEWER = "a newer grade already started this test case"


def is_long(cost, started, boundary_s, *, now=None) -> bool:
    """Whether one running case is long: priced beyond the boundary, or already running past it.

    Without a positive boundary (no cadence) there are no lanes.
    """
    if boundary_s is None or boundary_s <= 0:
        return False
    now = time.monotonic() if now is None else now
    return (cost is not None and cost > boundary_s) or now - started > boundary_s


class InFlight:
    """The cases one grade is running, as the scheduler reports them, with their price and start time."""

    def __init__(self, group=None, order=None):
        self.group = group  # trackers of one run (its run directory) withhold each other's long cases
        self.order = order  # this grade's snapshot token: a larger one is a newer snapshot
        self.lock = threading.Lock()
        self.running: dict[str, tuple[float | None, float]] = {}
        self.started: set[str] = set()  # every case this grade has started (running or finished)
        with _TRACKERS_LOCK:
            _TRACKERS.add(self)

    def update(self, running: Mapping[str, float | None]) -> None:
        """Scheduler callback: ``{case: estimated seconds or None}`` of every case now running."""
        now = time.monotonic()
        with self.lock:
            self.running = {
                name: (cost, self.running[name][1] if name in self.running else now)
                for name, cost in running.items()
            }
            self.started.update(running)

    def has_started(self, name) -> bool:
        with self.lock:
            return name in self.started

    def long_cases(self, boundary_s, *, now=None) -> frozenset:
        with self.lock:
            return frozenset(name for name, (cost, started) in self.running.items()
                             if is_long(cost, started, boundary_s, now=now))

    def long_lane(self, boundary_s, *, now=None) -> bool:
        """Whether every case still running is long (and at least one runs)."""
        with self.lock:
            running = dict(self.running)
        return bool(running) and all(is_long(cost, started, boundary_s, now=now)
                                     for cost, started in running.values())


@contextmanager
def scope(boundary_s: float, tracker: InFlight):
    """Bind a background owner's lane boundary (its grade cadence in seconds) and its tracker."""
    token = _SCOPE.set((float(boundary_s), tracker))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def withheld(capsule) -> str | None:
    """Why this grade need not start ``capsule``: another grade runs it as long, or a newer grade
    already started it; else None."""
    selected = _SCOPE.get()
    if selected is None or not isinstance(capsule, Mapping):
        return None
    boundary, own = selected
    name = str(capsule.get("name"))
    with _TRACKERS_LOCK:
        others = [tracker for tracker in _TRACKERS if tracker is not own and tracker.group == own.group]
    if any(name in tracker.long_cases(boundary) for tracker in others):
        return WITHHELD
    if own.order is not None and any(tracker.order is not None and tracker.order > own.order
                                     and tracker.has_started(name) for tracker in others):
        return NEWER
    return None


def relevance():
    """The admission ``superseded`` hook of a background owner (:func:`withheld`); ``None`` outside one."""
    return withheld if _SCOPE.get() is not None else None
