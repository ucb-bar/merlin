"""Pipelined, dependency-aware capsule grading: no wave barriers, a short calibration head.

The suite used to grade in dependency WAVES, each fully finished before the next began, and to open
every wave with a serial calibration head. Both serialize on the slowest member: one 3600 s RTL run
held twelve small capsules of the next wave for ~50 min while four of five workers idled, and a
corpus whose first member was a full-shape layer ran that one capsule alone for hours.

Here a capsule starts the moment a worker is free and the ``extends`` sibling it cites (if selected)
has FINISHED -- a dependent still never starts before its evidence exists, which is the one ordering
the waves were for. Ready members start longest first (declared operand volume, or with
``MERLIN_CAPSULE_ORDER=measured-longest-first`` the costs this run already measured). Calibration runs
once for the whole suite, on the CHEAPEST ready members, and only while tiers remain unpriced. Results
come back in input order.
"""

from __future__ import annotations

import heapq
import json
import os
import statistics
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from math import isfinite, prod
from pathlib import Path
from typing import Any

#: Wall seconds a serial calibration member may run before the pool fans out around it. Calibration
#: pays off when the head is cheap (a 0.29 s tier priced before 7 workers pay a 24.5 s one); a
#: full-shape layer at the head of a perf corpus took hours, all of it with every other worker idle.
CALIBRATION_BUDGET_S = 120.0
_DEFER_PROBE: ContextVar[Callable[[], str | None] | None] = ContextVar("merlin_capsule_defer_probe", default=None)

#: Operator-selected directories (``os.pathsep``-separated) whose immediate subdirectories are earlier
#: runs roots of this run; their completed capsule results price ``measured-longest-first``.
TIMING_HISTORY_ENV = "MERLIN_CAPSULE_TIMING_HISTORY"
_TIMING_HISTORY: ContextVar[tuple[Path, ...] | None] = ContextVar("merlin_capsule_timing_history", default=None)
_ORDERS = ("", "legacy", "operators-small-first", "measured-longest-first")
#: Parsed timing rows by (path, size, mtime_ns): one long run re-reads its history every grade.
_PARSED: dict[tuple, tuple[str, dict[str, float], dict[str, int]] | None] = {}
_PARSED_LOCK = threading.Lock()


def active_defer_probe() -> Callable[[], str | None] | None:
    """The scheduler's observed deferral reason for this worker, if it is a background grade."""
    return _DEFER_PROBE.get()


@contextmanager
def timing_history(runs_roots: Iterable[str | Path]) -> Iterator[None]:
    """Select earlier runs roots (``<root>/runs/<suite>/<capsule>/``) that price this caller's grades.

    The history only orders ready work. It is never a result, a reuse of a result, or a reason to
    skip a capsule; an unreadable or absent history leaves capsules on the static fallback.
    """
    token = _TIMING_HISTORY.set(tuple(Path(root) for root in runs_roots))
    try:
        yield
    finally:
        _TIMING_HISTORY.reset(token)


_ADMISSION_OBSERVER: ContextVar[tuple[Callable[[str], None], Callable[[], bool] | None] | None] = ContextVar("merlin_admission_observer", default=None)


@contextmanager
def observe_admission(callback, *, stop_requested=None):
    """Notify the trusted snapshot owner when this queue can admit no further work."""
    token = _ADMISSION_OBSERVER.set((callback, stop_requested))
    try:
        yield
    finally:
        _ADMISSION_OBSERVER.reset(token)


def background_admission_selected() -> bool:
    """Whether this context belongs to a trusted automatic snapshot owner.

    This is routing context, not a cancellation or correctness decision. Final
    frozen-compiler checking outside that owner is deliberately separate.
    """
    return _ADMISSION_OBSERVER.get() is not None


_ADMISSION_UNSET = object()
_ROLLING_ADMISSION: ContextVar[dict | None] = ContextVar("merlin_rolling_admission", default=None)


@contextmanager
def rolling_admission(*, on_start=None, admission_deadline_monotonic=None,
                      admission_costs=None, admission_priority=None, admission_window_s=None,
                      admission_allow_oversized_first=False):
    """Trusted per-snapshot admission scope; never a numerical/performance authority."""
    token = _ROLLING_ADMISSION.set(dict(on_start=on_start,
        admission_deadline_monotonic=admission_deadline_monotonic,
        admission_costs=admission_costs, admission_priority=admission_priority,
        admission_window_s=admission_window_s,
        admission_allow_oversized_first=admission_allow_oversized_first))
    try:
        yield
    finally:
        _ROLLING_ADMISSION.reset(token)


def selected_history() -> tuple[Path, ...]:
    """The caller's :func:`timing_history`, else every subdirectory of ``MERLIN_CAPSULE_TIMING_HISTORY``."""
    selected = _TIMING_HISTORY.get()
    if selected is not None:
        return selected
    roots = []
    for raw in os.environ.get(TIMING_HISTORY_ENV, "").split(os.pathsep):
        parent = Path(raw.strip())
        if raw.strip() and parent.is_absolute() and parent.is_dir():
            roots.extend(child for child in sorted(parent.iterdir()) if child.is_dir() and not child.is_symlink())
    return tuple(roots)


def _seconds(value) -> float | None:
    return float(value) if type(value) in (int, float) and isfinite(value) and value >= 0 else None


def _measured_row(path: Path, *, requested_tiers=None, engine=None) -> tuple[str, dict[str, float], dict[str, int]] | None:
    """``(capsule, {tier: measured seconds}, {tier: cycles})`` from one completed result, or None.

    A tier counts as measured only when THIS result executed it (``measured_now``) and recorded its
    own timing; a carried certificate costs nothing to grade again and says nothing about its cost.
    The cost is build plus active simulation as recorded. Queue wait belongs to contention, so the
    adapter wall time is used only when the active split was not recorded. Recorded seconds include
    harness work inside the timing window, so they rank members relative to each other; they are
    not a simulator stopwatch.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns, requested_tiers, engine)
    with _PARSED_LOCK:
        if key in _PARSED:
            return _PARSED[key]
    row = None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        document = None  # a result still being written is simply not history yet
    if isinstance(document, dict) and isinstance(document.get("capsule"), str):
        seconds: dict[str, float] = {}
        cycles: dict[str, int] = {}
        for tier, record in (document.get("tiers") or {}).items():
            if not isinstance(record, dict):
                continue
            if requested_tiers is not None and str(tier) not in requested_tiers:
                continue
            if engine is not None and record.get("engine") != engine:
                continue
            if type(record.get("cycles")) is int and record["cycles"] > 0:
                cycles[str(tier)] = record["cycles"]
            timing = record.get("timing")
            if record.get("measured_now") is not True or not isinstance(timing, dict):
                continue
            fields = ("sim_active_s", "build_s", "adapter_wall_s")
            active, build, wall = (_seconds(timing.get(field)) for field in fields)
            cost = active + (build or 0.0) if active is not None else wall
            if cost is not None:
                seconds[str(tier)] = cost
        row = (document["capsule"], seconds, cycles)
    if row is not None or document is not None:
        with _PARSED_LOCK:
            if len(_PARSED) > 65536:
                _PARSED.clear()
            _PARSED[key] = row
    return row


def measured_costs(runs_roots: Iterable[str | Path], *, requested_tiers=None, engine=None) -> tuple[dict[str, float], dict[str, int]]:
    """Per-capsule seconds from the newest measured run of each tier across earlier runs roots.

    Optional requested tiers/engine filter BEFORE merging history. A complete cost is
    returned only when every requested tier has a matching actual measurement. Unknown
    engine/timing fields cannot price that scope. These are scheduling costs only.

    Each runs root holds ``runs/<suite>/<capsule>/capsule_result.json``. For every capsule each tier
    contributes its newest MEASURED cost, so a capsule whose latest grade failed early still keeps
    the price of the deepest tier it last reached. Also returns the newest observed cycle count
    (carried certificates included), the static-cost fallback for capsules with no measured seconds.
    """
    if requested_tiers is not None:
        requested_tiers = tuple(sorted(set(requested_tiers)))
        if not requested_tiers or any(not isinstance(t, str) or not t for t in requested_tiers):
            raise ValueError("requested timing tiers must be nonempty names")
    if engine is not None and (not isinstance(engine, str) or not engine):
        raise ValueError("timing engine must be a nonempty name")
    files = []
    for root in runs_roots:
        base = Path(root)
        if base.is_absolute() and base.is_dir():
            files.extend((base / "runs").glob("*/*/capsule_result.json"))
    dated = []
    for path in files:
        try:
            dated.append((path.stat().st_mtime_ns, str(path), path))
        except OSError:
            continue
    newest_seconds: dict[str, dict[str, float]] = {}
    newest_cycles: dict[str, int] = {}
    for _, _, path in sorted(dated, reverse=True):
        row = _measured_row(path, requested_tiers=requested_tiers, engine=engine)
        if row is None:
            continue
        name, seconds, cycles = row
        tiers = newest_seconds.setdefault(name, {})
        for tier, cost in seconds.items():
            tiers.setdefault(tier, cost)
        if cycles and name not in newest_cycles:
            newest_cycles[name] = max(cycles.values())
    return {name: sum(t.values()) for name, t in newest_seconds.items()
            if t and (requested_tiers is None or set(requested_tiers) <= t.keys())}, newest_cycles


def cost_estimates(
    capsules: Sequence[Mapping], seconds: Mapping[str, float], cycles: Mapping[str, int]
) -> dict[str, float]:
    """Seconds per capsule: measured, else predicted from observed cycles, else the median.

    The cycle prediction is a least-squares line through this history's own measured
    ``(cycles, seconds)`` pairs (a simulator start-up floor plus a per-cycle slope), used only when
    at least three capsules priced both. A capsule with neither ranks as the median measured one, so
    it is neither starved behind known work nor allowed to displace a known long member.
    """
    names = [str(cap.get("name")) for cap in capsules]
    known = {name: float(seconds[name]) for name in names if name in seconds}
    pairs = [(float(cycles[name]), float(seconds[name])) for name in seconds if name in cycles]
    line = None
    if len(pairs) >= 3:
        mean_x = sum(x for x, _ in pairs) / len(pairs)
        mean_y = sum(y for _, y in pairs) / len(pairs)
        spread = sum((x - mean_x) ** 2 for x, _ in pairs)
        if spread > 0:
            slope = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / spread
            if slope > 0:
                line = (mean_y - slope * mean_x, slope)
    estimates = dict(known)
    for name in names:
        if name not in estimates and line is not None and name in cycles:
            estimates[name] = max(0.0, line[0] + line[1] * cycles[name])
    fallback = statistics.median(estimates.values()) if estimates else 0.0
    return {name: estimates.get(name, fallback) for name in names}


def dependencies(capsules: Sequence[Mapping]) -> dict[str, set[str]]:
    """``name -> {selected sibling}`` from ``extends``; validates names and rejects cycles."""
    names = [str(cap.get("name") or "") for cap in capsules]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("capsule dependency scheduling requires unique nonempty names")
    selected = set(names)
    deps = {
        name: ({str(cap["extends"])} if cap.get("extends") and str(cap["extends"]) in selected else set())
        for name, cap in zip(names, capsules, strict=True)
    }
    done: set[str] = set()
    pending = set(names)
    while pending:
        ready = {name for name in pending if deps[name] <= done}
        if not ready:
            raise ValueError("cyclic capsule extends dependencies: " + ", ".join(sorted(pending)))
        done |= ready
        pending -= ready
    return deps


def operand_volume(capsule: Mapping) -> int:
    """Declared input elements: a cheap, deterministic proxy for a member's simulation cost."""
    total = 0
    for leaf in capsule.get("inputs") or ():
        shape = leaf.get("shape") if isinstance(leaf, Mapping) else None
        if isinstance(shape, list) and all(isinstance(x, int) and x > 0 for x in shape):
            total += prod(shape)
    return total


class BoundaryReadiness:
    """Invocation-local ordering evidence, never cross-case correctness authority.

    Trusted callers bind opaque route keys to their frozen compiler/profile/ABI.
    The first ready small case per route runs normally and is reused as its own
    result. Other routes continue. A bounded wait releases fanout even when a
    probe is slow; failed case emissions cannot prove a route-global failure.
    """

    def __init__(self, routes: Mapping[str, str], *, wait_s: float = 30.0):
        if _seconds(wait_s) is None or wait_s <= 0:
            raise ValueError("boundary readiness wait must be finite positive seconds")
        self.routes = dict(routes)
        self.wait_s = wait_s
        self.probes = {}
        self.completed_routes = set()

    def route(self, cap):
        return self.routes.get(str(cap["name"]))

    def allows(self, cap):
        route = self.route(cap)
        probe = self.probes.get(route)
        return (route is None or probe is None or route in self.completed_routes
                or time.monotonic() >= probe[1])

    def first(self, cap):
        route = self.route(cap)
        return route is not None and route not in self.probes

    def started(self, cap):
        route = self.route(cap)
        if route is not None and route not in self.probes:
            self.probes[route] = (str(cap["name"]), time.monotonic() + self.wait_s)

    def completed(self, name, result):
        route = self.routes.get(name)
        probe = self.probes.get(route)
        if probe is None or probe[0] != name:
            return result
        self.completed_routes.add(route)
        return {**result, "boundary_readiness": {
            "route_sha256": route, "probe_capsule": name,
            "scope": "case", "authority": "ordering_only",
            "result_reused": True, "cross_case_pass_credit": False,
            "route_global_failure_proven": False,
        }}

    def next_wait(self):
        pending = [deadline - time.monotonic() for route, (_, deadline) in self.probes.items()
                   if route not in self.completed_routes and deadline > time.monotonic()]
        return min(pending) if pending else None


def run_pipelined(
    capsules: Sequence[dict],
    run_one: Callable[[dict, int], dict],
    *,
    max_workers: int,
    boundary_readiness: BoundaryReadiness | None = None,
    calibrate: Callable[[], set[str]] | None = None,
    calibration_cap: int = 0,
    calibration_budget_s: float | None = CALIBRATION_BUDGET_S,
    on_complete: Callable[[dict], None] | None = None,
    can_start: Callable[[dict], bool] | None = None,
    blocked_result: Callable[[dict], dict] | None = None,
    calibration_eligible: Callable[[dict], bool] | None = None,
    priority: Callable[[dict], int] | None = None,
    defer_pending: Callable[[], str | None] | None = None,
    defer_running: bool = True,
    on_start: Callable[[dict], None] | None = _ADMISSION_UNSET,
    admission_deadline_monotonic: float | None = _ADMISSION_UNSET,
    admission_costs: Mapping[str, float] | None = _ADMISSION_UNSET,
    admission_priority: Callable[[dict], Any] | None = _ADMISSION_UNSET,
    admission_window_s: float | None = _ADMISSION_UNSET,
    admission_allow_oversized_first: bool = _ADMISSION_UNSET,
) -> list[dict]:
    """Grade ``capsules`` with at most ``max_workers`` in flight, each as soon as its sibling finished.

    ``run_one(capsule, n_parallel)`` grades one member. ``calibrate()`` returns the tiers priced so far;
    while it keeps growing, up to ``calibration_cap`` of the cheapest ready members run first, alone,
    so the ladder learns its tier order before the fan-out (a member that passes ends calibration).
    A head member still running after ``calibration_budget_s`` ends calibration and the pool fans out
    around it (``None`` waits for it). Explicit ``MERLIN_CAPSULE_ORDER=operators-small-first``
    prioritizes ready operators by ascending operand volume before models; absent selection
    preserves legacy priorities. ``measured-longest-first`` starts ready members by descending
    measured cost from the selected :func:`timing_history` (or ``MERLIN_CAPSULE_TIMING_HISTORY``),
    falling back to observed cycles, then declared operand volume, then name, after any caller
    ``priority``. No profile bypasses dependencies or readiness gates, and none changes a result.
    With ``defer_running=False``, deferral drains started members and gates only unstarted ones.
    Optional admission controls bound new submissions, never cancel started work. ``on_start``
    records a submission (not success); costs are trusted scoped wall estimates, not performance
    scores. Unknown costs may start before the deadline. ``admission_priority`` supplies aging
    ahead of ordinary cost ordering, without bypassing dependencies or readiness.
    ``admission_window_s`` begins after queue preparation and is capped by any absolute deadline.
    Optional oversized-first admission prevents an aged expensive member starving forever; it
    retains its original execution timeout. Supplied costs replace unscoped cost/cycle ordering.
    """
    scope = _ROLLING_ADMISSION.get() or {}
    on_start = scope.get("on_start") if on_start is _ADMISSION_UNSET else on_start
    admission_deadline_monotonic = (scope.get("admission_deadline_monotonic")
        if admission_deadline_monotonic is _ADMISSION_UNSET else admission_deadline_monotonic)
    admission_costs = scope.get("admission_costs") if admission_costs is _ADMISSION_UNSET else admission_costs
    admission_priority = scope.get("admission_priority") if admission_priority is _ADMISSION_UNSET else admission_priority
    admission_window_s = scope.get("admission_window_s") if admission_window_s is _ADMISSION_UNSET else admission_window_s
    admission_allow_oversized_first = (scope.get("admission_allow_oversized_first", False)
        if admission_allow_oversized_first is _ADMISSION_UNSET else admission_allow_oversized_first)
    if admission_window_s is not None and (_seconds(admission_window_s) is None or admission_window_s <= 0):
        raise ValueError("admission window must be finite positive seconds")
    if type(admission_allow_oversized_first) is not bool:
        raise ValueError("oversized first admission must be boolean")
    if admission_deadline_monotonic is not None and _seconds(admission_deadline_monotonic) is None:
        raise ValueError("admission deadline must be a finite nonnegative monotonic time")
    scoped_costs_supplied = admission_costs is not None
    admission_costs = dict(admission_costs or {})
    if any(not isinstance(n, str) or _seconds(v) is None for n, v in admission_costs.items()):
        raise ValueError("admission costs must be finite nonnegative wall seconds")
    order = os.environ.get("MERLIN_CAPSULE_ORDER", "").strip()
    if order not in _ORDERS:
        raise ValueError("MERLIN_CAPSULE_ORDER must be legacy, operators-small-first or measured-longest-first")
    small_operators_first = order == "operators-small-first"
    measured_first = order == "measured-longest-first"
    estimate: dict[str, float] = {}
    if measured_first:
        from merlin.targetgen.tier_policy import budget_seconds

        if budget_seconds() is not None:
            # Under a certify budget the start order decides which members are certified at all.
            raise ValueError("measured-longest-first requires an unlimited certify budget")
        estimate = (cost_estimates(capsules, admission_costs, {}) if scoped_costs_supplied
                    else cost_estimates(capsules, *measured_costs(selected_history())))
    head_wait = os.environ.get("MERLIN_CAPSULE_CALIBRATION_S")
    if head_wait is not None:
        try:
            calibration_budget_s = float(head_wait)
        except ValueError as error:
            raise ValueError("MERLIN_CAPSULE_CALIBRATION_S requires finite nonnegative seconds") from error
        if not isfinite(calibration_budget_s) or calibration_budget_s < 0:
            raise ValueError("MERLIN_CAPSULE_CALIBRATION_S requires finite nonnegative seconds")
    deps = dependencies(capsules)
    # Linux affinity is per thread. Keep unleased preparation on the selected
    # manager cores; actual build/simulator step owners lease shared compute slots.
    # Never hold a capsule-wide lease across a child/RPC wait. Restore affinity
    # on every exit; shared mode takes precedence over the legacy grading pool.
    from merlin.common.cpu_pool import grading_cpus, preparation_cpus, selected_pool, shared_compute_enabled

    grade_cpus = grading_cpus()
    prep_cpus = preparation_cpus()
    if shared_compute_enabled():
        max_workers = max(1, min(max_workers, len(grade_cpus)))
    if prep_cpus:
        unpinned_run_one = run_one

        def run_one(capsule, parallel):
            with selected_pool(prep_cpus):
                return unpinned_run_one(capsule, parallel)

    names = [str(cap["name"]) for cap in capsules]
    index = {name: i for i, name in enumerate(names)}
    children: dict[str, list[str]] = {name: [] for name in names}
    for name, parents in deps.items():
        for parent in parents:
            children[parent].append(name)
    waiting = {name: len(parents) for name, parents in deps.items()}
    results: dict[int, dict] = {}
    deferred = None
    admission_closed = None
    admitted: set[str] = set()
    admission_notified = False

    def notify_closed(reason):
        nonlocal admission_notified
        if not admission_notified:
            admission_notified = True
            observer = _ADMISSION_OBSERVER.get()
            if observer is not None:
                observer[0](reason)

    def admission_fits(cap):
        if admission_deadline_monotonic is None:
            return True
        remaining = admission_deadline_monotonic - time.monotonic()
        # An unknown cost remains eligible before the deadline; it is never a zero-cost claim.
        return remaining > 0 and ((admission_allow_oversized_first and not admitted)
                                  or admission_costs.get(str(cap["name"]), 0.0) <= remaining)

    def submit(pool, capsule, parallel):
        from merlin_experiments.phase1.feedback import setup_timing

        with setup_timing.queued():
            return pool.submit(copy_context().run, run_one, capsule, parallel)

    def start(cap):
        if on_start is not None:
            on_start(cap)
        admitted.add(str(cap["name"]))
        if boundary_readiness is not None:
            boundary_readiness.started(cap)

    # Background snapshot replacement may drain started work at capsule boundaries.
    # Other callers retain cooperative cancellation of an owned running child.
    if defer_pending is not None:
        original_run_one = run_one

        def run_one(capsule, parallel):
            token = _DEFER_PROBE.set((lambda: deferred) if defer_running else None)
            try:
                return original_run_one(capsule, parallel)
            finally:
                _DEFER_PROBE.reset(token)

    def should_defer():
        nonlocal deferred, admission_closed
        observer = _ADMISSION_OBSERVER.get()
        if observer is not None and observer[1] is not None and observer[1]():
            admission_closed = "snapshot owner stopped new admission"
        if admission_deadline_monotonic is not None and time.monotonic() >= admission_deadline_monotonic:
            admission_closed = "snapshot admission deadline reached"
        if deferred is None and defer_pending is not None:
            reason = defer_pending()
            deferred = reason if isinstance(reason, str) and reason else None
        if deferred is not None or admission_closed is not None:
            notify_closed(deferred or admission_closed)
        return deferred is not None or admission_closed is not None

    def deferred_results():
        for i, cap in enumerate(capsules):
            if i not in results:
                results[i] = {
                    "capsule": cap["name"],
                    "kind": cap.get("kind"),
                    "label": cap.get("label"),
                    "status": "gated",
                    "background_deferred": deferred or admission_closed,
                    **({"admission_deferred": True, "execution_status": "not_started",
                        "correctness": "unknown"} if admission_closed and cap["name"] not in admitted else {}),
                    "failure": {"plane": "gate",
                        "category": "ADMISSION_DEFERRED" if admission_closed else "BACKGROUND_DEFERRED",
                        "detail": deferred or admission_closed},
                }
        return [results[i] for i in range(len(capsules))]

    # LONGEST FIRST (by declared operand volume, then input order): the classical makespan rule. The
    # member that bounds the suite starts as soon as a worker is free instead of after the small ones.
    volume = {name: operand_volume(cap) for name, cap in zip(names, capsules, strict=True)}
    ready: list[tuple[int, int, str]] = [(-volume[n], index[n], n) for n in names if waiting[n] == 0]
    heapq.heapify(ready)
    # Start the residency window at admission, after caller/source and queue preparation.
    if admission_window_s is not None:
        window_end = time.monotonic() + admission_window_s
        admission_deadline_monotonic = (window_end if admission_deadline_monotonic is None
                                        else min(window_end, admission_deadline_monotonic))

    def take_ready(*, calibration=False):
        if (
            not calibration
            and boundary_readiness is None
            and can_start is None
            and priority is None
            and admission_priority is None
            and admission_deadline_monotonic is None
            and not small_operators_first
            and not measured_first
        ):
            return heapq.heappop(ready) if ready else None
        selectable = [
            item
            for item in ready
            if (can_start is None or can_start(capsules[item[1]]))
            and admission_fits(capsules[item[1]])
            and (boundary_readiness is None or boundary_readiness.allows(capsules[item[1]]))
            and (not calibration or calibration_eligible is None or calibration_eligible(capsules[item[1]]))
        ]
        if not selectable:
            return None
        if small_operators_first:
            # Explicit feedback scheduling: only ready/admitted members compete.
            # Keep models behind ready operators, then prefer smaller declared operands.
            def key(entry):
                return capsules[entry[1]].get("kind") == "model", -entry[0], entry[1]
        elif measured_first and not calibration:
            # Makespan: the member that bounds the grade starts first. Caller priority (models when
            # admitted) still leads; the name is the final deterministic tie-break.
            def key(entry):
                return ((priority(capsules[entry[1]]) if priority else 0), -estimate[entry[2]], entry[0], entry[2])
        else:
            key = (
                (lambda entry: (-entry[0], entry[1]))
                if calibration
                else (lambda entry: ((priority(capsules[entry[1]]) if priority else 0), entry))
            )
        base_key = key
        if admission_priority is not None:
            key = lambda entry: (admission_priority(capsules[entry[1]]), base_key(entry))
        if boundary_readiness is not None:
            # Select the smallest currently admissible member of each untried
            # route, then preserve the caller's fairness across those routes.
            smallest = {}
            for entry in selectable:
                cap = capsules[entry[1]]
                if boundary_readiness.first(cap):
                    route = boundary_readiness.route(cap)
                    if route not in smallest or (-entry[0], entry[1]) < (-smallest[route][0], smallest[route][1]):
                        smallest[route] = entry
            selectable = [entry for entry in selectable
                if not boundary_readiness.first(capsules[entry[1]])
                or smallest[boundary_readiness.route(capsules[entry[1]])] == entry]
        item = min(selectable, key=key)
        ready.remove(item)
        heapq.heapify(ready)
        return item

    def finish_blocked():
        nonlocal admission_closed
        # Work that cannot fit this window is deferred, not an operator/correctness refusal.
        if admission_deadline_monotonic is not None and any(
            (can_start is None or can_start(capsules[item[1]]))
            and not admission_fits(capsules[item[1]]) for item in ready
        ):
            admission_closed = "remaining ready work exceeds snapshot admission window"
            return
        if blocked_result is None:
            raise ValueError("capsule readiness gate cannot progress and has no terminal result")
        *_, name = heapq.heappop(ready)
        finished(name, blocked_result(capsules[index[name]]))

    def finished(name: str, result: dict) -> None:
        # Completion consumers bind this exact native row to its durable receipt.
        # Ordering diagnostics belong to the collected scheduling summary only;
        # adding them to the callback would invalidate that custody comparison.
        completed_result = result
        if boundary_readiness is not None:
            result = boundary_readiness.completed(name, result)
        results[index[name]] = result
        if on_complete is not None:
            on_complete(completed_result)
        for child in children[name]:
            waiting[child] -= 1
            if waiting[child] == 0:
                heapq.heappush(ready, (-volume[child], index[child], child))

    if max_workers <= 1:
        while ready:
            if should_defer():
                return deferred_results()
            item = take_ready()
            if item is None:
                finish_blocked()
                continue
            *_, name = item
            start(capsules[index[name]])
            finished(name, run_one(capsules[index[name]], 1))
        return [results[i] for i in range(len(capsules))]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        running: dict[Any, str] = {}
        if calibrate is not None and calibration_cap > 0:
            seen = calibrate()
            for _ in range(calibration_cap):
                if should_defer():
                    return deferred_results()
                if not ready:
                    break
                item = take_ready(calibration=True)
                if item is None:
                    break
                start(capsules[item[1]])
                future = submit(pool, capsules[item[1]], 1)
                done, _ = wait([future], timeout=calibration_budget_s)
                if not done:
                    # A slow head teaches its tier order too late to be worth the idle workers: keep it
                    # running and fan out now, instead of grading one capsule alone for hours.
                    running[future] = item[2]
                    break
                result = future.result()
                finished(item[2], result)
                now = calibrate()
                if result.get("status") == "pass" or now == seen:
                    break
                seen = now
        while ready or running:
            should_defer()
            while ready and not should_defer() and len(running) < max_workers:
                item = take_ready()
                if item is None:
                    break
                *_, name = item
                start(capsules[index[name]])
                running[submit(pool, capsules[index[name]], max_workers)] = name
            if len(results) + len(running) == len(capsules):
                notify_closed("all selected work admitted or completed")
            if not running:
                if deferred or admission_closed:
                    return deferred_results()
                finish_blocked()
                continue
            # Observe obsolescence even while running work drains; pending members
            # never start after it, with optional cooperative running-child cancellation.
            wait_s = 5 if defer_pending is not None or admission_deadline_monotonic is not None or _ADMISSION_OBSERVER.get() is not None else None
            if boundary_readiness is not None:
                head_wait = boundary_readiness.next_wait()
                if head_wait is not None:
                    wait_s = head_wait if wait_s is None else min(wait_s, head_wait)
            done, _ = wait(running, timeout=wait_s, return_when=FIRST_COMPLETED)
            for future in done:
                finished(running.pop(future), future.result())
    return [results[i] for i in range(len(capsules))]
