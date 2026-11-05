"""Use idle reserved CPUs to keep the performance evidence of passing performance-cohort cases current.

Between grades the reserved CPUs sat near idle (a 20-CPU run averaged about 10) while the performance
ranking the author works from was built on estimates of older source, or on none: the timing model
only runs when the author asks for a scalability self-check, and the largest cases were rarely asked.
This module spends that idle time on exactly those estimates.

While shared CPU slots are free (beyond a small reserve while a grade is admitting, so a starting
grade still finds slots, and beyond the CPUs that queued requests are about to take) and the volume has
room, it submits operator requests through the self-check broker for passing performance-cohort cases
whose estimate does not describe the current submission and model, until the free CPUs are spoken for
(see :data:`KNOBS`; the caps are optional). It is the same request an author self-check makes
(``suite: scalability``, Spike only), so custody, snapshotting and response binding are identical; the
broker runs it at the advisory :data:`~merlin.common.cpu_pool.AUXILIARY` lease class and dispatches it
after every author request, so author requests and grades lease CPUs first. Author priority is that
lease class and dispatch order: a waiting author request does not stop this work, it only keeps the
CPUs it is about to take. The request and its response carry ``origin: operator_idle_performance`` and
an ``op_idle_`` id. It certifies nothing and starts no promotion: the scalability suite has no cert tier.

In a joint campaign the same gates also admit the gSIM lane (:mod:`.idle_gsim`, policy section
``gsim_measurement``): measured cycles of passing scalability cases, largest headroom first, within an
explicit share of the campaign's gSIM budget; it precedes ordinary timing and tuning.
Before that lane and tuning, the two largest declared dense performance cases
each get one early Spike timing slot once they pass L2 and lack a passing model estimate.
This finite allowance survives compiler edits; normal coverage then resumes.

The same free CPUs run tuning experiments (:mod:`.idle_tuning`, ``MERLIN_IDLE_TUNING=0`` disables only
those): baseline timing requests and experiments alternate, each in its own value order, and an
experiment's missing baselines are timed beside it. Experiments do not wait for every estimate to be
current -- with authors editing continuously that never held.

Leases are not preemptive, so this lane yields instead: whenever a process ranked above it (an author
check, a grade, a draining grade or its preparation) waits for a shared slot while none is free, the
newest running timing requests are withdrawn (the broker's cancellation: the request file goes away and
the broker stops its worker) until they cover the waiters, and their cases become eligible again. This
is checked on every grader poll, not on the gate interval. Tuning experiments and the gSIM lane keep
their own rules.

A claimed request counts as about to take its CPUs only while it starts (:data:`CLAIM_GRACE_S`); a
long author check that publishes no progress record holds its slots itself after that, and counting it
again as queued kept this lane shut for whole runs.

Disable with ``MERLIN_IDLE_PERF_MEASURE=0`` (which also stops tuning experiments). Failures are logged
and never reach the grader.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from . import scratch_retention

ENV = "MERLIN_IDLE_PERF_MEASURE"
ORIGIN = "operator_idle_performance"
#: Origin of :mod:`.idle_tuning` experiments; both run at the AUXILIARY lease class.
TUNING_ORIGIN = "operator_idle_tuning"
#: Origin of the gSIM lane (:mod:`.idle_gsim`); its requests also run at the AUXILIARY lease class.
from ..joint_gsim_measurement import ORIGIN as GSIM_ORIGIN  # noqa: E402 -- a constant beside its siblings

OPERATOR_ORIGINS = (ORIGIN, TUNING_ORIGIN, GSIM_ORIGIN)
#: Every operator background request id starts with this, so it is dispatched after author requests.
REQUEST_PREFIX = "op_idle_"
#: Knobs, each an environment override (name: default). Work is sized from the free capacity seen at
#: submission; the ceilings are only safety caps.
KNOBS = {
    "MERLIN_IDLE_PERF_MIN_FREE_SLOTS": 1,  # free shared CPU slots needed to submit anything
    "MERLIN_IDLE_PERF_MAX_CASES": 4,  # ceiling on cases per request (actual: min(this, free slots))
    "MERLIN_IDLE_PERF_MAX_REQUESTS": 0,  # optional cap on background requests in flight (0: sized from free CPUs)
    "MERLIN_IDLE_PERF_INTERVAL_S": 60,  # seconds between two submissions
    "MERLIN_IDLE_PERF_CHECK_S": 30,  # seconds between two evaluations of the gates
    "MERLIN_IDLE_PERF_TIMEOUT_S": 1800,  # client timeout each background request declares
    # Slots left free while a grade is admitting (0: a fifth of the pool, at least 2). An editing
    # author keeps some grade admitting almost all the time, so waiting for none never measured.
    "MERLIN_IDLE_PERF_GRADE_RESERVE_SLOTS": 0,
}
_MAX_RECORD_BYTES = 64 << 20
#: Seconds a claimed request without a progress record still counts as about to lease its CPUs: a
#: starting handler leases within seconds; after this its CPUs are held slots or ranked pool waiters.
CLAIM_GRACE_S = 30
_VERDICTS_READ = 6
# A finite first-evidence allowance, not a per-source refresh quota.
_EARLY_DENSE_CASES = 2


def knob(name: str, environ=None) -> int:
    """``name``'s positive integer value from the environment, else its default."""
    raw = (os.environ if environ is None else environ).get(name, "").strip()
    return int(raw) if raw.isascii() and raw.isdecimal() and int(raw) > 0 else KNOBS[name]


def pool_size(environ=None) -> int | None:
    from merlin.common import cpu_pool

    selection = cpu_pool._shared_compute_selection(os.environ if environ is None else environ)
    return None if selection is None else len(selection[0])


def grade_reserve(environ=None) -> int:
    """Slots kept free for grades while one is admitting (see the knob)."""
    chosen = knob("MERLIN_IDLE_PERF_GRADE_RESERVE_SLOTS", environ)
    if chosen > 0:
        return chosen
    size = pool_size(environ) or 0
    return max(2, size // 5)


def enabled(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(ENV, "1").strip() != "0"


def _load(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_RECORD_BYTES:
            return None
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _deadline_s(body: dict, path: Path) -> float:
    explicit = body.get("deadline_unix_ns")
    if isinstance(explicit, int):
        return explicit / 1e9
    try:
        return path.stat().st_mtime + int(body.get("timeout", 1800)) + 240
    except (OSError, ValueError, TypeError):
        return 0.0


def outstanding(channel: Path, *, now: float | None = None) -> tuple[int, int]:
    """``(author waiting, operator in flight)``: live requests without a response; an author request
    counts only until the broker claims it (a running author check is not waiting on anything)."""
    now = time.time() if now is None else now
    author = operator = 0
    for request in channel.glob("req_*.json"):
        rid = request.name[len("req_") : -len(".json")]
        if (channel / f"resp_{rid}.json").exists():
            continue
        body = _load(request)
        if not isinstance(body, dict) or now > _deadline_s(body, request):
            continue
        if rid.startswith(REQUEST_PREFIX):
            operator += 1
        elif not (channel / f"claimed_{rid}").exists():
            author += 1
    return author, operator


def _request_cpus(body: dict) -> int:
    """CPUs a request will lease at once: one per named case; a corpus selector takes the whole pool."""
    raw = body.get("capsules")
    names = [name for name in raw.split(",") if name.strip()] if isinstance(raw, str) else []
    if not names or any(name.strip() in ("all", "changed", "failing") or ":" in name for name in names):
        return 1 << 20
    return len(names)


def _starting(channel: Path, rid: str, now: float) -> bool:
    """Whether a request is unclaimed, or claimed so recently (no progress yet) that it is still starting."""
    try:
        claimed_at = (channel / f"claimed_{rid}").stat().st_mtime
    except OSError:
        return True
    return not (channel / f"progress_{rid}.json").exists() and now - claimed_at < CLAIM_GRACE_S


def queued(channel: Path, *, now: float | None = None) -> dict:
    """Live requests that hold no CPU yet, by the CPUs they will ask for.

    ``author_cases`` and ``operator_cases``: requests the broker has not claimed, or claimed within
    :data:`CLAIM_GRACE_S` without a progress record (a starting handler leases within seconds).
    ``operator_live``: operator requests without a response. Their CPUs are about to be taken, so they
    are not idle capacity."""
    now = time.time() if now is None else now
    found = {"author_cases": 0, "operator_cases": 0, "operator_live": 0}
    for request in channel.glob("req_*.json"):
        rid = request.name[len("req_") : -len(".json")]
        if (channel / f"resp_{rid}.json").exists():
            continue
        body = _load(request)
        if not isinstance(body, dict) or now > _deadline_s(body, request):
            continue
        operator = rid.startswith(REQUEST_PREFIX)
        found["operator_live"] += operator
        if not _starting(channel, rid, now):
            continue
        found["operator_cases" if operator else "author_cases"] += _request_cpus(body)
    return found


def free_slots(environ=None) -> int | None:
    """Free slots of the selected shared compute pool, or ``None`` when no pool is selected."""
    from merlin.common import cpu_pool

    selection = cpu_pool._shared_compute_selection(os.environ if environ is None else environ)
    if selection is None:
        return None
    cpus, root = selection
    if not root.is_dir():
        return len(cpus)
    return len(cpu_pool._free_cpus(root, cpus, fresh=True))


def contention(environ=None) -> int:
    """Processes ranked above this lane waiting for a shared compute slot while none is free (0: none,
    no pool selected, or an unreadable queue)."""
    from merlin.common import cpu_pool

    try:
        selection = cpu_pool._shared_compute_selection(os.environ if environ is None else environ)
        if selection is None:
            return 0
        cpus, root = selection
        root = Path(root)
        if not root.is_dir() or cpu_pool._free_cpus(root, cpus, fresh=True):
            return 0
        return sum(1 for key in cpu_pool._queued_keys(root / cpu_pool._WAITER_DIR) if key[0] < cpu_pool.AUXILIARY)
    except Exception:  # noqa: BLE001 -- an unreadable queue never withdraws work by itself
        return 0


def withdraw(channel: Path, wanted: int, *, now: float | None = None) -> list[tuple[str, list[str], str]]:
    """Cancel the newest running idle Spike timing requests until they cover ``wanted`` CPUs.

    Only this lane's own requests (:data:`ORIGIN`, claimed, unanswered): a request the broker has not
    started holds no CPU. Removing the request file is the broker's cancellation. Returns
    ``(request id, cases, source digest)`` of each withdrawn request."""
    now = time.time() if now is None else now
    running = []
    for request in channel.glob(f"req_{REQUEST_PREFIX}*.json"):
        rid = request.name[len("req_") : -len(".json")]
        if (channel / f"resp_{rid}.json").exists() or not (channel / f"claimed_{rid}").exists():
            continue
        body = _load(request)
        if not isinstance(body, dict) or body.get("origin") != ORIGIN or now > _deadline_s(body, request):
            continue
        running.append((body.get("requested_at_unix_ns") or 0, rid, request, body))
    withdrawn = []
    for _, rid, request, body in sorted(running, key=lambda row: row[:2], reverse=True):
        if wanted <= 0:
            break
        try:
            request.unlink()
        except FileNotFoundError:
            continue  # answered or cancelled meanwhile
        names = [name.strip() for name in str(body.get("capsules") or "").split(",") if name.strip()]
        withdrawn.append((rid, names, str(body.get("submission_sha256"))))
        wanted -= max(1, len(names))
    return withdrawn


def _verdicts(run_dir: Path) -> list[dict]:
    paths = sorted(
        (Path(run_dir) / "qa_history").glob("verdict_fast_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    from .result_details import hydrate

    documents = (hydrate(_load(path), document_path=path) for path in paths[:_VERDICTS_READ])
    return [d for d in documents if isinstance(d, dict)]


def _cohort_l2(verdicts: list[dict], *, workspace=None) -> tuple[list[str], dict]:
    """The performance cohort and each member's newest stated L2 status."""
    cohort: list[str] | None = None
    status: dict[str, str] = {}
    for verdict in verdicts:  # newest first
        if cohort is None:
            members = ((verdict.get("joint_test_selection") or {}).get("performance_cohorts") or {}).get("primary")
            if isinstance(members, list):
                roles = (verdict.get("joint_test_selection") or {}).get("workload_roles")
                if roles is not None:
                    from ..workload_roles import PRIMARY
                    members = [name for name in members if roles.get(name) in PRIMARY]
                cohort = [str(name) for name in members]
        for row in verdict.get("per_capsule") or ():
            name = row.get("capsule") if isinstance(row, dict) else None
            tier = (row.get("tiers") or {}).get("L2") if isinstance(row, dict) else None
            tier = tier.get("status") if isinstance(tier, dict) else tier
            if name and name not in status and tier in ("pass", "fail"):
                status[str(name)] = tier
    if workspace is not None:
        from ..workload_roles import PRIMARY, workspace_roles
        from . import family_rollup, reconcile

        if cohort is None:
            # The same staged public suite/roles are available before the first
            # completed grade. The broker still admits every named source case.
            roles, members = workspace_roles(workspace) or {}, family_rollup.inventory(workspace)
            cohort = [name for name, role in roles.items() if role in PRIMARY
                      and (members.get(name) or {}).get("selfcheck_suite") == "scalability"]
        for name, row in family_rollup.readiness_rows(workspace).items():
            if name not in (cohort or ()):
                continue
            if reconcile.is_older(row):
                status.pop(name, None)
            elif (state := family_rollup.verdict(row)) in ("pass", "fail"):
                status[name] = state
    return cohort or [], status


def estimates(channel: Path, *, with_status: bool = False, selection_sha256: str | None = None) -> dict:
    """``{case: [(graded_at_ns, submission_sha256, performance)]}`` from submission self-check responses.

    Variant responses (and knob-override copies, which carry a ``variant`` block) never describe
    ``submission/``. ``with_status`` appends whether the row passed its numerical check.
    An explicit selection pin excludes older model epochs without deleting their evidence.
    """
    found: dict[str, list] = {}
    from .result_details import load_row

    for path in channel.glob("resp_*.json"):
        document = _load(path)
        if (
            not isinstance(document, dict)
            or document.get("variant")
            or document.get("selfcheck_submission") not in (None, "submission")
        ):
            continue
        digest, graded = document.get("submission_sha256"), document.get("graded_at_unix_ns") or 0
        for compact in document.get("cases") or document.get("per_capsule") or ():
            row = load_row(compact, document_path=path)
            if isinstance(compact, dict) and compact.get("detail") and row is compact:
                continue  # A summary with unavailable details cannot establish a timing identity.
            performance = row.get("performance") if isinstance(row, dict) else None
            if selection_sha256 is not None and (not isinstance(performance, dict)
                    or performance.get("selection_sha256") != selection_sha256):
                continue
            cycles = performance.get("estimated_whole_soc_cycles") if isinstance(performance, dict) else None
            if isinstance(digest, str) and type(cycles) in (int, float) and cycles > 0:
                entry = (graded, digest, performance) + ((passed(row),) if with_status else ())
                found.setdefault(str(row.get("capsule")), []).append(entry)
    for rows in found.values():
        rows.sort(key=lambda r: r[0], reverse=True)
    return found


def _selection_identity() -> str | None:
    """Latest trusted selection at an idle admission, never an active request's model."""
    from .. import joint_control, joint_model_activation
    from . import joint_performance

    state = os.environ.get(joint_control.ENV)
    if state:
        pin = joint_model_activation.current(state)
    else:
        raw = os.environ.get(joint_performance.ENV)
        pin = json.loads(raw) if raw else None
    if pin is None:
        return None  # legacy runs without a selected model retain their existing behavior
    joint_performance._read(pin)
    return pin["sha256"]


def passed(row: dict) -> bool:
    """Whether a response row passed its numerical check (the L2 tier, else its own pass flags)."""
    tiers = row.get("tiers")
    if isinstance(tiers, dict) and "L2" in tiers:
        tier = tiers["L2"]
        return (tier.get("status") if isinstance(tier, dict) else tier) == "pass"
    return row.get("pass") is True or row.get("numerical_execution_passed") is True or row.get("status") == "pass"


def select(cohort, l2, known: dict, digest: str, requested: set, *, limit: int, preferred=frozenset(),
           work=None) -> list[str]:
    """Passing cohort cases without an estimate of ``digest``.

    Cases never timed and cases timed on older source alternate, so the largest stale estimates are
    refreshed without starving the cases that have none (the source digest moves with every edit).
    Stale cases go largest previous estimate first; untimed ones largest declared ``work`` first
    (:func:`_declared_work`), then in cohort order. Within each, a ``preferred`` case (a bundle
    representative, :func:`.family_rollup.preferred_cases`) goes first: an ordering preference only.
    """
    work = work or {}
    eligible = [
        name
        for name in cohort
        if l2.get(name) == "pass"
        and (name, digest) not in requested
        and not any(row[1] == digest for row in known.get(name) or ())
    ]
    untimed = sorted((name for name in eligible if not known.get(name)),
                     key=lambda name: (name not in preferred, -work.get(name, 0)))
    stale = sorted(
        (name for name in eligible if known.get(name)),
        key=lambda name: (name not in preferred, -(known[name][0][2].get("estimated_whole_soc_cycles") or 0), name),
    )
    chosen = []
    while (untimed or stale) and len(chosen) < limit:
        for group in (untimed, stale):
            if group and len(chosen) < limit:
                chosen.append(group.pop(0))
    return chosen


def _declared_work(workspace: Path, cohort, *, family: str | None = None) -> dict[str, int]:
    """``{case: declared work}`` of the cases whose declaration states it (of ``family`` when given)."""
    from .family_rollup import describe, inventory

    members = inventory(workspace)
    found = {}
    for name in cohort:
        _, shape = describe(name, members.get(name), workspace)
        if (shape and (family is None or shape.get("family") == family)
                and type(shape.get("work")) is int and shape["work"] > 0):
            found[name] = shape["work"]
    return found


def _largest_dense(workspace: Path, cohort) -> list[str]:
    """The largest declared dense work, independent of names, timing and passing order."""
    ranked = sorted((-work, name) for name, work in _declared_work(workspace, cohort, family="dense").items())
    return [name for _, name in ranked[:_EARLY_DENSE_CASES]]


class IdleMeasurer:
    """One per background grader; :meth:`tick` is cheap until every gate opens."""

    def __init__(self, workspace, run_dir, *, digest=None, clock=time.monotonic):
        self.workspace, self.run_dir = Path(workspace), Path(run_dir)
        self._digest = digest
        self._clock = clock
        self._next = 0.0
        self._next_check = 0.0
        self._timing_turn = True
        self.requested: set[tuple[str, str]] = set()
        self._early_dense_attempted: set[str] = set()
        self._selection_sha256: str | None = None
        self._preferred_for: tuple | None = None
        self._preferred_cases: frozenset = frozenset()
        from .idle_tuning import IdleTuner

        self.tuner = IdleTuner(
            self.workspace, digest=None if digest is None else (lambda label: digest(self.workspace / label))
        )
        from .idle_gsim import GsimLane

        self.gsim = GsimLane(self.workspace, self.run_dir)

    def _submission_digest(self) -> str:
        if self._digest is not None:
            return self._digest(self.workspace / "submission")
        from merlin.common.cpu_pool import AUXILIARY, cpu_priority

        from ..brokers.selfcheck import _submission_digest

        with cpu_priority(AUXILIARY):  # holds no slot while it waits for one
            return _submission_digest(self.workspace / "submission")

    def tick(self, *, grading: bool) -> str:
        """Fill free shared capacity with background requests; return their status. Never raises."""
        try:
            if enabled() and (withdrawn := self._yield()):
                return withdrawn
            return self._tick(grading)
        except Exception as error:  # noqa: BLE001 -- background measurement never reaches the grader
            self._next = self._clock() + knob("MERLIN_IDLE_PERF_INTERVAL_S")
            scratch_retention.say(f"[idle-perf] skipped: {type(error).__name__}: {str(error)[:200]}")
            return "error"

    def _yield(self) -> str | None:
        """Withdraw running timing requests while work ranked above them waits for a slot."""
        channel = self.workspace / ".qa_channel"
        if not any(channel.glob(f"req_{REQUEST_PREFIX}*.json")):
            return None
        wanted = contention()
        withdrawn = withdraw(channel, wanted) if wanted else []
        if not withdrawn:
            return None
        for _, names, digest in withdrawn:  # never measured: eligible again once CPUs are idle
            self.requested.difference_update((name, digest) for name in names)
            self._early_dense_attempted.difference_update(names)
        self._next = self._clock() + knob("MERLIN_IDLE_PERF_INTERVAL_S")
        scratch_retention.say(
            f"[idle-perf] {wanted} higher-priority waiter(s) for a CPU: withdrew background Spike timing of "
            + "; ".join(", ".join(names) for _, names, _ in withdrawn))
        return "yielded"

    def _tick(self, grading: bool) -> str:
        if not enabled():
            return "disabled"
        now = self._clock()
        if now < self._next or now < self._next_check:
            return "rate limited"
        self._next_check = now + knob("MERLIN_IDLE_PERF_CHECK_S")
        channel = self.workspace / ".qa_channel"
        if not channel.is_dir():
            return "no channel"
        free = free_slots()
        if free is None or os.environ.get("MERLIN_SELFCHECK_BROKER_JOBS", "").strip() == "1":
            return "single-lane broker"  # one lane would queue the next author request behind this work
        # Leases are not preemptive: while a grade admits, use only the slots beyond its reserve.
        free -= grade_reserve() if grading else 0
        if free < knob("MERLIN_IDLE_PERF_MIN_FREE_SLOTS"):
            return "cpus reserved for the admitting grade" if grading else "cpus busy"
        # Author requests need no gate here: the broker dispatches them first and they lease at the AUTHOR
        # class. The CPUs a queued request is about to take are not idle, so they are not offered.
        waiting = queued(channel)
        budget = free - waiting["author_cases"] - waiting["operator_cases"]
        if budget < knob("MERLIN_IDLE_PERF_MIN_FREE_SLOTS"):
            return "cpus promised to queued requests"
        cap = knob("MERLIN_IDLE_PERF_MAX_REQUESTS")
        live = waiting["operator_live"] - self.gsim.inflight(channel)  # the gSIM lane bounds its own jobs
        if cap and live >= cap:
            return "background requests in flight"
        if not scratch_retention.free_space_ok(self.run_dir):
            return "low disk"
        self._next = now + knob("MERLIN_IDLE_PERF_INTERVAL_S")
        verdicts = _verdicts(self.run_dir)
        cohort, l2 = _cohort_l2(verdicts, workspace=self.workspace)
        if not cohort:
            return "no performance cohort"
        digest = self._submission_digest()
        selection = _selection_identity()
        if selection != self._selection_sha256:
            self.requested.clear()
            self._early_dense_attempted.clear()
            self._selection_sha256 = selection
        known = estimates(channel, with_status=True, selection_sha256=selection)
        preferred = self._preferred(cohort)
        # Measured cycles of the largest passing scalability cases, within the campaign's gSIM share.
        from .idle_gsim import verdict_performance

        records = verdict_performance(verdicts)
        statuses = []
        timed = tuned = 0
        # Cohort order and source churn must not keep the largest dense cases
        # behind untimed siblings, gSIM, or repeated small tuning experiments.
        # Spend one existing AUXILIARY slot, once per selected timing epoch for
        # each of the two targets. All ordinary admission and timeout gates hold.
        early = (_largest_dense(self.workspace, cohort)
                 if len(self._early_dense_attempted) < _EARLY_DENSE_CASES else ())
        for name in early:
            if (l2.get(name) == "pass" and not any(len(row) < 4 or row[3] for row in known.get(name) or ())
                    and name not in self._early_dense_attempted
                    and (name, digest) not in self.requested):
                budget -= self._time(channel, [name], digest, records, why="early largest dense baseline")
                self._early_dense_attempted.add(name)
                timed, live = 1, live + 1
                break
        lane = self.gsim.tick(
            channel,
            cohort=cohort,
            l2=l2,
            known=known,
            digest=digest,
            free=budget,
            records=records,
            # Final grading keeps the same reserve an admitting grade does; ``free`` already excludes it
            # while one admits.
            final_reserve=0 if grading else grade_reserve(),
            preferred=preferred,
        )
        if lane:
            statuses.append(lane)
            budget -= 1
        # Retain comparisons from older model epochs too; only admission uses the current selection.
        historical = known if selection is None else estimates(channel, with_status=True)
        self._tuning(self.tuner.refresh, channel, digest, historical, cohort=cohort, l2=l2, progress_known=known)
        timing = select(cohort, l2, known, digest, self.requested, limit=len(cohort), preferred=preferred,
                        work=_declared_work(self.workspace, cohort))
        jobs = self._tuning(self.tuner.plan, channel, digest, cohort, l2, known, selection=selection) or []
        # Fill the free CPUs, alternating baseline timing and experiments (each in its own value order);
        # an experiment's missing baselines are timed beside it.
        timing_turn = self._timing_turn
        while budget >= knob("MERLIN_IDLE_PERF_MIN_FREE_SLOTS") and (timing or jobs) and not (cap and live >= cap):
            if (timing_turn and timing) or not jobs:
                size = min(knob("MERLIN_IDLE_PERF_MAX_CASES"), budget)
                if jobs and budget > 1:
                    size = min(size, max(1, budget // 2))
                from ..tools.selfcheck import _case_groups

                suite, names = _case_groups(self.workspace, timing[:size])[0]
                timing = [name for name in timing if name not in names]
                budget -= self._time(channel, names, digest, records, suite=suite)
                timed, live = timed + 1, live + 1
            else:
                spent = self._tuning(self._experiment, channel, jobs.pop(0), digest, known, budget, records)
                if spent:
                    budget -= spent
                    tuned += 1
                    live = queued(channel)["operator_live"] - self.gsim.inflight(channel)
                    timing = [name for name in timing if (name, digest) not in self.requested]
            timing_turn = not timing_turn
        self._timing_turn = timing_turn
        # Publish queued work immediately while retaining historical result identities.
        if tuned:
            self._tuning(self.tuner.refresh, channel, digest, historical, cohort=cohort, l2=l2, progress_known=known)
        if timed:
            statuses.append("submitted")
        if tuned:
            statuses.append("tuning submitted")
        if statuses:
            return "; ".join(statuses)
        return "estimates current; no untried tuning candidate" if not timing else "cpus busy"

    def _time(self, channel: Path, names: list[str], digest: str, records: dict, why: str = "",
              *, suite: str = "scalability") -> int:
        """Submit one Spike timing request of ``submission/`` for ``names``; returns the CPUs it asks for."""
        timeout, deferred = request_timeout(names, records)
        self._submit(channel, names, digest, timeout=timeout, suite=suite)
        self.requested.update((name, digest) for name in names)
        scratch_retention.say(
            f"[idle-perf] background Spike timing request for {len(names)} case(s): {', '.join(names)}"
            + (f" ({why})" if why else "")
            + (f" (timeout {timeout}s for observations a grade could not fit: {', '.join(deferred)})"
               if deferred else "")
        )
        return len(names)

    def _experiment(self, channel: Path, job: dict, digest: str, known: dict, budget: int, records: dict) -> int:
        """Start one proposed experiment sized to ``budget`` free CPUs, timing the baselines it needs
        beside it; returns the CPUs both ask for (0: it did not fit)."""
        candidate, cases = job["candidate"], list(job["cases"])

        def missing(name):
            if (name, digest) in self.requested:
                return False
            if candidate["kind"] == "knob":  # a knob copy is made of this digest: compare with it
                return not any(row[1] == digest for row in known.get(name) or ())
            return not known.get(name)

        size = len(cases)
        while size > 1 and size + sum(map(missing, cases[:size])) > budget:
            size -= 1
        names = cases[:size]
        need = [name for name in names if missing(name)]
        if not names or len(names) + len(need) > budget:
            return 0
        request_cap = knob("MERLIN_IDLE_PERF_MAX_REQUESTS")
        live = queued(channel)["operator_live"] - self.gsim.inflight(channel)
        if request_cap and live + 1 + bool(need) > request_cap:
            return 0
        from .idle_tuning import knob as tuning_knob

        cap = tuning_knob("MERLIN_IDLE_TUNE_MAX_REQUESTS")
        if cap and self.tuner.in_flight(channel) >= cap:
            return 0
        source_cap = tuning_knob("MERLIN_IDLE_TUNE_MAX_PER_SOURCE")
        if source_cap and sum(r["operator"] and (r["experiment"] or {}).get("base_sha256") == digest
                              for r in self.tuner._requests(channel)) >= source_cap:
            return 0
        if need:
            self._time(channel, need, digest, records, why=f"baseline for {candidate['key']}",
                       suite=job.get("suite", "scalability"))
        timeout, _ = request_timeout(names, records, base=tuning_knob("MERLIN_IDLE_TUNE_TIMEOUT_S"))
        self.tuner.submit(channel, job, names, digest, timeout=timeout)
        return len(names) + len(need)

    def _preferred(self, cohort) -> frozenset:
        """Bundle representatives, refreshed after source or completed numerical evidence changes."""
        from .family_rollup import preferred_cases, selfcheck_identity

        path = self.workspace / "qa" / "latest_completed_grade.json"
        try:
            stat = path.stat()
            identity = (stat.st_mtime_ns, stat.st_size, tuple(cohort))
        except OSError:
            identity = (None, None, tuple(cohort))
        identity += (selfcheck_identity(self.workspace),)
        if self._preferred_for != identity:
            self._preferred_for = identity
            self._preferred_cases = frozenset(preferred_cases(self.workspace, cohort))
        return self._preferred_cases

    @staticmethod
    def _tuning(method, *args, **kwargs):
        """A tuning step that can fail without stopping the timing that its baselines come from."""
        try:
            return method(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 -- tuning is advice; timing continues without it
            scratch_retention.say(f"[idle-tune] skipped: {type(error).__name__}: {str(error)[:200]}")
            return None

    def _submit(self, channel: Path, names: list[str], digest: str, timeout: int | None = None,
                *, suite: str = "scalability") -> str:
        return submit(
            channel, names, digest, origin=ORIGIN, prefix=REQUEST_PREFIX,
            timeout=knob("MERLIN_IDLE_PERF_TIMEOUT_S") if timeout is None else timeout,
            extra={"suite": suite, **({"tiers": "L0,L1,L2"} if suite == "core" else {})},
        )


def request_timeout(names, performance: dict, *, base: int | None = None) -> tuple[int, list[str]]:
    """``(timeout, deferred cases)`` of one background Spike timing request for ``names``.

    The knob's timeout covers ordinary cases. A case whose timed observation a grade could not fit
    (``observation_deferred``, with its predicted ``observation_need_s``) adds that need, so the request
    runs it to completion asynchronously instead of reporting it unobserved again; a joint campaign's
    remaining window bounds the sum."""
    base = knob("MERLIN_IDLE_PERF_TIMEOUT_S") if base is None else base
    deferred, extra = [], 0
    for name in names:
        record = performance.get(name) if isinstance(performance, dict) else None
        need = (record or {}).get("observation_need_s")
        if (record or {}).get("observation_deferred") and type(need) is int and need > 0:
            deferred.append(name)
            extra += need
    timeout = base + extra
    try:
        from .. import joint_control

        if joint_control.selected() is not None:
            timeout = max(base, min(timeout, int(joint_control.remaining(drain=True) or 0)))
    except (RuntimeError, OSError, ValueError):
        pass
    return timeout, deferred


def submit(
    channel: Path,
    names: list[str],
    digest: str,
    *,
    origin: str,
    prefix: str,
    timeout: int,
    extra: dict | None = None,
    rid: str | None = None,
) -> str:
    """Write one detached operator request (``suite: scalability``, Spike unless ``extra`` says) for ``names``.

    ``rid`` is given when the id was authorized before the request became visible (the gSIM lane)."""
    now_ns = time.time_ns()
    identity = json.dumps([digest, names, extra], sort_keys=True, default=str)
    if rid is None:
        rid = prefix + f"{now_ns}_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    elif not rid.startswith(prefix):
        raise ValueError("operator request id does not carry its lane prefix")
    body = {
        "protocol": 3,
        "request_id": rid,
        "origin": origin,
        "requested_at_unix_ns": now_ns,
        "deadline_unix_ns": now_ns + (timeout + 240) * 1_000_000_000,
        "submission": "submission",
        "submission_sha256": digest,
        "suite": "scalability",
        "sim": "spike",
        "capsules": ",".join(names),
        "workers": 0,
        "timeout": timeout,
        "shape_coverage": False,
        "offload_census": False,
        "model_layers": False,
        "candidate_diagnostics": False,
        "compile_only": False,
        "scale_probe": False,
        "dynamic_census": False,
        "caller_layout": "",
        **(extra or {}),
    }
    # No author process waits on this request: the detached marker is its lease, as for an author
    # client that returned early, and it is written before the request becomes visible.
    (channel / f"detached_{rid}").write_text(json.dumps({"origin": origin, "at_unix_ns": now_ns}))
    staged = channel / f".{rid}.staged"
    staged.write_text(json.dumps(body))
    staged.replace(channel / f"req_{rid}.json")
    return rid
