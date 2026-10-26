"""Use idle reserved CPUs to keep the performance evidence of passing performance-cohort cases current.

Between grades the reserved CPUs sat near idle (a 20-CPU run averaged about 10) while the performance
ranking the author works from was built on estimates of older source, or on none: the timing model
only runs when the author asks for a scalability self-check, and the largest cases were rarely asked.
This module spends that idle time on exactly those estimates.

When no author request is waiting, shared CPU slots are free (beyond a small reserve while a grade is
admitting, so a starting grade still finds slots) and the volume has room, it submits an operator
request through the self-check broker for passing performance-cohort cases, sized from free slots
and a cap, whose estimate does not describe the current submission and model. More requests may
follow while capacity stays free (see :data:`KNOBS`). It is the same request an author self-check
makes (``suite: scalability``,
Spike only), so custody, snapshotting and response binding are identical; the broker runs it at the
advisory :data:`~merlin.common.cpu_pool.AUXILIARY` lease class and dispatches it after every author
request, so author requests and grades lease CPUs first; it adds no wait or lock to an existing path.
The request and its response carry ``origin: operator_idle_performance`` and an
``op_idle_`` id. It certifies nothing and starts no promotion: the scalability suite has no cert tier.

In a joint campaign the same gates also admit the gSIM lane (:mod:`.idle_gsim`, policy section
``gsim_measurement``): measured cycles of passing scalability cases, largest headroom first, within an
explicit share of the campaign's gSIM budget; it is tried before the Spike requests of a tick.

Once every passing cohort case has an estimate of the current source, the same gates and budget run
bounded tuning experiments instead (:mod:`.idle_tuning`, ``MERLIN_IDLE_TUNING=0`` disables only those).

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
    "MERLIN_IDLE_PERF_MAX_REQUESTS": 4,  # ceiling on background requests in flight at once
    "MERLIN_IDLE_PERF_INTERVAL_S": 60,  # seconds between two submissions
    "MERLIN_IDLE_PERF_CHECK_S": 30,  # seconds between two evaluations of the gates
    "MERLIN_IDLE_PERF_TIMEOUT_S": 1800,  # client timeout each background request declares
    # Slots left free while a grade is admitting (0: a fifth of the pool, at least 2). An editing
    # author keeps some grade admitting almost all the time, so waiting for none never measured.
    "MERLIN_IDLE_PERF_GRADE_RESERVE_SLOTS": 0,
}
_MAX_RECORD_BYTES = 64 << 20
_VERDICTS_READ = 6


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


def _verdicts(run_dir: Path) -> list[dict]:
    paths = sorted(
        (Path(run_dir) / "qa_history").glob("verdict_fast_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    documents = (_load(path) for path in paths[:_VERDICTS_READ])
    return [d for d in documents if isinstance(d, dict)]


def _cohort_l2(verdicts: list[dict]) -> tuple[list[str], dict]:
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
    return cohort or [], status


def estimates(channel: Path, *, with_status: bool = False, selection_sha256: str | None = None) -> dict:
    """``{case: [(graded_at_ns, submission_sha256, performance)]}`` from submission self-check responses.

    Variant responses (and knob-override copies, which carry a ``variant`` block) never describe
    ``submission/``. ``with_status`` appends whether the row passed its numerical check.
    An explicit selection pin excludes older model epochs without deleting their evidence.
    """
    found: dict[str, list] = {}
    for path in channel.glob("resp_*.json"):
        document = _load(path)
        if (
            not isinstance(document, dict)
            or document.get("variant")
            or document.get("selfcheck_submission") not in (None, "submission")
        ):
            continue
        digest, graded = document.get("submission_sha256"), document.get("graded_at_unix_ns") or 0
        for row in document.get("cases") or ():
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


def select(cohort, l2, known: dict, digest: str, requested: set, *, limit: int) -> list[str]:
    """Passing cohort cases without an estimate of ``digest``.

    Cases never timed and cases timed on older source alternate, so the largest stale estimates are
    refreshed without starving the cases that have none (the source digest moves with every edit).
    Stale cases go largest previous estimate first; untimed ones in cohort order.
    """
    eligible = [
        name
        for name in cohort
        if l2.get(name) == "pass"
        and (name, digest) not in requested
        and not any(row[1] == digest for row in known.get(name) or ())
    ]
    untimed = [name for name in eligible if not known.get(name)]
    stale = sorted(
        (name for name in eligible if known.get(name)),
        key=lambda name: (-(known[name][0][2].get("estimated_whole_soc_cycles") or 0), name),
    )
    chosen = []
    while (untimed or stale) and len(chosen) < limit:
        for group in (untimed, stale):
            if group and len(chosen) < limit:
                chosen.append(group.pop(0))
    return chosen


class IdleMeasurer:
    """One per background grader; :meth:`tick` is cheap until every gate opens."""

    def __init__(self, workspace, run_dir, *, digest=None, clock=time.monotonic):
        self.workspace, self.run_dir = Path(workspace), Path(run_dir)
        self._digest = digest
        self._clock = clock
        self._next = 0.0
        self._next_check = 0.0
        self.requested: set[tuple[str, str]] = set()
        self._selection_sha256: str | None = None
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
        """Submit at most one background request; returns why it did or did not. Never raises."""
        try:
            return self._tick(grading)
        except Exception as error:  # noqa: BLE001 -- background measurement never reaches the grader
            self._next = self._clock() + knob("MERLIN_IDLE_PERF_INTERVAL_S")
            scratch_retention.say(f"[idle-perf] skipped: {type(error).__name__}: {str(error)[:200]}")
            return "error"

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
        author, operator = outstanding(channel)
        if author:
            return "author request waiting"
        operator -= self.gsim.inflight(channel)  # the gSIM lane bounds its own jobs
        free = free_slots()
        if free is None or os.environ.get("MERLIN_SELFCHECK_BROKER_JOBS", "").strip() == "1":
            return "single-lane broker"  # one lane would queue the next author request behind this work
        # Leases are not preemptive: while a grade admits, use only the slots beyond its reserve.
        free -= grade_reserve() if grading else 0
        if free < knob("MERLIN_IDLE_PERF_MIN_FREE_SLOTS"):
            return "cpus reserved for the admitting grade" if grading else "cpus busy"
        cases = min(knob("MERLIN_IDLE_PERF_MAX_CASES"), free)
        if operator >= min(knob("MERLIN_IDLE_PERF_MAX_REQUESTS"), max(1, free // cases)):
            return "background requests in flight"
        if not scratch_retention.free_space_ok(self.run_dir):
            return "low disk"
        self._next = now + knob("MERLIN_IDLE_PERF_INTERVAL_S")
        verdicts = _verdicts(self.run_dir)
        cohort, l2 = _cohort_l2(verdicts)
        if not cohort:
            return "no performance cohort"
        digest = self._submission_digest()
        selection = _selection_identity()
        if selection != self._selection_sha256:
            self.requested.clear()
            self._selection_sha256 = selection
        known = estimates(channel, with_status=True, selection_sha256=selection)
        # Measured cycles of the largest passing scalability cases, within the campaign's gSIM share.
        from .idle_gsim import verdict_performance

        lane = self.gsim.tick(
            channel,
            cohort=cohort,
            l2=l2,
            known=known,
            digest=digest,
            free=free,
            records=verdict_performance(verdicts),
            # Final grading keeps the same reserve an admitting grade does; ``free`` already excludes it
            # while one admits.
            final_reserve=0 if grading else grade_reserve(),
        )
        if lane:
            return lane
        self._tuning(self.tuner.refresh, channel, digest, known)  # results of an older source are dropped
        names = select(cohort, l2, known, digest, self.requested, limit=cases)
        if not names:  # every estimate is current: the same idle capacity runs tuning experiments
            return self._tuning(self.tuner.tick, channel, digest, cohort, l2, known, free=free) or "error"
        timeout, deferred = request_timeout(names, verdict_performance(verdicts))
        self._submit(channel, names, digest, timeout=timeout)
        self.requested.update((name, digest) for name in names)
        scratch_retention.say(
            f"[idle-perf] background Spike timing request for {len(names)} case(s): {', '.join(names)}"
            + (f" (timeout {timeout}s for observations a grade could not fit: {', '.join(deferred)})"
               if deferred else "")
        )
        return "submitted"

    @staticmethod
    def _tuning(method, *args, **kwargs):
        """A tuning step that can fail without stopping the timing that its baselines come from."""
        try:
            return method(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 -- tuning is advice; timing continues without it
            scratch_retention.say(f"[idle-tune] skipped: {type(error).__name__}: {str(error)[:200]}")
            return None

    def _submit(self, channel: Path, names: list[str], digest: str, timeout: int | None = None) -> str:
        return submit(
            channel, names, digest, origin=ORIGIN, prefix=REQUEST_PREFIX,
            timeout=knob("MERLIN_IDLE_PERF_TIMEOUT_S") if timeout is None else timeout,
        )


def request_timeout(names, performance: dict) -> tuple[int, list[str]]:
    """``(timeout, deferred cases)`` of one background Spike timing request for ``names``.

    The knob's timeout covers ordinary cases. A case whose timed observation a grade could not fit
    (``observation_deferred``, with its predicted ``observation_need_s``) adds that need, so the request
    runs it to completion asynchronously instead of reporting it unobserved again; a joint campaign's
    remaining window bounds the sum."""
    base = knob("MERLIN_IDLE_PERF_TIMEOUT_S")
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
