"""Bounded idle-time gSIM measurement of passing scalability cases: policy, budget and admission.

The joint policy refuses every author gSIM request for the scalability suite, so the largest shapes --
the cases holding most of the cycle headroom -- only ever had Spike timing-model estimates, while the
calibration pairs, the headroom ranking and the console's measured column were built from small
witnesses. This module lets the operator's idle-capacity measurer (:mod:`.feedback.idle_gsim`) run a
bounded sample of those cases on gSIM:

* the campaign policy section ``gsim_measurement`` (absent: every default applies) sets the lane's
  share of the campaign gSIM worker budget and concurrency; nothing here changes the ``gsim`` or
  ``gsim_correctness`` budgets, their rosters or their ledgers;
* the measurer *authorizes* one request id at a time in a run-side ledger the author cannot write
  (``joint_gsim_measurement_budget.json`` beside ``joint_campaign.json``), reserving the request's
  whole timeout against the lane budget before the request is visible; the self-check broker forwards a
  scalability gSIM request only for such an id, and the evaluation child re-checks the ledger before any
  simulator starts (:func:`admission`), so a forged or replayed request is refused exactly as before;
* the per-job timeout is derived from the case's cycles and the gSIM throughput of this run's own
  recorded simulator jobs (:func:`throughput`, :func:`job_timeout`). The cycles are the case's modeled
  estimate; a case without one is predicted from its structural floor (:func:`structural_floor`: compute
  floor or ideal request service, from the target facts) times the inefficiency this run's measured gSIM
  rows show over their floors (:func:`inefficiency`, :func:`predicted_cycles`). A run with too few
  recorded jobs measures nothing rather than guess a duration;
* concurrency follows the idle capacity seen at each submission (free shared slots beyond an admitting
  grade's reserve), within the gSIM slots the correctness and witness jobs leave and the budget share.

Results are advisory performance evidence: they certify nothing, satisfy no physical obligation, start
no promotion and never enter ``qa/tier_state.json``.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

SECTION = "gsim_measurement"
#: Set by the self-check broker in the evaluation child it starts for an authorized request id.
ENV = "MERLIN_JOINT_GSIM_MEASUREMENT_REQUEST"
ORIGIN = "operator_idle_gsim_measurement"
#: Starts with :data:`.feedback.idle_performance.REQUEST_PREFIX`, so the broker serves it after author work.
REQUEST_PREFIX = "op_idle_gsim_"
LEDGER = "joint_gsim_measurement_budget.json"
#: Policy knobs (``gsim_measurement`` in the campaign policy) and their defaults. ``None`` is derived.
DEFAULTS = {
    "enabled": True,
    # Share of the campaign gSIM worker-seconds budget (gsim.worker_budget_s) the lane may reserve, and
    # of the idle capacity seen at a submission it may hold.
    "worker_budget_share": 0.25,
    # Ceiling on concurrent measurement jobs. None: derived at each submission -- the share of the free
    # idle shared slots beyond an admitting grade's reserve (lane jobs included), within the gSIM slots the
    # correctness and witness jobs leave and the share's remaining worker-seconds.
    "max_jobs": None,
    # Quantile of observed/predicted runtime over recorded jobs, used as the timeout margin.
    "timeout_quantile": 0.95,
    # Recorded gSIM jobs needed before a throughput is derived (two parameters are fitted).
    "min_observations": 4,
    # Long measurements in flight at once (0: none). A case whose timeout does not fit before the drain
    # boundary may still run when its expected runtime ends before the final-model window; its timeout
    # then reaches at most the campaign's end, and it runs into the final window only on slots beyond
    # the grade reserve.
    "long_max_jobs": 1,
    # Wall seconds one measurement may take (:mod:`.gsim_wall_cap`); None: no cap. A job projected beyond
    # it is not started, and every job's timeout is held to it.
    "max_job_wall_s": None,
}
#: Why :func:`job_plan` skips a job no run of the selected engine can complete (the reason starts with it).
CEILING_SKIP = "expected runtime exceeds the selected gSIM process ceiling"
#: Seconds a request may wait to be claimed beyond its timeout (the broker's own request lease).
CLAIM_GRACE_S = 240
_STATUSES_ACTIVE = ("authorized", "running")
_MAX_FILE_BYTES = 16 << 20


def validate(section) -> None:
    """Refuse a malformed ``gsim_measurement`` section; an absent one uses :data:`DEFAULTS`."""
    if section is None:
        return
    if not isinstance(section, dict) or set(section) - set(DEFAULTS):
        raise ValueError("joint gSIM measurement policy must be an object of known knobs")
    if "enabled" in section and type(section["enabled"]) is not bool:
        raise ValueError("invalid joint gSIM measurement enabled")
    share = section.get("worker_budget_share", DEFAULTS["worker_budget_share"])
    if type(share) not in (int, float) or not 0 < share <= 1:
        raise ValueError("invalid joint gSIM measurement worker_budget_share")
    jobs = section.get("max_jobs")
    if jobs is not None and (type(jobs) is not int or not 1 <= jobs <= 64):
        raise ValueError("invalid joint gSIM measurement max_jobs")
    quantile = section.get("timeout_quantile", DEFAULTS["timeout_quantile"])
    if type(quantile) not in (int, float) or not 0.5 <= quantile <= 1:
        raise ValueError("invalid joint gSIM measurement timeout_quantile")
    observations = section.get("min_observations", DEFAULTS["min_observations"])
    if type(observations) is not int or not 2 <= observations <= 10000:
        raise ValueError("invalid joint gSIM measurement min_observations")
    long_jobs = section.get("long_max_jobs", DEFAULTS["long_max_jobs"])
    if type(long_jobs) is not int or not 0 <= long_jobs <= 8:
        raise ValueError("invalid joint gSIM measurement long_max_jobs")
    from .gsim_wall_cap import validate as validate_cap

    validate_cap(section.get("max_job_wall_s"))


def _slot_capacity() -> int | None:
    try:
        from merlin.targetgen.rtl_engine_policy import CAPSULE_WORKER_CAP

        value = CAPSULE_WORKER_CAP.get("gsim")
        return value if type(value) is int and value > 0 else None
    except Exception:  # noqa: BLE001 -- an unknown capacity leaves only the campaign job cap
        return None


def settings(policy: dict) -> dict:
    """The lane's effective knobs for ``policy`` (the campaign policy), with every derivation named."""
    section = {**DEFAULTS, **((policy or {}).get(SECTION) or {})}
    gsim = (policy or {}).get("gsim") or {}
    share = float(section["worker_budget_share"])
    capacity = _slot_capacity()
    budget = gsim.get("worker_budget_s") if type(gsim.get("worker_budget_s")) is int else 0
    final_window = (policy or {}).get("final_model_s") if type((policy or {}).get("final_model_s")) is int else 0
    return {
        "enabled": bool(section["enabled"]),
        "worker_budget_share": share,
        "budget_s": int(share * budget),
        "max_jobs": section["max_jobs"],
        "timeout_quantile": float(section["timeout_quantile"]),
        "min_observations": int(section["min_observations"]),
        "long_max_jobs": int(section["long_max_jobs"]),
        "max_job_wall_s": section["max_job_wall_s"],
        "final_window_s": final_window,
        "gsim_slot_capacity": capacity,
        "basis": {
            "long_horizon": f"a long job's expected runtime must end before the final-model window "
            f"(policy final_model_s {final_window}); its timeout reaches at most the campaign's end",
            "final_window": "a job still running in the final-model window starts only on free slots beyond "
            "the grade reserve, as an admitting grade's reserve is kept",
            "budget_s": f"worker_budget_share x gsim.worker_budget_s ({budget})",
            "max_jobs": "policy ceiling"
            if section["max_jobs"]
            else f"at each submission: worker_budget_share x (lane jobs in flight + free idle shared slots beyond the "
            f"grade reserve), at least 1, at most the gSIM slot capacity ({capacity}) less live correctness/witness "
            "gSIM jobs; the reservations of all jobs stay within budget_s",
            "timeout": "runtime margin x (setup + Spike tier seconds + cycles / recorded gSIM cycles per second); "
            "cycles = the modeled estimate, else overhead + inefficiency x structural floor fitted on "
            "this run's gSIM rows (upper quantile), else the worst observed cycles / floor ratio",
        },
        "advisory_only": True,
        "certification_authority": False,
    }


def _state():
    from . import joint_control

    return joint_control.selected()


def guidance() -> str:
    """One author-facing sentence when the selected campaign runs this lane, else ``""``."""
    try:
        state = _state()
    except (OSError, ValueError):
        return ""
    if state is None or not settings(state.get("policy") or {})["enabled"]:
        return ""
    return (
        "On idle capacity the operator measures passing scalability cases on gSIM, largest headroom "
        "first, within its own budget; read those cycles as advisory measured evidence that certifies "
        "nothing."
    )


def ledger_path(state=None) -> Path:
    from . import joint_control

    state = _state() if state is None else state
    explicit = (state or {}).get("gsim_measurement_ledger")
    return Path(explicit) if explicit else Path(os.environ[joint_control.ENV]).with_name(LEDGER)


def _alive(pid) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _settle(data: dict, now: float) -> None:
    """Expire unclaimed authorizations (full refund) and mark dead running jobs lost (charge kept)."""
    for job in data["jobs"].values():
        if job.get("status") == "authorized" and now > job.get("claim_deadline_epoch", 0):
            job.update(status="expired", outcome="not_started", finished_at_epoch=now)
            data["reserved_s"] = max(0.0, data["reserved_s"] - job.get("timeout_s", 0))
        elif job.get("status") == "running" and not _alive(job.get("pid")):
            # A killed worker keeps its reservation, as the other gSIM ledgers do.
            job.update(status="lost", outcome="lost", finished_at_epoch=now)


@contextmanager
def _ledger(path: Path | None = None):
    path = ledger_path() if path is None else path
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        data = json.loads(raw) if raw.strip() else {"version": 1, "reserved_s": 0.0, "jobs": {}}
        if data.get("version") != 1 or not isinstance(data.get("jobs"), dict) or len(data["jobs"]) > 65536:
            raise ValueError("invalid joint gSIM measurement ledger")
        _settle(data, time.time())
        yield data
        handle.seek(0)
        handle.truncate()
        json.dump(data, handle, sort_keys=True)
        handle.flush()


def read(path: Path | None = None) -> dict:
    """The ledger as it stands (empty when absent); never written."""
    path = ledger_path() if path is None else path
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"version": 1, "reserved_s": 0.0, "jobs": {}}
    return (
        data
        if isinstance(data, dict) and isinstance(data.get("jobs"), dict)
        else {"version": 1, "reserved_s": 0.0, "jobs": {}}
    )


def _other_physical_active(state) -> int:
    """Live gSIM jobs of the campaign's correctness and witness ledgers (their own capacity first)."""
    from . import joint_control

    count = 0
    for path in (joint_control.gsim_ledger(), joint_control.correctness_ledger()):
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        for entry in data.get("active") or ():
            pid = entry.get("pid") if isinstance(entry, dict) else entry
            count += _alive(pid)
    return count


def authorize(
    rid: str,
    capsule: str,
    *,
    timeout_s: int,
    executable: str | None,
    modeled_cycles,
    free_slots: int | None = None,
    long: bool = False,
    now: float | None = None,
    engine_bound: dict | None = None,
    throughput: dict | None = None,
    wall: dict | None = None,
) -> str | None:
    """Reserve ``timeout_s`` of the lane budget for request ``rid``; ``None`` on success, else why not.

    ``free_slots`` is the idle measurer's count of free shared slots beyond an admitting grade's reserve
    at this submission (``None``: no shared pool is selected, only the slot capacity bounds the lane).
    A ``long`` job's timeout must fit before the campaign's end rather than its drain boundary, and at
    most ``long_max_jobs`` of them are in flight (the lane decides which cases qualify,
    :mod:`.feedback.idle_gsim`). ``engine_bound`` (:func:`.gsim_engine_bound.engine_cycle_bound`) is the
    simulated-cycle bound derived from ``timeout_s``; the evaluation child applies it to its gSIM runs.
    """
    from . import joint_control

    state = _state()
    if state is None:
        return "no joint campaign"
    knobs = settings(state["policy"])
    if not knobs["enabled"]:
        return "gSIM measurement disabled by policy"
    if not rid.startswith(REQUEST_PREFIX) or type(timeout_s) is not int or timeout_s < 1:
        return "invalid measurement request"
    try:
        if joint_control.stop_reason() is not None:
            return "campaign window closed"
        if (joint_control.remaining(drain=not long) or 0) < timeout_s:
            return "remaining campaign window shorter than the derived timeout"
    except RuntimeError:
        return "campaign clock not started"
    if long and knobs["long_max_jobs"] < 1:
        return "long measurements disabled by policy"
    now = time.time() if now is None else now
    capacity = knobs["gsim_slot_capacity"]
    with _ledger() as data:
        live = [job for job in data["jobs"].values() if job.get("status") in _STATUSES_ACTIVE]
        active = len(live)
        if knobs["max_jobs"] and active >= knobs["max_jobs"]:
            return "measurement jobs in flight"
        if long and sum(bool(job.get("long")) for job in live) >= knobs["long_max_jobs"]:
            return "long measurement in flight"
        if free_slots is not None:
            if free_slots < 1:
                return "no free idle slot"
            # The lane holds at most its share of the idle capacity (its own jobs count as idle), so
            # Spike timing and tuning keep the rest.
            if active >= max(1, math.floor(knobs["worker_budget_share"] * (active + free_slots))):
                return "measurement jobs in flight"
        if capacity and active + _other_physical_active(state) >= capacity:
            return "gSIM slots held by correctness or witness work"
        if data["reserved_s"] + timeout_s > knobs["budget_s"]:
            return "gSIM measurement budget exhausted"
        if rid in data["jobs"]:
            return "request id already authorized"
        data["jobs"][rid] = {
            "capsule": capsule,
            "executable": executable,
            "modeled_cycles": modeled_cycles,
            "timeout_s": timeout_s,
            "long": bool(long),
            "engine_max_cycles": (engine_bound or {}).get("max_cycles"),
            "engine_bound_basis": (engine_bound or {}).get("basis"),
            # The wall predictor's inputs and projection; the measured outcome later refits it.
            "wall_features": (wall or {}).get("features"),
            "predicted_wall_s": (wall or {}).get("predicted_s"),
            "status": "authorized",
            "authorized_at_epoch": now,
            "claim_deadline_epoch": now + timeout_s + CLAIM_GRACE_S,
        }
        data["reserved_s"] += timeout_s
        data["settings"] = knobs
        if isinstance(throughput, dict):
            # The rate this run measured, for projecting author requests against the wall cap.
            data["throughput"] = throughput
    return None


def broker_admission(rid: str, request: dict) -> dict | None:
    """The authorization of a scalability gSIM request, or ``None`` (the request stays refused)."""
    if (
        _state() is None
        or not isinstance(request, dict)
        or not rid.startswith(REQUEST_PREFIX)
        or request.get("origin") != ORIGIN
        or request.get("suite") != "scalability"
        or request.get("sim") != "gsim"
    ):
        return None
    try:
        job = read().get("jobs", {}).get(rid)
    except (OSError, KeyError, ValueError):
        return None
    if (
        not isinstance(job, dict)
        or job.get("status") != "authorized"
        or job.get("capsule") != request.get("capsules")
        or time.time() > job.get("claim_deadline_epoch", 0)
    ):
        return None
    return dict(job)


def requested() -> str | None:
    """The authorized request id this evaluation child was started for, if any."""
    value = os.environ.get(ENV, "").strip()
    return value if value.startswith(REQUEST_PREFIX) else None


@contextmanager
def admission(capsule):
    """Claim this child's authorization before any simulator starts; yields its timeout seconds."""
    from . import joint_control

    rid = requested()
    if rid is None or _state() is None:
        raise ValueError("gSIM request is outside the selected small witness roster/window")
    if joint_control.stop_reason() is not None:
        raise ValueError("gSIM measurement outside the campaign window")
    begun = time.monotonic()
    with _ledger() as data:
        job = data["jobs"].get(rid)
        if not isinstance(job, dict) or job.get("status") != "authorized" or job.get("capsule") != capsule:
            raise ValueError("gSIM measurement request is not authorized")
        job.update(status="running", pid=os.getpid(), started_at_epoch=time.time())
        timeout, long = job["timeout_s"], bool(job.get("long"))
    try:
        # A long job may run past the drain boundary up to the campaign's end; a normal one may not.
        yield max(1, min(timeout, int(joint_control.remaining(drain=not long) or 0)))
    finally:
        elapsed = time.monotonic() - begun
        with _ledger() as data:
            job = data["jobs"].get(rid) or {}
            if job.get("status") == "running" and job.get("pid") == os.getpid():
                job.update(status="done", finished_at_epoch=time.time(), elapsed_s=round(elapsed, 1))
                job.setdefault("outcome", "timeout" if elapsed >= timeout else "no_result")
                data["reserved_s"] = max(0.0, data["reserved_s"] - max(0.0, timeout - elapsed))


def long_request(rid: str, *, running: bool = False) -> bool:
    """Whether ``rid`` is a long measurement the lane authorized that has not finished (never raises);
    with ``running``, only one whose evaluation child has claimed it and is alive."""
    if not str(rid).startswith(REQUEST_PREFIX):
        return False
    try:
        job = read().get("jobs", {}).get(rid)
    except Exception:  # noqa: BLE001 -- no campaign or no ledger: an ordinary request
        return False
    if not isinstance(job, dict) or not job.get("long"):
        return False
    if running:
        return job.get("status") == "running" and _alive(job.get("pid"))
    return job.get("status") in _STATUSES_ACTIVE


def shared_pool_contended() -> bool:
    """Whether some process waits for a shared compute slot while none is free.

    A long measurement that outlives the broker's stop yields to this: after the stop the waiter is
    final grading (or another operator check), and a measurement must never hold the slot it needs.
    ``False`` when no shared pool is selected or its queue cannot be read (no one to yield to).
    """
    try:
        from merlin.common import cpu_pool

        selection = cpu_pool._shared_compute_selection(os.environ)
        if selection is None:
            return False
        cpus, root = selection
        if not Path(root).is_dir() or cpu_pool._free_cpus(root, cpus, fresh=True):
            return False
        return bool(cpu_pool._queued_keys(Path(root) / cpu_pool._WAITER_DIR))
    except Exception:  # noqa: BLE001 -- an unreadable queue never stops a measurement by itself
        return False


#: What an evaluation child may record about its run: the measured outcome, and how its gSIM engine runs
#: went (:mod:`.gsim_engine_bound`), which later jobs derive their simulated-cycle bound from.
_RESULT_KEYS = (
    "outcome",
    "barrier_cycles",
    "numerics_passed",
    "execution_digest",
    "elf_sha256",
    "engine_runs",
    "engine_stop",
    "engine_timeout_ceiling_s",
)


def note_result(rid: str, result: dict) -> None:
    """Record the measured outcome of the running request ``rid`` (cycles, numerics); never raises."""
    try:
        with _ledger() as data:
            job = data["jobs"].get(rid)
            if isinstance(job, dict) and job.get("status") == "running":
                job.update(
                    {
                        key: value
                        for key, value in result.items()
                        if key in _RESULT_KEYS
                    }
                )
    except (OSError, ValueError, KeyError):
        pass


# --- throughput and timeouts --------------------------------------------------------------------


def _load(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def observations(channel: Path) -> list[tuple[float, float]]:
    """``(cycles, wall seconds)`` of this run's recorded single-case gSIM jobs.

    Simulator jobs: ``simtiming_<id>.json`` (the broker's worker runtime) joined with ``simresp_<id>.json``
    (the gSIM row's barrier cycles). Measurement responses (``resp_op_idle_gsim_*``) add their own
    start/finish times. A job without both numbers is skipped, never estimated.
    """
    channel = Path(channel)
    found: list[tuple[float, float]] = []
    for timing in channel.glob("simtiming_*.json"):
        record = _load(timing)
        response = _load(channel / ("simresp_" + timing.name[len("simtiming_") :]))
        if not isinstance(record, dict) or not isinstance(response, dict) or response.get("sim") != "gsim":
            continue
        rows = [r for r in response.get("per_capsule") or () if isinstance(r, dict)]
        seconds = record.get("worker_runtime_s")
        cycles = rows[0].get("barrier_cycles") if len(rows) == 1 else None
        if (
            type(cycles) is int
            and cycles > 0
            and type(seconds) in (int, float)
            and seconds > 0
            and rows[0].get("barrier_tier") == "L3"
        ):
            found.append((float(cycles), float(seconds)))
    for path in channel.glob(f"resp_{REQUEST_PREFIX}*.json"):
        document = _load(path)
        if not isinstance(document, dict) or document.get("sim") != "gsim":
            continue
        start, end = document.get("grade_started_at_unix_ns"), document.get("graded_at_unix_ns")
        rows = [r for r in document.get("per_capsule") or () if isinstance(r, dict)]
        cycles = rows[0].get("barrier_cycles") if len(rows) == 1 else None
        if type(start) is int and type(end) is int and end > start and type(cycles) is int and cycles > 0:
            found.append((float(cycles), (end - start) / 1e9))
    return sorted(found)


def _affine(points, *, quantile: float):
    """``y = intercept + slope * x`` over ``[(x, y)]`` sorted by x, for extrapolating to large x.

    ``intercept`` is the median y of the smaller half; ``slope`` is the least-squares slope of the larger
    half's y beyond the intercept through the origin (``sum(x * dy) / sum(x * x)``), so the largest
    points -- the ones nearest the large cases the fit is extrapolated to -- weigh most. A per-point
    median, or a ratio of sums, was set by the many points whose y was nearly all intercept and noise.
    ``margin`` is the ``quantile`` of observed / predicted y over the larger half, at least 1. ``None``
    when no point of the larger half lies above the intercept.
    """
    points = sorted(points)
    half = len(points) // 2
    intercept = statistics.median(y for _, y in points[: max(1, half)])
    beyond = [(x, y - intercept) for x, y in points[half:] if y > intercept and x > 0]
    if not beyond:
        return None
    slope = sum(x * extra for x, extra in beyond) / sum(x * x for x, _ in beyond)
    ratios = sorted(y / (intercept + slope * x) for x, y in points[half:])
    index = min(len(ratios) - 1, max(0, math.ceil(quantile * len(ratios)) - 1))
    return intercept, slope, max(1.0, ratios[index])


def throughput(channel: Path, *, min_observations: int, quantile: float) -> dict | None:
    """A two-parameter fit ``seconds = setup + cycles / cycles_per_s`` of the recorded gSIM jobs
    (:func:`_affine` of seconds over cycles). ``None`` when fewer than ``min_observations`` jobs, or no
    job ran measurably longer than the setup, were recorded."""
    points = observations(channel)
    if len(points) < max(2, min_observations):
        return None
    fit = _affine(points, quantile=quantile)
    if fit is None:
        return None
    setup, seconds_per_cycle, margin = fit
    return {
        "setup_s": round(setup, 2),
        "cycles_per_s": round(1 / seconds_per_cycle, 3),
        "margin": round(margin, 3),
        "observations": len(points),
        "quantile": quantile,
    }


def structural_floor(roofline) -> tuple[float | None, str | None]:
    """The case's cycle floor: the largest of its compute floor (useful MACs / peak MACs per cycle) and
    its ideal read/write request service (logical bytes / the target's request bytes per cycle).

    The roofline the timing model attaches carries both terms already derived from the target's facts;
    a term it states only as its inputs is recomputed from them (a missing request rate is taken from
    the other direction of the same roofline). ``(None, None)`` when nothing positive is stated.
    """
    from .feedback.performance_headroom import _bound, _positive

    roofline = roofline if isinstance(roofline, dict) else {}
    bound, limiter = _bound(roofline)
    terms = {limiter: bound} if bound is not None else {}
    useful, peak = roofline.get("useful_macs"), roofline.get("structural_peak_macs_per_cycle")
    if "compute" not in terms and _positive(useful) and _positive(peak):
        terms["compute"] = useful / peak
    service = roofline.get("logical_request_ideal_service") or {}
    sides = {side: value for side, value in service.items() if isinstance(value, dict)}
    rates = [
        v["requested_bytes"] / v["ideal_requested_service_cycles"]
        for v in sides.values()
        if _positive(v.get("requested_bytes")) and _positive(v.get("ideal_requested_service_cycles"))
    ]
    for side, value in sides.items():
        if f"{side} requests" not in terms and _positive(value.get("requested_bytes")) and rates:
            terms[f"{side} requests"] = value["requested_bytes"] / max(rates)
    terms = {name: value for name, value in terms.items() if name and _positive(value)}
    if not terms:
        return None, None
    name = max(terms, key=terms.get)
    return terms[name], name


def _bound_class(limiter) -> str:
    return "compute" if limiter == "compute" else "requests"


def bound_observations(channel: Path) -> list[tuple[float, float, str]]:
    """``(cycle floor, measured gSIM cycles, bound class)`` of this run's passing gSIM rows that state a
    structural roofline: the simulator responses and the lane's own measurements."""
    channel = Path(channel)
    found = []
    paths = [*channel.glob("simresp_*.json"), *channel.glob(f"resp_{REQUEST_PREFIX}*.json")]
    for path in paths:
        document = _load(path)
        if not isinstance(document, dict) or document.get("sim") != "gsim":
            continue
        for row in document.get("per_capsule") or ():
            if not isinstance(row, dict) or row.get("pass") is False or row.get("barrier_tier") != "L3":
                continue
            cycles = row.get("barrier_cycles")
            floor, limiter = structural_floor((row.get("performance") or {}).get("structural_roofline"))
            if type(cycles) is int and cycles > 0 and floor is not None:
                found.append((float(floor), float(cycles), _bound_class(limiter)))
    return found


def inefficiency(channel: Path, *, min_observations: int, quantile: float) -> dict | None:
    """How far measured gSIM cycles sit above the structural floor, from this run's own rows.

    Per bound class (compute, requests) with at least ``min_observations`` rows, else over all rows: an
    :func:`_affine` fit ``cycles = overhead + factor * floor`` (``factor`` is the inefficiency, ``margin``
    its upper quantile). With fewer rows than that, the worst observed cycles / floor ratio stands in for
    both (no overhead). ``None`` when no measured row states a floor.
    """
    points = bound_observations(channel)
    if not points:
        return None
    model = {
        "observations": len(points),
        "worst_ratio": round(max(c / f for f, c, _ in points), 3),
        "quantile": quantile,
        "classes": {},
    }
    groups = {"all": points, **{k: [p for p in points if p[2] == k] for k in ("compute", "requests")}}
    for name, group in groups.items():
        if len(group) >= max(2, min_observations):
            fit = _affine([(f, c) for f, c, _ in group], quantile=quantile)
            if fit is not None:
                model["classes"][name] = {
                    "overhead_cycles": round(fit[0], 1),
                    "factor": round(fit[1], 3),
                    "margin": round(fit[2], 3),
                    "observations": len(group),
                }
    return model


def predicted_cycles(model: dict | None, floor, limiter) -> dict | None:
    """``{cycles, upper_cycles, basis}`` for a case of cycle floor ``floor`` without a modeled estimate."""
    if not isinstance(model, dict) or type(floor) not in (int, float) or not floor > 0:
        return None
    klass = _bound_class(limiter)
    fit = model["classes"].get(klass) or model["classes"].get("all")
    if fit is None:
        cycles = model["worst_ratio"] * floor
        return {"cycles": cycles, "upper_cycles": cycles, "basis": "worst observed gSIM cycles / structural floor"}
    cycles = fit["overhead_cycles"] + fit["factor"] * floor
    scope = klass if klass in model["classes"] else "all"
    return {
        "cycles": cycles,
        "upper_cycles": cycles * fit["margin"],
        "basis": f"overhead + inefficiency x structural floor ({scope} bound gSIM rows)",
    }


def expected_seconds(fit: dict, cycles, *, spike_seconds: float = 0.0) -> float | None:
    """The central runtime of one measurement (setup + Spike tier seconds + cycles / rate), no margin."""
    if not isinstance(fit, dict) or type(cycles) not in (int, float) or not cycles > 0:
        return None
    return fit["setup_s"] + max(0.0, float(spike_seconds or 0.0)) + float(cycles) / fit["cycles_per_s"]


def job_plan(
    fit: dict,
    cycles,
    central_cycles,
    *,
    spike_seconds: float = 0.0,
    retries: int = 0,
    remaining_drain_s: float,
    remaining_end_s: float,
    final_window_s: float,
    long_allowed: bool,
    final_slots_free: bool,
    long_lane: bool = False,
    engine_ceiling_s: int | None = None,
    max_job_wall_s: int | None = None,
) -> tuple[dict | None, str]:
    """How one measurement may run now: ``({timeout_s, long, expected_s}, "")``, else ``(None, why)``.

    A normal job's timeout (from the upper ``cycles``) must end before the drain boundary. Otherwise,
    when a long slot is free, the job runs long if its expected runtime (from ``central_cycles``, no
    margin) ends before the final-model window; its timeout then reaches at most the campaign's end.
    Any job that may still run in the final-model window starts only when ``final_slots_free`` (slots
    beyond the grade reserve), since final grading must not wait for a measurement's slot.
    ``long_lane``: the caller judged this job long relative to the smaller work still pending; it then
    runs only in a long slot, so it never holds a normal slot those smaller cases need.
    ``engine_ceiling_s``: the selected gSIM process ceiling, which bounds every run whatever its timeout. A
    job whose expected runtime exceeds it is skipped (:data:`CEILING_SKIP`) rather than started to be
    killed there; otherwise its timeout is held to the ceiling.
    ``max_job_wall_s``: the operator's wall cap (:mod:`.gsim_wall_cap`); every timeout is held to it
    (whether a job is launched at all is the wall predictor's decision, :mod:`.gsim_wall_predictor`).
    """
    timeout = job_timeout(fit, cycles, spike_seconds=spike_seconds, retries=retries)
    if timeout is None:
        return None, "no cycles to time"
    expected = expected_seconds(fit, central_cycles, spike_seconds=spike_seconds)
    if type(engine_ceiling_s) is int and engine_ceiling_s > 0 and timeout > engine_ceiling_s:
        if expected is not None and expected > engine_ceiling_s:
            return None, f"{CEILING_SKIP} (expected {round(expected)} s, ceiling {engine_ceiling_s} s)"
        timeout = engine_ceiling_s
    if type(max_job_wall_s) is int and max_job_wall_s > 0:
        timeout = min(timeout, max_job_wall_s)
    long = False
    final_start = remaining_end_s - final_window_s
    if long_lane:
        if not long_allowed:
            return None, "long job waits for the long lane while smaller cases are pending"
        if expected is None or expected > final_start:
            return None, "expected runtime does not end before the final-model window"
        timeout, long = max(1, min(timeout, int(remaining_end_s))), True
    elif timeout > remaining_drain_s:
        if not long_allowed:
            return None, "derived timeout does not fit before the drain boundary and no long slot is free"
        if expected is None or expected > final_start:
            return None, "expected runtime does not end before the final-model window"
        timeout, long = max(1, min(timeout, int(remaining_end_s))), True
    if timeout > final_start and not final_slots_free:
        return None, "final-window slots are reserved for final grading"
    return {"timeout_s": timeout, "long": long, "expected_s": expected}, ""


def job_timeout(fit: dict, modeled_cycles, *, spike_seconds: float = 0.0, retries: int = 0) -> int | None:
    """Seconds one measurement may run: margin x (setup + Spike tier seconds + cycles / rate).

    Each earlier timed-out or lost attempt at the case doubles it. ``None`` without positive cycles.
    """
    if not isinstance(fit, dict) or type(modeled_cycles) not in (int, float) or not modeled_cycles > 0:
        return None
    seconds = fit["setup_s"] + max(0.0, float(spike_seconds or 0.0)) + float(modeled_cycles) / fit["cycles_per_s"]
    return max(1, math.ceil(fit["margin"] * seconds * (2 ** max(0, int(retries)))))


__all__ = [
    "CEILING_SKIP",
    "DEFAULTS",
    "ENV",
    "LEDGER",
    "ORIGIN",
    "REQUEST_PREFIX",
    "SECTION",
    "admission",
    "authorize",
    "bound_observations",
    "expected_seconds",
    "broker_admission",
    "inefficiency",
    "job_plan",
    "long_request",
    "job_timeout",
    "ledger_path",
    "note_result",
    "observations",
    "predicted_cycles",
    "read",
    "requested",
    "settings",
    "shared_pool_contended",
    "structural_floor",
    "throughput",
    "validate",
]
