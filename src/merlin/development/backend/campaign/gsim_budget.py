"""Predict a gSIM job's WALL time before launching it, from the case's work and this campaign's runs.

The wall cap (:mod:`.gsim_wall_cap`) must not be how a too-long job is discovered: a job that runs
until it is killed holds a slot for the whole cap and returns nothing. Recorded jobs show why simulated
cycles alone do not predict wall time: the simulation rate ranges from ~100 to ~2,000 simulated cycles
per second and falls with the data a kernel moves (a 32x1024x1024 GEMM took 667 s for 213k cycles, a
256^3 GEMM 204 s for 422k), because the emulated host prepares every input byte and the coherent
export reads every output byte back.

The model is therefore linear in the job's work:

    wall = setup + a * simulated_cycles + b * read_bytes + c * write_bytes

which is the same as ``setup + cycles / rate`` with the rate falling as bytes per cycle (the data
movement density) rises. ``read_bytes``/``write_bytes`` are the case's logical requested bytes from its
structural roofline (declared shape); ``simulated_cycles`` is the case's timing-model estimate, else its
structural floor times this run's measured inefficiency (the lane computes it). Coefficients are fitted
by relative-error least squares, never negative (a feature whose fitted weight would be negative is
dropped), over this run's measured gSIM jobs, seeded with earlier runs' measured jobs of the same
target when this run has too few (:func:`history`). The decision margin is the upper quantile of
observed/predicted wall over the fitted jobs -- the fit's own error spread, not a constant.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

# ---- gsim_wall_predictor -----------------------------------------------------------------------

FEATURES = ("cycles", "read_bytes", "write_bytes")

_QUANTILE = 0.95

_MIN_POINTS = 4

_LEDGER = "joint_gsim_measurement_budget.json"


def roofline_bytes(roofline) -> tuple[float, float] | None:
    """``(read, write)`` logical requested bytes the case's structural roofline declares, or ``None``."""
    service = (roofline or {}).get("logical_request_ideal_service") if isinstance(roofline, dict) else None
    if not isinstance(service, dict):
        return None
    values = []
    for side in ("read", "write"):
        amount = (service.get(side) or {}).get("requested_bytes") if isinstance(service.get(side), dict) else 0
        if amount is None:
            amount = 0
        if type(amount) not in (int, float) or amount < 0:
            return None
        values.append(float(amount))
    return (values[0], values[1]) if any(values) else None


def features(cycles, roofline) -> dict | None:
    """The predictor's inputs for one case, or ``None`` when its cycles or declared bytes are unknown."""
    moved = roofline_bytes(roofline)
    if moved is None or type(cycles) not in (int, float) or not cycles > 0:
        return None
    return {"cycles": float(cycles), "read_bytes": moved[0], "write_bytes": moved[1]}


def _solve(rows, targets):
    """Least squares by the normal equations (small, dense); ``None`` when singular."""
    n = len(rows[0])
    a = [
        [sum(r[i] * r[j] for r in rows) for j in range(n)] + [sum(r[i] * t for r, t in zip(rows, targets))]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-30:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                factor = a[r][col] / a[col][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[col])]
    return [a[i][n] / a[i][i] for i in range(n)]


def fit(points) -> dict | None:
    """``{coefficients, margin, observations}`` over ``[(features, wall_s)]``; ``None`` with too few.

    Each point is weighted by its own wall (relative error), so short jobs count as much as long ones."""
    points = [(f, float(w)) for f, w in points if isinstance(f, dict) and type(w) in (int, float) and w > 0]
    if len(points) < _MIN_POINTS:
        return None
    active = ["setup", *FEATURES]
    while active:
        rows = [[(1.0 if name == "setup" else f[name]) / w for name in active] for f, w in points]
        solution = _solve(rows, [1.0] * len(points))
        if solution is None:
            # A feature that never varies (or is zero) cannot be fitted: drop the last feature first.
            active.pop()
            continue
        negative = [(value, name) for value, name in zip(solution, active) if value < 0]
        if not negative:
            break
        active.remove(min(negative)[1])
    else:
        return None
    coefficients = {name: 0.0 for name in ("setup", *FEATURES)}
    coefficients.update(dict(zip(active, solution)))
    model = {"coefficients": coefficients, "observations": len(points)}
    ratios = sorted(w / predicted for f, w in points if (predicted := predict(model, f)) and predicted > 0)
    if not ratios:
        return None
    index = min(len(ratios) - 1, max(0, math.ceil(_QUANTILE * len(ratios)) - 1))
    model["margin"] = max(1.0, ratios[index])
    model["quantile"] = _QUANTILE
    return model


def predict(model: dict | None, case: dict | None) -> float | None:
    """Central predicted wall seconds of one case."""
    if not isinstance(model, dict) or not isinstance(case, dict):
        return None
    c = model["coefficients"]
    return c["setup"] + sum(c[name] * case[name] for name in FEATURES)


def decide(model, case, *, cap_s=None) -> dict:
    """``{run, predicted_s, timeout_s, reason}`` for one job under the wall cap ``cap_s``.

    The projection is the prediction times the fit's margin (its observed error spread): a job whose
    projection exceeds the cap is not run, since it could only be stopped there; otherwise its timeout
    is that projection, at most the cap. Without a prediction the caller's own rule stands (``run``
    True, no timeout)."""
    predicted = predict(model, case)
    if predicted is None:
        return {
            "run": True,
            "predicted_s": None,
            "projected_s": None,
            "timeout_s": None,
            "reason": "no wall prediction",
        }
    projected = model["margin"] * predicted
    row = {"predicted_s": round(predicted, 1), "projected_s": round(projected, 1)}
    if cap_s is not None and projected > cap_s:
        return {"run": False, **row, "timeout_s": None, "reason": message(projected, cap_s)}
    timeout = max(1, math.ceil(projected))
    return {"run": True, **row, "timeout_s": min(timeout, cap_s) if cap_s is not None else timeout, "reason": ""}


def _rooflines(run_dir: Path) -> dict:
    """``{case: structural roofline}`` from a run's verdict history (newest first wins)."""
    found = {}
    paths = sorted(
        (Path(run_dir) / "qa_history").glob("verdict_fast_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for path in paths[:8]:
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for row in (document.get("per_capsule") or ()) if isinstance(document, dict) else ():
            name = row.get("capsule") if isinstance(row, dict) else None
            roofline = (row.get("performance") or {}).get("structural_roofline") if isinstance(row, dict) else None
            if isinstance(name, str) and name not in found and isinstance(roofline, dict):
                found[name] = roofline
    return found


def ledger_points(jobs, rooflines=None) -> list:
    """``[(features, wall)]`` of a ledger's measured jobs: their recorded features, else rebuilt from the
    case's roofline in ``rooflines`` and the measured cycles."""
    points = []
    for job in (jobs or {}).values() if isinstance(jobs, dict) else ():
        if not isinstance(job, dict) or job.get("outcome") != "measured":
            continue
        wall, cycles = job.get("elapsed_s"), job.get("barrier_cycles")
        case = job.get("wall_features") if isinstance(job.get("wall_features"), dict) else None
        if case is None:
            case = features(cycles, (rooflines or {}).get(job.get("capsule")))
        elif type(cycles) in (int, float) and cycles > 0:
            case = {**case, "cycles": float(cycles)}  # the measured cycles replace the estimate
        if case is not None and type(wall) in (int, float) and wall > 0:
            points.append((case, float(wall)))
    return points


_HISTORY: dict = {}


def history(run_dir: Path) -> list:
    """Measured jobs of the other runs beside ``run_dir`` (same target, same suite).

    Read once per process while those runs' ledgers are unchanged: their verdict histories are large."""
    run_dir = Path(run_dir)
    try:
        siblings = sorted(p for p in run_dir.parent.iterdir() if p.is_dir() and p != run_dir)
        ledgers = tuple(
            (str(p / _LEDGER), (p / _LEDGER).stat().st_mtime_ns) for p in siblings if (p / _LEDGER).is_file()
        )
    except OSError:
        return []
    key = (str(run_dir), ledgers)
    if key not in _HISTORY:
        points = []
        for path, _ in ledgers:
            try:
                jobs = json.loads(Path(path).read_text()).get("jobs")
            except (OSError, ValueError, AttributeError):
                continue
            if jobs and any(isinstance(j, dict) and j.get("outcome") == "measured" for j in jobs.values()):
                points += ledger_points(jobs, _rooflines(Path(path).parent))
        _HISTORY.clear()
        _HISTORY[key] = points
    return list(_HISTORY[key])


def campaign_model(jobs, run_dir: Path | None) -> dict | None:
    """The fit over this run's measured jobs, seeded with earlier runs' when this run has too few."""
    measured = [j for j in (jobs or {}).values() if isinstance(j, dict) and j.get("outcome") == "measured"]
    # Jobs recorded by this module carry their features; older ones need the run's rooflines.
    needs_rooflines = run_dir is not None and any(not isinstance(j.get("wall_features"), dict) for j in measured)
    rooflines = _rooflines(run_dir) if needs_rooflines else {}
    own = ledger_points(jobs, rooflines)
    if len(own) >= _MIN_POINTS:
        model = fit(own)
        if model is not None:
            return {**model, "basis": f"{len(own)} measured gSIM job(s) of this run"}
    seeded = own + (history(run_dir) if run_dir is not None else [])
    model = fit(seeded)
    if model is None:
        return None
    return {
        **model,
        "basis": f"{len(own)} measured gSIM job(s) of this run and {len(seeded) - len(own)} of earlier runs",
    }


# ---- gsim_engine_bound -------------------------------------------------------------------------
# The simulated-cycle bound of one budgeted gSIM measurement, derived from its wall budget.
#
# The selected gSIM command carries a ``+max-cycles`` hang bound. That bound counts every simulated cycle
# of the run -- boot, the host's input preparation, the kernel and the coherent output export -- while the
# measurement lane (:mod:`.feedback.idle_gsim`) budgets wall seconds from the kernel's cycles. A case whose
# whole run needs more simulated cycles than the selected bound is stopped by the harness (its cycle-cap
# exit status) long before its wall budget ends, whatever that budget is: in one campaign 20 of 34 lane
# jobs ended this way after about 700 s, against budgets of 520 to 11,327 s.
#
# This module carries the budget through to the emulator instead:
#
# * :func:`engine_runs` reads, from a measurement child's own invocation records, how many simulated
#   cycles each gSIM harness run reached in how many wall seconds (the harness's completion line, or its
#   cycle bound when it stopped there);
# * :func:`engine_rate` summarizes the runs the lane's ledger holds into the engine's observed rate;
# * :func:`engine_cycle_bound` turns a job's wall budget into the simulated cycles it may reach (the
#   fastest observed rate, with the observed spread as margin), so the wall budget -- not the cap -- ends
#   the run; without an observed rate the selected bound stays and the result says so;
# * :func:`engine_cycle_budget` applies a job's derived bound to the gSIM backend of this process tree
#   (:data:`merlin.runtime.backends.rocc.tool_selection.GSIM_BUDGET_MAX_CYCLES_ENV`); it only ever raises the
#   selected bound, and the selected wall deadline still bounds the process.
#
# Nothing here raises into the lane or the grader: unreadable records are skipped, never estimated.

_CYCLE_FLAG = "+max-cycles="

_COMPLETION = "Cycles:"

_MAX_RECORD_BYTES = 1 << 20

_MAX_STREAM_BYTES = 16 << 20


def _cap_exit() -> int | None:
    try:
        from merlin.runtime.backends.rocc import chipyard_backend as chipyard_rocc

        return chipyard_rocc._GSIM_CYCLE_CAP_EXIT  # noqa: SLF001 -- the harness's own cycle-cap status
    except Exception:  # noqa: BLE001 -- without the convention a stopped run is simply not recognized
        return None


def _json(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_RECORD_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _completed_cycles(stream) -> int | None:
    """The harness's ``Cycles: N`` completion line in the run's stderr, if present."""
    path = Path(stream["path"]) if isinstance(stream, dict) and isinstance(stream.get("path"), str) else None
    try:
        if path is None or path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_STREAM_BYTES:
            return None
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None
    cycles = None
    for line in text.splitlines():
        head, _, value = line.strip().partition(" ")
        if head == _COMPLETION and value.strip().isdigit():
            cycles = int(value.strip())
    return cycles


def engine_runs(root: Path) -> list[dict]:
    """``[{cycles, wall_s, capped}]`` of the gSIM harness runs recorded under ``root``.

    A run is a harness run when its process record's argv carries exactly one ``+max-cycles=`` bound.
    Its cycles are the harness's completion line, or that bound when the process exited with the
    harness's cycle-cap status (``capped``). Its wall seconds are the record's own start and finish.
    A run without both numbers (still running, killed at its wall deadline, unreadable) is skipped.
    """
    cap_exit = _cap_exit()
    found = []
    try:
        records = sorted(Path(root).rglob("invocations/*/invocation.json"))
    except OSError:
        return found
    for path in records:
        record = _json(path)
        if not isinstance(record, dict) or record.get("kind") != "subprocess":
            continue
        bounds = [str(arg)[len(_CYCLE_FLAG) :] for arg in record.get("argv") or () if str(arg).startswith(_CYCLE_FLAG)]
        start, end, code = record.get("started_ns"), record.get("finished_ns"), record.get("returncode")
        if len(bounds) != 1 or not bounds[0].isdigit() or type(start) is not int or type(end) is not int:
            continue
        wall = (end - start) / 1e9
        if not wall > 0 or type(code) is not int:
            continue
        capped = cap_exit is not None and code == cap_exit
        cycles = int(bounds[0]) if capped else _completed_cycles(record.get("stderr")) if code == 0 else None
        if cycles is not None and cycles > 0:
            found.append({"cycles": cycles, "wall_s": round(wall, 3), "capped": capped})
    return found


def engine_rate(jobs) -> dict | None:
    """The engine's observed simulated cycles per wall second over the runs ``jobs`` recorded, or ``None``.

    ``jobs`` are the lane ledger's job records; each finished measurement child adds its
    :func:`engine_runs` as ``engine_runs``. ``spread`` is the fastest over the slowest rate (at least 1).
    """
    rates = []
    for job in (jobs or {}).values() if isinstance(jobs, dict) else ():
        if not isinstance(job, dict):
            continue
        for run in job.get("engine_runs") or ():
            if not isinstance(run, dict):
                continue
            cycles, wall = run.get("cycles"), run.get("wall_s")
            if type(cycles) is int and cycles > 0 and type(wall) in (int, float) and wall > 0:
                rates.append(cycles / wall)
    if not rates:
        return None
    fastest, slowest = max(rates), min(rates)
    return {
        "max_cycles_per_s": round(fastest, 3),
        "min_cycles_per_s": round(slowest, 3),
        "spread": round(fastest / slowest, 3),
        "observations": len(rates),
    }


def engine_ceiling(jobs) -> int | None:
    """The selected gSIM process ceiling (seconds) the most recent measurement child reported, if any."""
    latest, ceiling = None, None
    for job in (jobs or {}).values() if isinstance(jobs, dict) else ():
        if not isinstance(job, dict):
            continue
        value = job.get("engine_timeout_ceiling_s")
        when = job.get("finished_at_epoch") or job.get("started_at_epoch") or 0
        if type(value) in (int, float) and value > 0 and (latest is None or when >= latest):
            latest, ceiling = when, int(value)
    return ceiling


def engine_cycle_bound(timeout_s, rate: dict | None, *, margin: float = 1.0) -> dict:
    """``{max_cycles, basis}``: the simulated cycles a run with wall budget ``timeout_s`` may reach.

    ``max_cycles`` is the budget times the fastest observed rate times the larger of ``margin`` (the
    lane's runtime margin) and the observed rate spread, so a run that goes as fast as any observed one
    still ends at its wall budget, not at the cap. ``None`` without an observed rate: the selected
    command's bound stays, and ``basis`` says so.
    """
    if not isinstance(rate, dict) or type(timeout_s) not in (int, float) or not timeout_s > 0:
        return {"max_cycles": None, "basis": "selected command cycle bound (no gSIM engine rate observed yet)"}
    factor = max(1.0, float(margin or 1.0), float(rate.get("spread") or 1.0))
    cycles = math.ceil(float(timeout_s) * float(rate["max_cycles_per_s"]) * factor)
    return {
        "max_cycles": cycles,
        "basis": f"wall budget {round(timeout_s)} s x fastest observed engine rate "
        f"{round(rate['max_cycles_per_s'])} cycles/s x margin {round(factor, 3)} "
        f"({rate['observations']} recorded run(s))",
    }


@contextmanager
def engine_cycle_budget(max_cycles):
    """Within, gSIM runs of this process tree may reach ``max_cycles`` (``None``: the selected bound)."""
    from merlin.runtime.backends.rocc.tool_selection import GSIM_BUDGET_MAX_CYCLES_ENV as ENV

    if type(max_cycles) is not int or max_cycles <= 0:
        yield None
        return
    previous = os.environ.get(ENV)
    os.environ[ENV] = str(max_cycles)
    try:
        yield max_cycles
    finally:
        if previous is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = previous


def engine_stop(runs) -> str | None:
    """Why the harness ended a run before the program did (its cycle bound), for the response; else ``None``."""
    capped = [run for run in runs or () if isinstance(run, dict) and run.get("capped")]
    if not capped:
        return None
    run = capped[-1]
    return (
        f"the gSIM harness stopped at its cycle bound ({run['cycles']} simulated cycles) after "
        f"{round(run['wall_s'])} s, before the program finished"
    )


# ---- joint_gsim_queue --------------------------------------------------------------------------
# Waiting order for the joint gSIM correctness slots, kept in the flock'd correctness ledger.
#
# A full concurrency cap with budget left is a queue, not a refusal: an author request waits for a
# slot across every process that shares the ledger, and is served before background work (promotions,
# operator measurements) of other cases, because the author is blocked on it. The order is a priority,
# never a quota: background work still takes any slot no author is waiting for.
#
# Within a class, waiting requests are interleaved across their requesters (the caller thread a
# self-check names, :data:`REQUESTER_ENV`): each requester's oldest waiting request goes first, then each
# one's second, and so on, each round in arrival order. A sweep of many single-case requests keeps its
# own order but no longer holds another requester's single check behind all of it. Requests that name
# no requester share one anonymous requester, which keeps arrival order among them.
#
# Background work does not wait inside its evaluation child (its broker keeps it queued until a slot
# is free, :func:`background_slots`); a background child that loses the race reports a retryable
# refusal (:class:`SlotBusy`) and its broker leaves the request queued. A waiting entry belongs to a
# live process: one whose pid is gone is dropped, as dead workers already release their slot.

CLASS_ENV = "MERLIN_JOINT_GSIM_QUEUE_CLASS"

REQUEST_ENV = "MERLIN_JOINT_GSIM_REQUEST"

REQUESTER_ENV = "MERLIN_JOINT_GSIM_REQUESTER"

AUTHOR = "author"

BACKGROUND = "background"

_RANK = {AUTHOR: 0, BACKGROUND: 1}

WAITING_STATUS = "queued_for_gsim_slot"

ADMITTED_STATUS = "gsim_slot_admitted"


class SlotBusy(ValueError):
    """Every slot is busy and this request does not wait; the request itself stays valid."""

    retryable = True


def request_class(env=None) -> str:
    value = (os.environ if env is None else env).get(CLASS_ENV, "").strip()
    return value if value in _RANK else AUTHOR


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


def settle(data: dict) -> None:
    """Drop slots and waiting entries of processes that no longer exist (charges are kept)."""
    data["active"] = [row for row in data.get("active", ()) if _alive(row.get("pid"))]
    data["waiting"] = [row for row in data.get("waiting", ()) if _alive(row.get("pid"))]


def requester(env=None) -> str | None:
    value = (os.environ if env is None else env).get(REQUESTER_ENV, "").strip()
    return value[:128] or None


def name_requester(env: dict, request: dict) -> dict:
    """Set the evaluation child's requester from a self-check request's caller thread (else none).

    Used only to interleave waiting requests; it grants nothing and a request without it waits as one
    anonymous requester.
    """
    env.pop(REQUESTER_ENV, None)
    caller = request.get("caller_thread_id") if isinstance(request, dict) else None
    if isinstance(caller, str) and caller.strip():
        env[REQUESTER_ENV] = caller.strip()[:128]
    return env


def _ordered(data: dict) -> list:
    """Class first; within a class, round-robin over requesters, each round in arrival order."""
    rows = sorted(data.get("waiting", ()), key=lambda row: (_RANK.get(row.get("class"), 1), row.get("seq", 0)))
    turns: dict[tuple, int] = {}
    keyed = []
    for row in rows:
        rank = _RANK.get(row.get("class"), 1)
        own = (rank, row.get("requester"))
        keyed.append(((rank, turns.get(own, 0), row.get("seq", 0)), row))
        turns[own] = turns.get(own, 0) + 1
    return [row for _, row in sorted(keyed, key=lambda item: item[0])]


def enqueue(data: dict, token: str, *, capsule: str, request_class: str, now: float, requester=None) -> None:
    """Add ``token`` behind its requester's earlier entries (idempotent: an entry keeps its place)."""
    waiting = data.setdefault("waiting", [])
    if any(row.get("token") == token for row in waiting):
        return
    seq = int(data.get("next_seq", 0))
    data["next_seq"] = seq + 1
    waiting.append(
        {
            "token": token,
            "pid": os.getpid(),
            "capsule": capsule,
            "class": request_class,
            "seq": seq,
            "enqueued_at_epoch": now,
            **({"requester": requester} if requester else {}),
        }
    )


def leave(data: dict, token: str) -> None:
    data["waiting"] = [row for row in data.get("waiting", ()) if row.get("token") != token]


def ahead(data: dict, token: str) -> int:
    """Waiting entries served before ``token``."""
    for index, row in enumerate(_ordered(data)):
        if row.get("token") == token:
            return index
    return len(data.get("waiting", ()))


def admissible(data: dict, token: str, max_jobs: int) -> bool:
    """Whether a free slot is left for ``token`` after every entry ahead of it is served."""
    return ahead(data, token) < max_jobs - len(data.get("active", ()))


def background_slots(data: dict, max_jobs: int, starting=()) -> int:
    """Slots background work may start now on a settled ledger: none while an author waits.

    ``starting`` names request ids a broker launched whose child has not yet taken its slot (it
    registers ``REQUEST_ENV`` on its row); they count as busy, so a broker polling faster than its
    children start cannot launch more jobs than there are slots.
    """
    if any(row.get("class") == AUTHOR for row in data.get("waiting", ())):
        return 0
    active = data.get("active", ())
    registered = {row.get("request") for row in active}
    return max(0, max_jobs - len(active) - len(set(starting) - registered))


def describe(data: dict, token: str, max_jobs: int) -> dict:
    n_ahead = ahead(data, token)
    return {
        "state": "waiting",
        "ahead": n_ahead,
        "slots": max_jobs,
        "busy": len(data.get("active", ())),
        "note": f"queued for a gSIM slot ({n_ahead} ahead)",
    }


def publish(path, status: str, queue: dict) -> None:
    """Write the queue state as request progress; never a verdict, so a failure to write is ignored."""
    if not path:
        return
    path = Path(path)
    document = {
        "status": status,
        "gsim_slot_queue": queue,
        "updated_at_unix_ns": time.time_ns(),
        "note": "Waiting for a gSIM correctness slot; not a verdict and not a refusal.",
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(document))
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)


def waiting(path) -> dict | None:
    """The queue block of a request whose child is waiting for a slot, else ``None`` (never raises)."""
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if isinstance(document, dict) and document.get("status") == WAITING_STATUS:
        queue = document.get("gsim_slot_queue")
        return queue if isinstance(queue, dict) else {}
    return None


# ---- gsim_wall_cap -----------------------------------------------------------------------------
# The operator's wall-time cap on one gSIM job, read from the campaign policy.
#
# A gSIM run is worth its slot only while it returns soon enough to steer the next edit. The policy
# sections ``gsim`` (witnesses), ``gsim_correctness`` and ``gsim_measurement`` may each carry
# ``max_job_wall_s``; nothing here supplies a default, so a policy without it keeps its previous
# behaviour. Where a cap is selected, a job's wall time is PREDICTED before it is launched
# (:mod:`.gsim_wall_predictor`), so the cap is never how a too-long job is discovered:
#
# * the idle measurement lane launches a job only when its projection fits the cap, with that projection
#   (at most the cap) as its timeout, carried through to the emulator (:mod:`.gsim_engine_bound`);
# * an author or promotion gSIM request projected beyond its section's cap is refused before it takes
#   or waits for a slot, and the refusal shows the projection and what to use instead (:func:`message`).
#
# A prediction needs the case's cycles and declared bytes and enough measured jobs; without them nothing
# is refused here.

KEY = "max_job_wall_s"

SKIP = "gSIM not run"

SECTIONS = ("gsim", "gsim_correctness", "gsim_measurement")


def validate(value) -> None:
    """Refuse a malformed cap; ``None`` (absent) is no cap."""
    if value is not None and (type(value) is not int or not 1 <= value <= 172800):
        raise ValueError(f"invalid joint gSIM {KEY}")


def max_job_wall_s(section: str, policy: dict | None) -> int | None:
    """The selected cap of ``section`` (one of :data:`SECTIONS`), or ``None``."""
    if section not in SECTIONS:
        raise ValueError(f"unknown gSIM policy section {section!r}")
    value = ((policy or {}).get(section) or {}).get(KEY)
    validate(value)
    return value


def _minutes(seconds: float) -> str:
    return f"{max(1, math.ceil(seconds / 60))} min"


def message(projected_s: float, cap_s: int) -> str:
    """The explanatory refusal of a job projected beyond the cap."""
    return (
        f"{SKIP}: projected ~{_minutes(projected_s)} > {_minutes(cap_s)} cap; "
        "use the timing model or a smaller representative case"
    )


def _case_features(run_dir: Path, names: list[str]) -> dict | None:
    """The predictor inputs of the largest of ``names``: newest cycles (measured, else modeled) and
    the case's declared bytes, from the run's verdict history."""

    paths = sorted(
        (Path(run_dir) / "qa_history").glob("verdict_fast_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    found: dict[str, dict] = {}
    for path in paths[:8]:
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for row in (document.get("per_capsule") or ()) if isinstance(document, dict) else ():
            name = row.get("capsule") if isinstance(row, dict) else None
            if name not in names or name in found:
                continue
            performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
            case = features(
                row.get("barrier_cycles") or performance.get("estimated_whole_soc_cycles"),
                performance.get("structural_roofline"),
            )
            if case is not None:
                found[name] = case
    if not found:
        return None
    return max(found.values(), key=lambda case: (case["read_bytes"] + case["write_bytes"], case["cycles"]))


def request_refusal(capsules) -> str | None:
    """Why a gSIM request for ``capsules`` must not start (projected beyond its cap), else ``None``."""
    try:
        from merlin.development.backend.campaign import control as joint_control
        from merlin.development.backend.campaign import joint_gsim_measurement as M

        state = joint_control.selected()
        if state is None:
            return None
        names = [name.strip() for name in str(capsules).split(",") if name.strip()]
        correctness = joint_control.correctness_capsules(state)
        section = "gsim_correctness" if names and all(name in correctness for name in names) else "gsim"
        cap = max_job_wall_s(section, state.get("policy"))
        if cap is None:
            return None
        # The run directory holds the campaign state, its verdict history and the measurement ledger.
        run_dir = Path(os.environ[joint_control.ENV]).parent
        decision = decide(campaign_model(M.read().get("jobs", {}), run_dir), _case_features(run_dir, names), cap_s=cap)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None  # no evidence is no refusal
    return None if decision["run"] else decision["reason"]


__all__ = [
    "FEATURES",
    "KEY",
    "SECTIONS",
    "SKIP",
    "campaign_model",
    "decide",
    "engine_ceiling",
    "engine_cycle_bound",
    "engine_cycle_budget",
    "engine_rate",
    "engine_runs",
    "engine_stop",
    "features",
    "fit",
    "history",
    "ledger_points",
    "max_job_wall_s",
    "message",
    "predict",
    "request_refusal",
    "validate",
]
