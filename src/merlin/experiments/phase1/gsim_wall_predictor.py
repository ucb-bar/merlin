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
from pathlib import Path

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
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] + [sum(r[i] * t for r, t in zip(rows, targets))]
         for i in range(n)]
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
        return {"run": True, "predicted_s": None, "projected_s": None, "timeout_s": None,
                "reason": "no wall prediction"}
    projected = model["margin"] * predicted
    row = {"predicted_s": round(predicted, 1), "projected_s": round(projected, 1)}
    if cap_s is not None and projected > cap_s:
        from .gsim_wall_cap import message

        return {"run": False, **row, "timeout_s": None, "reason": message(projected, cap_s)}
    timeout = max(1, math.ceil(projected))
    return {"run": True, **row, "timeout_s": min(timeout, cap_s) if cap_s is not None else timeout, "reason": ""}


def _rooflines(run_dir: Path) -> dict:
    """``{case: structural roofline}`` from a run's verdict history (newest first wins)."""
    found = {}
    paths = sorted((Path(run_dir) / "qa_history").glob("verdict_fast_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
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
        ledgers = tuple((str(p / _LEDGER), (p / _LEDGER).stat().st_mtime_ns)
                        for p in siblings if (p / _LEDGER).is_file())
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
    return {**model, "basis": f"{len(own)} measured gSIM job(s) of this run and {len(seeded) - len(own)} of earlier runs"}


__all__ = ["FEATURES", "campaign_model", "decide", "features", "fit", "history", "ledger_points", "predict"]
