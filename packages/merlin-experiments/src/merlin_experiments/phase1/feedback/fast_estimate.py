"""A calibrated cycle estimate for every passing case, from its emitted command census, in milliseconds.

The Spike timing observer needs about 18 minutes for a large convolution (about 0.19 ms per accelerator
command, twice), while a grade or self-check gives each case a few minutes; such a case returned
``timing_status: UNKNOWN`` with "observation budget too small", and the author was told UNKNOWN stays
unranked. This module gives it an estimate instead, labelled ``evidence_kind: fast_estimate``:

* :func:`census` reads the case's decoded instruction stream (``generated/instruction_trace.json``, the
  same decode ``isa_tools.py disasm`` and :mod:`.fence_overlap` read) and multiplies every site by its
  proved executions per kernel call (the counted-loop proof of the decoder). Fences are the sites the
  decoder identifies structurally as architectural fences; every other decoded site is one accelerator
  command. A site whose executions are not proved counts once and marks the census partial;
* :func:`estimate` prices it as ``max(structural floor, commands x c_cmd) + fences x c_fence``: the
  floor is the case's own structural roofline (compute floor or ideal request service, derived from the
  target's facts), ``c_cmd`` the per-command host issue cost and ``c_fence`` the cost of draining the
  device at a fence;
* :func:`fit` calibrates ``c_cmd`` and ``c_fence`` against complete Spike timing-model estimates of the
  same programs (minimum median absolute log error) and states its error band as the 10th/90th
  percentiles of complete/estimated over the calibration pairs;
* :func:`calibration` uses the run's own pairs (:func:`record`, a ledger beside the joint campaign
  state, appended whenever an observation completes) once there are enough, else the reviewed seed in
  ``resources/fast_estimate_calibration.json`` whose ``facts_sha256`` matches the selected facts. No
  calibration for these facts leaves the estimate UNKNOWN with its reason -- never a guessed constant.

A complete Spike timing-model estimate stays preferred wherever one exists; this one fills the gap and
carries its band. Nothing here names a target, an opcode or a class.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import statistics
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "merlin.phase1.fast_estimate.v1"
EVIDENCE_KIND = "fast_estimate"
LEDGER = "joint_fast_estimate_pairs.json"
SEED = "fast_estimate_calibration.json"
TRACE = "instruction_trace.json"
#: Calibration pairs kept in the run ledger (the newest) and fewest needed before it replaces the seed.
_KEEP = 512
MIN_PAIRS = 12
_MAX_TRACE_BYTES = 32 << 20
#: The coarse grid of the fit: 0 and 2**(i/2) cycles; the fine pass searches 2**(j/8) around the best.
_COARSE = (0.0, *(2.0 ** (i / 2) for i in range(-4, 25)))


def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def census(trace) -> dict | None:
    """``{commands, fences, sites, unproved_sites, complete}`` per kernel call, or ``None`` without sites."""
    if not isinstance(trace, dict):
        return None
    from .fence_overlap import _is_fence

    commands = fences = sites = unproved = 0
    for row in trace.get("instructions") or ():
        if not isinstance(row, dict):
            continue
        fence = _is_fence(row)
        if not fence and row.get("funct") is None:
            continue  # neither a structural fence nor a decoded accelerator command
        runs = row.get("executions")
        if type(runs) is not int or runs < 0:
            unproved, runs = unproved + 1, 1
        sites += 1
        if fence:
            fences += runs
        else:
            commands += runs
    if not sites:
        return None
    return {"commands": commands, "fences": fences, "sites": sites, "unproved_sites": unproved,
            "complete": unproved == 0}


def trace_for(root) -> dict | None:
    """The decoded instruction stream beside a case's run directory (``generated/``), else ``None``."""
    if root is None:
        return None
    root = Path(root)
    for path in (root / "generated" / TRACE, root / TRACE):
        try:
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= _MAX_TRACE_BYTES:
                return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def floor(structural) -> float | None:
    """The structural floor of the case's roofline (compute or ideal request service), else ``None``."""
    from ..joint_gsim_measurement import structural_floor

    value, _ = structural_floor(structural)
    return float(value) if _positive(value) else None


def _price(bound, commands, fences, per_command, per_fence) -> float:
    return max(bound or 0.0, commands * per_command) + fences * per_fence


def _score(points, per_command, per_fence) -> float:
    errors = []
    for bound, commands, fences, cycles in points:
        priced = _price(bound, commands, fences, per_command, per_fence)
        errors.append(abs(math.log(priced / cycles)) if priced > 0 else math.inf)
    return statistics.median(errors)


def fit(points) -> dict | None:
    """Fit ``c_cmd``/``c_fence`` to ``(floor, commands, fences, complete cycles)`` points (module docstring)."""
    points = [
        (float(b or 0.0), int(c), int(f), float(y)) for b, c, f, y in points
        if type(c) is int and type(f) is int and c >= 0 and f >= 0 and _positive(y) and (b is None or b >= 0)
    ]
    if len(points) < MIN_PAIRS:
        return None
    best = min(((_score(points, a, b), a, b) for a in _COARSE for b in _COARSE), key=lambda t: (t[0], t[1], t[2]))
    _, a0, b0 = best
    fine = lambda v: (0.0,) if v == 0 else tuple(v * 2.0 ** (j / 8) for j in range(-4, 5))  # noqa: E731
    best = min(((_score(points, a, b), a, b) for a in fine(a0) for b in fine(b0)), key=lambda t: (t[0], t[1], t[2]))
    error, per_command, per_fence = best
    ratios = sorted(y / _price(b, c, f, per_command, per_fence) for b, c, f, y in points
                    if _price(b, c, f, per_command, per_fence) > 0)
    quantile = lambda q: ratios[min(len(ratios) - 1, int(q * (len(ratios) - 1) + 0.5))]  # noqa: E731
    return {
        "cycles_per_command": round(per_command, 4),
        "cycles_per_fence": round(per_fence, 4),
        "pairs": len(points),
        "median_abs_log_error": round(error, 4),
        # complete / estimated over the calibration pairs: the band an estimate carries.
        "ratio_quantiles": {"p10": round(quantile(0.10), 3), "p25": round(quantile(0.25), 3),
                            "p50": round(quantile(0.50), 3), "p75": round(quantile(0.75), 3),
                            "p90": round(quantile(0.90), 3)},
    }


def _ledger_path() -> Path | None:
    from .. import joint_control

    state = os.environ.get(joint_control.ENV, "").strip()
    return Path(state).with_name(LEDGER) if state else None


@contextmanager
def _locked(path: Path):
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read()
        try:
            document = json.loads(raw) if raw.strip() else {}
        except ValueError:
            document = {}
        if not isinstance(document, dict) or not isinstance(document.get("pairs"), list):
            document = {"schema": SCHEMA, "pairs": []}
        yield document
        handle.seek(0)
        handle.truncate()
        json.dump(document, handle, sort_keys=True)
        handle.flush()


def record(structural, counted, cycles, *, facts_sha256=None, path: Path | None = None) -> None:
    """Add one complete estimate and the census of the same program to the run ledger and refit; never raises."""
    try:
        if not isinstance(counted, dict) or not _positive(cycles) or not counted.get("complete"):
            return
        path = _ledger_path() if path is None else path
        if path is None:
            return
        pair = {"floor": floor(structural), "commands": counted["commands"], "fences": counted["fences"],
                "cycles": cycles, "facts_sha256": facts_sha256}
        with _locked(path) as document:
            document["pairs"] = [*document["pairs"], pair][-_KEEP:]
            same = [p for p in document["pairs"] if p.get("facts_sha256") == facts_sha256]
            fitted = fit([(p.get("floor"), p.get("commands"), p.get("fences"), p.get("cycles")) for p in same])
            document.setdefault("fits", {})[str(facts_sha256)] = fitted
    except (OSError, ValueError, TypeError, KeyError):
        pass


def _seed(facts_sha256) -> dict | None:
    from importlib.resources import files

    try:
        document = json.loads(files("merlin_experiments").joinpath(f"resources/{SEED}").read_text(encoding="utf-8"))
    except (OSError, ValueError, ModuleNotFoundError):
        return None
    for row in (document or {}).get("calibrations") or ():
        if isinstance(row, dict) and facts_sha256 and row.get("facts_sha256") == facts_sha256:
            return dict(row, source="reviewed seed: " + str(row.get("source") or SEED))
    return None


def calibration(facts_sha256, *, path: Path | None = None) -> dict | None:
    """The run's own fit for these facts once it has enough pairs, else the matching reviewed seed."""
    path = _ledger_path() if path is None else path
    if path is not None:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            fitted = ((document or {}).get("fits") or {}).get(str(facts_sha256))
            if isinstance(fitted, dict) and fitted.get("pairs", 0) >= MIN_PAIRS:
                return dict(fitted, source="this run's complete timing-model estimates")
        except (OSError, ValueError, AttributeError):
            pass
    return _seed(facts_sha256)


def estimate(counted, structural, calibrated) -> dict:
    """The ``fast_estimate`` block: cycles, error band and basis, or status UNKNOWN with its reason."""
    block = {"schema": SCHEMA, "evidence_kind": EVIDENCE_KIND, "status": "UNKNOWN", "cycles": None}
    if counted is None:
        return dict(block, reason="no decoded instruction stream with accelerator commands for this case")
    if calibrated is None:
        return dict(block, census=counted,
                    reason="no fast-estimate calibration for the selected hardware facts yet")
    bound = floor(structural)
    per_command, per_fence = calibrated["cycles_per_command"], calibrated["cycles_per_fence"]
    cycles = _price(bound, counted["commands"], counted["fences"], per_command, per_fence)
    if not cycles > 0:
        return dict(block, census=counted, reason="census and roofline state no work to price")
    band = calibrated.get("ratio_quantiles") or {}
    issue = counted["commands"] * per_command
    terms = {"structural_floor_cycles": round(bound, 1) if bound else None,
             "command_issue_cycles": round(issue, 1),
             "fence_drain_cycles": round(counted["fences"] * per_fence, 1)}
    return dict(
        block,
        status="ESTIMATED",
        cycles=int(round(cycles)),
        low_cycles=int(round(cycles * band["p10"])) if _positive(band.get("p10")) else None,
        high_cycles=int(round(cycles * band["p90"])) if _positive(band.get("p90")) else None,
        band="p10-p90 of complete/estimated over the calibration pairs",
        limited_by="command issue" if issue >= (bound or 0.0) else "structural floor",
        terms=terms,
        census=counted,
        lower_bound=not counted["complete"],
        calibration={key: calibrated.get(key) for key in (
            "cycles_per_command", "cycles_per_fence", "pairs", "median_abs_log_error", "ratio_quantiles", "source")},
        basis=("max(structural floor, commands x per-command issue) + fences x per-fence drain, calibrated "
               "against complete Spike timing-model estimates; an estimate with a band, not a measurement"),
    )


def for_case(root, structural, *, facts_sha256, path: Path | None = None) -> dict:
    """The ``fast_estimate`` block for the case whose run directory is ``root``; never raises."""
    try:
        return estimate(census(trace_for(root)), structural, calibration(facts_sha256, path=path))
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError) as error:
        return {"schema": SCHEMA, "evidence_kind": EVIDENCE_KIND, "status": "UNKNOWN", "cycles": None,
                "reason": "fast estimate unavailable: " + type(error).__name__}


def attach(performance: dict, root, structural, *, facts_sha256, path: Path | None = None) -> None:
    """Add ``fast_estimate`` to a passing case's performance record; a complete estimate also calibrates.

    Runs whether or not the timing observer ran, so every passing case carries an estimate; the complete
    estimate, when present, stays the preferred evidence. Never raises."""
    try:
        counted = census(trace_for(root))
        cycles = performance.get("estimated_whole_soc_cycles")
        if performance.get("complete_event_accounting") is True and _positive(cycles):
            record(structural, counted, cycles, facts_sha256=facts_sha256, path=path)
        performance["fast_estimate"] = estimate(counted, structural, calibration(facts_sha256, path=path))
        from .joint_outcome_brief import decide_on_fast_estimate

        decide_on_fast_estimate((performance.get("feedback_brief") or {}).get("outcomes"), performance["fast_estimate"])
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError, AttributeError) as error:
        performance["fast_estimate"] = {"schema": SCHEMA, "evidence_kind": EVIDENCE_KIND, "status": "UNKNOWN",
                                        "cycles": None, "reason": "fast estimate unavailable: " + type(error).__name__}


def cycles_of(performance) -> int | None:
    """The fast estimate's cycles from a performance record, when one was estimated."""
    block = (performance or {}).get("fast_estimate") if isinstance(performance, dict) else None
    if isinstance(block, dict) and block.get("status") == "ESTIMATED" and _positive(block.get("cycles")):
        return int(block["cycles"])
    return None


def label(performance) -> str | None:
    """``estimated: fast census estimate (...)`` for rankings, or ``None`` without an estimate."""
    if cycles_of(performance) is None:
        return None
    block = performance["fast_estimate"]
    low, high = block.get("low_cycles"), block.get("high_cycles")
    band = f"; band {low}-{high}" if low and high else ""
    bound = "; lower bound (unproved loop counts)" if block.get("lower_bound") else ""
    return f"estimated: fast census estimate{band}{bound}"


__all__ = [
    "EVIDENCE_KIND",
    "LEDGER",
    "SCHEMA",
    "attach",
    "calibration",
    "census",
    "cycles_of",
    "estimate",
    "fit",
    "floor",
    "for_case",
    "label",
    "record",
    "trace_for",
]
