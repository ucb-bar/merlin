"""The simulated-cycle bound of one budgeted gSIM measurement, derived from its wall budget.

The selected gSIM command carries a ``+max-cycles`` hang bound. That bound counts every simulated cycle
of the run -- boot, the host's input preparation, the kernel and the coherent output export -- while the
measurement lane (:mod:`.feedback.idle_gsim`) budgets wall seconds from the kernel's cycles. A case whose
whole run needs more simulated cycles than the selected bound is stopped by the harness (its cycle-cap
exit status) long before its wall budget ends, whatever that budget is: in one campaign 20 of 34 lane
jobs ended this way after about 700 s, against budgets of 520 to 11,327 s.

This module carries the budget through to the emulator instead:

* :func:`engine_runs` reads, from a measurement child's own invocation records, how many simulated
  cycles each gSIM harness run reached in how many wall seconds (the harness's completion line, or its
  cycle bound when it stopped there);
* :func:`engine_rate` summarizes the runs the lane's ledger holds into the engine's observed rate;
* :func:`engine_cycle_bound` turns a job's wall budget into the simulated cycles it may reach (the
  fastest observed rate, with the observed spread as margin), so the wall budget -- not the cap -- ends
  the run; without an observed rate the selected bound stays and the result says so;
* :func:`engine_cycle_budget` applies a job's derived bound to the gSIM backend of this process tree
  (:data:`merlin.runtime.backends.chipyard_rocc_bound.GSIM_BUDGET_MAX_CYCLES_ENV`); it only ever raises the
  selected bound, and the selected wall deadline still bounds the process.

Nothing here raises into the lane or the grader: unreadable records are skipped, never estimated.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

_CYCLE_FLAG = "+max-cycles="
_COMPLETION = "Cycles:"
_MAX_RECORD_BYTES = 1 << 20
_MAX_STREAM_BYTES = 16 << 20


def _cap_exit() -> int | None:
    try:
        from merlin.runtime.backends import chipyard_rocc

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
    from merlin.runtime.backends.chipyard_rocc_bound import GSIM_BUDGET_MAX_CYCLES_ENV as ENV

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


__all__ = [
    "engine_ceiling",
    "engine_cycle_bound",
    "engine_cycle_budget",
    "engine_rate",
    "engine_runs",
    "engine_stop",
]
