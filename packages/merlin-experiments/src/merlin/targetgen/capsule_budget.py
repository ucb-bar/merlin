"""The per-capsule wall budget, raised to cover what this run already measured a capsule needs.

One number, the grade's per-capsule ``timeout`` (``--qa-timeout`` for in-turn and formal grades, the
self-check's ``--timeout``), bounds every step of a capsule, including its elaborated-RTL run. When it
is below what a capsule is known to cost, that capsule's cert tier can never finish: in one run a
capsule whose gSIM run took 559 s in a self-check was killed after 287 s by a 300 s budget in every
later grade, and stayed ``cert_not_measured`` to the end with nothing naming the budget as the cause.

The budget is derived only from this run's own completed results (:func:`capsule_scheduling.
selected_history`), never from a constant about a target. It can only RAISE the requested value, by
at most :data:`MAX_MULTIPLE`, so it still bounds a hang. A tier that completed prices it at
:data:`SAFETY` times its wall; a tier that was stopped on its wall clock proves only a lower bound, so
the next grade gets :data:`SAFETY` times that wall, and a capsule that keeps timing out escalates to the
cap instead of failing the same way forever. Carried certificates were not run and price nothing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

#: Margin over a measured wall, the same 2x the measured-T_obs certificate budget applies.
SAFETY = 2.0
#: The derived budget never exceeds this multiple of the requested one.
MAX_MULTIPLE = 4
#: Absolute ceiling, the CLI's own bound on a per-capsule budget.
CEILING_S = 86400
_TIMEOUT_CATEGORIES = {"testbench_timeout", "TESTBENCH_TIMEOUT"}


def _seconds(value) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _observations(document: Mapping):
    """``(tier, wall_s, completed)`` for each tier this result actually ran and timed."""
    failure = document.get("failure") if isinstance(document.get("failure"), Mapping) else {}
    stopped_tier = failure.get("tier") if failure.get("category") in _TIMEOUT_CATEGORIES else None
    for tier, record in (document.get("tiers") or {}).items():
        if not isinstance(record, Mapping) or record.get("measured_now") is not True:
            continue
        timing = record.get("timing")
        if not isinstance(timing, Mapping):
            continue
        wall = _seconds(timing.get("adapter_wall_s"))
        if timing.get("partial") is True:
            # Stopped before its split: only a wall-clock stop says the work needs more than this.
            if tier == stopped_tier and wall is not None:
                yield str(tier), wall, False
            continue
        if wall is None:
            active, build = _seconds(timing.get("sim_active_s")), _seconds(timing.get("build_s"))
            wall = None if active is None else active + (build or 0.0)
        if wall is not None:
            yield str(tier), wall, True


def measured_capsule_budget(
    capsules: Sequence[Mapping], runs_roots: Iterable[str | Path], *, timeout: int | float
) -> dict:
    """The budget for ``capsules`` and the measurement it rests on.

    ``runs_roots`` are earlier runs roots of this run (``<root>/runs/<suite>/<capsule>/``). The record
    always states the requested value, the value used and, when raised, which capsule and tier set it.
    """
    requested = max(1, int(timeout))
    names = {str(capsule.get("name")) for capsule in capsules if capsule.get("name")}
    record = {"requested_s": requested, "budget_s": requested, "raised": False, "basis": None}
    if not names:
        return record
    worst = None
    for root in runs_roots:
        base = Path(root)
        if not base.is_absolute() or not base.is_dir():
            continue
        for path in (base / "runs").glob("*/*/capsule_result.json"):
            if path.parent.name not in names:
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # a result still being written is not history yet
            if not isinstance(document, dict):
                continue
            for tier, wall, completed in _observations(document):
                need = math.ceil(SAFETY * wall)
                if worst is None or need > worst[0]:
                    worst = (need, path.parent.name, tier, wall, completed, base.name)
    if worst is None or worst[0] <= requested:
        return record
    need, capsule, tier, wall, completed, source = worst
    cap = min(CEILING_S, MAX_MULTIPLE * requested)
    budget = max(requested, min(need, cap))
    record.update(
        budget_s=budget,
        raised=budget > requested,
        capped=need > cap,
        capsule=capsule,
        tier=tier,
        measured_wall_s=round(wall, 3),
        measured=("completed" if completed else "stopped on its wall clock"),
        history=source,
        basis=(
            f"{capsule} {tier} {'completed in' if completed else 'was stopped on its wall clock after'} "
            f"{wall:.0f} s in {source}; the requested {requested} s per-capsule budget is raised to "
            f"{budget} s ({SAFETY:g}x that wall{', capped at ' + str(MAX_MULTIPLE) + 'x the request' if need > cap else ''})"
        ),
    )
    return record


def timed_out(results: Sequence[Mapping], budget: Mapping) -> dict:
    """``{capsule: {tier, budget_s, ...}}`` for each result stopped by this grade's wall budget."""
    stopped = {}
    for result in results:
        failure = result.get("failure") if isinstance(result.get("failure"), Mapping) else {}
        if failure.get("category") not in _TIMEOUT_CATEGORIES:
            continue
        stopped[str(result.get("capsule"))] = {
            "tier": failure.get("tier"),
            "status": result.get("status"),
            "budget_s": budget.get("budget_s"),
            "requested_s": budget.get("requested_s"),
            "remedy": (
                "the per-capsule wall budget ran out before this tier finished, so it is unmeasured, "
                "not failed; raise the grade's per-capsule timeout (--qa-timeout for in-turn and formal "
                "grades, --timeout for a self-check) above this capsule's measured cost"
            ),
        }
    return stopped
