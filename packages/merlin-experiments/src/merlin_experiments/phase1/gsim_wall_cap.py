"""The operator's wall-time cap on one gSIM job, read from the campaign policy.

A gSIM run is worth its slot only while it returns soon enough to steer the next edit. The policy
sections ``gsim`` (witnesses), ``gsim_correctness`` and ``gsim_measurement`` may each carry
``max_job_wall_s``; nothing here supplies a default, so a policy without it keeps its previous
behaviour. Where a cap is selected, a job's wall time is PREDICTED before it is launched
(:mod:`.gsim_wall_predictor`), so the cap is never how a too-long job is discovered:

* the idle measurement lane launches a job only when its projection fits the cap, with that projection
  (at most the cap) as its timeout, carried through to the emulator (:mod:`.gsim_engine_bound`);
* an author or promotion gSIM request projected beyond its section's cap is refused before it takes
  or waits for a slot, and the refusal shows the projection and what to use instead (:func:`message`).

A prediction needs the case's cycles and declared bytes and enough measured jobs; without them nothing
is refused here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

KEY = "max_job_wall_s"
#: Every reason this module gives starts with it (the lane logs such a skip once per case).
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
    from .gsim_wall_predictor import features

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
            case = features(row.get("barrier_cycles") or performance.get("estimated_whole_soc_cycles"),
                            performance.get("structural_roofline"))
            if case is not None:
                found[name] = case
    if not found:
        return None
    return max(found.values(), key=lambda case: (case["read_bytes"] + case["write_bytes"], case["cycles"]))


def request_refusal(capsules) -> str | None:
    """Why a gSIM request for ``capsules`` must not start (projected beyond its cap), else ``None``."""
    try:
        from . import joint_control
        from . import joint_gsim_measurement as M
        from .gsim_wall_predictor import campaign_model, decide

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


__all__ = ["KEY", "SECTIONS", "SKIP", "max_job_wall_s", "message", "request_refusal", "validate"]
