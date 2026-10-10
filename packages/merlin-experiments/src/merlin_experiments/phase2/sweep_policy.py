"""The authoring gSIM sweep's declared environment: its fan-out and its member budget."""

from __future__ import annotations

import os

from .contracts import StageGateError

SWEEP_WORKERS_ENV = "MERLIN_PERF_SWEEP_WORKERS"
#: Roofline-floor cycles above which a tuning member is NOT swept on gSIM during authoring: it is
#: measured on gSIM only in the final measurement cells. Unset: every member is swept.
AUTHORING_GSIM_FLOOR_BUDGET_ENV = "MERLIN_PERF_AUTHORING_GSIM_FLOOR_BUDGET"


def authoring_gsim_floor_budget() -> int | None:
    """The declared authoring gSIM budget (roofline-floor cycles), or None when every member is swept."""
    raw = (os.environ.get(AUTHORING_GSIM_FLOOR_BUDGET_ENV) or "").strip()
    if not raw:
        return None
    try:
        budget = int(raw)
    except ValueError:
        raise StageGateError(f"{AUTHORING_GSIM_FLOOR_BUDGET_ENV}={raw!r} is not an integer cycle budget") from None
    if budget <= 0:
        raise StageGateError(f"{AUTHORING_GSIM_FLOOR_BUDGET_ENV}={budget} is not a positive cycle budget")
    return budget


def sweep_workers() -> int:
    """The declared sweep fan-out, or 1. Refuses a value it cannot read rather than guessing one."""
    raw = (os.environ.get(SWEEP_WORKERS_ENV) or "").strip()
    if not raw:
        return 1
    try:
        workers = int(raw)
    except ValueError:
        raise StageGateError(f"{SWEEP_WORKERS_ENV}={raw!r} is not an integer; refusing to guess a fan-out") from None
    if workers < 1:
        raise StageGateError(f"{SWEEP_WORKERS_ENV}={workers} is not a positive fan-out")
    return workers
