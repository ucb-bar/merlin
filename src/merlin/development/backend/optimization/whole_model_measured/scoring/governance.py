"""The whole-model objective a phase-2 ``whole_model_measured`` loop optimises.

TWO MACHINES, TWO JOBS, NEVER ONE NUMBER.  The SCREEN (a board: a whole model in minutes) measures
EVERY candidate the loop produces, so an agent sees the consequence of an edit while it still
remembers making it.  The CERTIFIER (the elaborated-RTL emulator: hours per model) runs only when a
candidate becomes the new best on the screen, and a best that fails there is RETRACTED.  Each
machine has its own reference arm -- the vendor library measured on THAT machine -- and every
comparison is made against the reference on the same device.  A cycle count from one machine is
never compared with one from the other.

A best must also hold up SOLO: a promoted best is re-measured alone (``repeats_on_best``), and a best
whose solo repeat does not beat the previous best's solo result by more than the NOISE MARGIN is
demoted to a tie.  The noise margin is the median batched-vs-solo repeat spread measured in THIS
store, floored at :data:`NOISE_FLOOR` -- a candidate crowned on a margin narrower than the noise the
board itself produces on a rerun is not a measured improvement, it is the spread.

``measure`` is the loop's per-candidate hook.  It never blocks on a simulator: it requests a
measurement of the exact bytes and returns what is known about them now -- a result, or ``PENDING``
-- together with the feedback the agent acts on.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.analysis.whole_model.build.inputs import resolve_capsules
from merlin.analysis.whole_model.grading import verdict as V
from merlin.development.backend.optimization.whole_model_measured.measurement import board_batches as BATCH
from merlin.development.backend.optimization.whole_model_measured.measurement import jobs as J
from merlin.development.backend.optimization.whole_model_measured.measurement.board_batches import (
    PROMOTED_REPEAT_PRIORITY,
)
from merlin.development.backend.optimization.whole_model_measured.measurement.identity import (
    epoch_of,
    package_digest,
    write_json_atomic,
)
from merlin.development.backend.optimization.whole_model_measured.measurement.jobs import (
    ROLE_CANDIDATE,
    TIMING_PENDING,
)
from merlin.development.backend.optimization.whole_model_measured.measurement.service import (
    MeasurementService,
)
from merlin.development.backend.optimization.whole_model_measured.scoring import feedback as F
from merlin.development.backend.optimization.whole_model_measured.scoring import gates as G
from merlin.development.backend.optimization.whole_model_measured.scoring import transfer as TC

# ---- objective ---------------------------------------------------------------------------------

DONE_STATE = "done"

NOISE_FLOOR = 0.001

CERT_KEEP_FRACTION = 0.5


def _key_of(job: Mapping[str, Any]) -> str:
    return str(job.get("job_key") or job["package_sha256"])


def _epoch(stamp: Any) -> float | None:
    return epoch_of(stamp)


COMPACT_TEXT = 2200

COMPACT_FAILING = 12

WAIT_MAX_SECONDS = 540.0

WAIT_POLL_SECONDS = 10.0


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 40] + f"\n... [{len(text) - limit + 40} chars clipped]"


def _compact(result: Mapping[str, Any]) -> dict[str, Any]:
    verdict = result.get("verdict") or {}
    return {
        "timing_status": result.get("timing_status"),
        "job_state": result.get("job_state"),
        "whole_window_cycles": verdict.get("whole_window_cycles"),
        "objective_cycles": result.get("objective_cycles"),
        "correctness": verdict.get("correctness"),
        "vendor_also_fails_count": verdict.get("vendor_also_fails_count"),
        "refusal": result.get("refusal") or verdict.get("refusal"),
        "invalid_reason": verdict.get("invalid_reason"),
        "pre_measure_check": {
            key: (result.get("pre_measure_check") or {}).get(key) for key in ("label", "required", "passed", "summary")
        }
        if result.get("pre_measure_check")
        else None,
        "device": (result.get("device") or {}).get("artifact"),
        "job_id": (result.get("run") or {}).get("job_id"),
    }


__all__ = [
    "CERTIFIED",
    "CERT_PENDING",
    "DEFERRED_INFEASIBLE",
    "NOISE_FLOOR",
    "RETRACTED",
    "SCHEMA",
    "UNCERTIFIED",
    "WholeModelObjective",
    "resolve_capsules",
]
