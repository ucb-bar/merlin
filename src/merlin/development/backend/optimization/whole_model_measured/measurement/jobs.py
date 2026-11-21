"""The job record every owner of a whole-model measurement reads and writes: states, roles, result shape.

A measurement is a JOB, requested, run in a detached process and read back whenever it lands, keyed
by the digest of the exact package bytes measured (see :mod:`.service`).  This module holds the
vocabulary those owners share so no two of them spell a state differently.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.analysis.whole_model.grading import verdict as V
from merlin.development.backend.optimization.whole_model_measured.measurement.identity import now

# ---- jobs --------------------------------------------------------------------------------------

JOB_SCHEMA = "merlin_whole_model_job_v1"

RESULT_SCHEMA = "merlin_whole_model_measurement_result_v1"

PENDING, RUNNING, DONE, FAILED, SUPERSEDED = "pending", "running", "done", "failed", "superseded"

SCREENING = "screening"

SCREEN_FAILED = "screen_failed"

BOARD = "awaiting_board"

TERMINAL = (DONE, FAILED, SUPERSEDED, SCREEN_FAILED)

TIMING_PENDING = "PENDING"

ROLE_REFERENCE = "reference"

ROLE_CANDIDATE = "candidate"

LOCAL_VERDICT = "local_verdict.json"

LOCAL_VERDICT_SCHEMA = "merlin_whole_model_local_verdict_v1"

BOARD_REQUEST = "board_request.json"

AWAITING_BOARD = "_awaiting_board"

ATTRIBUTION_ONLY = "invalid, attribution only"

ATTRIBUTION_FILE = "attribution.json"

ATTRIBUTION_SCHEMA = "merlin.phase2.whole_model_measured.attribution.v1"

ATTRIBUTION_PENDING = "pending"

ATTRIBUTION_AUTHORED = "authored"

ATTRIBUTION_UNAUTHORED = "unauthored"

UNATTRIBUTED = (ATTRIBUTION_PENDING, ATTRIBUTION_UNAUTHORED)


def attribution_state(record: Mapping[str, Any] | None) -> str | None:
    return (record or {}).get("state")


def attributable(record: Mapping[str, Any] | None) -> bool:
    """Whether a job whose attribution record is ``record`` may become the best or a champion."""
    return attribution_state(record) not in UNATTRIBUTED


def merge_attribution(
    record: Mapping[str, Any] | None, change: Mapping[str, Any], *, at: str, created: bool
) -> dict[str, Any]:
    """The attribution record after one ``change`` (``state``, ``round``, ``why``).  Every change is kept in
    ``history``; the state moves unless the job is authored, or is an EXISTING job the harness requested
    (``created`` says the request carrying ``change`` is the one creating the job)."""
    state = str(change["state"])
    if state not in (ATTRIBUTION_PENDING, ATTRIBUTION_AUTHORED, ATTRIBUTION_UNAUTHORED):
        raise ServiceError(f"unknown attribution state {state!r}")
    entry = {"state": state, "round": change.get("round"), "why": change.get("why"), "at": at}
    current = attribution_state(record) if record is not None else None
    harness = (record is None and not created) or (record is not None and current is None)
    return {
        "schema": ATTRIBUTION_SCHEMA,
        "state": current if (current == ATTRIBUTION_AUTHORED or harness) else state,
        "history": [*((record or {}).get("history") or ()), entry],
    }


ATTEMPTS_DIR = "attempts"

ATTEMPT_RECORD = "attempt.json"

ATTEMPT_SCHEMA = "merlin.phase2.whole_model_measured.attempt.v1"

ANNOTATIONS_FILE = "annotations.json"

ANNOTATIONS_SCHEMA = "merlin.phase2.whole_model_measured.annotations.v1"

RESULT_FILE = "result.json"

KEPT_ACROSS_ATTEMPTS = frozenset(
    {
        "job.json",
        "package",
        ".lock",
        ATTRIBUTION_FILE,
        ATTEMPTS_DIR,
        "pre_measure_check.json",
        "pre_measure_check_result.json",
        "selfcheck_out",
    }
)

ARCHIVED_ATTEMPT_PREFIXES = ("lost_attempt_", "unlinkable_attempt_", "paused_attempt_", "board_lost_attempt_")

INFRA_WORKER_LOST = "infra_worker_lost"

INFRA_BOARD_OBJECTS_MISSING = "infra_board_objects_missing"

BOARD_UNAVAILABLE_NOTICE = "infra_board_unavailable: board unavailable, measurement deferred"

CONTROL_DRIFT_NOTICE = (
    "infra_control_drift: the batch's control measured differently than alone, so the batch's timing is "
    "not trusted; re-measured alone (not a verdict)"
)


class ServiceError(RuntimeError):
    """The service cannot accept or run this request."""


def result(job: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    """A result document for ``job``: its identity fields, then ``fields``."""
    return {
        "schema": RESULT_SCHEMA,
        "package_sha256": job["package_sha256"],
        "label": job.get("label"),
        "requested_at": job.get("requested_at"),
        "finished_at": now(),
        "machine": job.get("machine"),
        "builder": job.get("builder"),
        # WHAT THE MACHINE COULD DO when this was measured, and what it lacked against its peers.
        "machine_capabilities": job.get("machine_capabilities"),
        **fields,
    }


def refused(job: Mapping[str, Any], reason: str, **fields: Any) -> dict[str, Any]:
    return result(job, timing_status=V.TIMING_REFUSED, objective_cycles=None, refusal=reason, **fields)


def failing_summary(document: Mapping[str, Any] | None, *, limit: int = 40) -> list[dict[str, Any]]:
    """``[{group, op, detail}]`` for every failed group (or failing capsule) of a result, compactly."""
    if not isinstance(document, Mapping):
        return []
    verdict = document.get("verdict") or {}
    routes = {
        str(r.get("group")): r for r in ((document.get("build") or {}).get("groups") or ()) if isinstance(r, Mapping)
    }
    rows = []
    # Groups where a divergence STARTS lead; groups that only inherited different inputs follow.
    for row in sorted(V.group_table(verdict), key=lambda row: bool(row.get("inherited_from"))):
        if row.get("state") != V.GROUP_FAILED:
            continue
        group = str(row.get("group"))
        rows.append(
            {
                "group": group,
                "op": (routes.get(group) or {}).get("op") or row.get("kind"),
                "detail": str(row.get("detail") or "")[:160],
            }
        )
    if not rows and document.get("screen_failed"):
        summary = (document.get("pre_measure_check") or {}).get("summary") or {}
        for name, mismatch in (summary.get("mismatches") or {}).items():
            rows.append(
                {
                    "capsule": name,
                    "detail": f"{mismatch.get('mismatch_count')} mismatches, max_abs {mismatch.get('max_abs_diff')}"[
                        :160
                    ],
                }
            )
    return rows[:limit]


def archive_attempt(
    job_dir: Path,
    prefix: str,
    count: int = 0,
    *,
    keep_extra: Sequence[str] = (),
    why: str | None = None,
    retracted: bool = False,
) -> Path:
    """Move everything a partial (or finished) attempt left in ``job_dir`` aside, as ``attempts/<n>/``.

    ``prefix`` names WHY (``lost_attempt_``, ``paused_attempt_``, ...: recorded as the attempt's
    ``kind``); ``count`` is accepted for the callers that used to number by kind and is not needed --
    attempts are numbered once, in the order they were set aside.  Never a deletion: the archive is kept
    (its build trees are pruned by retention) and its ``result.json`` is moved, not rewritten, so a
    finished measurement survives every later attempt (:mod:`.attempts` reads them back).  The job
    record, the package snapshot and the screen results stay, so the retry starts clean and nothing the
    earlier attempt wrote is read as the later one's.  ``retracted`` says the attempt's verdict no longer
    stands (an operator's decision, with ``why``); otherwise a verdict it holds stays the job's until a
    later attempt reaches one of its own."""
    del count  # numbering is the attempts directory's own
    job_dir = Path(job_dir)
    root = job_dir / ATTEMPTS_DIR
    root.mkdir(exist_ok=True)
    taken = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    index = max(taken, default=-1) + 1
    attempt = root / str(index)
    attempt.mkdir()
    moved = []
    for entry in sorted(job_dir.iterdir()):
        if (
            entry.name in KEPT_ACROSS_ATTEMPTS
            or entry.name in keep_extra
            or entry.name.startswith(ARCHIVED_ATTEMPT_PREFIXES)
        ):
            continue
        shutil.move(str(entry), str(attempt / entry.name))
        moved.append(entry.name)
    held = _read(attempt / RESULT_FILE)
    record = {
        "schema": ATTEMPT_SCHEMA,
        "attempt": index,
        "kind": str(prefix).rstrip("_") or "attempt",
        "archived_at": now(),
        "why": why,
        "retracted": bool(retracted),
        "moved": moved,
        "result": None
        if held is None
        else {k: held.get(k) for k in ("timing_status", "objective_cycles", "finished_at", "refusal")},
    }
    (attempt / ATTEMPT_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return attempt


class ResultExists(ServiceError):
    """A job directory already holds a result; a result is written once and never replaced."""


def write_result(job_dir: Path, document: Mapping[str, Any]) -> Path:
    """Write ``job_dir``'s ``result.json`` ONCE: the bytes land under a temporary name and are LINKED into
    place, which fails when a result is already there -- so a late worker, a second batch or a resubmit
    can never overwrite a finished measurement (:class:`ResultExists`).  A new measurement of the same
    job is a new attempt (:func:`archive_attempt` first)."""
    path = Path(job_dir) / RESULT_FILE
    temporary = path.with_name(f".{RESULT_FILE}.{os.getpid()}.{time.monotonic_ns()}")
    temporary.write_text(json.dumps(dict(document), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ResultExists(f"{path} already holds a result; a result is never overwritten") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def preserve_result(job_dir: Path, document: Mapping[str, Any], *, why: str) -> Path:
    """Keep a result that arrived for a job which already has one, as an attempt of its own (never in place
    of the existing result): ``attempts/<n>/result.json`` with ``kind: late_result`` and why."""
    root = Path(job_dir) / ATTEMPTS_DIR
    root.mkdir(exist_ok=True)
    taken = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    attempt = root / str(max(taken, default=-1) + 1)
    attempt.mkdir()
    write_result(attempt, document)
    record = {
        "schema": ATTEMPT_SCHEMA,
        "attempt": int(attempt.name),
        "kind": "late_result",
        "archived_at": now(),
        "why": why,
        "retracted": False,
        "moved": [],
        "result": {k: document.get(k) for k in ("timing_status", "objective_cycles", "finished_at", "refusal")},
    }
    (attempt / ATTEMPT_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return attempt


def _read(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


__all__ = [
    "ANNOTATIONS_FILE",
    "ARCHIVED_ATTEMPT_PREFIXES",
    "ATTEMPTS_DIR",
    "ATTEMPT_RECORD",
    "ATTRIBUTION_AUTHORED",
    "ATTRIBUTION_FILE",
    "ATTRIBUTION_ONLY",
    "ATTRIBUTION_PENDING",
    "ATTRIBUTION_UNAUTHORED",
    "AWAITING_BOARD",
    "BOARD",
    "BOARD_REQUEST",
    "DONE",
    "FAILED",
    "JOB_SCHEMA",
    "LOCAL_VERDICT",
    "PENDING",
    "RESULT_FILE",
    "RESULT_SCHEMA",
    "ROLE_CANDIDATE",
    "ROLE_REFERENCE",
    "RUNNING",
    "ResultExists",
    "SCREENING",
    "SCREEN_FAILED",
    "SUPERSEDED",
    "ServiceError",
    "TERMINAL",
    "TIMING_PENDING",
    "UNATTRIBUTED",
    "archive_attempt",
    "attributable",
    "attribution_state",
    "failing_summary",
    "merge_attribution",
    "preserve_result",
    "refused",
    "result",
    "write_result",
]
