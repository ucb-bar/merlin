"""What the author agent can actually see of its own feedback, read from its workspace.

Every other source reads the operator's records; this one reads only the files the agent reads, so a
grade that finished but never reached the agent, a verdict without rows or a stale progress stream is
visible here instead of looking like an idle grader. Bounded reads only: the newest operator grade in
``run_dir/qa_history``, the agent's ``qa/verdict.json``, ``qa/latest_completed_grade.json`` and
``qa/progress.json``, ``OPTIMIZATION_SUGGESTIONS.json``, and the names (not contents) of the self-check
channel plus the newest few responses. Display telemetry; it grades nothing.

Section::

    {"workspace", "live", "verdict": {present, at, rows, snapshot_current, graded_source, graded_at,
     notices: [..], headroom}, "latest_completed": {...same...}, "progress": {present, at, rows,
     graded_source}, "suggestions": {present, suggestions, pending_workload_check, discovery},
     "selfchecks": {requests, answered, pending, oldest_pending_at, last_response_at, recent_errors,
     last_error, recent_refusals}, "newest_grade": {token, at, rows, executed, graded_source,
     delivered: [file names]},
     "notes"}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...progress_health import policy_refusal
from ..io import clean, epoch, mtime, num

#: Grade files examined in the operator history and self-check responses opened per refresh.
MAX_GRADES = 4096
RECENT_RESPONSES = 6
NOTICE_KEYS = ("package_failure_notice", "grade_error_notice", "operator_epoch_notice")
#: Row statuses of a case the grade never ran (admission deferred, gated, or carried by regression
#: sampling with its earlier result): nothing to deliver.
NOT_EXECUTED = frozenset({"gated", "deferred", "carried"})


def executed_rows(rows) -> int:
    """Rows of cases the grade actually ran; deferred and gated rows carry no result."""
    if not isinstance(rows, list):
        return 0
    return sum(
        1
        for row in rows
        if isinstance(row, dict) and row.get("status") not in NOT_EXECUTED and not row.get("background_deferred")
    )


def _headroom(document: dict) -> dict | None:
    brief = document.get("feedback_brief") if isinstance(document.get("feedback_brief"), dict) else {}
    priority = brief.get("performance_priority") if isinstance(brief.get("performance_priority"), dict) else {}
    block = priority.get("development_headroom")
    if not isinstance(block, dict):
        return None
    counts = block.get("counts") if isinstance(block.get("counts"), dict) else {}
    return {
        "ranked": num(counts.get("ranked")),
        "passing_l2": num(counts.get("passing_l2")),
        "median_utilization_pct": num(block.get("median_array_utilization_pct")),
    }


def _grade_file(ctx, path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    document = ctx.reader.document(path, limit=64 << 20, stale_ok=True)
    if not isinstance(document, dict):
        return {"present": True, "at": mtime(path), "unreadable": True}
    rows = document.get("per_capsule")
    latest = document.get("latest_completed_grade") if isinstance(document.get("latest_completed_grade"), dict) else {}
    return {
        "present": True,
        "at": mtime(path),
        "rows": len(rows) if isinstance(rows, list) else 0,
        "snapshot_current": latest.get("snapshot_current", document.get("snapshot_current")),
        "graded_source": clean(latest.get("graded_source_sha256") or document.get("graded_source_sha256"), 64) or None,
        "graded_at": epoch(latest.get("graded_at") or document.get("graded_at")),
        "notices": [key for key in NOTICE_KEYS if document.get(key)],
        "headroom": _headroom(document),
        "model_gate_only": set(document) <= {"model_gate", "all_pass", "operator_epoch_notice"},
    }


def _newest_grade(ctx, run_dir: Path) -> dict[str, Any] | None:
    history = run_dir / "qa_history"
    best = None
    try:
        with os.scandir(history) as entries:
            for count, entry in enumerate(entries):
                if count >= MAX_GRADES:
                    break
                name = entry.name
                if not (name.startswith("verdict_fast_") and name.endswith(".json")):
                    continue
                token = name[len("verdict_fast_") : -len(".json")]
                if token.isdigit() and (best is None or int(token) > int(best[0])):
                    best = (token, Path(entry.path))
    except OSError:
        return None
    if best is None:
        return None
    document = ctx.reader.document(best[1], limit=64 << 20) or {}
    rows = document.get("per_capsule")
    return {
        "token": best[0],
        "at": epoch(document.get("graded_at")) or mtime(best[1]),
        "rows": len(rows) if isinstance(rows, list) else 0,
        "executed": executed_rows(rows),
        "graded_source": clean(document.get("graded_source_sha256"), 64) or None,
        "path": str(best[1]),
    }


def _selfchecks(ctx, channel: Path) -> dict[str, Any]:
    requests: dict[str, float | None] = {}
    responses: dict[str, str] = {}
    try:
        with os.scandir(channel) as entries:
            for entry in entries:
                name = entry.name
                if not name.endswith(".json"):
                    continue
                if name.startswith("req_"):
                    rid = name[4:-5]
                    parts = rid.split("_")
                    # req_<lane>_<ns>_<nonce> or req_op_idle_<ns>_<nonce>: the first long number is the time.
                    stamp = next((int(p) for p in parts if p.isdigit() and len(p) >= 16), None)
                    requests[rid] = epoch(stamp) if stamp else None
                elif name.startswith("resp_"):
                    responses[name[5:-5]] = entry.path
    except OSError:
        return {"requests": 0, "answered": 0, "pending": 0}
    pending = [at for rid, at in requests.items() if rid not in responses]
    newest = sorted(responses.items(), key=lambda kv: mtime(kv[1]) or 0, reverse=True)[:RECENT_RESPONSES]
    errors, refusals = [], 0
    for _rid, path in newest:
        document = ctx.reader.document(Path(path), limit=8 << 20)
        if isinstance(document, dict) and document.get("error"):
            if policy_refusal(document):
                refusals += 1  # refused before any simulator ran, and asked again later
            else:
                errors.append(clean(document.get("error"), 160))
    known = [at for at in pending if at is not None]
    return {
        "requests": len(requests),
        "answered": sum(1 for rid in requests if rid in responses),
        "pending": len(pending),
        "oldest_pending_at": min(known) if known else None,
        "last_response_at": mtime(newest[0][1]) if newest else None,
        "recent_errors": len(errors),
        "last_error": errors[0] if errors else None,
        "recent_refusals": refusals,
    }


def read(ctx) -> dict[str, Any]:
    experiment = ctx.experiment
    workspace = Path(experiment.get("workspace") or "")
    run_dir = Path(experiment.get("run_dir") or "")
    if not experiment.get("workspace") or not workspace.is_dir():
        return {"workspace": None, "live": ctx.live, "notes": ["no agent workspace recorded for this run"]}
    qa = workspace / "qa"
    verdict = _grade_file(ctx, qa / "verdict.json")
    latest = _grade_file(ctx, qa / "latest_completed_grade.json")
    progress_path = qa / "progress.json"
    progress: dict[str, Any] = {"present": progress_path.is_file()}
    if progress["present"]:
        document = ctx.reader.document(progress_path, limit=64 << 20, stale_ok=True) or {}
        rows = document.get("per_capsule")
        progress.update(
            at=epoch(document.get("updated_at")) or mtime(progress_path),
            rows=len(rows) if isinstance(rows, list) else 0,
            graded_source=clean(document.get("graded_source_sha256"), 64) or None,
        )
    suggestions_path = workspace / "OPTIMIZATION_SUGGESTIONS.json"
    suggestions: dict[str, Any] = {"present": suggestions_path.is_file()}
    if suggestions["present"]:
        document = ctx.reader.document(suggestions_path, limit=16 << 20) or {}
        for key in ("suggestions", "pending_workload_check", "discovery"):
            value = document.get(key)
            suggestions[key] = len(value) if isinstance(value, list) else None
    newest = _newest_grade(ctx, run_dir) if run_dir.is_dir() else None
    if newest is not None:
        delivered = []
        for label, view in (
            ("verdict.json", verdict),
            ("latest_completed_grade.json", latest),
            ("progress.json", progress),
        ):
            if (
                view.get("present")
                and view.get("rows")
                and newest["graded_source"]
                and (
                    view.get("graded_source") == newest["graded_source"]
                    or (view.get("graded_at") or view.get("at") or 0) >= (newest["at"] or 0)
                )
            ):
                delivered.append(label)
        newest["delivered"] = delivered
    return {
        "workspace": str(workspace),
        "live": ctx.live,
        "verdict": verdict,
        "latest_completed": latest,
        "progress": progress,
        "suggestions": suggestions,
        "selfchecks": _selfchecks(ctx, workspace / ".qa_channel"),
        "newest_grade": newest,
        "notes": [],
    }
