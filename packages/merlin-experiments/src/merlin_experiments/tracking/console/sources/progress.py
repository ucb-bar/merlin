"""How far the experiment has come: grade progress, cohort coverage, the passing count over time.

Reads ``workspace/qa/progress.json`` (``merlin.phase1.grade_progress.v1``), the cohort projection of
:class:`..terminal_capsules.Capsules`, the bounded diagnostics of :class:`..progress_health.ProgressHealth`,
the run's ``qa_history/verdict_*.json`` and ``oot_commits.jsonl`` checkpoints (the passing-count series),
``plateau.json`` and the completion record ``joint_campaign_summary.json``. Retained-snapshot totals are
never presented as current-compiler coverage: ``snapshot_current`` travels with the counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import bounded_glob, clean, epoch, num, obj
from . import downsample

PROGRESS_SCHEMA = "merlin.phase1.grade_progress.v1"
MAX_VERDICTS = 160


def _series(ctx, run: Path) -> dict[str, list]:
    points: dict[float, int] = {}
    verdicts = sorted(bounded_glob(run / "qa_history", "verdict_*.json", 4096))[-MAX_VERDICTS:]
    for path in verdicts:
        doc = ctx.reader.document(path, limit=16 << 20) or {}
        t, passed = epoch(doc.get("graded_at")), num(doc.get("n_passed"))
        if t is not None and passed is not None:
            points[t] = int(passed)
    commits = ctx.state.setdefault("commits", [])
    ctx.reader.poll(run / "oot_commits.jsonl", commits.append, commits.clear)
    for row in commits:
        t, passed = epoch(obj(row.get("oot")).get("committed_at")), num(row.get("n_passed"))
        if t is not None and passed is not None:
            points.setdefault(t, int(passed))
    ordered = downsample(sorted(points.items()))
    return {"t": [t for t, _ in ordered], "n_passed": [n for _, n in ordered]}


def _completion(ctx, run: Path) -> dict[str, Any] | None:
    path = run / "joint_campaign_summary.json"
    if not path.is_file():
        return None
    doc = ctx.reader.document(path, limit=64 << 20)
    if doc is None:
        return {"reason": "completion record present but not read this refresh", "source": str(path)}
    return {
        "reason": clean(doc.get("completion_reason"), 80) or None,
        "converged": doc.get("converged"),
        "formal_complete": doc.get("formal_complete"),
        "elapsed_s": num(doc.get("elapsed_wall_s")),
        "rounds": len(doc["rounds"]) if isinstance(doc.get("rounds"), list) else num(doc.get("rounds")),
        "source": str(path),
    }


def _plateau(ctx, run: Path) -> dict[str, Any] | None:
    doc = ctx.reader.document(run / "plateau.json", limit=4 << 20)
    if not doc:
        return None
    keep = {
        k: doc.get(k)
        for k in ("plateaued", "status", "reason", "rounds_without_progress", "best_n_passed", "window", "since")
        if k in doc
    }
    return {k: (clean(v, 120) if isinstance(v, str) else v) for k, v in keep.items()} or None


def _cohorts(ctx, run: Path, workspace: Path | None) -> list[dict[str, Any]]:
    if workspace is None:
        return []
    from ...terminal_capsules import Capsules

    sampler = ctx.state.get("capsules")
    if sampler is None:
        sampler = ctx.state["capsules"] = Capsules(interval=10)
    value = sampler.sample(run, workspace)
    out = []
    for row in value.get("cohorts") or []:
        out.append(
            {
                "name": row.get("cohort"),
                "total": row.get("total"),
                "all": row.get("all"),
                "L2": row.get("L2"),
                "L3": row.get("L3"),
                "refusal": row.get("refusal"),
                "L3_obligation": row.get("L3_obligation"),
            }
        )
    if out and isinstance(value.get("total"), dict):
        total = value["total"]
        out.append(
            {
                "name": "total",
                "total": total.get("total"),
                "all": total.get("all"),
                "L2": total.get("L2"),
                "L3": total.get("L3"),
                "refusal": total.get("refusal"),
                "L3_obligation": None,
            }
        )
    return out


def _health(ctx, run: Path, workspace: Path | None) -> dict[str, Any]:
    if workspace is None:
        return {}
    from ...progress_health import ProgressHealth

    health = ctx.state.get("health")
    if health is None:
        task = Path(ctx.experiment.get("task") or run)
        health = ctx.state["health"] = ProgressHealth(task, operator_private=ctx.operator_private)
    return health.sample({"campaign": {"run_dir": str(run), "workspace": str(workspace)}}) or {}


def read(ctx) -> dict[str, Any]:
    run = Path(ctx.experiment["run_dir"])
    workspace = Path(ctx.experiment["workspace"]) if ctx.experiment.get("workspace") else None
    notes = []
    progress = ctx.reader.document(workspace / "qa" / "progress.json") if workspace else None
    if progress is not None and progress.get("schema") not in (None, PROGRESS_SCHEMA):
        notes.append(f"progress.json schema {clean(progress.get('schema'), 60)!r} not read")
        progress = None
    progress = progress or {}
    if not progress:
        notes.append("grade progress not recorded")
    health = _health(ctx, run, workspace)
    numeric = obj(health.get("numeric"))
    requests = health.get("requests") or []
    latest = obj(requests[0]) if requests else {}
    alerts = []
    from ...terminal_health import alert as alert_title

    for alert in health.get("alerts") or []:
        alert = obj(alert)
        detail = alert.get("message")
        if alert.get("category") or alert.get("stage"):
            # One row per (snapshot, failure plane/category, stage); name what failed and how often.
            detail = (
                f"{alert.get('count')} capsules: {alert.get('plane') or '?'} / {alert.get('category') or '?'}"
                f" at {alert.get('stage') or 'unknown stage'}"
            )
            if alert.get("capsules"):
                detail += (
                    " ("
                    + ", ".join(map(str, alert["capsules"][:3]))
                    + (", ..." if len(alert["capsules"]) > 3 else "")
                    + ")"
                )
        elif alert.get("count") is not None:
            detail = f"{alert.get('count')} occurrences. {detail or ''}"
        severity = {"warning": "warn", "error": "error"}.get(alert.get("severity"), "info")
        alerts.append(
            {
                "code": alert.get("code"),
                "severity": severity,
                "title": clean(alert_title({"alerts": [alert]}) or alert.get("code"), 120),
                "detail": clean(detail, 240),
            }
        )
    source = progress.get("graded_source_sha256")
    return {
        "state": progress.get("state"),
        "n_selected": num(progress.get("n_selected")),
        "n_completed": num(progress.get("n_completed")),
        "n_passed": num(progress.get("n_passed")),
        "n_execution_passed": num(progress.get("n_execution_passed")),
        "snapshot_current": progress.get("snapshot_current"),
        "graded_source": source[:12] if isinstance(source, str) else None,
        "updated_at": epoch(progress.get("updated_at")),
        "cohorts": _cohorts(ctx, run, workspace),
        "numeric": {
            "completed": numeric.get("completed"),
            "passed": numeric.get("passed"),
            "failed": numeric.get("failed"),
        },
        "series": _series(ctx, run),
        "completion": _completion(ctx, run),
        "plateau": _plateau(ctx, run),
        "alerts": alerts,
        "latest_request": (
            {
                "source": str(latest.get("source_sha256") or "")[:12] or None,
                "queue_wait_s": num(latest.get("queue_wall_s")),
                "wall_s": num(latest.get("started_to_completion_wall_s")),
                "stage": clean(latest.get("stage"), 60) or None,
            }
            if latest
            else None
        ),
        "notes": notes + [clean(n, 160) for n in health.get("notices") or []][:4],
    }


__all__ = ["read"]
