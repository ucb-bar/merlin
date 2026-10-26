"""How far the experiment has come: grade progress, cohort coverage, the passing count over time.

The background grader runs several grades at once, each of an immutable snapshot of the agent's
compiler. Each grade has an owner record ``_qa_work/background_snapshot_owners/<token>.json``
(``merlin.background_snapshot_owner.v1``), a progress record ``_qa_work/fruns_<token>/snapshot_progress.json``
rewritten as each capsule finishes, and, once it ends, ``qa_history/verdict_fast_<token>.json``. Those are
what move while a run works, so ``grades`` is read from them. ``workspace/qa/progress.json`` is only the
copy the grader last published to the agent (``published``), which can trail the grades by an hour.

Also read: the cohort projection of :class:`..terminal_capsules.Capsules`, the bounded diagnostics of
:class:`..progress_health.ProgressHealth`, the certificate ledger ``qa/tier_state.json`` and
``qa/joint_physical_correctness.json`` (``certificates``), ``oot_commits.jsonl``, ``plateau.json`` and
the completion record ``joint_campaign_summary.json``. A grade that ended before grading any capsule
is not a point of the passing-count series: it measured nothing. Retained-snapshot totals are never
presented as current-compiler coverage: ``snapshot_current`` travels with the counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import clean, epoch, num, obj, rows, scan_dir
from . import downsample

PROGRESS_SCHEMA = "merlin.phase1.grade_progress.v1"
OWNER_SCHEMA = "merlin.background_snapshot_owner.v1"
PHYSICAL_SCHEMA = "merlin.joint_physical_correctness.v1"
#: Grades kept in the section, newest first; older ones only feed the passing-count series.
MAX_GRADES = 8
MAX_SERIES_GRADES = 160
PASS = ("pass", "passed")
FAIL = ("fail", "failed", "mismatch")
ERROR = ("error", "timeout", "timed_out")


def _l2_awaiting_l3(row: dict) -> bool:
    """Passed the loop tier (Spike L2) and is waiting for its L3 certificate, which this grade does not run."""
    tiers = obj(row.get("tiers"))
    return (
        row.get("status") == "incomplete"
        and tiers.get("L2") in PASS
        and row.get("failure_tier") == "L3"
        and row.get("failure_plane") == "oracle_unavailable"
    )


def _tally(doc: dict) -> dict[str, int]:
    out = {"pass": 0, "fail": 0, "error": 0, "l2": 0}
    for row in rows(doc.get("per_capsule")):
        row = obj(row)
        status = row.get("status")
        key = "pass" if status in PASS else "fail" if status in FAIL else "error" if status in ERROR else None
        if key is None and _l2_awaiting_l3(row):
            key = "l2"
        if key:
            out[key] += 1
    return out


def _tokens(run: Path) -> list[str]:
    """Grade tokens (nanosecond stamps), newest first, from owner records and ``fruns_*`` directories."""
    work, found = run / "_qa_work", set()
    for entry in scan_dir(work / "background_snapshot_owners", 4096):
        if entry.name.endswith(".json"):
            found.add(entry.name.removesuffix(".json"))
    for entry in scan_dir(work, 8192):
        if entry.name.startswith("fruns_"):
            found.add(entry.name.removeprefix("fruns_"))
    return sorted((t for t in found if t.isascii() and t.isdigit()), key=int, reverse=True)


def _grade(ctx, run: Path, token: str, *, full: bool) -> dict[str, Any] | None:
    """One background grade: who owns it, how far it got, how it ended. ``full`` reads the per-capsule rows."""
    work = run / "_qa_work"
    owner = obj(ctx.reader.document(work / "background_snapshot_owners" / f"{token}.json", limit=1 << 20))
    if owner.get("schema") not in (None, OWNER_SCHEMA):
        owner = {}
    progress_path = work / f"fruns_{token}" / "snapshot_progress.json"
    progress = obj(ctx.reader.document(progress_path, limit=16 << 20))
    if progress.get("schema") not in (None, PROGRESS_SCHEMA):
        progress = {}
    verdict_path = run / "qa_history" / f"verdict_fast_{token}.json"
    verdict = obj(ctx.reader.document(verdict_path, limit=16 << 20)) if full or not progress else {}
    if not (owner or progress or verdict):
        return None
    completed, passed = num(progress.get("n_completed")), num(progress.get("n_passed"))
    if passed is None:
        passed = num(verdict.get("n_passed"))
    tally = _tally(progress) if full else {}
    if full and not progress and verdict:
        tally = _tally(verdict)
        completed = completed if completed is not None else sum(tally.values()) or None
    ended = (
        owner.get("state") == "complete"
        or progress.get("state") in ("complete", "superseded_snapshot")
        or bool(verdict)
    )
    if not ended:
        state = "grading" if ctx.live else "unfinished"
    elif not completed:
        state = "stopped"  # Ended before grading any capsule.
    elif progress.get("state") == "superseded_snapshot":
        state = "superseded"
    else:
        state = "done"
    source = progress.get("graded_source_sha256") or owner.get("source")
    started = num(owner.get("started_at_epoch")) or int(token) / 1e9
    updated = epoch(progress.get("updated_at")) or epoch(verdict.get("graded_at"))
    return {
        "token": token,
        "state": state,
        "owner_state": clean(owner.get("state"), 24) or None,
        "reason": clean(owner.get("admission_reason") or owner.get("deferred_reason") or owner.get("error"), 80)
        or None,
        "started_at": started,
        "updated_at": updated,
        "n_selected": num(progress.get("n_selected")) or num(verdict.get("n_capsules")),
        "n_completed": completed,
        "n_passed": passed,
        "n_failed": tally.get("fail"),
        "n_errors": tally.get("error"),
        "n_l2_awaiting_l3": tally.get("l2"),
        "source": source[:12] if isinstance(source, str) else None,
        "progress_source": str(progress_path) if ctx.operator_private else None,
    }


def _grades(ctx, run: Path) -> list[dict[str, Any]]:
    tokens = _tokens(run)
    out = []
    for i, token in enumerate(tokens[:MAX_SERIES_GRADES]):
        grade = _grade(ctx, run, token, full=i < MAX_GRADES)
        if grade is not None:
            out.append(grade)
    return out


def _certificates(ctx, workspace: Path | None) -> dict[str, Any] | None:
    """Capsules holding an L2/L3 certificate in the tier ledger, and the joint L3 physical obligation."""
    if workspace is None:
        return None
    ledger = obj(ctx.reader.document(workspace / "qa" / "tier_state.json", limit=16 << 20, stale_ok=True))
    tiers: dict[str, dict[str, set]] = {t: {"pass": set(), "pending": set(), "fail": set()} for t in ("L2", "L3")}
    for name, state in ledger.items():
        if name.startswith("<"):
            continue
        for tier, buckets in tiers.items():
            records = [obj(state).get(tier), *obj(obj(obj(state).get("<certs>")).get(tier)).values()]
            statuses = {obj(r).get("status") for r in records if obj(r)}
            if statuses & set(PASS):
                buckets["pass"].add(name)
            elif statuses & set(FAIL):
                buckets["fail"].add(name)
            elif statuses:
                buckets["pending"].add(name)
    physical = obj(
        ctx.reader.document(workspace / "qa" / "joint_physical_correctness.json", limit=16 << 20, stale_ok=True)
    )
    if physical.get("schema") != PHYSICAL_SCHEMA:
        physical = {}
    if not ledger and not physical:
        return None
    return {
        "L2": {k: len(v) for k, v in tiers["L2"].items()} if ledger else None,
        "L3": {k: len(v) for k, v in tiers["L3"].items()} if ledger else None,
        "physical": {k: num(physical.get(k)) for k in ("required", "passed", "failed", "pending")}
        if physical
        else None,
    }


def _series(ctx, run: Path, grades: list[dict[str, Any]]) -> tuple[dict[str, list], str | None]:
    """``(series, scope)``: capsules passing per grade, at the time each grade last reported.

    Background grades and round-end grades select different capsule sets (all of them against the
    capsules a round touched), so a series never mixes the two: the round verdicts and OOT commits
    are used only when the run has no background grade. Grades that graded nothing are left out."""
    points: dict[float, tuple[int, int | None]] = {}
    for grade in grades:
        t, passed = grade["updated_at"], grade["n_passed"]
        if t is not None and passed is not None and grade["n_completed"]:
            points[t] = (int(passed), grade["n_completed"])
    scope = "background grades"
    if not points:
        scope = "round grades"
        for entry in scan_dir(run / "qa_history", 4096):
            if not (entry.name.startswith("verdict_round_") and entry.name.endswith(".json")):
                continue
            doc = ctx.reader.document(Path(entry.path), limit=16 << 20) or {}
            t, passed, graded = epoch(doc.get("graded_at")), num(doc.get("n_passed")), sum(_tally(doc).values())
            if t is not None and passed is not None and graded:
                points[t] = (int(passed), graded)
        commits = ctx.state.setdefault("commits", [])
        ctx.reader.poll(run / "oot_commits.jsonl", commits.append, commits.clear)
        for row in commits:
            t, passed = epoch(obj(row.get("oot")).get("committed_at")), num(row.get("n_passed"))
            if t is not None and passed is not None:
                points.setdefault(t, (int(passed), num(row.get("n_capsules"))))
    ordered = downsample(sorted(points.items()))
    series = {
        "t": [t for t, _ in ordered],
        "n_passed": [n for _, (n, _) in ordered],
        "n_completed": [c for _, (_, c) in ordered],
    }
    return series, scope if ordered else None


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
    grades = _grades(ctx, run)
    series, series_scope = _series(ctx, run, grades)
    progress = ctx.reader.document(workspace / "qa" / "progress.json", stale_ok=True) if workspace else None
    if progress is not None and progress.get("schema") not in (None, PROGRESS_SCHEMA):
        notes.append(f"progress.json schema {clean(progress.get('schema'), 60)!r} not read")
        progress = None
    progress = progress or {}
    if not progress and not grades:
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
            detail = f"{alert.get('count')}x: {detail or ''}".strip(": ")
        severity = {"warning": "warn", "error": "error"}.get(alert.get("severity"), "info")
        alerts.append(
            {
                "code": alert.get("code"),
                "severity": severity,
                "title": clean(alert_title({"alerts": [alert]}) or alert.get("code"), 120),
                "detail": clean(detail, 240),
            }
        )
    verdict = (
        obj(ctx.reader.document(workspace / "qa" / "verdict.json", limit=16 << 20, stale_ok=True)) if workspace else {}
    )
    source = progress.get("graded_source_sha256")
    published = (
        {
            "updated_at": epoch(progress.get("updated_at")),
            "n_selected": num(progress.get("n_selected")),
            "n_completed": num(progress.get("n_completed")),
            "n_passed": num(progress.get("n_passed")),
        }
        if progress
        else None
    )
    return {
        "grades": grades[:MAX_GRADES],
        "grades_total": len(grades),
        "published": published,
        "certificates": _certificates(ctx, workspace),
        "verdict": {
            "stage": clean(verdict.get("stage"), 48) or None,
            "tiers_graded": [clean(t, 8) for t in verdict.get("tiers_graded") or [] if isinstance(t, str)],
            "tiers_not_run": [clean(t, 8) for t in verdict.get("tiers_not_run") or [] if isinstance(t, str)],
            "all_pass": verdict.get("all_pass"),
        }
        if verdict
        else None,
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
        "series": series,
        "series_scope": series_scope,
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
