"""Grader pipeline health: how fast verdicts arrive, how many are delivered, what holds grades back.

Read from the run's background grade records: each finished fast verdict
(``qa_history/verdict_fast_<tick>.json``: when it was graded, whether its snapshot was still current,
its ``background_grade`` status) and each snapshot owner
(``_qa_work/background_snapshot_owners/<tick>.json``: state, admission closure and deferral reasons),
plus the live grades :mod:`.progress` already follows, the channel's deferred simulations
(``simdefer_*``) and operator idle-timing requests (``req_op_idle_*`` / ``resp_op_idle_*``), and the
campaign log's ``[idle-perf]`` lines. A grade's duration is its verdict time minus its tick (the owner
start). A verdict counts as delivered when its snapshot was current or when the agent's
``qa/latest_completed_grade.json`` records merging it (``completed_rows.recorded_grades``). Display
telemetry; it grades nothing.

Panel fields beyond the shared shape: ``verdicts`` {n, delivered, superseded, first_after_s, p50_s,
p95_s}, ``owners`` {n, states, admission, deferred, max_concurrent}, ``idle`` {requests, answered,
last_at, log_lines}, ``backlog`` {grading, remaining}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import clean, epoch, is_dir, mtime, num, obj, rows, scan_dir
from .insight_common import ago, campaign_log, duration, line, log_scan, panel, quantile, row, series, table

MAX_RECORDS = 512


def _tick(name: str, prefix: str) -> int | None:
    stem = name[len(prefix) :].partition(".")[0].partition("_")[0]
    return int(stem) if stem.isdigit() else None


def _verdicts(ctx, run_dir: Path) -> list[dict[str, Any]]:
    """One summary per finished fast verdict (immutable once written, so summarised once)."""
    cache: dict = ctx.state.setdefault("verdicts", {})
    history = run_dir / "qa_history"
    names = sorted(
        e.name for e in scan_dir(history, 8192) if e.name.startswith("verdict_fast_") and e.name.endswith(".json")
    )[-MAX_RECORDS:]
    for name in reversed(names):  # newest first: a short budget summarises the verdicts that matter now
        if name in cache:
            continue
        document = ctx.reader.document(history / name, limit=32 << 20)
        if document is None:
            continue
        tick = _tick(name, "verdict_fast_")
        statuses: dict[str, int] = {}
        for r in rows(document.get("per_capsule"), 8192):
            status = str(r.get("status") or "not recorded")
            statuses[status] = statuses.get(status, 0) + 1
        background = obj(document.get("background_grade"))
        cache[name] = {
            "token": str(tick) if tick is not None else name,
            "tick": tick / 1e9 if tick is not None else None,
            "graded_at": epoch(document.get("graded_at")),
            "status": background.get("status"),
            "reason": background.get("reason"),
            "snapshot_current": document.get("snapshot_current"),
            "n_capsules": num(document.get("n_capsules")),
            "n_passed": num(document.get("n_passed")),
            "statuses": statuses,
            "stage": document.get("stage"),
        }
    ctx.state["verdicts_listed"] = len(names)
    return [cache[n] for n in names if n in cache]


def _owners(ctx, run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir / "_qa_work" / "background_snapshot_owners"
    found = []
    for entry in sorted(scan_dir(root, 8192), key=lambda e: e.name)[-MAX_RECORDS:]:
        if not entry.name.endswith(".json"):
            continue
        document = ctx.reader.document(Path(entry.path), limit=1 << 20)
        if document is None:
            continue
        found.append(
            {
                "token": str(document.get("token") or entry.name[:-5]),
                "started_at": num(document.get("started_at_epoch")),
                "updated_at": mtime(entry.path),
                "state": document.get("state"),
                "admission_closed": document.get("admission_closed"),
                "admission_reason": document.get("admission_reason"),
                "deferred_reason": document.get("deferred_reason"),
                "error": document.get("error"),
            }
        )
    return found


def _channel(ctx, workspace: str | None) -> dict[str, Any]:
    out = {"idle_requests": 0, "idle_answered": 0, "idle_last_at": None, "deferred": {}}
    channel = Path(workspace or "") / ".qa_channel"
    if not workspace or not is_dir(channel):
        return out
    defers = []
    for entry in scan_dir(channel, 65536):
        name = entry.name
        if name.startswith("req_op_idle_"):
            out["idle_requests"] += 1
            tick = _tick(name, "req_op_idle_")
            if tick is not None:
                out["idle_last_at"] = max(out["idle_last_at"] or 0, tick / 1e9)
        elif name.startswith("resp_op_idle_"):
            out["idle_answered"] += 1
        elif name.startswith("simdefer_") and name.endswith(".json"):
            defers.append(entry.path)
    for path in sorted(defers)[-64:]:
        reason = obj(ctx.reader.document(Path(path), limit=1 << 20)).get("reason")
        key = clean(reason, 60) if reason else "not recorded"
        out["deferred"][key] = out["deferred"].get(key, 0) + 1
    return out


def _recorded(ctx, workspace: str | None) -> set[str]:
    """Grade tokens the agent's newest completed-grade file has merged (delivered despite superseding)."""
    path = Path(workspace or "") / "qa" / "latest_completed_grade.json"
    if not workspace or not path.is_file():
        return set()
    summary = obj(obj(ctx.reader.document(path, limit=64 << 20)).get("completed_rows"))
    return {str(token) for token in summary.get("recorded_grades") or () if str(token).isdigit()}


def _max_concurrent(owners: list[dict], verdicts: dict[str, dict], now: float) -> tuple[int | None, float]:
    """Most snapshot owners alive at once, and seconds spent with three or more alive."""
    spans = []
    for owner in owners:
        start = owner["started_at"]
        if start is None:
            continue
        verdict = verdicts.get(owner["token"])
        end = (verdict or {}).get("graded_at") or (
            now if owner["state"] not in ("complete", "failed") else owner["updated_at"]
        )
        if end and end > start:
            spans.append((start, end))
    if not spans:
        return None, 0.0
    marks = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    alive = best = 0
    third = 0.0
    previous = marks[0][0]
    for at, delta in marks:
        if alive >= 3:
            third += at - previous
        alive += delta
        best = max(best, alive)
        previous = at
    return best, third


def read(ctx) -> dict[str, Any]:
    now = ctx.now
    run_dir = Path(ctx.experiment.get("run_dir") or "")
    verdicts = _verdicts(ctx, run_dir)
    owners = _owners(ctx, run_dir)
    by_token = {v["token"]: v for v in verdicts}
    channel = _channel(ctx, ctx.experiment.get("workspace"))
    log = log_scan(ctx, campaign_log(ctx))

    durations = [v["graded_at"] - v["tick"] for v in verdicts if v["graded_at"] and v["tick"]]
    recorded = _recorded(ctx, ctx.experiment.get("workspace"))
    superseded = sum(1 for v in verdicts if v["status"] == "superseded_snapshot" and v["token"] not in recorded)
    delivered = sum(
        1
        for v in verdicts
        if v["snapshot_current"] is True
        or v["token"] in recorded
        or (v["status"] and v["status"] != "superseded_snapshot")
    )
    turns = rows(obj(ctx.shared.get("lifecycle")).get("turns"))
    agent_start = None
    if turns:
        agent_start = num(turns[0].get("agent_start")) or num(turns[0].get("startup_end"))
    graded = sorted(v["graded_at"] for v in verdicts if v["graded_at"])
    first_after = graded[0] - agent_start if graded and agent_start else None

    states: dict[str, int] = {}
    admission: dict[str, int] = {}
    deferred: dict[str, int] = dict(channel["deferred"])
    errors = 0
    for owner in owners:
        states[str(owner["state"] or "not recorded")] = states.get(str(owner["state"] or "not recorded"), 0) + 1
        if owner["admission_reason"]:
            key = clean(owner["admission_reason"], 60)
            admission[key] = admission.get(key, 0) + 1
        if owner["deferred_reason"]:
            key = "owner: " + clean(owner["deferred_reason"], 52)
            deferred[key] = deferred.get(key, 0) + 1
        if owner["error"]:
            errors += 1
    max_alive, third_s = _max_concurrent(owners, by_token, now)

    live = [g for g in rows(obj(ctx.shared.get("progress")).get("grades")) if g.get("state") == "grading"]
    remaining = sum(max(0, (num(g.get("n_selected")) or 0) - (num(g.get("n_completed")) or 0)) for g in live)
    newest = verdicts[-1] if verdicts else None
    gated = (newest or {}).get("statuses", {}).get("gated") if newest else None
    newest_owner = next((o for o in owners if newest and o["token"] == newest["token"]), None)
    gated_reason = (newest_owner or {}).get("admission_reason") if gated else None
    idle_log = log.counts.get("idle_perf", 0) if log else None

    n = len(verdicts)
    frac = superseded / n if n else None
    summary = [
        line(
            "verdicts done / delivered",
            (f"{n} / {delivered}" + (" · reading more" if (ctx.state.get("verdicts_listed") or 0) > n else ""))
            if n
            else "no verdict yet",
            "bad" if n >= 3 and delivered == 0 else ("warn" if frac and frac > 0.5 else None),
            key="verdicts",
        ),
        line(
            "superseded before delivery",
            f"{100 * frac:.0f}% ({superseded}/{n})" if n else None,
            "warn" if frac and frac > 0.5 else None,
        ),
        line("first verdict after start", duration(first_after)),
        line(
            "grade time p50 / p95",
            f"{duration(quantile(durations, 0.5))} / {duration(quantile(durations, 0.95))}" if durations else None,
        ),
        line("grading now (cases left)", f"{len(live)} ({remaining})" if live else "idle"),
        line(
            "admission closures",
            ", ".join(f"{k} ×{v}" for k, v in admission.items()) or "none",
            "info" if admission else None,
            key="owners",
        ),
        line(
            "deferred", ", ".join(f"{k} ×{v}" for k, v in sorted(deferred.items(), key=lambda kv: -kv[1])[:2]) or "none"
        ),
        line(
            "newest verdict gated",
            f"{gated} ({gated_reason or 'reason not recorded'})" if gated else ("none" if newest else None),
            "warn" if gated else None,
        ),
        line(
            "idle timing requests",
            f"{channel['idle_requests']} sent · {channel['idle_answered']} answered · last "
            f"{ago(now, channel['idle_last_at'])}"
            if channel["idle_requests"]
            else (
                "none" + (f" ({idle_log} log lines)" if idle_log else "") if ctx.experiment.get("workspace") else None
            ),
        ),
        line("max concurrent owners", f"{max_alive} · ≥3 alive {duration(third_s)}" if max_alive else None),
    ]
    table_rows = []
    for v in reversed(verdicts[-12:]):
        dur = v["graded_at"] - v["tick"] if v["graded_at"] and v["tick"] else None
        if v["snapshot_current"] is True:
            status = "delivered"
        elif v["token"] in recorded:
            status = "merged"
        else:
            status = v["status"] or "not recorded"
        table_rows.append(
            row(
                [
                    ago(now, v["graded_at"]),
                    duration(dur),
                    v["n_passed"],
                    status.replace("_snapshot", ""),
                    v["statuses"].get("gated"),
                ],
                "warn" if v["status"] == "superseded_snapshot" and v["token"] not in recorded else None,
                key=v["token"],
            )
        )
    points = [
        (v["graded_at"], round((v["graded_at"] - v["tick"]) / 60, 2)) for v in verdicts if v["graded_at"] and v["tick"]
    ]
    notes = []
    listed = ctx.state.get("verdicts_listed") or 0
    if listed > n:
        notes.append(f"{n} of {listed} verdicts read so far; the rest arrive over the next refreshes")
    if errors:
        notes.append(f"{errors} snapshot owner(s) recorded an error")
    if not verdicts and not owners:
        notes.append("no background grade records in this run")
    return panel(
        "GRADER",
        summary,
        table_=table(["graded", "took", "pass", "outcome", "gated"], table_rows),
        series_=[series("grade minutes", points)],
        notes=notes,
        verdicts={
            "n": n,
            "delivered": delivered,
            "superseded": superseded,
            "first_after_s": first_after,
            "p50_s": quantile(durations, 0.5),
            "p95_s": quantile(durations, 0.95),
        },
        owners={
            "n": len(owners),
            "states": states,
            "admission": admission,
            "deferred": deferred,
            "max_concurrent": max_alive,
            "three_alive_s": third_s,
            "errors": errors,
        },
        idle={
            "requests": channel["idle_requests"],
            "answered": channel["idle_answered"],
            "last_at": channel["idle_last_at"],
            "log_lines": idle_log,
        },
        backlog={"grading": len(live), "remaining": remaining},
    )


def inspect(ctx, key: str) -> dict[str, Any]:
    section = obj(ctx.shared.get("grader"))
    verdicts = {v["token"]: v for v in ctx.state.get("verdicts", {}).values()}
    if key in verdicts:
        v = verdicts[key]
        lines = [
            f"{k}: {v[k]}"
            for k in ("graded_at", "status", "reason", "snapshot_current", "stage", "n_capsules", "n_passed")
        ]
        lines += [f"  {status}: {count}" for status, count in sorted(v["statuses"].items())]
        return {
            "title": f"verdict {key}",
            "sections": [{"title": "fast verdict", "lines": lines}],
            "paths": [str(Path(ctx.experiment.get("run_dir") or "") / "qa_history" / f"verdict_fast_{key}.json")],
        }
    owners = obj(section.get("owners"))
    return {
        "title": "snapshot owners",
        "sections": [
            {"title": "states", "lines": [f"{k}: {v}" for k, v in obj(owners.get("states")).items()]},
            {"title": "admission closures", "lines": [f"{k}: {v}" for k, v in obj(owners.get("admission")).items()]},
            {"title": "deferred", "lines": [f"{k}: {v}" for k, v in obj(owners.get("deferred")).items()]},
        ],
        "paths": [],
    }
