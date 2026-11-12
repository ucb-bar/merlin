"""What the author agent is working on now: the files it edits, the operation families they touch, what is open.

Edits are the agents' own file changes (:class:`.insight_common.AgentActivity`) over the last
``WINDOW_S`` seconds, shown relative to the workspace and grouped by file (operator-private: file names
are the agent's choice) or by directory; the
family of an edited file is the operation family its path names (:func:`.performance.family`), which is
a hint, not a measurement. The lead's latest message and command come from :mod:`.agents`; open
self-checks and pending suggestions from :mod:`.agent_view`. Display telemetry.

Panel fields beyond the shared shape: ``files`` [{path, edits, last_at}], ``directories`` {dir: n},
``families`` {family: n}, ``lead`` {message, command}, ``open`` {selfchecks, oldest_pending_at}.
"""

from __future__ import annotations

from typing import Any

from ..io import clean, num, obj, rows
from .insight_common import activity, ago, line, panel, row, table
from .performance import family

WINDOW_S = 1800.0
DIRECTORY_DEPTH = 3


def _relative(path: str, workspace: str | None) -> str:
    if workspace and path.startswith(workspace.rstrip("/") + "/"):
        return path[len(workspace.rstrip("/")) + 1 :]
    return path.rpartition("/")[2] if path.startswith("/") else path


def read(ctx) -> dict[str, Any]:
    now = ctx.now
    tracker = activity(ctx)
    workspace = ctx.experiment.get("workspace")
    recent = [(t, _relative(p, workspace)) for t, p in tracker.edits if t >= now - WINDOW_S]
    files: dict[str, dict[str, Any]] = {}
    directories: dict[str, int] = {}
    families: dict[str, int] = {}
    private = ctx.operator_private
    for t, path in recent:
        # Public mode shows where in the tree the agent works, not the file names it chose.
        directory = path.rpartition("/")[0] or "."
        shown = path if private else "/".join(directory.split("/")[:DIRECTORY_DEPTH]) + "/…"
        f = files.setdefault(shown, {"path": shown, "edits": 0, "last_at": t})
        f["edits"] += 1
        f["last_at"] = max(f["last_at"], t)
        directory = "/".join(directory.split("/")[:DIRECTORY_DEPTH])
        directories[directory] = directories.get(directory, 0) + 1
        fam = family(path.rpartition("/")[2])
        if fam != "other":
            families[fam] = families.get(fam, 0) + 1
    ranked = sorted(files.values(), key=lambda f: (-f["edits"], -f["last_at"]))
    agents = rows(obj(ctx.shared.get("agents")).get("agents"))
    lead = next((a for a in agents if a.get("role") == "lead"), {})
    working = [a for a in agents if a.get("state") == "working" and a.get("role") != "lead"]
    view = obj(ctx.shared.get("agent_view"))
    checks = obj(view.get("selfchecks"))
    suggestions = obj(view.get("suggestions"))
    last_edit = max((t for t, _ in tracker.edits), default=None)

    summary = [
        line(
            "lead says",
            clean(lead.get("last_message"), 150) or None if ctx.operator_private else "operator-private",
            key="lead",
        ),
        line(
            "lead running",
            clean(lead.get("current"), 120) or None
            if ctx.operator_private
            else f"{lead.get('state') or 'not recorded'} · {num(lead.get('commands'))} commands",
        ),
        line(
            f"edits in last {WINDOW_S / 60:.0f}m",
            f"{len(recent)} in {len(files)} files · last {ago(now, last_edit)}"
            if recent
            else (f"none · last {ago(now, last_edit)}" if last_edit else "none recorded"),
        ),
        line(
            "directories",
            ", ".join(f"{d} ×{n}" for d, n in sorted(directories.items(), key=lambda kv: -kv[1])[:3]) or None,
        ),
        line(
            "families in edited files",
            ", ".join(f"{k} ×{v}" for k, v in sorted(families.items(), key=lambda kv: -kv[1])) or None,
        ),
        line("helpers working", ", ".join(clean(a.get("name") or a.get("path"), 24) for a in working[:4]) or "none"),
        line(
            "open self-checks",
            f"{num(checks.get('pending'))} pending · oldest {ago(now, num(checks.get('oldest_pending_at')))}"
            if checks.get("pending")
            else ("none" if checks else None),
            "warn"
            if (num(checks.get("pending")) or 0) and now - (num(checks.get("oldest_pending_at")) or now) > 900
            else None,
        ),
        line(
            "suggestions waiting",
            f"{num(suggestions.get('suggestions'))} suggestions · "
            f"{num(suggestions.get('pending_workload_check'))} pending workload checks"
            if suggestions.get("present")
            else None,
        ),
    ]
    table_rows = [
        row([f["path"], f["edits"], ago(now, f["last_at"])], key=f["path"] if private else None) for f in ranked[:10]
    ]
    return panel(
        "AGENT FOCUS",
        summary,
        table_=table(["file" if private else "directory", "edits", "last"], table_rows),
        notes=["families are named by file paths: a hint about the work, not a measurement"],
        files=ranked[:24],
        directories=directories,
        families=families,
        lead={
            "message": clean(lead.get("last_message"), 400) or None,
            "command": clean(lead.get("current"), 400) or None,
        },
        open={"selfchecks": num(checks.get("pending")), "oldest_pending_at": num(checks.get("oldest_pending_at"))},
    )


def inspect(ctx, key: str) -> dict[str, Any]:
    section = obj(ctx.shared.get("focus"))
    if key == "lead":
        lead = obj(section.get("lead"))
        return {
            "title": "lead agent",
            "sections": [
                {"title": "latest message", "lines": [lead.get("message") or "—"]},
                {"title": "running", "lines": [lead.get("command") or "—"]},
            ],
            "paths": [],
        }
    workspace = ctx.experiment.get("workspace") or ""
    return {
        "title": f"edits · {key}",
        "sections": [{"title": "file", "lines": [key]}],
        "paths": [f"{workspace.rstrip('/')}/{key}"] if workspace and not key.startswith("/") else [],
    }
