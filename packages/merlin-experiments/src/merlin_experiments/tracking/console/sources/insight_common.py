"""Shared pieces of the insight panels: the panel shape, the campaign chain, the agent activity and log tails.

Not a console source itself (it is not in :data:`..model.SOURCES`). The insight sources
(:mod:`.performance`, :mod:`.cases`, :mod:`.tools`, :mod:`.grader`, :mod:`.integrity`, :mod:`.attempts`,
:mod:`.certification`, :mod:`.focus`, :mod:`.cost`) all return one generic panel shape so a single
frontend renderer draws them::

    {"title", "summary": [{"label", "value", "level", "key"}],
     "table": {"columns": [..], "rows": [{"cells": [..], "level", "key"}]}|None,
     "series": [{"label", "t": [..], "v": [..]}], "notes": [..], ...source-specific fields}

``level`` is one of ``ok``, ``info``, ``warn``, ``bad`` or ``None``. A value that was not recorded is
``"—"`` in text and ``None`` in a number field, never 0.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..io import bounded_glob, clean, epoch, is_dir, is_file, obj
from . import downsample

LEVELS = ("ok", "info", "warn", "bad")
#: Agent activity events kept per experiment, and rollout files followed.
MAX_EVENTS = 60_000
MAX_ROLLOUTS = 96
ACTIVITY_STEP = 16 << 20
CHAIN_RECHECK_S = 60.0


def line(label: str, value: Any, level: str | None = None, key: str | None = None) -> dict[str, Any]:
    text = "—" if value is None or value == "" else clean(value, 160)
    return {"label": clean(label, 40), "value": text, "level": level if level in LEVELS else None, "key": key}


def table(columns: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"columns": [clean(c, 24) for c in columns], "rows": rows}


def row(cells: list[Any], level: str | None = None, key: str | None = None) -> dict[str, Any]:
    return {
        "cells": ["—" if c is None or c == "" else clean(c, 80) for c in cells],
        "level": level if level in LEVELS else None,
        "key": key,
    }


def series(label: str, points: list[tuple[float, float]]) -> dict[str, Any]:
    points = downsample(sorted(points))
    return {"label": clean(label, 40), "t": [p[0] for p in points], "v": [p[1] for p in points]}


def panel(
    title: str,
    summary: list[dict],
    *,
    table_: dict | None = None,
    series_: list | None = None,
    notes: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {"title": title, "summary": summary, "table": table_, "series": series_ or [], "notes": notes or [], **extra}


def ratio_text(value: float | None, digits: int = 2) -> str | None:
    return None if value is None else f"×{value:.{digits}f}"


def ago(now: float, then: float | None) -> str | None:
    if then is None:
        return None
    seconds = max(0.0, now - then)
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]


def out_root(experiment: dict) -> Path | None:
    """``<repo>/out`` of a run laid out as ``out/runs/<target>/<suite>/<run>``."""
    run = Path(os.path.abspath(experiment.get("run_dir") or "."))
    parents = run.parents
    if len(parents) >= 4 and parents[2].name == "runs":
        return parents[3]
    return None


# -- per-row records kept out of public sections ----------------------------------------------------
_ROWS: dict[str, dict[str, list[dict]]] = {}


def share_rows(ctx, source: str, rows_: list[dict]) -> None:
    """Keep a source's per-row records for the insight panels' aggregates.

    :mod:`.capsules` and :mod:`.perf` publish their rows only with ``operator_private``; the insight
    panels still need them to count by cohort and family. The rows stay inside this process and
    are never put in a section; a panel shows names from them only with ``operator_private``."""
    key = str(ctx.experiment.get("id") or ctx.experiment.get("run_dir"))
    _ROWS.setdefault(key, {})[source] = rows_


def private_rows(ctx, source: str) -> list[dict]:
    key = str(ctx.experiment.get("id") or ctx.experiment.get("run_dir"))
    return _ROWS.get(key, {}).get(source) or []


# -- the campaign chain (every attempt of a run) --------------------------------------------------
def chain(ctx) -> dict[str, Any]:
    """The attempt chain naming this run: ``{manifest, root, attempts: [{...}], current}``.

    Read from a campaign manifest (``merlin.agentreport.campaign_manifest.v1``) under the run's
    ``out/artifacts/agentic-report/campaign-figures/`` whose live attempt is this run. Without one the
    chain is the current run alone. Cached; rechecked at most once a minute."""
    state = ctx.state.setdefault("_chain", {})
    if state.get("at") and time.monotonic() - state["at"] < CHAIN_RECHECK_S:
        return state["value"]
    run_dir = Path(ctx.experiment.get("run_dir") or "")
    current = {
        "label": "current",
        "run_dir": str(run_dir),
        "workspace": ctx.experiment.get("workspace"),
        "live": ctx.live,
        "launch_log": None,
        "agent_home": None,
        "accounting_log": None,
        "accounting_entry": None,
    }
    value: dict[str, Any] = {"manifest": None, "root": None, "attempts": [current], "current": 0}
    here = os.path.realpath(run_dir)
    complete = True
    root = out_root(ctx.experiment)
    if root is not None:
        for manifest in sorted(
            bounded_glob(root / "artifacts" / "agentic-report" / "campaign-figures", "*/*.campaign.yaml", 16)
        ):
            document = ctx.reader.yaml(manifest, limit=2 << 20)
            if document is None:
                complete = False  # over the refresh budget or unreadable: look again next refresh
                continue
            document = obj(document)
            base = Path(document.get("root") or "")
            attempts = []
            for entry in document.get("attempts") or []:
                entry = obj(entry)
                if not entry.get("run_dir"):
                    continue

                def resolve(key: str, entry=entry) -> str | None:
                    raw = entry.get(key)
                    if not isinstance(raw, str) or not raw:
                        return None
                    return str(Path(raw) if raw.startswith("/") else base / raw)

                label = clean(entry.get("continuation"), 12)
                if entry.get("attempt") is not None:
                    label += f".{entry.get('attempt')}"
                attempts.append(
                    {
                        "label": label,
                        "run_dir": resolve("run_dir"),
                        "workspace": resolve("workspace"),
                        "live": bool(entry.get("live")),
                        "launch_log": resolve("launch_log"),
                        "agent_home": resolve("agent_home"),
                        "accounting_log": resolve("accounting_log"),
                        "accounting_entry": entry.get("accounting_entry"),
                    }
                )
            live = [i for i, a in enumerate(attempts) if a["run_dir"] and os.path.realpath(a["run_dir"]) == here]
            if live:
                value = {"manifest": str(manifest), "root": str(base), "attempts": attempts, "current": live[-1]}
                break
    if complete or value["manifest"]:
        state.update(at=time.monotonic(), value=value)
    return value


def previous_workspace(ctx) -> str | None:
    """The newest earlier attempt of the chain that has an agent workspace on disk."""
    info = chain(ctx)
    for attempt in reversed(info["attempts"][: info["current"]]):
        workspace = attempt.get("workspace")
        if workspace and is_dir(Path(workspace) / ".qa_channel"):
            return workspace
    return None


# -- the agent activity tail (shared by tools and focus) -------------------------------------------
_ACTIVITY: dict[str, AgentActivity] = {}

#: ``(category, needles)``: a command or edit counts when any needle occurs in its text. Needles are
#: tool and file names of the harness, never target names.
COMMAND_RULES = (
    ("isa_debug", ("isa_tools.py debug",)),
    ("evaluate_variants", ("evaluate_variants",)),
    ("suggestions", ("OPTIMIZATION_SUGGESTIONS", "optimization_suggest")),
    ("ranked_list", ("development_headroom", "performance_brief", "latest_completed_grade")),
    ("ab_variants", ("--submission variants",)),
    ("gsim_check", ("--sim gsim",)),
    ("scalability_check", ("--suite scalability",)),
    ("selfcheck", ("agent_selfcheck",)),
    ("perf_fields", ("estimated_whole_soc_cycles", "floor_to_estimate", "structural_roofline")),
    ("sched_library", ("merlin.sched", "merlin.compile.scheduling")),
)


class Needles:
    """Which library passes a command or an edit mentions, by structured tokens (no regex).

    ``words`` maps an identifier (a module leaf with an underscore, a symbol, a pass name) to the pass
    keys it names; ``pairs`` maps an adjacent ``parent.leaf`` / ``parent/leaf`` pair (a module leaf
    without an underscore, such as ``capacity``) to its pass keys. A text is cut at every character
    other than letters, digits, ``_``, ``.`` and ``/``; each piece is split at ``.`` and ``/``."""

    _CUT = {c: " " for c in range(128) if not (chr(c).isalnum() or chr(c) in "_./")}

    def __init__(self, words: dict[str, set[str]], pairs: dict[tuple[str, str], set[str]]) -> None:
        self.words = words
        self.pairs = pairs

    def touched(self, text: str) -> set[str]:
        found: set[str] = set()
        for piece in text.translate(self._CUT).split():
            parts = piece.replace("/", ".").split(".")
            for index, part in enumerate(parts):
                if not part:
                    continue
                found.update(self.words.get(part, ()))
                if index and parts[index - 1]:
                    found.update(self.pairs.get((parts[index - 1], part), ()))
        return found


class AgentActivity:
    """Commands, edits and helper spawns of every agent stream, tailed incrementally.

    With :class:`Needles` it also counts, per library pass, the commands and edits that mention it
    (``library``: ``{key: [count, last time]}``)."""

    def __init__(self, needles: Needles | None = None) -> None:
        self.events: deque[tuple[float, str, str]] = deque(maxlen=MAX_EVENTS)  # (t, category, detail)
        self.edits: deque[tuple[float, str]] = deque(maxlen=8192)  # (t, path)
        self.commands = 0
        self.files: list[str] = []
        self.tails: dict[str, LineTail] = {}
        self.catching_up = False
        self.needles = needles
        self.library: dict[str, list] = {}
        self.streams: dict[str, str] = {}  # rollout file name -> the copy this tracker follows

    def _touch(self, when: float, text: str) -> None:
        if self.needles is None or not text:
            return
        for key in self.needles.touched(text):
            entry = self.library.setdefault(key, [0, when])
            entry[0] += 1
            entry[1] = max(entry[1], when)

    def consume(self, row: dict) -> None:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            return
        when = epoch(row.get("timestamp"))
        if when is None:
            return
        if row.get("type") == "response_item" and payload.get("type") == "function_call":
            name = payload.get("name")
            if name == "spawn_agent":
                self.events.append((when, "spawn", ""))
            return
        if row.get("type") != "event_msg" or payload.get("type") != "item_completed":
            return
        item = obj(payload.get("item"))
        kind = item.get("type")
        if kind == "CommandExecution":
            command = item.get("command")
            text = command[-1] if isinstance(command, list) and command else command
            if not isinstance(text, str):
                return
            self.commands += 1
            self.events.append((when, "command", ""))
            self._touch(when, text)
            for category, needles in COMMAND_RULES:
                if any(n in text for n in needles):
                    self.events.append((when, category, ""))
        elif kind == "FileChange":
            for path, change in obj(item.get("changes")).items():
                if not isinstance(path, str):
                    continue
                self.edits.append((when, path))
                self.events.append((when, "edit", ""))
                body = obj(change)
                text = body.get("unified_diff") or body.get("content") or ""
                self._touch(when, path + "\n" + (text if isinstance(text, str) else ""))
                if isinstance(text, str) and ("merlin.sched" in text or "merlin.compile.scheduling" in text):
                    self.events.append((when, "sched_library", "edit"))
                if path.lower().endswith("iteration_notes.md"):  # ITERATION_NOTES.md, or the legacy name
                    self.events.append((when, "iteration_notes", ""))

    def counts(self, now: float, window_s: float | None) -> dict[str, int]:
        out: dict[str, int] = {}
        for when, category, _ in self.events:
            if window_s is None or when >= now - window_s:
                out[category] = out.get(category, 0) + 1
        return out


def unique_streams(paths: list[str]) -> list[str]:
    """One path per agent stream: a run keeps a snapshot copy of each live rollout (same file name, in
    ``rounds/*rollout_snapshot``), and reading both counted every command twice. The larger copy (the
    more complete one) is kept, in the order first seen."""
    best: dict[str, tuple[int, str]] = {}
    order: list[str] = []
    for path in paths:
        name = os.path.basename(path)
        try:
            size = os.stat(path).st_size
        except OSError:
            size = -1
        if name not in best:
            order.append(name)
        if name not in best or size > best[name][0]:
            best[name] = (size, path)
    return [best[name][1] for name in order]


def activity(ctx) -> AgentActivity:
    """The experiment's shared activity tail, advanced over every known agent stream.

    Each stream is tailed by its own :class:`LineTail` (the :mod:`.agents` reader keeps its own
    offsets), at most ``ACTIVITY_STEP`` bytes per refresh, and only lines that can be a completed item
    or a helper spawn are parsed."""
    from .agents import index_for
    from .library import touch_needles

    key = str(ctx.experiment.get("id") or ctx.experiment.get("run_dir"))
    needles = touch_needles(ctx)
    tracker = _ACTIVITY.get(key)
    # A tracker started before the library manifest was readable starts over once it is, so every
    # stream is counted against the same needles.
    if tracker is None or (needles is not None and tracker.needles is None):
        tracker = _ACTIVITY[key] = AgentActivity(needles)
    index = index_for(ctx)
    files = []
    for path in unique_streams([str(p) for p in getattr(index, "files", [])])[:MAX_ROLLOUTS]:
        # A stream keeps the copy it was first read from (its tail offset belongs to that file).
        kept = tracker.streams.setdefault(os.path.basename(path), path)
        files.append(kept if os.path.exists(kept) else path)
    tracker.files = files
    budget = ACTIVITY_STEP
    for path in files:
        if budget <= 0:
            tracker.catching_up = True
            break
        tail = tracker.tails.get(path)
        if tail is None:
            tail = tracker.tails[path] = LineTail(path)
        before = tail.offset
        lines, _ = tail.advance(ctx.reader, budget)
        budget -= max(0, tail.offset - before)
        for raw in lines:
            if b'"item_completed"' not in raw and b'"spawn_agent"' not in raw:
                continue
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict):
                tracker.consume(row)
    else:
        tracker.catching_up = not all(t.complete for t in tracker.tails.values())
    return tracker


# -- the campaign log (shared by grader, integrity and attempts) -----------------------------------
#: ``(code, needles)``: a log line counts when any needle occurs in it (case-sensitive substrings of
#: the harness's own messages).
LOG_RULES = (
    ("marker", ("differs from its marker", "differs from its durable result")),
    # Loop grades say "grade error:"; background snapshot grades say "[snapshot N] grade failed:".
    ("grade_error", ("grade error", "grade failed:")),
    ("promotion_skipped", ("promotion skipped",)),
    ("snapshot_unavailable", ("snapshot unavailable",)),
    ("broker", ("[broker]",)),
    ("restart", ("restarted", "restarting", "supervisor")),
    ("dropped", ("writes dropped", "dropped write", "write dropped")),
    ("enospc", ("No space left",)),
    ("traceback", ("Traceback (most recent call last)",)),
    ("error_line", ("Error: ",)),
    ("idle_perf", ("[idle-perf]",)),
)
LOG_STEP = 1 << 20
_LOGS: dict[str, LogScan] = {}


class LineTail:
    """Complete lines appended to a file since the last call, at most ``step`` bytes per call.

    Read-only; a file replaced (another inode) or truncated is read again from the start and
    :meth:`advance` reports it through ``reset``. Bytes read are charged to the reader's budget."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.offset = 0
        self.ident: tuple | None = None
        self.partial = b""
        self.complete = False

    def advance(self, reader, step: int) -> tuple[list[bytes], bool]:
        try:
            st = os.stat(self.path)
        except OSError:
            return [], False
        ident = (st.st_dev, st.st_ino)
        reset = self.ident is not None and (ident != self.ident or st.st_size < self.offset)
        if reset or self.ident is None:
            self.offset, self.partial, self.ident = 0, b"", ident
        if st.st_size == self.offset:
            self.complete = True
            return [], reset
        try:
            with open(self.path, "rb") as stream:
                stream.seek(self.offset)
                data = stream.read(int(min(step, reader.available())))
        except OSError:
            return [], reset
        reader.remaining -= len(data)
        self.offset += len(data)
        self.complete = self.offset >= st.st_size
        lines = (self.partial + data).split(b"\n")
        self.partial = lines.pop()
        return lines, reset


class LogScan:
    """Counts of :data:`LOG_RULES` over a whole text log, read incrementally (1 MiB per refresh)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.tail = LineTail(path)
        self._reset()

    def _reset(self) -> None:
        self.lines = 0
        self.counts: dict[str, int] = {}
        self.last: dict[str, tuple[int, str]] = {}  # code -> (line number, text)
        self.first_at: float | None = None

    @property
    def complete(self) -> bool:
        return self.tail.complete

    def advance(self, reader) -> LogScan:
        lines, reset = self.tail.advance(reader, LOG_STEP)
        if reset:
            self._reset()
        for raw in lines:
            self.lines += 1
            text = raw.decode(errors="replace")
            if self.first_at is None and text.startswith("{"):
                self.first_at = epoch(text.partition('"at": "')[2].partition('"')[0])
            for code, needles in LOG_RULES:
                if any(n in text for n in needles):
                    self.counts[code] = self.counts.get(code, 0) + 1
                    self.last[code] = (self.lines, clean(text.strip(), 160))
        return self


def log_scan(ctx, path: str | None) -> LogScan | None:
    if not path or not is_file(path):
        return None
    scan = _LOGS.get(path)
    if scan is None:
        scan = _LOGS[path] = LogScan(path)
    return scan.advance(ctx.reader)


def campaign_log(ctx) -> str | None:
    """The live campaign log: the runner's standard output when it is a file, else the chain's log."""
    pid = ctx.experiment.get("runner_pid")
    linked = proc_link(pid, 1) if ctx.live else None
    if linked and linked.startswith("/") and is_file(linked):
        return linked
    info = chain(ctx)
    return info["attempts"][info["current"]].get("launch_log")


def proc_link(pid: Any, fd: int) -> str | None:
    """Where a live process's descriptor points (read-only, ``None`` when not visible)."""
    if type(pid) is not int:
        return None
    try:
        return os.readlink(f"/proc/{pid}/fd/{fd}")
    except OSError:
        return None


def proc_env(pid: Any, name: str) -> str | None:
    if type(pid) is not int:
        return None
    try:
        with open(f"/proc/{pid}/environ", "rb") as stream:
            data = stream.read(1 << 20)
    except OSError:
        return None
    prefix = name.encode() + b"="
    for item in data.split(b"\0"):
        if item.startswith(prefix):
            return item[len(prefix) :].decode(errors="replace")
    return None


__all__ = [
    "AgentActivity",
    "COMMAND_RULES",
    "LOG_RULES",
    "LineTail",
    "LogScan",
    "Needles",
    "activity",
    "ago",
    "campaign_log",
    "chain",
    "log_scan",
    "duration",
    "line",
    "out_root",
    "panel",
    "previous_workspace",
    "private_rows",
    "proc_env",
    "proc_link",
    "quantile",
    "ratio_text",
    "row",
    "series",
    "share_rows",
    "table",
    "unique_streams",
]
