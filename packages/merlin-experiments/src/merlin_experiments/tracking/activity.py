"""What the authoring agent did, from its own event stream: a feed, tool-call spans and token use.

Reads ``rounds/round_NN.codex_events.timestamped.jsonl``: one ``{"seq", "arrived_at", "event"}`` line
per event the provider CLI printed, stamped by the harness as it arrived (the provider module
:mod:`..phase1.providers.codex_agent` owns that wrapper and the event vocabulary, whose constants are
used here).  From it come:

* turns (``turn.started`` .. ``turn.completed`` / ``turn.failed``) with their recorded usage;
* tool calls (``item.started`` .. ``item.completed`` of a command, a file change, an MCP or web call),
  with exit code and status; a call with no completion is RUNNING;
* assistant messages, reasoning summaries, plan (``todo_list``) updates and errors.

A command is labelled (self-check, test, build, read, verdict wait, other) by the words it runs -- the
label is a reading aid and says so on the page; it never decides a grade.  Nothing is inferred about a
round whose stream is absent: it is "not recorded".
"""

from __future__ import annotations

import shlex
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import records as R
from .tail import Tails

#: Labels a command can carry, and the words that put it there (first match wins, in this order).
SELFCHECK, TEST, BUILD, READ, WAIT, OTHER = "selfcheck", "test", "build", "read", "verdict wait", "command"
_WORDS = (
    (WAIT, ("await_verdict",)),
    (SELFCHECK, ("agent_selfcheck", "selfcheck")),
    (TEST, ("pytest", "ctest", "lit", "llvm-lit", "FileCheck", "check-mlir", "check-all")),
    (BUILD, ("cmake", "ninja", "make", "clang", "clang++", "gcc", "g++", "mlir-tblgen")),
    (READ, ("cat", "sed", "rg", "grep", "ls", "find", "head", "tail", "wc", "nl", "tree", "stat", "git")),
)
_SHELLS = ("bash", "/bin/bash", "/usr/bin/bash", "sh", "/bin/sh", "zsh", "/bin/zsh")
FEED_LIMIT = 160
TEXT_LIMIT = 400


def _vocabulary() -> Any:
    from ..phase1.providers import codex_agent as CA

    return CA


def unwrap(command: Any) -> str:
    """The command a shell wrapper runs (``bash -lc '<cmd>'`` -> ``<cmd>``); anything else as given."""
    if isinstance(command, list):
        command = " ".join(str(c) for c in command)
    text = str(command or "").strip()
    try:
        words = shlex.split(text)
    except ValueError:
        return text
    if len(words) >= 3 and words[0] in _SHELLS and words[1].startswith("-") and "c" in words[1]:
        return words[2].strip()
    return text


def label(command: str) -> str:
    """The reading-aid label of a command, from the words in it."""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    names = [w.rpartition("/")[2] for w in words]
    stems = [n.partition(".")[0] for n in names]
    for kind, keys in _WORDS:
        if kind == READ:
            # Only when the command STARTS with a reader (a pipeline segment start counts too).
            starts = [names[0]] if names else []
            starts += [names[i + 1] for i, w in enumerate(words[:-1]) if w in ("|", "&&", ";", "||")]
            if any(s in keys for s in starts):
                return kind
            continue
        if any(k in names or k in stems for k in keys) or any(k in w for w in words for k in keys if "_" in k):
            return kind
    return OTHER


def _clip(value: Any, limit: int = TEXT_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _round_number(path: Path) -> int | None:
    stem = path.name.partition(".")[0]
    digits = stem.partition("_")[2]
    return int(digits) if digits.isdigit() else None


def _stream_files(run_dir: Path) -> list[Path]:
    rounds = Path(run_dir) / "rounds"
    if not rounds.is_dir():
        return []
    return sorted(rounds.glob("round_*.codex_events.timestamped.jsonl"), key=lambda p: (_round_number(p) or 0, p.name))


#: Bounds on what a long run keeps in memory between polls (counters cover everything regardless).
FEED_KEEP = 600
CALL_KEEP = 6000
TURN_KEEP = 4000


def _round_stat(number: int) -> dict[str, Any]:
    return {
        "round": number,
        "first": None,
        "last": None,
        "turns": 0,
        "calls": 0,
        "failed_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "usage_turns": 0,
        "errors": 0,
    }


class ActivityState:
    """Everything the views derive from the streams, updated one event at a time (bounded memory)."""

    def __init__(self) -> None:
        self.CA = _vocabulary()
        self.turns: deque[dict[str, Any]] = deque(maxlen=TURN_KEEP)
        self.open_turns: dict[int, dict[str, Any]] = {}
        self.open_calls: dict[str, dict[str, Any]] = {}
        self.closed: deque[dict[str, Any]] = deque(maxlen=CALL_KEEP)
        self.feed: deque[dict[str, Any]] = deque(maxlen=FEED_KEEP)
        self.rounds: dict[int, dict[str, Any]] = {}
        self.edits: dict[str, int] = {}
        self.counts = {k: 0 for k in (SELFCHECK, TEST, BUILD, READ, WAIT, OTHER)}
        self.n_commands = 0
        self.failed_commands = 0
        self.unknown = 0
        self.todo: list[dict[str, Any]] | None = None
        self.last_message: dict[str, Any] | None = None
        self.last_reasoning: dict[str, Any] | None = None
        self.last_event: float | None = None

    def _note(self, entry: dict[str, Any]) -> None:
        self.feed.append(entry)
        if entry["kind"] == "message":
            self.last_message = entry
        elif entry["kind"] == "reasoning":
            self.last_reasoning = entry

    def add(self, number: int, row: Mapping[str, Any]) -> None:
        CA = self.CA
        stat = self.rounds.get(number)
        if stat is None:
            stat = self.rounds[number] = _round_stat(number)
        at = R.epoch(row.get("arrived_at"))
        if at is not None:
            stat["first"] = at if stat["first"] is None else min(stat["first"], at)
            stat["last"] = at if stat["last"] is None else max(stat["last"], at)
            self.last_event = at if self.last_event is None else max(self.last_event, at)
        event = row.get("event")
        if not isinstance(event, Mapping):
            if "unparsed" in row:
                self._note(
                    {"at": at, "kind": "unparsed", "text": _clip(row.get("unparsed")), "round": number, "error": True}
                )
            return
        etype = event.get("type")
        if etype == CA.EVENT_TURN_STARTED:
            turn = {"round": number, "start": at, "end": None, "status": "running", "usage": None}
            self.open_turns[number] = turn
            self.turns.append(turn)
            stat["turns"] += 1
        elif etype in (CA.EVENT_TURN_COMPLETED, CA.EVENT_TURN_FAILED):
            self._end_turn(number, at, etype, event, stat)
        elif etype == CA.EVENT_ERROR:
            stat["errors"] += 1
            self._note({"at": at, "kind": "error", "round": number, "error": True, "text": _clip(event.get("message"))})
        elif etype in (CA.EVENT_ITEM_STARTED, CA.EVENT_ITEM_UPDATED, CA.EVENT_ITEM_COMPLETED):
            item = event.get("item")
            if isinstance(item, Mapping):
                self._item(etype, item, at, number, stat)
        elif etype != CA.EVENT_THREAD_STARTED:
            self.unknown += 1

    def _end_turn(self, number, at, etype, event, stat) -> None:
        CA = self.CA
        turn = self.open_turns.pop(number, None)
        if turn is None:
            turn = {"round": number, "start": at, "end": None, "status": "running", "usage": None}
            self.turns.append(turn)
        turn["end"] = at
        turn["status"] = "completed" if etype == CA.EVENT_TURN_COMPLETED else "failed"
        usage = event.get("usage") if isinstance(event.get("usage"), Mapping) else None
        turn["usage"] = dict(usage) if usage else None
        if usage:
            stat["usage_turns"] += 1
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    stat[key] += value
        if etype == CA.EVENT_TURN_FAILED:
            stat["errors"] += 1
            error = event.get("error")
            self._note(
                {
                    "at": at,
                    "kind": "turn failed",
                    "round": number,
                    "error": True,
                    "text": _clip(error.get("message") if isinstance(error, Mapping) else error),
                }
            )
        # A turn that ended closes every call it left open: those never recorded a completion.
        for key in [k for k, c in self.open_calls.items() if c["round"] == number]:
            call = self.open_calls.pop(key)
            call["end"], call["status"] = at, "no completion recorded"
            self.closed.append(call)

    def _item(self, etype, item, at, number, stat) -> None:
        CA = self.CA
        itype = item.get("type")
        key = f"{number}:{item.get('id')}"
        completed = etype == CA.EVENT_ITEM_COMPLETED
        if itype in (CA.ITEM_COMMAND_EXECUTION, CA.ITEM_FILE_CHANGE, CA.ITEM_MCP_TOOL_CALL, CA.ITEM_WEB_SEARCH):
            call = self.open_calls.get(key)
            if call is None:
                call = {"round": number, "id": item.get("id"), "type": itype, "start": at, "end": None, "status": None}
                stat["calls"] += 1
                if itype == CA.ITEM_COMMAND_EXECUTION:
                    command = unwrap(item.get("command"))
                    call["command"], call["label"] = _clip(command, 600), label(command)
                    self.n_commands += 1
                    self.counts[call["label"]] += 1
                if not completed:
                    self.open_calls[key] = call
            elif completed:
                self.open_calls.pop(key, None)
            if itype == CA.ITEM_FILE_CHANGE:
                changes = item.get("changes") if isinstance(item.get("changes"), list) else []
                call["changes"] = [
                    {"path": str(c.get("path")), "kind": c.get("kind")} for c in changes if isinstance(c, Mapping)
                ]
            elif itype == CA.ITEM_MCP_TOOL_CALL:
                call["command"] = f"{item.get('server') or '?'}.{item.get('tool') or '?'}"
            elif itype == CA.ITEM_WEB_SEARCH:
                call["command"] = _clip(item.get("query"))
            if completed:
                self._complete(call, item, at, number, stat)
            return
        if itype == CA.ITEM_AGENT_MESSAGE and completed:
            self._note({"at": at, "kind": "message", "round": number, "text": _clip(item.get("text"), 1200)})
        elif itype == CA.ITEM_REASONING and completed:
            self._note({"at": at, "kind": "reasoning", "round": number, "text": _clip(item.get("text"), 600)})
        elif itype == CA.ITEM_ERROR:
            stat["errors"] += 1
            self._note({"at": at, "kind": "error", "round": number, "error": True, "text": _clip(item.get("message"))})
        elif itype == CA.ITEM_TODO_LIST:
            entries = item.get("items") if isinstance(item.get("items"), list) else []
            self.todo = [
                {"text": _clip(e.get("text"), 200), "completed": bool(e.get("completed"))}
                for e in entries
                if isinstance(e, Mapping)
            ]

    def _complete(self, call, item, at, number, stat) -> None:
        CA = self.CA
        call["end"] = at if at is not None else call["start"]
        call["status"] = item.get("status")
        code = item.get("exit_code")
        call["exit_code"] = code if isinstance(code, int) and not isinstance(code, bool) else None
        call["failed"] = bool(call["exit_code"]) or item.get("status") == "failed"
        if call["failed"]:
            stat["failed_calls"] += 1
        output = str(item.get("aggregated_output") or "")
        call["output_tail"] = _clip(output[-600:], 300) if output else None
        self.closed.append(call)
        itype = call["type"]
        if itype == CA.ITEM_COMMAND_EXECUTION:
            if call["failed"]:
                self.failed_commands += 1
            detail = f"exit {call['exit_code']}" + (
                f" \u2014 {call['output_tail']}" if call["failed"] and call["output_tail"] else ""
            )
            self._note(
                {
                    "at": at,
                    "kind": call["label"],
                    "round": number,
                    "error": call["failed"],
                    "text": _clip(call["command"]),
                    "detail": detail,
                }
            )
        elif itype == CA.ITEM_FILE_CHANGE:
            for change in call.get("changes") or ():
                self.edits[change["path"]] = self.edits.get(change["path"], 0) + 1
            paths = ", ".join(f"{c['kind'] or '?'} {c['path']}" for c in call.get("changes") or ())
            self._note(
                {
                    "at": at,
                    "kind": "edit",
                    "round": number,
                    "error": call["failed"],
                    "text": _clip(paths or "file change"),
                }
            )
        else:
            self._note(
                {"at": at, "kind": itype, "round": number, "error": call["failed"], "text": _clip(call.get("command"))}
            )

    def snapshot(self, files: list[Path], tails: dict[str, Any]) -> dict[str, Any]:
        calls = sorted(
            [*self.closed, *self.open_calls.values()], key=lambda c: c["start"] if c["start"] is not None else 0.0
        )
        running = sorted(self.open_calls.values(), key=lambda c: c["start"] if c["start"] is not None else 0.0)
        return {
            "files": [str(p) for p in files],
            "tails": tails,
            "turns": list(self.turns),
            "calls": calls,
            "feed": sorted(self.feed, key=lambda f: f["at"] if f["at"] is not None else 0.0),
            "rounds": [self.rounds[k] for k in sorted(self.rounds)],
            "edits": dict(sorted(self.edits.items(), key=lambda kv: (-kv[1], kv[0]))),
            "command_counts": dict(self.counts),
            "failed_commands": self.failed_commands,
            "n_commands": self.n_commands,
            "running": running,
            "now": {
                "last_message": self.last_message,
                "last_reasoning": self.last_reasoning,
                "running": running[-5:],
                "todo": self.todo,
                "last_event": self.last_event,
            },
            "unknown_events": self.unknown,
            "kept": {"feed": FEED_KEEP, "calls": CALL_KEEP, "turns": TURN_KEEP},
        }


def read(run_dir: Path, inventory: R.Inventory, tails: Tails | None = None) -> dict[str, Any] | None:
    """The agent's activity across every round's timestamped stream, or None when no stream exists.

    With ``tails`` (a live view's), each stream is read from where the last poll stopped; a replaced or
    truncated stream rebuilds the state from the start."""
    files = _stream_files(run_dir)
    if not files:
        inventory.note("rounds/round_NN.codex_events.timestamped.jsonl", Path(run_dir) / "rounds", "absent")
        return None
    tails = tails or Tails()
    key = f"activity:{Path(run_dir)}"
    for _attempt in range(2):
        state = tails.state.get(key)
        if state is None:
            state = tails.state[key] = ActivityState()
        reset = []
        for path in files:
            number = _round_number(path) or 0
            tails.tail(path).poll(lambda row, n=number: state.add(n, row), on_reset=lambda: reset.append(1))
        if not reset:
            break
        # A stream was replaced or truncated: forget everything and read every stream from the start.
        tails.state.pop(key, None)
        for path in files:
            tails.files.pop(str(path), None)
    info = {}
    for path in files:
        tail = tails.tail(path)
        detail = f"{tail.rows_read} rows to byte {tail.offset:,}" + (f", {tail.bad} unreadable" if tail.bad else "")
        inventory.note(f"rounds/{path.name}", path, "read" if tail.present else "absent", detail)
        info[path.name] = {"offset": tail.offset, "rows": tail.rows_read, "bad": tail.bad}
    return state.snapshot(files, info)


# --------------------------------------------------------------------------- Gantt lanes
def lanes(activity: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Gantt lanes for the agent: one for turns, one per tool-call kind."""
    if not activity:
        return []
    turn_spans = [
        {
            "start": t["start"],
            "end": t["end"],
            "kind": "turn" if t["status"] != "failed" else "failed",
            "tip": f"round {t['round']} turn ({t['status']})\n{R.stamp(t['start'])} → {R.stamp(t['end'])}"
            + (f"\nout {t['usage'].get('output_tokens')} tok" if t.get("usage") else ""),
        }
        for t in activity["turns"]
    ]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for c in activity["calls"]:
        kind = c.get("label") or ("edit" if c["type"] == "file_change" else c["type"])
        text = c.get("command") or ", ".join(ch["path"] for ch in c.get("changes") or ())
        by_kind.setdefault(kind, []).append(
            {
                "start": c["start"],
                "end": c["end"],
                "kind": "failed" if c.get("failed") else kind,
                "tip": f"{kind}: {_clip(text, 200)}\n{R.stamp(c['start'])}"
                + (f"  exit {c.get('exit_code')}" if c.get("end") is not None else ""),
            }
        )
    order = (SELFCHECK, TEST, BUILD, WAIT, READ, OTHER, "edit", "mcp_tool_call", "web_search")
    out = [{"lane": "agent turns", "spans": turn_spans}]
    out += [{"lane": f"tool: {k}", "spans": by_kind[k]} for k in order if k in by_kind]
    out += [{"lane": f"tool: {k}", "spans": v} for k, v in by_kind.items() if k not in order]
    return out


__all__ = ["label", "lanes", "read", "unwrap"]
