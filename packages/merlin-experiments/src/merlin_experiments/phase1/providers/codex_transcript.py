"""Normalize driver tool events and retain bounded, paired transcript records."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ITEM_COMMAND_EXECUTION = "command_execution"


ITEM_FILE_CHANGE = "file_change"


_INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md", "AGENT.md", "TASK.md")


def _instruction_files(ws: Path) -> dict[str, int]:
    """Which instruction files the workspace carries, and their sizes."""
    found = {}
    for name in _INSTRUCTION_FILES:
        path = ws / name
        if path.is_file():
            found[name] = path.stat().st_size
    return found


_TOOL_OUTPUT_CAP = 20000


def _clip_tool_output(text: str, cap: int = _TOOL_OUTPUT_CAP) -> str:
    """Bound a recorded tool result WITHOUT discarding its tail.

    Head-only truncation destroys evidence silently. A discovery command that prints a large
    structure and THEN its summary -- ``print(facts); print('DERIVED_LEVERS', levers)`` -- loses
    exactly the closing lines the conformance checks read, so the run is graded as though the agent
    never ran the tool. Measured on the atlas arm-4 run of 2026-09-04: 29 of 467 results were cut at
    the cap, and the one carrying the arm-4 discovery evidence was among them. Keeping both ends
    preserves the structure and the summary inside the same budget.
    """
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    return f"{text[:head]}\n...[{len(text) - head - tail} chars elided by the harness]...\n{text[-tail:]}"


def _tool_block(item: dict, tool_use_id: str) -> dict:
    """Render a Codex tool item as a Claude ``tool_use`` block.

    ``tool_use_id`` is REQUIRED and must be the same string the paired ``tool_result`` carries: it is
    the only key joining a call to its result, so every consumer of the transcript keys on it. Emitting
    the block WITHOUT an id silently costs the run its entire tool telemetry -- aet's stream parser
    records a call only ``if tc.tool_use_id``, so the atlas round of 2026-09-04 reported
    ``tool_call_count=0`` and ``unique_tools_used=[]`` for a round that made 125 tool calls, and the
    per-call latency (measured as the gap between a tool_use and its tool_result) had nothing to pair.
    """
    itype = item.get("type")
    if itype == ITEM_COMMAND_EXECUTION:
        return {"type": "tool_use", "id": tool_use_id, "name": "Bash", "input": {"command": item.get("command", "")}}
    if itype == ITEM_FILE_CHANGE:
        changes = item.get("changes")
        return {
            "type": "tool_use",
            "id": tool_use_id,
            "name": "Edit",
            "input": {"changes": changes if isinstance(changes, list) else []},
        }
    return {
        "type": "tool_use",
        "id": tool_use_id,
        "name": str(itype or "tool"),
        "input": {k: v for k, v in item.items() if k not in ("id", "type")},
    }


class _Transcript:
    """Writes the harness's Claude-stream JSONL, one flushed line at a time."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w")
        self.path = path

    def emit(self, obj: dict) -> None:
        self._f.write(json.dumps(obj) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass


def last_message_path(ws: Path, final_path: Path, sandbox: str) -> Path:
    """Where codex may actually WRITE its last message (``-o``).

    ``final_path`` lives under the run directory, which is on /scratch -- and the sandbox tmpfs-hides
    /scratch* by design, so codex could not create it and reported ``Failed to write last message file
    ... (os error 2)`` on every sandboxed round. The read afterwards is guarded, so the loss was
    SILENT: the round's ``result`` was simply empty. Inside the sandbox the workspace is the one
    writable tree, so the file goes there and is copied out (and removed) once the process exits.
    """
    if sandbox != "bwrap":
        return final_path
    return ws / ".codex_out" / final_path.name


def _error_text(value: Any) -> str:
    """Render an error payload without losing the message nested inside it."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str):
            return message
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)
