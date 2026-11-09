"""The author agent tree: who exists, what each is doing, its tokens and its activity spans.

Read from the agents' own rollout streams (``session_meta``, ``task_*``, ``item_completed``,
``token_usage_record``, ``token_count``), found the way :mod:`..terminal_metadata` finds them: the
run's ``rounds/round_*.codex_rollout_snapshot``, the codex home a round summary names, and the
policy's home root. Streams are tailed incrementally; a finished run's ``timing_detailed.json``
``author_agents`` block (``merlin.phase1_author_agents.v1``) supplies authoritative totals. With no
rollout at all, the run-side lead stream ``rounds/round_NN.codex_events.timestamped.jsonl`` is the
fallback. The parsed index is kept per experiment so :mod:`.tokens` and :mod:`.events` reuse it.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..io import bounded_glob, clean, epoch, is_dir, is_file, num, obj, rows, scan_dir
from . import TOKEN_FIELDS, empty_tokens

MAX_SPANS = 400
MAX_INTERVALS = 50_000
MAX_FILES = 64
REDISCOVER_S = 30.0
#: Rows a resumed session writes before its next turn; they are not the agent working.
BOOKKEEPING = frozenset({"turn_context", "thread_settings_applied", "task_started"})
_INDEXES: dict[str, RolloutIndex] = {}


def tokens_of(usage: Any) -> dict | None:
    """The console TOKENS mapping of one recorded usage object."""
    usage = obj(usage)
    if not usage:
        return None
    total_in = num(usage.get("input_tokens"))
    cached = num(usage.get("cached_input_tokens")) or 0
    written = num(usage.get("cache_write_input_tokens")) or 0
    out = {
        "input": total_in,
        "cache_read": num(usage.get("cached_input_tokens")),
        "cache_write": num(usage.get("cache_write_input_tokens")),
        "fresh_input": (max(0, total_in - cached - written)
                        if total_in is not None and usage.get("cached_input_tokens") is not None else None),
        "output": num(usage.get("output_tokens")),
        "reasoning": num(usage.get("reasoning_output_tokens")),
        "total": num(usage.get("total_tokens")),
    }
    return out if any(v is not None for v in out.values()) else None


def _thread(thread_id: str) -> dict[str, Any]:
    return {
        "id": thread_id,
        "parent_id": None,
        "name": None,
        "path": None,
        "role": None,
        "depth": None,
        "state": "observed",
        "started_at": None,
        "last_activity_at": None,
        "turns": 0,
        "closed_turn_s": 0.0,
        "open_turn": None,
        "commands": 0,
        "failed_commands": 0,
        "current": None,
        "last_message": None,
        "model": None,
        "effort": None,
        "tokens": None,
        "token_source": None,
        "samples": [],
        "spans": deque(maxlen=MAX_SPANS),
        "intervals": {"turn": [], "tool": [], "wait": [], "tool_call": []},
        "transport_calls": {},
        "transport_records": 0,
        "intervals_complete": True,
        "recent_commands": deque(maxlen=40),
        "messages": deque(maxlen=8),
        "file": None,
        "authoritative": None,
    }


class RolloutIndex:
    """Parsed agent streams of one experiment, updated incrementally."""

    def __init__(self) -> None:
        self.threads: dict[str, dict[str, Any]] = {}
        self.file_thread: dict[str, str] = {}
        self.files: list[Path] = []
        self.discovered_at = 0.0
        self.rate_limit: dict | None = None
        self.context_window: int | None = None
        self.feed: deque = deque(maxlen=300)
        self.pending_child: dict[str, dict] = {}
        self.fallback = False
        self.helper_ceiling: int | None = None
        self.rounds_seen: tuple | None = None
        self.run_name = ""

    # -- discovery -----------------------------------------------------------------------------
    def discover(self, ctx) -> None:
        run = Path(ctx.experiment.get("run_dir") or "")
        if not is_dir(run):
            return
        rounds = run / "rounds"
        self.run_name = run.name
        roots: list[Path] = [p for p in bounded_glob(rounds, "round_*.codex_rollout_snapshot", 16)]
        for summary in sorted(bounded_glob(rounds, "round_*.codex_summary.json", 16)):
            doc = obj(ctx.reader.document(summary, limit=4 << 20))
            home = obj(doc.get("codex_home")).get("codex_home")
            if isinstance(home, str):
                roots.append(Path(home) / "sessions")
            ceiling = obj(doc.get("subagent_capacity")).get("max_concurrent_threads_per_session")
            if type(ceiling) is int:
                self.helper_ceiling = ceiling
        env = obj(ctx.reader.yaml(run / "environment.yaml", limit=16 << 20))
        config = obj(env.get("run_config"))
        ceiling = obj(config.get("author_subagents")).get("max_concurrent_threads_per_session")
        if self.helper_ceiling is None and type(ceiling) is int:
            self.helper_ceiling = ceiling
        for policy in (obj(env.get("codex_runtime_policy")), obj(config.get("codex_runtime"))):
            home_root = policy.get("home_root")
            if isinstance(home_root, str) and home_root.startswith("/"):
                for entry in scan_dir(Path(home_root), 64):
                    suffix = entry.name[len(run.name) + 2 :]
                    if entry.name.startswith(run.name + "_r") and suffix.isdigit():
                        roots.append(Path(entry.path) / "sessions")
        files = list(
            itertools.islice(
                itertools.chain.from_iterable(
                    bounded_glob(root, "*/*/*/rollout-*.jsonl", MAX_FILES) for root in dict.fromkeys(roots)
                ),
                MAX_FILES,
            )
        )
        self.files = files
        self.discovered_at = time.monotonic()

    # -- rollout rows --------------------------------------------------------------------------
    def _row(self, path: str, row: dict) -> None:
        payload = row.get("payload")
        kind = row.get("type")
        when = epoch(row.get("timestamp"))
        if kind == "session_meta":
            if path in self.file_thread or not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
                return
            thread_id = payload["id"]
            if thread_id in self.threads and self.threads[thread_id]["file"] not in (None, path):
                self.file_thread[path] = ""  # Another copy of a stream already indexed (snapshot vs home).
                return
            spawn = obj(obj(obj(payload.get("source")).get("subagent")).get("thread_spawn"))
            thread = self.threads.setdefault(thread_id, _thread(thread_id))
            parent = payload.get("parent_thread_id") or spawn.get("parent_thread_id")
            thread.update(
                parent_id=parent if isinstance(parent, str) else None,
                name=payload.get("agent_nickname") or spawn.get("agent_nickname"),
                path=payload.get("agent_path") or spawn.get("agent_path") or (None if parent else "/root"),
                role="subagent" if parent else "lead",
                depth=spawn.get("depth") if type(spawn.get("depth")) is int else (1 if parent else 0),
                started_at=epoch(payload.get("timestamp")) or when,
                file=path,
                start_ordinal=payload.get("subagent_history_start_ordinal"),
            )
            thread.update({k: v for k, v in self.pending_child.pop(thread_id, {}).items() if v})
            thread["round"] = round_of(path, self.run_name)
            self.file_thread[path] = thread_id
            return
        thread_id = self.file_thread.get(path)
        if not thread_id or not isinstance(payload, dict):
            return
        thread = self.threads[thread_id]
        start = thread.get("start_ordinal")
        if type(start) is int and (type(row.get("ordinal")) is not int or row["ordinal"] < start):
            return  # Forked parent history is not this helper's own work.
        if payload.get("thread_id", thread_id) != thread_id:
            return
        previous = thread.get("work_at")
        if when:
            thread["last_activity_at"] = max(thread["last_activity_at"] or 0, when)
            if kind not in BOOKKEEPING and payload.get("type") not in BOOKKEEPING:
                thread["work_at"] = max(thread.get("work_at") or 0, when)
        if kind == "token_usage_record":
            tokens = tokens_of(payload.get("thread_token_usage") or payload.get("usage"))
            if tokens and payload.get("thread_token_usage"):
                thread["token_source"] = "token_usage_record"
                self._sample(thread, when, tokens)
            return
        if kind == "response_item" and when is not None:
            self._transport(thread, payload, when)
        if kind != "event_msg":
            return
        event = payload.get("type")
        if event == "token_count":
            info = obj(payload.get("info"))
            if type(info.get("model_context_window")) is int:
                self.context_window = info["model_context_window"]
            primary = obj(obj(payload.get("rate_limits")).get("primary"))
            if primary:
                self.rate_limit = {
                    "used_percent": num(primary.get("used_percent")),
                    "window_minutes": num(primary.get("window_minutes")),
                    "resets_at": epoch(primary.get("resets_at")),
                }
            tokens = tokens_of(info.get("total_token_usage"))
            if tokens and thread["token_source"] != "token_usage_record":
                thread["token_source"] = "token_count"
                self._sample(thread, when, tokens)
        elif event == "task_started":
            begin = epoch(payload.get("started_at")) or when
            if thread["open_turn"] and previous and previous >= thread["open_turn"]:
                # The previous turn never closed (its process was stopped): it ended at its last record.
                thread["closed_turn_s"] += previous - thread["open_turn"]
                self._span(thread, thread["open_turn"], previous, "turn")
            thread["turns"] += 1
            thread["state"] = "working"
            thread["open_turn"] = begin
        elif event in ("task_complete", "task_aborted"):
            begin = epoch(payload.get("started_at")) or thread["open_turn"]
            end = epoch(payload.get("completed_at")) or when
            if begin and end and end >= begin:
                thread["closed_turn_s"] += end - begin
                self._span(thread, begin, end, "turn")
            thread["open_turn"] = None
            thread["state"] = "completed" if event == "task_complete" else "aborted"
            message = payload.get("last_agent_message")
            if isinstance(message, str) and message.strip():
                thread["last_message"] = clean(message, 400)
        elif event == "item_completed":
            self._item(thread, obj(payload.get("item")), payload, when)

    def _transport(self, thread: dict, payload: dict, when: float) -> None:
        """Match outer call/return metadata; background command lifetimes are separate."""
        kind, key = payload.get("type"), payload.get("call_id")
        if not isinstance(key, str):
            return
        if kind in ("function_call", "custom_tool_call"):
            thread["transport_records"] += 1
            thread["transport_calls"].setdefault(key, when)
        elif kind in ("function_call_output", "custom_tool_call_output"):
            begin = thread["transport_calls"].pop(key, None)
            if begin is not None and when >= begin:
                self._span(thread, begin, when, "tool_call")

    def _item(self, thread: dict, item: dict, payload: dict, when: float | None) -> None:
        kind = item.get("type")
        begin, end = epoch(payload.get("started_at_ms")), epoch(payload.get("completed_at_ms"))
        if kind == "CommandExecution":
            command = item.get("command")
            text = clean(command[-1] if isinstance(command, list) and command else command, 240)
            code = item.get("exit_code")
            failed = item.get("status") == "failed" or (type(code) is int and code != 0)
            thread["commands"] += 1
            thread["failed_commands"] += int(failed)
            thread["current"] = text
            if begin and end:
                self._span(thread, begin, end, "tool")
            duration = obj(item.get("duration"))
            seconds = (num(duration.get("secs")) or 0) + (num(duration.get("nanos")) or 0) / 1e9 if duration else None
            thread["recent_commands"].append(
                {"t": end or when, "command": text, "exit_code": code, "duration_s": seconds, "failed": failed}
            )
            if failed:
                self.feed.append(
                    {
                        "t": end or when,
                        "kind": "command_failed",
                        "severity": "warn",
                        "agent": thread["id"],
                        "text": f"exit {code}: {text}",
                    }
                )
        elif kind == "AgentMessage":
            parts = [clean(p.get("text"), 400) for p in rows(item.get("content")) if p.get("text")]
            if parts:
                thread["last_message"] = " ".join(parts)[:400]
                thread["messages"].append({"t": end or when, "text": thread["last_message"]})
                self.feed.append(
                    {
                        "t": end or when,
                        "kind": "message",
                        "severity": "info",
                        "agent": thread["id"],
                        "text": thread["last_message"],
                    }
                )
        elif kind == "SubAgentActivity" and item.get("kind") == "started":
            child = item.get("agent_thread_id")
            if isinstance(child, str):
                extra = {"model": item.get("model"), "effort": item.get("reasoning_effort")}
                if child in self.threads:
                    self.threads[child].update({k: v for k, v in extra.items() if v})
                else:
                    self.pending_child[child] = extra
        elif kind == "Extension" and begin and end and end > begin:
            self._span(thread, begin, end, "wait")

    @staticmethod
    def _span(thread: dict, begin: float, end: float, kind: str) -> None:
        spans = thread["spans"]
        if spans and spans[-1][2] == kind and begin - spans[-1][1] < 1.0:
            spans[-1][1] = max(spans[-1][1], end)  # Merge adjacent spans of one kind.
        else:
            spans.append([begin, end, kind])
        # The exact union per kind, kept whole (the drawn spans are bounded) for time attribution.
        exact = thread["intervals"][kind]
        if exact and exact[-1][0] <= begin <= exact[-1][1]:
            exact[-1][1] = max(exact[-1][1], end)
        else:
            exact.append([begin, end])
            if len(exact) > MAX_INTERVALS:
                thread["intervals_complete"] = False
                exact[:] = [[a[0], max(a[1], b[1])] for a, b in zip(exact[::2], exact[1::2])] + exact[
                    len(exact) // 2 * 2 :
                ]

    @staticmethod
    def _sample(thread: dict, when: float | None, tokens: dict) -> None:
        thread["tokens"] = tokens
        if when:
            samples = thread["samples"]
            if samples and samples[-1][0] == when:
                samples[-1] = (when, tokens)
            else:
                samples.append((when, tokens))
            if len(samples) > 4000:
                del samples[1 : len(samples) // 2 : 2]  # Thin the older half; keep the newest dense.

    def _reset(self, path: str) -> None:
        thread_id = self.file_thread.pop(path, None)
        if thread_id:
            self.threads.pop(thread_id, None)

    # -- run-side fallback stream --------------------------------------------------------------
    def _lead_row(self, row: dict) -> None:
        event = obj(row.get("event"))
        when = epoch(row.get("arrived_at"))
        kind = event.get("type")
        if kind == "thread.started" and isinstance(event.get("thread_id"), str):
            thread = self.threads.setdefault(event["thread_id"], _thread(event["thread_id"]))
            thread.update(role="lead", path="/root", depth=0, started_at=when, file="run-stream")
            self.file_thread["run-stream"] = event["thread_id"]
            return
        thread = self.threads.get(self.file_thread.get("run-stream") or "")
        if thread is None:
            return
        if when:
            thread["last_activity_at"] = max(thread["last_activity_at"] or 0, when)
        item = obj(event.get("item"))
        if kind == "turn.started":
            thread.update(state="working", open_turn=when)
            thread["turns"] += 1
        elif kind in ("turn.completed", "turn.failed"):
            if thread["open_turn"] and when:
                thread["closed_turn_s"] += when - thread["open_turn"]
                self._span(thread, thread["open_turn"], when, "turn")
            thread.update(open_turn=None, state="completed" if kind == "turn.completed" else "aborted")
            usage = tokens_of(event.get("usage"))
            if usage:
                previous = thread["tokens"] or empty_tokens()
                summed = {k: (previous.get(k) or 0) + (usage.get(k) or 0) for k in TOKEN_FIELDS}
                thread["token_source"] = "turn usage (run stream)"
                self._sample(thread, when, summed)
        elif item.get("type") == "command_execution":
            thread["current"] = clean(item.get("command"), 240)
            if kind == "item.completed":
                code = item.get("exit_code")
                failed = item.get("status") == "failed" or (type(code) is int and code != 0)
                thread["commands"] += 1
                thread["failed_commands"] += int(failed)
                thread["recent_commands"].append(
                    {"t": when, "command": thread["current"], "exit_code": code, "duration_s": None, "failed": failed}
                )
                if failed:
                    self.feed.append(
                        {
                            "t": when,
                            "kind": "command_failed",
                            "severity": "warn",
                            "agent": thread["id"],
                            "text": f"exit {code}: {thread['current']}",
                        }
                    )
        elif item.get("type") == "agent_message" and kind == "item.completed" and item.get("text"):
            thread["last_message"] = clean(item["text"], 400)
            thread["messages"].append({"t": when, "text": thread["last_message"]})
            self.feed.append(
                {
                    "t": when,
                    "kind": "message",
                    "severity": "info",
                    "agent": thread["id"],
                    "text": thread["last_message"],
                }
            )

    # -- refresh -------------------------------------------------------------------------------
    def refresh(self, ctx) -> None:
        # A new round or turn writes into rounds/ and may bring a new agent home: look again at once.
        rounds = ctx.reader.identity(Path(ctx.experiment.get("run_dir") or "") / "rounds")
        if not self.files or time.monotonic() - self.discovered_at > REDISCOVER_S or rounds != self.rounds_seen:
            self.rounds_seen = rounds
            self.discover(ctx)
        for path in self.files:
            key = str(path)
            ctx.reader.poll(path, lambda row, key=key: self._row(key, row), lambda key=key: self._reset(key))
        self.fallback = not any(t.get("file") not in (None, "run-stream") for t in self.threads.values())
        if self.fallback:
            run = Path(ctx.experiment.get("run_dir") or "")
            for stream in sorted(bounded_glob(run / "rounds", "round_*.codex_events.timestamped.jsonl", 16)):
                ctx.reader.poll(stream, self._lead_row)
        self._authoritative(ctx)

    def _authoritative(self, ctx) -> None:
        run = Path(ctx.experiment.get("run_dir") or "")
        timing = run / "timing_detailed.json"
        if not is_file(timing):
            return
        doc = obj(ctx.reader.document(timing, limit=64 << 20))
        block = obj(doc.get("author_agents"))
        for row in rows(block.get("agents")):
            thread_id = row.get("thread_id")
            if not isinstance(thread_id, str):
                continue
            thread = self.threads.setdefault(thread_id, _thread(thread_id))
            tok = obj(row.get("tokens"))
            thread["authoritative"] = {
                "active_s": num(row.get("active_s")),
                "turns": row.get("turns"),
                "state": row.get("state"),
                "commands": obj(row.get("commands")).get("n"),
                "failed": obj(row.get("commands")).get("failed"),
                "tokens": {
                    "input": num(tok.get("input_total")),
                    "fresh_input": num(tok.get("fresh_input")),
                    "cache_read": num(tok.get("cache_read")),
                    "cache_write": num(tok.get("cache_write")),
                    "output": num(tok.get("output")),
                    "reasoning": num(tok.get("reasoning")),
                    "total": num(tok.get("total")),
                }
                if tok.get("available") is not False
                else None,
            }
            for key, value in (
                ("parent_id", row.get("parent_thread_id")),
                ("path", row.get("agent_path")),
                ("name", row.get("agent_nickname")),
                ("role", row.get("role")),
                ("depth", row.get("depth")),
                ("started_at", epoch(row.get("started_at"))),
                ("last_activity_at", epoch(row.get("last_event_at"))),
            ):
                if thread.get(key) is None and value is not None:
                    thread[key] = value


def round_of(path: str, run_name: str) -> int | None:
    """The round an agent stream belongs to: its codex home ``<run>_rNN`` or snapshot ``round_NN``."""
    for part in Path(path).parts:
        for prefix in (f"{run_name}_r", "round_"):
            digits = part[len(prefix) :].split(".", 1)[0] if part.startswith(prefix) else ""
            if digits.isdigit():
                return int(digits)
    return None


def index_for(ctx) -> RolloutIndex:
    """The experiment's shared index (created on first use)."""
    key = str(ctx.experiment.get("id") or ctx.experiment.get("run_dir"))
    index = _INDEXES.get(key)
    if index is None:
        index = _INDEXES[key] = RolloutIndex()
    return index


def view(thread: dict, now: float, private: bool, live: bool = True) -> dict[str, Any]:
    """One agent row of the section contract.

    In a run that is no longer live, a turn that never closed ends at the agent's last recorded
    activity and its state reads ``stopped`` (the stream ended mid-turn), never ``working``."""
    auth = thread.get("authoritative") or {}
    closed = thread.get("closed_at")
    end = closed if closed is not None else now if live else (thread["last_activity_at"] or thread["open_turn"] or 0)
    active = thread["closed_turn_s"] + (max(0.0, end - thread["open_turn"]) if thread["open_turn"] else 0.0)
    has_activity = thread["turns"] or thread["commands"] or thread["spans"]
    state = thread["state"] if has_activity or not auth.get("state") else clean(auth["state"], 20)
    if state == "working" and (not live or closed is not None):
        state = "stopped"
    return {
        "id": thread["id"],
        "parent_id": thread["parent_id"],
        "name": clean(thread["name"], 40) or None,
        "path": clean(thread["path"], 80) or None,
        "role": thread["role"],
        "depth": thread["depth"],
        "state": state,
        "started_at": thread["started_at"],
        "last_activity_at": thread["last_activity_at"],
        # Time worked comes from the agent's own stream when it has one; the run's author_agents record
        # can count a turn's host startup as agent time, so it is kept beside it, not instead of it.
        "active_s": active if has_activity else auth.get("active_s"),
        "recorded_active_s": auth.get("active_s"),
        "outer_tool_calls_recorded": thread["transport_records"] > 0,
        "intervals_complete": thread["intervals_complete"],
        "turns": auth.get("turns") if type(auth.get("turns")) is int else thread["turns"],
        "commands": auth.get("commands") if type(auth.get("commands")) is int else thread["commands"],
        "failed_commands": auth.get("failed") if type(auth.get("failed")) is int else thread["failed_commands"],
        "current": thread["current"] if private else None,
        "last_message": thread["last_message"] if private else None,
        "model": clean(thread["model"], 40) or None,
        "effort": clean(thread["effort"], 20) or None,
        "tokens": auth.get("tokens") or thread["tokens"] or empty_tokens(),
        "spans": [list(s) for s in thread["spans"]] + open_span(thread, now, live),
        "stage": None,
        "round": thread.get("round"),
    }


def open_span(thread: dict, now: float, live: bool) -> list[list]:
    """The turn still open: to now in a live run, to the last recorded activity otherwise."""
    begin = thread["open_turn"]
    if not begin:
        return []
    end = thread.get("closed_at")
    if end is None:
        end = now if live else (thread["last_activity_at"] or begin)
    return [[begin, end, "turn"]] if end >= begin else []


def intervals(thread: dict, now: float, live: bool) -> dict[str, list[tuple[float, float]]]:
    """The thread's exact activity per kind (``turn``/``tool``/``wait``), including an open turn."""
    out = {kind: [tuple(iv) for iv in ivs] for kind, ivs in thread["intervals"].items()}
    out["turn"] += [(a, b) for a, b, _ in open_span(thread, now, live)]
    end = thread.get("closed_at") or (now if live else thread.get("last_activity_at"))
    if end is not None:
        out["tool_call"] += [(begin, end) for begin in thread["transport_calls"].values() if end >= begin]
    return out


def close_orphans(index: RolloutIndex, lifecycle: dict[str, Any] | None, live: bool) -> dict[str, Any] | None:
    """End turns whose author process is no longer running (the stream stopped without a close record).

    The lifecycle section says whether the author runs: when the host interrupted it or the session
    ended, every open turn ends at its thread's last record; while a later turn runs, a thread with no
    record since that turn began belongs to the earlier one. Returns the session note for the section."""
    life = lifecycle or {}
    turns = [t for t in life.get("turns") or [] if isinstance(t, dict)]
    current = turns[-1] if turns else {}
    running = life.get("author_running")
    for thread in index.threads.values():
        thread["closed_at"] = None
        if not thread["open_turn"]:
            continue
        last = thread.get("work_at") or thread["last_activity_at"] or thread["open_turn"]
        if not live:
            # A stopped run: an open turn ended at the thread's last work (not its last bookkeeping row).
            thread["closed_at"] = max(thread["open_turn"], last)
            continue
        began = current.get("startup_start") or current.get("agent_start")
        if running is False or (running and began and last < began):
            thread["closed_at"] = max(thread["open_turn"], last)
    if (running is None and not turns) or not live:
        return (
            None
            if not turns
            else {
                "running": False,
                "phase": "ended",
                "since": life.get("phase_since"),
                "detail": life.get("phase_detail"),
            }
        )
    return {
        "running": running,
        "phase": life.get("phase"),
        "since": life.get("phase_since"),
        "detail": life.get("phase_detail"),
    }


def read(ctx) -> dict[str, Any]:
    index = index_for(ctx)
    index.refresh(ctx)
    session = close_orphans(index, ctx.shared.get("lifecycle"), ctx.live)
    agents = [
        view(t, ctx.now, ctx.operator_private, ctx.live)
        for t in index.threads.values()
        if t.get("file") or t.get("authoritative")
    ]
    agents.sort(key=lambda a: (a["depth"] if type(a["depth"]) is int else 9, a["started_at"] or 0))
    notes = []
    if not agents:
        notes.append("No agent stream recorded for this run.")
    if index.fallback and agents:
        notes.append("No per-thread rollout found; the lead's run-side stream is shown alone.")
    if any(a["tokens"].get("total") is None for a in agents):
        notes.append("Token usage not recorded for some agents.")
    sources = {t.get("file") for t in index.threads.values()}
    authoritative = any(t.get("authoritative") for t in index.threads.values())
    source = "run_stream" if index.fallback else ("rollout" if sources - {None} else None)
    if authoritative:
        source = f"timing_detailed+{source}" if source else "timing_detailed"
    return {
        "agents": agents,
        "totals": {
            "agents": len(agents),
            "working": sum(a["state"] == "working" for a in agents),
            "helpers_working": sum(a["role"] != "lead" and a["state"] == "working" for a in agents),
            "subagents": sum(a["role"] == "subagent" for a in agents),
            "helper_ceiling": index.helper_ceiling,
        },
        "usage_completeness": {
            "observed_threads": len(agents),
            "threads_with_usage": sum(a["tokens"].get("total") is not None for a in agents),
            "all_observed_threads_have_usage": bool(agents) and all(
                a["tokens"].get("total") is not None for a in agents),
            "all_run_threads_verified": False,
            "totals_are_lower_bounds": True,
            "scope": "discovered thread streams; live, missing or undiscovered usage is not zero",
        },
        "source": source,
        "session": session,
        "notes": notes,
    }


def inspect(ctx, key: str) -> dict[str, Any]:
    index = index_for(ctx)
    thread = index.threads.get(key)
    if thread is None:
        return {"title": f"agent {clean(key, 40)}: not recorded", "sections": [], "paths": []}
    row = view(thread, ctx.now, ctx.operator_private, ctx.live)
    tok = row["tokens"]

    def fmt(value):
        return (
            "not recorded"
            if value is None
            else (f"{value:,.0f}" if isinstance(value, int | float) else clean(value, 200))
        )

    sections = [
        {
            "title": "Identity",
            "lines": [
                f"id {row['id']}",
                f"parent {row['parent_id'] or '(lead)'}",
                f"path {row['path'] or 'not recorded'}  name {row['name'] or '-'}",
                f"model {row['model'] or 'not recorded'}  effort {row['effort'] or '-'}",
                f"state {row['state']}  turns {row['turns']}  worked {fmt(row['active_s'])} s (from its stream)",
                f"the run's author_agents record says {fmt(row['recorded_active_s'])} s"
                + (
                    " (it counts host turn startup as agent time)"
                    if isinstance(row["recorded_active_s"], int | float)
                    and isinstance(row["active_s"], int | float)
                    and row["recorded_active_s"] > row["active_s"] + 60
                    else ""
                ),
            ],
        },
        {"title": "Tokens", "lines": [f"{k:<12}{fmt(tok.get(k))}" for k in TOKEN_FIELDS]},
        {"title": "Commands", "lines": [f"{row['commands']} run, {row['failed_commands']} failed"]},
    ]
    if ctx.operator_private:
        recent = list(thread["recent_commands"])[-15:]
        sections.append(
            {
                "title": "Recent commands",
                "lines": [
                    f"{'FAIL' if c['failed'] else 'ok  '} exit {c['exit_code']} {fmt(c['duration_s'])}s  {c['command']}"
                    for c in reversed(recent)
                ]
                or ["none recorded"],
            }
        )
        sections.append(
            {
                "title": "Recent messages",
                "lines": [m["text"] for m in reversed(thread["messages"])] or ["none recorded"],
            }
        )
    paths = [thread["file"]] if ctx.operator_private and thread.get("file") not in (None, "run-stream") else []
    name = row["name"] or ("lead" if row["role"] == "lead" else row["path"] or row["id"][:8])
    return {"title": f"agent {name}", "sections": sections, "paths": paths}


__all__ = ["RolloutIndex", "index_for", "inspect", "read", "tokens_of", "view"]
