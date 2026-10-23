"""Every Codex author thread -- the lead and each sub-agent -- accounted from its own rollout.

WHY THE ROLLOUTS AND NOT THE STREAM. ``codex exec --json`` reports the LEAD thread only. Measured on
codex-cli 0.162.0 with two spawned helpers: the stream carried no spawn event, the lead's
``collab_tool_call`` items (``tool: wait``) named no receiver, none of the helpers' commands, edits
or messages appeared, and ``turn.completed.usage`` equalled the sum of the lead's own responses --
the helpers' tokens were simply absent. Every thread instead persists its own
``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<stamp>-<thread>.jsonl``, written live, in which:

* the first ``session_meta`` is the thread's own (``id``; for a helper ``parent_thread_id``,
  ``source.subagent.thread_spawn`` with ``depth``/``agent_path``/``agent_nickname``/``agent_role``,
  and ``subagent_history_start_ordinal``);
* a forked helper's file then COPIES the parent's history -- a second ``session_meta`` and rows with
  ordinals below the start ordinal. Those rows are the parent's work and are not counted again;
* ``token_usage_record`` rows carry ``thread_id``, ``response_id`` and per-response ``usage``;
* ``event_msg``/``item_completed`` rows carry ``thread_id``, ``started_at_ms``/``completed_at_ms``
  and an item: ``CommandExecution`` (argv, status, exit code), ``FileChange`` (paths), the lead's
  ``SubAgentActivity`` (``kind: started|completed``, ``agent_thread_id``), ``CollabAgentToolCall``
  (``tool: wait``) and ``Extension`` (``clock.sleep``);
* ``event_msg`` ``task_started``/``task_complete`` bound each turn.

Commands run by a thread see ``CODEX_THREAD_ID`` (the thread's own id). Which thread wrote each
self-check request, and which simulations each request caused, is
:mod:`merlin_experiments.phase1.telemetry.agent_requests`.

The analysis is READ-ONLY: it never writes under a workspace or a Codex home and feeds no grade,
gate or integrity verdict. Its only writes are the operator-side live record (:func:`write_live`,
refused inside the run or workspace) and, at finalization, the sub-agent fields of the run's own
cost record (:func:`annotate_cost_record`). A missing or partially written file is reported
unavailable, never as zero, and the record says how each attribution was made.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from merlin_experiments.phase1.telemetry import agent_commands as AC
from merlin_experiments.phase1.telemetry import agent_requests as AR
from merlin_experiments.phase1.telemetry import timeline

SCHEMA = "merlin.phase1_author_agents.v1"
WINDOW_S = 1800
#: An open turn with no own record for this long is reported ``open_stale``, not ``active``. A thread
#: blocked in one long command writes nothing until it completes, so stale is not proof of death.
STALE_S = 900.0


# --- small helpers ------------------------------------------------------------------------------


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).isoformat()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _json_lines(path: Path) -> list[dict]:
    """Complete JSON rows; a live trailing partial row is not yet a record."""
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    rows = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _ms(value) -> float | None:
    return value / 1000.0 if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _union(intervals) -> float:
    return timeline._union_seconds([(a, b) for a, b in intervals if b > a])


def _environment(run_dir: Path) -> dict:
    import yaml

    try:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        document = yaml.load((run_dir / "environment.yaml").read_text(encoding="utf-8"), Loader=loader)
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return document if isinstance(document, dict) else {}


# --- discovery ----------------------------------------------------------------------------------


def _round_of(name: str) -> int | None:
    head = name.split(".", 1)[0]
    digits = head.removeprefix("round_")
    return int(digits) if head.startswith("round_") and digits.isdigit() else None


def lead_thread_ids(run_dir: Path) -> dict[str, int]:
    """Lead thread ids announced by each round's provider stream: ``{thread_id: round}``."""
    leads: dict[str, int] = {}
    rounds = Path(run_dir) / "rounds"
    for path in sorted(rounds.glob("round_*.codex_events.raw.jsonl")):
        number = _round_of(path.name)
        for row in _json_lines(path):
            if row.get("type") == "thread.started" and isinstance(row.get("thread_id"), str) and number is not None:
                leads.setdefault(row["thread_id"], number)
    return leads


def session_roots(run_dir: Path, codex_homes=()) -> list[tuple[int | None, Path]]:
    """Every sessions directory that may hold this run's rollouts, with the round it belongs to."""
    run_dir = Path(run_dir)
    env = _environment(run_dir)
    found: list[tuple[int | None, Path]] = []
    declared = {
        (env.get("codex_runtime_policy") or {}).get("home_root"),
        ((env.get("run_config") or {}).get("codex_runtime") or {}).get("home_root"),
    }
    for root in sorted(v for v in declared if isinstance(v, str) and Path(v).is_absolute()):
        for home in sorted(Path(root).glob(f"{run_dir.name}_r*")):
            suffix = home.name[len(run_dir.name) + 2 :]
            if suffix.isdigit():
                found.append((int(suffix), home / "sessions"))
    rounds = run_dir / "rounds"
    for summary in sorted(rounds.glob("round_*.codex_summary.json")):
        home = ((_read_json(summary) or {}).get("codex_home") or {}).get("codex_home")
        if isinstance(home, str) and Path(home).is_absolute():
            found.append((_round_of(summary.name), Path(home) / "sessions"))
    for snapshot in sorted(rounds.glob("round_*.codex_rollout_snapshot")):
        found.append((_round_of(snapshot.name), snapshot))
    for home in codex_homes:
        home = Path(home)
        found.append((None, home / "sessions" if (home / "sessions").is_dir() else home))
    unique, seen = [], set()
    for number, root in found:
        key = str(root)
        if key not in seen and root.is_dir():
            seen.add(key)
            unique.append((number, root))
    return unique


# --- one rollout --------------------------------------------------------------------------------


def read_rollout(path: Path) -> dict | None:
    """The thread's OWN records from one rollout file (a fork's copied parent history excluded)."""
    rows = _json_lines(path)
    if not rows or rows[0].get("type") != "session_meta" or not isinstance(rows[0].get("payload"), dict):
        return None
    meta = rows[0]["payload"]
    thread = meta.get("id")
    if not isinstance(thread, str) or not thread:
        return None
    start = meta.get("subagent_history_start_ordinal")
    start = start if isinstance(start, int) and not isinstance(start, bool) else None
    own = []
    for row in rows[1:]:
        if row.get("type") == "session_meta":
            continue
        ordinal = row.get("ordinal")
        if start is not None and (not isinstance(ordinal, int) or ordinal < start):
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if isinstance(payload.get("thread_id"), str) and payload["thread_id"] != thread:
            continue
        own.append(row)
    source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    spawn = (source.get("subagent") or {}).get("thread_spawn") if isinstance(source.get("subagent"), dict) else None
    spawn = spawn if isinstance(spawn, dict) else {}
    parent = meta.get("parent_thread_id") or spawn.get("parent_thread_id")
    return {
        "thread_id": thread,
        "parent_thread_id": parent if isinstance(parent, str) and parent else None,
        "agent_path": meta.get("agent_path") or spawn.get("agent_path") or (None if parent else "/root"),
        "agent_nickname": meta.get("agent_nickname") or spawn.get("agent_nickname"),
        "agent_role": spawn.get("agent_role"),
        "depth": spawn.get("depth") if isinstance(spawn.get("depth"), int) else (0 if not parent else None),
        "cwd": meta.get("cwd"),
        "source": "exec" if meta.get("source") == "exec" else ("subagent" if parent else str(meta.get("source"))),
        "meta_timestamp": meta.get("timestamp"),
        "path": path,
        "rows": own,
    }


def _thread_metrics(rollout: dict, workspace: str | None) -> dict:
    rows = rollout["rows"]
    stamps = [t for t in (timeline._stamp(row.get("timestamp")) for row in rows) if t is not None]
    first = min(stamps) if stamps else timeline._stamp(rollout.get("meta_timestamp"))
    last = max(stamps) if stamps else first
    turns: dict[str, list] = {}
    order: list[str] = []
    commands, edits, waits, spawned, finished_children = [], [], [], [], set()
    usage: dict[str, dict] = {}
    fallback_usage = None
    for row in rows:
        kind = row.get("type")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        at = timeline._stamp(row.get("timestamp"))
        ptype = payload.get("type")
        if kind == "event_msg" and ptype in ("task_started", "task_complete"):
            turn = str(payload.get("turn_id") or f"anon{len(order)}")
            if turn not in turns:
                turns[turn] = [None, None]
                order.append(turn)
            turns[turn][0 if ptype == "task_started" else 1] = at
        elif kind == "token_usage_record":
            values = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
            if values is not None:
                key = str(payload.get("response_id") or f"row{row.get('ordinal')}")
                usage.setdefault(key, values)
        elif kind == "event_msg" and ptype == "token_count":
            info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            if isinstance(info.get("total_token_usage"), dict):
                fallback_usage = info["total_token_usage"]
        elif kind == "event_msg" and ptype == "item_completed" and isinstance(payload.get("item"), dict):
            item = payload["item"]
            begin, end = _ms(payload.get("started_at_ms")), _ms(payload.get("completed_at_ms"))
            begin = begin if begin is not None else at
            end = end if end is not None else at
            itype = item.get("type")
            if itype == "CommandExecution":
                commands.append({"start": begin, "end": end, "item": item})
            elif itype == "FileChange":
                edits.append(item)
            elif (itype == "CollabAgentToolCall" and item.get("tool") == "wait") or (
                itype == "Extension" and item.get("kind") == "clock.sleep"
            ):
                if begin is not None and end is not None:
                    waits.append((begin, end))
            elif itype == "SubAgentActivity":
                child = item.get("agent_thread_id")
                if item.get("kind") == "started" and isinstance(child, str):
                    spawned.append({"thread_id": child, "agent_path": item.get("agent_path"), "at": _iso(begin)})
                elif item.get("kind") == "completed" and isinstance(child, str):
                    finished_children.add(child)
    intervals, open_turn = [], False
    for turn in order:
        begin, end = turns[turn]
        begin = begin if begin is not None else first
        if end is None:
            open_turn = True
            end = last  # observed through the last own record; never extrapolated to "now"
        if begin is not None and end is not None and end >= begin:
            intervals.append((begin, end))
    turn_wall = _union(intervals)
    wait_s = _union([(max(a, lo), min(b, hi)) for a, b in waits for lo, hi in intervals if min(b, hi) > max(a, lo)])
    tokens = _tokens(list(usage.values()), fallback_usage)
    edited = []
    for item in edits:
        changes = item.get("changes")
        names = (
            list(changes)
            if isinstance(changes, dict)
            else [c.get("path") for c in changes or [] if isinstance(c, dict)]
        )
        for name in names:
            if isinstance(name, str):
                rel = name
                if workspace and name.startswith(workspace.rstrip("/") + "/"):
                    rel = name[len(workspace.rstrip("/")) + 1 :]
                edited.append(rel)
    failed = sum(
        1
        for c in commands
        if c["item"].get("status") == "failed"
        or (isinstance(c["item"].get("exit_code"), int) and c["item"]["exit_code"])
    )
    return {
        "first": first,
        "last": last,
        "intervals": intervals,
        "turns": len(order),
        "open_turn": open_turn,
        "turn_wall_s": turn_wall,
        "wait_s": wait_s,
        "active_s": max(turn_wall - wait_s, 0.0),
        "tokens": tokens,
        "commands": commands,
        "n_failed_commands": failed,
        "command_union_s": _union(
            [(c["start"], c["end"]) for c in commands if c["start"] is not None and c["end"] is not None]
        ),
        "edit_items": len(edits),
        "edited_paths": sorted(set(edited)),
        "spawned": spawned,
        "finished_children": finished_children,
    }


def _tokens(responses: list[dict], fallback: dict | None) -> dict:
    """Provider usage in the run's bucket names; cached/cache-write are subsets of input."""

    def value(row, key):
        v = row.get(key, 0)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0

    if responses:
        rows, source = responses, "token_usage_record"
    elif isinstance(fallback, dict):
        rows, source = [fallback], "token_count_total"
    else:
        return {"available": False, "source": None}
    total_in = sum(value(r, "input_tokens") for r in rows)
    cached = sum(value(r, "cached_input_tokens") for r in rows)
    write = sum(value(r, "cache_write_input_tokens") for r in rows)
    out = sum(value(r, "output_tokens") for r in rows)
    reasoning = sum(value(r, "reasoning_output_tokens") for r in rows)
    return {
        "available": True,
        "source": source,
        "responses": len(responses),
        "input_total": total_in,
        "fresh_input": max(total_in - cached - write, 0),
        "cache_read": cached,
        "cache_write": write,
        "output": out,
        "reasoning": reasoning,
        "total": total_in + out,
    }


# --- threads of one run -------------------------------------------------------------------------


def threads(run_dir: Path, *, codex_homes=()) -> dict:
    """The run's lead threads and all their descendant helpers, one metrics record each."""
    run_dir = Path(run_dir)
    env = _environment(run_dir)
    workspace = env.get("workspace_path") if isinstance(env.get("workspace_path"), str) else None
    by_thread: dict[str, tuple[int | None, dict]] = {}
    files = 0
    for number, root in session_roots(run_dir, codex_homes):
        for path in sorted(root.rglob("*.jsonl")):
            files += 1
            rollout = read_rollout(path)
            if rollout is None:
                continue
            prior = by_thread.get(rollout["thread_id"])
            # A live home and its retained snapshot hold the same thread; keep the longer record.
            if prior is None or len(rollout["rows"]) > len(prior[1]["rows"]):
                by_thread[rollout["thread_id"]] = (number, rollout)
    leads = lead_thread_ids(run_dir)
    if not leads:
        leads = {tid: number for tid, (number, r) in by_thread.items() if r["parent_thread_id"] is None}
    included: dict[str, tuple[int | None, dict]] = {}
    for tid, number in leads.items():
        if tid in by_thread:
            rollout = by_thread[tid][1]
            if workspace is None or rollout.get("cwd") in (None, workspace):
                included[tid] = (number if number is not None else by_thread[tid][0], rollout)
    changed = True
    while changed:
        changed = False
        for tid, (number, rollout) in by_thread.items():
            parent = rollout["parent_thread_id"]
            if tid not in included and parent in included:
                included[tid] = (included[parent][0] if number is None else number, rollout)
                changed = True
    records = {}
    for tid, (number, rollout) in included.items():
        records[tid] = {"round": number, "rollout": rollout, "metrics": _thread_metrics(rollout, workspace)}
    announced = {child["thread_id"] for rec in records.values() for child in rec["metrics"]["spawned"]}
    return {
        "records": records,
        "leads": leads,
        "workspace": workspace,
        "files_scanned": files,
        "missing_leads": sorted(set(leads) - set(records)),
        "missing_subagents": sorted(announced - set(records)),
    }


# --- composition --------------------------------------------------------------------------------


def _concurrency(intervals_by_thread: dict, window_s: int) -> dict:
    edges = []
    for spans in intervals_by_thread.values():
        for a, b in spans:
            edges += [(a, 1), (b, -1)]
    if not edges:
        return {"max_active": 0, "windows": []}
    edges.sort(key=lambda e: (e[0], e[1]))
    level, peak, points = 0, 0, []
    for t, step in edges:
        level += step
        peak = max(peak, level)
        points.append((t, level))
    start, end = edges[0][0], edges[-1][0]
    windows = []
    lo = math.floor(start / window_s) * window_s
    while lo < end:
        hi = lo + window_s
        a, b = max(lo, start), min(hi, end)
        weighted, w_peak, level = 0.0, 0, 0
        prev = a
        for t, lvl in points:
            if t <= a:
                level = lvl
                continue
            if t > b:
                break
            weighted += level * (t - prev)
            w_peak = max(w_peak, level)
            prev, level = t, lvl
        weighted += level * max(b - prev, 0.0)
        w_peak = max(w_peak, level)
        seen = sorted(tid for tid, spans in intervals_by_thread.items() if any(x < b and y > a for x, y in spans))
        if b > a:
            windows.append(
                {
                    "start": _iso(a),
                    "end": _iso(b),
                    "active_max": w_peak,
                    "active_mean": round(weighted / (b - a), 3),
                    "threads": len(seen),
                }
            )
        lo = hi
    return {"max_active": peak, "windows": windows}


def _sum_tokens(rows: list[dict]) -> dict:
    keys = ("input_total", "fresh_input", "cache_read", "cache_write", "output", "reasoning", "total", "responses")
    present = [r for r in rows if r.get("available")]
    return (
        {"available": bool(present), **{k: sum(r.get(k, 0) for r in present) for k in keys}}
        if present
        else {"available": False}
    )


def _add(target: dict, source: dict) -> None:
    for engine, n in (source or {}).items():
        target[engine] = target.get(engine, 0) + n


def account(run_dir: Path, *, codex_homes=(), now: float | None = None, window_s: int = WINDOW_S) -> dict:
    """Per-thread and total author accounting for one run (live or finished)."""
    run_dir = Path(run_dir)
    now = datetime.now(UTC).timestamp() if now is None else now
    found = threads(run_dir, codex_homes=codex_homes)
    records = found["records"]
    if not records:
        return {
            "schema": SCHEMA,
            "available": False,
            "reason": "no Codex rollout for this run's lead thread(s)",
            "files_scanned": found["files_scanned"],
            "missing_leads": found["missing_leads"],
        }
    joined = AR.collect(run_dir, found["workspace"], records)
    requests, promotions, simjobs, calls = (joined[k] for k in ("requests", "promotions", "simjobs", "calls"))
    owner_of = {r["request_id"]: r["thread_id"] for r in requests}

    finished_children = {c for rec in records.values() for c in rec["metrics"]["finished_children"]}
    total_active = sum(rec["metrics"]["active_s"] for rec in records.values()) or 0.0
    lead_ids = [tid for tid in records if records[tid]["rollout"]["parent_thread_id"] is None]
    starts = [rec["metrics"]["first"] for rec in records.values() if rec["metrics"]["first"] is not None]
    ends = [rec["metrics"]["last"] for rec in records.values() if rec["metrics"]["last"] is not None]
    run_wall = (max(ends) - min(starts)) if starts and ends else None
    rows = []
    for tid, rec in sorted(records.items(), key=lambda kv: (kv[1]["metrics"]["first"] or 0, kv[0])):
        m, r = rec["metrics"], rec["rollout"]
        mine = [q for q in requests if q["thread_id"] == tid]
        sims: dict[str, int] = {}
        for q in mine:
            _add(sims, (q.get("simulations") or {}).get("by_engine"))
        promoted = [p for p in promotions if p["request_id"] and owner_of.get(p["request_id"]) == tid]
        promo_sims: dict[str, int] = {}
        for p in promoted:
            _add(promo_sims, {str(p["sim"]): 1} if p["sim"] else {})
        my_jobs = [j for j in simjobs if j["thread_id"] == tid]
        job_sims: dict[str, int] = {}
        for j in my_jobs:
            _add(job_sims, {str(j["sim"]): 1} if j["sim"] else {})
        selfcheck_calls = [c for c in calls if c["thread_id"] == tid and c["client"] == AC.SELFCHECK]
        if not m["open_turn"] or tid in finished_children:
            state = "finished"
        elif m["last"] is not None and now - m["last"] <= STALE_S:
            state = "active"
        else:
            state = "open_stale"  # a turn never closed: still running a long command, or interrupted
        rows.append(
            {
                "thread_id": tid,
                "parent_thread_id": r["parent_thread_id"],
                "role": "lead" if r["parent_thread_id"] is None else "subagent",
                "agent_path": r["agent_path"],
                "agent_nickname": r["agent_nickname"],
                "agent_role": r["agent_role"],
                "depth": r["depth"],
                "round": rec["round"],
                "rollout": str(r["path"]),
                "state": state,
                "started_at": _iso(m["first"]),
                "last_event_at": _iso(m["last"]),
                "turns": m["turns"],
                "turn_wall_s": round(m["turn_wall_s"], 3),
                "wait_s": round(m["wait_s"], 3),
                "active_s": round(m["active_s"], 3),
                "share_of_agent_activity": round(m["active_s"] / total_active, 4) if total_active else None,
                "share_of_run_wall": round(m["active_s"] / run_wall, 4) if run_wall else None,
                "tokens": m["tokens"],
                "commands": {
                    "n": len(m["commands"]),
                    "failed": m["n_failed_commands"],
                    "union_s": round(m["command_union_s"], 3),
                    "selfcheck_invocations": len(selfcheck_calls),
                    "selfcheck_attaches": sum(1 for c in selfcheck_calls if c["mode"] == "attach"),
                },
                "edits": {
                    "items": m["edit_items"],
                    "paths": len(m["edited_paths"]),
                    "first_paths": m["edited_paths"][:20],
                },
                "spawned": [child["thread_id"] for child in m["spawned"]],
                "selfcheck_requests": [
                    {k: v for k, v in q.items() if not k.startswith("_") and k not in ("thread_id", "caller_thread_id")}
                    for q in mine
                ],
                "promotions": [p["job_id"] for p in promoted],
                "simjobs": [j["job_id"] for j in my_jobs],
                "simulations": {
                    "selfcheck_by_engine": dict(sorted(sims.items())),
                    "promotion_by_engine": dict(sorted(promo_sims.items())),
                    "simjob_by_engine": dict(sorted(job_sims.items())),
                },
            }
        )
    lead_tokens = [row["tokens"] for row in rows if row["role"] == "lead"]
    sub_tokens = [row["tokens"] for row in rows if row["role"] == "subagent"]
    all_sims: dict[str, int] = {}
    for row in rows:
        for part in row["simulations"].values():
            _add(all_sims, part)
    unattributed = [q for q in requests if q["thread_id"] is None]
    complete = not found["missing_leads"] and not found["missing_subagents"]
    report = {
        "schema": SCHEMA,
        "available": True,
        "complete": complete,
        "basis": (
            "per-thread Codex rollouts (lead + every descendant helper); requests by forwarded thread id, else by "
            "the command whose interval holds requested_at with matching --capsules; simulations by retained raw "
            "results (capsule set + grade window), else redacted response statuses; a promotion by the passing "
            "verdict published just before it"
        ),
        "files_scanned": found["files_scanned"],
        "missing_leads": found["missing_leads"],
        "missing_subagents": found["missing_subagents"],
        "run_wall_s": round(run_wall, 3) if run_wall else None,
        "totals": {
            "threads": len(rows),
            "leads": len(lead_ids),
            "subagents": len(rows) - len(lead_ids),
            "active_now": sum(1 for row in rows if row["state"] == "active"),
            "active_s": round(total_active, 3),
            "tokens_all": _sum_tokens([row["tokens"] for row in rows]),
            "tokens_lead": _sum_tokens(lead_tokens),
            "tokens_subagents": _sum_tokens(sub_tokens),
            "commands_all": sum(row["commands"]["n"] for row in rows),
            "commands_subagents": sum(row["commands"]["n"] for row in rows if row["role"] == "subagent"),
            "edit_items_all": sum(row["edits"]["items"] for row in rows),
            "edit_items_subagents": sum(row["edits"]["items"] for row in rows if row["role"] == "subagent"),
            "selfcheck_requests": len(requests),
            "selfcheck_requests_attributed": len(requests) - len(unattributed),
            "selfcheck_requests_unattributed": len(unattributed),
            "promotions": len(promotions),
            "simjobs": len(simjobs),
            "simulations_by_engine": dict(sorted(all_sims.items())),
        },
        "concurrency": _concurrency(
            {row["thread_id"]: records[row["thread_id"]]["metrics"]["intervals"] for row in rows}, window_s
        ),
        "agents": rows,
        "unattributed_requests": [{k: v for k, v in q.items() if not k.startswith("_")} for q in unattributed],
        "unattributed_promotions": [p["job_id"] for p in promotions if p["request_id"] is None],
        "unattributed_simjobs": [j["job_id"] for j in simjobs if j["thread_id"] is None],
        "note": (
            "active_s = turn time minus explicit waits (wait_agent, clock sleep); shares divide by the sum over "
            "threads and by the run's observed span. A running command appears only once it completes."
        ),
    }
    from . import token_budget

    policy = (_environment(run_dir).get("run_config") or {}).get("author_token_budget")
    report["token_budget"] = token_budget.assess(report, policy)
    return report


def _k(n) -> str:
    return "n/a" if not isinstance(n, (int, float)) else (f"{n / 1000:.0f}k" if n >= 1000 else str(n))


def heartbeat_line(report: dict) -> str:
    """One ``[agents]`` operator-log line; field syntax matches the convergence report's reader."""
    if not report.get("available"):
        return f"[agents] available=false reason={json.dumps(report.get('reason'))}"
    totals = report["totals"]
    helpers = totals["tokens_subagents"].get("total", 0) if totals["subagents"] else 0
    return (
        f"[agents] active={totals['active_now']} total={totals['threads']} subagents={totals['subagents']} "
        f"max_concurrent={report['concurrency']['max_active']} "
        f"tokens={_k(totals['tokens_all'].get('total'))} subagent_tokens={_k(helpers)} "
        f"commands={totals['commands_all']} selfchecks={totals['selfcheck_requests']} "
        f"unattributed={totals['selfcheck_requests_unattributed']} "
        f"sims={json.dumps(totals['simulations_by_engine'], sort_keys=True, separators=(',', ':'))} "
        f"complete={str(report['complete']).lower()} "
        f"token_budget_status={report.get('token_budget', {}).get('status', 'not_selected')} "
        f"token_target={report.get('token_budget', {}).get('policy', {}).get('target', 'unselected')}"
    )


def _inside(path: Path, parent: Path | None) -> bool:
    if parent is None:
        return False
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def write_live(run_dir: Path, out: Path, *, codex_homes=(), now: float | None = None) -> str:
    """Refresh ``out`` (an operator-side JSON) atomically and return the heartbeat line."""
    run_dir, out = Path(run_dir), Path(out)
    workspace = _environment(run_dir).get("workspace_path")
    if _inside(out, run_dir) or (isinstance(workspace, str) and _inside(out, Path(workspace))):
        raise ValueError("the live agent record is operator-side; refusing to write inside the run or workspace")
    report = account(run_dir, codex_homes=codex_homes, now=now)
    report["observed_at"] = _iso(datetime.now(UTC).timestamp() if now is None else now)
    out.parent.mkdir(parents=True, exist_ok=True)
    staged = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    staged.chmod(0o600)
    staged.replace(out)
    return heartbeat_line(report)


def cost_fields(report: dict) -> dict:
    """The cost record's sub-agent fields; ``tool_calls`` there stays the lead stream's own count."""
    if not report.get("available"):
        return {
            "subagent_tool_calls_tracked": False,
            "author_agents": {"schema": SCHEMA, "available": False, "reason": report.get("reason")},
        }
    totals = report["totals"]
    return {
        "subagent_tool_calls_tracked": bool(report["complete"]),
        "subagent_tool_calls": totals["commands_subagents"] + totals["edit_items_subagents"],
        "tool_calls_all_agents": totals["commands_all"] + totals["edit_items_all"],
        "tokens_total_all_agents": totals["tokens_all"].get("total") if totals["tokens_all"].get("available") else None,
        "author_agents": {
            "schema": SCHEMA,
            "complete": report["complete"],
            "missing_subagents": report["missing_subagents"],
            "threads": totals["threads"],
            "subagents": totals["subagents"],
            "max_concurrent_active": report["concurrency"]["max_active"],
            "tokens_lead": totals["tokens_lead"],
            "tokens_subagents": totals["tokens_subagents"],
            "selfcheck_requests": totals["selfcheck_requests"],
            "selfcheck_requests_unattributed": totals["selfcheck_requests_unattributed"],
            "simulations_by_engine": totals["simulations_by_engine"],
            "per_agent": [
                {
                    "thread_id": row["thread_id"],
                    "parent_thread_id": row["parent_thread_id"],
                    "agent_path": row["agent_path"],
                    "active_s": row["active_s"],
                    "tokens_total": row["tokens"].get("total"),
                    "commands": row["commands"]["n"],
                    "edit_items": row["edits"]["items"],
                    "selfcheck_requests": len(row["selfcheck_requests"]),
                }
                for row in report["agents"]
            ],
            "tool_calls_semantics": "commands + file-change items from each thread's rollout",
        },
    }


def annotate_cost_record(run_dir: Path, *, report: dict | None = None) -> dict | None:
    """Merge the sub-agent fields into ``cost_time_toolcalls.yaml``; accounting only, never a gate."""
    import yaml

    path = Path(run_dir) / "cost_time_toolcalls.yaml"
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        return None
    fields = cost_fields(account(run_dir) if report is None else report)
    document.update(fields)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return fields


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--codex-home", action="append", default=[], type=Path, help="extra Codex home (repeatable)")
    ap.add_argument("--out", type=Path, help="operator-side live JSON (never inside the run or workspace)")
    ap.add_argument("--json", action="store_true", help="print the full JSON report")
    args = ap.parse_args(argv)
    if not args.run_dir.is_dir():
        ap.error(f"--run-dir {args.run_dir} is not a directory")
    if args.out:
        try:
            print(write_live(args.run_dir, args.out, codex_homes=args.codex_home))
        except ValueError as exc:
            ap.error(str(exc))
        return 0
    report = account(args.run_dir, codex_homes=args.codex_home)
    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(heartbeat_line(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
