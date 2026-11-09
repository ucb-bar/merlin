"""The author session's lifecycle: each turn's admission, startup, agent work and how it ended.

Two run-side records, both tailed incrementally:

* ``rounds/round_NN.transcript.jsonl`` -- the driver's lifecycle rows: ``codex_token_admission``
  (a turn admitted, its phase and token allowance; monotonic time), ``codex_operator_interrupt`` (the
  host stopped the author, with its reason), ``codex_session_turn``, ``codex_summary`` and ``result``
  (the session ended);
* ``rounds/author_startup.jsonl`` (``merlin.author_startup_stage.v1``) -- the host's preparation of each
  turn's sandbox: nested stages with ``start``/``finish`` events, wall time and both an ISO and a
  monotonic timestamp (which also converts the admission's monotonic time to epoch seconds).

A turn's *startup* runs from its first root stage start to its last root stage finish; its *agent*
time runs from there to the interrupt or session end that follows. ``phase`` is what the author
side is doing now: ``startup`` (a stage open), ``authoring`` (the agent runs), ``ended`` (the
session ended or the host interrupted it) or ``unknown``. Nothing here is inferred from processes.

Section::

    {"phase", "phase_since", "phase_detail", "author_running": bool|None,
     "turns": [{round, turn, phase, admitted_at, allowance, startup_start, startup_end, startup_s,
                stages: [{stage, start, end, wall_s, depth}], open_stage, agent_start, agent_end,
                agent_s, end_reason}],
     "session": {ended_at, reason, exit_code}|None, "notes"}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import bounded_glob, clean, epoch, mtime, num

STARTUP_SCHEMA = "merlin.author_startup_stage.v1"
MAX_TURNS = 64
MAX_STAGES = 48


class Lifecycle:
    """Incrementally parsed lifecycle rows of one run."""

    def __init__(self) -> None:
        self.offset: float | None = None  # epoch - monotonic, from the startup records
        self.admissions: list[dict[str, Any]] = []
        self.interrupts: list[dict[str, Any]] = []
        self.sessions: dict[int, dict[str, Any]] = {}  # round -> how its session ended
        self.written: dict[int, float] = {}  # round -> its transcript's modification time
        self.started_at: float | None = None
        self.stages: dict[str, dict[str, Any]] = {}  # stage_id -> stage row

    # -- transcript ----------------------------------------------------------------------------
    def transcript(self, row: dict, rnd: int) -> None:
        kind = row.get("type")
        end = self.sessions.get(rnd) or {}
        if kind == "system" and row.get("subtype") == "init":
            self.started_at = self.started_at or epoch(row.get("started_at"))
        elif kind == "codex_token_admission":
            turn = str(row.get("turn") or "")
            rnd, _, idx = turn.partition(":")
            self.admissions.append(
                {
                    "round": int(rnd) if rnd.isdigit() else row.get("round"),
                    "turn": int(idx) if idx.isdigit() else None,
                    "phase": clean(row.get("phase"), 24) or None,
                    "admitted": row.get("admitted"),
                    "reason": clean(row.get("reason"), 160) or None,
                    "monotonic": num(row.get("at_monotonic_s")),
                    "allowance": num(row.get("next_turn_allowance")),
                }
            )
            del self.admissions[:-MAX_TURNS]
        elif kind == "codex_operator_interrupt":
            self.interrupts.append(
                {
                    "at": epoch(row.get("arrived_at")),
                    "round": rnd,
                    "reason": clean(row.get("reason"), 160) or "interrupted",
                }
            )
            del self.interrupts[:-MAX_TURNS]
        elif kind == "codex_summary":
            reason = row.get("interruption_reason") if row.get("operator_interrupted") else None
            if row.get("timed_out"):
                reason = "timed out"
            errors = [e for e in row.get("unrecovered_errors") or row.get("errors") or [] if isinstance(e, str)]
            if reason is None and errors:
                reason = errors[0]
            self.sessions[rnd] = {
                **end,
                "reason": clean(reason, 160) or end.get("reason") or "session ended",
                "exit_code": row.get("exit_code"),
                # A session that never started a turn ran no agent, whatever its startup did.
                "no_turns": row.get("turns_started") == 0,
            }
        elif kind == "result":
            text = clean(row.get("result"), 160)
            reason = end.get("reason") if end.get("reason") not in (None, "session ended") else None
            self.sessions[rnd] = {**end, "reason": reason or text or "session ended"}
        elif kind == "codex_session_turn" and row.get("continuing") is False:
            self.sessions[rnd] = {
                **end,
                "at": epoch(row.get("arrived_at")),
                "reason": clean(row.get("stopped_because"), 160) or "session ended",
            }

    # -- startup stages ------------------------------------------------------------------------
    def startup(self, row: dict) -> None:
        if row.get("schema") != STARTUP_SCHEMA or not isinstance(row.get("stage_id"), str):
            return
        at, mono = epoch(row.get("at")), num(row.get("monotonic_s"))
        if at is not None and mono is not None:
            self.offset = at - mono
        stage = self.stages.setdefault(
            row["stage_id"],
            {
                "stage": clean(row.get("stage"), 48) or "?",
                "parent": row.get("parent_stage_id"),
                "round": row.get("round"),
                "turn": row.get("turn"),
                "start": None,
                "end": None,
                "wall_s": None,
            },
        )
        if row.get("event") == "start":
            stage["start"] = at
        elif row.get("event") == "finish":
            stage["end"] = at
            stage["wall_s"] = num(row.get("wall_s"))
        if len(self.stages) > MAX_TURNS * MAX_STAGES:
            for key in list(self.stages)[: len(self.stages) - MAX_TURNS * MAX_STAGES]:
                del self.stages[key]

    # -- view ----------------------------------------------------------------------------------
    def turns(self, now: float) -> list[dict[str, Any]]:
        keyed: dict[tuple, dict[str, Any]] = {}

        def turn_row(rnd, idx) -> dict[str, Any]:
            return keyed.setdefault(
                (rnd, idx),
                {
                    "round": rnd,
                    "turn": idx,
                    "phase": None,
                    "admitted_at": None,
                    "allowance": None,
                    "startup_start": None,
                    "startup_end": None,
                    "startup_s": None,
                    "stages": [],
                    "open_stage": None,
                    "agent_start": None,
                    "agent_end": None,
                    "agent_s": None,
                    "end_reason": None,
                },
            )

        for adm in self.admissions:
            if adm["admitted"] is False:
                continue
            row = turn_row(adm["round"], adm["turn"])
            row["phase"], row["allowance"] = adm["phase"], adm["allowance"]
            if adm["monotonic"] is not None and self.offset is not None:
                row["admitted_at"] = adm["monotonic"] + self.offset
        depth: dict[str, int] = {}

        def depth_of(stage_id: str) -> int:
            if stage_id not in depth:
                parent = self.stages.get(stage_id, {}).get("parent")
                depth[stage_id] = 0 if parent not in self.stages else min(8, depth_of(parent) + 1)
            return depth[stage_id]

        for stage_id, stage in self.stages.items():
            if stage["start"] is None:
                continue
            row = turn_row(stage["round"], stage["turn"])
            level = depth_of(stage_id)
            if len(row["stages"]) < MAX_STAGES:
                row["stages"].append(
                    {
                        "stage": stage["stage"],
                        "start": stage["start"],
                        "end": stage["end"],
                        "wall_s": stage["wall_s"],
                        "depth": level,
                    }
                )
            if level == 0:
                row["startup_start"] = min(filter(None, (row["startup_start"], stage["start"])))
                if stage["end"] is not None:
                    row["startup_end"] = max(filter(None, (row["startup_end"], stage["end"])))
            if stage["end"] is None:
                open_stage = row["open_stage"]
                if open_stage is None or level >= open_stage["depth"]:
                    row["open_stage"] = {"stage": stage["stage"], "since": stage["start"], "depth": level}
        turns = sorted(keyed.values(), key=lambda r: (r["round"] or 0, r["turn"] if r["turn"] is not None else -1))
        for i, row in enumerate(turns):
            ends = sorted(
                (e for e in self.interrupts if e["at"] is not None and e["round"] == row["round"]),
                key=lambda e: e["at"],
            )
            same_round = i + 1 < len(turns) and turns[i + 1]["round"] == row["round"]
            session = self.sessions.get(row["round"]) if not same_round else None
            if row["open_stage"] is not None:
                row["startup_end"] = None
            if row["startup_start"] is not None and row["startup_end"] is not None:
                row["startup_s"] = round(row["startup_end"] - row["startup_start"], 1)
            elif row["startup_start"] is not None:
                row["startup_s"] = round(now - row["startup_start"], 1)
            row["agent_start"] = row["startup_end"]
            begin = row["startup_start"] or row["admitted_at"]
            following = turns[i + 1]["startup_start"] if same_round else None
            # The interrupt that ends a turn is the first one after the turn began (and before the next).
            end = next(
                (
                    e
                    for e in ends
                    if begin is not None and e["at"] >= begin and (following is None or e["at"] <= following)
                ),
                None,
            )
            if end is not None:
                row["agent_end"], row["end_reason"] = end["at"], end["reason"]
            elif session and (session.get("at") or self.written.get(row["round"])):
                # A session end without its own time ended when its transcript was last written.
                at = session.get("at") or self.written[row["round"]]
                row["agent_end"], row["end_reason"] = at, session.get("reason")
                if session.get("no_turns"):
                    row["agent_start"], row["end_reason"] = None, f"refused: {session.get('reason')}"
            elif following is not None:
                row["agent_end"], row["end_reason"] = following, "next turn started"
            if row["agent_start"] is not None:
                if row["agent_end"] is not None and row["agent_end"] < row["agent_start"]:
                    row["agent_start"] = None  # Interrupted during startup: the agent never ran.
                else:
                    row["agent_s"] = round((row["agent_end"] or now) - row["agent_start"], 1)
            row["stages"].sort(key=lambda s: s["start"])
        return turns[-MAX_TURNS:]


def _state(ctx) -> Lifecycle:
    key = str(ctx.experiment.get("run_dir"))
    if ctx.state.get("key") != key:
        ctx.state.clear()
        ctx.state.update(key=key, lifecycle=Lifecycle())
    return ctx.state["lifecycle"]


def read(ctx) -> dict[str, Any]:
    run = Path(ctx.experiment.get("run_dir") or "")
    life = _state(ctx)
    rounds = run / "rounds"
    for path in sorted(bounded_glob(rounds, "round_*.transcript.jsonl", 64)):
        digits = path.name[len("round_") :].split(".", 1)[0]
        rnd = int(digits) if digits.isdigit() else -1
        ctx.reader.poll(path, lambda row, rnd=rnd: life.transcript(row, rnd))
        written = mtime(path)
        if written is not None:
            life.written[rnd] = written
    ctx.reader.poll(rounds / "author_startup.jsonl", life.startup)
    now = ctx.now if ctx.live else (num(ctx.experiment.get("last_record_at")) or ctx.now)
    turns = life.turns(now)
    notes: list[str] = []
    phase, since, detail, running = "unknown", None, None, None
    last = turns[-1] if turns else None
    session = life.sessions.get(last["round"]) if last is not None else None
    if last is not None:
        if last["agent_end"] is not None or (session and not ctx.live):
            phase, since = "ended", last["agent_end"]
            detail, running = last["end_reason"] or (session or {}).get("reason"), False
        elif last["open_stage"] is not None:
            open_stage = last["open_stage"]
            phase, since, running = "startup", last["startup_start"], False
            detail = f"{open_stage['stage']} for {int(now - open_stage['since'])}s"
        elif last["agent_start"] is not None:
            phase, since, running = "authoring", last["agent_start"], ctx.live
    elif session:
        phase, detail, running = "ended", session.get("reason"), False
    if not turns:
        notes.append("no author lifecycle records (transcript or author_startup.jsonl) in this run")
    if not ctx.live and phase in ("startup", "authoring"):
        phase, running = "ended", False
        detail = "the run stopped while the author " + ("was starting" if last and last["open_stage"] else "ran")
    return {
        "phase": phase,
        "phase_since": since,
        "phase_detail": clean(detail, 200) or None,
        "author_running": running,
        "turns": turns,
        "session": {"ended_at": (session or {}).get("at"), **{k: v for k, v in (session or {}).items() if k != "at"}}
        if session
        else None,
        "started_at": life.started_at,
        "notes": notes,
    }


def inspect(ctx, key: str) -> dict[str, Any]:
    section = read(ctx)
    sections = []
    for turn in section["turns"]:
        label = f"round {turn['round']} turn {turn['turn']}"
        lines = [
            f"phase {turn['phase'] or 'not recorded'}  token allowance {turn['allowance'] or 'not recorded'}",
            f"startup {turn['startup_s'] if turn['startup_s'] is not None else 'not recorded'} s"
            + (f" (open: {turn['open_stage']['stage']})" if turn["open_stage"] else ""),
            (f"agent worked {turn['agent_s']} s" if turn["agent_s"] is not None else "the agent did not run")
            + (f"; ended: {turn['end_reason']}" if turn["end_reason"] else ""),
        ]
        for stage in turn["stages"]:
            wall = stage["wall_s"]
            lines.append(
                "  " * (stage["depth"] + 1)
                + f"{stage['stage']:<28} {'running' if stage['end'] is None else f'{wall:.1f}s' if wall else '-'}"
            )
        sections.append({"title": label, "lines": lines})
    run = Path(ctx.experiment.get("run_dir") or "")
    return {
        "title": f"author lifecycle · {section['phase']}",
        "sections": sections or [{"title": "lifecycle", "lines": section["notes"]}],
        "paths": [str(run / "rounds" / "author_startup.jsonl")],
    }


__all__ = ["Lifecycle", "inspect", "read"]
