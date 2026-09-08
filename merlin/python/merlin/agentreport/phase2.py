"""The performance lane's own telemetry, which is richer than the functional lane's.

A performance stage records three things the functional runs do not:

``agent/tools.jsonl``
    one row per tool call with real ``t_start_s`` / ``t_end_s`` offsets. No id join, no arrival-stamp
    reconstruction -- the driver wrote the span. Measured over 1,362 ``command_execution`` rows: four
    overlapping pairs out of 1,336 consecutive gaps, peak concurrency 5. **The performance lane is
    essentially serial**, which is a finding about how the lane is shaped rather than a defect, and
    it is the reason the parallelism story lives in the functional lane.

``control/round_NN/receipts.jsonl``
    one row per BROKER action, with ``elapsed_s`` and a return code. This is where the lane's cost
    actually goes: the compiles are seconds, the analysis action is free by construction, and the
    measurement action is tens of seconds.

``agent_workspaces/round_NN/STAGE_CONTEXT.json``
    ``broker_actions`` -- the exact tool surface this run exposed. It is DERIVED per run from the
    candidate's own manifest plus descriptor-declared probes, so it is read here rather than assumed:
    a report that hardcoded the action list would describe one campaign and mislabel the next.

``kind == "file_change"`` rows are instantaneous events, not spans (99% land under 10 ms), so they
are counted and excluded from any duration or occupancy figure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from merlin.agentreport.availability import Availability, measured, unavailable
from merlin.agentreport.spans import Span, SpanSet

SOURCE_TOOLS_JSONL = "agent_tools_jsonl"

#: Row kinds that occupy time. Anything else is an event the driver logged at a point.
SPAN_KINDS = frozenset({"command_execution", "mcp_tool_call", "web_search"})
POINT_KINDS = frozenset({"file_change"})


@dataclass
class BrokerCall:
    action: str
    elapsed_s: float
    returncode: int | None
    state: str
    index: int


@dataclass
class Phase2Facts:
    spanset: SpanSet = field(default_factory=SpanSet)
    n_point_events: int = 0
    broker_calls: list[BrokerCall] = field(default_factory=list)
    broker_actions: list[str] = field(default_factory=list)
    availability: Availability = field(default_factory=Availability)

    def action_totals(self) -> dict[str, tuple[int, float]]:
        """``action -> (calls, seconds)``. Where the brokered half of the run's time went."""
        out: dict[str, tuple[int, float]] = {}
        for call in self.broker_calls:
            n, total = out.get(call.action, (0, 0.0))
            out[call.action] = (n + 1, total + call.elapsed_s)
        return out


def _rows(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def read_tool_spans(stage_dir: Path) -> tuple[SpanSet, int]:
    """Spans straight from the driver's own ledger, plus the count of point events excluded."""
    out = SpanSet(source=SOURCE_TOOLS_JSONL)
    path = stage_dir / "agent" / "tools.jsonl"
    rows = _rows(path)
    points = 0
    for row in rows:
        kind = str(row.get("kind") or "")
        if kind in POINT_KINDS:
            points += 1
            continue
        if kind not in SPAN_KINDS:
            continue
        start, end = row.get("t_start_s"), row.get("t_end_s")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        out.spans.append(Span(float(start), float(end), kind, str(row.get("command") or "")[:160]))
    if out.spans:
        out.wall_s = max(s.end_s for s in out.spans)
        out.n_pairs = len(out.spans)
        out.availability.set("spans", measured(SOURCE_TOOLS_JSONL))
    else:
        out.source = ""
        # A GLOBAL CAMPAIGN NEVER HAS THIS FILE, and the reason is worth stating rather than
        # leaving a reader to conclude the lane made no tool calls. `agent/tools.jsonl` is written
        # by `perf_agent_stage.finalize_agent_telemetry`, which is called only from phase 1's
        # `run_stage`; the global launcher never calls it. The raw driver stream IS on disk at
        # `agent/events.NN.raw.jsonl` -- but spans are NOT reconstructed from it here, because
        # arrival stamps are when an event reached the reader, not when the tool ran, and a
        # plausible span built from the wrong clock is exactly the kind of number this package
        # exists to refuse.
        raw = sorted((stage_dir / "agent").glob("events.*.raw.jsonl")) \
            if (stage_dir / "agent").is_dir() else []
        out.availability.set("spans", unavailable(
            f"{stage_dir.name} has no agent/tools.jsonl row carrying a start and end offset"
            + (f" ({points} point event(s) were present but occupy no time)" if points else "")
            + (f"; {len(raw)} raw driver event stream(s) are present but carry arrival stamps, "
               f"not tool spans, and finalize_agent_telemetry (which writes the spans) is called "
               f"only from the phase-1 stage" if raw else "")))
    return out, points


#: Where a run keeps its broker receipts. A phase-1 stage writes ``control/``; a global phase-2
#: campaign writes ``global_control/`` with four-digit round indices. BOTH are read, because reading
#: one meant every global campaign's entire tool cost reported UNAVAILABLE while the receipts sat on
#: disk -- and an availability report that cannot find its own evidence is worse than no report.
CONTROL_DIRS = ("control", "global_control")


def read_receipts(stage_dir: Path) -> list[BrokerCall]:
    """Every brokered action this stage invoked, in order, across all its rounds."""
    calls: list[BrokerCall] = []
    directories = [stage_dir / name for name in CONTROL_DIRS]
    if not any(d.is_dir() for d in directories):
        return calls
    for receipts in sorted(r for d in directories if d.is_dir()
                           for r in d.glob("round_*/receipts.jsonl")):
        for row in _rows(receipts):
            action = str(row.get("action") or "")
            if not action:
                continue
            rc = row.get("returncode")
            calls.append(BrokerCall(
                action=action,
                elapsed_s=float(row.get("elapsed_s") or 0.0),
                returncode=int(rc) if isinstance(rc, int) else None,
                state=str(row.get("state") or ""),
                index=int(row.get("index")) if isinstance(row.get("index"), int) else -1))
    return calls


def read_broker_actions(stage_dir: Path) -> list[str]:
    """The tool surface this run declared. Derived per run, so it is read and never assumed."""
    for context in sorted((stage_dir / "agent_workspaces").glob("round_*/STAGE_CONTEXT.json")) \
            if (stage_dir / "agent_workspaces").is_dir() else []:
        try:
            doc = json.loads(context.read_text(encoding="utf-8", errors="ignore"))
        except (ValueError, OSError):
            continue
        actions = doc.get("broker_actions") if isinstance(doc, dict) else None
        if isinstance(actions, list) and actions:
            names = []
            for entry in actions:
                if isinstance(entry, str):
                    names.append(entry)
                elif isinstance(entry, dict) and entry.get("name"):
                    names.append(str(entry["name"]))
            if names:
                return sorted(set(names))
    return []


def read_phase2(stage_dir: Path) -> Phase2Facts:
    facts = Phase2Facts()
    facts.spanset, facts.n_point_events = read_tool_spans(stage_dir)
    facts.availability.set("spans", facts.spanset.availability.get("spans"))

    facts.broker_calls = read_receipts(stage_dir)
    if facts.broker_calls:
        facts.availability.set("broker_calls", measured("receipts.jsonl"))
    else:
        facts.availability.set("broker_calls", unavailable(
            f"{stage_dir.name} recorded no control/round_*/receipts.jsonl, so how the brokered half "
            f"of this run spent its time is unrecoverable"))

    facts.broker_actions = read_broker_actions(stage_dir)
    if facts.broker_actions:
        facts.availability.set("broker_actions", measured("STAGE_CONTEXT"))
    else:
        facts.availability.set("broker_actions", unavailable(
            f"{stage_dir.name} recorded no STAGE_CONTEXT.json with a broker_actions list, so the "
            f"tool surface it exposed cannot be stated. It must not be inferred from another run: "
            f"the action set is derived per run from the candidate's own manifest."))
    return facts
