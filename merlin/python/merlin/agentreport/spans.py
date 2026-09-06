"""Tool-call spans for a run, from whichever stream can still supply them.

THE PROBLEM THIS SOLVES. ``merlin.agent_trace.timeline`` pairs a ``tool_use`` block to its
``tool_result`` by id, which is the right join when the id is there. On every run written before the
harness fix on 2026-09-04 it is NOT there: the merged transcript carries ``tool_use`` blocks with no
``id`` at all, against ``tool_result`` blocks that do carry ``tool_use_id``. The join key exists on
one side only, so the timeline comes back with a wall clock and ZERO spans -- which plots as a blank
activity chart and reads as "this agent used no tools". Measured over the two main roots: radiance
had 49 such runs and 0 usable ones.

The same runs keep the driver's own event stream, ``rounds/round_NN.codex_events.timestamped.jsonl``,
where each tool call appears as an ``item.started`` / ``item.completed`` pair sharing an ``item.id``.
That join key is intact. So this module tries the transcript first and falls back to the raw stream,
and it RECORDS WHICH ONE IT USED, because the two do not have the same fidelity.

WHERE THE RAW STREAM IS WEAKER, AND HOW MUCH. ``arrived_at`` marks when the harness READ a line, so a
call whose start and completion are flushed together gets a duration near zero. Measured: 56-65% of
pairs come back under 10 ms, while p90 is 27-43 s and the tail runs to 34 minutes. The long
durations are real -- the harness genuinely waited -- and the short ones are an artifact of the read.

That split is survivable because the quantities we care about are carried by the long calls: tool
WAIT is where the simulators and builds live, and THINKING is the gap between a completion and the
next start, which the flush does not touch. What it forbids is any claim that rests on a short call's
duration. So :func:`concurrency` is computed twice -- over all spans and over only those long enough
to be trusted -- and REFUSES when the two disagree, because a concurrency reading that changes when
you drop the unreliable spans was measuring the flush. On the two runs checked the two agree to
within 0.3% (2206 s vs 2201 s; 1239 s vs 1236 s), so the overlap those runs show is real.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from merlin.agentreport.availability import Availability, derived, measured, unavailable

#: Item kinds in the driver's raw stream that represent a tool CALL (rather than a message or a
#: reasoning step). Kept as driver vocabulary, not target facts.
TOOL_ITEM_KINDS = frozenset({"command_execution", "file_change", "mcp_tool_call", "web_search"})

ITEM_STARTED = "item.started"
ITEM_COMPLETED = "item.completed"

SOURCE_TRANSCRIPT = "transcript_tool_use_ids"
SOURCE_RAW_ITEMS = "codex_item_events"

#: A span at or under this many seconds cannot be distinguished from a start/finish flushed together
#: in one read. It is a property of how the stream is READ, not of any target, so it is a constant
#: here rather than a derived fact -- but nothing depends on its exact value: the concurrency guard
#: re-runs the computation without these spans and refuses if the answer moves.
FLUSH_FLOOR_S = 0.010

#: How far the two concurrency computations may differ before the reading is judged flush-contaminated.
_CONCURRENCY_TOLERANCE = 0.05


@dataclass
class Span:
    start_s: float
    end_s: float
    kind: str = ""
    detail: str = ""

    @property
    def duration_s(self) -> float:
        return max(self.end_s - self.start_s, 0.0)


@dataclass
class SpanSet:
    """Every tool span for one run, on one clock, plus how it was obtained."""

    spans: list[Span] = field(default_factory=list)
    source: str = ""
    wall_s: float = 0.0
    n_pairs: int = 0
    n_unterminated: int = 0
    availability: Availability = field(default_factory=Availability)

    @property
    def ok(self) -> bool:
        return bool(self.spans) and self.wall_s > 0

    @property
    def flush_collapsed_fraction(self) -> float:
        """Share of spans too short to have a trustworthy duration. High is normal, not an error."""
        if not self.spans:
            return 0.0
        return sum(1 for s in self.spans if s.duration_s <= FLUSH_FLOOR_S) / len(self.spans)

    def trusted(self) -> list[Span]:
        return [s for s in self.spans if s.duration_s > FLUSH_FLOOR_S]


def _stamp(value) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _event_time(evt: dict) -> float | None:
    """When this event was read, whatever the driver named the field."""
    for key in ("arrived_at", "timestamp", "started_at"):
        t = _stamp(evt.get(key))
        if t is not None:
            return t
    return None


def _lines(path: Path) -> Iterable[dict]:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:  # a malformed line must not lose the rest of the run
            continue
        if isinstance(obj, dict):
            yield obj


def _from_transcript(paths: Sequence[Path]) -> tuple[list[Span], int, int, int]:
    """Spans by ``tool_use.id`` -> ``tool_result.tool_use_id``. Returns also how many ``tool_use``
    blocks carried NO id, which is the diagnostic that sends the caller to the raw stream."""
    spans: list[Span] = []
    idless = 0
    unterminated = 0
    offset = 0.0
    for path in paths:
        open_calls: dict[str, tuple[float, str]] = {}
        t0: float | None = None
        last = 0.0
        for evt in _lines(path):
            if evt.get("type") not in ("assistant", "user"):
                continue
            t = _event_time(evt)
            if t is None:
                continue
            if t0 is None:
                t0 = t
            rel = t - t0 + offset
            last = max(last, rel)
            content = (evt.get("message") or {}).get("content")
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    cid = str(block.get("id") or "")
                    if not cid:
                        idless += 1
                        continue
                    open_calls[cid] = (rel, str(block.get("name") or ""))
                elif block.get("type") == "tool_result":
                    cid = str(block.get("tool_use_id") or "")
                    started = open_calls.pop(cid, None)
                    if started is not None:
                        spans.append(Span(started[0], rel, started[1]))
        unterminated += len(open_calls)
        offset = last
    return spans, idless, unterminated, len(spans)


def _from_raw_items(paths: Sequence[Path]) -> tuple[list[Span], int]:
    """Spans by ``item.id`` across ``item.started`` / ``item.completed`` in the driver's own stream."""
    spans: list[Span] = []
    unterminated = 0
    offset = 0.0
    for path in paths:
        open_items: dict[str, tuple[float, str, str]] = {}
        t0: float | None = None
        last = 0.0
        for evt in _lines(path):
            t = _event_time(evt)
            if t is None:
                continue
            if t0 is None:
                t0 = t
            rel = t - t0 + offset
            last = max(last, rel)
            inner = evt.get("event") if isinstance(evt.get("event"), dict) else evt
            item = inner.get("item")
            if not isinstance(item, dict) or item.get("type") not in TOOL_ITEM_KINDS:
                continue
            iid = str(item.get("id") or "")
            if not iid:
                continue
            if inner.get("type") == ITEM_STARTED:
                open_items[iid] = (rel, str(item.get("type") or ""), str(item.get("command") or "")[:160])
            elif inner.get("type") == ITEM_COMPLETED:
                started = open_items.pop(iid, None)
                if started is not None:
                    spans.append(Span(started[0], rel, started[1], started[2]))
        unterminated += len(open_items)
        offset = last
    return spans, unterminated


def round_transcripts(run_dir: Path) -> list[Path]:
    rounds = run_dir / "rounds"
    if rounds.is_dir():
        found = sorted(rounds.glob("round_*.transcript.jsonl"))
        if found:
            return found
    flat = run_dir / "transcript.jsonl"
    return [flat] if flat.is_file() else []


def raw_event_streams(run_dir: Path) -> list[Path]:
    rounds = run_dir / "rounds"
    if not rounds.is_dir():
        return []
    return sorted(rounds.glob("round_*.codex_events.timestamped.jsonl"))


def read_spans(run_dir: Path) -> SpanSet:
    """Spans for one run, transcript first, then the driver's raw stream, then a stated refusal."""
    out = SpanSet()
    transcripts = round_transcripts(run_dir)
    raws = raw_event_streams(run_dir)

    spans, idless, unterminated, n = _from_transcript(transcripts)
    if spans:
        out.spans, out.source, out.n_pairs, out.n_unterminated = spans, SOURCE_TRANSCRIPT, n, unterminated
        out.wall_s = max(s.end_s for s in spans)
        out.availability.set("spans", measured(SOURCE_TRANSCRIPT))
        return out

    raw_spans, raw_unterminated = _from_raw_items(raws)
    if raw_spans:
        out.spans, out.source = raw_spans, SOURCE_RAW_ITEMS
        out.n_pairs, out.n_unterminated = len(raw_spans), raw_unterminated
        out.wall_s = max(s.end_s for s in raw_spans)
        why = (f"the merged transcript carried {idless} tool_use block(s) with no id, so the "
               f"tool_use -> tool_result join was impossible; spans recovered from the driver's own "
               f"item.started/item.completed stream instead") if idless else (
               "no span was recoverable from the merged transcript; spans recovered from the "
               "driver's item.started/item.completed stream")
        out.availability.set("spans", derived(why, source=SOURCE_RAW_ITEMS))
        return out

    if not transcripts and not raws:
        why = f"no transcript and no driver event stream under {run_dir.name}"
    elif idless:
        why = (f"the merged transcript's {idless} tool_use block(s) carry no id and no driver event "
               f"stream is present, so no tool call can be placed on a clock")
    else:
        why = (f"{len(transcripts)} transcript(s) and {len(raws)} event stream(s) read, but no "
               f"tool call carried a usable pair of arrival stamps")
    out.availability.set("spans", unavailable(why))
    return out


def _sweep(spans: Sequence[Span]) -> tuple[float, int, float]:
    """``(overlap_seconds, max_concurrency, wall_seconds)`` over a span list."""
    live = [s for s in spans if s.duration_s > 0]
    if not live:
        return 0.0, 0, 0.0
    events = sorted([(s.start_s, 1) for s in live] + [(s.end_s, -1) for s in live])
    k = 0
    peak = 0
    overlap = 0.0
    last: float | None = None
    for t, delta in events:
        if last is not None and k >= 2:
            overlap += t - last
        k += delta
        peak = max(peak, k)
        last = t
    wall = max(s.end_s for s in live) - min(s.start_s for s in live)
    return overlap, peak, wall


@dataclass
class Concurrency:
    """How much of the run had more than one tool call in flight, and whether that survives scrutiny."""

    overlap_s: float = 0.0
    max_concurrent: int = 0
    wall_s: float = 0.0
    overlap_share: float = 0.0
    #: The same numbers recomputed with flush-suspect spans dropped. Published only if they agree.
    overlap_s_trusted: float = 0.0
    max_concurrent_trusted: int = 0
    availability: Availability = field(default_factory=Availability)


def concurrency(spanset: SpanSet) -> Concurrency:
    """Overlap over the run, with the flush-robustness check that decides whether to believe it.

    The check is the point. A start and a finish read in one flush produce a zero-length span, and a
    pile of those can manufacture apparent concurrency that no clock ever saw. Recomputing over only
    the spans long enough to have survived a real wait, and requiring the two answers to agree,
    distinguishes "these tool calls genuinely overlapped" from "the reader batched them"."""
    out = Concurrency()
    if not spanset.ok:
        out.availability.set("concurrency", unavailable(
            spanset.availability.get("spans").reason or "no spans to sweep"))
        return out

    overlap, peak, wall = _sweep(spanset.spans)
    t_overlap, t_peak, _ = _sweep(spanset.trusted())
    out.overlap_s, out.max_concurrent, out.wall_s = overlap, peak, wall
    out.overlap_s_trusted, out.max_concurrent_trusted = t_overlap, t_peak
    out.overlap_share = overlap / wall if wall > 0 else 0.0

    if overlap <= 0.0 and t_overlap <= 0.0:
        out.availability.set("concurrency", measured(spanset.source))
        return out
    drift = abs(overlap - t_overlap) / max(overlap, t_overlap, 1e-9)
    if drift > _CONCURRENCY_TOLERANCE:
        out.availability.set("concurrency", unavailable(
            f"the overlap reading depends on spans too short to trust: {overlap:.1f}s over all "
            f"spans vs {t_overlap:.1f}s over spans longer than {FLUSH_FLOOR_S*1000:.0f} ms "
            f"({drift:.0%} apart). A start and a finish read in one flush cannot be told from a "
            f"tool call that took no time, so this run's concurrency is not measurable.",
            source=spanset.source))
        return out
    # A transcript-sourced sweep is a direct measurement. A raw-stream one is DERIVED: it survived
    # the flush check, which is what makes it publishable, but it is reconstructed rather than read.
    if spanset.source == SOURCE_TRANSCRIPT:
        out.availability.set("concurrency", measured(spanset.source))
    else:
        out.availability.set("concurrency", derived(
            f"overlap is stable when flush-suspect spans are dropped ({overlap:.1f}s vs "
            f"{t_overlap:.1f}s, {drift:.1%} apart)", source=spanset.source))
    return out
