"""Turn ANY agentic run's transcript into a wall-clock activity timeline.

WHY THIS EXISTS. Every agentic experiment in this repo writes a transcript of the same shape
(``assistant`` messages carrying ``tool_use`` blocks, ``user`` messages carrying ``tool_result``
blocks), but nothing derived WHEN each thing happened. The existing trajectory figure says so in its
own docstring -- "per-message wall stamps don't exist; within a round messages are laid by weighted
time across the round's measured duration" -- so its x-axis is synthesised from round totals rather
than measured. That was true when written and is no longer: a driver that stamps each event on
arrival (``arrived_at``) gives a real axis, and this module uses it when it is there.

WHAT IT REFUSES TO DO. When a transcript carries no arrival stamps, this returns
``basis="unstamped"`` and NO spans -- it does not fall back to a synthetic layout and label it time.
A plot that silently mixes a measured axis with an invented one is worse than a plot that says which
runs it cannot chart, and this repo has repeatedly been bitten by exactly that substitution.

The classification is DERIVED, never a per-target list:
  * ``thinking``   -- the gap between a tool result and the agent's next action: model time.
  * ``reading``    -- a tool that inspects without changing (declared read-only tools; shell readers).
  * ``writing``    -- a tool that edits the submission.
  * ``bash``       -- interactive shell work.
  * ``tool_wait``  -- a call whose own duration dominates it (a simulator, a build). Split from
                      ``bash`` BY MEASURED DURATION rather than by naming known simulators, so a new
                      target's toolchain lands in the right band with no code change here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

THINKING = "thinking"
READING = "reading"
WRITING = "writing"
BASH = "bash"
TOOL_WAIT = "tool_wait"
#: Stacked bottom-to-top, matching the reference figure's reading order.
ACTIVITIES = (THINKING, READING, WRITING, BASH, TOOL_WAIT)

#: A shell call at or above this many seconds is dominated by WAITING on an external tool rather than
#: by the shell itself. Derived boundary, not a tool list: it separates an `ls` from a cycle-accurate simulation
#: without this module ever learning what a simulator is called.
WAIT_SECONDS = 20.0

#: Headroom on the tool-time-vs-wall-time invariant, for clock jitter and the odd genuinely
#: backgrounded call. Anything beyond this is a stamping artefact, not concurrency.
_BUSY_TOLERANCE = 1.05

#: Tool names that inspect without mutating, and those that author. Tool NAMES are harness vocabulary
#: (the same strings `conformance` matches on), not target facts.
_READ_TOOLS = frozenset({"read", "read_file", "open_file", "glob", "grep", "ls", "notebookread"})
_WRITE_TOOLS = frozenset({"write", "edit", "multiedit", "write_file", "apply_patch", "notebookedit"})
_SHELL_TOOLS = frozenset({"bash", "shell", "command_execution", "run_command"})
#: argv[0] of a shell command that only inspects. Used to split `reading` out of `bash`.
_READ_EXES = frozenset({"cat", "head", "tail", "less", "grep", "rg", "find", "ls", "stat", "wc",
                        "file", "diff", "jq", "sed", "awk", "tree", "du", "nl", "od", "xxd"})


@dataclass
class Span:
    """One contiguous stretch of one activity, in seconds from the session's first event."""
    start_s: float
    end_s: float
    activity: str
    tool: str = ""
    detail: str = ""

    @property
    def duration_s(self) -> float:
        return max(self.end_s - self.start_s, 0.0)


@dataclass
class Timeline:
    """Spans + token samples + the BASIS, which a caller must check before plotting a time axis."""
    basis: str                       # "wall_clock" | "unstamped"
    reason: str = ""                 # why, when basis is not wall_clock
    spans: list[Span] = field(default_factory=list)
    #: (t_seconds, cumulative_input_tokens, cumulative_output_tokens) at each usage report.
    tokens: list[tuple[float, int, int]] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def measured(self) -> bool:
        return self.basis == "wall_clock"

    def share(self, bins: int = 120) -> tuple[list[float], dict[str, list[float]]]:
        """Activity share per time bin, each bin summing to 1 (or 0 where nothing ran).

        This is the stacked area of the reference figure. Shares are computed from MEASURED span
        overlap with each bin, so a single long tool wait spreads across the bins it really covered
        instead of being counted once at its start."""
        if not self.spans or self.wall_s <= 0:
            return [], {a: [] for a in ACTIVITIES}
        width = self.wall_s / bins
        acc = {a: [0.0] * bins for a in ACTIVITIES}
        for sp in self.spans:
            if sp.activity not in acc:
                continue
            lo, hi = sp.start_s, sp.end_s
            first, last = int(lo // width), min(int(hi // width), bins - 1)
            for b in range(max(first, 0), max(last, 0) + 1):
                b_lo, b_hi = b * width, (b + 1) * width
                acc[sp.activity][b] += max(min(hi, b_hi) - max(lo, b_lo), 0.0)
        centres = [(b + 0.5) * width for b in range(bins)]
        for b in range(bins):
            total = sum(acc[a][b] for a in ACTIVITIES)
            if total > 0:
                for a in ACTIVITIES:
                    acc[a][b] /= total
        return centres, acc

    def totals(self) -> dict[str, float]:
        out = {a: 0.0 for a in ACTIVITIES}
        for sp in self.spans:
            if sp.activity in out:
                out[sp.activity] += sp.duration_s
        return out


def _stamp(obj: dict) -> float | None:
    raw = obj.get("arrived_at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _argv0(command: str) -> str:
    """argv[0] of a shell command, unwrapping a `bash -lc "..."` wrapper. Structural, no regex."""
    import shlex
    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()
    if not toks:
        return ""
    if Path(toks[0]).name in {"bash", "sh", "zsh", "dash"}:
        for i, tok in enumerate(toks[1:-1], start=1):
            if tok.startswith("-") and tok.endswith("c"):
                try:
                    inner = shlex.split(toks[i + 1])
                except ValueError:
                    inner = toks[i + 1].split()
                return Path(inner[0]).name if inner else ""
    return Path(toks[0]).name


def classify(tool: str, command: str, duration_s: float) -> str:
    """Which activity a completed tool call belongs to.

    ``duration_s`` participates deliberately: the difference between shell work and waiting on a
    simulator is how long it took, not what it was called. Naming simulators here would need editing
    for every new target -- the thing this repo's cardinal rule forbids."""
    name = (tool or "").strip().lower().split(".")[-1]
    if name in _WRITE_TOOLS:
        return WRITING
    if name in _READ_TOOLS:
        return READING
    if name in _SHELL_TOOLS or command:
        if duration_s >= WAIT_SECONDS:
            return TOOL_WAIT
        return READING if _argv0(command) in _READ_EXES else BASH
    return BASH


def _events(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = __import__("json").loads(line)
        except Exception:  # noqa: BLE001 - a malformed line must not lose the rest of the run
            continue
        if isinstance(obj, dict):
            yield obj


def timeline(transcript: Path) -> Timeline:
    """Build the wall-clock timeline for one transcript, or say why it cannot be built."""
    stamped = 0
    total = 0
    opens: dict[str, tuple[float, str, str]] = {}
    spans: list[Span] = []
    tokens: list[tuple[float, int, int]] = []
    cum_in = cum_out = 0
    t0: float | None = None
    last_end: float | None = None

    for obj in _events(transcript):
        kind = obj.get("type")
        if kind not in ("assistant", "user"):
            continue
        total += 1
        t = _stamp(obj)
        if t is None:
            continue
        stamped += 1
        if t0 is None:
            t0 = t
        rel = t - t0
        msg = obj.get("message") or {}
        content = msg.get("content")
        blocks = content if isinstance(content, list) else []
        if kind == "assistant":
            usage = msg.get("usage") or {}
            if usage:
                cum_in += int(usage.get("input_tokens") or 0)
                cum_out += int(usage.get("output_tokens") or 0)
                tokens.append((rel, cum_in, cum_out))
            for blk in blocks:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "tool_use":
                    tid = str(blk.get("id") or "")
                    cmd = str((blk.get("input") or {}).get("command") or "")
                    opens[tid] = (rel, str(blk.get("name") or ""), cmd)
                    # The agent was THINKING between the last result and issuing this call. Emitted
                    # ONCE per gap: a message carrying several tool_use blocks would otherwise repeat
                    # the same interval per block and inflate thinking past the wall clock.
                    if last_end is not None and rel > last_end:
                        spans.append(Span(last_end, rel, THINKING))
                        last_end = rel
        else:
            for blk in blocks:
                if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                    continue
                tid = str(blk.get("tool_use_id") or "")
                started = opens.pop(tid, None)
                if started is None:
                    continue
                s_rel, tool, cmd = started
                spans.append(Span(s_rel, rel, classify(tool, cmd, rel - s_rel), tool, cmd[:120]))
                last_end = rel

    # A BURST OF IDENTICAL END STAMPS MEANS THE STAMPS ARE THE READER'S, NOT THE EVENT'S.
    # `arrived_at` records when the harness READ a line. When the agent process buffers and flushes
    # many completions at once, every one of them is stamped with the flush instant, and the tool
    # durations derived from them are fiction: measured on a real run, EIGHT calls all ended
    # at 36.98 min, which made 169 min of tool time appear inside 43.6 min of wall. Those numbers
    # look plausible on a chart, which is exactly why this refuses instead of drawing them.
    tool_spans = [sp for sp in spans if sp.activity != THINKING]
    wall_est = max((sp.end_s for sp in spans), default=0.0)
    # EVERY span counts: thinking that overlaps a running tool is as impossible as two tools at once,
    # and an inflated thinking share is the easiest error to miss because it looks like a busy agent.
    busy = sum(sp.duration_s for sp in spans)
    ends: dict[float, int] = {}
    for sp in tool_spans:
        ends[round(sp.end_s, 3)] = ends.get(round(sp.end_s, 3), 0) + 1
    worst = max(ends.values(), default=0)
    # THE INVARIANT: a single-threaded agent cannot spend more tool time than wall time. When it
    # appears to, the stamps are the reader's rather than the event's, and every duration built on
    # them is fiction. Both symptoms are reported because they point at the same cause and the tie
    # count says how badly.
    if wall_est > 0 and busy > wall_est * _BUSY_TOLERANCE:
        return Timeline(basis="bursty", tokens=tokens, wall_s=wall_est, reason=(
            f"tool time ({busy / 60:.1f} min) exceeds wall time ({wall_est / 60:.1f} min), and "
            f"{worst} calls share one end stamp: the transcript's `arrived_at` marks when the harness "
            f"READ each event, not when the tool finished, so per-tool durations cannot be derived "
            f"from this stream. Fix the reader (or the agent's output buffering) before charting a "
            f"time axis from it."))

    if not stamped:
        return Timeline(basis="unstamped", reason=(
            f"no event in {transcript.name} carries an `arrived_at` stamp ({total} assistant/user "
            f"events read), so this run has no measured time axis. Charting it would mean inventing "
            f"one. The driver must stamp events on arrival for this run to be plottable."))
    spans.sort(key=lambda s: s.start_s)
    wall = max((s.end_s for s in spans), default=0.0)
    return Timeline(basis="wall_clock", spans=spans, tokens=tokens, wall_s=wall)
