"""Cumulative token consumption over wall time -- the curve behind a token-rate plot.

WHY THIS IS NOT ALWAYS AVAILABLE, AND WHY THAT MATTERS. A rate is a derivative, so it needs several
samples. Drivers disagree about how often they report usage:

* one driver stamps ``usage`` on EVERY assistant message, giving hundreds of samples across a run;
* another reports only at ``turn.completed`` -- one sample per turn, which on a continuous run is
  **one sample for two hours of work**;
* a third reports none this reader can place on a clock.

Measured over the corpus: 101 runs carry five or more usable samples, 73 carry between one and four,
and 154 carry none. A two-point "curve" drawn through a two-hour round is not a rate; it is a
straight line whose slope is an average, and plotting it beside a dense one invites the reader to
compare shapes that mean different things. So this reports the sample count and
:func:`token_rate` REFUSES below a floor rather than drawing that line.

USAGE REPORTS ARE DELTAS, NOT RUNNING TOTALS -- and one driver emits each of them several times.
Both drivers report what a single message or turn consumed, so the cumulative curve is their running
SUM. The trap is that a streaming transcript re-emits the same assistant message as it grows: on one
run, 1,301 usage reports covered 585 distinct messages, and summing them all doubled the output
count and overshot cached input by 2.2x. Deduplicating on the message id and keeping the last report
for each lands within 0.2% of the figure the harness recorded independently
(output 12,213 vs 12,233; cached 91,093,603 vs 91,221,980, the remainder being messages the harness
attributes to no model).

Because that agreement is the only thing standing between this curve and a plausible-looking wrong
one, :func:`read_token_series` takes the independently recorded total as an argument and records
whether the two agree. A counter nobody cross-checked is how this repo has been bitten before.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from merlin.agentreport.availability import Availability, measured, unavailable

#: Below this many samples a rate curve is an average drawn as a trend. Refuse instead.
MIN_RATE_SAMPLES = 5

#: How far the reconstructed total may sit from the harness's own before the curve is disowned.
_CROSS_CHECK_TOLERANCE = 0.05


@dataclass
class TokenSample:
    t_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def billed_input(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens


@dataclass
class TokenSeries:
    """Cumulative samples on the run's own clock, plus whether they can carry a rate."""

    samples: list[TokenSample] = field(default_factory=list)
    source: str = ""
    wall_s: float = 0.0
    n_reports: int = 0            # raw usage reports read
    n_duplicates: int = 0         # reports superseded by a later one for the same message id
    availability: Availability = field(default_factory=Availability)

    @property
    def ok(self) -> bool:
        return len(self.samples) >= 2

    @property
    def can_rate(self) -> bool:
        return len(self.samples) >= MIN_RATE_SAMPLES


def _stamp(value) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _files(run_dir: Path) -> tuple[list[Path], list[Path]]:
    rounds = run_dir / "rounds"
    transcripts = sorted(rounds.glob("round_*.transcript.jsonl")) if rounds.is_dir() else []
    if not transcripts and (run_dir / "transcript.jsonl").is_file():
        transcripts = [run_dir / "transcript.jsonl"]
    raws = sorted(rounds.glob("round_*.codex_events.timestamped.jsonl")) if rounds.is_dir() else []
    return transcripts, raws


def _raw_samples(paths: Sequence[Path]) -> list[tuple[float, str, dict]]:
    """``(epoch, message_id, usage)`` from whichever shape the line carries.

    The message id is what makes deduplication possible; a line without one is kept under a unique
    synthetic key so it is never silently merged with a neighbour."""
    out: list[tuple[float, str, dict]] = []
    for path in paths:
        for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
            if '"usage"' not in line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            t = _stamp(obj.get("timestamp")) or _stamp(obj.get("arrived_at"))
            if t is None:
                continue
            message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            usage = message.get("usage")
            key = str(message.get("id") or "")
            if not isinstance(usage, dict):
                inner = obj.get("event") if isinstance(obj.get("event"), dict) else {}
                usage = inner.get("usage")
                key = ""          # a turn report has no message id and is never a duplicate
            if isinstance(usage, dict) and usage:
                out.append((t, key or f"{path.name}#{index}", usage))
    return out


def _buckets(usage: dict) -> tuple[int, int, int, int]:
    """``(input, output, cache_read, cache_write)`` from either driver's spelling.

    The two vocabularies are read side by side rather than branched on a driver name: a driver id is
    not always recorded, and a reader keyed on one would silently return zeros for the other."""
    def _i(*keys) -> int:
        for k in keys:
            v = usage.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return 0
    cache_read = _i("cache_read_input_tokens", "cached_input_tokens")
    cache_write = _i("cache_creation_input_tokens", "cache_write_input_tokens")
    raw_input = _i("input_tokens")
    # One driver reports `input_tokens` as the TOTAL prompt (cached included); the other reports it
    # as the uncached remainder. Subtracting when it is clearly the total keeps the two comparable
    # instead of double-counting the cache on one of them.
    uncached = raw_input - cache_read - cache_write if raw_input >= cache_read + cache_write > 0 else raw_input
    return max(uncached, 0), _i("output_tokens"), cache_read, cache_write


def read_token_series(run_dir: Path, *, recorded_totals: dict | None = None) -> TokenSeries:
    """Cumulative token samples for one run, or a stated reason there is no curve.

    ``recorded_totals`` is the harness's own end-of-run figure (``output``/``cache_read`` keys). When
    given, the reconstructed total is compared against it and the disagreement is recorded."""
    series = TokenSeries()
    transcripts, raws = _files(run_dir)
    reports = _raw_samples(transcripts)
    source = "transcript_usage"
    raw_reports = _raw_samples(raws)
    if len(raw_reports) > len(reports):
        reports, source = raw_reports, "driver_turn_usage"
    if not reports:
        series.availability.set("token_series", unavailable(
            f"no event in {run_dir.name} carries a usage report this reader can place on a clock, so "
            f"the run has no token curve — only the end-of-run totals"))
        return series

    series.n_reports = len(reports)
    # Keep the LAST report for each message id: a streaming transcript emits partials first and the
    # complete usage last, so the final one is the whole message.
    reports.sort(key=lambda r: r[0])
    latest: dict[str, tuple[float, dict]] = {}
    for t, key, usage in reports:
        if key in latest:
            series.n_duplicates += 1
        latest[key] = (t, usage)

    running = [0, 0, 0, 0]
    for t, usage in sorted(latest.values(), key=lambda r: r[0]):
        delta = _buckets(usage)
        running = [a + b for a, b in zip(running, delta)]
        series.samples.append(TokenSample(t, *running))
    t0 = series.samples[0].t_s
    for sample in series.samples:
        sample.t_s -= t0
    series.source = source
    series.wall_s = series.samples[-1].t_s

    if series.can_rate:
        series.availability.set("token_series", measured(source))
    else:
        series.availability.set("token_series", unavailable(
            f"only {len(series.samples)} usage sample(s): this driver reports usage once per turn "
            f"rather than per message, so a rate drawn through them would be an average shown as a "
            f"trend", source=source))

    if recorded_totals:
        final = series.samples[-1]
        checks = {"output": (final.output_tokens, recorded_totals.get("output")),
                  "cache_read": (final.cache_read_tokens, recorded_totals.get("cache_read"))}
        drift = []
        for name, (ours, theirs) in checks.items():
            if not isinstance(theirs, (int, float)) or theirs <= 0:
                continue
            off = abs(ours - theirs) / theirs
            if off > _CROSS_CHECK_TOLERANCE:
                drift.append(f"{name} {ours:,} vs {theirs:,} ({off:.1%} apart)")
        if drift:
            series.availability.set("token_series_crosscheck", unavailable(
                "the curve reconstructed from per-message usage disagrees with the total the harness "
                "recorded independently: " + "; ".join(drift) + ". Trust the recorded total; this "
                "curve's SHAPE may still be informative but its magnitude is not."))
        else:
            series.availability.set("token_series_crosscheck", measured("cost_time_toolcalls"))
    return series


def rate_curve(series: TokenSeries, which: str = "output",
               window: int = 5) -> tuple[list[float], list[float]]:
    """``(minutes, tokens-per-minute)`` — a smoothed derivative, or empty when it cannot be drawn."""
    if not series.can_rate:
        return [], []
    key = {"output": "output_tokens", "input": "input_tokens",
           "cache_read": "cache_read_tokens", "billed_input": "billed_input"}[which]
    xs = [s.t_s / 60.0 for s in series.samples]
    ys = [getattr(s, key) for s in series.samples]
    rates: list[float] = []
    times: list[float] = []
    for i in range(1, len(xs)):
        lo = max(0, i - window)
        dt = xs[i] - xs[lo]
        if dt <= 0:
            continue
        rates.append(max(ys[i] - ys[lo], 0) / dt)
        times.append(xs[i])
    return times, rates
