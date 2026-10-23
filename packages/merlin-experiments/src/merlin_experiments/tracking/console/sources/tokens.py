"""Cumulative token use across every author agent: totals, a time series, rates, budget and limits.

Built from the agents' parsed streams (:func:`.agents.index_for`; nothing is read twice): each thread's
cumulative usage samples are summed at every sample time, so the series is the run's total as it grew.
Totals prefer the agents section (which prefers a finished run's authoritative ``author_agents``).
The budget is the run's ``merlin.author_token_budget.v1`` (``run_config.author_token_budget``) under its
own accounting: all authors' input including cache plus output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import num, obj
from . import TOKEN_FIELDS, downsample, empty_tokens
from .agents import index_for

SERIES_FIELDS = ("fresh_input", "cache_read", "cache_write", "output", "reasoning")
RATE_WINDOW_S = 600.0


def merged_series(threads) -> list[tuple[float, dict]]:
    """``(t, summed cumulative tokens)`` at each sample time of any thread."""
    events = sorted(
        ((t, thread_id, tokens) for thread_id, thread in threads.items() for t, tokens in thread.get("samples") or []),
        key=lambda e: (e[0], e[1]),
    )
    latest: dict[str, dict] = {}
    out: list[tuple[float, dict]] = []
    for when, thread_id, tokens in events:
        latest[thread_id] = tokens
        summed = {k: sum((v.get(k) or 0) for v in latest.values()) for k in TOKEN_FIELDS}
        if out and out[-1][0] == when:
            out[-1] = (when, summed)
        else:
            out.append((when, summed))
    return out


def _sum(rows: list[dict]) -> dict:
    total = empty_tokens()
    for key in TOKEN_FIELDS:
        values = [r.get(key) for r in rows if isinstance(r.get(key), int | float)]
        total[key] = sum(values) if values else None
    return total


def read(ctx) -> dict[str, Any]:
    index = index_for(ctx)
    agents = [a for a in (obj(ctx.shared.get("agents")).get("agents") or []) if isinstance(a, dict)]
    total = _sum([obj(a.get("tokens")) for a in agents])
    points = merged_series(index.threads)
    sampled = downsample(points)
    series = {"t": [p[0] for p in sampled], **{k: [p[1].get(k) for p in sampled] for k in SERIES_FIELDS}}
    rate = {"input": None, "output": None}
    if len(points) >= 2:
        end_t, end = points[-1]
        start_t, start = next(((t, v) for t, v in points if t >= end_t - RATE_WINDOW_S), points[0])
        minutes = (end_t - start_t) / 60
        if minutes > 0:
            rate = {
                "input": ((end.get("input") or 0) - (start.get("input") or 0)) / minutes,
                "output": ((end.get("output") or 0) - (start.get("output") or 0)) / minutes,
            }
    grand = (
        sum(
            a_total
            for a_total in [obj(a.get("tokens")).get("total") for a in agents]
            if isinstance(a_total, int | float)
        )
        or None
    )
    per_agent = []
    for agent in agents:
        tok = obj(agent.get("tokens"))
        value = tok.get("total")
        per_agent.append(
            {
                "id": agent.get("id"),
                "name": agent.get("name") or agent.get("path") or agent.get("id"),
                "total": value,
                "output": tok.get("output"),
                "share": value / grand if isinstance(value, int | float) and grand else None,
            }
        )
    per_agent.sort(key=lambda r: -(r["total"] or 0))
    cache_hit = (
        total["cache_read"] / total["input"]
        if isinstance(total["cache_read"], int | float) and total["input"]
        else None
    )
    env = obj(ctx.reader.yaml(Path(ctx.experiment.get("run_dir") or "") / "environment.yaml", limit=16 << 20))
    declared = obj(obj(env.get("run_config")).get("author_token_budget"))
    budget = None
    target, reserve = num(declared.get("target")), num(declared.get("reserve"))
    if target:
        used = (total["input"] or 0) + (total["output"] or 0) if total["input"] is not None else None
        budget = {
            "total": target,
            "reserve": reserve,
            "working": target - reserve if reserve is not None else None,
            "used_pct": 100 * used / target if used is not None else None,
            "mode": declared.get("mode"),
            "accounting": declared.get("accounting"),
        }
    notes = []
    if not points:
        notes.append("No token usage samples recorded; the curve is empty.")
    if budget is None:
        notes.append("No author token budget recorded for this run.")
    return {
        "total": total,
        "series": series,
        "per_agent": per_agent,
        "rate_per_min": rate,
        "cache_hit_ratio": cache_hit,
        "budget": budget,
        "rate_limit": index.rate_limit,
        "context_window": index.context_window,
        "notes": notes,
    }


__all__ = ["merged_series", "read"]
