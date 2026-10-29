"""What the run costs in tokens, per hour and per newly certified test case.

Token totals and the cumulative series come from :mod:`.tokens` (all author agents; ``uncached`` is
fresh input plus output, the part a cache does not discount). Certificates come from
:mod:`.certification`; *newly certified* is the L3 count now minus the L3 count in the previous
attempt's certificate ledger (the chain's newest earlier workspace), and, separately, minus the count
when this console first watched the run. Display telemetry; prices are not modeled.

Panel fields beyond the shared shape: ``per_hour`` {total, uncached}, ``per_new_l3`` {attempt,
watched}, ``new_l3`` {attempt, watched}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import num, obj
from .certification import _held
from .insight_common import duration, line, panel, previous_workspace, row, series, table

HOUR = 3600.0


def _compact(n: float | None) -> str | None:
    if n is None:
        return None
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= scale:
            return f"{n / scale:.1f}{suffix}"
    return f"{n:.0f}"


def _previous_l3(ctx) -> int | None:
    workspace = previous_workspace(ctx)
    if not workspace:
        return None
    cached = ctx.state.get("previous_l3")
    if cached and cached[0] == workspace:
        return cached[1]
    ledger = obj(ctx.reader.document(Path(workspace) / "qa" / "tier_state.json", limit=16 << 20))
    if not ledger:
        return None
    count = sum(1 for name, state in ledger.items() if not str(name).startswith("<") and _held(state, "L3") == "pass")
    ctx.state["previous_l3"] = (workspace, count)
    return count


def read(ctx) -> dict[str, Any]:
    now = ctx.now
    tokens = obj(ctx.shared.get("tokens"))
    total = obj(tokens.get("total"))
    s = obj(tokens.get("series"))
    t = s.get("t") or []

    def cumulative(i: int, keys: tuple[str, ...]) -> float:
        return sum(num((s.get(k) or [0] * len(t))[i]) or 0 for k in keys)

    all_keys = ("fresh_input", "cache_read", "cache_write", "output")
    uncached_keys = ("fresh_input", "output")
    per_hour = {"total": None, "uncached": None}
    points = []
    if len(t) >= 2:
        last = len(t) - 1
        base = next((i for i, at in enumerate(t) if at >= t[last] - HOUR), 0)
        span = t[last] - t[base]
        if span >= 300:
            scale = HOUR / span
            per_hour = {
                "total": (cumulative(last, all_keys) - cumulative(base, all_keys)) * scale,
                "uncached": (cumulative(last, uncached_keys) - cumulative(base, uncached_keys)) * scale,
            }
        j = 0
        for i in range(len(t)):
            while t[j] < t[i] - HOUR:
                j += 1
            if t[i] - t[j] >= 300:
                points.append(
                    (t[i], round((cumulative(i, all_keys) - cumulative(j, all_keys)) * HOUR / (t[i] - t[j]) / 1e6, 3))
                )
    cert = obj(obj(ctx.shared.get("certification")).get("totals"))
    # Held for some bytes, as the previous attempt is counted: an invalidation re-proving is no loss.
    l3 = num(cert.get("L3_held", cert.get("L3")))
    previous = _previous_l3(ctx)
    first = ctx.state.setdefault("first_seen", {})
    if l3 is not None and "l3" not in first:
        first.update(l3=l3, tokens=num(total.get("total")), at=now)
    spent = num(total.get("total"))
    new_attempt = (l3 - previous) if (l3 is not None and previous is not None) else None
    new_watched = (l3 - first["l3"]) if (l3 is not None and "l3" in first) else None
    spent_watched = (spent - first["tokens"]) if (spent is not None and first.get("tokens") is not None) else None
    per_new = {
        "attempt": (spent / new_attempt) if (spent and new_attempt and new_attempt > 0) else None,
        "watched": (spent_watched / new_watched) if (spent_watched and new_watched and new_watched > 0) else None,
    }
    budget = obj(tokens.get("budget"))
    uncached = (num(total.get("fresh_input")) or 0) + (num(total.get("output")) or 0)
    summary = [
        line(
            "tokens this attempt",
            f"{_compact(spent)} (uncached {_compact(uncached)})" if spent is not None else None,
        ),
        line(
            "last hour",
            f"{_compact(per_hour['total'])}/h (uncached {_compact(per_hour['uncached'])}/h)"
            if per_hour["total"] is not None
            else None,
        ),
        line(
            "budget used",
            f"{budget.get('used_pct'):.0f}% of {_compact(num(budget.get('total')))} ({budget.get('mode')})"
            if num(budget.get("used_pct")) is not None
            else None,
            "warn" if (num(budget.get("used_pct")) or 0) > 100 else None,
        ),
        line(
            "new L3 this attempt",
            f"{new_attempt} (previous attempt held {previous})"
            if new_attempt is not None
            else ("no previous attempt ledger" if l3 is not None else None),
            "warn" if new_attempt is not None and new_attempt <= 0 else None,
        ),
        line(
            "tokens per new L3",
            _compact(per_new["attempt"])
            if per_new["attempt"]
            else ("no new certificate" if new_attempt is not None else None),
        ),
        line(
            "since console watched",
            f"+{new_watched} L3 for {_compact(spent_watched)} tokens over {duration(now - first['at'])}"
            if new_watched is not None and first.get("at") and now - first["at"] >= 60
            else "watching from now",
        ),
    ]
    table_rows = [
        row([key, _compact(num(total.get(key)))], key=key)
        for key in ("fresh_input", "cache_read", "cache_write", "output", "reasoning")
    ]
    return panel(
        "COST",
        summary,
        table_=table(["tokens", "this attempt"], table_rows),
        series_=[series("M tokens per hour", points)],
        notes=["token counts only; no prices are modeled", "new certificates compare the certificate ledgers"],
        per_hour=per_hour,
        per_new_l3=per_new,
        new_l3={"attempt": new_attempt, "watched": new_watched, "previous": previous},
    )
