"""Historical regression tracking: store and compare results across runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


HISTORY_DIR = Path(__file__).parent.parent / "results" / "history"


def save_to_history(results: list[dict], metadata: dict) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = HISTORY_DIR / f"run_{ts}.json"
    with open(path, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2, default=str)
    return path


def _load_history() -> list[dict]:
    if not HISTORY_DIR.exists():
        return []
    runs = []
    for p in sorted(HISTORY_DIR.glob("run_*.json")):
        try:
            with open(p) as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def compare_with_history(
    current_results: list[dict],
    threshold: float = 0.1,
) -> list[dict]:
    """Compare current results against the most recent historical run.

    Returns a list of regressions where p50 latency increased by more than
    `threshold` (fraction, e.g. 0.1 = 10%).
    """
    history = _load_history()
    if len(history) < 1:
        return []

    prev = history[-1]
    prev_by_key = {}
    for r in prev.get("results", []):
        key = (r.get("workload"), r.get("size"), r.get("backend"))
        prev_by_key[key] = r

    regressions = []
    for r in current_results:
        key = (r.get("workload"), r.get("size"), r.get("backend"))
        prev_r = prev_by_key.get(key)
        if not prev_r:
            continue

        curr_p50 = r.get("subsequent_p50_ms", 0)
        prev_p50 = prev_r.get("subsequent_p50_ms", 0)
        if prev_p50 <= 0 or curr_p50 <= 0:
            continue

        change = (curr_p50 - prev_p50) / prev_p50
        if change > threshold:
            regressions.append({
                "workload": r.get("workload"),
                "size": r.get("size"),
                "backend": r.get("backend"),
                "previous_p50_ms": prev_p50,
                "current_p50_ms": curr_p50,
                "change_pct": change * 100,
            })

    return regressions
