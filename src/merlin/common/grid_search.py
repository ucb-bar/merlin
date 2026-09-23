"""Deterministic explicit sweeps shared by compiler tuning and optional DSE."""

from __future__ import annotations

import itertools


def grid_search(space: dict, evaluate) -> list[dict]:
    """Score each Cartesian-product point, sorting descending by scalar/priority key."""
    keys = list(space)
    rows = []
    for combo in itertools.product(*(space[k] for k in keys)):
        point = dict(zip(keys, combo))
        rows.append({**point, "score": evaluate(point)})
    return _sorted(rows)


def _key(score):
    return score.priority_key() if hasattr(score, "priority_key") else score


def _sorted(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: _key(row["score"]), reverse=True)
