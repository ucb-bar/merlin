"""Grid search: enumerate a small explicit space and score every point. Deterministic; no LLM.

Two entry points: a generic ``grid_search`` over a ``space`` dict (matching
``search_space.schema.yaml``), and ``grid_search_strategies`` which scores a fixed strategy set
over the workload regions (the resident-regime scoreboard).
"""
from __future__ import annotations

import itertools


def grid_search(space: dict, evaluate) -> list[dict]:
    """Cartesian product over ``space`` (name -> list of values); ``evaluate(point)`` -> score.

    Returns rows ``{**point, "score": <evaluate result>}`` sorted by score descending. The score
    may be a float or any object with a ``priority_key()`` method.
    """
    keys = list(space)
    rows = []
    for combo in itertools.product(*(space[k] for k in keys)):
        point = dict(zip(keys, combo))
        rows.append({**point, "score": evaluate(point)})
    return _sorted(rows)


def grid_search_strategies(candidates, evaluator) -> list[dict]:
    """Score each candidate strategy with ``evaluator``; return rows sorted by score."""
    rows = []
    for c in candidates:
        score = evaluator.evaluate(c)
        c.score = score
        rows.append({
            "strategy": c.artifact.get("id"),
            "variant_class": c.artifact.get("variant_class"),
            "features": ";".join(c.artifact.get("interface_features", [])),
            "score": score,
            "candidate": c,
        })
    return _sorted(rows)


def _key(score):
    return score.priority_key() if hasattr(score, "priority_key") else score


def _sorted(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: _key(r["score"]), reverse=True)
