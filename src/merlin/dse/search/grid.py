"""Grid search: enumerate a small explicit space and score every point. Deterministic; no LLM.

Two entry points: a generic ``grid_search`` over a ``space`` dict (matching
``search_space.schema.yaml``), and ``grid_search_strategies`` which scores a fixed strategy set
over the workload regions (the resident-regime scoreboard).
"""

from __future__ import annotations

from merlin.common.grid_search import _key, _sorted, grid_search  # noqa: F401 -- compatibility exports


def grid_search_strategies(candidates, evaluator) -> list[dict]:
    """Score each candidate strategy with ``evaluator``; return rows sorted by score."""
    rows = []
    for c in candidates:
        score = evaluator.evaluate(c)
        c.score = score
        rows.append(
            {
                "strategy": c.artifact.get("id"),
                "variant_class": c.artifact.get("variant_class"),
                "features": ";".join(c.artifact.get("interface_features", [])),
                "score": score,
                "candidate": c,
            }
        )
    return _sorted(rows)
