"""Quality-Diversity / MAP-Elites: illuminate the space, keeping many good families.

Fills a behavior-keyed archive (memory/control/granularity x workload regime). Each iteration
picks a random elite, mutates it, evaluates, and inserts into its cell if it beats the
incumbent. The output is a portfolio, not a single winner.
"""
from __future__ import annotations

import random

from merlin.search.archive import best_overall, update_archive
from merlin.search.mutations import mutate


def map_elites_search(seeds, evaluator, iterations: int = 40, seed: int = 0,
                      workload_regime: str | None = None) -> dict:
    """Run MAP-Elites from ``seeds``. Returns ``{archive, best, occupied_cells}``."""
    rng = random.Random(seed)
    archive: dict = {}

    for c in seeds:
        update_archive(archive, c, evaluator.evaluate(c), workload_regime)

    for _ in range(iterations):
        if not archive:
            break
        parent = rng.choice(list(archive.values()))["candidate"]
        child = mutate(parent, rng)
        update_archive(archive, child, evaluator.evaluate(child), workload_regime)

    return {"archive": archive, "best": best_overall(archive),
            "occupied_cells": len(archive)}
