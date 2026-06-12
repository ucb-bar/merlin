"""Evolutionary search: mutate -> evaluate -> keep if better. Improves candidate strategies.

A simple (mu + lambda) loop with elitism and deterministic RNG. Selection and acceptance use
the shared ``Score.priority_key`` (correctness first), so search cannot trade correctness for
speed. Keeps candidate lineage.
"""
from __future__ import annotations

import random

from merlin.search.mutations import mutate


def evolutionary_search(seeds, evaluator, generations: int = 10, population: int = 6,
                        seed: int = 0) -> dict:
    """Evolve ``seeds`` for ``generations``. Returns ``{best, population, history}``."""
    rng = random.Random(seed)

    pop = []
    for c in seeds:
        c.score = evaluator.evaluate(c)
        pop.append(c)
    pop = _truncate(pop, population)

    history = [_snapshot(pop)]
    for _ in range(generations):
        children = []
        for parent in pop:
            child = mutate(parent, rng)
            child.score = evaluator.evaluate(child)
            children.append(child)
        pop = _truncate(_dedup(pop + children), population)
        history.append(_snapshot(pop))

    best = max(pop, key=lambda c: c.score.priority_key())
    return {"best": best, "population": pop, "history": history}


def _truncate(candidates, k):
    return sorted(candidates, key=lambda c: c.score.priority_key(), reverse=True)[:k]


def _dedup(candidates):
    seen, out = set(), []
    for c in candidates:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


def _snapshot(pop):
    best = max(pop, key=lambda c: c.score.priority_key())
    return {"best_strategy": best.artifact.get("id"), "best_total": round(best.score.total, 4)}
