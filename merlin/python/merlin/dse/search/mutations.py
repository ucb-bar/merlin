"""Mutation/repair operators over candidate strategies.

Deterministic edits on a strategy's effect-pass set (add / remove / toggle a pass), which in
turn changes the exposed interface features and behavior. LLMs would plug in here as additional
operators — they are operators, never the search method itself.
"""
from __future__ import annotations

from merlin.dse.strategy import _EFFECT_ORDER, effect_passes, strategy_from_passes
from merlin.dse.search.candidate import make_candidate

# The toggleable effect passes (lowering passes are structural and not mutated).
TOGGLEABLE = [p for p in _EFFECT_ORDER]


def _child(parent, passes) -> object:
    strategy = strategy_from_passes(passes)
    return make_candidate(strategy, parents=(parent.id,))


def mutate(candidate, rng) -> object:
    """Return one mutated child of ``candidate`` (a toggled effect pass)."""
    current = set(effect_passes(candidate.strategy()))
    pass_to_toggle = rng.choice(TOGGLEABLE)
    if pass_to_toggle in current:
        current.discard(pass_to_toggle)
    else:
        current.add(pass_to_toggle)
    return _child(candidate, current)


def neighbours(candidate) -> list:
    """All single-toggle neighbours of a candidate (deterministic, for exhaustive local moves)."""
    current = set(effect_passes(candidate.strategy()))
    out = []
    for p in TOGGLEABLE:
        nxt = set(current)
        nxt.discard(p) if p in nxt else nxt.add(p)
        out.append(_child(candidate, nxt))
    return out
