"""Pareto-frontier computation over multi-objective DSE points.

Generic and dependency-free: given a list of points (dicts) and the objective keys with a
direction per objective (minimise or maximise), return the non-dominated subset. Used to
compare the hardware-only and interface-aware design frontiers (latency vs area / energy).
"""
from __future__ import annotations


def _normalise_modes(objectives: list[str], modes) -> list[str]:
    if modes is None:
        return ["min"] * len(objectives)
    if isinstance(modes, str):
        return [modes] * len(objectives)
    if len(modes) != len(objectives):
        raise ValueError("modes length must match objectives")
    return list(modes)


def dominates(a: dict, b: dict, objectives: list[str], modes: list[str]) -> bool:
    """True iff ``a`` dominates ``b``: no worse on every objective, strictly better on one."""
    strictly_better = False
    for key, mode in zip(objectives, modes):
        av, bv = a[key], b[key]
        if mode == "min":
            if av > bv:
                return False
            if av < bv:
                strictly_better = True
        else:  # max
            if av < bv:
                return False
            if av > bv:
                strictly_better = True
    return strictly_better


def compute_pareto(points: list[dict], objectives: list[str], modes=None) -> list[dict]:
    """Return the non-dominated subset of ``points`` (order preserved)."""
    m = _normalise_modes(objectives, modes)
    frontier: list[dict] = []
    for p in points:
        if any(dominates(q, p, objectives, m) for q in points if q is not p):
            continue
        # Drop earlier points this one now dominates (keeps duplicates out of the frontier).
        if any(_same(p, f, objectives) for f in frontier):
            continue
        frontier.append(p)
    return frontier


def _same(a: dict, b: dict, objectives: list[str]) -> bool:
    return all(a[k] == b[k] for k in objectives)


def frontier_dominates(front_a: list[dict], front_b: list[dict],
                       objectives: list[str], modes=None) -> bool:
    """True iff every point on ``front_b`` is dominated by (or equal to) some point on ``front_a``.

    Used to assert the interface-aware frontier dominates the hardware-only frontier.
    """
    m = _normalise_modes(objectives, modes)
    for b in front_b:
        ok = any(dominates(a, b, objectives, m) or _same(a, b, objectives) for a in front_a)
        if not ok:
            return False
    return True
