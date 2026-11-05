"""Rank pass-exploration candidates on training geometries, then check the winner on held-out ones.

``merlin.transforms.pass_exploration`` explores one source module at a time
(:func:`explore_candidates`) and ranks one set of same-domain observations
(:func:`rank_observed_metrics`). Choosing a schedule for an operator family needs both across several
geometries, with the held-out geometries frozen before anything is tuned. This wrapper is that loop
and nothing more: the caller supplies the space, the source modules, the facts and emission adapters
and a ``metrics`` function computing static observations of an emitted program (for example command
counts or bytes moved, read from its own emission). Nothing here simulates, scores by default or
qualifies performance; confirm a winner by timing it in a variant.

Use from the author's own exploration script::

    import sys; sys.path.insert(0, "<this directory>")
    from holdout_rank import rank_on_holdout
    report = rank_on_holdout(partition, modules, space, facts, output="program", emission=emission,
                             metrics=metrics, objectives=("commands",), domain="my-counts-v1")

Only the standard library and the workspace's own compiler library are imported.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

NOTE = ("Static screening of emitted programs; not cycles. The held-out check guards against a "
        "schedule fitted to the training shapes. Time the winner against submission/ in a variant "
        "before keeping it.")


def _observe(module, space, facts, *, output, emission, metrics, session_options) -> tuple[dict, dict]:
    """``({candidate: {metric: value}}, {candidate: status})`` for one source module."""
    from merlin.transforms.pass_exploration import explore_candidates

    values, statuses, by_program = {}, {}, {}
    for candidate in explore_candidates(module, space, facts, output=output, emission=emission,
                                        **session_options):
        name = candidate.evaluation.variant
        statuses[name] = candidate.evaluation.status
        if candidate.evaluation.status != "compiled" or candidate.emission is None:
            continue
        key = candidate.summary()["program_sha256"]
        if key not in by_program:  # identical emitted programs observe identically
            observed = metrics(candidate.emission, candidate.evaluation.result.receipt)
            if not isinstance(observed, Mapping):
                raise ValueError("metrics must return a mapping of metric name to a number or None")
            by_program[key] = dict(observed)
        values[name] = by_program[key]
    return values, statuses


def _aggregate(per_geometry: list[dict], names, objectives) -> dict:
    """Per candidate and objective, the sum over geometries; unknown if any geometry lacks it."""
    totals = {}
    for name in names:
        row = {}
        for objective in objectives:
            parts = [observed.get(name, {}).get(objective) for observed in per_geometry]
            row[objective] = None if any(p is None for p in parts) else sum(parts)
        totals[name] = row
    return totals


def _rank(totals: dict, objectives, *, basis: str, domain: str) -> dict:
    from merlin.transforms.pass_exploration import ObservedMetrics, rank_observed_metrics

    observations = tuple((name, ObservedMetrics(basis, domain, tuple((o, totals[name][o]) for o in objectives)))
                         for name in sorted(totals))
    return rank_observed_metrics(observations, objectives=tuple(objectives))


def rank_on_holdout(partition, modules: Mapping, space, facts: Callable, *, output: str, emission: Callable,
                    metrics: Callable, objectives: tuple, domain: str, basis: str = "static_emitted_ir",
                    **session_options) -> dict:
    """Explore every geometry of ``partition``, rank on training, and report the winner's held-out rank.

    ``partition`` is a ``ValidationPartition`` frozen before tuning; ``modules`` maps each of its
    ``ExplorationGeometry`` values to the source module of that geometry; ``space`` comes from
    ``generate_candidates`` once, so candidate names agree across geometries. ``session_options``
    (``context``, ``clone``, ``fingerprint``, ``verify_ir``, ``ir_level``, ``prefix``) pass through.
    """
    objectives = tuple(objectives)
    missing = [g for g in (*partition.training, *partition.held_out) if g not in modules]
    if missing:
        raise ValueError(f"no source module for {len(missing)} partition geometry(ies)")
    names = [variant.name for variant in space.variants]
    splits, statuses = {}, {}
    for split, geometries in (("training", partition.training), ("held_out", partition.held_out)):
        observed = []
        statuses[split] = {}
        for geometry in geometries:
            values, status = _observe(modules[geometry], space, facts, output=output, emission=emission,
                                      metrics=metrics, session_options=session_options)
            observed.append(values)
            statuses[split][geometry.identity] = status
        splits[split] = _rank(_aggregate(observed, names, objectives), objectives, basis=basis, domain=domain)
    training, held_out = splits["training"], splits["held_out"]
    winner = training["ranked"][0][0] if training["ranked"] else None
    held_ranked = [name for name, _ in held_out["ranked"]]
    position = held_ranked.index(winner) if winner in held_ranked else None
    best = held_out["ranked"][0][1] if held_out["ranked"] else None
    holds = position is not None and dict(held_out["ranked"])[winner] == best
    variant = next((v for v in space.variants if v.name == winner), None)
    return {
        "schema": "merlin.author_holdout_ranking.v1",
        "partition": partition.manifest(),
        "objectives": list(objectives), "basis": basis, "domain": domain,
        "training": training, "held_out": held_out,
        "winner": winner,
        "winner_bindings": [dict(space.bindings[key]) for key in variant.passes] if variant else [],
        "winner_held_out_position": position,
        "holds_on_held_out": holds,
        "statuses": statuses,
        "search_truncated": bool(space.truncated),
        "performance_qualification": False,
        "note": NOTE,
    }


def dumps(report: dict) -> str:
    """The report as JSON (tuples become lists)."""
    return json.dumps(report, indent=2, sort_keys=True, default=str)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        prog="holdout_rank.py",
        description="Library module, not a command: import rank_on_holdout from your own exploration "
                    "script (see this file's docstring and the explore-schedules skill).",
    ).parse_args()
    print(__doc__)
