"""Compilation strategy: a first-class, hashable compilation approach.

Generalizes the baseline/software_visible/hardware_managed/oracle variant enum in
`variants.py`. Loaded from `compilation_strategy.schema.yaml` artifacts; assembled into a
runnable xDSL pipeline via `merlin.pipelines.builder.build_pipeline`.

Placeholder module. No real logic yet.
"""
from __future__ import annotations


def load_strategies(*args, **kwargs):
    """TODO: load compilation_strategy YAML artifacts into a registry of Strategy objects."""
    raise NotImplementedError("load_strategies is a scaffold stub; not implemented yet.")


def strategy_id(*args, **kwargs):
    """TODO: return a stable hash for a strategy (keys output dirs + enables run caching)."""
    raise NotImplementedError("strategy_id is a scaffold stub; not implemented yet.")
