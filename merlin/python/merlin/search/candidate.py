"""Candidate: a mutable compiler artifact plus its behavior descriptors and lineage.

In M1 the primary candidate type is a ``compilation_strategy`` (wrapping a
:class:`~merlin.dse.strategy.Strategy`). A candidate carries the artifact dict, the MAP-Elites
behavior descriptors derived from it, a lineage of parent ids, and its last score.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from merlin.dse import strategy as strat


@dataclass
class Candidate:
    """A searchable artifact (a compilation strategy) with behavior + lineage."""

    id: str
    artifact: dict
    behavior: dict = field(default_factory=dict)
    lineage: tuple[str, ...] = ()
    score: object = None

    def strategy(self) -> strat.Strategy:
        return strat.from_dict(self.artifact)


def _artifact_id(artifact: dict) -> str:
    key = "|".join([
        artifact.get("id", ""), artifact.get("lowering_pipeline", ""),
        ",".join(sorted(artifact.get("interface_features", []))),
    ])
    return "cand_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def make_candidate(artifact, behavior: dict | None = None, parents=()) -> Candidate:
    """Build a Candidate from a Strategy or a compilation_strategy dict."""
    strategy = artifact if isinstance(artifact, strat.Strategy) else strat.from_dict(artifact)
    d = strategy.to_dict()
    return Candidate(
        id=_artifact_id(d),
        artifact=d,
        behavior=behavior or strat.behavior_descriptors(strategy),
        lineage=tuple(parents),
    )


def seed_candidates() -> list[Candidate]:
    """The default strategies as seed candidates."""
    return [make_candidate(s) for s in strat.default_strategies()]
