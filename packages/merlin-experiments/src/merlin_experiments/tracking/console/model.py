"""Assemble one experiment's snapshot from the console sources, on a fast and a slow cadence.

Each source module in :mod:`.sources` exposes ``read(ctx) -> dict`` and, optionally,
``inspect(ctx, key) -> dict`` for the drill-down drawer. A source sees the outputs of the sources
before it in :data:`SOURCES` through ``ctx.shared`` (its latest refresh, even when that refresh was on
the slow cadence), so dependencies are explicit in the order. A source that raises becomes an
``{"error": ...}`` section plus a notice; it never stops the console.

Snapshot (``merlin.console.snapshot.v1``)::

    {"schema", "generated_at", "experiment": {...discovery row...}, "operator_private",
     "sections": {name: section}, "changed": [names refreshed this time],
     "timings": {name: seconds}, "diagnostics": [...], "notices": [...]}

Section shapes are documented in each source module's docstring and in
``docs/guides/experiment_dashboard.md``.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any

from . import SNAPSHOT_SCHEMA
from .io import Reader, clean

#: ``(name, tier)`` in dependency order. ``fast`` refreshes every interval; ``slow`` on the slow interval.
SOURCES: tuple[tuple[str, str], ...] = (
    ("resources", "fast"),
    ("machine", "fast"),
    ("agents", "fast"),
    ("tokens", "fast"),
    ("jobs", "fast"),
    ("events", "fast"),
    ("timeline", "fast"),
    ("capsules", "slow"),
    ("progress", "slow"),
    ("perf", "slow"),
    ("golden", "slow"),
    ("passes", "slow"),
    ("stages", "slow"),
)
#: Sources whose content exists only for an operator (never shown in public mode).
OPERATOR_ONLY = frozenset({"golden"})


@dataclass
class Context:
    """What a source may use: the experiment row, the shared reader and its own persistent state."""

    experiment: dict[str, Any]
    reader: Reader
    state: dict[str, Any]
    operator_private: bool
    now: float
    shared: dict[str, Any] = field(default_factory=dict)
    live: bool = False


def _module(name: str):
    return importlib.import_module(f"{__package__}.sources.{name}")


class Collector:
    """One experiment's sources, reader and per-source state between refreshes."""

    def __init__(
        self,
        experiment: dict[str, Any],
        *,
        operator_private: bool = False,
        clock=time.time,
        sources: tuple[tuple[str, str], ...] = SOURCES,
    ) -> None:
        self.experiment = experiment
        self.operator_private = operator_private
        self.clock = clock
        self.sources = tuple((n, t) for n, t in sources if operator_private or n not in OPERATOR_ONLY)
        self.reader = Reader()
        self.state: dict[str, dict[str, Any]] = {name: {} for name, _ in self.sources}
        self.sections: dict[str, Any] = {}
        self.timings: dict[str, float] = {}

    def update_experiment(self, experiment: dict[str, Any]) -> None:
        """Adopt a fresher discovery row (state, last record time) for the same run."""
        if experiment.get("id") == self.experiment.get("id"):
            self.experiment = experiment

    def context(self, name: str, now: float) -> Context:
        return Context(
            experiment=self.experiment,
            reader=self.reader,
            state=self.state[name],
            operator_private=self.operator_private,
            now=now,
            shared=self.sections,
            live=self.experiment.get("state") in ("LIVE", "STALLED", "ACTIVE"),
        )

    def refresh(self, tier: str = "all") -> dict[str, Any]:
        """Re-read the sources of ``tier`` (``fast``, ``slow`` or ``all``); return the whole snapshot."""
        now = self.clock()
        self.reader.begin()
        changed = []
        for name, source_tier in self.sources:
            if tier != "all" and source_tier != tier and name in self.sections:
                continue
            started = time.monotonic()
            try:
                section = _module(name).read(self.context(name, now))
            except Exception as exc:  # noqa: BLE001 -- one broken reader never stops the console
                section = {"error": f"{type(exc).__name__}: {clean(exc, 160)}"}
            self.timings[name] = round(time.monotonic() - started, 4)
            self.sections[name] = section
            changed.append(name)
        from .diagnose import diagnose

        snapshot = {
            "schema": SNAPSHOT_SCHEMA,
            "generated_at": now,
            "experiment": self.experiment,
            "operator_private": self.operator_private,
            "sections": self.sections,
            "changed": changed,
            "timings": dict(self.timings),
            "notices": sorted(self.reader.notices),
        }
        snapshot["diagnostics"] = diagnose(snapshot)
        return snapshot

    def inspect(self, kind: str, key: str) -> dict[str, Any]:
        """The drill-down record for one row: ``kind`` names the source, ``key`` the row."""
        names = {name for name, _ in self.sources}
        if kind not in names:
            return {"title": f"{kind}: not a console source", "sections": [], "paths": []}
        module = _module(kind)
        if not hasattr(module, "inspect"):
            return {"title": f"{kind} {key}", "sections": [], "paths": [], "notice": "this panel has no deeper record"}
        self.reader.begin()
        try:
            return module.inspect(self.context(kind, self.clock()), key)
        except Exception as exc:  # noqa: BLE001
            return {
                "title": f"{kind} {key}",
                "sections": [],
                "paths": [],
                "error": f"{type(exc).__name__}: {clean(exc, 160)}",
            }


__all__ = ["Collector", "Context", "OPERATOR_ONLY", "SOURCES"]
