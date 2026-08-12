"""Synthetic bespoke-sim ORACLE contributed by the phantom5 fixture target.

This module is NEVER imported by name; capsule_runner._ensure_sim_oracles_discovered loads it BY FILE
PATH under the synthetic name ``merlin._oot_sim_oracles.phantom5`` (because phantom5's contract names
it via ``plugin.sim_oracle``). At import time it self-registers a fake sim engine with the
capsule_runner oracle registry, exactly as a real out-of-tree accelerator's sim oracle would — proving
a target contributes an oracle as DATA (a plugin path), with zero edit to the core ``_SIM_ORACLES``
literal. It carries no real toolchain: ``available()`` always fails closed, so the fixture can never be
mistaken for a runnable oracle.
"""
from __future__ import annotations

from merlin.targetgen.capsule_runner import register_sim_oracle


def _adapters(target: str) -> dict:
    """A trivial (empty) per-tier adapter map — the fixture proves only the discover/register plumbing,
    so it wires no real grading tiers."""
    return {}


def _available(target: str) -> tuple[bool, str]:
    """Fail closed: this is a test fixture, never runnable."""
    return (False, "phantom sim: test fixture, not runnable")


# Module-level self-registration under a DISTINCT sim-engine name — not "chipyard"/"cyclotron" — so its
# presence in capsule_runner._SIM_ORACLES uniquely proves the plugin loaded. exclusive=True mirrors a
# self-hosted core that grades its own kernel artifact.
register_sim_oracle("phantomsim", adapters=_adapters, available=_available, exclusive=True)
