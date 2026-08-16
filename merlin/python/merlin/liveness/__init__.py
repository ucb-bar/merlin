"""HW-agnostic *liveness / progress* oracle — an L2.5 tier between functional (L2) and RTL (L3).

A functional oracle (cyclotron L2) executes instructions correctly but models nothing about the
interconnect, finite on-chip storage, ordering, or the host-assist boundary — so it cannot tell you
whether a program would *stall, deadlock, or fault on real silicon*. Full RTL (L3) models all of that
but is slow and heavy. This package fills the middle: it answers the single question "would this even
have a chance on silicon?" — fast, and derived entirely from each target's OWN facts (never a
per-target literal), per the repo's cardinal rule.

Two complementary tools, both consuming HW-agnostic inputs:

* :mod:`merlin.liveness.preconditions` — (B) a *static* silicon-precondition linter. Each rule encodes a
  real "ran in sim, hung on silicon" failure (HTIF first-print hang, untranscodable ``fence``, unmapped
  address, unmanaged ``mstatus.VS``, under-declared VLEN, ``medany`` blob span, unaligned readback) as a
  check against a *derived* fact.
* :mod:`merlin.liveness.interconnect` — (A) a *dynamic* transaction-level resource/deadlock model. It
  replays a decoded memory-movement stream (the RoCC ``instruction_trace``) through finite on-chip
  resources (scratchpad / accumulator / DMA) sized from introspected capacities, detecting overflow,
  eviction-before-use, unmapped/misaligned DRAM, back-pressure, and visibility (drain) hazards.

Both are *screening* oracles — fast, conservative, and honest about what they cannot derive
(``UNKNOWN`` findings are surfaced, never silently dropped) — not cycle-accurate proofs.
"""
from __future__ import annotations

from .report import Finding, LivenessReport, Severity
from .facts import SiliconFacts, silicon_facts
from .oracle import Program, assess, persist

__all__ = [
    "Finding",
    "LivenessReport",
    "Severity",
    "SiliconFacts",
    "silicon_facts",
    "Program",
    "assess",
    "persist",
]
