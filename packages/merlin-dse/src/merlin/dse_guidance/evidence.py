"""Evidence types, confidence *weights*, and cost-tier *proxies* — the single source of truth.

These tables encode how much we trust a number by *where it came from*, and how expensive an
axis is to actually build. Two things must stay honest:

  * ``CONFIDENCE_WEIGHTS`` are **weights**, not measurements. A ``measured`` weight of 1.0
    does not mean the measurement is perfect; it means a measured quantity is weighted above
    an analytical one when ranking axes. They scale ``priority_score`` and nothing else.

  * ``COST_TIERS`` are an ordinal **build-cost proxy** (software-only ... major RTL redesign),
    not measured PPA. They divide ``priority_score`` so a cheap axis with the same gap-closure
    ranks above an expensive one.

The ordering of ``EVIDENCE_TYPES`` is strongest -> weakest; ``weakest_evidence`` uses it to
combine the per-component tags an axis touches (an axis is only as trustworthy as its softest
input).
"""

from __future__ import annotations

from merlin.common.evidence import CONFIDENCE_WEIGHTS, EVIDENCE_TYPES, _rank, confidence_for, weakest_evidence

# Ordinal build-cost PROXY (not measured PPA):
#   1 = software-only / lowering change
#   2 = runtime change
#   3 = HW/SW contract change
#   4 = small RTL / control feature
#   5 = major datapath or memory-hierarchy redesign
COST_TIERS: dict[str, int] = {
    "PE_count_2x": 5,
    "SRAM_capacity_increase": 5,
    "DMA_bandwidth_2x": 5,
    "resident_packed_weights": 3,
    "resident_prefix_kv": 3,
    "command_batching": 2,
    "autonomous_K_loop": 4,
    "accumulator_commit": 4,
    "event_tokens": 4,
}
