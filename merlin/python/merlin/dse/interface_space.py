"""The interface design space: the set of software-visible contracts the DSE may choose.

This is the axis hardware-only DSE does *not* have. A point in the interface space is a
:class:`~merlin.dse.strategy.Strategy` (the contract + its lowering). ``build_interface_space``
returns the candidate strategies; ``baseline_only`` returns the single opaque strategy used as
the fixed interface for the hardware-only DSE baseline.
"""
from __future__ import annotations

from merlin.dse.strategy import Strategy, default_strategies


def build_interface_space(target: str = "toy_npu",
                          variant_classes: list[str] | None = None) -> list[Strategy]:
    """Candidate interface strategies, optionally filtered to some variant classes."""
    strategies = default_strategies(target=target)
    if variant_classes is not None:
        strategies = [s for s in strategies if s.variant_class in variant_classes]
    return strategies


def baseline_only(target: str = "toy_npu") -> list[Strategy]:
    """Just the opaque baseline — the fixed interface for hardware-only DSE."""
    return [s for s in default_strategies(target=target) if s.variant_class == "baseline"]
