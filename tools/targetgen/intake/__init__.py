"""Source intake scanners for TargetGen.

Each scanner is deterministic, pure-Python pattern matching. No LLM. They
inspect a target's source tree (a foreign repo, RTL block, MLIR dialect,
ISA docs, etc.) and emit ``SourceFinding`` records that aggregate into a
``SourceInventory``. The classifier and stage-map planner consume the
inventory to derive Merlin integration styles and patch surfaces.
"""

from __future__ import annotations

from .classifier import SOURCE_TO_TARGETGEN, classify_inventory
from .source_index import AVAILABLE_SCANNERS, build_source_inventory

__all__ = [
    "AVAILABLE_SCANNERS",
    "SOURCE_TO_TARGETGEN",
    "build_source_inventory",
    "classify_inventory",
]
