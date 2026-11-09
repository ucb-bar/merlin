"""Compatibility imports for the canonical compiler owner."""

from merlin.compiler.partition.semantic import (
    SemanticRegionOutlining,
    extract_semantic_region,
    outline_semantic_regions,
    partition_semantic_dag,
    semantic_dispatch,
    semantic_staging_buffers,
)

__all__ = [
    "semantic_dispatch",
    "extract_semantic_region",
    "partition_semantic_dag",
    "semantic_staging_buffers",
    "SemanticRegionOutlining",
    "outline_semantic_regions",
]
