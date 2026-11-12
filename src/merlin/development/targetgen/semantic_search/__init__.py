"""Bounded semantic instruction selection and scratchpad allocation.

``search`` compares exact captured linalg semantics with selected instruction
semantics. It enumerates equivalent instruction choices, including a small
sound integer commutativity rewrite, and retries allocation with Z3. The
returned plan is diagnostic until a target compiler emits and executes it.
"""

from merlin.development.targetgen.semantic_search.search import (
    SearchLimits,
    kernel_from_linalg_inventory,
    search,
    search_linalg_inventory,
)

__all__ = ["SearchLimits", "kernel_from_linalg_inventory", "search", "search_linalg_inventory"]
