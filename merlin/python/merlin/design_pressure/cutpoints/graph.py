"""Cut point: graph.

The graph view (pre-linalg, op-level dataflow) is an honest pass-through in M1: it records
that the cutpoint was visited and carries the op list forward. Full graph-level dataflow
analysis (producer/consumer distance, fusion boundaries) is M2.
"""
from __future__ import annotations


def cut_graph(region: dict) -> dict:
    """Pass-through marker for the graph cutpoint (M1)."""
    return {"cutpoint": "present", "ops": list(region.get("ops", []) or [])}
