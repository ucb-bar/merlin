"""Cut point: trace.

Honest pass-through in M1. Real trace-level analysis (event counts, queue occupancy,
pipeline bubbles) requires the command/event simulator and is M2.
"""
from __future__ import annotations


def cut_trace(region: dict) -> dict:
    """Pass-through marker for the trace cutpoint (M1)."""
    return {"cutpoint": "present"}
