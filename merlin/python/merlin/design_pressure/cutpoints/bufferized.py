"""Cut point: bufferized.

Honest pass-through in M1. Real bufferization-level analysis (actual buffer allocation,
in-place reuse, spill bytes) requires lowering and is M2.
"""
from __future__ import annotations


def cut_bufferized(region: dict) -> dict:
    """Pass-through marker for the bufferized cutpoint (M1)."""
    return {"cutpoint": "present"}
