"""Estimate model allocation pressure and place a Spike-only activation arena.

The estimate sums literal ``malloc`` call sites in generated LLVM IR. It is a
planning heuristic: branches can skip sites and loops can repeat them. The
caller must still treat a guest arena failure as a failed run. This module has
no knowledge of an accelerator or its command ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


GIB = 1 << 30
MIB = 1 << 20
_MALLOC = re.compile(r"\bcall ptr @malloc\(i64 ([^)]+)\)")


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class StaticMallocEstimate:
    callsites: int
    literal_bytes: int
    dynamic_sites: int


def estimate_llvm_malloc(path: str | Path) -> StaticMallocEstimate:
    """Count direct malloc sites and the bytes known from literal arguments."""
    source = Path(path).read_text(encoding="utf-8")
    arguments = _MALLOC.findall(source)
    literals = [int(value) for value in arguments if value.isdecimal()]
    return StaticMallocEstimate(len(arguments), sum(literals),
                                len(arguments) - len(literals))


def recommended_arena_bytes(estimate: StaticMallocEstimate) -> int:
    """Round a measured static sum up with headroom, keeping dynamic sites visible."""
    if estimate.callsites < 0 or estimate.literal_bytes < 0 or estimate.dynamic_sites < 0:
        raise ValueError("allocation estimate cannot be negative")
    # Ten percent plus 256 MiB accommodates harness buffers and alignment;
    # one additional GiB covers small dynamic sites but is not a proof of fit.
    planned = (estimate.literal_bytes * 11 + 9) // 10 + 256 * MIB
    if estimate.dynamic_sites:
        planned += GIB
    return max(2 * GIB, _round_up(planned, GIB))


def spike_session_layout(
    *, weights_bytes: int, arena_bytes: int,
    weights_base: int = 0x200000000, dram_base: int = 0x80000000,
    early_arena_base: int = 0xC0000000,
) -> dict[str, int]:
    """Keep a linked weight blob fixed and place a disjoint Spike arena."""
    if weights_bytes < 0 or arena_bytes <= 0 or weights_base <= early_arena_base:
        raise ValueError("invalid Spike weight or arena extent")
    weight_end = weights_base + weights_bytes
    if early_arena_base + arena_bytes <= weights_base:
        arena_base = early_arena_base
    else:
        arena_base = _round_up(weight_end, 256 * MIB)
    end = max(weight_end, arena_base + arena_bytes)
    return {"arena_base": arena_base, "arena_bytes": arena_bytes,
            "weights_base": weights_base,
            "mem_bytes": _round_up(end - dram_base, GIB)}
