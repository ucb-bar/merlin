"""Estimate model allocation pressure and place a Spike-only activation arena.

The estimate sums literal ``malloc`` call sites in generated LLVM IR. It is a
planning heuristic: branches can skip sites and loops can repeat them. The
caller must still treat a guest arena failure as a failed run. This module has
no knowledge of an accelerator or its command ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.llvmlower.late_quant_rne import _tokens

#: The arena lives here (literal-addressed, inside the -m memory map this backend passes to spike).
ARENA_BASE = 0xC0000000  # derived-ok: address chosen by this backend's own -m map, not read from a target
DRAM_BASE = 0x80000000  # derived-ok: RISC-V platform DRAM base used by spike/fesvr; the -m map is passed explicitly


GIB = 1 << 30
MIB = 1 << 20


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class StaticMallocEstimate:
    callsites: int
    literal_bytes: int
    dynamic_sites: int


def estimate_llvm_malloc(path: str | Path) -> StaticMallocEstimate:
    """Count direct malloc sites and the bytes known from literal arguments."""
    from collections import deque

    callsites = literal_bytes = dynamic_sites = 0
    with Path(path).open(encoding="utf-8") as source:
        # Stream lines so large model modules do not retain a full token array.
        tokens = (token.text for line in source for token in _tokens(line))
        prefix = ("call", "ptr", "@malloc", "(", "i64")
        window = deque(maxlen=len(prefix))
        for token in tokens:
            window.append(token)
            if tuple(window) != prefix:
                continue
            argument, depth = [], 1
            for token in tokens:
                depth += (token == "(") - (token == ")")
                if depth == 0:
                    break
                argument.append(token)
            if depth:
                raise ValueError("unterminated LLVM malloc call")
            callsites += 1
            if len(argument) == 1 and argument[0].isascii() and argument[0].isdecimal():
                literal_bytes += int(argument[0])
            else:
                dynamic_sites += 1
            window.clear()
    return StaticMallocEstimate(callsites, literal_bytes, dynamic_sites)


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
    *,
    weights_bytes: int,
    arena_bytes: int,
    weights_base: int = 0x200000000,
    dram_base: int = 0x80000000,
    early_arena_base: int = 0xC0000000,
    static_end: int | None = None,
) -> dict[str, int]:
    """Keep static data, the arena, and a linked weight blob disjoint."""
    if weights_bytes < 0 or arena_bytes <= 0 or weights_base <= early_arena_base:
        raise ValueError("invalid Spike weight or arena extent")
    if static_end is not None:
        if static_end < dram_base or static_end > weights_base:
            raise ValueError("linked static extent overlaps fixed weights")
        early_arena_base = max(early_arena_base, _round_up(static_end, 256 * MIB))
    weight_end = weights_base + weights_bytes
    if early_arena_base + arena_bytes <= weights_base:
        arena_base = early_arena_base
    else:
        arena_base = _round_up(weight_end, 256 * MIB)
    end = max(weight_end, arena_base + arena_bytes)
    layout = {
        "arena_base": arena_base,
        "arena_bytes": arena_bytes,
        "weights_base": weights_base,
        "mem_bytes": _round_up(end - dram_base, GIB),
    }
    if static_end is not None:
        layout["static_end"] = static_end
    return layout


def _layout(
    arena_bytes: int,
    weights_bytes: int,
    *,
    dram_base: int = DRAM_BASE,
    dram_bytes: int | None = None,
    code_reserve: int = 64 * 1024 * 1024,
    spike_arena_base: int = ARENA_BASE,
) -> dict:
    """Absolute-address memory map for the bare-metal image.

    Default (``dram_bytes=None``) is the historical spike map: code@0x80000000, arena@0xC0000000,
    weights 256 MB-aligned above it and at least 0x2_0000_0000. spike is told to span it with ``-m``,
    so "above the DRAM a real board has" costs nothing there.

    On a REAL board it costs everything: an arena at 0xC0000000 and weights at 0x2_0000_0000 are simply
    not memory, so the image faults on its first activation. Given ``dram_bytes`` the map is packed
    inside ``[dram_base, dram_base + dram_bytes)`` instead — code first, then the weights blob, then the
    arena taking the rest — and it FAILS CLOSED if the model does not fit rather than emitting an image
    that addresses memory the chip does not have.
    """
    if dram_bytes is None:
        if spike_arena_base < dram_base or spike_arena_base % (1 << 20):
            raise ValueError("Spike arena base must be in DRAM and MiB-aligned")
        weights_base = spike_arena_base + arena_bytes
        weights_base = (weights_base + 0xFFFFFFF) & ~0xFFFFFFF  # 256MB align
        weights_base = max(weights_base, 0x200000000)
        mem_end = weights_base + weights_bytes
        mem_bytes = ((mem_end - DRAM_BASE) + 0x3FFFFFFF) & ~0x3FFFFFFF  # round to 1GB
        return {"arena_base": spike_arena_base, "weights_base": weights_base, "mem_bytes": mem_bytes}
    if spike_arena_base != ARENA_BASE:
        raise ValueError("Spike arena base override is not valid for a board DRAM layout")

    align = 1 << 20  # 1 MB is enough for a blob base
    weights_base = (dram_base + code_reserve + align - 1) & ~(align - 1)
    arena_base = (weights_base + weights_bytes + align - 1) & ~(align - 1)
    end = dram_base + dram_bytes
    if arena_base + arena_bytes > end:
        raise RuntimeError(
            f"does not fit: code+static-I/O reserve {code_reserve / 2**20:.0f} MB + weights "
            f"{weights_bytes / 2**20:.1f} MB + arena {arena_bytes / 2**20:.0f} MB exceeds the board's "
            f"{dram_bytes / 2**20:.0f} MB at {hex(dram_base)}. Shrink the arena or the model."
        )
    return {
        "arena_base": arena_base,
        "weights_base": weights_base,
        "mem_bytes": dram_bytes,
        "code_reserve": code_reserve,
    }
