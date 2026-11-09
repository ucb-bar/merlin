"""Zephyr image memory layout from the selected board and program requirements.

This module performs no compilation, simulator discovery or execution. Physical
capacity and port limits are supplied by the selected board contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ZephyrBuildError(RuntimeError):
    """The selected program and platform cannot produce an admitted image."""


_WEIGHTS_ALIGN_BYTES = 4096  # Zephyr memory-region alignment, not a board address


@dataclass(frozen=True)
class MemoryPlan:
    external: bool
    ram_region_bytes: int
    spike_span_bytes: int
    weights_base: int | None = None
    weights_region_bytes: int = 0


def memory_plan(brd: Any, weights_bytes: int, linked_region: int, override: int | None = None) -> MemoryPlan:
    """Choose an in-DRAM layout using only the selected board's memory facts.

    A missing external-weights layout is a refusal, not a guess at another board's
    16 GiB map. The returned spike span covers every byte of the emitted ELF.
    """
    default = brd.zephyr_default_ram_bytes
    limit = brd.zephyr_link_limit_bytes
    if default is None or limit is None:
        raise ZephyrBuildError(f"{brd.name}: Zephyr memory facts are missing from the selected board")
    if linked_region < 0 or weights_bytes < 0:
        raise ZephyrBuildError(f"{brd.name}: negative model memory requirement")
    if brd.dram_base > 0xFFFFFFFF or default > 0xFFFFFFFF:
        raise ZephyrBuildError(f"{brd.name}: Zephyr RAM overlay requires one-cell DRAM address and size")
    if override is not None:
        if override < default or override < weights_bytes or override > limit or override > brd.dram_bytes:
            raise ZephyrBuildError(f"{brd.name}: RAM override cannot hold weights or exceeds board limits")
        return MemoryPlan(False, override, override)
    if linked_region < default:
        raise ZephyrBuildError(f"{brd.name}: computed region is smaller than the port's default RAM region")
    if linked_region <= limit:
        if linked_region > brd.dram_bytes or linked_region > 0xFFFFFFFF:
            raise ZephyrBuildError(f"{brd.name}: linked region exceeds physical DRAM")
        return MemoryPlan(False, linked_region, linked_region)
    low = brd.zephyr_external_ram_bytes
    tail = brd.zephyr_external_tail_reserve_bytes
    if low is None or tail is None:
        raise ZephyrBuildError(
            f"{brd.name}: linked region exceeds the port's limit and the board declares no external-weights layout"
        )
    if low <= 0 or low > limit or low > 0xFFFFFFFF or tail < 0 or low + tail >= brd.dram_bytes:
        raise ZephyrBuildError(f"{brd.name}: invalid external-weights layout")
    base = brd.dram_base + low
    aligned = (weights_bytes + _WEIGHTS_ALIGN_BYTES - 1) & ~(_WEIGHTS_ALIGN_BYTES - 1)
    if base % _WEIGHTS_ALIGN_BYTES or aligned + low + tail > brd.dram_bytes:
        raise ZephyrBuildError(f"{brd.name}: aligned weights blob does not fit the declared DRAM layout")
    return MemoryPlan(True, low, low + aligned, base, aligned)


def require_external_arena(brd: Any, ram_region_bytes: int, segments: list[Any], demand: int) -> None:
    """Refuse an external image whose *linked* low-memory footprint crowds its arena."""
    low_end = brd.dram_base + ram_region_bytes
    low_segments = [sg for sg in segments if sg.vaddr < low_end]
    if any(sg.vaddr < brd.dram_base or sg.end > low_end for sg in low_segments):
        raise ZephyrBuildError(f"{brd.name}: linked ELF crosses the declared low RAM region")
    available = ram_region_bytes - sum(sg.memsz for sg in low_segments)
    if available < demand:
        raise ZephyrBuildError(
            f"{brd.name}: external-weights ELF leaves {available} bytes "
            f"of low RAM for {demand} bytes of activation demand; declare a larger "
            "external RAM region in the target-owned catalog"
        )


def ram_for_weights(
    weights_bytes: int,
    activation_bytes: int | None = None,
    allocation_bytes_total: int | None = None,
    *,
    default_ram_bytes: int,
) -> int:
    """Build the selected image; see docs/design/zephyr_model_memory.md for sizing rationale."""
    headroom = (weights_bytes * 3) // 10 + 128 * 1024 * 1024
    demand = int(allocation_bytes_total or 0) or int(activation_bytes or 0)
    if demand:
        headroom = max(headroom, demand + 128 * 1024 * 1024)
    total = weights_bytes + headroom
    align = 16 * 1024 * 1024
    total = ((total + align - 1) // align) * align
    return max(default_ram_bytes, total)
