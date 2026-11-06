"""How much of a board's DRAM a Zephyr model image needs, and the refusal when it does not fit.

An image's RAM region holds three things, and this module prices each from the build's own products
rather than from a formula:

* the IMAGE itself -- code, the weights blob (linked ``.data``), the embedded inputs, the per-hart
  stacks, the matrix unit's pack scratch, and the STATIC ACTIVATION ARENA. The arena is planned at
  build time (:mod:`merlin.llvmlower.arena_bind`) over the lowered model's own allocations, with each
  buffer freed at its last use (``sink-deallocs``), so it is the PEAK of what is live at once rather than
  the sum of everything ever allocated. It lands in ``.bss``, so the linker measures it with the rest;
  the image's size is read off the linked ELF, never estimated.
* the HEAP RESIDUE -- the allocations the plan could not seat (see the binder's refusal reasons) still go
  through ``malloc``. They are priced at their SUM, which is the bound that holds whatever the allocator
  does with frees: an allocation is carved from a hole or from the untouched tail, the tail only shrinks
  by allocations already counted, so a heap of the sum never runs out.
* the ALLOCATOR's own needs, derived from the Zephyr tree the image is built against: per allocation a
  chunk header, the round-up to the heap's chunk unit and the alignment padding; once, the heap's
  bookkeeping and the alignment of its start.

That replaces the old sizing -- weights + 30% + 128 MiB, or the sum of every allocation + 128 MiB -- whose
slack hid how little of it was needed and refused models that fit: a 2-layer TinyLlama at 128 tokens asked
for a 528 MiB region on a 512 MiB board.

The refusal names every term, because it is the first thing someone with a new model reads. The runtime
guard (``merlin_alloc_guard_zephyr.c``) stays linked: an allocation whose size is computed at run time is
counted but cannot be priced, and if one ever exceeds the bound the log names it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Zephyr memory-region alignment of an external weights blob (not a board address).
_WEIGHTS_ALIGN_BYTES = 4096

#: What one heap allocation looks like in the emitted LLVM IR.
_ALLOC_SYMBOL = "@malloc("

#: The granule a region is rounded up to. A region is a device-tree ``reg``; rounding keeps the overlay
#: stable across builds whose image differs by a few KiB, so their build hashes do not churn.
REGION_GRANULE_BYTES = 16 * 1024 * 1024

#: The passes that turn per-intermediate heap allocations into the static arena. ON for a Zephyr image
#: unless deselected (``--no-pass``); all three are exact (bit-identical output) and check themselves.
#: ``fold-uniform-fill-copy`` belongs to the plan because of what it frees: CSE shares ONE zero-filled
#: accumulator init across every contraction of the same shape, each contraction copies it, and that one
#: buffer is live -- and, its ownership resolved only at run time, freed only at the end of the call --
#: from the first layer to the last. Measured on an 8-layer TinyLlama section: 73 such buffers, 110 MiB
#: the arena could not seat; filling each destination directly leaves none.
MEMORY_PLAN_PASSES = ("sink-deallocs", "static-arena", "fold-uniform-fill-copy")

#: The largest alignment ``malloc`` pads an allocation to. The C library asks the heap for
#: ``alignof(max_align_t)``; this bound is the memref lowering's own buffer alignment, which no ABI's
#: ``max_align_t`` exceeds, so the margin does not depend on knowing which ABI it is.
_MALLOC_ALIGN_BOUND = 64

#: The heap's bookkeeping, once: ``struct z_heap`` plus one free-list bucket per power of two of the heap
#: size (at most 64 of them), and the round-up of the heap's start to a page. A bound, not a guess: the
#: struct and buckets come to a few hundred bytes for any heap a 64-bit address space can hold.
_HEAP_FIXED_BYTES = 8 * 1024


def _error(message: str) -> Exception:
    from .zephyr_model import ZephyrModelError  # the backend's error type; imported late (no cycle)

    return ZephyrModelError(message)


def _mib(n: int | None) -> str:
    return "?" if n is None else f"{n / 2**20:.1f}"


# ---- the lowered model's allocations ---------------------------------------------------------------


def allocation_bytes(ll_path: str | Path) -> tuple[int, int]:
    """``(total_constant_bytes, dynamic_calls)`` requested by the lowered model's ``malloc`` calls.

    Parsed STRUCTURALLY -- ``partition`` on the call, then on the argument's type -- not by pattern. The
    IR spells the same call several ways (``call``, ``tail call``, with and without ``noalias``, with a
    trailing attribute group), and a pattern narrow enough to be readable silently misses one spelling,
    which would under-report the requirement and produce an image that dies in its tail. The
    ``declare`` of ``malloc`` is not a call and is not counted (it used to be, as one "dynamic" call).

    ``dynamic_calls`` counts allocations whose size is a value rather than a literal. They are REPORTED,
    never assumed to be zero: their bytes are genuinely unknown at build time.
    """
    total = dynamic = 0
    for line in Path(ll_path).read_text(errors="replace").splitlines():
        if line.startswith("declare "):
            continue
        _, sym, rest = line.partition(_ALLOC_SYMBOL)
        if not sym:
            continue
        args, close, _ = rest.partition(")")
        if not close:
            continue
        # `i64 1024`, and also `i64 noundef 1024` -- the parameter attributes sit BETWEEN the type and
        # the value, so the size is the last token, not the second one.
        tokens = args.split()
        if len(tokens) < 2 or not tokens[0].startswith("i") or not tokens[-1].isdigit():
            dynamic += 1
            continue
        total += int(tokens[-1])
    return total, dynamic


@dataclass(frozen=True)
class HeapResidue:
    """What still goes through ``malloc`` once the arena plan has seated what it could.

    ``bytes`` counts each call site once, except that a site in a function other than the entry (an
    outlined parallel region) is counted once per hart, since every hart may hold one at the same time.
    ``dynamic`` sites have a computed size and are not in ``bytes``; ``in_loop`` sites are counted once
    (the ownership-based deallocation frees a loop body's buffer inside the iteration that made it).
    """

    bytes: int = 0
    allocations: int = 0
    dynamic: int = 0
    in_loop: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"bytes": self.bytes, "allocations": self.allocations, "dynamic": self.dynamic, "in_loop": self.in_loop}


def heap_residue(ll_text: str, *, entry: str = "forward", harts: int = 1) -> HeapResidue:
    """Read the heap residue off the final LLVM module (after the arena binder ran)."""
    from ...llvmlower import arena_bind as ab

    lines = ll_text.split("\n")
    total = count = dynamic = in_loop = 0
    for name, start, end in ab._defines(lines):
        blocks = ab._blocks_of(lines, start, end)
        by_label = {b.label: k for k, b in enumerate(blocks)}
        cyclic = ab._cyclic_blocks([[by_label[s] for s in b.succs] for b in blocks])
        copies = 1 if name == entry else max(1, harts)
        for k, b in enumerate(blocks):
            for li in b.insts:
                inst = lines[li].strip()
                if ab._call_target(inst) != "malloc":
                    continue
                size = ab._malloc_size(inst)
                if size is None:
                    dynamic += 1
                    continue
                in_loop += k in cyclic
                total += size * copies
                count += copies
    return HeapResidue(total, count, dynamic, in_loop)


# ---- the allocator underneath --------------------------------------------------------------------


@dataclass(frozen=True)
class AllocatorFacts:
    """The Zephyr heap's granule, read from the tree the image links against."""

    chunk_unit: int
    source: str

    def per_allocation(self) -> int:
        """Overhead one allocation can add: its chunk header (at most one unit -- 4 or 8 bytes, by heap
        size), the round-up of its size to whole units, and the alignment padding of an aligned
        allocation, which is how ``malloc`` asks the heap."""
        return 2 * self.chunk_unit + _MALLOC_ALIGN_BOUND

    def margin(self, allocations: int) -> int:
        return _HEAP_FIXED_BYTES + allocations * self.per_allocation()


def allocator_facts(zephyr_base: str | Path | None = None) -> AllocatorFacts:
    """``CHUNK_UNIT`` from ``lib/heap/heap.h`` of the selected Zephyr tree.

    Read, not assumed: it is the granule every allocation is rounded to, and a different tree may choose
    another. A tree that does not define it is refused rather than guessed at.
    """
    base = Path(zephyr_base or os.environ.get("ZEPHYR_BASE") or "")
    header = base / "lib" / "heap" / "heap.h"
    try:
        text = header.read_text(errors="replace")
    except OSError as exc:
        raise _error(f"cannot read the Zephyr heap's granule from {header}: {exc}") from exc
    for line in text.splitlines():
        words = line.split()
        if len(words) >= 3 and words[0] == "#define" and words[1] == "CHUNK_UNIT":
            value = words[2].rstrip("uUlL")
            if value.isdigit() and int(value) > 0:
                return AllocatorFacts(int(value), str(header))
    raise _error(f"{header} defines no CHUNK_UNIT; cannot price the heap's per-allocation overhead")


# ---- the budget -----------------------------------------------------------------------------------


@dataclass
class MemoryBudget:
    """Every term of an image's RAM requirement, and the region it was linked for.

    ``allocation_sum_bytes`` is what the model's buffers add up to if none is ever reused -- what the
    region used to be sized from. ``arena_bytes`` is what the plan seats them in; ``peak_live_bytes`` is
    the most of them live at once, the floor any plan could reach.
    """

    board: str
    dram_bytes: int
    policy: str
    weights_bytes: int
    arena: dict[str, Any] = field(default_factory=dict)
    heap: HeapResidue = field(default_factory=HeapResidue)
    allocator: AllocatorFacts | None = None
    image_bytes: int | None = None
    region_bytes: int | None = None
    link_limit_bytes: int | None = None

    @property
    def arena_bytes(self) -> int:
        return int(self.arena.get("arena_bytes") or 0)

    @property
    def peak_live_bytes(self) -> int:
        return int(self.arena.get("peak_live_bytes") or 0)

    @property
    def allocation_sum_bytes(self) -> int:
        return int(self.arena.get("naive_total_bytes") or 0) + self.heap.bytes

    @property
    def margin_bytes(self) -> int:
        return self.allocator.margin(self.heap.allocations) if self.allocator else 0

    @property
    def code_bytes(self) -> int | None:
        """The image less the weights and the arena: code, data, stacks, embedded inputs, scratch."""
        if self.image_bytes is None:
            return None
        return max(0, self.image_bytes - self.weights_bytes - self.arena_bytes)

    @property
    def capacity_bytes(self) -> int:
        """The largest region this board can link: its DRAM, and the port's link window."""
        return min(self.dram_bytes, self.link_limit_bytes or self.dram_bytes)

    def need_bytes(self, image_bytes: int | None = None) -> int:
        image = self.image_bytes if image_bytes is None else image_bytes
        return int(image or 0) + self.heap.bytes + self.margin_bytes

    def fits(self) -> bool:
        return self.image_bytes is not None and self.need_bytes() <= self.capacity_bytes

    def refusal(self, *, image_at_least: bool = False) -> str:
        need = self.need_bytes()
        image = "at least " if image_at_least else ""
        lines = [
            f"{self.board}: the model does not fit: the image needs {image}{_mib(need)} MiB and the board has "
            f"{_mib(self.capacity_bytes)} MiB"
            + (" of DRAM" if self.capacity_bytes == self.dram_bytes else " it can link"),
            f"  weights            {_mib(self.weights_bytes)} MiB",
            f"  activation arena   {_mib(self.arena_bytes)} MiB (peak live {_mib(self.peak_live_bytes)} MiB; "
            f"every buffer summed {_mib(self.allocation_sum_bytes)} MiB)",
            f"  heap residue       {_mib(self.heap.bytes)} MiB in {self.heap.allocations} allocation(s) the plan "
            f"could not seat{self._refusal_reasons()}",
            "  code and runtime   "
            + (
                "not linked: the terms above already exceed the board"
                if image_at_least and not self.code_bytes
                else f"{image}{_mib(self.code_bytes)} MiB (code, data, stacks, embedded inputs)"
            ),
            f"  allocator margin   {self.margin_bytes / 1024:.0f} KiB",
        ]
        if self.policy != "static-arena":
            lines.append("  the static arena plan is deselected; selecting it (the default) reuses buffers")
        return "\n".join(lines)

    def _refusal_reasons(self) -> str:
        reasons = self.arena.get("refusals") or {}
        return f" ({', '.join(f'{k}: {v}' for k, v in sorted(reasons.items()))})" if reasons else ""

    def to_dict(self) -> dict[str, Any]:
        """The build record's ``memory`` block: why this image does or does not fit its board."""
        return {
            "policy": self.policy,
            "board": self.board,
            "dram_bytes": self.dram_bytes,
            "region_bytes": self.region_bytes,
            "need_bytes": self.need_bytes() if self.image_bytes is not None else None,
            "fits": self.fits() if self.image_bytes is not None else None,
            "weights_bytes": self.weights_bytes,
            "image_bytes": self.image_bytes,
            "code_bytes": self.code_bytes,
            "allocation_sum_bytes": self.allocation_sum_bytes,
            "arena_bytes": self.arena_bytes,
            "peak_live_bytes": self.peak_live_bytes,
            "arena": dict(self.arena),
            "heap_residue": self.heap.to_dict(),
            "allocator_margin_bytes": self.margin_bytes,
            "allocator": {"chunk_unit": self.allocator.chunk_unit, "source": self.allocator.source}
            if self.allocator
            else None,
        }


def summary_line(memory: dict[str, Any]) -> str:
    """One line for a console or a manifest table: sum vs peak vs arena vs region."""

    def mib(key: str) -> str:
        return _mib(memory.get(key))

    return (
        f"buffers sum {mib('allocation_sum_bytes')} MiB, peak live {mib('peak_live_bytes')} MiB, arena "
        f"{mib('arena_bytes')} MiB, heap residue {_mib((memory.get('heap_residue') or {}).get('bytes'))} MiB, "
        f"weights {mib('weights_bytes')} MiB, image {mib('image_bytes')} MiB -> region {mib('region_bytes')} "
        f"of {mib('dram_bytes')} MiB"
    )


def region_for(budget: MemoryBudget, need: int, default_ram_bytes: int) -> int:
    """The region to link for ``need`` bytes: rounded up to the granule, never under the port's default
    (a FireSim image boots reliably only at the stock ``ram0`` size), and clamped to what the board can
    link when the rounding alone would cross it."""
    granule = REGION_GRANULE_BYTES
    region = max(default_ram_bytes, (need + granule - 1) // granule * granule)
    return min(region, budget.capacity_bytes) if need <= budget.capacity_bytes else region


def overflowed_by(message: str) -> int | None:
    """Bytes the linker says the image overflowed its memory region by, from its error text."""
    _, sep, tail = message.partition("' overflowed by ")
    if not sep:
        return None
    word = tail.split(" ", 1)[0]
    return int(word) if word.isdigit() else None


def image_span(elf: str | Path, dram_base: int, limit: int) -> int:
    """Bytes from the start of DRAM to the end of the last thing the image places below ``limit``.

    The span, not the sum of segment sizes: the heap starts where the image ENDS, so a gap the linker
    left between two segments is as unavailable to it as the segments are."""
    from ..elf_audit import read_elf

    _entry, segments, sections = read_elf(elf)
    ends = [sg.end for sg in segments if dram_base <= sg.vaddr < dram_base + limit]
    ends += [addr + size for addr, size, _kind in sections.values() if dram_base <= addr < dram_base + limit]
    return max(ends) - dram_base if ends else 0


def linked_ram_bytes(build_dir: str | Path) -> int | None:
    """The RAM size the image was CONFIGURED for: ``CONFIG_SRAM_SIZE`` (KiB) in the build's ``.config``.

    It is what the linker's RAM region and the C library's heap are both sized from -- not the devicetree
    overlay, which only seeds it on a fresh configure."""
    config = Path(build_dir) / "zephyr" / ".config"
    try:
        text = config.read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "CONFIG_SRAM_SIZE" and value.strip().isdigit():
            return int(value.strip()) * 1024
    return None


def require_linked_ram(build_dir: str | Path, region_bytes: int) -> None:
    """Refuse an image configured for a different RAM size than the region it is reported under.

    MEASURED: a relink that changed only the overlay left ``CONFIG_SRAM_SIZE`` at the first configure's
    value, so a 2-layer TinyLlama "relinked for 288 MiB" kept a 256 MiB link region and a 19.5 MiB heap
    for 39 MiB of heap residue -- an image that would end in ``FAIL alloc`` -- while its record said 288.
    """
    linked = linked_ram_bytes(build_dir)
    if linked != region_bytes:
        raise _error(
            f"the image was configured for {_mib(linked)} MiB of RAM, not the {_mib(region_bytes)} MiB region "
            f"it is linked for (CONFIG_SRAM_SIZE in {Path(build_dir) / 'zephyr' / '.config'})"
        )


def link_to_fit(
    budget: MemoryBudget,
    first_region: int,
    link: Callable[[int], str],
    measure: Callable[[], int],
    *,
    default_ram_bytes: int,
) -> tuple[int, str]:
    """Link for ``first_region``; then for exactly what the linked image needs. Returns (region, hash).

    The region and the image are circular -- the heap is whatever the region leaves after the image, and
    the image's size is not known until it is linked -- so the first link is at an estimate and the
    region is corrected to the MEASURED image. A first link the image overflows (the linker names by how
    much) is retried at the size that implies. The image does not change size with the region, so one
    correction settles it. A need beyond what the board can link is refused with every term named.
    """
    region = min(first_region, budget.capacity_bytes)
    for _ in range(3):
        try:
            build_hash = link(region)
        except Exception as exc:  # noqa: BLE001 -- only a region overflow is ours to handle
            over = overflowed_by(str(exc))
            if over is None:
                raise
            budget.image_bytes = region + over
            if budget.need_bytes() > budget.capacity_bytes or region >= budget.capacity_bytes:
                raise _error(budget.refusal(image_at_least=True)) from exc
            region = region_for(budget, budget.need_bytes(), default_ram_bytes)
            continue
        budget.image_bytes = measure()
        need = budget.need_bytes()
        if need > budget.capacity_bytes:
            raise _error(budget.refusal())
        if need <= region:
            budget.region_bytes = region
            return region, build_hash
        print(
            f"[zephyr_model] region {_mib(region)} MiB leaves {_mib(region - budget.image_bytes)} MiB of heap "
            f"for {_mib(budget.heap.bytes + budget.margin_bytes)} MiB of heap residue and allocator margin; "
            f"the linked image measures {_mib(budget.image_bytes)} MiB. Relinking for "
            f"{_mib(region_for(budget, need, default_ram_bytes))} MiB.",
            flush=True,
        )
        region = region_for(budget, need, default_ram_bytes)
    raise _error(f"{budget.board}: the region did not settle after three links ({_mib(region)} MiB)")


@contextlib.contextmanager
def planned_lowering() -> Iterator[str]:
    """Select the arena plan's passes for one lowering, unless the build deselected them by name.

    Yields the resulting policy: ``static-arena`` (the arena over sunk frees, the default),
    ``static-arena-unsunk`` (the arena over end-of-function frees, which reuses nothing) or ``heap`` (the
    arena deselected). Process-wide for the duration, as the pass selection is (the lowering reads it in
    child processes too), so the caller holds the IR lock around it.
    """
    from ...llvmlower import optional_passes as op

    current = op.active()
    wanted = [name for name in MEMORY_PLAN_PASSES if current.state(name) is None]
    with op.applied(op.Selection.of(enable=wanted)) as combined:
        arena, sunk = (combined.state(name) is True for name in ("static-arena", "sink-deallocs"))
        yield "static-arena" if arena and sunk else ("static-arena-unsunk" if arena else "heap")


# ---- the board's layout --------------------------------------------------------------------------


@dataclass(frozen=True)
class _MemoryPlan:
    external: bool
    ram_region_bytes: int
    spike_span_bytes: int
    weights_base: int | None = None
    weights_region_bytes: int = 0


def _memory_plan(brd: Any, weights_bytes: int, linked_region: int, override: int | None = None) -> _MemoryPlan:
    """Choose an in-DRAM layout from the selected board's memory facts only: a missing external-weights layout is
    refused, never guessed. The spike span covers the whole ELF; a region over the DRAM is refused with sizes."""
    default = brd.zephyr_default_ram_bytes
    limit = brd.zephyr_link_limit_bytes
    if default is None or limit is None:
        raise _error(f"{brd.name}: Zephyr memory facts are missing from the selected board")
    if linked_region < 0 or weights_bytes < 0:
        raise _error(f"{brd.name}: negative model memory requirement")
    if brd.dram_base > 0xFFFFFFFF or default > 0xFFFFFFFF:
        raise _error(f"{brd.name}: Zephyr RAM overlay requires one-cell DRAM address and size")
    if override is not None:
        if override < default or override < weights_bytes or override > limit or override > brd.dram_bytes:
            raise _error(f"{brd.name}: RAM override cannot hold weights or exceeds board limits")
        return _MemoryPlan(False, override, override)
    if linked_region < default:
        raise _error(f"{brd.name}: computed region is smaller than the port's default RAM region")
    if linked_region <= limit:
        if linked_region > brd.dram_bytes or linked_region > 0xFFFFFFFF:
            need, w, has = linked_region >> 20, weights_bytes >> 20, brd.dram_bytes >> 20
            raise _error(f"{brd.name}: linked region exceeds physical DRAM: the image needs {need} MiB ({w} "
                         f"MiB of weights, the rest code and arena), the board has {has} MiB")  # fmt: skip
        return _MemoryPlan(False, linked_region, linked_region)
    low = brd.zephyr_external_ram_bytes
    tail = brd.zephyr_external_tail_reserve_bytes
    if low is None or tail is None:
        raise _error(
            f"{brd.name}: linked region exceeds the port's limit and the board declares no external-weights layout"
        )
    if low <= 0 or low > limit or low > 0xFFFFFFFF or tail < 0 or low + tail >= brd.dram_bytes:
        raise _error(f"{brd.name}: invalid external-weights layout")
    base = brd.dram_base + low
    aligned = (weights_bytes + _WEIGHTS_ALIGN_BYTES - 1) & ~(_WEIGHTS_ALIGN_BYTES - 1)
    if base % _WEIGHTS_ALIGN_BYTES or aligned + low + tail > brd.dram_bytes:
        raise _error(f"{brd.name}: aligned weights blob does not fit the declared DRAM layout")
    return _MemoryPlan(True, low, low + aligned, base, aligned)


def _require_external_arena(brd: Any, ram_region_bytes: int, segments: list[Any], demand: int) -> None:
    """Refuse an external image whose *linked* low-memory footprint crowds its heap."""
    low_end = brd.dram_base + ram_region_bytes
    low_segments = [sg for sg in segments if sg.vaddr < low_end]
    if any(sg.vaddr < brd.dram_base or sg.end > low_end for sg in low_segments):
        raise _error(f"{brd.name}: linked ELF crosses the declared low RAM region")
    available = ram_region_bytes - sum(sg.memsz for sg in low_segments)
    if available < demand:
        raise _error(
            f"{brd.name}: external-weights ELF leaves {available} bytes "
            f"of low RAM for {demand} bytes of activation demand; declare a larger "
            "external RAM region in the target-owned catalog"
        )
