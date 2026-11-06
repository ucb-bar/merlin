"""The Zephyr image's memory planner: buffer lifetimes, their packing into one arena, and the refusal.

Three things are pinned here, each the one a person with a new model depends on:

* LIFETIMES. A buffer freed at its last use is dead from there on, so the arena holds the PEAK of what
  is live at once -- not the sum of every buffer ever made, which is what the region used to be sized
  from (a 2-layer TinyLlama at 128 tokens asked for a 528 MiB region on a 512 MiB board).
* THE REGION. It is the linked image (measured, never estimated) plus the heap residue plus the
  allocator's derived margin; an estimate that was too small is corrected by relinking, an overflow the
  linker reports is retried at the size it implies.
* THE REFUSAL. When a model does not fit, the message names every term -- weights, arena, peak, sum,
  heap residue, code, the board's DRAM -- because it is the first thing someone reads.
"""

from __future__ import annotations

import pytest

from merlin.llvmlower import arena_bind as ab
from merlin.llvmlower import optional_passes as op
from merlin.runtime.backends import zephyr_memory as zmem
from merlin.runtime.backends.zephyr_model import ZephyrModelError

MIB = 1 << 20


def _alloc(name: str, size: int) -> str:
    return f"""  %{name} = call ptr @malloc(i64 {size})
  %{name}i = ptrtoint ptr %{name} to i64
  %{name}p = inttoptr i64 %{name}i to ptr
  store float 1.000000e+00, ptr %{name}p, align 4"""


def _free(name: str) -> str:
    return f"  call void @free(ptr %{name})"


#: Three buffers in a chain, each freed right after the next one is made (dealloc at last use, as the
#: sunk deallocation places it): a and b coexist, b and c coexist, a and c never do.
CHAIN = f"""declare ptr @malloc(i64)

declare void @free(ptr)

define void @forward(ptr %out) {{
{_alloc("a", 4096)}
{_alloc("b", 8192)}
{_free("a")}
{_alloc("c", 4096)}
{_free("b")}
{_free("c")}
  ret void
}}
"""


def test_buffers_freed_at_last_use_pack_to_the_peak_not_the_sum():
    """a+b and b+c are the live pairs; the arena is the heavier pair, and c reuses a's bytes."""
    out, rep = ab.bind_arena(CHAIN)
    assert rep.bound == 3 and rep.refusals == {}
    assert rep.naive_total_bytes == 4096 + 8192 + 4096
    assert rep.peak_live_bytes == 4096 + 8192
    assert rep.arena_bytes == rep.peak_live_bytes
    offsets = {
        line.split()[0]: int(line.rsplit(" ", 1)[1]) for line in out.splitlines() if "@merlin_arena, i64" in line
    }
    assert offsets["%a"] == offsets["%c"] and offsets["%b"] != offsets["%a"]


#: Four buffers whose largest-first placement wastes a slot: b2 and b3 overlap, b3 and b1, b1 and b0.
#: The peak is b1 + b0 = 256 bytes; placing the 128-byte buffers first strands b3 above them (320).
STAGGERED = f"""declare ptr @malloc(i64)

declare void @free(ptr)

define void @forward(ptr %out) {{
{_alloc("b2", 128)}
{_alloc("b3", 64)}
{_free("b2")}
{_alloc("b1", 128)}
{_free("b3")}
{_alloc("b0", 128)}
{_free("b0")}
{_free("b1")}
  ret void
}}
"""


def test_the_packing_tries_more_than_one_order_and_keeps_the_smallest():
    """No greedy order is best for every program; on a 512-token TinyLlama section largest-first left
    the arena 11.5 MiB over its peak and the image 2.2 MiB over the board. Here largest-first needs 320
    bytes and allocation order reaches the 256-byte peak."""
    from merlin.xdsl_dialects.lowering.arena_plan import pack_disjoint

    out, rep = ab.bind_arena(STAGGERED)
    assert rep.peak_live_bytes == 256 and rep.arena_bytes == 256, rep.to_dict()
    conflicts = {"%b2": {"%b3"}, "%b3": {"%b2", "%b1"}, "%b1": {"%b3", "%b0"}, "%b0": {"%b1"}}
    largest_first = [("%b0", 128), ("%b1", 128), ("%b2", 128), ("%b3", 64)]
    assert pack_disjoint(largest_first, conflicts)[1] == 320


def test_the_same_buffers_freed_at_the_end_pack_to_their_sum():
    """Unsunk, every free sits at the end of the function and every buffer is live for the whole call:
    the plan is then exactly as large as the sum, which is why the frees have to move first."""
    unsunk = CHAIN.replace(_free("a") + "\n", "").replace(_free("b"), _free("a") + "\n" + _free("b"))
    _, rep = ab.bind_arena(unsunk)
    assert rep.arena_bytes == rep.naive_total_bytes == 4096 + 8192 + 4096


def _budget(**kw) -> zmem.MemoryBudget:
    facts = dict(
        board="kodiak-like",
        dram_bytes=512 * MIB,
        policy="static-arena",
        weights_bytes=92 * MIB,
        arena={"arena_bytes": 37 * MIB, "peak_live_bytes": 36 * MIB, "naive_total_bytes": 300 * MIB,
               "refusals": {"multiple_frees": 54}},
        heap=zmem.HeapResidue(bytes=62 * MIB, allocations=59),
        allocator=zmem.AllocatorFacts(chunk_unit=8, source="heap.h"),
        link_limit_bytes=1900 * MIB,
    )  # fmt: skip
    facts.update(kw)
    return zmem.MemoryBudget(**facts)


def test_the_margin_is_derived_from_the_allocator_not_a_fixed_slack():
    """Per allocation a header, the round-up to the chunk unit and the alignment padding; once, the heap's
    bookkeeping. Kilobytes for a model with tens of allocations -- where the old rule added 128 MiB."""
    b = _budget()
    assert b.margin_bytes == zmem._HEAP_FIXED_BYTES + 59 * (2 * 8 + zmem._MALLOC_ALIGN_BOUND)
    assert b.margin_bytes < MIB


def test_the_allocator_granule_is_read_from_the_zephyr_tree(tmp_path):
    heap = tmp_path / "lib" / "heap"
    heap.mkdir(parents=True)
    (heap / "heap.h").write_text("#ifndef X\n#define CHUNK_UNIT 16U\n#endif\n")
    assert zmem.allocator_facts(tmp_path).chunk_unit == 16
    (heap / "heap.h").write_text("/* nothing */\n")
    with pytest.raises(ZephyrModelError, match="defines no CHUNK_UNIT"):
        zmem.allocator_facts(tmp_path)


def test_a_first_link_that_fits_is_kept():
    b = _budget()
    links: list[int] = []
    region, h = zmem.link_to_fit(b, 288 * MIB, lambda r: links.append(r) or "h", lambda: 160 * MIB,
                                 default_ram_bytes=256 * MIB)  # fmt: skip
    assert (region, h, links) == (288 * MIB, "h", [288 * MIB])
    assert b.fits() and b.region_bytes == region
    assert b.to_dict()["need_bytes"] == 160 * MIB + 62 * MIB + b.margin_bytes


def test_an_image_larger_than_estimated_is_relinked_for_what_it_measures():
    b = _budget()
    links: list[int] = []
    region, _ = zmem.link_to_fit(b, 256 * MIB, lambda r: links.append(r) or "h", lambda: 230 * MIB,
                                 default_ram_bytes=256 * MIB)  # fmt: skip
    assert links == [256 * MIB, 304 * MIB] and region == 304 * MIB


def test_a_link_the_image_overflows_is_retried_at_the_size_the_linker_reports():
    b = _budget()
    links: list[int] = []

    def link(r: int) -> str:
        links.append(r)
        if r < 300 * MIB:
            raise ZephyrModelError("ld.bfd: region `RAM' overflowed by 4194304 bytes\ncollect2: error")
        return "h"

    region, _ = zmem.link_to_fit(b, 256 * MIB, link, lambda: 260 * MIB, default_ram_bytes=256 * MIB)
    assert links[0] == 256 * MIB and region >= 260 * MIB + 62 * MIB and len(links) == 2


def test_a_model_that_does_not_fit_is_refused_naming_every_term():
    """ResNet-50 at 224 used to ask for 624 MiB of a 512 MiB board with no say in why. The refusal now
    names the weights, the arena and its peak, the sum it replaced, the heap residue, the code and the
    DRAM -- enough to see which term to attack."""
    b = _budget(weights_bytes=400 * MIB)
    with pytest.raises(ZephyrModelError) as err:
        zmem.link_to_fit(b, 512 * MIB, lambda r: "h", lambda: 470 * MIB, default_ram_bytes=256 * MIB)
    text = str(err.value)
    for term in ("does not fit", "512.0 MiB of DRAM", "weights            400.0", "activation arena   37.0",
                 "peak live 36.0", "summed 362.0", "heap residue       62.0", "multiple_frees: 54",
                 "code and runtime   33.0"):  # fmt: skip
        assert term in text, (term, text)


def test_a_floor_over_the_board_says_the_code_was_never_linked():
    """Refused before linking (weights + arena + heap already exceed the DRAM), the code term is not a
    number: printing it as 0 MiB read as "the code is free"."""
    b = _budget(weights_bytes=500 * MIB)
    b.image_bytes = b.weights_bytes + b.arena_bytes
    text = b.refusal(image_at_least=True)
    assert "needs at least" in text and "not linked" in text and "0.0 MiB (code" not in text
    assert "allocator margin   " in text and "KiB" in text


def test_an_overflow_at_the_whole_board_is_refused_with_a_lower_bound():
    b = _budget()

    def link(r: int) -> str:
        raise ZephyrModelError("region `RAM' overflowed by 1048576 bytes")

    with pytest.raises(ZephyrModelError, match="needs at least"):
        zmem.link_to_fit(b, 512 * MIB, link, lambda: 0, default_ram_bytes=256 * MIB)


def test_a_link_failure_that_is_not_an_overflow_is_not_swallowed():
    with pytest.raises(ZephyrModelError, match="undefined reference"):
        zmem.link_to_fit(_budget(), 288 * MIB, lambda r: (_ for _ in ()).throw(ZephyrModelError("undefined reference")),
                         lambda: 0, default_ram_bytes=256 * MIB)  # fmt: skip


def test_the_plan_is_on_for_an_image_unless_deselected_by_name(monkeypatch):
    monkeypatch.delenv(op.ENV, raising=False)
    with zmem.planned_lowering() as policy:
        assert policy == "static-arena"
        assert op.switched("sink-deallocs", False) and op.switched("static-arena", False)
        assert "fold_uniform_fill_copy" in (op.selected_features(frozenset()) or ())
    assert op.active() == op.Selection()  # restored
    monkeypatch.setenv(op.ENV, "-static-arena")
    with zmem.planned_lowering() as policy:
        assert policy == "heap"
    monkeypatch.setenv(op.ENV, "-sink-deallocs")
    with zmem.planned_lowering() as policy:
        assert policy == "static-arena-unsunk"


def test_the_report_carries_sum_peak_arena_and_region():
    b = _budget()
    zmem.link_to_fit(b, 288 * MIB, lambda r: "h", lambda: 160 * MIB, default_ram_bytes=256 * MIB)
    d = b.to_dict()
    assert d["allocation_sum_bytes"] == 362 * MIB and d["peak_live_bytes"] == 36 * MIB
    assert d["arena_bytes"] == 37 * MIB and d["region_bytes"] == 288 * MIB and d["fits"] is True
    line = zmem.summary_line(d)
    assert "buffers sum 362.0" in line and "peak live 36.0" in line and "region 288.0 of 512.0" in line


def test_an_image_configured_for_a_stale_ram_size_is_refused(tmp_path):
    """Kconfig keeps CONFIG_SRAM_SIZE from the first configure, and the heap is sized from it: a relink that
    changed only the overlay reported 288 MiB while linking a 256 MiB image with a 19.5 MiB heap."""
    (tmp_path / "zephyr").mkdir()
    (tmp_path / "zephyr" / ".config").write_text("CONFIG_FOO=y\nCONFIG_SRAM_SIZE=262144\n")
    assert zmem.linked_ram_bytes(tmp_path) == 256 * MIB
    zmem.require_linked_ram(tmp_path, 256 * MIB)
    with pytest.raises(ZephyrModelError, match="configured for 256.0 MiB of RAM, not the 288.0 MiB region"):
        zmem.require_linked_ram(tmp_path, 288 * MIB)
