"""Spike arena planning exposes its uncertainty and avoids weight overlap."""

from merlin.runtime.backends.spike_arena import (
    GIB,
    StaticMallocEstimate,
    estimate_llvm_malloc,
    recommended_arena_bytes,
    spike_session_layout,
)


def test_literal_malloc_sum_and_dynamic_site_are_reported(tmp_path):
    source = tmp_path / "model.ll"
    source.write_text(
        "%a = call ptr @malloc(i64 1024)\n%b = call ptr @malloc(i64 %size)\n%c = call ptr @malloc(i64 4096)\n"
    )
    estimate = estimate_llvm_malloc(source)
    assert estimate == StaticMallocEstimate(3, 5120, 1)
    assert recommended_arena_bytes(estimate) == 2 * GIB


def test_malloc_reader_ignores_comments_and_strings_and_accepts_line_breaks(tmp_path):
    source = tmp_path / "model.ll"
    source.write_text(
        "; call ptr @malloc(i64 999)\n"
        '@text = constant [30 x i8] c"call ptr @malloc(i64 999)"\n'
        "%a = call ptr @malloc(\n i64 1024\n)\n"
        "%b = call ptr @malloc(i64 add (i64 2, i64 3))\n"
    )
    assert estimate_llvm_malloc(source) == StaticMallocEstimate(2, 1024, 1)


def test_large_static_graph_moves_arena_past_fixed_weights():
    estimate = StaticMallocEstimate(690, 4_660_000_000, 0)
    arena = recommended_arena_bytes(estimate)
    assert arena == 6 * GIB
    layout = spike_session_layout(weights_bytes=4_400_000_000, arena_bytes=arena)
    assert layout["arena_base"] >= layout["weights_base"] + 4_400_000_000
    assert layout["mem_bytes"] >= layout["arena_base"] + arena - 0x80000000


def test_smaller_arena_uses_space_before_fixed_weights():
    layout = spike_session_layout(weights_bytes=900_000_000, arena_bytes=4 * GIB)
    assert layout["arena_base"] == 0xC0000000
    assert layout["arena_base"] + layout["arena_bytes"] <= layout["weights_base"]


def test_linked_static_extent_moves_arena_above_program():
    static_end = 0xC49108F0
    layout = spike_session_layout(weights_bytes=4_400_000_000, arena_bytes=2 * GIB, static_end=static_end)
    assert layout["static_end"] == static_end
    assert layout["arena_base"] == 0xD0000000
    assert layout["arena_base"] >= static_end
    assert layout["arena_base"] + layout["arena_bytes"] <= layout["weights_base"]
