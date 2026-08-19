"""How much DRAM an image must be linked for, and why it is the SUM and not the peak.

`activation_peak_bytes` reports LIVE bytes and documents itself as a lower bound. Whether the live figure
is the one that has to fit depends on the allocator underneath, and the bare-metal one never reclaims
(`free` is a no-op), so there every byte ever requested has to be resident at once. A flat peak-plus-slack
came 27 MB short on a 26-layer model, and that kind of shortfall arrives at the END of a run -- after the
upload, after the layers, on hardware someone waited hours for. So the requirement is read off the IR that
will actually run.
"""
from __future__ import annotations

from merlin.runtime.backends import zephyr_model as zm


def _ll(tmp_path, body: str):
    f = tmp_path / "model.ll"
    f.write_text(body)
    return f


def test_every_spelling_of_an_allocation_is_counted(tmp_path):
    """The IR spells the same call several ways. A pattern narrow enough to read misses one of them, and a
    missed allocation under-reports the requirement -- which is the direction that produces a broken image
    rather than a wasteful one."""
    total, dynamic = zm.allocation_bytes(_ll(tmp_path, """
  %1 = call ptr @malloc(i64 1024)
  %2 = tail call noalias ptr @malloc(i64 2048) #3
  %3 = call noalias align 16 ptr @malloc(i64 noundef 4096)
  %4 = call ptr @llvm.memcpy(ptr %a, ptr %b, i64 8)
"""))
    assert (total, dynamic) == (1024 + 2048 + 4096, 0)


def test_a_computed_size_is_reported_not_assumed_zero(tmp_path):
    """Its bytes are genuinely unknown at build time. Counting it as nothing would make an incomplete
    answer look like a complete one."""
    total, dynamic = zm.allocation_bytes(_ll(tmp_path, """
  %1 = call ptr @malloc(i64 512)
  %2 = call ptr @malloc(i64 %n)
"""))
    assert total == 512
    assert dynamic == 1


def test_the_region_covers_the_sum_of_allocations(tmp_path):
    """The whole point: a model whose buffers dwarf its weights gets a region sized to the buffers."""
    w = 16 * 1024 * 1024
    # Weights-scaled and peak-based sizing would both leave this model far short.
    got = zm._ram_for_weights(w, 100 * 1024 * 1024, 900 * 1024 * 1024)
    assert got >= w + 900 * 1024 * 1024


def test_the_measured_total_replaces_a_pessimistic_estimate():
    """The measured sum is an UPPER bound on live bytes -- every live byte was allocated -- and it prices
    only what comes out of the arena. The MLIR estimate prices tensors that never reach the heap, and on a
    12-layer Gemma section it claimed 2827 MB against a measured 1533 MB: an estimate larger than the sum
    of every allocation the program makes. Sizing from it asked for 4416 MB on a 4096 MB design and refused
    a model that fits in 3136 MB."""
    w = 1531545600
    estimate, measured = 2827 * 1024 * 1024, 1607049952
    assert zm._ram_for_weights(w, estimate) // 2**20 == 4416
    assert zm._ram_for_weights(w, estimate, measured) // 2**20 == 3136


def test_the_estimate_is_still_used_when_nothing_was_measured():
    """A caller with no emitted IR to read (an external-weights layout, an older path) must not silently
    fall back to the flat headroom that under-provisioned whisper_tiny."""
    w = 120 * 1024 * 1024
    assert zm._ram_for_weights(w, 210 * 1024 * 1024) > zm._ram_for_weights(w, None)


def test_a_model_whose_buffers_are_small_is_unaffected():
    w = 1200 * 1024 * 1024
    assert (zm._ram_for_weights(w, 64 * 1024 * 1024, 8 * 1024 * 1024)
            == zm._ram_for_weights(w, None))
