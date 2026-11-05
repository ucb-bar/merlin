"""The console is the scarce resource, so the epilogue prints by evidence-per-byte.

Earned by a whole-model Gemma 2 2B FireSim run: it reached ``STAGE compute_done`` having executed
all 11,160 of its ops, then spent the rest of an eight-hour run cap emitting 2,469 lines of per-op
profile over a console measured at 7.6 bytes/s, and was killed before a single logit was printed.
Eight hours of correct computation produced nothing gradeable, because a debug-only dump sat
between the result and the console.

So the order is: metrics (tens of bytes), whole-tensor checks (tens of bytes), the OUT sample
(tens of kilobytes), the per-op profile (tens of kilobytes). A run truncated at any point has then
produced the most evidence it could have.
"""

from __future__ import annotations

import numpy as np
import pytest

from merlin.runtime.backends import zephyr_model as zm

_DEBUG_MEMORY = {"dram_base": 0x80000000, "region_bytes": 256 << 20}


def test_debug_memory_probe_requires_selected_board_memory():
    with pytest.raises(zm.ZephyrModelError, match="selected board's DRAM"):
        zm._main_c(0, debug=True)


def _at(src: str, marker: str) -> int:
    i = src.find(marker)
    assert i >= 0, f"{marker!r} missing from the generated harness"
    return i


def test_the_epilogue_is_ordered_by_evidence_per_byte():
    src = zm._main_c(0, debug=True, **_DEBUG_MEMORY)
    order = [
        "METRIC cycles",
        'printk("ARGMAX %d',
        'printk("SUM %u',
        'printk("HASH fnv1a32',
        'printk("OUT %d',
        "merlin_prof_dump();",
        'printk("DONE',
    ]
    pos = [_at(src, m) for m in order]
    assert pos == sorted(pos), (
        "epilogue out of order; cheap decisive evidence must precede expensive evidence:\n"
        + "\n".join(f"  {p:6d}  {m}" for p, m in zip(pos, order))
    )


def test_compute_done_still_marks_the_end_of_compute():
    """The cheap stage marker stays put — it is what tells a stalled run from a slow one."""
    src = zm._main_c(0, debug=True, **_DEBUG_MEMORY)
    assert _at(src, 'merlin_stage("compute_done")') < _at(src, "METRIC cycles")


def test_a_delivery_image_carries_no_dump_at_all():
    """The whole debug apparatus stays provably inert in a delivery build."""
    src = zm._main_c(0, debug=False)
    assert "merlin_prof_dump();" not in src
    assert "merlin_report_stacks();" not in src
    # ...but the answer, and the whole-tensor check on it, ship in every image
    assert 'printk("OUT %d' in src and 'printk("HASH fnv1a32' in src


def test_the_hash_the_harness_computes_is_the_one_the_host_checks():
    """FNV-1a over the output BYTES, in C and in numpy, must agree bit for bit."""
    src = zm._main_c(0, debug=True, **_DEBUG_MEMORY)
    assert "2166136261u" in src and "16777619u" in src, "not FNV-1a 32"

    def fnv1a32(b: bytes) -> int:
        h = 2166136261
        for byte in b:
            h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
        return h

    # the reference vector every FNV-1a implementation agrees on
    assert fnv1a32(b"") == 2166136261
    assert fnv1a32(b"a") == 0xE40C292C
    assert fnv1a32(b"foobar") == 0xBF9CF968

    # and it distinguishes a one-bit flip in a float tensor, which is the whole point
    a = np.arange(64, dtype=np.float32)
    b = a.copy()
    b[37] = np.frombuffer((np.frombuffer(b[37].tobytes(), dtype=np.uint32)[0] ^ 1).tobytes(), dtype=np.float32)[0]
    assert fnv1a32(a.tobytes()) != fnv1a32(b.tobytes())


def test_the_host_parser_reads_the_whole_tensor_digest():
    """A digest nobody parses is a digest nobody checks."""
    a = np.arange(8, dtype=np.float32)
    h = zm.out_hash_fnv1a32(a)
    console = f"METRIC cycles 5\nARGMAX 2 3 4\nSUM 1065353216\nHASH fnv1a32 8 {h}\nOUT 2 0 1065353216\nDONE\n"
    res = zm._parse_console(console, 0)
    assert res["out_hash"] == {"algo": "fnv1a32", "elems": 8, "value": h}
    # and it round-trips: the same array hashes to the value the console carried
    assert zm.out_hash_fnv1a32(a) == res["out_hash"]["value"]


def test_a_console_without_a_digest_still_parses():
    """Older images predate HASH; reading one must not become an error."""
    res = zm._parse_console("OUT 2 0 1065353216\nDONE\n", 0)
    assert "out_hash" not in res


@pytest.mark.parametrize("iters", [1, 4])
def test_no_console_write_inside_the_timed_window(iters):
    """A printk between the two mcycle reads is free on spike but blocks on FireSim's UART: a
    0.8M-cycle OPU kernel and a 0.95M-cycle RVV kernel both timed ~17.04M cycles (jobs 3852/3853)."""
    src = zm._main_c(0, iters=iters, warmup=1)
    start, end = _at(src, "uint64_t c0 = rd_mcycle();"), _at(src, "uint64_t c1 = rd_mcycle();")
    assert "printk(" not in src[start:end].replace('printk("FAIL', "")
    assert _at(src, "METRIC iter_cycles") > end
