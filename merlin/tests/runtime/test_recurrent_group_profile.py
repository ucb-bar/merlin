"""Execute the generated profiler across resets and reject misleading transport."""

import ctypes
import subprocess

import pytest

from merlin.runtime.whole_model_readback import (
    TRACE_CAPACITY,
    GroupCallee,
    ReadbackError,
    parse_group_profile,
    render_group_profile,
)


def test_generated_reset_bounds_recurrent_invocations_and_clears_overflow(tmp_path):
    source = render_group_profile([GroupCallee("kernel", "c_interface", None, 2)], [("first", "kernel")])
    # Exercise the identical generated wrapper on the build host with a deterministic
    # counter. This test does not stand in for target cycle measurements.
    source = source.replace('  __asm__ volatile("rdcycle %0" : "=r"(c));', '  c = ++ticks;')
    source = "static unsigned long long ticks;\n" + source
    source += """
static char console[1048576]; static unsigned at;
void htif_putc(char c) { console[at++] = c; console[at] = 0; }
void htif_puts(const char *s) { while (*s) htif_putc(*s++); }
void __real_kernel(void *r, const void *a, const void *b) { (void)r; (void)a; (void)b; ++ticks; }
void invoke(unsigned n) { for (unsigned i=0;i<n;i++) __wrap_kernel(0,0,0); }
const char *dump(void) { at=0; merlin_group_profile_dump(0,ticks+1); return console; }
"""
    c, so = tmp_path / "profile.c", tmp_path / "profile.so"
    c.write_text(source)
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", str(c), "-o", str(so)], check=True, capture_output=True)
    rt = ctypes.CDLL(str(so))
    rt.dump.restype = ctypes.c_char_p
    rt.invoke(TRACE_CAPACITY + 3)
    overflow = parse_group_profile(rt.dump().decode())
    assert overflow["calls"] == TRACE_CAPACITY and overflow["dropped"] == 3
    for _ in range(4):
        rt.merlin_group_profile_reset()
        rt.invoke(3)
        profile = parse_group_profile(rt.dump().decode())
        assert profile["calls"] == 3 and profile["dropped"] == 0
        assert profile["groups"][0]["name"] == "first"


@pytest.mark.parametrize("record", ["GROUP 0 5", "GAP 0 3", "GROUP_ID 0 a", "METRIC group_calls 1"])
def test_duplicate_profile_records_are_not_silently_overwritten(record):
    text = "GROUP_ID 0 a\nGROUP 0 5\nGAP 0 3\nGAP 1 2\nMETRIC group_calls 1\n"
    with pytest.raises(ReadbackError, match="duplicate"):
        parse_group_profile(text + record + "\n")


@pytest.mark.parametrize("record", ["GROUP 0 -1", "GAP -1 1", "METRIC group_calls_dropped -1"])
def test_negative_counters_cannot_be_attributed_to_host_work(record):
    with pytest.raises(ReadbackError, match="negative"):
        parse_group_profile(record)


def test_zero_call_profile_keeps_the_complete_host_window():
    profile = parse_group_profile("GAP 0 7\nMETRIC group_calls 0\nMETRIC group_calls_dropped 0\n")
    assert profile == {"groups": [], "tail_gap": 7, "calls": 0, "dropped": 0}
