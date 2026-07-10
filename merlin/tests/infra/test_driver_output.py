"""Regex-free parsers for our own driver / simulator stdout markers."""
from __future__ import annotations

from merlin.common.driver_output import (int_after, int_field, is_vector_mnemonic, kv_pairs,
                                         line_after_marker)


def test_int_after():
    assert int_after("junk  KIND 2 COUNT 3 CYCLES 4210 tail", "CYCLES") == 4210
    assert int_after("REGION_CYCLES 77", "REGION_CYCLES") == 77
    assert int_after("CYCLES_FULL 9 CYCLES 5", "CYCLES") == 5          # exact-token, not CYCLES_FULL
    assert int_after("negative CYCLES -12", "CYCLES") == -12
    assert int_after("no marker here", "CYCLES") is None
    assert int_after("CYCLES notanint", "CYCLES") is None


def test_int_field_both_forms():
    assert int_field("errors=0 done", "errors") == 0
    assert int_field("MR=4 foo", "MR") == 4
    assert int_field("MR 8", "MR") == 8                                 # space form falls through
    assert int_field("nope", "MR") is None


def test_line_after_marker_and_kv():
    assert line_after_marker("a\nMERLIN_E2E ticks=10 wall_ns=5\nb", "MERLIN_E2E") == "ticks=10 wall_ns=5"
    assert line_after_marker("MERLIN_REGION", "MERLIN_REGION") == ""
    assert line_after_marker("x", "MERLIN_E2E") is None
    assert kv_pairs("ticks=10 name=foo_bar calls=3 bare") == {"ticks": "10", "name": "foo_bar", "calls": "3"}


def test_is_vector_mnemonic_matches_old_regex():
    import re
    rx = re.compile(r"^v[a-z0-9]+(?:\.[a-z0-9]+)*$")
    for m in ["vsetvli", "vfmacc.vv", "vle32.v", "vadd.vi", "v", "v.x", "vmv1r.v", "add",
              "fadd.s", "vse8.v", "vwmacc.vx", "v0", "vfncvt.f.f.w", "", "VADD", "vle."]:
        assert bool(is_vector_mnemonic(m)) == bool(rx.match(m)), m
