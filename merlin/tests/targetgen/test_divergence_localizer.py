"""Hermetic tests for the advisory two-artifact divergence localizer (D2).

Pure-Python: synthetic command buffers + decoded traces (no build, no oracle, no external artifacts), so
the test is location-independent and fast. Asserts the localizer FIRES with a concrete op+field on the
byte-width stride/offset bug, STAYS SILENT on a well-formed artifact (fail-closed), and never returns a
golden value.
"""
from __future__ import annotations

from merlin.targetgen import divergence_localizer as DL


def _cb(out_cols: int, dtype: str = "i32"):
    return {"tensors": {"A0": {"shape": [16, out_cols], "dtype": "i8", "role": "input"},
                        "Y0": {"shape": [16, out_cols], "dtype": dtype, "role": "output"}}}


def _trace(insns):
    return {"instructions": insns}


def test_config_st_stride_wrong_element_width_is_localized():
    # output is 64-wide i32 -> row stride must be 256 B; the artifact emitted 64 B (i8 width).
    cb = _cb(64, "i32")
    trace = _trace([
        {"index": 5, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": 64}},
        {"index": 9, "class": "MVOUT", "decoded": {"readout": "i32",
                                                   "dram": {"arg_index": 2, "offset": 0},
                                                   "rows": 16, "cols": 16}},
    ])
    f = DL.localize(cb, {}, trace)
    assert f is not None
    assert f["op_index"] == 5 and f["class"] == "CONFIG_ST" and f["field"] == "out_stride_bytes"
    assert f["your_value"] == 64 and f["intended_value"] == 256
    # the hint carries no golden output value — only the agent's own stride + the derived intent
    assert "256" in DL.format_finding(f)


def test_correct_stride_is_not_flagged():
    cb = _cb(16, "i32")                         # 16 cols i32 -> 64 B stride, and the artifact used 64
    trace = _trace([
        {"index": 5, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": 64}},
        {"index": 9, "class": "MVOUT", "decoded": {"readout": "i32",
                                                   "dram": {"arg_index": 2, "offset": 0},
                                                   "rows": 16, "cols": 16}},
    ])
    assert DL.localize(cb, {}, trace) is None


def test_mvout_tile_offset_undersized_is_localized():
    # 36x8 i32 output tiled by rows; consecutive tiles must step 16*8*4=512 B, artifact stepped 128 (i8).
    cb = {"tensors": {"Y0": {"shape": [36, 8], "dtype": "i32", "role": "output"}}}
    trace = _trace([
        {"index": 3, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": 32}},   # 8*4 correct
        {"index": 10, "class": "MVOUT", "decoded": {"dram": {"arg_index": 2, "offset": 0},
                                                    "rows": 16, "cols": 8}},
        {"index": 11, "class": "MVOUT", "decoded": {"dram": {"arg_index": 2, "offset": 128},
                                                    "rows": 16, "cols": 8}},
    ])
    f = DL.localize(cb, {}, trace)
    assert f is not None and f["class"] == "MVOUT" and f["op_index"] == 11
    assert f["your_value"] == 128 and f["intended_value"] == 512


def test_fail_closed_when_stride_undecodable():
    # a decode that came back None (not derivable on this RTL) must NOT be flagged.
    cb = _cb(64, "i32")
    trace = _trace([{"index": 5, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": None}}])
    assert DL.localize(cb, {}, trace) is None


def test_no_outputs_no_flag():
    assert DL.localize({"tensors": {}}, {}, _trace([])) is None
