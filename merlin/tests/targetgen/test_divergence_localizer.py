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
    f = DL.localize(cb, {}, trace, tile_edge=16)
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
    assert DL.localize(cb, {}, trace, tile_edge=16) is None


def test_mvout_tile_offset_undersized_is_localized():
    # 36x8 i32 output tiled by rows; consecutive tiles must step 16*8*4=512 B, artifact stepped 128 (i8).
    # The output-store tiles carry the accumulator-readout direction (readout/acc_addr), as the real decode does.
    cb = {"tensors": {"Y0": {"shape": [36, 8], "dtype": "i32", "role": "output"}}}
    trace = _trace([
        {"index": 3, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": 32}},   # 8*4 correct
        {"index": 10, "class": "MVOUT", "decoded": {"readout": "i32", "acc_addr": 0,
                                                    "dram": {"arg_index": 2, "offset": 0},
                                                    "rows": 16, "cols": 8}},
        {"index": 11, "class": "MVOUT", "decoded": {"readout": "i32", "acc_addr": 0,
                                                    "dram": {"arg_index": 2, "offset": 128},
                                                    "rows": 16, "cols": 8}},
    ])
    f = DL.localize(cb, {}, trace, tile_edge=16)
    assert f is not None and f["class"] == "MVOUT" and f["op_index"] == 11
    assert f["your_value"] == 128 and f["intended_value"] == 1024


def test_input_load_tiles_do_not_false_fire():
    """An INPUT load (a DRAM->scratchpad move: it carries dram+rows+cols but a spad_addr, not the
    accumulator-readout direction) must NOT be mistaken for an output store — even though its tile step
    would be 'undersized' for the OUTPUT element width. This locks the input/output distinction that a
    pure field-presence match would break."""
    cb = {"tensors": {"Y0": {"shape": [36, 8], "dtype": "i32", "role": "output"}}}
    trace = _trace([
        {"index": 10, "class": "MVIN", "decoded": {"spad_addr": 0,
                                                   "dram": {"arg_index": 0, "offset": 0},
                                                   "rows": 16, "cols": 8}},
        {"index": 11, "class": "MVIN", "decoded": {"spad_addr": 64,
                                                   "dram": {"arg_index": 0, "offset": 128},
                                                   "rows": 16, "cols": 8}},
    ])
    assert DL.localize(cb, {}, trace, tile_edge=16) is None


def test_selection_is_class_name_agnostic():
    """Op selection keys on the DERIVED decoded fields (out_stride_bytes / dram+rows+cols+readout), NOT a
    class-name literal — so a target whose decode names its output-store ops differently still localizes,
    and the hint reports that target's OWN class name."""
    cb = _cb(64, "i32")                            # 64-wide i32 -> 256 B stride; artifact emitted 64
    trace = _trace([
        {"index": 5, "class": "ST_CFG", "decoded": {"out_stride_bytes": 64}},   # differently-named class
    ])
    f = DL.localize(cb, {}, trace, tile_edge=16)
    assert f is not None and f["class"] == "ST_CFG" and f["field"] == "out_stride_bytes"
    assert f["your_value"] == 64 and f["intended_value"] == 256


def test_fail_closed_when_stride_undecodable():
    # a decode that came back None (not derivable on this RTL) must NOT be flagged.
    cb = _cb(64, "i32")
    trace = _trace([{"index": 5, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": None}}])
    assert DL.localize(cb, {}, trace, tile_edge=16) is None


def test_no_outputs_no_flag():
    assert DL.localize({"tensors": {}}, {}, _trace([]), tile_edge=16) is None


# --- tile-padded row stride (the advisory that fought a correct fix) -------------------------------

def test_tile_padded_row_stride_is_not_flagged():
    """REGRESSION. A harness that drives a tiled array allocates each operand at the TILE-PADDED row
    stride, not the packed one. Measured on ``A7_edge_padding``: a [20, 12] i32 output is declared
    ``T_Y0[512]`` (32 rows x 16 elements), i.e. a 64 B row — but the localizer derived the intent from
    ``cols * elem_bytes`` = 48 B and told the agent its CORRECT 64 was wrong. The agent's own iteration
    notes record it following that advice ("A7 reported 48, not 64"), and a replay on the corrected trace
    re-emitted the same advisory, so it fought the fix every round. Both strides are now accepted."""
    cb = _cb(12, "i32")                            # 12 cols i32: packed 48 B, tile-padded 16*4 = 64 B
    trace = _trace([{"index": 5, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": 64}}])
    assert DL.localize(cb, {}, trace, tile_edge=16) is None
    # ... and the packed row is still accepted too (a caller may legitimately allocate it)
    packed = _trace([{"index": 5, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": 48}}])
    assert DL.localize(cb, {}, packed, tile_edge=16) is None


def test_wrong_element_width_on_a_padded_output_reports_the_padded_stride():
    """A genuinely wrong stride on the SAME padded output still fires — and now names the tile-padded
    row (64 B), not the packed one, so following the advisory produces the stride the harness allocated."""
    cb = _cb(12, "i32")
    trace = _trace([{"index": 5, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": 16}}])  # i8 width
    f = DL.localize(cb, {}, trace, tile_edge=16)
    assert f is not None and f["your_value"] == 16 and f["intended_value"] == 64
    assert "48" in f["basis"]                      # the packed row is named, but is not the intent


def test_underivable_tile_edge_yields_no_finding_and_records_why():
    """FAIL CLOSED. With no derivable tile edge the intended row stride is unknowable, so the localizer
    emits NO finding at all rather than an advisory computed from a guess — and records WHY, so "no
    advisory" is never mistaken for "the advisory looked and found nothing"."""
    cb = _cb(12, "i32")
    trace = _trace([{"index": 5, "class": "CONFIG_ST", "decoded": {"out_stride_bytes": 999}}])
    notes: list[str] = []
    assert DL.localize(cb, {}, trace, target=None, notes=notes) is None
    assert notes and "UNKNOWN tile edge" in notes[0]
    # a target with no fact bundle on disk is the same UNKNOWN, never a baked default
    notes2: list[str] = []
    assert DL.localize(cb, {}, trace, target="__no_such_target__", notes=notes2) is None
    assert notes2 and "UNKNOWN tile edge" in notes2[0]


def test_tile_edge_is_derived_from_the_targets_own_facts():
    """The edge is DERIVED (capability manifest / RTL fact bundle), never a literal in this module. A
    target whose facts publish no array geometry returns UNKNOWN with a reason instead of a number."""
    edge, basis = DL.tile_cols("__no_such_target__")
    assert edge is None and basis
    edge, basis = DL.tile_cols(None)
    assert edge is None and "no target" in basis
