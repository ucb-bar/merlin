"""The answer-free structural lints in rtl_checks: store coverage, ragged-extent legalization, conv depth.

All three are ADVISORY (severity ``warn``, so they can never move the report's verdict to ``reject`` and
can never cost a run its oracle) and all three derive every bound from either the capsule's DECLARED
shapes or the RTL-derived array geometry. Nothing here reads a golden.

Each lint is exercised in both directions — a conformant stream passes, and the specific defect it exists
for is DETECTED — plus the fail-closed direction, because a check that could not run must never report
success (a recurring defect in this repo).
"""
from __future__ import annotations

import copy

from merlin.targetgen import rtl_checks as RC

#: RTL-shaped facts, passed as the explicit override so these tests exercise the CHECKS rather than the
#: fact extractor. The mesh here plays the role of "whatever the target's array turns out to be": every
#: bound below is computed from it, never from the number.
_FACTS = {"mesh": [16, 16], "scratchpad_bytes": 262144, "legal_funct": None,
          "custom_opcode": None, "funct3": None, "from": "test override"}
_EDGE = _FACTS["mesh"][0]

#: The three lints under test (the report also carries the pre-existing checks).
_LINTS = {"T0.output_store_coverage", "T0.extent_tile_legalization", "T0.conv_lowering"}


def _matmul_capsule(M=16, K=16, N=16, out_dtype="i32", epilogue=None):
    return {"name": "unit", "inputs": [
                {"name": "W", "role": "weight", "shape": [K, N], "dtype": "i8"},
                {"name": "A0", "role": "input", "shape": [M, K], "dtype": "i8"}],
            "operation": {"op": "matmul", "attributes": {
                "lhs": "A0", "weight": "W", "out": "Y0",
                "epilogue": epilogue or [], "output_dtype": out_dtype}}}


def _conv_capsule(kh=3, kw=3, ci=4, co=8, H=8, W=8):
    return {"name": "unit", "inputs": [
                {"name": "W", "role": "weight", "shape": [kh * kw * ci, co], "dtype": "i8"},
                {"name": "IFM", "role": "input", "shape": [1, H, W, ci], "dtype": "i8"}],
            "operation": {"op": "conv2d", "attributes": {
                "ifm": "IFM", "weight": "W", "out": "Y0", "ci": ci, "kh": kh, "kw": kw,
                "stride": [1, 1], "padding": [0, 0, 0, 0], "dilation": [1, 1], "layout": "nhwc",
                "epilogue": [], "output_dtype": "i32"}}}


def _trace(stores, *, n_compute=1, n_mvin=2, out_stride_bytes=64, out_arg=2):
    """A well-formed stream: FENCE-bracketed, configs before use, PRELOAD before COMPUTE. ``stores`` is a
    list of ``(byte_offset, rows, cols)`` MVOUTs addressing kernel argument ``out_arg``."""
    ins = [{"index": 0, "class": "FENCE", "funct": None, "decoded": {}},
           {"index": 1, "class": "CONFIG_EX", "funct": 0, "decoded": {"subtype": "EX"}},
           {"index": 2, "class": "CONFIG_LD", "funct": 0, "decoded": {"subtype": "LD", "stride": 16}},
           {"index": 3, "class": "CONFIG_ST", "funct": 0,
            "decoded": {"subtype": "ST", "out_stride_bytes": out_stride_bytes}}]
    for a in range(n_mvin):
        ins.append({"index": len(ins), "class": "MVIN", "funct": 2,
                    "decoded": {"rows": _EDGE, "cols": _EDGE, "spad_addr": a * _EDGE,
                                "dram": {"kind": "argbase", "arg_index": a % 2, "offset": 0}}})
    for _ in range(n_compute):
        ins.append({"index": len(ins), "class": "PRELOAD", "funct": 6, "decoded": {}})
        ins.append({"index": len(ins), "class": "COMPUTE_PRELOADED", "funct": 4, "decoded": {}})
    for (off, rows, cols) in stores:
        ins.append({"index": len(ins), "class": "MVOUT", "funct": 3,
                    "decoded": {"rows": rows, "cols": cols,
                                "dram": {"kind": "argbase", "arg_index": out_arg, "offset": off}}})
    ins.append({"index": len(ins), "class": "FENCE", "funct": None, "decoded": {}})
    return {"instructions": ins, "abi": {}}


def _check(rep, cid):
    return next(c for c in rep.checks if c.id == cid)


def _screen(trace, capsule):
    return RC.screen(trace, capsule, _FACTS, target="unit-test-target")


# ------------------------------------------------------------------- output store coverage
def test_a_fully_covered_output_passes():
    cap = _matmul_capsule()
    rep = _screen(_trace([(0, 16, 16)]), cap)          # 16x16 i32, row pitch 64 bytes
    c = _check(rep, "T0.output_store_coverage")
    assert c.status == "pass", c.message
    assert c.severity == "warn"                        # advisory: never gates


def test_a_declared_output_with_no_store_is_reported_as_a_dropped_store():
    """The measured failure: the numeric plane reports mismatches with max_abs_error 0 because the bytes
    were never produced. Detected here with no simulation at all."""
    rep = _screen(_trace([]), _matmul_capsule())
    c = _check(rep, "T0.output_store_coverage")
    assert c.status == "fail"
    assert "NO store" in c.message and "Y0" in c.message
    assert "DROPPED or MIS-ADDRESSED" in c.message
    assert c.severity == "warn"


def test_these_lints_alone_never_push_the_report_to_reject():
    """Advisory means advisory: a stream in which ONLY these three checks fail must stay at `warn`, so a
    caller that skips an expensive oracle on `reject` can never lose an oracle run over one of them."""
    cap = _matmul_capsule(M=16, K=16, N=12)
    rep = _screen(_trace([(0, 16, 16)], out_stride_bytes=48), cap)   # full tile over a 12-wide extent
    failed = {c.id for c in rep.checks if c.status == "fail"}
    assert failed and failed <= _LINTS, f"other checks also failed: {failed - _LINTS}"
    assert rep.verdict == "warn"
    assert rep.n_error == 0


def test_a_store_to_the_wrong_kernel_argument_is_reported():
    rep = _screen(_trace([(0, 16, 16)], out_arg=1), _matmul_capsule())
    c = _check(rep, "T0.output_store_coverage")
    assert c.status == "fail" and "kernel argument #2" in c.message


def test_a_missing_tail_band_names_the_uncovered_rows():
    """M=20 over a 16-row edge: committing only the first whole tile leaves rows 16..19 unwritten."""
    cap = _matmul_capsule(M=20, K=16, N=12)
    rep = _screen(_trace([(0, 16, 12)], out_stride_bytes=48), cap)
    c = _check(rep, "T0.output_store_coverage")
    assert c.status == "fail"
    assert "192 of 240" in c.message
    assert c.evidence["Y0"]["uncovered_rows"] == [16, 17, 18, 19]


def test_a_full_tile_store_over_a_ragged_extent_is_reported_as_an_overrun():
    """N=12 committed as a whole 16-wide tile writes 4 columns the output buffer does not own."""
    cap = _matmul_capsule(M=16, K=16, N=12)
    rep = _screen(_trace([(0, 16, 16)], out_stride_bytes=48), cap)
    c = _check(rep, "T0.output_store_coverage")
    assert c.status == "fail" and "PAST the declared extent" in c.message


def test_many_short_bands_are_accepted_when_they_cover_the_extent():
    """Legalization is judged by COVERAGE, not by loop shape: 15 one-row stores cover a 15-row output."""
    cap = _matmul_capsule(M=15, K=16, N=15)
    stores = [(r * 60, 1, 15) for r in range(15)]
    rep = _screen(_trace(stores, out_stride_bytes=60), cap)
    assert _check(rep, "T0.output_store_coverage").status == "pass"
    assert _check(rep, "T0.extent_tile_legalization").status == "pass"


def test_a_store_whose_address_could_not_be_decoded_makes_the_answer_unknown_not_absent():
    """"Your kernel dropped this store" and "our decoder could not read your addresses" must never be
    collapsed. A store with no DRAM provenance at all might BE the missing one."""
    t = _trace([(0, 16, 16)])
    for i in t["instructions"]:
        if i["class"] == "MVOUT":
            i["decoded"] = {}                          # the decoder resolved no address operand
    c = _check(_screen(t, _matmul_capsule()), "T0.output_store_coverage")
    assert c.status == "skipped" and "UNKNOWN" in c.message
    assert "no decodable DRAM address" in c.message


def test_coverage_is_unknown_not_passed_when_the_row_pitch_was_never_configured():
    """FAIL CLOSED: a check that could not run must not report success."""
    t = _trace([(0, 16, 16)])
    for i in t["instructions"]:                        # drop the store configuration entirely
        if i["class"] == "CONFIG_ST":
            i["decoded"] = {"subtype": "ST"}
    c = _check(_screen(t, _matmul_capsule()), "T0.output_store_coverage")
    assert c.status == "skipped" and "UNKNOWN" in c.message


def test_multiple_declared_outputs_are_each_checked():
    """A residency capsule commits one tensor per declared matmul; a dropped second output is named."""
    cap = {"name": "unit", "inputs": [
               {"name": "W", "role": "weight", "shape": [16, 16], "dtype": "i8"},
               {"name": "A0", "role": "input", "shape": [16, 16], "dtype": "i8"},
               {"name": "A1", "role": "input", "shape": [16, 16], "dtype": "i8"}],
           "operation": {"op": "resident_reuse", "attributes": {
               "weight": "W",
               "matmuls": [{"lhs": "A0", "out": "Y0", "epilogue": [], "output_dtype": "i32"},
                           {"lhs": "A1", "out": "Y1", "epilogue": [], "output_dtype": "i32"}]}}}
    both = _trace([(0, 16, 16)], out_arg=3)
    both["instructions"].insert(-1, {"index": 99, "class": "MVOUT", "funct": 3,
                                     "decoded": {"rows": 16, "cols": 16,
                                                 "dram": {"kind": "argbase", "arg_index": 4,
                                                          "offset": 0}}})
    assert _check(_screen(both, cap), "T0.output_store_coverage").status == "pass"
    only_first = _trace([(0, 16, 16)], out_arg=3)
    c = _check(_screen(only_first, cap), "T0.output_store_coverage")
    assert c.status == "fail" and "['Y1']" in c.message and "1 of 2 declared outputs" in c.message


# ---------------------------------------------------------------- ragged-extent legalization
def test_extents_that_divide_the_derived_edge_pass():
    c = _check(_screen(_trace([(0, 16, 16)]), _matmul_capsule()), "T0.extent_tile_legalization")
    assert c.status == "pass" and "whole multiple" in c.message


def test_a_ragged_column_extent_with_no_tail_store_is_reported_against_the_derived_edge():
    cap = _matmul_capsule(M=16, K=16, N=17)
    rep = _screen(_trace([(0, 16, 16)], out_stride_bytes=68), cap)
    c = _check(rep, "T0.extent_tile_legalization")
    assert c.status == "fail"
    assert "committed columns of Y0 is 17" in c.message
    assert f"tile edge {_EDGE}" in c.message and "arrays[mesh]" in c.message


def test_a_ragged_contraction_whose_tail_is_never_accumulated_is_reported():
    """K=24 over a 16-deep operand tile needs 2 accumulation steps; a stream carrying 1 is missing a
    slice of the sum while every store still covers its output perfectly."""
    cap = _matmul_capsule(M=16, K=24, N=16)
    rep = _screen(_trace([(0, 16, 16)], n_compute=1), cap)
    c = _check(rep, "T0.extent_tile_legalization")
    assert c.status == "fail"
    assert "contraction length is 24" in c.message
    assert "never accumulated" in c.message
    rep_ok = _screen(_trace([(0, 16, 16)], n_compute=2), cap)
    assert _check(rep_ok, "T0.extent_tile_legalization").status == "pass"


def test_legalization_is_skipped_when_the_array_geometry_is_unknown():
    """FAIL CLOSED: an unknown mesh means no edge, and no edge means no verdict — never a pass."""
    facts = dict(_FACTS, mesh=None)
    rep = RC.screen(_trace([(0, 16, 16)]), _matmul_capsule(M=15, K=16, N=17), facts,
                    target="unit-test-target")
    c = _check(rep, "T0.extent_tile_legalization")
    assert c.status == "skipped" and "UNKNOWN" in c.message


# ------------------------------------------------------------------------- conv lowering
def test_a_conv_lowered_as_an_im2col_contraction_passes():
    cap = _conv_capsule()                              # P=36, K=3*3*4=36, Co=8 -> 3*3*1 = 9 steps
    rep = _screen(_trace([(0, 16, 8), (512, 16, 8), (1024, 4, 8)], n_compute=9, out_stride_bytes=32), cap)
    c = _check(rep, "T0.conv_lowering")
    assert c.status == "pass" and "im2col'd contraction of depth 36" in c.message


def test_a_conv_contracted_over_the_channel_depth_alone_is_detected():
    """The defect: a well-formed, fully covering stream that computes a different operation, because the
    Kh*Kw window was never folded into the contraction."""
    cap = _conv_capsule()                              # a ci-deep contraction would take 3*1*1 = 3 steps
    rep = _screen(_trace([(0, 16, 8), (512, 16, 8), (1024, 4, 8)], n_compute=3, out_stride_bytes=32), cap)
    c = _check(rep, "T0.conv_lowering")
    assert c.status == "fail"
    assert "at least 9 compute step(s)" in c.message and "carries 3" in c.message
    assert "RAW channel depth" in c.message
    assert _check(rep, "T0.output_store_coverage").status == "pass"   # the stores are fine; the math is not


def test_a_conv_is_accepted_when_the_target_names_a_fused_conv_loop_and_the_stream_uses_it():
    cap = _conv_capsule()
    t = _trace([(0, 16, 8), (512, 16, 8), (1024, 4, 8)], n_compute=0, out_stride_bytes=32)
    t["instructions"].insert(-1, {"index": 99, "class": "LOOP_CONV", "funct": 15, "decoded": {}})
    monkey = RC._vocabulary_classes
    try:
        RC._vocabulary_classes = lambda target, stems: (
            ({"LOOP_CONV"} if "conv" in stems else {"LOOP_WS", "LOOP_CONV"}), "test vocabulary")
        c = _check(_screen(t, cap), "T0.conv_lowering")
    finally:
        RC._vocabulary_classes = monkey
    assert c.status == "pass" and "fused convolution loop" in c.message


def test_conv_check_is_skipped_when_the_declared_kernel_geometry_is_missing():
    cap = _conv_capsule()
    del cap["operation"]["attributes"]["kh"]
    c = _check(_screen(_trace([(0, 16, 8)], n_compute=1, out_stride_bytes=32), cap), "T0.conv_lowering")
    assert c.status == "skipped"


# ---------------------------------------------------------------------------- answer-freeness
def test_no_finding_carries_anything_but_structure():
    """Every message/evidence these lints produce is derived from declarations, RTL geometry and the
    author's own instruction stream. Guard the boundary explicitly."""
    cases = [(_trace([]), _matmul_capsule()),
             (_trace([(0, 16, 16)], out_stride_bytes=68), _matmul_capsule(M=16, K=16, N=17)),
             (_trace([(0, 16, 8)], n_compute=3, out_stride_bytes=32), _conv_capsule())]
    banned = ("golden", "expected value", "reference output", "answer")
    for trace, cap in cases:
        rep = _screen(trace, cap)
        for c in (c for c in rep.checks if c.id in _LINTS):
            blob = (c.message + " " + str(c.evidence) + " " + str(c.fix_hint)).lower()
            for word in banned:
                assert word not in blob, f"{c.id} leaked {word!r}"


# ------------------------------------------------- fail-closed on an unrecognized compute vocabulary
def _rename_compute(trace, new_class="MMA"):
    """A trace whose matrix-compute steps carry a class name this module's vocabulary does not know —
    what a target that spells its compute differently looks like to these checks."""
    t = copy.deepcopy(trace)
    for i in t["instructions"]:
        if i["class"] in ("COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"):
            i["class"] = new_class
    return t


def test_a_ragged_contraction_is_unknown_not_failed_when_no_compute_class_is_recognized():
    """The count bound has nothing to count on a target that names its compute step differently. That is
    a limit of this check, not evidence about the kernel — it must not be reported as a defect."""
    cap = _matmul_capsule(M=16, K=24, N=16)
    t = _rename_compute(_trace([(0, 16, 16)], n_compute=1))
    c = _check(_screen(t, cap), "T0.extent_tile_legalization")
    assert c.status == "skipped"
    assert "no decodable extent" in c.message or "could not be judged" in c.message


def test_conv_depth_is_unknown_not_failed_when_no_compute_class_is_recognized():
    cap = _conv_capsule()
    t = _rename_compute(_trace([(0, 16, 8), (512, 16, 8), (1024, 4, 8)], n_compute=3,
                               out_stride_bytes=32))
    c = _check(_screen(t, cap), "T0.conv_lowering")
    assert c.status == "skipped" and "UNKNOWN" in c.message
