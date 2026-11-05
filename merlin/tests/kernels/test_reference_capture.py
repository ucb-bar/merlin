"""Actual native issue capture and complete symbolic tile contraction recovery."""

import copy
import shutil

import pytest

from merlin.kernels.reference_capture import capture_native_function, source_function
from merlin.kernels.tile_reference import recover_tile_region


def test_generated_declarations_are_source_syntax_not_function_names():
    pytest.importorskip("tree_sitter_c")
    source = b'''// arbitrary(
// A : i8[1, 2] @DRAM,
// B : i8[2, 2] @DRAM,
// O : i32[1, 2] @DRAM
// )
void arbitrary(void *ctxt, const int8_t *A, const int8_t *B, int32_t *O) { ISSUE(0,A,B,7); FINISH(); }
'''
    function = source_function(source, "arbitrary")
    assert function["declarations"]["B"] == dict(dtype="i8", shape=[2, 2])
    assert function["declarations"]["O"] == dict(dtype="i32", shape=[1, 2])


def test_native_capture_executes_real_loop_and_original_header_operand_encoding(tmp_path):
    pytest.importorskip("tree_sitter_c")
    if shutil.which("cc") is None:
        pytest.skip("native compiler unavailable")
    header = tmp_path / "selected.h"
    header.write_text('''#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#define ISSUE(slot,x,y,f) do {} while(0)
#define FINISH() do {} while(0)
#define MOVE(x,y) ISSUE(2,x,((uint64_t)(y)<<9)|3,7)
''')
    source = b'''// arbitrary(
// A : i8[1, 4] @DRAM,
// O : i32[1, 4] @DRAM
// )
void arbitrary(void *ctxt,const int8_t *A,int32_t *O) {
 for(int i=0;i<4;i++) MOVE(A+i,i);
 FINISH();
}
'''
    capture = capture_native_function(source, "arbitrary", settings=dict(
        header_root=str(tmp_path), header="selected.h", sink_macro="ISSUE", fence_macro="FINISH",
        probe_expressions={"parameter": "sizeof(int32_t)"}), workspace=tmp_path / "capture")
    commands = [row for row in capture["records"] if row["kind"] == "command"]
    assert len(commands) == 4 and [row["rs2"] for row in commands] == [3, 515, 1027, 1539]
    assert [row["rs1"] - commands[0]["rs1"] for row in commands] == list(range(4))
    assert any(pin["kind"] == "source_header" and pin["uri"] == str(header) for pin in capture["artifacts"])
    assert not capture["native_timing"]
    with pytest.raises(ValueError, match="failed or exceeded"):
        capture_native_function(source, "arbitrary", settings=dict(
            header_root=str(tmp_path), header="selected.h", sink_macro="ISSUE", fence_macro="FINISH",
            max_commands=2), workspace=tmp_path / "bounded")


def _tiles():
    declarations = {"A": dict(dtype="i8", shape=[1, 2]), "W": dict(dtype="i8", shape=[2, 2]),
                    "O": dict(dtype="i32", shape=[1, 2])}
    events = [dict(kind="load", tensor="A", offset=0, stride=2, rows=1, cols=2,
                   tile_cols=1, address=0, block_stride=1)]
    for k in range(2):
        events.append(dict(kind="load", tensor="W", offset=2*k, stride=2, rows=1, cols=2,
                           tile_cols=1, address=100, block_stride=1))
        for n in range(2):
            events.extend([dict(kind="bind", weight=100+n, address=200+n, accumulate=bool(k)),
                           dict(kind="compute", lhs=k, rows=1, cols=1)])
    events.append(dict(kind="store", tensor="O", offset=0, stride=8, address=200,
                       rows=1, cols=2, tile_cols=1, block_stride=1,
                       numeric=dict(accumulator_dtype="i32", bias="none", scale=1,
                                    activation="none", rounding="none", saturation="none")))
    return events, declarations


def test_tile_algebra_recovers_complete_ordered_dense_region_and_scoped_reuse():
    events, declarations = _tiles()
    result = recover_tile_region(events, declarations, evidence_id="synthetic:capture")
    assert result["problem"]["m"] == 1 and result["problem"]["k"] == result["problem"]["n"] == 2
    assert result["semantic_region"].complete and len(result["semantic_region"].operations) == 1
    assert result["tile_dataflow"]["lhs_retained_complete_invocation"]
    assert result["tile_dataflow"]["loaded_bytes"] == dict(A=2, W=4)
    assert all(row["reduction_segments"] == 2 for row in result["tile_dataflow"]["output_tiles"])


@pytest.mark.parametrize("corruption", ["reduction", "output_coverage", "readout_stride", "uninitialized"])
def test_tile_algebra_refuses_incomplete_or_wrong_mathematics(corruption):
    events, declarations = _tiles()
    events = copy.deepcopy(events)
    if corruption == "reduction":
        events[3]["lhs"] = 1
    elif corruption == "output_coverage":
        events[-1]["cols"] = 1
    elif corruption == "readout_stride":
        events[-1]["stride"] = 3
    else:
        events[2]["accumulate"] = True
    with pytest.raises(ValueError):
        recover_tile_region(events, declarations, evidence_id="synthetic:capture")
