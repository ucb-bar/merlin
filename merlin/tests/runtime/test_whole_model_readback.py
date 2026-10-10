"""Full readback and group profiling for whole-model images: generation, transport and parsing."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from merlin.runtime import out_bin
from merlin.runtime import whole_model_readback as W


def _frame(name: str, array: np.ndarray, width: int, signed: bool) -> bytes:
    raw = array.tobytes()
    rows = int(np.prod(array.shape[:-1])) if array.ndim > 1 else 1
    cols = int(array.shape[-1]) if array.ndim else 1
    digest = 0xCBF29CE484222325
    for byte in raw:
        digest = ((digest ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    head = f"OUT_BIN_BEGIN v1 {name} {rows} {cols} {width} {'s' if signed else 'u'} {len(raw)}\n".encode()
    return head + raw + f"OUT_BIN_END v1 {digest:016x}\n".encode()


def test_every_result_round_trips_bit_for_bit_in_its_own_dtype():
    logits = (np.arange(24, dtype=np.float32).reshape(2, 12) - 7.5) / 3
    cache = np.arange(-6, 6, dtype=np.int64).reshape(3, 4)
    mask = np.array([[1, 0, 1]], dtype=np.uint8)
    specs = [([2, 12], "f32"), ([3, 4], "i64"), ([1, 3], "i1")]
    console = (
        b"booting\n"
        + _frame("out0", logits, 4, False)
        + _frame("out1", cache, 8, True)
        + _frame("out2", mask, 1, False)
        + b"METRIC cycles 10\nMETRIC build_hash abc123\nMETRIC memref_rank_mismatch 0\nDONE\n"
    )
    outputs, _metrics, frames = out_bin.parse_binary_console_details(console)
    decoded = W.decode_outputs(outputs, frames, specs)
    assert decoded[0].dtype == np.float32 and np.array_equal(decoded[0].view(np.uint32), logits.view(np.uint32))
    assert np.array_equal(decoded[1], cache) and decoded[2].tolist() == [[1, 0, 1]]


def test_a_missing_or_reformatted_result_is_refused():
    specs = [([2], "f32"), ([2], "f32")]
    one = _frame("out0", np.array([[1.0, 2.0]], dtype=np.float32), 4, False) + b"DONE\n"
    outputs, _m, frames = out_bin.parse_binary_console_details(one)
    with pytest.raises(W.ReadbackError, match="no complete frame for result 1"):
        W.decode_outputs(outputs, frames, specs)
    wrong = _frame("out0", np.array([[1, 2]], dtype=np.int32), 4, True) + b"DONE\n"
    outputs, _m, frames = out_bin.parse_binary_console_details(wrong)
    with pytest.raises(W.ReadbackError, match="format differs"):
        W.decode_outputs(outputs, frames, [([2], "f32")])


def test_the_output_table_states_every_result():
    header = W.render_outputs_header([([2, 3], "f32"), ([5], "i64"), ([], "i32")])
    assert '{"out0", "out1", "out2"}' in header
    assert "MERLIN_OUTPUT_BYTES[3] = {24ULL, 40ULL, 4ULL}" in header
    assert "MERLIN_OUTPUT_SIGN[3] = {'u', 's', 's'}" in header
    with pytest.raises(W.ReadbackError, match="cannot carry"):
        W.render_outputs_header([([2], "complex64")])


_ROUTED = [
    {"symbol": "dev_0", "group": 4},
    {"symbol": "dev_1", "group": 7},
    {"symbol": "dev_0", "group": 9},
]
_SIGS = {"dev_0": (16, 32, 64), "dev_1": (2, 8, 8, 8)}


def test_only_referenced_group_symbols_are_wrapped_under_the_name_the_model_calls():
    callees = W.callees(_ROUTED, _SIGS, {"dev_0", "_mlir_ciface_dev_1", "memcpy"})
    assert [(c.symbol, c.abi, c.rank) for c in callees] == [
        ("dev_0", "expanded", 2),
        ("_mlir_ciface_dev_1", "c_interface", 3),
    ]
    assert W.wrap_flags(callees) == ["-Wl,--wrap=dev_0", "-Wl,--wrap=_mlir_ciface_dev_1"]
    with pytest.raises(W.ReadbackError, match="under no known name"):
        W.callees(_ROUTED, _SIGS, {"dev_0"})


def test_the_generated_profile_compiles_and_names_calls_by_program_order():
    callees = W.callees(_ROUTED, _SIGS, {"dev_0", "_mlir_ciface_dev_1"})
    text = W.render_group_profile(callees, W.group_names(_ROUTED))
    assert '{"g4_dev_0", "g7_dev_1", "g9_dev_0"}' in text
    assert "merlin_group_name_callee[] = {0, 1, 0}" in text
    assert "__wrap_dev_0(" in text and "__real__mlir_ciface_dev_1(result, a, b, c)" in text
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no host C compiler to syntax-check the generated unit")
    proc = subprocess.run([compiler, "-fsyntax-only", "-x", "c", "-"], input=text, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


def test_the_profile_lines_parse_into_groups_and_gaps():
    console = "\n".join(
        [
            "GROUP_ID 0 g4_dev_0",
            "GROUP_ID 1 g7_dev_1",
            "GAP 0 100",
            "GROUP 0 900",
            "GAP 1 50",
            "GROUP 1 400",
            "GAP 2 25",
            "METRIC group_calls 2",
            "METRIC group_calls_dropped 0",
        ]
    )
    profile = W.parse_group_profile(console)
    assert profile["calls"] == 2 and profile["tail_gap"] == 25
    assert profile["groups"][1] == {"index": 1, "name": "g7_dev_1", "cycles": 400, "gap_before": 50}
    assert W.parse_group_profile("METRIC cycles 3\nDONE") is None
    with pytest.raises(W.ReadbackError, match="incomplete"):
        W.parse_group_profile("GROUP_ID 0 a\nGROUP 0 5\n")
