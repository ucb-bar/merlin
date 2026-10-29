"""Raw observations require complete counters and preserve unknown ownership."""

import copy

import pytest

from merlin.runtime.operation_profile import OperationProfileError, summarize

TABLE = [
    {"id": 0, "mlir_op": "tensor.empty"},
    {"id": 1, "mlir_op": "linalg.generic", "body_ops": ["arith.addf"]},
    {"id": 2, "mlir_op": "func.call", "callee": "@opaque_writer"},
]
TRACE = "PROF 0 0 2\nPROF 1 80 2\nPROF 2 20 2\nPROF 3 123 1\nMETRIC prof_total_ticks 223\nMETRIC prof_marks 8\nDONE\n"


def test_zero_ticks_are_observed_and_invocation_gaps_are_separate():
    original = copy.deepcopy(TABLE)
    report = summarize(TABLE, TRACE)
    assert TABLE == original
    assert report["executions"] == 2
    assert report["operation_counter_ticks"] == 100
    assert report["inter_invocation_counter_ticks"] == 123
    assert report["total_counter_ticks"] == 223
    assert sum(report["category_counter_ticks"].values()) == 100
    assert report["category_counter_ticks"]["unclassified:func.call"] == 20
    assert all(row["execution_owner"] == "unknown" for row in report["operations"])
    assert report["unit"] == "raw_counter_ticks" and report["timing_qualified"] is False
    assert not any(key.endswith(("_ms", "_hz", "_cycles")) for key in report)


@pytest.mark.parametrize("old,new", [
    ("PROF 1 80 2\n", ""),
    ("PROF 1 80 2", "PROF 1 80 1"),
    ("PROF 1 80 2", "PROF 1 80 2\nPROF 1 80 2"),
    ("PROF 3 123 1", "PROF -1 123 1"),
    ("PROF 3 123 1", "PROF 4 123 1"),
    ("PROF 3 123 1", "PROF 3 123 2"),
    ("PROF 3 123 1\n", ""),
    ("PROF 1 80 2", "PROF 1 -1 2"),
    ("PROF 1 80 2", "PROF 1 80 0"),
    ("PROF 1 80 2", "PROF 1 80 4294967296"),
    ("PROF 1 80 2", "PROF 1 18446744073709551616 2"),
    ("PROF 1 80 2", "PROF 1 unknown 2"),
    ("PROF 1 80 2", "PROF 1 80"),
    ("prof_total_ticks 223", "prof_total_ticks 100"),
    ("prof_marks 8", "prof_marks 9"),
    ("METRIC prof_marks 8", "METRIC prof_marks 8\nMETRIC prof_marks 8"),
    ("METRIC prof_marks 8\n", ""),
    ("METRIC prof_marks 8", "METRIC prof_marks junk"),
    ("METRIC prof_marks 8", "METRIC prof_marks -1"),
    ("DONE\n", ""),
    ("DONE", "DONE\nPROF 0 1 1"),
    ("DONE", "DONE\nDONE"),
])
def test_partial_corrupt_overflow_and_replayed_traces_refuse(old, new):
    with pytest.raises(OperationProfileError):
        summarize(TABLE, TRACE.replace(old, new))


@pytest.mark.parametrize("table", [[], [TABLE[1]], [{"id": True, "mlir_op": "arith.constant"}],
    [{"id": 0}], [{"id": 0, "mlir_op": "func.call", "function": "nested"}]])
def test_unknown_table_membership_refuses(table):
    with pytest.raises(OperationProfileError, match="operation table"):
        summarize(table, TRACE)


def test_binary_output_payload_cannot_supply_or_corrupt_profiler_records():
    from merlin.runtime.out_bin import binary_console_diagnostics

    payload = b"\0\xff\nPROF -1 999 1\nMETRIC prof_marks 0\nDONE\n"
    checksum = 0xCBF29CE484222325
    for byte in payload:
        checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    frame = (
        f"OUT_BIN_BEGIN v1 out0 1 {len(payload)} 1 u {len(payload)}\n".encode()
        + payload + f"OUT_BIN_END v1 {checksum:016x}\n".encode()
    )
    text = binary_console_diagnostics(frame + TRACE.encode()).decode()
    assert summarize(TABLE, text) == summarize(TABLE, TRACE)
