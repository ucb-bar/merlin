"""Each level detector must be able to FIRE, and must not fire on a clean program.

Measured 2026-09-04: all 78 emitted command buffers in this corpus produce ZERO findings. That is a
true negative, not a broken detector -- these capsules are single-layer kernels that commit each
accumulator once and never read it back, so there is no inter-layer round trip to find. A detector
whose only evidence is silence on a clean corpus is indistinguishable from one that cannot fire at
all, which is why every pattern here carries a positive control.

The value of these detectors on this corpus is therefore as a REGRESSION guard: a restructuring that
introduces a round trip, re-stages a weight, or leaves a fusable producer unfused will be seen for
free, before any oracle time is spent on it.
"""
from __future__ import annotations

from merlin.perf import structural_levels as SL

STAGE, RELEASE, COMMIT = SL.STAGE_OPCODES[0], SL.RELEASE_OPCODES[0], SL.COMMIT_OPCODES[0]


def _buffer(*commands):
    return {"abi_version": "0.1", "commands": list(commands)}


def _clean():
    """The shape every capsule in this corpus actually emits."""
    return _buffer(
        {"opcode": STAGE, "operands": {"src": "W", "dst": "W_res"},
         "attributes": {"layout": "packed_rhs"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
        {"opcode": COMMIT, "operands": {"src": "acc0", "dst": "Y0"},
         "attributes": {"epilogue": [], "output_dtype": "i32"}},
        {"opcode": RELEASE, "operands": {"handle": "W_res"}},
    )


def test_a_clean_program_reports_nothing():
    report = SL.findings(_clean())
    assert report["status"] == "read"
    assert report["findings"] == []
    assert set(report["by_level"]) == set(SL.LEVELS), "every level is reported, including the zeros"


def test_a_memory_round_trip_fires():
    """A value drained to memory and then read back by a later command."""
    report = SL.findings(_buffer(
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W", "dst": "acc0"}},
        {"opcode": COMMIT, "operands": {"src": "acc0", "dst": "Y0"}, "attributes": {"epilogue": []}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "acc0", "rhs": "W", "dst": "acc1"}},
    ))
    kinds = [f["kind"] for f in report["findings"]]
    assert "memory_round_trip" in kinds
    assert report["by_level"]["L3_inter_layer"] == 1


def test_a_restaged_value_fires():
    """A weight staged, released, and staged again pays the residency cost twice."""
    report = SL.findings(_buffer(
        {"opcode": STAGE, "operands": {"src": "W", "dst": "W_res"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
        {"opcode": RELEASE, "operands": {"handle": "W"}},
        {"opcode": STAGE, "operands": {"src": "W", "dst": "W_res2"}},
    ))
    assert "residency_restaged" in [f["kind"] for f in report["findings"]]
    assert report["by_level"]["L2_intra_layer"] == 1


def test_an_unfused_single_consumer_fires():
    """An empty epilogue whose result is read by exactly the next command."""
    report = SL.findings(_buffer(
        {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "W", "dst": "t0"},
         "attributes": {"epilogue": []}},
        {"opcode": "VECTOR_MAP", "operands": {"src": "t0", "dst": "Y0"}},
    ))
    assert "unfused_single_consumer" in [f["kind"] for f in report["findings"]]
    assert report["by_level"]["L5_fusion"] == 1


def test_a_declared_epilogue_is_not_reported_as_unfused():
    report = SL.findings(_buffer(
        {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "W", "dst": "t0"},
         "attributes": {"epilogue": ["relu"]}},
        {"opcode": "VECTOR_MAP", "operands": {"src": "t0", "dst": "Y0"}},
    ))
    assert report["by_level"]["L5_fusion"] == 0


def test_an_unreadable_buffer_is_unknown_not_clean():
    """Silence must never be reported as efficiency."""
    for bad in (None, {}, {"commands": []}, {"commands": "not a list"}):
        assert SL.findings(bad)["status"] == SL.UNKNOWN
