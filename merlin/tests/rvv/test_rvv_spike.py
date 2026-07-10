"""Phase-2 milestone: the full core-dialect pipeline executes on spike as a
multicore RVV CPU, and the parsed outputs equal the independent reference.

Codegen-only tests run everywhere; compile/run tests auto-skip without the chipyard
toolchain (set MERLIN_CHIPYARD, default /path/to/chipyard).
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

HARTS = 4


def _toolchain():
    from merlin.runtime.backends import spike

    return spike.available()


@pytest.fixture(scope="module")
def saturn_lowered():
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    return lower_repeated_rhs_matmul(reuse=4, m=16, k=24, n=20, target="saturn")


def test_saturn_pipeline_descends_and_simulates(saturn_lowered):
    """The saturn descent verifies and the Python engine remains the oracle."""
    from merlin.xdsl_dialects.lowering import execute

    for mod in saturn_lowered.modules():
        mod.verify()
    names = {op.name for op in saturn_lowered.target_module.walk()} - {
        "builtin.module", "func.func", "func.return"}
    assert names == {"saturn.pack", "saturn.matmul", "saturn.commit", "saturn.release"}
    cb = saturn_lowered.command_buffer
    assert cb["target"] == "saturn"
    assert cb["backend"] == "baremetal"
    assert execute(saturn_lowered)["correct"] is True


def test_rvv_codegen_emits_real_driver(saturn_lowered):
    """Codegen is verifiable without the toolchain: real data, kernel calls, barriers."""
    from merlin.runtime.backends.rvv_codegen import generate_driver

    src = generate_driver(saturn_lowered.command_buffer, nharts=HARTS)
    assert "merlin_rvv_matmul_i8" in src
    assert "row_lo(hart" in src                # multicore partitioning
    assert src.count("barrier();") >= 8        # per-command sync
    assert "static const int8_t T_W[" in src   # embedded real tensor data
    assert "csrr %0, mcycle" in src


@pytest.mark.skipif(not _toolchain(), reason="chipyard toolchain/spike not available")
def test_full_pipeline_executes_on_spike_multicore(saturn_lowered, tmp_path):
    """contract->schedule->interface->saturn->runtime->cb -> RVV -> spike -p4."""
    from merlin.runtime import reference_outputs, simulate
    from merlin.runtime.backends import spike

    cb = saturn_lowered.command_buffer
    res = spike.run_command_buffer(cb, harts=HARTS, workdir=tmp_path)
    # The correctness gate: spike outputs == independent reference recomputation.
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb)
    # And identical to the Python simulator (three-way agreement).
    assert res["outputs"] == simulate(cb)["outputs"]
    m = res["metrics"]
    assert m["cycles"] > 0                      # real mcycle delta
    assert m["pack_count"] == 1
    assert m["resident_hits"] == 4
    assert m["accumulator_commits"] == 4
    assert m["evictions"] == 1
    assert m["target_specific"]["harts"] == HARTS


@pytest.mark.skipif(not _toolchain(), reason="chipyard toolchain/spike not available")
def test_spike_runs_full_epilogue_buffer(tmp_path):
    """bias_add + requant + relu + saturating i8 on spike == engine reference."""
    from merlin.runtime.backends import spike

    tensors = {"W": {"shape": [8, 6], "dtype": "i8", "role": "weight"},
               "bias": {"shape": [6], "dtype": "i32", "role": "bias"},
               "A0": {"shape": [5, 8], "dtype": "i8", "role": "input"}}
    cb = {"abi_version": "0.1", "target": "saturn", "backend": "baremetal",
          "tensors": tensors, "params": {"requant_shift": 4},
          "commands": [
              {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
               "attributes": {"layout": "packed_rhs"}},
              {"opcode": "MATMUL_RESIDENT",
               "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
              {"opcode": "COMMIT",
               "operands": {"src": "acc0", "dst": "Y0", "bias": "bias"},
               "attributes": {"epilogue": ["bias_add", "requant", "relu"],
                              "requant_shift": 4, "output_dtype": "i8"}},
              {"opcode": "EVICT", "operands": {"handle": "W_res"}},
          ]}
    res = spike.run_command_buffer(cb, harts=2, workdir=tmp_path)
    assert res["correct"] is True


@pytest.mark.skipif(not _toolchain(), reason="chipyard toolchain/spike not available")
def test_single_vs_multi_hart_outputs_identical(tmp_path):
    """Parallelization must never change results."""
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul
    from merlin.runtime.backends import spike

    cb = lower_repeated_rhs_matmul(reuse=2, m=8, k=12, n=10,
                                   target="saturn").command_buffer
    one = spike.run_command_buffer(cb, harts=1, workdir=tmp_path / "p1")
    four = spike.run_command_buffer(cb, harts=4, workdir=tmp_path / "p4")
    assert one["correct"] and four["correct"]
    assert one["outputs"] == four["outputs"]


def test_saturn_targetgen_plans_validate(tmp_path):
    """TargetGen synthesizes the curated saturn plans and they pass the schemas."""
    from merlin.targetgen import pipeline as tg

    result = tg.build("saturn", out=tmp_path / "saturn", emit=["contract-only"])
    assert result.schema_problems == []
    tc = result.plans["target_contract"]
    assert tc["name"] == "saturn"
    assert "rvv" in tc["features"]
    assert tc["runtime"]["backends"] == ["simulator", "baremetal", "vcs", "zephyr"]
    dp = result.plans["dialect_plan"]
    assert {r["from"]: r["to"] for r in dp["lowering"]} == {
        "interface.resident_pack": "saturn.pack",
        "interface.matmul": "saturn.matmul",
        "interface.commit": "saturn.commit",
        "interface.resident_evict": "saturn.release",
    }
