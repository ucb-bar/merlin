"""The automated op x datatype optimization sweep (the framework, not hand-fixing) — scalar-fail gate
+ parallel matrix orchestration, mock beam (no board)."""
from __future__ import annotations

from merlin.common.paths import merlin_dir
from merlin.mining.op_sweep import OpCell, is_scalar_kernel, run_op_sweep


def test_is_scalar_kernel_flags_scalar_and_passes_rvv():
    # a real XNNPACK RVV kernel objdump -> NOT scalar (has vsetvli/vfmacc/...)
    rvv = (merlin_dir() / "tests" / "data" / "cca_asm" / "xnnpack_f32_gemm_rvv.objdump").read_text()
    assert is_scalar_kernel(rvv) is False
    # a hand scalar loop (fmadd.s/flw/fsw, NO vector ops) -> scalar FAIL. Real objdump -d format
    # (address : hexword : mnemonic : operands) so rvv_audit's line parser recognizes each insn.
    scalar = (
        "0000000000001000 <gemm_scalar>:\n"
        "    1000:\t00052507          \tflw\tfa0,0(a0)\n"
        "    1004:\t0005a587          \tflw\tfa1,0(a1)\n"
        "    1008:\t68b57543          \tfmadd.s\tfa2,fa0,fa1,fa2\n"
        "    100c:\t00c52027          \tfsw\tfa2,0(a2)\n"
        "    1010:\t00008067          \tret\n"
    )
    assert is_scalar_kernel(scalar) is True


def test_run_op_sweep_fans_out_and_ranks_on_attainment(tmp_path):
    # a mock beam: attainment scales with the cell's shape (bigger regime -> "closer to XNNPACK"),
    # so the sweep collects a CellResult per cell without touching the board.
    def mock_beam(*, seed_pkg, model_dir, expert_objdump, op, dtype, shape_regime, targets,
                  width, depth, top_k, expert_wall_ns):
        att = 0.4 if "128" in shape_regime else 0.9
        return {"best": {"run_id": f"{op}_{dtype}_win", "attainment_vs_expert": att,
                         "speedup": 1.5, "gate_ok": True},
                "parent_run_dir": None, "nodes": [], "deferred": []}

    cells = [
        OpCell(op="matmul", dtype="f32", shape_regime="square_128", workload_dir=tmp_path / "f32",
               expert_objdump=merlin_dir() / "tests/data/cca_asm/xnnpack_f32_gemm_rvv.objdump",
               expert_wall_ns=9424),
        OpCell(op="matmul", dtype="int8", shape_regime="square_64", workload_dir=tmp_path / "i8",
               expert_objdump=merlin_dir() / "tests/data/cca_asm/xnnpack_f32_gemm_rvv.objdump",
               expert_wall_ns=1544),
    ]
    results = run_op_sweep(cells, beam_fn=mock_beam, max_workers=2)
    assert len(results) == 2
    by = {r.cell_key: r for r in results}
    assert by["matmul:f32:square_128"].attainment_vs_expert == 0.4
    assert by["matmul:int8:square_64"].attainment_vs_expert == 0.9
    assert all(r.gate_ok for r in results)


def test_sweep_never_aborts_on_one_cell_error(tmp_path):
    def boom(**kw):
        raise RuntimeError("cell blew up")
    cells = [OpCell(op="matmul", dtype="f32", shape_regime="s", workload_dir=tmp_path,
                    expert_objdump=tmp_path / "x.objdump")]
    results = run_op_sweep(cells, beam_fn=boom)
    assert len(results) == 1 and results[0].gate_ok is False and "error" in results[0].note
