"""Merlin-FAITHFUL Gemmini codegen: the RoCC sequence emitted from MLIR (llvm.inline_asm),
lowered by merlin's compiler — no C kernel. Certified bit-exact against the reference.
"""
from __future__ import annotations

import pytest

# HW certification (builds + runs spike/verilator RTL) — heavy; deselect with `-m "not slow"` for the
# fast gate. Runs in the full suite (no filter) / nightly.
pytestmark = pytest.mark.slow

from merlin.runtime import reference_outputs, simulate
from merlin.runtime.backends import gemmini_codegen_mlir as gm
from merlin.runtime.backends import gemmini as gem
from merlin.targetgen.eval.gemmini_conformance import build


NONREQUANT = ["C0", "C1", "C4", "C4e", "C5"]
QUANT = ["Q0", "Q1", "Q2", "Q1t"]   # Gemmini i8 readout: float acc_scale (round-near-even) + clamp


def test_kernel_is_mlir_inline_asm_not_c():
    """The Gemmini kernel is MLIR (llvm.inline_asm .insn), not a C kernel."""
    text, _ = gm.emit_kernel_mlir(build("C0"))
    assert "llvm.inline_asm" in text and ".insn r 0x7b" in text
    assert "llvm.func @gemmini_kernel" in text
    assert "#include" not in text and "gemmini_mvin" not in text  # not C / not libgemmini macros


def test_requant_rejected_on_mlir_path():
    """The MLIR-faithful path explicitly refuses requant (not bit-exact; documented)."""
    from merlin.runtime.backends.gemmini_codegen import CodegenError
    cb = build("C0")
    commit = next(c for c in cb["commands"] if c["opcode"] == "COMMIT")
    commit.setdefault("attributes", {})["epilogue"] = ["requant"]
    with pytest.raises(CodegenError):
        gm.emit_kernel_mlir(cb)


try:
    from merlin.llvmlower import toolchain as _tc
    _HAVE_LLVM = _tc.available()
except Exception:  # pragma: no cover
    _HAVE_LLVM = False


@pytest.mark.skipif(not (_HAVE_LLVM and gem.available("spike")),
                    reason="merlin MLIR→LLVM + riscv toolchain unavailable")
def test_rocc_sequence_emitted_from_mlir(tmp_path):
    """Lowering the C0 MLIR kernel emits the full Gemmini RoCC custom-3 sequence (9 instrs)."""
    obj = gm.build_object(build("C0"), tmp_path)
    assert gm.rocc_instruction_count(obj) == 9


@pytest.mark.skipif(not gem.available("spike"), reason="spike-gemmini unavailable")
@pytest.mark.parametrize("rung", NONREQUANT + QUANT)
def test_gemmini_mlir_spike_bitexact(rung, tmp_path):
    """The MLIR-faithful Gemmini kernel is bit-exact on spike-gemmini (bootstrap).

    Includes the quantized rungs (Q*): Gemmini float acc_scale on the i8 readout, incl. a
    non-power-of-two scale (Q2) that exercises the float multiply, not just an exponent shift."""
    cb = build(rung)
    res = gm.run_on_spike(cb, workdir=tmp_path, simulator="spike", timeout=300)
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb) == simulate(cb)["outputs"]
    assert res["oracle"]["derived_from_rtl"] is False
    assert res["path"] == "mlir_inline_asm_rocc"


@pytest.mark.skipif(not gem.available("verilator"), reason="Gemmini Verilator sim unavailable")
@pytest.mark.parametrize("rung", NONREQUANT + QUANT)
def test_gemmini_mlir_verilator_cert(rung, tmp_path):
    """RTL certification of the MLIR-faithful Gemmini battery (derived_from_rtl, three-way).

    The Q* rungs certify Gemmini's float acc_scale requant (round-near-even + i8 clamp) — the
    useful-quantized-layer slice — bit-exact against merlin's reference on the RTL oracle."""
    cb = build(rung)
    res = gm.run_on_spike(cb, workdir=tmp_path, simulator="verilator", timeout=900)
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb) == simulate(cb)["outputs"]
    assert res["oracle"]["derived_from_rtl"] is True
    assert res["path"] == "mlir_inline_asm_rocc"
