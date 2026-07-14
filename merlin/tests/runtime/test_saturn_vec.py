"""Saturn-vectors SV1: vector-family ABI expressiveness (Level 0, no toolchain).

Tests that Merlin's command-buffer + reference/simulator express a NON-matmul (vector/SIMD)
workload, and that the semantics are correct against an independent hand-computation.
"""
from __future__ import annotations

import pytest

from merlin.runtime import outputs_match, reference_outputs, simulate
from merlin.runtime.commandbuffer import materialize_inputs
from merlin.targetgen.eval.saturn_vec_conformance import RUNGS, build


def _expected(rung: str, cb: dict) -> dict:
    leaves = materialize_inputs(cb)
    if rung in ("VEC0", "VEC1"):
        x, b = leaves["x"].data, leaves["b"].data
        y = [a + c for a, c in zip(x, b)]
        if rung == "VEC1":
            y = [v if v > 0 else 0 for v in y]
        return {"y": list(y)}
    if rung == "VEC2":
        x, w = leaves["x"].data, leaves["w"].data
        return {"s": [sum(a * c for a, c in zip(x, w))]}
    raise KeyError(rung)


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_vector_abi_expresses_and_matches(rung):
    """L0: the vector cb runs through simulate + reference, and matches an independent compute."""
    cb = build(rung)
    sim = simulate(cb)["outputs"]
    ref = reference_outputs(cb)
    assert outputs_match(sim, ref), f"sim={sim} ref={ref}"
    assert sim == _expected(rung, cb), f"sim={sim} expected={_expected(rung, cb)}"


def test_vector_cb_is_not_matmul_shaped():
    """The vector family uses its own opcodes — not RES_PACK/MATMUL_RESIDENT/COMMIT/EVICT."""
    opcodes = {c["opcode"] for r in RUNGS for c in build(r)["commands"]}
    assert opcodes <= {"VECTOR_MAP", "VREDUCE"}
    assert not (opcodes & {"RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"})


# --- RVV codegen + oracle ---
from merlin.runtime.backends import saturn_vec  # noqa: E402
from merlin.runtime.backends.saturn_vec_codegen import generate_driver  # noqa: E402


def test_vector_codegen_emits_rvv():
    """The vector codegen emits real RVV intrinsics (not matmul, not tiled_matmul_auto)."""
    src = generate_driver(build("VEC2"))
    for needle in ("__riscv_vsetvl_e32m1", "__riscv_vle32_v_i32m1", "__riscv_vmul_vv_i32m1",
                   "__riscv_vredsum", "VOUT", "DONE"):
        assert needle in src, f"vector driver missing {needle!r}"


@pytest.mark.skipif(not saturn_vec.available(), reason="spike rv64gcv toolchain unavailable")
@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_vector_spike_rv64gcv_cert(rung, tmp_path):
    """RVV vector kernel runs bit-exact on spike rv64gcv vs the Merlin reference."""
    cb = build(rung)
    res = saturn_vec.run_command_buffer(cb, workdir=tmp_path, timeout=180)
    assert res["correct"] is True
    assert res["outputs"] == reference_outputs(cb)
    assert res["oracle"]["kind"] == "spike_rv64gcv"


# --- MLIR-FAITHFUL path: compute expressed in MLIR, lowered by merlin's compiler (no C kernel) ---
def test_vector_mlir_emitter_is_linalg_not_c():
    """The merlin-faithful emitter produces MLIR (linalg), not C."""
    from merlin.runtime.backends.saturn_vec_mlir import emit_mlir
    text, inputs, out = emit_mlir(build("VEC2"))
    assert "linalg.generic" in text and "func.func @forward" in text
    assert 'iterator_types = ["reduction"]' in text  # the reduce expressed in MLIR
    assert "htif" not in text and "riscv_vector.h" not in text  # not a C kernel


try:
    from merlin.llvmlower import toolchain as _tc
    _HAVE_LLVM = _tc.available()
except Exception:  # pragma: no cover
    _HAVE_LLVM = False

# Disassembly-based tests also shell out to riscv64-unknown-elf-objdump (the chipyard toolchain, via
# spike.gcc_path()). Guard on it too — otherwise, with clang present but chipyard unset, they don't
# skip: they exec the /path/to/chipyard placeholder and raise FileNotFoundError.
try:
    from merlin.runtime.backends import spike as _spike
    _HAVE_RISCV_OBJDUMP = _spike.gcc_path().with_name("riscv64-unknown-elf-objdump").is_file()
except Exception:  # pragma: no cover
    _HAVE_RISCV_OBJDUMP = False


@pytest.mark.skipif(not _HAVE_LLVM, reason="merlin MLIR→LLVM toolchain (clang) unavailable")
@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_vector_mlir_host_cert(rung, tmp_path):
    """Vector compute lowered through merlin's real MLIR→LLVM compiler is bit-exact on host."""
    from merlin.runtime.backends import saturn_vec_mlir as vm
    res = vm.run_host(build(rung), workdir=tmp_path)
    assert res["correct"] is True
    assert res["oracle"]["kind"] == "merlin_mlir_host"


@pytest.mark.skipif(not (_HAVE_LLVM and _HAVE_RISCV_OBJDUMP),
                    reason="merlin MLIR→LLVM (clang) + riscv objdump toolchain unavailable")
@pytest.mark.parametrize("rung,expect", [("VEC0", "vadd.vv"), ("VEC2", "vredsum")])
def test_vector_native_rvv_emitted(rung, expect, tmp_path):
    """The elementwise/reduction transform schedule emits REAL RVV (incl. vectorized reduction)."""
    from merlin.runtime.backends import saturn_vec_mlir as vm
    r = vm.lower_rvv(build(rung), workdir=tmp_path)
    assert r["has_rvv"], f"no RVV vector ops in {rung}"
    assert "vsetivli" in r["rvv_ops"] or "vsetvli" in r["rvv_ops"]
    assert expect in r["rvv_ops"], f"{rung}: expected {expect} in {r['rvv_ops']}"


@pytest.mark.skipif(not (_HAVE_LLVM and _HAVE_RISCV_OBJDUMP),
                    reason="merlin MLIR→LLVM (clang) + riscv objdump toolchain unavailable")
def test_custom_instruction_via_mlir_inline_asm(tmp_path):
    """A custom accelerator instruction (Gemmini RoCC custom-3) declared in MLIR via
    merlin.inline_asm lowers to a raw .insn in the object — no LLVM fork, no C."""
    from merlin.llvmlower import custom_isa
    obj = custom_isa.build_rvv_object("rocc_c3", ".insn r 0x7b, 3, 0, x0, $0, $1", "r,r",
                                      ["i64", "i64"], None, str(tmp_path))
    dis = custom_isa.disassemble(obj)
    assert ".insn" in dis and "7b" in dis  # custom-3 opcode emitted, not a known mnemonic
