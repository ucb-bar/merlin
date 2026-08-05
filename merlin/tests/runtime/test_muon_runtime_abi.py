"""RG4 — the falsifiable de-overfit gate for the fork-free SIMT runtime.

The boot preamble, the harness (console + hart CSR), and the compile substrate are rendered ENTIRELY from a
target's derived ``runtime_abi`` fact — no Muon/radiance literal in the shared code. This suite proves it two
ways: (1) a PERTURBED fact (different CSR numbers, a different CUSTOM slot / funct3 for wspawn, a different
console aperture, a different base ISA) makes the rendered artifacts track the fact with ZERO code edits — so a
second SIMT target works by supplying its own fact; (2) an ABSENT fact fails closed (never a hardcoded default).
Hermetic — no toolchain / sim."""
from __future__ import annotations

import pytest

from merlin.targetgen.isa_model import IsaModel
from merlin.runtime.backends import muon
from merlin.runtime.backends import muon_harness as mh


def _model(runtime_abi: dict) -> IsaModel:
    return IsaModel(target="synth_simt", runtime_abi=runtime_abi)


# a SECOND SIMT target whose runtime ABI differs from radiance in every consumed value.
_PERTURBED = {
    "base_isa_family": "riscv64",
    "special_csrs": {"num_warps": 0xABC, "warp_id": 0xDE0, "warp_mask": 0xDE1, "mhartid": 0xF14},
    "sfu_ops": {"tmc": {"opcode": 0x5B, "funct3": 2}, "wspawn": {"opcode": 0x5B, "funct3": 3}},
    "apertures": {"console_mmio": 0x12340000, "stack_base": 0x40000000, "dram_base": 0x80000000},
}


def test_preamble_tracks_a_perturbed_fact_with_no_code_edit():
    pre = muon.render_simt_preamble(_model(_PERTURBED))
    # the num-warps CSR alias, the tmc/wspawn opcodes AND funct3 all come from THIS fact — not radiance's.
    assert ".set nw, 0xabc\n" in pre
    assert ".insn r 0x5b, 2, 0, x0, \\reg, x0\n" in pre        # tmc: perturbed opcode + funct3
    assert ".insn r 0x5b, 3, 0, x0, \\n, \\f\n" in pre         # wspawn: perturbed opcode + funct3
    # nothing from radiance's fact leaks through.
    assert "0xfc1" not in pre and "0x0b" not in pre


def test_harness_tracks_a_perturbed_console_and_hart_csr():
    h = mh._render_helpers(_model(_PERTURBED))
    assert "0x12340000u" in h                                  # perturbed console aperture
    assert "csrr %0,0xf14" in h                                # hart-id CSR (RISC-V spec)
    assert "0xff080000" not in h                               # radiance's console does not leak


def test_compile_triple_tracks_base_isa_family():
    assert muon.forkfree_compile_triple(_model(_PERTURBED)) == ("riscv64", "lp64")
    assert muon.forkfree_compile_triple(_model({**_PERTURBED, "base_isa_family": "riscv32"})) == ("riscv32", "ilp32")


def test_non_riscv_base_fails_closed():
    with pytest.raises(muon.MuonError):
        muon.forkfree_compile_triple(_model({**_PERTURBED, "base_isa_family": "arm64"}))


def test_absent_runtime_abi_fails_closed_never_defaults():
    empty = IsaModel(target="synth_simt")                      # no runtime_abi
    with pytest.raises(KeyError):
        muon.render_simt_preamble(empty)                       # needs sfu_ops + num_warps CSR
    with pytest.raises(KeyError):
        mh._render_helpers(empty)                              # needs mhartid + console_mmio


def test_simt_runtime_scaffold_is_rendered_from_derived_facts():
    # the multi-warp spawn/park/wait scaffold emits its SIMT-control ops + CSRs ENTIRELY from runtime_abi:
    # a perturbed fact makes the emitted C track it, with zero shared-code edits.
    prog = muon.render_simt_runtime(_model(_PERTURBED), num_warps=8,
                                    worker_body="/*compute*/", globals="volatile int C[1];")
    assert ".insn r 0x5b,3,0,x0,%0,%1" in prog          # wspawn: perturbed opcode 0x5b + funct3 3
    assert ".insn r 0x5b,2,0,x0,x0,x0" in prog           # tmc x0 park: perturbed opcode + funct3, rs1=x0
    assert "csrr %0,0xde0" in prog                        # perturbed warp_id CSR
    assert "csrr %0,0xde1" in prog                        # perturbed warp_mask CSR
    assert "MU_NUM_WARPS 8u" in prog and "volatile int C[1];" in prog
    assert "0xcc1" not in prog and "0x0b" not in prog     # radiance's facts do not leak


def test_simt_runtime_fails_closed_without_runtime_abi():
    with pytest.raises(KeyError):
        muon.render_simt_runtime(IsaModel(target="synth_simt"), num_warps=4, worker_body="")


def test_transcoder_refuses_a_non_riscv_base():
    # the fixed-format transcoder's rv32 decode taxonomy is gated on a RISC-V base family (explicit + fail-closed).
    from merlin.targetgen.isa_transcode import FixedFormatTranscoder, TranscodeError
    from merlin.targetgen.isa_model import isa_model_from_encoding
    fact = {"inst_width": 64,
            "fields": {"opcode": [6, 0], "rd": [16, 9], "f3": [19, 17], "rs1": [27, 20], "rs2": [35, 28]},
            "opcodes": {"OP": 0x33, "OP_IMM": 0x13},
            "runtime_abi": {"base_isa_family": "arm64"}}
    with pytest.raises(TranscodeError):
        FixedFormatTranscoder(isa_model_from_encoding("synth", fact))
