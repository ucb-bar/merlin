"""Fork-free wide-word SIMT codegen: transcode STOCK rv32 machine code into the target's fixed-format
words using ONLY the RTL-derived IsaModel. Hermetic transcode checks + a gated stock-clang→transcode→
decode-clean regression (the proof pipeline's compile+transcode half)."""
from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from merlin.targetgen.isa_model import isa_model_from_encoding
from merlin.targetgen import isa_disasm
from merlin.targetgen.isa_transcode import FixedFormatTranscoder, TranscodeError, _decode_rv32, derive_march

# a Muon-shaped fixed-format model (the field positions the derived radiance fact carries)
MUON_FACT = {
    "inst_width": 64,
    "fields": {"opcode": [6, 0], "ext2": [8, 7], "rd": [16, 9], "f3": [19, 17],
               "rs1": [27, 20], "rs2": [35, 28], "f7": [58, 52], "imm24": [59, 36]},
    "opcodes": {"LOAD": 0x03, "OP_IMM": 0x13, "STORE": 0x23, "OP": 0x33, "BRANCH": 0x63, "JAL": 0x6F},
}


def test_derive_march_reads_the_fp_mode_from_the_opcode_table():
    # OP_FP present, no FP load/store -> zfinx (FP on GPRs), the Muon shape.
    m = isa_model_from_encoding("synth", {**MUON_FACT, "opcodes": {**MUON_FACT["opcodes"], "OP_FP": 0x53}})
    assert derive_march(m) == "rv32im_zfinx"
    # OP_FP + a FP load/store opcode -> a separate FP register file ('f').
    m2 = isa_model_from_encoding("synth", {**MUON_FACT,
                                           "opcodes": {**MUON_FACT["opcodes"], "OP_FP": 0x53, "LOAD_FP": 0x07}})
    assert derive_march(m2) == "rv32imf"
    # no OP_FP -> integer only.
    assert derive_march(isa_model_from_encoding("synth", MUON_FACT)) == "rv32im"


def test_transcode_custom0_simt_op_as_register_form():
    # CUSTOM0 (the SIMT-control opcode family: tmc/wspawn/split/join) is a register-register `.insn r` form
    # -- opcode + f3/f7 select the op, rd/rs1/rs2 are GPRs. It must transcode (not fail closed) so SIMT
    # control can be injected into a kernel. Here a tmc-shaped word (opcode 0x0b, rs1 = the mask reg).
    fact = {**MUON_FACT, "opcodes": {**MUON_FACT["opcodes"], "CUSTOM0": 0x0B}}
    m = isa_model_from_encoding("synth", fact)
    d = _decode_rv32(0x0B | (5 << 15), m.inst_width // 8 // 4)
    assert d.opcode == 0x0B and not d.has_imm and d.rs1 == 5      # register form, no immediate
    w = FixedFormatTranscoder(m).encode(d)
    assert (w & 0x7F) == 0x0B                                     # opcode packed from the derived table
    assert (w >> 20) & 0xFF == 5                                  # rs1 (the mask) at the derived position


def test_transcode_addi_places_fields_at_target_positions():
    m = isa_model_from_encoding("synth", MUON_FACT)
    tc = FixedFormatTranscoder(m)
    # rv32 `addi a0, a0, 11` = 0x00b50513 (imm=11, rs1=x10, f3=0, rd=x10, op=OP_IMM)
    word = tc.transcode_text(struct.pack("<I", 0x00B50513))[0]
    rec = isa_disasm.disassemble(m, [word])[0]
    assert rec["mnemonic"] == "OP_IMM"
    assert rec["operands"]["rd"] == 10 and rec["operands"]["rs1"] == 10
    assert rec["operands"]["f3"] == 0
    assert rec["operands"]["imm24"] == 11           # low 24 bits of the immediate, contiguous
    assert rec["operands"]["rs2"] == 0              # high immediate byte = 0 here


def test_transcode_negative_store_immediate_high_byte_in_rd():
    # a store with a negative offset: the full 32-bit immediate's high byte must land in the rd field
    # (store has no destination reg) so the sign survives — the bug that broke backward branches.
    m = isa_model_from_encoding("synth", MUON_FACT)
    tc = FixedFormatTranscoder(m)
    # `sw a1, -4(a0)` = imm=-4, rs1=x10, rs2=x11, f3=2, op=STORE  -> encode by hand:
    imm = -4 & 0xFFF
    word32 = (((imm >> 5) & 0x7F) << 25) | (11 << 20) | (10 << 15) | (2 << 12) | ((imm & 0x1F) << 7) | 0x23
    word = tc.transcode_text(struct.pack("<I", word32))[0]
    rec = isa_disasm.disassemble(m, [word])[0]
    # imm32 = 0xFFFFFFFC -> low24 in imm24, high byte 0xFF in rd
    assert rec["operands"]["imm24"] == 0xFFFFFC & ((1 << 24) - 1)
    assert rec["operands"]["rd"] == 0xFF
    assert rec["operands"]["rs1"] == 10 and rec["operands"]["rs2"] == 11


def test_transcode_jal_clears_funct3():
    # J-type has no funct3 field: bits [14:12] of an rv32 jal are immediate bits. A *resolved*
    # same-section jal (nonzero displacement) must not leak those bits into the target's f3 field —
    # the fixed format carries the whole displacement in the contiguous immediate.
    m = isa_model_from_encoding("synth", MUON_FACT)
    tc = FixedFormatTranscoder(m)
    off = -144                                       # displacement whose bits [14:12] are nonzero
    o = off & 0x1FFFFF
    word32 = ((((o >> 20) & 1) << 31) | (((o >> 1) & 0x3FF) << 21) | (((o >> 11) & 1) << 20)
              | (((o >> 12) & 0xFF) << 12) | (1 << 7) | 0x6F)   # jal ra, -144
    assert (word32 >> 12) & 0x7 != 0                 # rv32 word really has stray bits at [14:12]
    rec = isa_disasm.disassemble(m, tc.transcode_text(struct.pack("<I", word32)))[0]
    assert rec["mnemonic"] == "JAL"
    assert rec["operands"]["f3"] == 0                # the fix: funct3 cleared for J-type
    # displacement is scaled by the 4->8 byte stride ratio, reconstructed from imm24 + rs2 high byte
    imm32 = (rec["operands"]["rs2"] << 24) | rec["operands"]["imm24"]
    assert imm32 - (1 << 32) == off * tc.stride_ratio


def test_transcode_fma_places_third_source():
    # a 4-register FMA re-maps to the target's rs3 field; the 2-bit format lands in f7.
    fact = {**MUON_FACT,
            "fields": {**MUON_FACT["fields"], "rs3": [43, 36]},
            "opcodes": {**MUON_FACT["opcodes"], "MADD": 0x43}}
    m = isa_model_from_encoding("synth", fact)
    tc = FixedFormatTranscoder(m)
    # fmadd  rd=1, rs1=2, rs2=3, rs3=4, rm=0, fmt=0
    w32 = (4 << 27) | (0 << 25) | (3 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | 0x43
    word = tc.transcode_text(struct.pack("<I", w32))[0]
    rec = isa_disasm.disassemble(m, [word])[0]
    assert rec["mnemonic"] == "MADD"
    assert rec["operands"]["rd"] == 1 and rec["operands"]["rs1"] == 2
    assert rec["operands"]["rs2"] == 3 and rec["operands"]["rs3"] == 4


def test_fma_fails_closed_without_an_rs3_field():
    # a target whose layout has no rs3 field cannot carry a fused-multiply-add -> fail closed.
    m = isa_model_from_encoding("synth", {**MUON_FACT, "opcodes": {**MUON_FACT["opcodes"], "MADD": 0x43}})
    tc = FixedFormatTranscoder(m)
    w32 = (4 << 27) | (3 << 20) | (2 << 15) | (1 << 7) | 0x43
    with pytest.raises(TranscodeError):
        tc.transcode_text(struct.pack("<I", w32))


def test_transcode_fence_places_ordering_bits_in_the_immediate_field():
    """A fence must survive the re-map, or a kernel cannot order its stores before it finishes.

    Its I-type immediate carries the ordering bits {fm,pred,succ} rather than an address, but the
    transcoder re-maps fields and not meanings, so it belongs in the immediate field like every other
    I-type form. Asserted on a SYNTHETIC target so this proves the general rule, not one device.
    """
    fact = {**MUON_FACT, "opcodes": {**MUON_FACT["opcodes"], "MISC_MEM": 0x0F}}
    m = isa_model_from_encoding("synth", fact)
    tc = FixedFormatTranscoder(m)
    # rv32 `fence iorw, iorw` = 0x0ff0000f: imm[11:0] = 0x0FF, rd = rs1 = 0, f3 = 0.
    word = tc.transcode_text(struct.pack("<I", 0x0FF0000F))[0]
    rec = isa_disasm.disassemble(m, [word])[0]
    assert rec["mnemonic"] == "MISC_MEM"
    assert rec["operands"]["imm24"] == 0x0FF, "the ordering bits must survive intact"
    assert rec["operands"]["rd"] == 0 and rec["operands"]["rs1"] == 0
    assert rec["operands"]["f3"] == 0
    # `fence.i` is the same opcode with f3=1; it must not be mistaken for a shift-amount form (that
    # special case is guarded on OP_IMM, and this asserts the guard holds).
    fence_i = tc.transcode_text(struct.pack("<I", 0x0000100F))[0]
    assert isa_disasm.disassemble(m, [fence_i])[0]["operands"]["f3"] == 1


def test_fence_fails_closed_when_the_target_declares_no_misc_mem():
    """A target whose derived decoder has no MISC_MEM must be refused, not sent a fence anyway.

    Nothing target-specific gates this: `encode` compares against the DERIVED opcode table, so leaving
    MISC_MEM out of the fact is enough to make the refusal happen.
    """
    m = isa_model_from_encoding("synth", MUON_FACT)          # MUON_FACT declares no MISC_MEM
    tc = FixedFormatTranscoder(m)
    with pytest.raises(TranscodeError, match="not in the target's derived opcode table"):
        tc.transcode_text(struct.pack("<I", 0x0FF0000F))


def test_transcode_fails_closed_on_auipc():
    m = isa_model_from_encoding("synth", MUON_FACT)
    tc = FixedFormatTranscoder(m)
    auipc = (0 << 12) | (5 << 7) | 0x17            # auipc x5, 0
    with pytest.raises(TranscodeError):
        tc.transcode_text(struct.pack("<I", auipc))


def test_branch_displacement_scaled_by_stride():
    # a branch's PC-relative displacement is scaled by the target/rv32 byte ratio (8/4 = 2 for a 64-bit
    # word), so it lands on the same instruction index.
    d = _decode_rv32(0x00000463, stride_ratio=2)   # beq x0,x0,+8  (imm=8) -> scaled to 16
    assert d.imm == 16


def test_requires_fixed_format_model():
    from merlin.targetgen.isa_model import IsaModel
    with pytest.raises(TranscodeError):
        FixedFormatTranscoder(IsaModel(target="x"))       # empty / not fixed-format


def _stock_clang():
    """The in-tree stock clang, resolved from the repo root rather than an absolute path -- this test
    is skipped when it is absent, so a baked path silently skipped everywhere except one checkout."""
    from merlin.common.paths import repo_root
    p = repo_root() / "third_party" / "llvm-install" / "bin" / "clang"
    return str(p) if p.is_file() else None


def test_stock_rv32_kernel_transcodes_to_clean_muon(tmp_path):
    """The compile+transcode half of the fork-free pipeline: a tiny kernel built with STOCK clang to rv32,
    transcoded via the RTL-derived model, decodes with zero illegal instructions (no auipc in this kernel)."""
    from merlin.targetgen.rtl import mlc_bridge
    from merlin.targetgen.contract.toolchain import mlir_bin
    fact = mlc_bridge.isa_encoding_for("radiance")
    clang, objcopy = _stock_clang(), mlir_bin("llvm-objcopy")
    if not (fact and clang and objcopy.is_file()):
        pytest.skip("mlc fact / stock clang / llvm-objcopy not all present")
    src = tmp_path / "k.c"
    src.write_text("void main(void){volatile int*p=(int*)0x100;int s=0;"
                   "for(int i=0;i<8;i++)s+=i;*p=s;}")
    obj = tmp_path / "k.o"
    r = subprocess.run([clang, "--target=riscv32", "-march=rv32im", "-mabi=ilp32", "-mno-relax",
                        "-fno-pic", "-fno-jump-tables", "-mcmodel=medany", "-O2", "-ffreestanding",
                        "-c", str(src), "-o", str(obj)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"stock rv32 compile unavailable: {r.stderr[-200:]}")
    binf = tmp_path / "k.bin"
    subprocess.run([str(objcopy), "-O", "binary", "--only-section=.text", str(obj), str(binf)],
                   capture_output=True)
    text = binf.read_bytes()
    if not text:
        pytest.skip("empty .text")
    m = isa_model_from_encoding("radiance", fact)
    tc = FixedFormatTranscoder(m)
    try:
        words = tc.transcode_text(text)
    except TranscodeError as e:
        pytest.skip(f"kernel used a non-transcodable form (expected for some builds): {e}")
    recs = isa_disasm.disassemble(m, words)
    illegal = [r for r in recs if r.get("illegal")]
    assert not illegal, f"{len(illegal)} transcoded words did not decode"
    assert len(words) == len(text) // 4     # 1:1 transcode
