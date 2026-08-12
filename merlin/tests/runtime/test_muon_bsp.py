"""Fork-free boot/BSP object build for the fixed-format SIMT backend (:mod:`muon_bsp`).

Hermetic, board-free checks of the instruction-level transcode that lets a STOCK-assembled boot object
(mixing standard RISC-V ops with the target's CUSTOM-slot SIMT ops) be re-mapped into the target's
fixed-format words without a vendor compiler fork. The full assemble+link+cyclotron proof runs out of
band with the local radiance toolchain; here we lock the CUSTOM-slot re-encode and the auipc contract.
"""
from __future__ import annotations

from merlin.targetgen.isa_model import isa_model_from_encoding
from merlin.targetgen.isa_asm import assemble_fixed
from merlin.targetgen.isa_transcode import FixedFormatTranscoder
from merlin.runtime.backends.base import get_backend
MB = get_backend("muon").muon_bsp               # evicted SIMT backend, resolved via plugin discovery

# A Muon-shaped fixed-format model that also defines the standard CUSTOM slots + AUIPC.
FACT = {
    "inst_width": 64,
    "fields": {"opcode": [6, 0], "ext2": [8, 7], "rd": [16, 9], "f3": [19, 17],
               "rs1": [27, 20], "rs2": [35, 28], "f7": [58, 52], "imm24": [59, 36]},
    "opcodes": {"LOAD": 0x03, "CUSTOM0": 0x0B, "OP_IMM": 0x13, "AUIPC": 0x17, "STORE": 0x23,
                "OP": 0x33, "BRANCH": 0x63, "JAL": 0x6F},
}


def _tc(m):
    return FixedFormatTranscoder(m), MB._custom_slot_opcodes(m)


def test_custom_slot_opcodes_are_derived_from_the_opcode_table():
    m = isa_model_from_encoding("synth", FACT)
    assert MB._custom_slot_opcodes(m) == {0x0B: "CUSTOM0"}


def test_custom_rtype_word_reencodes_through_derived_encoder():
    # rv32 `.insn r 0x0b, 1, 0, x0, t0, t1` (a warp-spawn-shaped CUSTOM0 op): funct3=1, rd=x0,
    # rs1=t0(5), rs2=t1(6). The re-mapped word must equal the derived assemble_fixed encoding.
    m = isa_model_from_encoding("synth", FACT)
    tc, customs = _tc(m)
    rv = 0x0B | (0 << 7) | (1 << 12) | (5 << 15) | (6 << 20) | (0 << 25)
    got = MB._transcode_word(rv, tc, customs, m)
    assert got == assemble_fixed(m, "CUSTOM0", {"f3": 1, "f7": 0, "rd": 0, "rs1": 5, "rs2": 6})


def test_auipc_becomes_clean_zero_immediate_word_with_rd_preserved():
    # auipc is the upper half of a la/call — a relocation site in a relocatable boot object. It must
    # transcode to opcode+rd with a zero immediate (the linker fills it), not fail closed.
    m = isa_model_from_encoding("synth", FACT)
    tc, customs = _tc(m)
    rv = 0x17 | (5 << 7) | (0x12345 << 12)           # auipc t0, 0x12345 (the imm is reloc-supplied)
    got = MB._transcode_word(rv, tc, customs, m)
    assert got == assemble_fixed(m, "AUIPC", {"rd": 5})
    # immediate fields (imm24 + rs2 high byte) are zero; only opcode + rd carry information
    assert (got >> 36) & 0xFFFFFF == 0 and (got >> 28) & 0xFF == 0


def test_base_isa_word_still_goes_through_the_transcoder():
    m = isa_model_from_encoding("synth", FACT)
    tc, customs = _tc(m)
    addi = 0x00B50513                                 # addi a0, a0, 11
    assert MB._transcode_word(addi, tc, customs, m) == tc.transcode_text(addi.to_bytes(4, "little"))[0]


def test_pcrel_pair_without_reloc_is_rescaled_to_target_stride():
    # a same-section `la`/&sym the assembler resolved into a bare auipc+addi (NO relocation) carries its
    # displacement at the rv32 (4-byte) stride; under the target's 8-byte stride it must be rescaled x2.
    import struct
    auipc = (0 << 12) | (10 << 7) | 0x17                        # auipc a0, 0   (high part 0, within ±2KB)
    addi = (40 << 20) | (10 << 15) | (0 << 12) | (10 << 7) | 0x13   # addi a0, a0, 40  (the pcrel low part)
    data = struct.pack("<II", auipc, addi)
    ov = MB._pcrel_lo_overrides(data, reloc_offsets=set(), stride_ratio=2)
    assert ov == {4: 80}                                        # the addi (byte off 4) carries 40*2 = 80


def test_pcrel_pair_at_a_reloc_site_is_left_to_the_linker():
    # when the auipc IS a relocation site, the linker resolves the pair — no override is emitted here.
    import struct
    auipc = (0 << 12) | (10 << 7) | 0x17
    addi = (40 << 20) | (10 << 15) | (10 << 7) | 0x13
    data = struct.pack("<II", auipc, addi)
    assert MB._pcrel_lo_overrides(data, reloc_offsets={0}, stride_ratio=2) == {}


def test_override_immediate_is_encoded_into_the_low_instruction():
    from merlin.targetgen import isa_disasm
    m = isa_model_from_encoding("synth", FACT)
    tc, customs = _tc(m)
    addi = (40 << 20) | (10 << 15) | (0 << 12) | (10 << 7) | 0x13
    w = MB._transcode_word(addi, tc, customs, m, override_imm=80)
    rec = isa_disasm.disassemble(m, [w])[0]
    assert rec["operands"]["imm24"] == 80                       # the rescaled displacement, contiguous
    assert rec["operands"]["rd"] == 10 and rec["operands"]["rs1"] == 10
