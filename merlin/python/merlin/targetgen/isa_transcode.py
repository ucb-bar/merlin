"""Fork-free codegen for a wide-word SIMT target: transcode STOCK-toolchain rv32 machine code into the
target's own fixed-format instruction words, driven entirely by the RTL-derived :class:`IsaModel`.

Why this exists: a self-hosted-ISA SIMT core (e.g. a Vortex-derived design) reuses the *standard*
RISC-V opcode/funct VALUES but lays them out in a wider, un-scattered fixed word (one field layout,
opcode-selected, with the immediate stored contiguously). So a kernel can be built with a **stock**
RISC-V compiler + assembler — no per-target clang fork — and its base-integer/FP machine code
*structurally re-mapped* into the target's word: decode each rv32 instruction at the standard bit
positions, then re-pack the SAME field values into the positions the target's own encoder uses (taken
from ``IsaModel.field_layout``). This owns the backend without forking the toolchain.

Two facts make it a clean 1:1 transcode (both hold for the fixed-format model mlc derives from RTL):

* **Immediate is one contiguous field.** The rv32 scattered B/J/S/U immediate is decoded to its full
  signed integer and re-packed. Because the target's immediate field is narrower than 32 bits, the
  reconstructed value is ``imm32 = (high_byte << 24) | imm_field``; the high byte occupies the word's
  otherwise-unused register field — the ``rd`` slot for store/branch forms (no destination register),
  the second-source slot for every other immediate form (no second source).
* **Wider instruction stride.** Each instruction grows from 4 bytes to ``inst_width/8`` bytes, so every
  PC-relative displacement (branch, jal) is scaled by that ratio. Absolute/data references are NOT
  transcodable this way (a PC-relative ``auipc`` pair cannot be a pure field re-map under a changed
  stride); the transcoder **fails closed** on them so a caller keeps such references relocation-based.

No target-name literal, no ``re``: field positions and opcode values are read from the derived model;
the rv32 side uses the fixed, external standard-RISC-V field positions (a property of the stock
toolchain's ISA, not of any target).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from .isa_model import IsaModel

# Standard RISC-V base-opcode field values (a fixed external fact of the stock rv32 toolchain — these
# are compared against the TARGET's own derived opcode_table before use, never assumed of the target).
_LOAD, _OP_IMM, _STORE, _OP = 0x03, 0x13, 0x23, 0x33
_LUI, _AUIPC, _JAL, _JALR, _SYSTEM = 0x37, 0x17, 0x6F, 0x67, 0x73
_OP_FP = 0x53                        # zfinx FP is register-register (GPRs)
_FMA = {0x43, 0x47, 0x4B, 0x4F}      # MADD / MSUB / NMSUB / NMADD — fused multiply-add (4 source regs)
# CUSTOM0..3 (the SIMT-control / accelerator opcodes: tmc, wspawn, split, join, barrier, ...) are
# register-register `.insn r` forms — opcode + f3 + f7 select the operation, rd/rs1/rs2 are GPRs. Standard
# RISC-V custom-opcode values; the target's own table is still what the packer uses (compared as data).
_CUSTOM = {0x0B, 0x2B, 0x5B, 0x7B}
_ITYPE = {_LOAD, _OP_IMM, _JALR, _SYSTEM}
_STYPE = {_STORE}
_BTYPE = {0x63}          # BRANCH
_JTYPE = {_JAL}
_UTYPE = {_LUI, _AUIPC}
_RTYPE = {_OP, _OP_FP} | _CUSTOM   # register-register (incl. zfinx FP on GPRs + the SIMT CUSTOM ops)


# Standard RISC-V opcode VALUES for the FP-extension family (a fixed external fact of stock RISC-V,
# compared as data against the target's DERIVED opcode table — never assumed present of the target).
_OPV_OP_FP, _OPV_LOAD_FP, _OPV_STORE_FP = 0x53, 0x07, 0x27


def derive_march(model: IsaModel) -> str:
    """Derive the stock-LLVM ``-march`` for a fork-free build from the target's DERIVED opcode table.

    Base is ``rv32im`` (the integer substrate the transcoder re-maps). The FP mode is read from WHICH FP
    opcodes the target's decoder actually defines, compared by standard VALUE (never by opcode name):

      * ``OP_FP`` present, NO FP load/store  -> ``_zfinx`` (FP arithmetic on GPRs; the compiler emits plain
        integer loads for FP data, which the base transcoder already handles);
      * ``OP_FP`` present WITH FP load/store  -> ``f`` (a separate FP register file — FLW/FSW appear);
      * no ``OP_FP``                          -> integer only.

    Fail-safe: an empty/opaque opcode table yields the integer base (a float kernel then fails closed at
    compile rather than silently mis-compiling). This is how the fork-free build learns a target's FP mode
    without a hand-set flag — e.g. Muon defines OP_FP + FMA but no FP load/store, so it derives to zfinx."""
    values = {int(v) for v in model.opcode_table.values()}
    march = "rv32im"
    if _OPV_OP_FP in values:
        march += "f" if (_OPV_LOAD_FP in values or _OPV_STORE_FP in values) else "_zfinx"
    return march


class TranscodeError(ValueError):
    """A word the derived model cannot faithfully re-map (an unknown opcode, or a PC-relative form the
    changed instruction stride would corrupt). Raised rather than emitting a silently-wrong word."""


def _sx(v: int, bits: int) -> int:
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


@dataclass
class _Decoded:
    opcode: int
    rd: int
    f3: int
    rs1: int
    rs2: int
    f7: int
    imm: int        # full signed immediate (already stride-scaled for branch/jal)
    has_imm: bool   # False for register-register ops (the second-source field is a real register)
    is_store_like: bool  # store/branch: the immediate's high byte lives in the rd field
    rs3: int = 0    # third source register (fused-multiply-add only); placed only for non-immediate forms


def _decode_rv32(word: int, stride_ratio: int) -> _Decoded:
    """Decode one 32-bit standard-RISC-V word into (opcode, regs, funct, full-immediate). Branch/jal
    displacements are scaled by ``stride_ratio`` (the target/rv32 instruction-byte ratio)."""
    op = word & 0x7F
    rd = (word >> 7) & 0x1F
    f3 = (word >> 12) & 0x7
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    f7 = (word >> 25) & 0x7F
    imm, has_imm, is_store_like, rs3 = 0, True, False, 0
    if op in _FMA:
        # R4-type: rs3 in bits [31:27], the 2-bit format in [26:25] (the target carries it in f7's low
        # bits), rm in f3; no immediate, three real source registers.
        has_imm, rs3, f7 = False, (word >> 27) & 0x1F, (word >> 25) & 0x3
    elif op in _ITYPE:
        rs2 = 0
        if op == _OP_IMM and f3 in (0x1, 0x5):        # slli/srli/srai: shamt is the imm; funct7 -> f7
            imm = (word >> 20) & 0x1F
        elif op == _SYSTEM:                            # csr number (unsigned 12-bit) is the imm
            imm, f7 = (word >> 20) & 0xFFF, 0
        else:
            imm, f7 = _sx((word >> 20) & 0xFFF, 12), 0
    elif op in _STYPE:
        rd, is_store_like, f7 = 0, True, 0
        imm = _sx((((word >> 25) & 0x7F) << 5) | ((word >> 7) & 0x1F), 12)
    elif op in _BTYPE:
        rd, is_store_like, f7 = 0, True, 0
        b = (((word >> 31) & 1) << 12) | (((word >> 7) & 1) << 11) \
            | (((word >> 25) & 0x3F) << 5) | (((word >> 8) & 0xF) << 1)
        imm = _sx(b, 13) * stride_ratio
    elif op in _JTYPE:
        # J-type has no funct3 field; bits [14:12] are immediate bits, so f3 must be cleared (the wide
        # format carries the whole displacement in the contiguous immediate).
        rs1, rs2, f7, f3 = 0, 0, 0, 0
        j = (((word >> 31) & 1) << 20) | (((word >> 12) & 0xFF) << 12) \
            | (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3FF) << 1)
        imm = _sx(j, 21) * stride_ratio
    elif op in _UTYPE:
        if op == _AUIPC:
            raise TranscodeError("auipc: a PC-relative pair is not a pure field re-map under a changed "
                                 "instruction stride; keep this reference relocation-based (fail closed)")
        # U-type likewise has no funct3 field; bits [14:12] are part of the upper immediate.
        rs1, rs2, f7, f3 = 0, 0, 0, 0
        imm = (word >> 12) & 0xFFFFF                    # lui upper immediate (hardware applies << 12)
    elif op in _RTYPE:
        has_imm = False                                # rs2 is a real register; funct7 already in f7
    else:
        raise TranscodeError(f"opcode {op:#04x} is not handled by the base-ISA transcoder")
    return _Decoded(op, rd, f3, rs1, rs2, f7, imm, has_imm, is_store_like, rs3)


class FixedFormatTranscoder:
    """Re-packs decoded standard-rv32 instructions into a target's fixed-format words using ONLY the
    derived :class:`IsaModel` (field bit-ranges + opcode table + instruction width)."""

    def __init__(self, model: IsaModel):
        if not model.is_fixed_format():
            raise TranscodeError("transcoder requires a fixed-format model (field layout + opcode table)")
        # The rv32 decode taxonomy below (_OP_IMM/_STYPE/_BTYPE/_AUIPC …) is the RISC-V base ISA — valid only
        # when the target re-encodes a RISC-V substrate. Make that assumption EXPLICIT + fail-closed: if the
        # derived runtime ABI declares a non-RISC-V base family, refuse rather than silently mis-decode. (A
        # target whose runtime ABI has not been derived leaves the family blank -> the legacy path is allowed.)
        fam = model.base_isa_family()
        if fam and not fam.startswith("riscv"):
            raise TranscodeError(f"fixed-format transcode assumes a RISC-V base substrate; derived "
                                 f"base_isa_family={fam!r} for target {model.target!r} (fail closed)")
        self.fl = dict(model.field_layout)
        self.width = int(model.inst_width)
        self.opcodes = {int(v) for v in model.opcode_table.values()}
        for req in ("opcode", "rd", "rs1", "rs2", "f3"):
            if req not in self.fl:
                raise TranscodeError(f"model field layout is missing the '{req}' field")
        # The immediate field is the widest field in the layout (the wide-word format stores the
        # immediate contiguously); derived, not named by a literal.
        self.imm_field = max(self.fl, key=lambda n: self.fl[n][0] - self.fl[n][1])
        self.imm_bits = self.fl[self.imm_field][0] - self.fl[self.imm_field][1] + 1
        if self.width % 8:
            raise TranscodeError(f"instruction width {self.width} is not a byte multiple")
        self.stride_ratio = (self.width // 8) // 4
        if self.stride_ratio < 1:
            raise TranscodeError(f"instruction width {self.width} is narrower than rv32")

    def _place(self, word: int, name: str, value: int, mask_bits: int | None = None) -> int:
        hi, lo = self.fl[name]
        width = hi - lo + 1
        if mask_bits is not None:
            value &= (1 << mask_bits) - 1
        if value >> width:
            raise TranscodeError(f"field '{name}'={value:#x} overflows its {width}-bit range")
        return word | (value << lo)

    def encode(self, d: _Decoded) -> int:
        if d.opcode not in self.opcodes:
            raise TranscodeError(f"opcode {d.opcode:#04x} is not in the target's derived opcode table")
        rd, rs2 = d.rd, d.rs2
        word = 0
        word |= d.opcode << self.fl["opcode"][1]       # opcode at its field's low bit (extension=0)
        if d.has_imm:
            imm32 = d.imm & 0xFFFFFFFF
            imm_hi = (imm32 >> 24) & 0xFF
            word = self._place(word, self.imm_field, imm32, mask_bits=self.imm_bits)
            # High byte spills into the otherwise-unused register field for this form.
            if d.is_store_like:
                rd = imm_hi
            else:
                rs2 = imm_hi
        word = self._place(word, "rd", rd)
        word = self._place(word, "f3", d.f3)
        word = self._place(word, "rs1", d.rs1)
        word = self._place(word, "rs2", rs2)
        if d.f7 and "f7" in self.fl:
            word = self._place(word, "f7", d.f7)
        # third source (fused-multiply-add) — only for non-immediate forms; its field overlaps the
        # contiguous immediate field, so placing it under an immediate would corrupt the immediate.
        if d.rs3:
            if d.has_imm or "rs3" not in self.fl:
                raise TranscodeError("this instruction needs a third-source (rs3) field the target's layout "
                                     "does not provide (or an immediate form overlaps it) — fail closed")
            word = self._place(word, "rs3", d.rs3)
        return word & ((1 << self.width) - 1)

    def transcode_text(self, text: bytes) -> list[int]:
        """Transcode a raw rv32 ``.text`` byte stream (little-endian 4-byte words) into target words."""
        if len(text) % 4:
            raise TranscodeError("rv32 .text length is not a multiple of 4 (compressed instructions?)")
        return [self.encode(_decode_rv32(w, self.stride_ratio)) for (w,) in struct.iter_unpack("<I", text)]


def to_data_lines(words: list[int], inst_width: int) -> str:
    """Render transcoded words as the assembler data directive stock ``llvm-mc`` emits little-endian:
    a wide (>32-bit) word becomes ``.quad`` (8 bytes), else ``.word`` (4 bytes)."""
    if inst_width > 32:
        return "".join(f".quad 0x{w & ((1 << 64) - 1):016x}\n" for w in words)
    return "".join(f".word 0x{w & 0xFFFFFFFF:08x}\n" for w in words)


def emit_kernel_asm(words: list[int], inst_width: int, entry: str = "main") -> str:
    """Wrap transcoded words as a `.text` assembly unit exporting ``entry`` — the input stock ``llvm-mc``
    assembles into the target object that links against the vendored runtime (BSP)."""
    body = to_data_lines(words, inst_width)
    return (f".section .text\n.global {entry}\n.type {entry},@function\n{entry}:\n"
            f"{body}.size {entry}, .-{entry}\n")
