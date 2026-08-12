"""Fork-free link + relocation patch for the fixed-format SIMT backend (Muon / RISC-V re-encoding).

Context
-------
The SIMT device this backend targets runs a fixed-64-bit re-encoding of RISC-V: the standard
opcode/funct *values* survive, but each instruction's fields are re-packed into a clean 64-bit format
whose immediate lives contiguously (see :mod:`merlin.targetgen.isa_model`). Device C is compiled with
a **stock** clang (rv32) and transcoded to the fixed format with no relocations; the only remaining
step that needed a compiler *fork* was the **link** of the boot/runtime (BSP) objects, which carry
RISC-V relocations. A stock ``ld.lld`` resolves symbols and lays out sections correctly, but it writes
each relocation's bits at the *standard* rv32 positions -- which, in a fixed-format word, land on the
wrong (register/funct) fields and corrupt the boot code.

Closure
-------
Link with a **stock** linker for LAYOUT ONLY (``--no-relax --emit-relocs``: final section addresses,
a resolved symbol table, and the relocation records preserved in the output), then re-apply every
relocation ourselves at the target's *derived* field positions. The clean (un-relocated) instruction
words are recovered from the input objects, so we never depend on the stock linker's (wrong) writes.

Two properties keep this honest and target-agnostic:

* **Field positions are derived, never hardcoded.** They come from the target's RTL-derived
  :class:`~merlin.targetgen.isa_model.IsaModel` (``field_layout``); the ``target`` is a parameter.
  Only ELF/RISC-V *ABI* relocation-type numbers appear as literals -- those are the psABI contract,
  identical for every RISC-V target, not a fact about any one accelerator.
* **Fail closed.** A relocation type we do not model, or an input placement that does not reproduce
  the stock layout byte-for-byte (outside the reloc sites), raises rather than emitting a boot image
  that silently mis-executes. A single wrong relocation panics the device at ``pc:0``.

The fixed-format immediate is contiguous, so a ``la`` (``auipc``/``addi`` pair) collapses: the
``auipc`` immediate is 0 and the paired ``addi``/load carries the full resolved 32-bit offset. A call
(``auipc``/``jalr``) collapses the same way, with the ``jalr`` at ``+inst_width`` bytes (the fixed
stride), not ``+4``. Branch/jump displacements are byte-accurate in the final layout (the addresses
already reflect the fixed stride), so they are *not* re-doubled here.
"""
from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

# RISC-V psABI relocation type numbers. These are the ELF ABI contract (the same for every RISC-V
# target); they are NOT target ISA facts, so listing them here does not bake in any accelerator.
R_RISCV_BRANCH = 16
R_RISCV_JAL = 17
R_RISCV_CALL = 18
R_RISCV_CALL_PLT = 19
R_RISCV_GOT_HI20 = 20
R_RISCV_PCREL_HI20 = 23
R_RISCV_PCREL_LO12_I = 24
R_RISCV_PCREL_LO12_S = 25

_HI20_KINDS = frozenset({R_RISCV_PCREL_HI20, R_RISCV_GOT_HI20})
_LO12_KINDS = frozenset({R_RISCV_PCREL_LO12_I, R_RISCV_PCREL_LO12_S})
_CALL_KINDS = frozenset({R_RISCV_CALL, R_RISCV_CALL_PLT})


class MuonLinkError(RuntimeError):
    """A relocation could not be applied at fixed-format positions (fail closed, never a silent pass)."""


# --- minimal structural ELF32-LE reader (no regex; struct-parsed) ---------------------------------
@dataclass
class _Section:
    name: str
    type: int
    flags: int
    addr: int
    off: int
    size: int
    link: int
    info: int
    align: int
    entsize: int


class Elf32:
    """Just enough little-endian ELF32 to read sections, symbols, and RELA records, and to patch bytes
    in place. Parsed structurally from the header tables -- no textual tool output, no regex."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.d = bytearray(self.path.read_bytes())
        d = self.d
        if d[:4] != b"\x7fELF" or d[4] != 1:
            raise MuonLinkError(f"{path}: not an ELF32 little-endian object")
        e_shoff = struct.unpack_from("<I", d, 0x20)[0]
        e_shentsize = struct.unpack_from("<H", d, 0x2e)[0]
        e_shnum = struct.unpack_from("<H", d, 0x30)[0]
        e_shstrndx = struct.unpack_from("<H", d, 0x32)[0]
        raw = []
        for i in range(e_shnum):
            o = e_shoff + i * e_shentsize
            (nameoff, typ, flags, addr, off, size,
             link, info, align, entsize) = struct.unpack_from("<IIIIIIIIII", d, o)
            raw.append((nameoff, typ, flags, addr, off, size, link, info, align, entsize))
        shstr_off = raw[e_shstrndx][4]
        shstr_size = raw[e_shstrndx][5]
        strtab = bytes(d[shstr_off:shstr_off + shstr_size])
        self.sections: list[_Section] = []
        for (nameoff, typ, flags, addr, off, size, link, info, align, entsize) in raw:
            self.sections.append(_Section(_cstr(strtab, nameoff), typ, flags, addr, off, size,
                                          link, info, align, entsize))
        self.by_name = {s.name: s for s in self.sections}

    def section_bytes(self, name: str) -> bytes:
        s = self.by_name[name]
        return bytes(self.d[s.off:s.off + s.size])

    def symbol_values(self) -> list[int]:
        """Resolved ``st_value`` per symbol index (from ``.symtab``); empty if none."""
        st = self.by_name.get(".symtab")
        if st is None:
            return []
        out = []
        for i in range(st.size // 16):
            o = st.off + i * 16
            _, value = struct.unpack_from("<II", self.d, o)[:2]
            out.append(value)
        return out

    def rela(self, name: str) -> list[tuple[int, int, int, int]]:
        """RELA records ``(r_offset, r_type, sym_index, addend)`` for a ``.rela.<sec>`` section."""
        rs = self.by_name.get(name)
        if rs is None:
            return []
        out = []
        for i in range(rs.size // 12):
            r_offset, r_info, r_addend = struct.unpack_from("<IIi", self.d, rs.off + i * 12)
            out.append((r_offset, r_info & 0xff, r_info >> 8, r_addend))
        return out

    def vaddr_fileoff(self, vaddr: int, secname: str) -> int:
        s = self.by_name[secname]
        return s.off + (vaddr - s.addr)

    def write(self, path: str | Path) -> None:
        Path(path).write_bytes(self.d)


def _cstr(tab: bytes, n: int) -> str:
    end = tab.index(b"\0", n)
    return tab[n:end].decode()


# --- fixed-format field packing (positions DERIVED from IsaModel.field_layout) --------------------
class _Fields:
    """Pack immediate fields into a fixed-format word using the target's derived field layout.

    A full 32-bit immediate splits into ``imm24`` (low 24 bits) plus one high byte; the high byte
    lives in the ``rs2`` field for register/upper/jump forms (whose ``rs2`` is otherwise unused) and
    in the ``rd`` field for the branch/store (SB) form (whose ``rd`` is otherwise unused).
    """

    def __init__(self, field_layout: dict[str, tuple[int, int]]):
        self.fl = field_layout
        for req in ("imm24", "rs2", "rd"):
            if req not in field_layout:
                raise MuonLinkError(f"field layout missing {req!r}; cannot pack relocation immediate")

    def _set(self, word: int, name: str, value: int) -> int:
        hi, lo = self.fl[name]
        mask = (1 << (hi - lo + 1)) - 1
        return (word & ~(mask << lo)) | ((value & mask) << lo)

    def put_imm32(self, word: int, imm32: int, *, sb: bool) -> int:
        imm32 &= 0xffffffff
        word = self._set(word, "imm24", imm32 & 0xffffff)
        word = self._set(word, "rd" if sb else "rs2", (imm32 >> 24) & 0xff)
        return word


# --- input-section placement (recover clean instruction words) ------------------------------------
def _placement(out_elf: Elf32, inputs: list[Elf32], out_sec: str) -> list[tuple[bytes, int]]:
    """Reproduce how the linker concatenated input sections into ``out_sec``: for each input, gather
    the section named ``out_sec`` or ``out_sec.*`` (the ``*(.text .text.*)`` idiom), placed in link
    order at its own alignment. Returns ``[(bytes, base_vaddr), ...]``."""
    base = out_elf.by_name[out_sec].addr
    cur = base
    items: list[tuple[bytes, int]] = []
    for obj in inputs:
        for s in obj.sections:
            if s.name != out_sec and not s.name.startswith(out_sec + "."):
                continue
            if s.size == 0 or not (s.flags & 0x2):  # SHF_ALLOC
                continue
            align = s.align or 1
            cur = (cur + align - 1) & ~(align - 1)
            items.append((obj.section_bytes(s.name), cur))
            cur += s.size
    return items


def _clean_word(placement: list[tuple[bytes, int]], vaddr: int, out_sec: str) -> int:
    for data, b in placement:
        if b <= vaddr < b + len(data):
            return struct.unpack_from("<Q", data, vaddr - b)[0]
    raise MuonLinkError(f"no input section covers {vaddr:#x} in {out_sec}")


# --- the two public steps -------------------------------------------------------------------------
def resolve_stock_linker(explicit: str | Path | None = None) -> str:
    """Locate a *stock* (unforked) ``ld.lld``. Honors an explicit path, then a generic ``ld.lld`` /
    ``lld`` on ``PATH``. Raises if none is found -- this module must never fall back to a fork."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p)
        raise MuonLinkError(f"explicit linker {explicit} not found")
    for cand in ("ld.lld", "lld"):
        found = shutil.which(cand)
        if found:
            return found
    raise MuonLinkError("no stock ld.lld found on PATH (set an explicit linker path)")


def stock_layout_link(objs: list[str | Path], linker_script: str | Path, out_path: str | Path,
                      *, linker: str | Path | None = None,
                      extra_args: list[str] | None = None) -> Path:
    """Run the stock linker for LAYOUT ONLY: ``--no-relax --emit-relocs`` so the output carries final
    addresses, a resolved symbol table, and the relocation records (which we re-apply ourselves)."""
    ld = resolve_stock_linker(linker)
    cmd = [ld, "--no-relax", "--emit-relocs", "-Bstatic",
           "-T", str(linker_script), "-z", "norelro", "-o", str(out_path),
           *[str(o) for o in objs]]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MuonLinkError(f"stock link failed:\n{' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return Path(out_path)


def patch_relocations(linked_elf: str | Path, objs: list[str | Path],
                      field_layout: dict[str, tuple[int, int]], out_path: str | Path,
                      *, code_sections: tuple[str, ...] = (".init", ".text", ".text.init"),
                      inst_width: int = 64) -> Path:
    """Re-apply the linked ELF's relocations at fixed-format field positions and write ``out_path``.

    ``field_layout`` / ``inst_width`` come from the target's derived IsaModel. Only *instruction*
    sections are re-patched; data relocations (``ADD32``/``SUB32``/``32_PCREL``) write at
    position-only offsets that the stock linker already resolved correctly, so data is left untouched.
    """
    out = Elf32(linked_elf)
    fields = _Fields(field_layout)
    inputs = [Elf32(o) for o in objs]
    symval = out.symbol_values()
    stride = inst_width // 8

    sections = [s for s in code_sections if s in out.by_name and (".rela" + s) in out.by_name]
    for sec in sections:
        recs = out.rela(".rela" + sec)
        placement = _placement(out, inputs, sec)
        _verify_placement(out, sec, recs, placement, stride)
        hi20_target = {r_off: symval[si] + add
                       for (r_off, r_type, si, add) in recs if r_type in _HI20_KINDS}
        for (vaddr, r_type, si, add) in recs:
            foff = out.vaddr_fileoff(vaddr, sec)
            if r_type in _HI20_KINDS:
                # Upper part of a la/call collapses to 0 (immediate is contiguous in the low half).
                w = fields.put_imm32(_clean_word(placement, vaddr, sec), 0, sb=False)
                struct.pack_into("<Q", out.d, foff, w)
            elif r_type in _LO12_KINDS:
                # The referenced symbol is the local hi label == the HI20 instruction address.
                if si >= len(symval) or symval[si] not in hi20_target:
                    raise MuonLinkError(f"LO12 at {vaddr:#x} has no paired HI20 in {sec}")
                off32 = (hi20_target[symval[si]] - symval[si]) & 0xffffffff
                w = fields.put_imm32(_clean_word(placement, vaddr, sec), off32, sb=False)
                struct.pack_into("<Q", out.d, foff, w)
            elif r_type in _CALL_KINDS:
                off32 = (symval[si] + add - vaddr) & 0xffffffff
                struct.pack_into("<Q", out.d, foff,
                                 fields.put_imm32(_clean_word(placement, vaddr, sec), 0, sb=False))
                jr = vaddr + stride
                struct.pack_into("<Q", out.d, out.vaddr_fileoff(jr, sec),
                                 fields.put_imm32(_clean_word(placement, jr, sec), off32, sb=False))
            elif r_type == R_RISCV_JAL:
                disp = (symval[si] + add - vaddr) & 0xffffffff
                w = fields.put_imm32(_clean_word(placement, vaddr, sec), disp, sb=False)
                struct.pack_into("<Q", out.d, foff, w)
            elif r_type == R_RISCV_BRANCH:
                disp = (symval[si] + add - vaddr) & 0xffffffff
                w = fields.put_imm32(_clean_word(placement, vaddr, sec), disp, sb=True)
                struct.pack_into("<Q", out.d, foff, w)
            else:
                raise MuonLinkError(
                    f"unhandled relocation type {r_type} at {vaddr:#x} in {sec}; refusing to emit a "
                    f"boot image that would mis-execute (fail closed)")
    out.write(out_path)
    return Path(out_path)


def _verify_placement(out: Elf32, sec: str, recs, placement, stride: int) -> None:
    """Confirm the reconstructed input placement reproduces the stock layout byte-for-byte outside the
    reloc sites (a mis-placement would silently corrupt boot). A stock rv32 linker dirties a reloc's
    low half; a call also writes the paired jalr's low half into ``+4`` -- i.e. the auipc word's HIGH
    half under the fixed stride -- so a call word is dirty in both halves; other relocs only low."""
    dirty_full: set[int] = set()
    dirty_low: set[int] = set()
    for (r_off, r_type, si, add) in recs:
        (dirty_full if r_type in _CALL_KINDS else dirty_low).add(r_off)
    stock = out.section_bytes(sec)
    base = out.by_name[sec].addr
    recon = bytearray(len(stock))
    for data, b in placement:
        recon[b - base:b - base + len(data)] = data
    for i in range(0, len(stock) - (len(stock) % 8), 8):
        v = base + i
        if v in dirty_full:
            continue
        if stock[i + 4:i + 8] != recon[i + 4:i + 8]:
            raise MuonLinkError(f"placement mismatch (high half) at {v:#x} in {sec}: input "
                                f"reconstruction does not reproduce the stock layout")
        if v not in dirty_low and stock[i:i + 4] != recon[i:i + 4]:
            raise MuonLinkError(f"placement mismatch (low half) at {v:#x} in {sec}")


def link_fork_free(objs: list[str | Path], linker_script: str | Path, out_path: str | Path,
                   *, target: str, isa_model=None, linker: str | Path | None = None,
                   code_sections: tuple[str, ...] = (".init", ".text", ".text.init")) -> Path:
    """Link ``objs`` into a fixed-format device ELF with NO fork component.

    Stock linker lays out + resolves symbols; we re-apply relocations at the target's derived field
    positions. ``target`` selects the RTL-derived IsaModel (``field_layout``/``inst_width``); pass an
    already-built ``isa_model`` to avoid re-deriving it.
    """
    if isa_model is None:
        from merlin.targetgen.isa_model import isa_model_from_encoding
        from merlin.targetgen.rtl import mlc_bridge
        isa_model = isa_model_from_encoding(target, mlc_bridge.isa_encoding_for(target))
    linked = Path(out_path).with_suffix(".layout.elf")
    stock_layout_link(objs, linker_script, linked, linker=linker)
    return patch_relocations(linked, objs, isa_model.field_layout, out_path,
                             code_sections=code_sections, inst_width=isa_model.inst_width)
