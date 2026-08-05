"""Fork-free build of the boot/BSP object (the crt0-like startup) for a fixed-format SIMT target.

Context
-------
:mod:`merlin.runtime.backends.muon_link` closed the fork-free *link*: a stock ``ld.lld`` lays out the
device image and the relocations are re-applied at the target's derived field positions. The device
*kernel* is already fork-free too — a stock rv32 compiler + :class:`~merlin.targetgen.isa_transcode.
FixedFormatTranscoder` produce its fixed-format words. The one step that still leaned on a vendor
*compiler fork* was assembling the boot object (``crt0``/``_start``): it mixes standard RISC-V ops with
the target's CUSTOM-slot SIMT ops (thread-mask control, warp spawn), which a stock assembler does not
know as mnemonics.

This module removes that last fork dependency. The boot source is assembled with a **stock** toolchain
(its SIMT pseudo-mnemonics supplied as explicit ``.insn`` CUSTOM-slot forms through a caller-provided
assembler preamble), then its executable sections are transcoded into the target's fixed-format words —
base ISA through the derived :class:`FixedFormatTranscoder`, the CUSTOM-slot ops through the derived
:func:`~merlin.targetgen.isa_asm.assemble_fixed` encoder. The result is an ordinary relocatable object:
its relocations are preserved (offsets scaled by the instruction-stride ratio) so ``muon_link`` resolves
them at link time exactly as it does for the fork-built boot. NO vendor-fork binary is invoked anywhere.

Target-agnostic + fail-closed
-----------------------------
Every ISA fact — field positions, opcode values, instruction width — is read from the target's
RTL-derived :class:`~merlin.targetgen.isa_model.IsaModel`; ``target`` is a parameter. The CUSTOM-slot
opcodes are discovered from the derived opcode table (the standard RISC-V ``CUSTOM0..3`` slots), not
named by any target literal, and are re-encoded assuming the register-register (``.insn r``) form the
boot uses — an operand the derived encoder cannot represent raises rather than emitting a wrong word.
An ``auipc`` (the upper half of a ``la``/``call``) is always a HI20 relocation site in a relocatable
boot object, so it is emitted as the clean opcode+``rd`` word with a zero immediate that the linker
fills; a non-relocatable ``auipc`` would still fail closed in the base transcoder.
"""
from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path

from ...targetgen.isa_asm import assemble_fixed
from ...targetgen import isa_transcode as _it
from ...targetgen.isa_transcode import FixedFormatTranscoder, TranscodeError

_SHT_PROGBITS, _SHT_SYMTAB, _SHT_STRTAB, _SHT_RELA, _SHT_NOBITS = 1, 2, 3, 4, 8
_SHF_ALLOC, _SHF_EXECINSTR = 0x2, 0x4


class BootBuildError(RuntimeError):
    """A boot object could not be built fork-free (an operand the derived model cannot encode, a
    malformed input object). Raised rather than emitting a boot image that would mis-execute."""


# --- instruction-level transcode (base ISA + CUSTOM-slot ops + relocation-site auipc) --------------
def _custom_slot_opcodes(model) -> dict[int, str]:
    """opcode-value -> name for the standard RISC-V CUSTOM slots this target defines (from the derived
    opcode table; the ``CUSTOM`` prefix is a standard opcode-slot name, not a target literal)."""
    return {int(v): n for n, v in model.opcode_table.items() if n.startswith("CUSTOM")}


def _transcode_word(word: int, tx: FixedFormatTranscoder, customs: dict[int, str], model) -> int:
    """Re-map one standard-rv32 word into the target's fixed-format word.

    * A CUSTOM-slot opcode is decoded at the standard register-register (``.insn r``) positions and
      re-encoded through the derived :func:`assemble_fixed`.
    * An ``auipc`` is a HI20 relocation site in a relocatable boot object: emit the clean opcode+``rd``
      word with a zero immediate (the linker fills it), instead of failing closed as the base transcoder
      does for a stride-changed PC-relative pair.
    * Everything else goes through the derived base-ISA transcoder.
    """
    op = word & 0x7F
    if op in customs:
        try:
            return assemble_fixed(model, customs[op], {
                "rd": (word >> 7) & 0x1F, "f3": (word >> 12) & 0x7,
                "rs1": (word >> 15) & 0x1F, "rs2": (word >> 20) & 0x1F,
                "f7": (word >> 25) & 0x7F})
        except Exception as e:  # AssembleError etc. — fail closed with context
            raise BootBuildError(f"cannot encode CUSTOM-slot word {word:#010x} "
                                 f"({customs[op]}) via the derived encoder: {e}") from None
    auipc = model.opcode_table.get("AUIPC")
    if auipc is not None and op == int(auipc):
        return assemble_fixed(model, "AUIPC", {"rd": (word >> 7) & 0x1F})
    return tx.encode(_it._decode_rv32(word, tx.stride_ratio))


def _transcode_section(data: bytes, tx: FixedFormatTranscoder, customs, model) -> bytes:
    if len(data) % 4:
        raise BootBuildError("executable section length is not a multiple of 4 (compressed insns?)")
    stride = tx.width // 8
    fmt = "<Q" if stride == 8 else "<I"
    out = bytearray()
    for (w,) in struct.iter_unpack("<I", data):
        out += struct.pack(fmt, _transcode_word(w, tx, customs, model))
    return bytes(out)


# --- ELF32-LE object transcode (grow code sections, scale symbol values + reloc offsets) -----------
def transcode_boot_object(rv32_obj: str | Path, out_obj: str | Path, *, isa_model) -> dict:
    """Transcode a stock-assembled rv32 boot object into the target's fixed-format object.

    Executable ``PROGBITS`` sections are re-mapped instruction-for-instruction (each word grows from
    4 bytes to ``inst_width/8``); symbol values into those sections and relocation offsets targeting
    them are scaled by the same ratio; relocation records (types, symbols, addends) and all other
    sections are preserved verbatim so :mod:`muon_link` can resolve them. ``isa_model`` is the target's
    derived model. Returns a small summary (stride ratio, transcoded section names, CUSTOM slots seen).
    """
    d = bytearray(Path(rv32_obj).read_bytes())
    if d[:4] != b"\x7fELF" or d[4] != 1:
        raise BootBuildError(f"{rv32_obj}: not an ELF32 little-endian object")
    e_shoff = struct.unpack_from("<I", d, 0x20)[0]
    e_shentsize = struct.unpack_from("<H", d, 0x2E)[0]
    e_shnum = struct.unpack_from("<H", d, 0x30)[0]
    e_shstrndx = struct.unpack_from("<H", d, 0x32)[0]
    secs = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        (name, typ, flags, addr, off, size, link, info, align, entsize) = struct.unpack_from(
            "<IIIIIIIIII", d, o)
        data = bytearray() if typ == _SHT_NOBITS else bytearray(d[off:off + size])
        secs.append(dict(idx=i, name=name, type=typ, flags=flags, addr=addr, off=off, size=size,
                         link=link, info=info, align=align, entsize=entsize, data=data))
    shstr = bytes(secs[e_shstrndx]["data"])
    for s in secs:
        n = s["name"]
        s["sname"] = shstr[n:shstr.index(b"\0", n)].decode()

    stride_ratio = (int(isa_model.inst_width) // 8) // 4
    if stride_ratio < 1:
        raise BootBuildError(f"instruction width {isa_model.inst_width} is narrower than rv32")
    tx = FixedFormatTranscoder(isa_model)
    customs = _custom_slot_opcodes(isa_model)

    code_idx = set()
    for s in secs:
        if s["type"] == _SHT_PROGBITS and (s["flags"] & _SHF_EXECINSTR):
            s["data"] = bytearray(_transcode_section(bytes(s["data"]), tx, customs, isa_model))
            s["size"] = len(s["data"])
            code_idx.add(s["idx"])
    if not code_idx:
        raise BootBuildError("no executable PROGBITS section found to transcode")

    for s in secs:
        if s["type"] == _SHT_SYMTAB:
            sd = s["data"]
            for i in range(len(sd) // 16):
                st_value = struct.unpack_from("<I", sd, i * 16 + 4)[0]
                st_shndx = struct.unpack_from("<H", sd, i * 16 + 14)[0]
                if st_shndx in code_idx:
                    struct.pack_into("<I", sd, i * 16 + 4, (st_value * stride_ratio) & 0xFFFFFFFF)
        elif s["type"] == _SHT_RELA and s["info"] in code_idx:
            rd = s["data"]
            for i in range(len(rd) // 12):
                r_off = struct.unpack_from("<I", rd, i * 12)[0]
                struct.pack_into("<I", rd, i * 12, (r_off * stride_ratio) & 0xFFFFFFFF)

    _serialize_elf32(d[:52], secs, out_obj)
    return {"stride_ratio": stride_ratio,
            "code_sections": [secs[i]["sname"] for i in sorted(code_idx)],
            "custom_slots": sorted(set(customs.values()))}


def _serialize_elf32(header: bytes, secs: list[dict], out_obj: str | Path) -> None:
    """Write ELF header + section data (each at its alignment) + section header table, fixing offsets."""
    out = bytearray(header)
    for s in secs:
        if s["type"] == 0:
            s["off"] = 0
            continue
        if s["type"] == _SHT_NOBITS:
            s["off"] = len(out)
            continue
        al = s["align"] or 1
        pad = (-len(out)) % al
        out += b"\0" * pad
        s["off"] = len(out)
        out += s["data"]
    out += b"\0" * ((-len(out)) % 4)
    new_shoff = len(out)
    for s in secs:
        out += struct.pack("<IIIIIIIIII", s["name"], s["type"], s["flags"], s["addr"], s["off"],
                           s["size"], s["link"], s["info"], s["align"], s["entsize"])
    struct.pack_into("<I", out, 0x20, new_shoff)
    Path(out_obj).write_bytes(out)


# --- convenience: stock assemble + transcode ------------------------------------------------------
def build_boot_object(boot_asm: str | Path, out_obj: str | Path, *, target: str, clang: str | Path,
                      isa_model=None, march: str = "rv32im", mabi: str = "ilp32",
                      asm_preamble: str = "", cpp: bool = True) -> dict:
    """Assemble a boot source with a STOCK toolchain and transcode it into the target's fixed-format
    object — no vendor compiler fork anywhere.

    ``boot_asm`` is the crt0/BSP source; ``clang`` is a stock RISC-V clang (used only to preprocess and
    assemble rv32 — never to emit target words). ``asm_preamble`` is the target's assembler shim text
    (its SIMT pseudo-mnemonics expressed as explicit ``.insn`` CUSTOM-slot forms, plus any custom-CSR
    ``.set`` names) prepended before assembly so a stock assembler accepts the source. ``isa_model`` is
    the target's derived model (built from ``target`` if omitted).
    """
    if isa_model is None:
        from ...targetgen.isa_model import isa_model_from_encoding
        from ...targetgen.rtl import mlc_bridge
        isa_model = isa_model_from_encoding(target, mlc_bridge.isa_encoding_for(target))
    clang = str(clang)
    triple = "--target=riscv32"
    with tempfile.TemporaryDirectory() as td:
        src = Path(boot_asm)
        body = src.read_text()
        if cpp:
            pre = subprocess.run([clang, triple, "-E", "-x", "assembler-with-cpp", str(src)],
                                 capture_output=True, text=True)
            if pre.returncode != 0:
                raise BootBuildError(f"stock preprocess failed:\n{pre.stderr[-2000:]}")
            body = pre.stdout
        staged = Path(td) / "boot.stock.S"
        staged.write_text((asm_preamble + "\n" if asm_preamble else "") + body)
        rv32_obj = Path(td) / "boot.stock.o"
        asm = subprocess.run([clang, triple, f"-march={march}", f"-mabi={mabi}", "-mno-relax",
                              "-c", str(staged), "-o", str(rv32_obj)],
                             capture_output=True, text=True)
        if asm.returncode != 0:
            raise BootBuildError(f"stock assemble failed:\n{asm.stderr[-2000:]}")
        return transcode_boot_object(rv32_obj, out_obj, isa_model=isa_model)


def build_bsp(boot_src: str | Path, tohost_src: str | Path, out_dir: str | Path, *, target: str,
              clang: str | Path, mc: str | Path, asm_preamble: str = "", num_warps: int = 1,
              isa_model=None) -> list[Path]:
    """Reproducibly build the WHOLE fork-free BSP for a fixed-format SIMT target from its shipped sources —
    no transient or committed binaries. Returns the object list a linker consumes: the transcoded boot
    object (via :func:`build_boot_object`) plus the data-only runtime shims. The occupancy shim
    (``__mu_num_warps``) is generated from ``num_warps``; ``tohost_src`` is the sim-exit mailbox source.
    Both shims are pure DATA (no instructions), so a stock assembler emits them directly — only the boot's
    executable sections need transcoding. Everything is stock-toolchain only; ``target`` is a parameter."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clang, mc = str(clang), str(mc)
    boot_o = out / "boot.forkfree.o"
    build_boot_object(boot_src, boot_o, target=target, clang=clang, asm_preamble=asm_preamble,
                      isa_model=isa_model)
    # occupancy shim: override the BSP's weak __mu_num_warps (a plain data word — no relocation, no
    # instruction, so it links as-is and needs no transcode).
    murt_s = out / "murt.S"
    murt_s.write_text(".section .data\n.globl __mu_num_warps\n.p2align 2\n"
                      f"__mu_num_warps: .word {int(num_warps)}\n")
    objs = [boot_o]
    for name, src in (("murt.o", murt_s), ("tohost.o", Path(tohost_src))):
        o = out / name
        r = subprocess.run([mc, "--triple=riscv32", "--filetype=obj", str(src), "-o", str(o)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise BootBuildError(f"stock assemble of {name} failed:\n{r.stderr[-1500:]}")
        objs.append(o)
    return objs
