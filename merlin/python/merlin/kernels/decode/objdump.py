"""Generic, ISA-agnostic disassembly tokenizer.

Turns an object file into a list of ``RawInsn`` by **structured field-splitting** of
``llvm-objdump`` output — no semantic regex (we do not guess meaning from mnemonic substrings;
that happens in the per-target semantic decoders, from explicit operands). Reusable by every
riscv-based target (RVV, Gemmini RoCC, scalar); a per-ISA decoder (``decode/rvv.py``,
``targetgen/rocc_decode.py``, …) consumes these ``RawInsn`` and lifts its own facet.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]
_LLVM_OBJDUMP = _REPO / "third_party" / "llvm-install" / "bin" / "llvm-objdump"


def objdump_bin() -> str:
    if _LLVM_OBJDUMP.is_file():
        return str(_LLVM_OBJDUMP)
    return shutil.which("llvm-objdump") or shutil.which("riscv64-unknown-elf-objdump") or "objdump"


@dataclass
class RawInsn:
    addr: int                      # byte address within the section
    mnemonic: str                  # e.g. "vsetivli", "vfmacc.vv", "addi"
    operands: list[str]            # comma-split, stripped: ["zero", "0x4", "e32", "m2", "ta", "ma"]
    hexcode: str = ""              # raw encoding word(s)
    section: str = ""              # enclosing section/symbol if known


def disassemble_text(obj_path: str | Path, triple: str = "riscv64") -> str:
    """Raw ``llvm-objdump -d`` text (no-aliases so the canonical mnemonics/vtype show)."""
    cmd = [objdump_bin(), "-d", f"--triple={triple}", "-M", "no-aliases", str(obj_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"objdump failed: {' '.join(cmd)}\n{p.stderr[-1500:]}")
    return p.stdout


def _parse_line(line: str, section: str) -> RawInsn | None:
    """One disassembly line -> RawInsn, or None for non-instruction lines (headers, labels, '...').

    Format: ``   <addr>: <hex>\\t<mnemonic>\\t<operands>``. We split structurally on ':' then
    whitespace; the operand string is comma-split. Anything that doesn't fit (a ``<sym>:`` label,
    a blank line, ``...``) returns None.
    """
    if ":" not in line:
        return None
    left, _, right = line.partition(":")
    left = left.strip()
    # an instruction's left side is a bare hex address; a label line is "<name>" (not hex).
    try:
        addr = int(left, 16)
    except ValueError:
        return None
    right = right.strip()
    if not right:
        return None
    parts = right.split(None, 2)          # [hexword, mnemonic, operands?]
    if len(parts) < 2:
        return None
    hexword, mnemonic = parts[0], parts[1]
    # the encoding word is hex; if parts[0] isn't hex this isn't an instruction line.
    try:
        int(hexword, 16)
    except ValueError:
        return None
    operands: list[str] = []
    if len(parts) == 3:
        operands = [o.strip() for o in parts[2].split(",") if o.strip()]
    return RawInsn(addr=addr, mnemonic=mnemonic, operands=operands, hexcode=hexword,
                   section=section)


def tokenize(obj_path: str | Path, triple: str = "riscv64") -> list[RawInsn]:
    """Object file -> ordered list of RawInsn (instructions only)."""
    text = disassemble_text(obj_path, triple=triple)
    out: list[RawInsn] = []
    section = ""
    for line in text.splitlines():
        s = line.strip()
        # section/symbol headers look like "<name>:" with no hex address, or
        # "Disassembly of section .text:".
        if s.startswith("Disassembly of section"):
            section = s.split("section", 1)[1].strip().rstrip(":")
            continue
        if s.endswith(">:") or (s.endswith(":") and "<" in s):
            section = s.rstrip(":")
            continue
        insn = _parse_line(line, section)
        if insn is not None:
            out.append(insn)
    return out
