"""Derive a custom-instruction table from the target's OWN intrinsics header.

Some targets do not ship an ISA definition a tool can read; they ship a C header whose inline-assembly
intrinsics ARE the definition. Each one names the operation and encodes it in the same place, so the
header is a decode table written in a different notation:

    inline void vx_barrier(unsigned id, unsigned n) {
        asm volatile (".insn r %0, 4, 0, x0, %1, %2" :: "i"(RISCV_CUSTOM0), ...);
    }

That is the target saying, in its own words, that opcode CUSTOM0 with funct3=4 is a warp barrier.
Reading it is the same act as reading a funct table out of RTL — the source differs, the derivation
does not — and it is the difference between a SIMT target whose accelerator surface is 8 unexplained
opcode spaces and one whose barriers, warp spawns and divergence points are legible.

Parsed structurally (``split``/``partition``, no regex), because a too-narrow pattern silently drops a
valid-but-differently-spelled declaration, which here would mean quietly losing an instruction rather
than failing. Anything not understood is REPORTED, never skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["HeaderInsn", "parse_insn_header", "table_for"]


@dataclass(frozen=True)
class HeaderInsn:
    """One custom instruction, as the target's own header declares it."""

    name: str                       # the intrinsic's function name — the target's own vocabulary
    opcode_macro: str               # the opcode symbol the .insn references
    opcode: int | None = None       # resolved from the header's own #define, when present
    funct3: int | None = None
    funct7: int | None = None
    form: str = "r"                 # the .insn form (r, r4, i, s, ...)

    def key(self) -> tuple:
        return (self.opcode, self.funct3)


def _int_of(token: str) -> int | None:
    """A C integer literal (decimal or 0x...), or None. No regex."""
    t = token.strip().rstrip("uUlL")
    if not t:
        return None
    neg = t.startswith("-")
    t = t[1:] if neg else t
    try:
        v = int(t, 16) if t.lower().startswith("0x") else int(t)
    except ValueError:
        return None
    return -v if neg else v


def _defines(text: str) -> dict[str, int]:
    """``#define NAME <int>`` pairs — the header's own opcode symbols."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#define"):
            continue
        parts = s.split(None, 2)
        if len(parts) < 3:
            continue
        val = _int_of(parts[2].split("//")[0].split("/*")[0])
        if val is not None:
            out[parts[1]] = val
    return out


def _enclosing_name(lines, index: int) -> str:
    """The function name whose body contains ``lines[index]``.

    Scans BACKWARDS to the nearest declaration, because the intrinsic's meaning lives in its name and
    the encoding lives in its body, and nothing else connects the two.
    """
    for i in range(index, max(-1, index - 12), -1):
        s = lines[i]
        # Strip comments BEFORE looking for a declaration: a comment like "...into a0 (x10) register"
        # otherwise reads as a call and yields `a0` as the instruction's name. Found on a real header.
        for marker in ("//", "/*"):
            cut = s.find(marker)
            if cut >= 0:
                s = s[:cut]
        s = s.strip()
        # A string literal is assembly text, not a declaration.
        if s.startswith('"') or "(" not in s or s.startswith((".insn", "asm", "*", "#")):
            continue
        head = s.split("(", 1)[0].strip()
        if not head or head.endswith((",", ";", "=")):
            continue
        name = head.split()[-1].lstrip("*")
        if name and (name[0].isalpha() or name[0] == "_"):
            return name
    return ""


def parse_insn_header(path: "str | Path") -> tuple[list[HeaderInsn], tuple[str, ...]]:
    """``(instructions, problems)`` from a C intrinsics header.

    ``problems`` names every ``.insn`` the parser could not fully place. A silently skipped declaration
    is an instruction that will later be reported as an unexplained opcode, with nothing linking the two.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    defines = _defines(text)
    lines = text.splitlines()
    out: list[HeaderInsn] = []
    problems: list[str] = []
    seen: set = set()

    for i, line in enumerate(lines):
        at = line.find(".insn")
        if at < 0:
            continue
        rest = line[at + len(".insn"):].strip()
        form, _, args = rest.partition(" ")
        form = form.strip().strip('"')
        # `.insn r %0, 4, 0, x0, ...` — the operand list after the form. The first token is the opcode
        # (a %N placeholder bound to an "i"(MACRO) constraint), then funct3, then funct7.
        args = args.split('"')[0]
        toks = [t.strip() for t in args.split(",")]
        if len(toks) < 3:
            problems.append(f"line {i + 1}: .insn with too few operands to place: {rest[:60]!r}")
            continue
        f3, f7 = _int_of(toks[1]), _int_of(toks[2])
        # The opcode arrives as a %N placeholder; its macro is the matching "i"(MACRO) constraint.
        macro = ""
        tail = line[at:]
        if '"i"(' in tail:
            macro = tail.split('"i"(', 1)[1].split(")", 1)[0].strip()
        name = _enclosing_name(lines, i)
        if not name:
            problems.append(f"line {i + 1}: .insn outside any named function; nothing gives it meaning")
            continue
        if f3 is None:
            problems.append(f"{name}: funct3 operand {toks[1]!r} is not a literal")
            continue
        key = (name, macro, f3, f7)
        if key in seen:
            continue
        seen.add(key)
        out.append(HeaderInsn(name=name, opcode_macro=macro, opcode=defines.get(macro),
                              funct3=f3, funct7=f7, form=form))
    return out, tuple(problems)


def table_for(target: str, endpoint) -> tuple[dict, tuple[str, ...]]:
    """``{(opcode, funct3): name}`` for a target whose endpoint declares an intrinsics header.

    Returns ``({}, problems)`` when nothing is declared or the pin does not verify — the honest
    degradation, with the custom opcodes then reported as unexplained rather than named wrongly.
    """
    try:
        from merlin.common import provenance as _prov
        from merlin.kernels import endpoints as _ep

        # An endpoint that already carries its declaration block hands it over directly; one that only
        # knows its name is looked up. Looking up by name ALWAYS returned nothing for the first kind,
        # and returned it with no problem recorded — an empty table that reads as "this target declares
        # no intrinsics" when it means "we asked the wrong way".
        block = getattr(endpoint, "block", None)
        if not block:
            block = ((_ep._spec().get("endpoints") or {}).get(getattr(endpoint, "name", "")) or {})
        decl = ((block.get("encoding") or {}).get("intrinsics") or {})
        if not decl.get("pin") or not decl.get("path"):
            return {}, ()
        root = Path(_prov.verify(str(decl["pin"])).observed.path)
        path = root / str(decl["path"])
        if not path.is_file():
            return {}, (f"{target}: declared intrinsics header {path} is absent",)
        insns, problems = parse_insn_header(path)
        # Keyed on (opcode, funct3, funct7). Dropping funct7 matched any word in the space whose
        # funct3 coincided with an intrinsic's — measured, that mislabelled a shared opcode space's
        # OTHER occupant as SIMT control, inflating a role count with another engine's instructions.
        table = {}
        for ins in insns:
            if ins.opcode is not None and ins.funct3 is not None:
                table.setdefault((ins.opcode, ins.funct3, ins.funct7 or 0), ins.name)
        return table, problems
    except Exception as exc:  # noqa: BLE001
        return {}, (f"{target}: intrinsics header unreadable ({type(exc).__name__}: {exc})",)
