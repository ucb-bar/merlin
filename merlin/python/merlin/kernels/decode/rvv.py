"""RVV (vector) semantic decoder — the first per-ISA instantiation over ``decode/objdump``.

Consumes ISA-agnostic ``RawInsn`` and runs a **vtype-state machine**: the SEW/LMUL/tail/mask are
read from the *explicit operands* of ``vsetvli``/``vsetivli``/``vsetvl`` (the ISA's own canonical
vtype spelling, e.g. ``e32, m2, ta, ma``) — never guessed from a mnemonic substring. Each vector
instruction is annotated with the *effective* vtype in force at its point. Output: an
``InsnStream`` that the CCA `vector` facet lifts from. This is the robust replacement for the
regex-over-objdump-text in ``build_asm``/``fingerprint``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .objdump import RawInsn, tokenize


@dataclass(frozen=True)
class VType:
    sew: int | None = None        # element width in bits (8/16/32/64)
    lmul: float | None = None     # group multiplier: 8,4,2,1, 0.5, 0.25, 0.125 (mf2/mf4/mf8)
    tail: str | None = None       # "ta" | "tu"
    mask: str | None = None       # "ma" | "mu"

    def __str__(self) -> str:
        if self.sew is None:
            return "vtype?"
        lm = (f"m{int(self.lmul)}" if self.lmul and self.lmul >= 1
              else f"mf{int(1/self.lmul)}" if self.lmul else "m?")
        return f"e{self.sew}{lm}{self.tail or ''}{self.mask or ''}"


@dataclass
class VInsn:
    raw: RawInsn
    is_vector: bool
    vtype: VType | None           # effective vtype at this instruction (vector insns only)


@dataclass
class InsnStream:
    insns: list[VInsn] = field(default_factory=list)

    def vector_histogram(self) -> dict[str, int]:
        return dict(Counter(i.raw.mnemonic for i in self.insns if i.is_vector))

    def vtype_histogram(self) -> dict[str, int]:
        """How many vector insns ran under each effective (SEW,LMUL,...) — the real LMUL usage,
        not a guess. The dominant entry is the kernel's working vtype."""
        return dict(Counter(str(i.vtype) for i in self.insns if i.is_vector and i.vtype))

    def has_loop(self) -> bool:
        return any(_is_backedge(i.raw) for i in self.insns)

    def count(self, *mnemonic_prefixes: str) -> int:
        return sum(1 for i in self.insns
                   if any(i.raw.mnemonic.startswith(p) for p in mnemonic_prefixes))


# vsetvl operand tokens carry the canonical vtype: e<SEW>, m<LMUL>|mf<frac>, ta|tu, ma|mu.
def _parse_vtype(operands: list[str]) -> VType:
    sew = lmul = tail = mask = None
    for tok in operands:
        if tok.startswith("e") and tok[1:].isdigit():
            sew = int(tok[1:])
        elif tok.startswith("mf") and tok[2:].isdigit():
            lmul = 1.0 / int(tok[2:])
        elif tok.startswith("m") and tok[1:].isdigit():
            lmul = float(int(tok[1:]))
        elif tok in ("ta", "tu"):
            tail = tok
        elif tok in ("ma", "mu"):
            mask = tok
    return VType(sew=sew, lmul=lmul, tail=tail, mask=mask)


_VSET = ("vsetvli", "vsetivli", "vsetvl")
_BRANCH = ("beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez", "bgez", "blez",
           "bgtz", "bltz", "j", "jal")


def _is_backedge(insn: RawInsn) -> bool:
    """A branch/jump whose target address is < its own address = a loop back-edge."""
    if not insn.mnemonic.startswith(_BRANCH) or not insn.operands:
        return False
    tgt = insn.operands[-1]
    try:
        return int(tgt, 16) < insn.addr
    except ValueError:
        return False


def decode(obj_path, triple: str = "riscv64") -> InsnStream:
    """Object -> InsnStream with per-insn effective vtype (RVV/vector facet)."""
    raws = tokenize(obj_path, triple=triple)
    cur = VType()
    out: list[VInsn] = []
    for r in raws:
        if r.mnemonic in _VSET:
            cur = _parse_vtype(r.operands)
            out.append(VInsn(raw=r, is_vector=True, vtype=cur))
            continue
        is_vec = r.mnemonic.startswith("v")          # RVV mnemonics are v-prefixed
        out.append(VInsn(raw=r, is_vector=is_vec, vtype=cur if is_vec else None))
    return InsnStream(insns=out)
