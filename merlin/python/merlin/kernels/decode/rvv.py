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

    def loop_spans(self) -> list[tuple[int, int]]:
        """[(start_addr, backedge_addr), ...] for every back-edge: the half-open address range a
        loop body covers (its target up to and including the branch). Read structurally from the
        decoded branch target, never from a mnemonic substring. Used by the CCA accumulator-
        residency / register-block inference to scope "inside the K loop"."""
        spans: list[tuple[int, int]] = []
        for i in self.insns:
            tgt = _branch_target(i.raw)
            if tgt is not None and tgt < i.raw.addr:
                spans.append((tgt, i.raw.addr))
        return spans

    def spans_reliable(self) -> bool:
        """False when the branch displacements look UNRELOCATED, so ``loop_spans()`` /
        ``innermost_loop()`` (and every loop-scoped count built on them) cannot be trusted.

        In an unlinked object the branch displacement is still a zero placeholder awaiting
        relocation, so llvm-objdump resolves each branch to ITS OWN address: no target is ever
        ``< addr``, ``loop_spans()`` reads EMPTY, and any loop-scoped count silently collapses to 0
        (MEASURED: a whole-model ``model.o`` reports 0 back-edge spans where the linked ELF reports
        thousands). A branch that resolves to its own address is the smoking gun -- a real
        branch-to-self is an infinite loop no compiler emits in compute code.

        Straight-line code (no resolvable branch) stays reliable=True: there is simply no loop
        structure to get wrong, so ``calls_in_loop`` is a sound 0. Mirrors the honesty contract of
        ``escape_audit.EscapeSite.depth_reliable``: report UNKNOWN rather than a confident wrong 0.
        """
        for i in self.insns:
            if not i.raw.mnemonic.startswith(_BRANCH):
                continue
            tgt = _branch_target(i.raw)
            if tgt is not None and tgt == i.raw.addr:
                return False
        return True

    def innermost_loop(self) -> tuple[int, int] | None:
        """The innermost (smallest-span) back-edge range, or None if straight-line. The innermost
        loop of a tiled GEMM is the K-reduction loop — where accumulator residency is decided."""
        spans = self.loop_spans()
        return min(spans, key=lambda s: s[1] - s[0]) if spans else None

    def innermost_vector_loop(self) -> tuple[int, int] | None:
        """The micro-kernel's vector loop: the FMA-bearing back-edge range, smallest-span if several.

        :meth:`innermost_loop` answers "the tightest loop", which is the K-reduction loop only when
        every loop in the kernel is vectorized. A recipe that PEELS (the VL-agnostic scalable one
        peels N so the main body is unmasked) leaves a REMAINDER loop beside the micro-kernel; that
        remainder is a shorter span, and the compiler may even auto-vectorize it into a
        ``vfredosum`` dot-product — so the plain innermost span, and even "smallest loop with any
        vector op", report the TAIL's mix rather than the micro-kernel's ``vfmacc``.

        The micro-kernel is the loop that issues a fused multiply-add (``vfmacc``/``vfmadd``/the
        integer ``vwmacc``); the remainder dot-product uses ``vfmul``+``vfredosum`` and has none. So
        we scope to FMA-bearing loops and take the smallest of those; if none exists (e.g. an int or
        non-FMA kernel) we fall back to any vector loop, preserving the old behaviour."""
        def _has_fma(span):
            return any(i.is_vector and i.raw.mnemonic.startswith(("vfmacc", "vfmadd", "vwmacc"))
                       for i in self.insns_in(span))
        def _has_vec(span):
            return any(i.is_vector for i in self.insns_in(span))
        fma = [s for s in self.loop_spans() if _has_fma(s)]
        if fma:
            return min(fma, key=lambda s: s[1] - s[0])
        vec = [s for s in self.loop_spans() if _has_vec(s)]
        return min(vec, key=lambda s: s[1] - s[0]) if vec else None

    def insns_in(self, span: tuple[int, int]) -> list["VInsn"]:
        lo, hi = span
        return [i for i in self.insns if lo <= i.raw.addr <= hi]

    def count_in(self, span: tuple[int, int], *mnemonic_prefixes: str) -> int:
        return sum(1 for i in self.insns_in(span)
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


def _branch_target(insn: RawInsn) -> int | None:
    """Resolve a branch/jump's target byte-address from its operands, robustly.

    llvm-objdump renders the target operand as either a bare hex (``0x1dc``) or a hex with a symbol
    annotation (``0x1dc <forward+0x1dc>``). The OLD code did ``int(operands[-1], 16)`` which silently
    failed on the annotated form (the last comma-split token is the whole ``0x1dc <forward+0x1dc>``),
    so ``has_loop`` reported False for any real disassembled loop. Here we scan the LAST operand for
    its first ``0x…`` token, so both spellings resolve. Non-branch / unresolved -> None."""
    if not insn.mnemonic.startswith(_BRANCH) or not insn.operands:
        return None
    for tok in insn.operands[-1].split():
        try:
            return int(tok, 16)
        except ValueError:
            continue
    return None


def _is_backedge(insn: RawInsn) -> bool:
    """A branch/jump whose target address is < its own address = a loop back-edge."""
    tgt = _branch_target(insn)
    return tgt is not None and tgt < insn.addr


def _stream_from_raws(raws) -> InsnStream:
    """Build an InsnStream (per-insn effective vtype) from decoded RawInsns — shared by decode /
    decode_text."""
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


def decode(obj_path, triple: str = "riscv64") -> InsnStream:
    """Object -> InsnStream with per-insn effective vtype (RVV/vector facet)."""
    return _stream_from_raws(tokenize(obj_path, triple=triple))


def decode_text(text: str) -> InsnStream:
    """Already-disassembled objdump text -> InsnStream (no toolchain needed). Lets a CCA be lifted
    from a saved objdump.txt (e.g. a beam fork's generated/objdump.txt) via cca.lift_asm(decode_text(...))."""
    from .objdump import tokenize_text
    return _stream_from_raws(tokenize_text(text))
