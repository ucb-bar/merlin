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


#: Key for the whole-stream bucket in the per-section index. A NUL cannot appear in an objdump symbol
#: name, so it cannot collide with a real section -- including the empty section a stream built
#: directly from RawInsn carries.
_ALL_SECTIONS = "\x00all"


@dataclass
class InsnStream:
    insns: list[VInsn] = field(default_factory=list)
    # addr -> enclosing section header, built once on first use. Without it every span resolution is
    # a full scan of the stream, and the callers ask per span: on a linked whole-model ELF (192k
    # instructions, 2138 back-edges) that turned an instant `loop_spans()` into 16 s.
    _section_at: dict[int, str] | None = field(default=None, repr=False, compare=False)
    # section -> (addresses, instructions) in stream order, plus whether the addresses ascend, built
    # once on first use. Same reason as `_section_at` one line up, for the other per-span scan:
    # `insns_in` filtered the WHOLE stream per call, and its callers ask per span. MEASURED on a
    # 30,397-instruction int8 whole-model objdump: `cca.lift_asm` spent 33.3 s of 33.9 s inside
    # `insns_in` across 5,672 calls, which made the beam's per-fork CCA audit ~55 s and a
    # 30-fork generation ~27 minutes.
    _by_section: "dict[str, tuple[list[int], list[VInsn], bool]] | None" = field(
        default=None, repr=False, compare=False)

    def _section_index(self) -> dict[int, str]:
        if self._section_at is None:
            # First occurrence wins, matching a linear scan that stops at the first match -- the two
            # differ only where one address appears twice, which a linked image does not do.
            idx: dict[int, str] = {}
            for i in self.insns:
                idx.setdefault(i.raw.addr, i.raw.section)
            self._section_at = idx
        return self._section_at

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
            if not _is_branch(i.raw.mnemonic):
                continue
            tgt = _branch_target(i.raw)
            if tgt is not None and tgt == i.raw.addr:
                return False
        return True

    def functions(self) -> tuple[str, ...]:
        """The enclosing function symbols present in the decoded stream, in address order.

        ``RawInsn.section`` carries the objdump section/symbol header verbatim
        (``0000000000000000 <forward>``); this reduces it to the bare symbol so callers can scope an
        analysis to one function. Empty tuple when the disassembly carries no symbol headers."""
        seen: list[str] = []
        for i in self.insns:
            fn = _function_of(i.raw.section)
            if fn and fn not in seen:
                seen.append(fn)
        return tuple(seen)

    def in_function(self, name: str) -> "InsnStream":
        """A sub-stream holding only the instructions of function ``name`` — the caller-driven way to
        scope every loop query below to the function actually under analysis."""
        return InsnStream(insns=[i for i in self.insns if _function_of(i.raw.section) == name])

    def kernel_loop_spans(self) -> list[tuple[int, int]]:
        """``loop_spans()`` minus the loops that live in compiler-emitted support functions.

        An object file is not one function. Besides the model's own code it carries whatever the
        lowering pipeline emitted beside it, and those helpers have loops of their own — MLIR's
        ``bufferization-lower-deallocations`` emits a ``dealloc_helper`` whose 3-instruction scalar
        loop is far tighter than any GEMM K-loop. Since :meth:`innermost_loop` selects by SMALLEST
        span, adding that pass silently moved "the innermost loop" from the micro-kernel to the
        deallocator: MEASURED on a 64x64x64 matmul, ``innermost_loop()`` returned ``(500, 506)`` in
        ``<dealloc_helper>`` while the real K-loop ``(164, 206)`` in ``<forward>`` — with its 4
        ``vfmacc.vf`` and zero spills — was untouched. Every loop-scoped count then reads the wrong
        body, and ``_lift_envelope`` additionally reclassifies the true K-loop as an OUTER loop and
        counts its calls as per-tile overhead.

        Support functions are named rather than inferred, the same way ``cca.RUNTIME_ESCAPE_SYMBOLS``
        names the runtime helpers it recognizes: which helpers exist is a property of the passes WE
        run, not of the target. Fail-safe — if the exclusion would leave nothing, the unfiltered
        spans are returned, so an object that is *only* a helper still reports its structure."""
        spans = self.loop_spans()
        kept = [s for s in spans if _function_of(self._span_section(s)) not in COMPILER_SUPPORT_FUNCTIONS]
        return kept or spans

    def innermost_loop(self) -> tuple[int, int] | None:
        """The innermost (smallest-span) back-edge range, or None if straight-line. The innermost
        loop of a tiled GEMM is the K-reduction loop — where accumulator residency is decided.

        Scoped to :meth:`kernel_loop_spans`, so a compiler-emitted support helper's tiny loop cannot
        win the smallest-span contest against the micro-kernel."""
        spans = self.kernel_loop_spans()
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
        fma = [s for s in self.kernel_loop_spans() if _has_fma(s)]
        if fma:
            return min(fma, key=lambda s: s[1] - s[0])
        vec = [s for s in self.kernel_loop_spans() if _has_vec(s)]
        return min(vec, key=lambda s: s[1] - s[0]) if vec else None

    def _span_section(self, span: tuple[int, int]) -> str:
        """The objdump section header enclosing ``span``, read off the back-edge instruction at its
        high end (falling back to the first instruction in range). ``""`` when the disassembly
        carries no symbol headers, which makes every function filter below a no-op."""
        lo, hi = span
        idx = self._section_index()
        if hi in idx:
            return idx[hi]
        for i in self.insns:                    # span not anchored on an instruction: fall back
            if lo <= i.raw.addr <= hi:
                return i.raw.section
        return ""

    def _section_buckets(self) -> "dict[str, tuple[list[int], list[VInsn], bool]]":
        """Instructions grouped by section in stream order, with an ascending-address flag.

        ``_ALL_SECTIONS`` holds the WHOLE stream, which is what a span with no resolvable section must
        be filtered against (an object with no symbol headers).
        """
        if self._by_section is None:
            groups: dict[str, list[VInsn]] = {}
            for i in self.insns:
                groups.setdefault(i.raw.section, []).append(i)
            # The whole-stream bucket goes under a key NO section can equal. Using "" for it collided
            # with instructions whose section IS "" -- a stream built straight from RawInsn, as the
            # hermetic tests do -- so `setdefault("").append` appended the stream to itself and every
            # loop-scoped count came back exactly DOUBLED (fma_in_loop 2 for a one-FMA loop). The real
            # objdump fixtures all carry symbol headers, which is why only a synthetic stream caught it.
            groups[_ALL_SECTIONS] = list(self.insns)
            built: dict[str, tuple[list[int], list[VInsn], bool]] = {}
            for name, items in groups.items():
                addrs = [i.raw.addr for i in items]
                ascending = all(a <= b for a, b in zip(addrs, addrs[1:]))
                built[name] = (addrs, items, ascending)
            self._by_section = built
        return self._by_section

    def insns_in(self, span: tuple[int, int]) -> list["VInsn"]:
        """The instructions of ``span``, confined to the function the span belongs to.

        Addresses are unique across an object only by convention; a range must not be allowed to
        swallow the tail of a neighbouring function just because the two are adjacent. Anchoring on
        the span's own section keeps every loop-scoped count (residency, spills, register block)
        about one function. Degrades to the plain address filter when there are no symbol headers.

        Bisects a per-section index rather than scanning the stream. Within one section a disassembly
        lists addresses in ascending order, so the span is a contiguous slice; where that does not
        hold (a stream assembled out of order) the flag is False and the section is filtered linearly,
        which is still only that section rather than everything. Results are IDENTICAL to the scan,
        stream order included -- asserted against it in the tests.
        """
        import bisect
        lo, hi = span
        sect = self._span_section(span)
        addrs, items, ascending = self._section_buckets().get(sect or _ALL_SECTIONS,
                                                              ([], [], True))
        if not ascending:
            return [i for i in items if lo <= i.raw.addr <= hi]
        return items[bisect.bisect_left(addrs, lo):bisect.bisect_right(addrs, hi)]

    def count_in(self, span: tuple[int, int], *mnemonic_prefixes: str) -> int:
        return sum(1 for i in self.insns_in(span)
                   if any(i.raw.mnemonic.startswith(p) for p in mnemonic_prefixes))


# Functions the LOWERING emits beside the model's own code, whose loops are not model compute. Named
# rather than inferred, exactly as ``cca.RUNTIME_ESCAPE_SYMBOLS`` names the runtime helpers it knows:
# a disassembly gives no derivable way to tell "kernel" from "support", but which support routines
# can appear IS a property of the passes we run, and each entry below traces to one of them.
#   dealloc_helper -- emitted by MLIR's ``bufferization-lower-deallocations`` (see
#                     ``llvmlower.pipeline._dealloc_passes``); a scalar retain/free loop.
COMPILER_SUPPORT_FUNCTIONS = ("dealloc_helper",)


def _function_of(section: str) -> str:
    """``"0000000000000000 <forward>"`` -> ``"forward"``. Structured split on the angle brackets
    objdump writes; ``""`` when the header carries no symbol (then no function filter applies)."""
    _, _, rest = section.partition("<")
    name, sep, _ = rest.partition(">")
    return name if sep else ""


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


def _is_branch(mnemonic: str) -> bool:
    """Whether a mnemonic is a branch/jump, COMPRESSED forms included.

    The compressed encodings are the same operations under a ``c.`` prefix, and matching the prefix list
    directly missed every one of them: ``c.bnez`` does not start with ``bnez``. That mattered a lot more
    than it looks. ``rv64gcv`` includes the C extension, so at -O2 a loop's back-edge is routinely
    ``c.bnez`` — which made ``loop_spans()`` omit the innermost loop, ``_fma_loop`` fall back to an
    enclosing one (or to None), and every loop-scoped count, accumulator residency included, answer for
    the wrong region while looking perfectly healthy.

    Register-indirect forms (``c.jr``/``c.jalr``) resolve to no static target, so they are matched here
    and then declined by :func:`_branch_target` — which is the correct outcome, not a special case.
    """
    return mnemonic.removeprefix("c.").startswith(_BRANCH)


def _branch_target(insn: RawInsn) -> int | None:
    """Resolve a branch/jump's target byte-address from its operands, robustly.

    llvm-objdump renders the target operand as either a bare hex (``0x1dc``) or a hex with a symbol
    annotation (``0x1dc <forward+0x1dc>``). The OLD code did ``int(operands[-1], 16)`` which silently
    failed on the annotated form (the last comma-split token is the whole ``0x1dc <forward+0x1dc>``),
    so ``has_loop`` reported False for any real disassembled loop. Here we scan the LAST operand for
    its first ``0x…`` token, so both spellings resolve. Non-branch / unresolved -> None."""
    if not _is_branch(insn.mnemonic) or not insn.operands:
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


def decode(obj_path, triple: str = "riscv64", mattr: str | None = None) -> InsnStream:
    """Object -> InsnStream with per-insn effective vtype (RVV/vector facet)."""
    return _stream_from_raws(tokenize(obj_path, triple=triple, mattr=mattr))


def decode_text(text: str) -> InsnStream:
    """Already-disassembled objdump text -> InsnStream (no toolchain needed). Lets a CCA be lifted
    from a saved objdump.txt (e.g. a beam fork's generated/objdump.txt) via cca.lift_asm(decode_text(...))."""
    from .objdump import tokenize_text
    return _stream_from_raws(tokenize_text(text))
