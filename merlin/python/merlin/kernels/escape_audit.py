"""Attribute RUNTIME ESCAPES in emitted code to their call sites, scoped and loop-depth-tagged.

A *runtime escape* is a call the compiler emitted instead of code: `memrefCopy`, `memcpy`, `memset`,
`malloc`, … (:data:`merlin.kernels.cca.RUNTIME_ESCAPE_SYMBOLS`). An expert kernel calls none of them.
The CCA envelope facet (:func:`merlin.kernels.cca._lift_envelope`) answers "does this region escape?"
for one lifted region; this module answers what a cross-cutting audit needs: **which helper, called
from which compiler-emitted function, at what loop depth**.

Two facts about the emitted artifacts dictate the design, and getting either wrong yields a
confidently WRONG clean bill of health:

1. **Loop depth must be read from the LINKED ELF.** In the unlinked ``model.o`` every branch
   displacement is still a zero placeholder awaiting relocation, so each branch appears to target
   itself, no target is ever ``< addr``, and ``loop_spans()`` reads EMPTY. A loop-scoped facet lifted
   from ``model.o`` therefore reports "no calls in loops" for a kernel whose hot loop is nothing but
   calls. (Measured: the same object reports 0 back-edge spans; the linked ELF reports 6017.)
2. **Escapes must be scoped to the compiler-emitted functions.** The linked ELF statically contains
   libc, whose own internals call ``memcpy``/``malloc`` constantly. Counting escapes over the whole
   binary drowns the signal in library noise. The compute region is exactly the set of functions
   defined by ``model.o`` (``forward``, ``_mlir_ciface_forward``, …), which we read from the object.

So: **names and scope from the object, structure from the ELF.**

Cost intuition the matrix relies on: a site's loop DEPTH, not the site COUNT, is what predicts cost.
The defect this module generalizes was a single ``memrefCopy`` site -- ranked harmless by count, but
sitting at depth > 0 it ran once per output tile and accounted for 77% of the kernel's instructions.

HONESTY CONTRACT (mirrors :func:`merlin.kernels.decode.objdump.undefined_symbols`): every field is
``None`` when it could not be READ and a possibly-empty tuple/dict when it was read and found empty.
An unreadable artifact must never be reported as "no escapes".
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .cca import RUNTIME_ESCAPE_SYMBOLS
from .decode.objdump import nm_bin, objdump_bin, undefined_symbols


@dataclass(frozen=True)
class EscapeSite:
    """One emitted call to a runtime helper, inside a compiler-emitted function."""

    helper: str               # the runtime helper called, e.g. "memrefCopy"
    caller: str               # enclosing compiler-emitted function, e.g. "forward"
    addr: int                 # call-site address in the linked ELF
    loop_depth: int           # number of back-edge spans enclosing this address (0 = straight-line)
    depth_reliable: bool = True   # False when the enclosing spans do not form a containment chain

    @property
    def in_loop(self) -> bool:
        return self.loop_depth > 0


@dataclass
class EscapeReport:
    """The runtime escapes of one emitted artifact pair (object + linked ELF).

    ``sites`` is ``None`` when the artifacts could not be read (UNKNOWN), never silently empty.
    """

    obj: str
    elf: str | None
    scope: tuple[str, ...] | None          # compiler-emitted functions the audit covered
    undefined: tuple[str, ...] | None      # nm -u of the object
    sites: tuple[EscapeSite, ...] | None
    loops_seen: int | None = None          # back-edge spans found across the scope functions

    @property
    def readable(self) -> bool:
        return self.sites is not None and self.scope is not None

    @property
    def loop_structure_suspect(self) -> bool:
        """True when the scope contains NO loop at all -- which for a compute kernel means the loop
        structure was probably unreadable rather than absent, so every depth read 0 and every escape
        was filed as harmless prologue.

        This is not hypothetical: on the K1 Linux build path the emitted ``model.o`` still has
        UNRELOCATED branch displacements (each branch literally targets itself), so an audit of the
        object rather than the linked ELF sees zero back-edges and pronounces a per-tile ``memrefCopy``
        to be a one-off. Callers should treat this flag as "re-check the artifact", not as a result.
        """
        return bool(self.readable and not self.loops_seen)

    @property
    def helpers(self) -> tuple[str, ...] | None:
        """Escape helpers the object references at all, or None if unreadable."""
        if self.undefined is None:
            return None
        return tuple(sorted(set(self.undefined).intersection(RUNTIME_ESCAPE_SYMBOLS)))

    def site_counts(self) -> dict[str, int] | None:
        """helper -> emitted call sites inside the compute scope."""
        return self._counts(lambda s: True)

    def in_loop_counts(self) -> dict[str, int] | None:
        """helper -> call sites INSIDE a loop body: the per-iteration escapes, the ones whose cost
        scales with the problem size and the only ones worth chasing."""
        return self._counts(lambda s: s.in_loop)

    def _counts(self, keep) -> dict[str, int] | None:
        if self.sites is None:
            return None
        counts: dict[str, int] = {}
        for s in self.sites:
            if keep(s):
                counts[s.helper] = counts.get(s.helper, 0) + 1
        return counts

    def max_depth(self) -> int | None:
        if self.sites is None:
            return None
        return max((s.loop_depth for s in self.sites), default=0)

    @property
    def depths_reliable(self) -> bool | None:
        """False when any in-loop site's enclosing spans do not nest, so its depth NUMBER is not
        trustworthy. ``in_loop`` itself still stands -- a back-edge does enclose the call."""
        if self.sites is None:
            return None
        return all(s.depth_reliable for s in self.sites if s.in_loop)


def emitted_functions(obj_path: str | Path) -> tuple[str, ...] | None:
    """Function symbols DEFINED by the compiler-emitted object -- the audit's compute scope.

    Returns None if the symbol table could not be read (never (), which would silently narrow the
    scope to nothing and make every binary look escape-free).
    """
    try:
        p = subprocess.run([nm_bin(), "--defined-only", str(obj_path)],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    out: list[str] = []
    for line in p.stdout.splitlines():
        parts = line.split()
        # "<addr> T <name>" -- text-section (function) symbols only.
        if len(parts) >= 3 and parts[-2].upper() == "T":
            out.append(parts[-1])
    return tuple(sorted(set(out)))


def _disasm(path: str | Path) -> str | None:
    cmd = [objdump_bin(), "-d", "--triple=riscv64", "-M", "no-aliases", str(path)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def _callee(operands: list[str]) -> str | None:
    """Callee symbol of a call instruction, from llvm-objdump's ``<name>`` annotation on the target
    operand (``jal ra, 0x21360 <memset>``). None when the target carries no symbol annotation."""
    if not operands:
        return None
    last = operands[-1]
    if "<" not in last or ">" not in last:
        return None
    name = last[last.index("<") + 1:last.rindex(">")]
    # A within-function target is annotated "<forward+0x30>" -- not a call to a named symbol.
    return None if "+" in name else name


def audit(obj_path: str | Path, elf_path: str | Path) -> EscapeReport:
    """Audit one build: escapes inside the compiler-emitted functions, tagged with loop depth.

    ``obj_path`` is the compiler's object (names + scope); ``elf_path`` is the linked binary, which
    is the only artifact with real branch displacements and therefore real loop structure.
    """
    obj_path, elf_path = Path(obj_path), Path(elf_path)
    scope = emitted_functions(obj_path)
    undef = undefined_symbols(obj_path)
    text = _disasm(elf_path)
    if scope is None or text is None:
        return EscapeReport(obj=str(obj_path), elf=str(elf_path), scope=scope,
                            undefined=undef, sites=None)

    in_scope = set(scope)
    # Group instructions by enclosing function: loops are INTRA-procedural, so a function's loop
    # depth must be computed from its own back-edges only. Taking back-edges over the whole linked
    # ELF instead lets a long backward jump anywhere in libc manufacture a span that encloses our
    # code, which inflates every depth (observed: a prologue malloc "at depth 40").
    per_fn: dict[str, list[tuple[int, str, list[str]]]] = {}
    caller = ""
    for addr, sym, mnemonic, operands in _tokenize_with_symbols(text):
        if sym is not None:
            caller = sym
            continue
        if caller in in_scope:
            per_fn.setdefault(caller, []).append((addr, mnemonic, operands))

    sites: list[EscapeSite] = []
    loops_seen = 0
    for fn, insns in per_fn.items():
        spans = _backedge_spans(insns)
        loops_seen += len(spans)
        for addr, mnemonic, operands in insns:
            if not mnemonic.startswith(("jal", "c.jal", "call")):
                continue
            callee = _callee(operands)
            if callee is None or callee not in RUNTIME_ESCAPE_SYMBOLS:
                continue
            enclosing = [(lo, hi) for lo, hi in spans if lo <= addr <= hi]
            sites.append(EscapeSite(helper=callee, caller=fn, addr=addr,
                                    loop_depth=len(enclosing),
                                    depth_reliable=_is_containment_chain(enclosing)))
    sites.sort(key=lambda s: s.addr)
    return EscapeReport(obj=str(obj_path), elf=str(elf_path), scope=scope,
                        undefined=undef, sites=tuple(sites), loops_seen=loops_seen)


def _is_containment_chain(spans: list[tuple[int, int]]) -> bool:
    """True when the spans nest properly (each contained in the next) -- the condition under which
    counting enclosing spans is a sound loop DEPTH.

    A real loop nest is nested or disjoint, never partially overlapping, so overlapping spans mean
    the address-range approximation has broken down: after loop rotation and block reordering a
    loop's body need not be address-contiguous, and a back-edge's [target, latch] range can then
    cover code belonging to a sibling loop. Observed on the int8 small-M path, where a call had four
    "enclosing" spans whose starts AND ends both increased -- an impossible nest. The call really is
    inside a loop there, but the number 4 is not a depth, and reporting it as one would be inventing
    precision the instrument does not have.
    """
    ordered = sorted(spans, key=lambda s: s[1] - s[0])
    return all(ordered[i + 1][0] <= ordered[i][0] and ordered[i][1] <= ordered[i + 1][1]
               for i in range(len(ordered) - 1))


def _backedge_spans(insns: list[tuple[int, str, list[str]]]) -> list[tuple[int, int]]:
    """Back-edge spans ``(target, branch)`` within ONE function: branches whose resolved target is a
    lower address inside the same function. Reuses the decoder's branch-target resolution so both
    objdump spellings (bare hex, and hex with a ``<sym+off>`` annotation) are handled."""
    from .decode.objdump import RawInsn
    from .decode.rvv import _branch_target

    lo_addr = min((a for a, _, _ in insns), default=0)
    spans: list[tuple[int, int]] = []
    for addr, mnemonic, operands in insns:
        tgt = _branch_target(RawInsn(addr=addr, mnemonic=mnemonic, operands=operands))
        if tgt is not None and lo_addr <= tgt < addr:
            spans.append((tgt, addr))
    return spans


def _tokenize_with_symbols(text: str):
    """Yield ``(addr, symbol_or_None, mnemonic, operands)`` over a disassembly, structurally.

    A function header ``0000000000010974 <memrefCopy>:`` yields ``(addr, name, "", [])`` so callers
    can track the enclosing function; instruction lines yield ``(addr, None, mnemonic, operands)``.
    """
    for line in text.splitlines():
        s = line.strip()
        if not s or ":" not in s:
            continue
        if s.endswith(">:") and "<" in s:
            head = s.split()[0]
            try:
                addr = int(head, 16)
            except ValueError:
                continue
            yield addr, s[s.index("<") + 1:s.rindex(">")], "", []
            continue
        left, _, right = s.partition(":")
        try:
            addr = int(left.strip(), 16)
        except ValueError:
            continue
        parts = right.strip().split(None, 2)
        if len(parts) < 2:
            continue
        try:
            int(parts[0], 16)              # encoding word; otherwise not an instruction line
        except ValueError:
            continue
        operands = [o.strip() for o in parts[2].split(",")] if len(parts) == 3 else []
        yield addr, None, parts[1], operands
