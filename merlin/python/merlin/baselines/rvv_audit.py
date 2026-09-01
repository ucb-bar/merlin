"""RVV-coverage audit — the mechanical honesty behind "push RVV, label scalar fallback".

We do not take a framework's word that it "used RVV". We disassemble the emitted binary/object
(``objdump -d``) and classify every instruction as **vector** (RVV) or **scalar**, per symbol, and
report a coverage fraction + an explicit list of compute-bearing symbols that fell back to scalar.
That per-region RVV% and fallback table is what every :class:`~merlin.baselines.contract.BaselineResult`
carries, so scalar fallback is recorded rather than averaged away.

Pure-stdlib and **structural**: the classifier operates on disassembly *text*, so it is unit-testable
without a board or even a binary (feed it a fixture ``.s``/objdump dump). Lines are taken apart with
``str.partition`` + index walks rather than regex — the objdump line shape differs between GNU and
LLVM (see :func:`_insn_mnemonic`) and a pattern narrow enough for one silently drops the other, which
is exactly how a vector-heavy binary gets reported as 0% RVV.

Why not pyelftools: the unit of measurement here is a *decoded instruction attributed to a symbol*,
which only the disassembler produces. pyelftools would supply symbol addresses (which the objdump
``<sym>:`` headers already carry) and no instruction stream, so it would replace nothing; it is also
not a declared dependency of this repo and is not installed. The ``decode/`` package parses a
different thing (RoCC/custom-opcode traces), not rv64gcv disassembly.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_HEX = frozenset("0123456789abcdef")        # objdump emits lowercase hex, as the old `[0-9a-f]` assumed
_LOWER = frozenset("abcdefghijklmnopqrstuvwxyz")
_MNEMONIC_CHARS = _LOWER | frozenset("0123456789._")      # the old mnemonic tail `[a-z0-9._]`

# Scalar *compute* mnemonic prefixes (the denominator for coverage). Excludes pure control-flow /
# jumps / nops / fences, which are neither vector nor "work we could have vectorized".
#
# The old detector was a PREFIX regex (`re.match`, no `$`):
#   ^(f[a-z]|add|addi|addw|addiw|sub|subw|mul|mulw|mulh|div|divw|divu|rem|remu|and|andi|or|ori|
#     xor|xori|sll|slli|sllw|srl|srli|srlw|sra|srai|sraw|slt|slti|sltu|
#     l[bhwd]|lbu|lhu|lwu|s[bhwd]|fl[wd]|fs[wd]|neg|not|mv|sext|zext)
# Because it matched a PREFIX, most alternatives were redundant longer spellings of a shorter one
# already listed: `add` subsumes addi/addw/addiw, `sub` subsumes subw, `mul` subsumes mulw/mulh,
# `div` subsumes divw/divu, `rem` subsumes remu, `and` subsumes andi, `or` subsumes ori, `xor`
# subsumes xori, `sll`/`srl`/`sra`/`slt` subsume their i/w/u forms, `l[bhwd]` subsumes lbu/lhu/lwu,
# and `f[a-z]` subsumes fl[wd]/fs[wd]. The tuple below is that alternation with the redundancy folded
# out, so it accepts EXACTLY the same set of mnemonics.
_SCALAR_COMPUTE_PREFIXES = ("add", "sub", "mul", "div", "rem", "and", "or", "xor",
                            "sll", "srl", "sra", "slt", "neg", "not", "mv", "sext", "zext")
_LOAD_STORE_WIDTHS = frozenset("bhwd")      # the old `l[bhwd]` / `s[bhwd]` character classes


def _is_rvv(mnemonic: str) -> bool:
    """RVV instructions are the only base-RISC-V mnemonics that start with 'v' followed by a letter
    (vsetvli/vsetivli, vle*/vse*, vadd/vfmacc/vmul/vredsum/vmv/...). Scalar float ops start with 'f',
    integer/branch/mem ops never start with 'v'.

    The old detector was ``^v[a-z][a-z0-9._]*$``. Its tail was already guaranteed by the line parser
    (which only ever yields mnemonics over that alphabet), so "'v' + a lowercase letter" is the whole
    test. Checked against a live spike ELF disassembly: ``vmv.v.i`` / ``vsetvli`` / ``vle64.v`` count
    as vector; ``li`` / ``fld`` / ``sd`` / ``ret`` do not.
    """
    return len(mnemonic) >= 2 and mnemonic[0] == "v" and mnemonic[1] in _LOWER


def _is_scalar_compute(mnemonic: str) -> bool:
    """Scalar arithmetic / load / store mnemonics — the coverage denominator (see the prefix table)."""
    if len(mnemonic) < 2:
        return False
    if mnemonic[0] == "f" and mnemonic[1] in _LOWER:                # f[a-z] : every scalar FP op
        return True
    if mnemonic[0] in "ls" and mnemonic[1] in _LOAD_STORE_WIDTHS:   # l[bhwd] / s[bhwd]
        return True
    return mnemonic.startswith(_SCALAR_COMPUTE_PREFIXES)


def _insn_mnemonic(line: str) -> str | None:
    r"""The mnemonic of one objdump instruction line, or ``None`` when the line is not one.

    Both disassemblers we accept put the mnemonic after the TAB that terminates the raw-bytes
    column, but they space the prefix differently. These are REAL lines, captured by disassembling a
    live spike ELF built from the saturn ``vec-igemm`` benchmark (shown as ``repr()`` so the tabs are
    visible)::

        GNU  riscv64-unknown-elf-objdump -d : '    80002000:\t5e003057          \tvmv.v.i\tv0,0'
        LLVM llvm-objdump -d                : '80002000: 5e003057     \tvmv.v.i\tv0, 0x0'

    The old pattern was ``^\s*[0-9a-f]+:\s+[0-9a-f ]+\t([a-z][a-z0-9._]*)`` — optional leading
    whitespace, a lowercase-hex address, ':', at least one whitespace character, a run of
    hex-digits-and-spaces (the raw-bytes column), a TAB, then the mnemonic. The walk below
    reproduces that acceptance exactly. A TAB cannot appear inside the ``[0-9a-f ]+`` run, so that
    run always ends at the very TAB the pattern requires: there is nothing for the old regex's
    backtracking to explore and a single left-to-right scan is equivalent.

    Still REFUSED, exactly as before: a ``--no-show-raw-insn`` dump (no bytes column). That is a
    fail-closed refusal, never a silent zero — such a dump yields no instructions at all, so
    ``coverage_overall`` comes back ``None`` (unknown) rather than 0.0.
    """
    addr, sep, tail = line.partition(":")
    if not sep:
        return None
    stripped = addr.lstrip()
    if addr[:len(addr) - len(stripped)].strip():        # `^\s*` : the lead must be pure whitespace
        return None
    if not stripped or any(c not in _HEX for c in stripped):        # `[0-9a-f]+`
        return None
    ws_end = 0
    while ws_end < len(tail) and tail[ws_end].isspace():            # the `\s+` run
        ws_end += 1
    if ws_end == 0:
        return None
    # `\s+` is greedy but backtracks, and a SPACE belongs to `[0-9a-f ]+` too: when the bytes
    # column is spaces-only the old pattern took one space for `\s+` and let `[0-9a-f ]+` absorb
    # the rest. Walking the split point from longest to shortest reproduces that search order.
    for start in range(ws_end, 0, -1):
        j = start
        while j < len(tail) and (tail[j] in _HEX or tail[j] == " "):    # `[0-9a-f ]+` (>= 1)
            j += 1
        # Shortening THIS run can never help: it would end on a hex/space character, and the
        # pattern requires a TAB right there. So only the `\s+` split point needs backtracking.
        if j == start or j >= len(tail) or tail[j] != "\t":  # ... terminated by the TAB
            continue
        rest = tail[j + 1:]
        if not rest or rest[0] not in _LOWER:           # `[a-z]`
            continue
        k = 1
        while k < len(rest) and rest[k] in _MNEMONIC_CHARS:  # `[a-z0-9._]*`
            k += 1
        return rest[:k]
    return None


def _symbol_name(line: str) -> str | None:
    r"""The symbol of an objdump symbol header, or ``None`` when the line is not one.

    Real line, byte-identical from GNU objdump and llvm-objdump on the live spike ELF::

        '0000000080000000 <_start>:'

    The old pattern was ``^[0-9a-f]+\s+<([^>]+)>:`` — lowercase hex from column 0 (no leading
    whitespace allowed), whitespace, then a non-empty ``<name>`` immediately followed by ':'. Since
    ``[^>]+`` cannot span a '>', the first '>' is always the closing one and ``partition('>')`` is
    exact.
    """
    i = 0
    while i < len(line) and line[i] in _HEX:            # `^[0-9a-f]+`
        i += 1
    if i == 0:
        return None
    j = i
    while j < len(line) and line[j].isspace():          # `\s+` (at least one)
        j += 1
    if j == i or j >= len(line) or line[j] != "<":
        return None
    name, sep, rest = line[j + 1:].partition(">")       # `<([^>]+)>` then ':'
    if not sep or not name or not rest.startswith(":"):
        return None
    return name


@dataclass
class SymbolCoverage:
    symbol: str
    vector: int = 0            # RVV instruction count
    scalar_compute: int = 0    # scalar arithmetic/mem instruction count
    total: int = 0             # every decoded instruction (incl. control flow)

    @property
    def coverage(self) -> float | None:
        """Fraction of *compute* instructions that are vector (None if no compute in this symbol)."""
        denom = self.vector + self.scalar_compute
        return (self.vector / denom) if denom else None

    @property
    def is_scalar_fallback(self) -> bool:
        """A compute-bearing symbol with zero vector instructions = a scalar fallback."""
        return self.vector == 0 and self.scalar_compute > 0


@dataclass
class AuditReport:
    by_symbol: dict[str, SymbolCoverage] = field(default_factory=dict)
    source: str = ""           # binary path or "<text>"

    @property
    def vector(self) -> int:
        return sum(s.vector for s in self.by_symbol.values())

    @property
    def scalar_compute(self) -> int:
        return sum(s.scalar_compute for s in self.by_symbol.values())

    @property
    def coverage_overall(self) -> float | None:
        denom = self.vector + self.scalar_compute
        return (self.vector / denom) if denom else None

    def scalar_fallback_symbols(self, *, ignore: tuple[str, ...] = ()) -> list[str]:
        """Compute-bearing symbols that are entirely scalar (candidates for a fallback label).

        ``ignore`` skips known-non-kernel symbols (e.g. libc/CRT) by substring.
        """
        out = []
        for name, sc in sorted(self.by_symbol.items()):
            if any(g in name for g in ignore):
                continue
            if sc.is_scalar_fallback:
                out.append(name)
        return out


def classify_disasm(text: str, *, source: str = "<text>") -> AuditReport:
    """Classify an ``objdump -d`` (or ``.s``) dump into per-symbol vector/scalar counts."""
    report = AuditReport(source=source)
    cur = SymbolCoverage(symbol="<none>")
    report.by_symbol[cur.symbol] = cur
    for line in text.splitlines():
        name = _symbol_name(line)
        if name is not None:
            cur = report.by_symbol.setdefault(name, SymbolCoverage(symbol=name))
            continue
        mnem = _insn_mnemonic(line)
        if mnem is None:
            continue
        cur.total += 1
        if _is_rvv(mnem):
            cur.vector += 1
        elif _is_scalar_compute(mnem):
            cur.scalar_compute += 1
    # drop the placeholder bucket if it stayed empty
    if report.by_symbol.get("<none>") and report.by_symbol["<none>"].total == 0:
        report.by_symbol.pop("<none>", None)
    return report


def _objdump() -> str | None:
    """Find an objdump that decodes rv64gcv CORRECTLY, preferring ``llvm-objdump``.

    Hard-won lesson (confirmed independently by the EXO and ExecuTorch arms): the SpacemiT / GNU
    ``riscv64-unknown-linux-gnu-objdump`` silently MIS-DECODES rv64gcv in bulk ``-d`` mode — it emits
    bare ``.insn`` words (or only a handful of vector mnemonics) for a vector-heavy binary, which
    would fabricate a false ~0% RVV coverage. ``llvm-objdump`` decodes the V extension correctly, so
    we prefer it everywhere (toolchain bin first, then PATH). GNU objdump remains a last resort.
    """
    from merlin.mining import k1
    root = k1._toolchain_root()
    cands = ("llvm-objdump", "riscv64-unknown-linux-gnu-objdump", "objdump")
    if root:
        for cand in cands:
            p = root / "bin" / cand
            if p.is_file():
                return str(p)
    for cand in cands:
        if shutil.which(cand):
            return shutil.which(cand)
    return None


def audit_binary(path: str | Path, *, objdump: str | None = None, timeout: int = 120) -> AuditReport:
    """Disassemble a rv64 ELF/object and classify RVV vs scalar coverage per symbol.

    Raises ``RuntimeError`` if no objdump is available or disassembly fails — never returns a
    fabricated coverage (fail-closed, matching not_run_is_not_pass).
    """
    tool = objdump or _objdump()
    if tool is None:
        raise RuntimeError("no objdump available to audit RVV coverage (install the riscv toolchain)")
    # llvm-objdump and GNU objdump both accept -d; GNU needs no extra flags for symbol headers.
    r = subprocess.run([tool, "-d", str(path)], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{tool} -d failed on {path}: {r.stderr[:200]}")
    return classify_disasm(r.stdout, source=str(path))


def enforce_rvv_march(march: str) -> str:
    """Reject a target march that does not enable the vector extension. Returns march on success.

    "No scalar!" starts at the build flags: a baseline compiled without ``+v``/``v`` can never be
    RVV, so we refuse it loudly rather than silently measure a scalar binary.
    """
    m = march.lower().replace("-", "").replace("_", "")
    # rv64gcv, rv64gc+v via -mattr=+v, or an explicit 'v' extension token.
    has_v = ("v" in m.split("rv64", 1)[-1][:8]) if "rv64" in m else False
    has_v = has_v or "+v" in march.lower() or march.lower().endswith("v")
    if not has_v:
        raise ValueError(f"march {march!r} does not enable RVV (need e.g. rv64gcv or -mattr=+v)")
    return march
