"""RVV-coverage audit — the mechanical honesty behind "push RVV, label scalar fallback".

We do not take a framework's word that it "used RVV". We disassemble the emitted binary/object
(``objdump -d``) and classify every instruction as **vector** (RVV) or **scalar**, per symbol, and
report a coverage fraction + an explicit list of compute-bearing symbols that fell back to scalar.
That per-region RVV% and fallback table is what every :class:`~merlin.baselines.contract.BaselineResult`
carries, so scalar fallback is recorded rather than averaged away.

Pure-stdlib + regex; the classifier operates on disassembly *text*, so it is unit-testable without a
board or even a binary (feed it a fixture ``.s``/objdump dump).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# RVV instructions are the only base-RISC-V mnemonics that start with 'v' followed by a letter
# (vsetvli/vsetivli, vle*/vse*, vadd/vfmacc/vmul/vredsum/vmv/…). Scalar float ops start with 'f',
# integer/branch/mem ops never start with 'v'. So `^v[a-z]` is a sound RVV detector for rv64gcv.
_RVV_RE = re.compile(r"^v[a-z][a-z0-9._]*$")

# Scalar *compute* mnemonics (the denominator for coverage). Excludes pure control-flow / jumps /
# nops / fences, which are neither vector nor "work we could have vectorized".
_SCALAR_COMPUTE_RE = re.compile(
    r"^(f[a-z]|add|addi|addw|addiw|sub|subw|mul|mulw|mulh|div|divw|divu|rem|remu|"
    r"and|andi|or|ori|xor|xori|sll|slli|sllw|srl|srli|srlw|sra|srai|sraw|slt|slti|sltu|"
    r"l[bhwd]|lbu|lhu|lwu|s[bhwd]|fl[wd]|fs[wd]|neg|not|mv|sext|zext)",
)

# objdump line:  "   1013c:\t02008557          \tvsetvli\ta0,a1,e32,m1,ta,ma"
_INSN_LINE = re.compile(r"^\s*[0-9a-f]+:\s+[0-9a-f ]+\t([a-z][a-z0-9._]*)")
# symbol header:  "0000000000010120 <merlin_gemm_f32_0>:"
_SYM_LINE = re.compile(r"^[0-9a-f]+\s+<([^>]+)>:")


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
        m_sym = _SYM_LINE.match(line)
        if m_sym:
            name = m_sym.group(1)
            cur = report.by_symbol.setdefault(name, SymbolCoverage(symbol=name))
            continue
        m = _INSN_LINE.match(line)
        if not m:
            continue
        mnem = m.group(1)
        cur.total += 1
        if _RVV_RE.match(mnem):
            cur.vector += 1
        elif _SCALAR_COMPUTE_RE.match(mnem):
            cur.scalar_compute += 1
    # drop the placeholder bucket if it stayed empty
    if report.by_symbol.get("<none>") and report.by_symbol["<none>"].total == 0:
        report.by_symbol.pop("<none>", None)
    return report


def _objdump() -> str | None:
    """Find an objdump that can disassemble rv64 (prefer the SpacemiT/riscv cross tool)."""
    from merlin.rvvgen import k1
    root = k1._toolchain_root()
    if root:
        for cand in ("riscv64-unknown-linux-gnu-objdump", "llvm-objdump", "objdump"):
            p = root / "bin" / cand
            if p.is_file():
                return str(p)
    for cand in ("riscv64-unknown-linux-gnu-objdump", "llvm-objdump", "objdump"):
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
