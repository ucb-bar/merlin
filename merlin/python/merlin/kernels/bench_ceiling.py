"""Kernel-performance CEILING harness (S4.2) — compile a CURATED expert kernel
*standalone* and MEASURE its cycle count on **spike**, to establish the bar our
compiler's RVV codegen is judged against.

    attainment = ceiling_cycles / compiler_cycles

(1.0 == we matched the expert; < 1.0 == the expert kernel is faster than what we
emit; > 1.0 == we beat the expert). The ceiling is *measured*, never modeled.

Why saturn first
----------------
The guaranteed-standalone corpus is ``tmp/kernels/saturn-vectors/benchmarks`` — each
``vec-*`` benchmark ships its own ``main`` + golden data + an htif printf path, builds
to a single ``.riscv`` ELF, and runs under ``spike`` printing a cycle count per kernel
invocation. We drive the saturn build recipe directly (the same flags the bench
Makefile uses, plus the one include dir for ``encoding.h`` that the empty ``env/`` dir
is missing), run on spike, and parse the bench's own cycle line. No bare-metal main is
generated; the bench already is one. ``VOPACC`` benches are excluded per the mining
contract.

Honesty contract
----------------
:func:`run_kernel_ceiling` returns ``None`` (never a fabricated number) when the
toolchain/spike is unavailable, the bench won't build, the run fails, or the requested
``(M,N,K)`` is not among the sizes the bench actually executes. Each ceiling row records
exactly which bench/kernel produced it and how the cycle count was parsed.

Reading the console is deliberately three-valued (:class:`CycleReading`): PARSED, ABSENT (this
bench does not sweep that size), and UNPARSEABLE (it DID measure the size but the console did not
say so in a shape we recognize). The last is a tooling defect and is reported, never quietly
downgraded to "not measured" and never turned into a cycle count of 0.

The ``fingerprint_key`` ``(op, dtype, shape_regime)`` matches
:class:`merlin.kernels.compare.RvvFingerprint`'s key, so a ceiling row joins 1:1 to a
compiler-measured cycle count for the same op-shape.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..common.paths import repo_root
from . import build_asm

# spike run conventions, lifted from the saturn benchmarks/Makefile RISCV_SIM line.
DEFAULT_ISA = "rv64gcv_zfh_zvfh"
SPIKE_MEM = "-m0x70020000:0x20000,0x80000000:0x10000000"
DEFAULT_CEILING_PATH = "out/artifacts/ceiling/ceiling.jsonl"

# The saturn Makefile cflags (RISCV_GCC_OPTS) verbatim, minus the per-bench -I flags
# (build_asm/the recipe add those). The empty benchmarks/env/ dir is missing encoding.h,
# which util.h/crt.S include; we add the chipyard libgloss include dir that ships it.
_SATURN_CFLAGS = (
    "-DPREALLOCATE=1", "-mcmodel=medany", "-static", "-O2", "-g", "-ffast-math",
    "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
    "-march=rv64gcv_zfh_zvfh", "-mabi=lp64d", "-std=gnu99",
)
_SATURN_LINK_OPTS = ("-static", "-nostdlib", "-nostartfiles", "-lm", "-lgcc")

# Candidate locations of the encoding.h the empty env/ dir lacks (read_csr/mcycle/minstret).
_ENCODING_H_CANDIDATES = (
    "toolchains/libgloss/include/encoding.h",
    "tools/torture/env/encoding.h",
)


# --------------------------------------------------------------------------- shape_regime
def shape_regime(op: str, M: int, N: int, K: int) -> str:
    """A compact, deterministic regime STRING for the fingerprint key.

    compare.py uses a single string (e.g. "square_small"), not the feature-extractor's
    regime list, so we produce one here that is stable for a given (op, M, N, K). The
    vocabulary mirrors features/shape_regime.py's intent (skinny / square / large) at a
    granularity that joins cleanly to a compiler-measured cycle count.
    """
    mn = min(M, N)
    if mn <= 1:                                   # dot / elementwise: N=K=1, M=length
        return "vector"
    if M == N == K:
        if M >= 256:
            return "square_large"
        if M >= 32:
            return "square_medium"
        return "square_small"
    if mn <= 16:
        return "skinny"
    return "rectangular"


# ------------------------------------------------------- console reading (three states, no regex)
#
# These benches have no machine-readable output mode: their printf'd console text IS the contract.
# It is fixed by the corpus's own C sources, so we anchor on those literals and then VALIDATE the
# token we pulled out, refusing anything that is not a plain non-negative decimal.
#
# Three outcomes, never collapsed:
#   ABSENT       the bench does not measure this size (its sweep never printed that header)
#   UNPARSEABLE  the size WAS measured but the cycle line could not be read -> the console format
#                drifted; we must say so
#   PARSED       a real number
# UNPARSEABLE never becomes ABSENT and neither ever becomes 0: an unmeasurable ceiling is UNKNOWN.

_PARSED = "parsed"
_ABSENT = "absent"
_UNPARSEABLE = "unparseable"

_DIGITS = frozenset("0123456789")


@dataclass(frozen=True)
class CycleReading:
    """What one bench console said about the requested size."""
    state: str                          # _PARSED / _ABSENT / _UNPARSEABLE
    cycles: int | None = None
    instructions: int | None = None
    note: str = ""                      # how it was read (goes on the ceiling row)
    detail: str = ""                    # what went wrong, when state is _UNPARSEABLE

    @classmethod
    def absent(cls) -> "CycleReading":
        return cls(_ABSENT)

    @classmethod
    def unparseable(cls, detail: str) -> "CycleReading":
        return cls(_UNPARSEABLE, detail=detail)


def _leading_count(text: str) -> tuple[int | None, int]:
    r"""Read ``\s*(\d+)`` off the FRONT of ``text`` -> ``(value, index just past the digits)``.

    Only ASCII digits: the spike console is ASCII, and refusing an exotic digit is the safe side of
    "refuse rather than coerce". A sign, a dot, or a letter stops the run, so ``-5`` and ``1.2`` are
    refused, exactly as the old ``(\d+)`` (which could not match a leading '-') did.
    """
    i = 0
    while i < len(text) and text[i].isspace():
        i += 1
    j = i
    while j < len(text) and text[j] in _DIGITS:
        j += 1
    if j == i:
        return None, 0
    return int(text[i:j]), j


def _square_shape_header(line: str, size: int) -> bool:
    r"""True for vec-igemm's sweep header at ``size``.

    Live console line (spike, saturn ``vec-igemm``)::

        Calculating a (32 x 32) x (32 x 32) matrix multiplication...

    printf source: ``"Calculating a (%d x %d) x (%d x %d) matrix multiplication...\n"`` with the
    same ``s`` in all four slots. The old pattern was
    ``\(\s*S\s*x\s*S\s*\)\s*x\s*\(\s*S\s*x\s*S\s*\)`` — the four-operand shape with arbitrary
    whitespace ANYWHERE inside it, including none. Deleting all whitespace from the line and
    looking for the canonical spelling accepts exactly that set. The token is specific enough to
    anchor on (it names the size four times), so it cannot pick a number out of another line: for
    S=4 it does not match ``(64 x 64) x (64 x 64)`` or ``(14 x 4) x (4 x 4)``.
    """
    return f"({size}x{size})x({size}x{size})" in "".join(line.split())


def _took_cycles(line: str) -> int | None:
    r"""``The execution took 26055 cycles.`` -> 26055; ``None`` when the line is not that.

    printf source: ``"The execution took %d cycles.\n"``. The old pattern was ``took\s+(\d+)\s+cycles``:
    whitespace-delimited, so the count is always a whole token. Anchoring on the exact token
    ``took`` is marginally stricter than the old substring match (which would also have fired on
    ``mistook``), which no spelling this bench emits can hit. A ``%d`` that printed negative has no
    all-digit token and is refused, as the old pattern also refused it.
    """
    parts = line.split()
    for i, token in enumerate(parts):
        if token != "took" or i + 2 >= len(parts):
            continue
        if any(c not in _DIGITS for c in parts[i + 1]) or not parts[i + 1]:
            continue
        if parts[i + 2].startswith("cycles"):
            return int(parts[i + 1])
    return None


def _length_header(line: str, avl: int) -> bool:
    r"""True for vec-dotprod's per-length header at ``avl``.

    Live console line::

        Calulating 64b dotp with vectors with length = 512

    printf source: ``"Calulating <w>b dotp with vectors with length = %lu\n"`` (the bench's own
    typo). The old pattern was ``length\s*=\s*AVL\b``; this walk is that literally — find
    ``length``, skip whitespace, require '=', skip whitespace, then the digits must be exactly
    ``avl`` and must not run on into another word character (so ``length = 512`` does not answer a
    request for 51).
    """
    want = str(avl)
    at = line.find("length")
    while at >= 0:
        i = at + len("length")
        while i < len(line) and line[i].isspace():
            i += 1
        if i < len(line) and line[i] == "=":
            i += 1
            while i < len(line) and line[i].isspace():
                i += 1
            if line.startswith(want, i):
                end = i + len(want)
                if end == len(line) or not (line[end].isalnum() or line[end] == "_"):
                    return True
        at = line.find("length", at + 1)
    return False


def _vector_cycles(line: str) -> tuple[int, int] | None:
    r"""``Vector cycles: 401 instructions: 401`` -> ``(401, 401)``; ``None`` otherwise.

    printf source: ``"Vector cycles: %ld instructions: %ld\n"``. The old pattern was
    ``Vector cycles:\s*(\d+)\s+instructions:\s*(\d+)`` — both counts anchored directly to their own
    named literal, which is what makes this safe to read positionally.
    """
    marker, gap = "Vector cycles:", "instructions:"
    at = line.find(marker)
    while at >= 0:
        rest = line[at + len(marker):]
        cycles, end = _leading_count(rest)
        if cycles is not None:
            after = rest[end:]
            if after[:1].isspace() and after.lstrip().startswith(gap):   # the old `\s+instructions:`
                instructions, _ = _leading_count(after.lstrip()[len(gap):])
                if instructions is not None:
                    return cycles, instructions
        at = line.find(marker, at + 1)
    return None


def _named_counter(line: str, name: str) -> int | None:
    r"""``mcycle = 137509`` -> 137509, for the saturn ``setStats`` counter dump.

    printf source: saturn ``benchmarks/common/syscalls.c`` prints ``"%s = %d\n"`` for each enabled
    counter, and ``setStats(0)`` stores the DELTA over the measured region (``csr -= counters[i]``),
    so ``mcycle`` here is the region's cycle count, not an absolute CSR read. Anchored on the
    counter name as the whole left-hand side, and the right-hand side must be a bare count.
    """
    lhs, sep, rhs = line.partition("=")
    if not sep or lhs.strip() != name:
        return None
    value, end = _leading_count(rhs)
    if value is None or rhs[end:].strip():      # nothing but the number may follow
        return None
    return value


def _colon_cycles(line: str) -> int | None:
    r"""``core   0: 137509 cycles, ... CPI`` -> 137509 (the riscv-tests / pk ``setStats`` spelling).

    This is what the retired ``:\s*(\d+)\s+cycles`` pattern read. The saturn corpus does NOT print
    it (see :func:`_parse_sgemm`), but a pk-hosted console would, so we keep accepting it.
    """
    at = line.find(":")
    while at >= 0:
        value, end = _leading_count(line[at + 1:])
        if value is not None:
            after = line[at + 1 + end:]
            if after[:1].isspace() and after.lstrip().startswith("cycles"):
                return value
        at = line.find(":", at + 1)
    return None


# ------------------------------------------------------------------------- bench registry
@dataclass(frozen=True)
class SaturnBench:
    """How to build + run a saturn benchmark and read a cycle number out of it.

    ``op``/``dtype`` are the curated kernel's canonical op + element type. ``parse`` maps
    a requested ``(M, N, K)`` to a :class:`CycleReading` — a measured count, an honest ABSENT when
    that size is not among the ones the bench executes, or UNPARSEABLE when the size WAS measured
    but the console did not say what we know how to read.
    """
    bench: str
    op: str
    dtype: str
    kernel_ref: str
    parse: Callable[[str, int, int, int], CycleReading]


def _parse_igemm(console: str, M: int, N: int, K: int) -> CycleReading:
    """vec-igemm sweeps squares s=4,8,16,32,64, printing a shape header then
    ``The execution took N cycles.`` a couple of lines later."""
    if not (M == N == K):
        return CycleReading.absent()
    lines = console.splitlines()
    first_window = ""
    for i, line in enumerate(lines):
        if not _square_shape_header(line, M):
            continue
        window = lines[i:i + 4]
        for nxt in window:
            cycles = _took_cycles(nxt)
            if cycles is not None:
                return CycleReading(_PARSED, cycles,
                                    note=f"'The execution took N cycles.' under the ({M} x {M}) header")
        # Keep scanning: a later header for the same size may carry the line (the old pattern
        # searched on too). Only once every one of them is exhausted is this UNPARSEABLE.
        first_window = first_window or " | ".join(x.strip() for x in window)
    if first_window:
        return CycleReading.unparseable(
            f"vec-igemm printed the ({M} x {M}) header but no 'took N cycles' line followed it: "
            + first_window)
    return CycleReading.absent()          # this size is simply not in the sweep


def _parse_dotprod(console: str, M: int, N: int, K: int) -> CycleReading:
    """vec-dotprod sweeps avl=8,64,512 for each element width, printing
    ``Calulating <w>b dotp with vectors with length = <avl>`` then
    ``Vector cycles: C instructions: I``.

    The requested length is M (N==K==1). Width is implied by dtype but the 64b block is measured
    first; we match on the length header regardless of width and take the FIRST (64-bit)
    occurrence so the key is deterministic."""
    if N != 1 or K != 1:
        return CycleReading.absent()
    lines = console.splitlines()
    first_window = ""
    for i, line in enumerate(lines):
        if not _length_header(line, M):
            continue
        window = lines[i:i + 3]
        for nxt in window:
            got = _vector_cycles(nxt)
            if got is not None:
                return CycleReading(_PARSED, got[0], got[1],
                                    note=f"'Vector cycles: C instructions: I' under length = {M}")
        # This header repeats once per element width, so keep scanning the later ones (as the old
        # pattern's search did) before calling the console unreadable.
        first_window = first_window or " | ".join(x.strip() for x in window)
    if first_window:
        return CycleReading.unparseable(
            f"vec-dotprod printed the length = {M} header but no 'Vector cycles:' line followed it: "
            + first_window)
    return CycleReading.absent()


def _parse_sgemm(console: str, M: int, N: int, K: int) -> CycleReading:
    r"""vec-sgemm runs ONE fixed 71x71x71 sgemm between ``setStats(1)``/``setStats(0)``, which dumps
    the enabled hardware counters. Only the fixed (71,71,71) shape is valid.

    The retired pattern here was ``:\s*(\d+)\s+cycles`` — the riscv-tests / pk spelling
    ``<code>: C cycles, ... CPI``. **The saturn corpus does not print that.** Verified by building
    and running this bench on spike from this tree; the whole console is::

        sgemm M,N,K = 71,71,71
        mcycle = 137509
        minstret = 137514

    (saturn ``benchmarks/common/syscalls.c`` prints ``"%s = %d\n"`` per counter, and ``setStats(0)``
    has already turned each into a delta over the measured region.) So the old pattern matched
    nothing and this bench's ceiling was silently unmeasurable — the exact failure the
    "parse structurally" rule exists to stop. We now read the counter dump the bench actually emits,
    and still accept the pk spelling for a pk-hosted console.
    """
    if not (M == N == K == 71):
        return CycleReading.absent()
    cycles = instructions = None
    for line in console.splitlines():
        if cycles is None:
            cycles = _named_counter(line, "mcycle")
        if instructions is None:
            instructions = _named_counter(line, "minstret")
    if cycles is not None:
        return CycleReading(_PARSED, cycles, instructions,
                            note="saturn setStats counter dump ('mcycle = N' / 'minstret = N')")
    for line in console.splitlines():
        legacy = _colon_cycles(line)
        if legacy is not None:
            return CycleReading(_PARSED, legacy, note="pk setStats line ('<code>: N cycles')")
    return CycleReading.unparseable(
        "vec-sgemm ran the fixed 71x71x71 shape but neither a 'mcycle = N' counter dump nor a "
        "'<code>: N cycles' stats line was found in its console: " + console.strip()[:300])


# Curated benches that build standalone AND run on spike with a parseable cycle line.
# dtype reflects the element type the kernel actually computes on.
SATURN_BENCHES: dict[str, SaturnBench] = {
    "vec-igemm": SaturnBench("vec-igemm", "matmul", "i64", "imatmul", _parse_igemm),
    "vec-dotprod": SaturnBench("vec-dotprod", "dotprod", "i64", "dotp_v64b", _parse_dotprod),
    "vec-sgemm": SaturnBench("vec-sgemm", "matmul", "f32", "vec_sgemm_nn", _parse_sgemm),
}


# ------------------------------------------------------------------------- build + run
def _encoding_include_dir() -> Path | None:
    """The dir holding the ``encoding.h`` the saturn env/ dir is missing, or None."""
    from ..runtime.backends import spike
    chip = spike.chipyard_root()
    for rel in _ENCODING_H_CANDIDATES:
        p = chip / rel
        if p.is_file():
            return p.parent
    return None


def saturn_available() -> bool:
    """True when the riscv gcc + spike + the saturn corpus + an encoding.h are all present."""
    from ..runtime.backends import spike
    root = build_asm.saturn_root() / "benchmarks"
    return (spike.available() and root.is_dir()
            and _encoding_include_dir() is not None)


def _build_saturn_elf(bench: str, workdir: Path, *, timeout: int = 300) -> Path | None:
    """Build a saturn benchmark to a spike ELF, mirroring the bench Makefile recipe.

    Returns the ELF path, or ``None`` on any failure (missing toolchain, bench, or a
    compile/link error) — never raises for an ordinary build failure.
    """
    from ..runtime.backends import spike
    gcc = spike.gcc_path()
    if not gcc.is_file():
        return None
    benchmarks = build_asm.saturn_root() / "benchmarks"
    bench_dir = benchmarks / bench
    common = benchmarks / "common"
    if not bench_dir.is_dir() or not common.is_dir():
        return None
    enc = _encoding_include_dir()
    if enc is None:
        return None

    test_ld = common / "test.ld"
    srcs: list[str] = []
    for d in (bench_dir, common, common / "ara"):
        if d.is_dir():
            srcs += [str(p) for p in sorted(d.glob("*.c"))]
            srcs += [str(p) for p in sorted(d.glob("*.S"))]
    incs = ["-I", str(benchmarks / "env"), "-I", str(common), "-I", str(bench_dir),
            "-I", str(enc)]
    elf = workdir / f"{bench}.riscv"
    cmd = [str(gcc), *incs, *_SATURN_CFLAGS, "-o", str(elf), *srcs,
           *_SATURN_LINK_OPTS, "-T", str(test_ld)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    # ld warns about RWX LOAD segments (harmless); only a nonzero exit / missing ELF fails.
    if proc.returncode != 0 or not elf.is_file():
        return None
    return elf


def _run_saturn_elf(elf: Path, *, isa: str = DEFAULT_ISA, harts: int = 4,
                    timeout: int = 300) -> str | None:
    """Run a saturn ELF on spike and return console text, or None on failure."""
    from ..runtime.backends import spike
    cmd = [str(spike.spike_path()), f"--isa={isa}", f"-p{harts}", SPIKE_MEM, str(elf)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


# ------------------------------------------------------------------------- public API
def run_kernel_ceiling(source: str, kernel_ref: str, op: str, dtype: str,
                       MNK: tuple[int, int, int], *, target: str = "spike",
                       isa: str = DEFAULT_ISA, timeout: int = 300,
                       diagnostics: list[str] | None = None) -> dict | None:
    """Build a curated kernel standalone, run it on ``target`` (spike), and return one
    ceiling row, or ``None`` if it cannot build/run or the requested size is not measured.

    ``kernel_ref`` doubles as the saturn bench name (``vec-igemm``) for the saturn source;
    when it is a registry key it also fixes op/dtype/parser, so callers may pass the
    bench name as ``kernel_ref`` and let the registry supply op/dtype.

    Returns a dict with: op, dtype, M, N, K, shape_regime, source, target, cycles,
    fingerprint_key (+ bench, kernel_ref, instructions, isa, note).

    ``None`` covers two different things and they must not be confused by the caller: the size was
    never measured (ordinary, silent), or the bench DID measure it but its console could not be
    read. The second is a tooling defect, so it is written to stderr and appended to
    ``diagnostics`` if a list is passed. Neither ever becomes a cycle count of 0.
    """
    if target != "spike":
        return None
    src = (source or "").lower()
    if src not in ("saturn", "saturn-vectors", "saturn_vectors"):
        return None
    if "vopacc" in (kernel_ref or "").lower():       # mining-contract exclusion
        return None

    bench_name = kernel_ref if kernel_ref in SATURN_BENCHES else _bench_for_op(op, dtype)
    spec = SATURN_BENCHES.get(bench_name)
    if spec is None:
        return None
    # Registry op/dtype win when the caller passed the bench name; else honor caller's.
    op = spec.op if kernel_ref in SATURN_BENCHES else (op or spec.op)
    dtype = spec.dtype if kernel_ref in SATURN_BENCHES else (dtype or spec.dtype)

    if not saturn_available():
        return None

    M, N, K = (int(x) for x in MNK)
    with tempfile.TemporaryDirectory(prefix="merlin_ceiling_") as tmp:
        elf = _build_saturn_elf(spec.bench, Path(tmp), timeout=timeout)
        if elf is None:
            return None
        console = _run_saturn_elf(elf, isa=isa, timeout=timeout)
    if console is None:
        return None
    reading = spec.parse(console, M, N, K)
    if reading.state == _UNPARSEABLE:
        message = (f"{spec.bench}: measured the requested size but its console could not be read "
                   f"-> ceiling UNKNOWN (not zero): {reading.detail}")
        if diagnostics is not None:
            diagnostics.append(message)
        print(f"kernel-bench: {message}", file=sys.stderr)
        return None
    if reading.state != _PARSED or reading.cycles is None:
        return None
    cycles, instr = reading.cycles, reading.instructions

    regime = shape_regime(op, M, N, K)
    row = {
        "op": op, "dtype": dtype, "M": M, "N": N, "K": K,
        "shape_regime": regime, "source": "saturn", "target": target,
        "bench": spec.bench, "kernel_ref": spec.kernel_ref,
        "cycles": int(cycles), "isa": isa,
        "fingerprint_key": fingerprint_key(op, dtype, regime),
        "note": f"saturn {spec.bench} on spike; read {reading.note}",
    }
    if instr is not None:
        row["instructions"] = int(instr)
    return row


def _bench_for_op(op: str, dtype: str) -> str | None:
    """Map (op, dtype) onto a registered saturn bench (best match), or None."""
    op = (op or "").lower()
    dtype = (dtype or "").lower()
    for name, spec in SATURN_BENCHES.items():
        if spec.op == op and (not dtype or spec.dtype == dtype):
            return name
    for name, spec in SATURN_BENCHES.items():
        if spec.op == op:
            return name
    return None


def fingerprint_key(op: str, dtype: str, regime: str) -> dict[str, str]:
    """The 1:1 join key to merlin.kernels.compare.RvvFingerprint (``{op,dtype,shape_regime}``)."""
    return {"op": op, "dtype": dtype, "shape_regime": regime}


# ------------------------------------------------------------------------- jsonl store
def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def append_ceiling(row: dict, path: str | Path = DEFAULT_CEILING_PATH) -> Path:
    """Append one ceiling row as a JSON line, creating the file/dir as needed."""
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def load_ceiling(path: str | Path = DEFAULT_CEILING_PATH) -> list[dict]:
    """Load all ceiling rows (skipping blank lines); empty list when the file is absent."""
    out = _resolve(path)
    if not out.is_file():
        return []
    rows: list[dict] = []
    for line in out.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def attainment(ceiling_row: dict, compiler_cycles: float) -> float | None:
    """attainment = ceiling_cycles / compiler_cycles.

    1.0 == our compiler matched the expert ceiling; < 1.0 == the expert is faster (we are
    leaving performance on the table); > 1.0 == we beat the expert. Returns ``None`` when
    inputs are missing or ``compiler_cycles`` is non-positive.
    """
    ceil = ceiling_row.get("cycles") if isinstance(ceiling_row, dict) else None
    if ceil is None or compiler_cycles is None or compiler_cycles <= 0:
        return None
    return float(ceil) / float(compiler_cycles)


def find_ceiling(rows: Iterable[dict], op: str, dtype: str, regime: str) -> dict | None:
    """First ceiling row whose fingerprint key matches (op, dtype, shape_regime)."""
    want = fingerprint_key(op, dtype, regime)
    for r in rows:
        if r.get("fingerprint_key") == want:
            return r
    return None


# ------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kernel-bench",
        description="Measure a curated kernel's spike cycle ceiling (S4.2).")
    ap.add_argument("--source", default="saturn",
                    help="curated corpus (only 'saturn' is wired today)")
    ap.add_argument("--bench", required=True,
                    help=f"saturn benchmark, one of: {', '.join(sorted(SATURN_BENCHES))}")
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("-K", type=int, required=True)
    ap.add_argument("--op", default="", help="override op (else from bench registry)")
    ap.add_argument("--dtype", default="", help="override dtype (else from bench registry)")
    ap.add_argument("--isa", default=DEFAULT_ISA)
    ap.add_argument("--out", default=DEFAULT_CEILING_PATH,
                    help="ceiling jsonl path (relative paths resolve under repo root)")
    ap.add_argument("--compiler-cycles", type=float, default=None,
                    help="if given, also print attainment = ceiling/compiler cycles")
    ap.add_argument("--no-append", action="store_true",
                    help="measure + print only; do not write the jsonl")
    args = ap.parse_args(argv)

    if not saturn_available():
        print("kernel-bench: spike/riscv-gcc/saturn-corpus/encoding.h unavailable; "
              "cannot measure a ceiling (set MERLIN_CHIPYARD / MERLIN_SATURN_REPO).")
        return 2

    diagnostics: list[str] = []
    row = run_kernel_ceiling(args.source, args.bench, args.op, args.dtype,
                             (args.M, args.N, args.K), isa=args.isa, diagnostics=diagnostics)
    if row is None:
        if diagnostics:
            # The bench ran and measured this size; only reading it back failed. Say which,
            # rather than blaming the build or the size.
            print(f"kernel-bench: UNKNOWN ceiling for bench={args.bench} "
                  f"(M,N,K)=({args.M},{args.N},{args.K}): " + "; ".join(diagnostics))
        else:
            print(f"kernel-bench: no ceiling for bench={args.bench} "
                  f"(M,N,K)=({args.M},{args.N},{args.K}) — build/run failed or size not measured.")
        return 1

    print(json.dumps(row, sort_keys=True))
    if not args.no_append:
        out = append_ceiling(row, args.out)
        print(f"appended -> {out}")
    if args.compiler_cycles is not None:
        att = attainment(row, args.compiler_cycles)
        print(f"attainment (ceiling/compiler) = {att:.4f}  "
              f"(ceiling={row['cycles']} cycles, compiler={args.compiler_cycles:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
