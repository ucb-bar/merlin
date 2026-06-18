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

The ``fingerprint_key`` ``(op, dtype, shape_regime)`` matches
:class:`merlin.kernels.compare.RvvFingerprint`'s key, so a ceiling row joins 1:1 to a
compiler-measured cycle count for the same op-shape.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..common.paths import repo_root
from . import build_asm

# spike run conventions, lifted from the saturn benchmarks/Makefile RISCV_SIM line.
DEFAULT_ISA = "rv64gcv_zfh_zvfh"
SPIKE_MEM = "-m0x70020000:0x20000,0x80000000:0x10000000"
DEFAULT_CEILING_PATH = "output/kernels/ceiling/ceiling.jsonl"

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


# ------------------------------------------------------------------------- bench registry
@dataclass(frozen=True)
class SaturnBench:
    """How to build + run a saturn benchmark and read a cycle number out of it.

    ``op``/``dtype`` are the curated kernel's canonical op + element type. ``parse`` maps
    a requested ``(M, N, K)`` to its measured cycle/instr counts in the console text, or
    ``None`` when that size is not among the ones the bench executes.
    """
    bench: str
    op: str
    dtype: str
    kernel_ref: str
    parse: Callable[[str, int, int, int], tuple[int, int | None] | None]


def _parse_igemm(console: str, M: int, N: int, K: int) -> tuple[int, int | None] | None:
    """vec-igemm sweeps square s=4,8,16,32,64 printing a header then 'The execution took
    N cycles.'  Match the header for the requested square size, then the next cycle line."""
    if not (M == N == K):
        return None
    lines = console.splitlines()
    hdr = re.compile(rf"\(\s*{M}\s*x\s*{M}\s*\)\s*x\s*\(\s*{M}\s*x\s*{M}\s*\)")
    took = re.compile(r"took\s+(\d+)\s+cycles")
    for i, line in enumerate(lines):
        if hdr.search(line):
            for nxt in lines[i:i + 4]:
                m = took.search(nxt)
                if m:
                    return int(m.group(1)), None
    return None


def _parse_dotprod(console: str, M: int, N: int, K: int) -> tuple[int, int | None] | None:
    """vec-dotprod sweeps avl=8,64,512 for each width printing
    'Calulating <w>b dotp ... length = <avl>' then 'Vector cycles: C instructions: I'.
    The requested length is M (N==K==1). Width is implied by dtype but the 64b block is
    measured first; we match on the length header regardless of width and take the FIRST
    (64-bit) occurrence so the key is deterministic."""
    if N != 1 or K != 1:
        return None
    lines = console.splitlines()
    hdr = re.compile(rf"length\s*=\s*{M}\b")
    cyc = re.compile(r"Vector cycles:\s*(\d+)\s+instructions:\s*(\d+)")
    for i, line in enumerate(lines):
        if hdr.search(line):
            for nxt in lines[i:i + 3]:
                m = cyc.search(nxt)
                if m:
                    return int(m.group(1)), int(m.group(2))
    return None


def _parse_sgemm(console: str, M: int, N: int, K: int) -> tuple[int, int | None] | None:
    """vec-sgemm runs ONE fixed 71x71x71 sgemm under setStats, which prints a stats line
    of the form '<code>: C cycles, ... CPI'. Only the fixed (71,71,71) shape is valid."""
    if not (M == N == K == 71):
        return None
    m = re.search(r":\s*(\d+)\s+cycles", console)
    if m:
        return int(m.group(1)), None
    return None


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
                       isa: str = DEFAULT_ISA, timeout: int = 300) -> dict | None:
    """Build a curated kernel standalone, run it on ``target`` (spike), and return one
    ceiling row, or ``None`` if it cannot build/run or the requested size is not measured.

    ``kernel_ref`` doubles as the saturn bench name (``vec-igemm``) for the saturn source;
    when it is a registry key it also fixes op/dtype/parser, so callers may pass the
    bench name as ``kernel_ref`` and let the registry supply op/dtype.

    Returns a dict with: op, dtype, M, N, K, shape_regime, source, target, cycles,
    fingerprint_key (+ bench, kernel_ref, instructions, isa, note).
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
    parsed = spec.parse(console, M, N, K)
    if parsed is None:
        return None
    cycles, instr = parsed

    regime = shape_regime(op, M, N, K)
    row = {
        "op": op, "dtype": dtype, "M": M, "N": N, "K": K,
        "shape_regime": regime, "source": "saturn", "target": target,
        "bench": spec.bench, "kernel_ref": spec.kernel_ref,
        "cycles": int(cycles), "isa": isa,
        "fingerprint_key": fingerprint_key(op, dtype, regime),
        "note": f"saturn {spec.bench} on spike; parsed M=N=K={M} cycle line",
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

    row = run_kernel_ceiling(args.source, args.bench, args.op, args.dtype,
                             (args.M, args.N, args.K), isa=args.isa)
    if row is None:
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
