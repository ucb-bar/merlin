"""Universal asm-normalizer for kernel mining (S8.3) — compile ONE curated kernel
translation unit to objdump RVV assembly, source-agnostic at the instruction level.

This feeds (a) the dossier's ``asm`` field (:func:`dossier_asm`) and (b) later the
kernel-performance ceiling (S4.2). The whole point is *honest graceful degradation*:
a kernel that cannot be built standalone returns ``None`` (with a recorded reason),
never a fabricated disassembly.

Design
------
We compile with ``-c`` (object only, no link) so we never need a ``main``/libc, then
disassemble the object with the chipyard ``objdump`` reused via
:func:`merlin.llvmlower.custom_isa.disassemble`. The emitted text is plain ``objdump -d``
so :meth:`merlin.kernels.compare.RvvFingerprint.from_objdump` consumes it unchanged.

Two corpora, two difficulty tiers:

* **saturn-vectors/benchmarks** — the GUARANTEED path. Each ``vec-*`` benchmark ships a
  standalone C kernel TU (e.g. ``vec-dotprod/dotproduct.c``) that includes only a local
  header. We compile that TU with ``-I<bench> -Icommon -Ienv`` and the RVV cflags.
* **XNNPACK / OpenBLAS** — BEST-EFFORT only. These need framework headers/params structs;
  a single-TU compile usually fails on a missing header/type. We try (rooted at the
  framework src root) and on failure return ``None`` with the reason. Full CMake builds
  are a deliberately-later task.

Which corpora exist, which ``kernel.source`` spellings name them, their layout and where their checkouts
live is data (``merlin/contract/corpora.yaml``, read via :mod:`merlin.targetgen.corpora`); this module
holds only the per-layout build recipes.

Toolchain
---------
clang-23 (``llvmlower.toolchain.clang``) ships no riscv ``<stdint.h>``/``riscv_vector.h``
that these intrinsic kernels need, so the proven RVV compiler in this repo is the chipyard
``riscv64-unknown-elf-gcc`` (``runtime.backends.spike.gcc_path``); the matching ``objdump``
is what :func:`disassemble` already uses. We default to that gcc and, when ``clang_first``
is set, attempt clang (pointed at the gcc sysroot) first and fall back to gcc.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from ..common.paths import repo_root
from ..llvmlower.custom_isa import disassemble

# RVV cflags (runtime/backends/zephyr_model.py pattern, plus the saturn Makefile's zfh/zvfh).
RVV_CFLAGS: tuple[str, ...] = (
    "-march=rv64gcv_zfh_zvfh",
    "-mabi=lp64d",
    "-mcmodel=medany",
    "-O2",
    "-ffreestanding",
    "-fno-builtin",
)


# --------------------------------------------------------------------------- toolchains
def _gcc() -> Path | None:
    """The chipyard riscv gcc (proven RVV compiler), or None when unavailable."""
    from ..runtime.backends import spike

    gcc = spike.gcc_path()
    return gcc if gcc.is_file() else None


def _objdump_available() -> bool:
    from ..runtime.backends import spike

    return spike.gcc_path().with_name("riscv64-unknown-elf-objdump").is_file()


def _gcc_sysroot_includes() -> tuple[str, ...]:
    """The newlib/gcc include dirs that clang needs to find ``stdint.h`` etc."""
    gcc = _gcc()
    if gcc is None:
        return ()
    try:
        proc = subprocess.run([str(gcc), "-E", "-Wp,-v", "-xc", os.devnull], capture_output=True, text=True, timeout=30)
    except Exception:
        return ()
    incs: list[str] = []
    capture = False
    for line in (proc.stderr or "").splitlines():
        if "search starts here" in line:
            capture = True
            continue
        if "End of search list" in line:
            break
        if capture and line.startswith(" "):
            p = Path(line.strip())
            if p.is_dir():
                incs.append(str(p.resolve()))
    return tuple(incs)


def asm_toolchain_available() -> bool:
    """True when a riscv compiler + objdump are present (else every build returns None)."""
    return _gcc() is not None and _objdump_available()


# ------------------------------------------------------------------------- core compile
def build_kernel_asm(
    kernel_path: str | Path, *, include_dirs=(), extra_cflags=(), timeout: int = 120, clang_first: bool = False
) -> str | None:
    """Compile ``kernel_path`` (``-c``, object only) to an rv64gcv object and return its
    ``objdump -d`` text; return ``None`` on any failure (missing toolchain, source not found,
    compile error, empty disassembly). Never raises for an ordinary build failure.

    Args:
        kernel_path: the single C translation unit to compile.
        include_dirs: extra ``-I`` roots (the bench dir, framework src root, ...).
        extra_cflags: appended after :data:`RVV_CFLAGS` (override/extend).
        timeout: per-compile wall-clock budget (seconds).
        clang_first: try clang-23 (pointed at the gcc sysroot) before gcc.
    """
    src = Path(kernel_path)
    if not src.is_file():
        return None
    gcc = _gcc()
    if gcc is None or not _objdump_available():
        return None

    inc_flags: list[str] = []
    for d in include_dirs:
        inc_flags += ["-I", str(d)]
    cflags = [*RVV_CFLAGS, *extra_cflags]

    with tempfile.TemporaryDirectory(prefix="merlin_asm_") as tmp:
        obj = Path(tmp) / (src.stem + ".o")

        commands: list[list[str]] = []
        if clang_first:
            clang = _clang_cmd(src, obj, inc_flags, cflags)
            if clang is not None:
                commands.append(clang)
        commands.append([str(gcc), *cflags, *inc_flags, "-c", str(src), "-o", str(obj)])

        for cmd in commands:
            if not _run(cmd, timeout) or not obj.is_file():
                continue
            text = disassemble(obj)
            if text and text.strip():
                return text
        return None


def _clang_cmd(src: Path, obj: Path, inc_flags: list[str], cflags: list[str]) -> list[str] | None:
    """A clang-23 compile command pointed at the gcc sysroot, or None when clang is absent.

    Note: clang-23 in this install often lacks the ``__riscv_*`` intrinsic surface these
    kernels use, so this is opportunistic — :func:`build_kernel_asm` falls back to gcc.
    """
    from ..llvmlower import toolchain

    clang = toolchain.clang()
    if not clang.is_file():
        return None
    sys_incs: list[str] = []
    for inc in _gcc_sysroot_includes():
        sys_incs += ["-isystem", inc]
    march = [c for c in cflags if not c.startswith("-march")]
    return [
        str(clang),
        "--target=riscv64-unknown-elf",
        "-march=rv64gcv",
        *march,
        *sys_incs,
        *inc_flags,
        "-c",
        str(src),
        "-o",
        str(obj),
    ]


def _run(cmd: list[str], timeout: int) -> bool:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


# ------------------------------------------------------------------- registered corpora
# Corpus layouts (the `layout` of a merlin/contract/corpora.yaml entry): a standalone-benchmark corpus
# builds and runs each benchmark as-is; a single-TU corpus is a framework whose kernel TU is compiled
# best-effort against its declared include roots.
STANDALONE_BENCHMARKS = "standalone_benchmarks"
SINGLE_TU = "single_tu"


def _corpus(source: str | None) -> tuple[str | None, dict | None]:
    """``(corpus name, registry spec)`` for the corpus ``source`` names, or ``(None, None)``."""
    from ..targetgen.corpora import kernel_corpora, kernel_corpus_for_source

    name = kernel_corpus_for_source(source)
    return (name, kernel_corpora()[name]) if name else (None, None)


def corpus_root(source: str) -> Path | None:
    """Checkout root of the expert corpus ``source`` names, or None when no corpus is registered for it.

    Resolved by :func:`merlin.targetgen.corpora.kernel_corpus_root`: ``MERLIN_<NAME>_REPO`` wins; then a
    local ``<repo>/tmp/kernels/<checkout>`` clone if one is actually there; then the corpus's declared
    fallback inside an external checkout. That last step is what makes the ceiling harness work without a
    second, hand-set env var when a corpus ships INSIDE a checkout this repo already configures: without it
    every expert column of the cross-framework matrix builds against a path that does not exist and reports
    NOT_RUN with ``fatal error: util.h``. Measured: all 30 cells of the matrix (openblas, xnnpack, ours x8,
    at 32/64/128) failed that way, which reads as "the comparison cannot be made" rather than as a missing
    include dir.

    The unresolvable case still returns the local-clone path, so a fresh checkout's error message names the
    location a clone is expected at rather than someone else's machine.
    """
    from ..targetgen.corpora import kernel_corpus_root

    name, _spec = _corpus(source)
    return kernel_corpus_root(name) if name else None


def benchmark_source(source: str | None = None) -> str | None:
    """The standalone-benchmark corpus ``source`` names, or None when it names none.

    With ``source`` omitted: the first corpus the registry declares with the ``standalone_benchmarks``
    layout -- the corpus whose benchmarks build and run as-is, which the guaranteed asm path and the kernel
    ceiling need. Pass ``""`` (not None) for "no source" so it is not taken as "the default"."""
    if source is None:
        from ..targetgen.corpora import kernel_corpora

        return next((n for n, s in kernel_corpora().items() if s.get("layout") == STANDALONE_BENCHMARKS), None)
    name, spec = _corpus(source)
    return name if spec and spec.get("layout") == STANDALONE_BENCHMARKS else None


def benchmarks_dir(source: str | None = None) -> Path | None:
    """The benchmark tree of the standalone-benchmark corpus ``source`` names (default: the registry's
    first), or None when there is no such corpus."""
    name = benchmark_source(source)
    if name is None:
        return None
    _name, spec = _corpus(name)
    return corpus_root(name) / str(spec.get("benchmarks") or "")


def _benchmark_kernel_tu(bench_dir: Path) -> Path | None:
    """Pick the kernel translation unit of a ``vec-*`` bench: the lone C file that is NOT a
    ``main``/``*_main`` driver (those carry the harness/data, not the kernel). When several
    remain, prefer the shortest name (the primary kernel, e.g. ``imatmul.c`` over a variant)."""
    cands = [p for p in sorted(bench_dir.glob("*.c")) if p.name != "main.c" and not p.name.endswith("_main.c")]
    if not cands:
        return None
    cands.sort(key=lambda p: (len(p.name), p.name))
    return cands[0]


def benchmark_asm(bench_name: str, source: str | None = None, *, timeout: int = 120) -> str | None:
    """Resolve a standalone benchmark's kernel TU + include flags and compile it to asm.

    ``source`` picks the corpus (default: the registry's first standalone-benchmark corpus). Returns
    ``None`` when there is no such corpus, the bench is missing, has no standalone kernel TU (asm-only
    bench), or won't compile. ``VOPACC`` benches are excluded by name per the mining contract.
    """
    if "vopacc" in bench_name.lower():
        return None
    benchmarks = benchmarks_dir(source)
    if benchmarks is None:
        return None
    bench = benchmarks / bench_name
    tu = _benchmark_kernel_tu(bench)
    if tu is None:
        return None
    incs = [bench, benchmarks / "common", benchmarks / "env"]
    return build_kernel_asm(tu, include_dirs=[d for d in incs if d.is_dir()], timeout=timeout)


# ------------------------------------------------------------------ best-effort frameworks
def framework_include_roots(source: str) -> list[Path]:
    """Header roots to try for a best-effort single-TU compile of a ``single_tu`` corpus's kernel -- its
    declared ``include_subdirs`` under the corpus root, in -I order. [] for a source naming no such corpus."""
    name, spec = _corpus(source)
    if not spec or spec.get("layout") != SINGLE_TU:
        return []
    root = corpus_root(name)
    return [root / str(sub) for sub in spec.get("include_subdirs") or []]


def framework_kernel_asm(source: str, kernel_path: str | Path, *, timeout: int = 120) -> str | None:
    """Best-effort compile a framework (xnnpack/openblas) RVV kernel TU to asm. Returns
    ``None`` on the (expected) missing-header/type failures — full builds are a later task."""
    roots = [d for d in framework_include_roots(source) if d.is_dir()]
    return build_kernel_asm(kernel_path, include_dirs=roots, timeout=timeout)


# ------------------------------------------------------------------------------- routing
def dossier_asm(nk, *, timeout: int = 120) -> str | None:
    """Route a :class:`~merlin.kernels.types.NormalizedKernel` to its build path by the LAYOUT of the
    corpus its source names (registry ``merlin/contract/corpora.yaml``).

    standalone_benchmarks -> :func:`benchmark_asm` (the bench is the ``vec-*`` path component);
    single_tu -> :func:`framework_kernel_asm` (best-effort). A source naming no registered corpus, or an
    unresolvable path, yields ``None``.
    """
    source = (getattr(nk, "source", "") or "").lower()
    path = getattr(nk, "path", "") or ""
    name, spec = _corpus(source)
    layout = (spec or {}).get("layout")
    if layout == STANDALONE_BENCHMARKS:
        bench = _bench_from_path(path)
        return benchmark_asm(bench, name, timeout=timeout) if bench else None
    if layout == SINGLE_TU:
        return framework_kernel_asm(name, _resolve_framework_path(name, path), timeout=timeout)
    return None


def _bench_from_path(path: str) -> str | None:
    """Extract the ``vec-*`` bench name from a saturn kernel path."""
    for part in Path(path).parts:
        if part.startswith("vec-"):
            return part
    return None


def _resolve_framework_path(source: str, path: str) -> Path:
    """Make a framework kernel path absolute against its src root when it is relative."""
    p = Path(path)
    if p.is_absolute():
        return p
    roots = framework_include_roots(source)
    if roots:
        cand = roots[0] / path
        if cand.is_file():
            return cand
    return p


def build_dossier_with_asm(nk, *, timeout: int = 120):
    """Convenience: build the kernel's asm then assemble the dossier with it set.

    A thin wrapper over :func:`dossier_asm` + :func:`merlin.kernels.dossier.build_dossier`;
    when no standalone build exists the dossier still carries the code-level layers, with
    ``to_dict()['has_asm'] == False``.
    """
    from .dossier import build_dossier

    asm = dossier_asm(nk, timeout=timeout)
    return build_dossier(nk, asm=asm)


# -------------------------------------------------------------------------------- helper
def top_mnemonics(objdump: str, n: int = 15) -> list[tuple[str, int]]:
    """The ``n`` most frequent RVV mnemonics in objdump text (handy for verification/CLI)."""
    from ..common.driver_output import is_vector_mnemonic

    h: Counter[str] = Counter()
    for line in objdump.splitlines():
        cols = line.split("\t")
        if len(cols) < 3 or not cols[2].strip():
            continue
        m = cols[2].strip().split()[0]
        if is_vector_mnemonic(m):
            h[m] += 1
    return h.most_common(n)
