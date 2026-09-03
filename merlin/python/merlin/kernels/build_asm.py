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
    "-march=rv64gcv_zfh_zvfh", "-mabi=lp64d", "-mcmodel=medany",
    "-O2", "-ffreestanding", "-fno-builtin",
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
        proc = subprocess.run([str(gcc), "-E", "-Wp,-v", "-xc", os.devnull],
                              capture_output=True, text=True, timeout=30)
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
def build_kernel_asm(kernel_path: str | Path, *, include_dirs=(), extra_cflags=(),
                     timeout: int = 120, clang_first: bool = False) -> str | None:
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
    return [str(clang), "--target=riscv64-unknown-elf", "-march=rv64gcv",
            *march, *sys_incs, *inc_flags, "-c", str(src), "-o", str(obj)]


def _run(cmd: list[str], timeout: int) -> bool:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


# ------------------------------------------------------------------------ saturn corpus
def saturn_root() -> Path:
    """saturn-vectors checkout root, in precedence order.

    ``MERLIN_SATURN_REPO`` wins; then a local ``<repo>/tmp/kernels/saturn-vectors`` clone if one is
    actually there; then ``ext_path('chipyard')/generators/saturn``, because saturn ships INSIDE the
    chipyard checkout this repo already configures. Falling back to the chipyard copy is what makes
    the ceiling harness work without a second, hand-set env var: without it every expert column of
    the cross-framework matrix builds against a path that does not exist and reports NOT_RUN with
    ``fatal error: util.h``. Measured: all 30 cells of the matrix (openblas, xnnpack, ours x8, at
    32/64/128) failed that way, which reads as "the comparison cannot be made" rather than as a
    missing include dir.

    The unresolvable case still returns the tmp path, so a fresh checkout's error message names the
    location a clone is expected at rather than someone else's machine.
    """
    env = os.environ.get("MERLIN_SATURN_REPO")
    if env:
        return Path(env)
    local = repo_root() / "tmp/kernels/saturn-vectors"
    if local.is_dir():
        return local
    try:
        from ..common.paths import ext_path
        cand = ext_path("chipyard") / "generators" / "saturn"
        if cand.is_dir():
            return cand
    except (KeyError, ImportError):
        pass
    return local


def _saturn_kernel_tu(bench_dir: Path) -> Path | None:
    """Pick the kernel translation unit of a ``vec-*`` bench: the lone C file that is NOT a
    ``main``/``*_main`` driver (those carry the harness/data, not the kernel). When several
    remain, prefer the shortest name (the primary kernel, e.g. ``imatmul.c`` over a variant)."""
    cands = [p for p in sorted(bench_dir.glob("*.c"))
             if p.name != "main.c" and not p.name.endswith("_main.c")]
    if not cands:
        return None
    cands.sort(key=lambda p: (len(p.name), p.name))
    return cands[0]


def saturn_benchmark_asm(bench_name: str, *, timeout: int = 120) -> str | None:
    """Resolve a saturn benchmark's kernel TU + include flags and compile it to asm.

    Returns ``None`` when the bench is missing, has no standalone kernel TU (asm-only bench),
    or won't compile. ``VOPACC`` benches are excluded by name per the mining contract.
    """
    if "vopacc" in bench_name.lower():
        return None
    root = saturn_root()
    bench = root / "benchmarks" / bench_name
    tu = _saturn_kernel_tu(bench)
    if tu is None:
        return None
    common = root / "benchmarks/common"
    env = root / "benchmarks/env"
    incs = [bench, common, env]
    return build_kernel_asm(tu, include_dirs=[d for d in incs if d.is_dir()], timeout=timeout)


# ------------------------------------------------------------------ best-effort frameworks
def _framework_src_roots(source: str) -> list[Path]:
    """Header roots to try for a best-effort XNNPACK/OpenBLAS single-TU compile."""
    base = repo_root() / "tmp/kernels"
    if source == "xnnpack":
        env = os.environ.get("MERLIN_XNNPACK_REPO")
        root = Path(env) if env else base / "XNNPACK"
        return [root, root / "src", root / "include"]
    if source == "openblas":
        env = os.environ.get("MERLIN_OPENBLAS_REPO")
        root = Path(env) if env else base / "OpenBLAS"
        return [root, root / "kernel/riscv64"]
    return []


def framework_kernel_asm(source: str, kernel_path: str | Path, *,
                         timeout: int = 120) -> str | None:
    """Best-effort compile a framework (xnnpack/openblas) RVV kernel TU to asm. Returns
    ``None`` on the (expected) missing-header/type failures — full builds are a later task."""
    roots = [d for d in _framework_src_roots(source) if d.is_dir()]
    return build_kernel_asm(kernel_path, include_dirs=roots, timeout=timeout)


# ------------------------------------------------------------------------------- routing
def dossier_asm(nk, *, timeout: int = 120) -> str | None:
    """Route a :class:`~merlin.kernels.types.NormalizedKernel` to its build path by source.

    saturn -> :func:`saturn_benchmark_asm` (the bench is the second path component);
    xnnpack/openblas -> :func:`framework_kernel_asm` (best-effort). Any other source, or an
    unresolvable path, yields ``None``.
    """
    source = (getattr(nk, "source", "") or "").lower()
    path = getattr(nk, "path", "") or ""
    if source in ("saturn", "saturn-vectors", "saturn_vectors"):
        bench = _bench_from_path(path)
        return saturn_benchmark_asm(bench, timeout=timeout) if bench else None
    if source in ("xnnpack", "openblas"):
        return framework_kernel_asm(source, _resolve_framework_path(source, path),
                                    timeout=timeout)
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
    roots = _framework_src_roots(source)
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
