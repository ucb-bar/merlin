"""Toolchain resolution for the whole-model path (all env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path
from merlin.common.paths import ext_path, repo_root


def _env(key: str, default: str | None = None) -> str | None:
    """Process env wins, then the gitignored ``<repo>/.env`` (same source ``ext_path`` reads), then
    ``default``. This lets a dev point at their model2MLIR / clang once in ``.env`` and have the
    toolchain resolve automatically — parity with how the ``aet`` sibling checkout is picked up —
    without exporting vars per shell or committing a personal path."""
    from merlin.common.paths import _dotenv
    return os.environ.get(key) or _dotenv().get(key) or default


DEFAULT_M2M_DIR = "/path/to/model2MLIR"  # external model2MLIR checkout; set MERLIN_M2M_DIR (or .env)
# Standalone LLVM-23 install (mlir-opt/mlir-translate) used where the torch-mlir wheel's
# in-process translate bridge is unreliable (its OpenMPIRBuilder segfaults on whole-model
# omp IR, whereas this build's mlir-translate handles it cleanly).
DEFAULT_LLVM_INSTALL = repo_root() / "third_party" / "llvm-install"


def m2m_dir() -> Path:
    return Path(_env("MERLIN_M2M_DIR", DEFAULT_M2M_DIR))


def m2m_python() -> Path:
    """Python of the model2MLIR venv (has the torch-mlir wheel: LLVM 23 passes)."""
    env = _env("MERLIN_M2M_VENV")
    base = Path(env) if env else m2m_dir() / ".venv"
    return base / "bin" / "python"


def _iree_bin() -> Path | None:
    """bin/ of the IREE-based Merlin build (ships clang-23), if configured. Set MERLIN_IREE_BIN, or
    MERLIN_EXT_MERLIN_IREE pointing at the third_party/baselines/merlin-iree submodule build.
    Resolved lazily so importing this module never requires the IREE build to be present."""
    env = _env("MERLIN_IREE_BIN")
    if env:
        return Path(env)
    try:
        return Path(ext_path("merlin_iree")) / "build" / "host-merlin-release" / "install" / "bin"
    except Exception:
        return None


def clang() -> Path:
    """clang able to target both x86-64 and riscv64. The repo's OWN toolchain always wins — resolution,
    first that exists: ``MERLIN_CLANG`` (explicit override) → the repo's OWN
    ``third_party/llvm-install`` ``clang-23`` (built with clang + the RISCV target; self-contained and
    authoritative) → the IREE build's ``clang-23`` (legacy fallback only, when that external build
    happens to be present) → ``clang-23`` on PATH. Preferring the repo's own install keeps the toolchain
    self-contained and independent of the retired IREE build."""
    env = _env("MERLIN_CLANG")
    if env:
        return Path(env)
    local = DEFAULT_LLVM_INSTALL / "bin" / "clang-23"
    if local.exists():
        return local
    b = _iree_bin()
    if b and (b / "clang-23").exists():
        return b / "clang-23"
    return (b / "clang-23") if b else Path("clang-23")


def mlir_translate() -> Path:
    """Standalone LLVM-23 ``mlir-translate`` (handles OpenMP -> LLVM-IR; the in-process
    torch-mlir bridge crashes on whole-model omp). Env-overridable."""
    env = _env("MERLIN_MLIR_TRANSLATE")
    return Path(env) if env else Path(DEFAULT_LLVM_INSTALL) / "bin" / "mlir-translate"


def available() -> bool:
    return m2m_python().is_file() and clang().is_file()


def objdump() -> Path:
    """LLVM ``objdump``, from the same install as :func:`clang`. Env-overridable.

    Used by the post-codegen census to count what was actually EMITTED for a symbol. It has to be
    the LLVM one, and the same one that produced the object: GNU objdump on a host build has no
    reason to know the cross target the object was compiled for, and a disassembler that decodes
    nothing would make an empty function indistinguishable from an unreadable one."""
    env = _env("MERLIN_OBJDUMP")
    if env:
        return Path(env)
    local = DEFAULT_LLVM_INSTALL / "bin" / "llvm-objdump"
    if local.exists():
        return local
    return Path(clang()).parent / "llvm-objdump"
