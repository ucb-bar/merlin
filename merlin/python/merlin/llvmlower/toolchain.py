"""Toolchain resolution for the whole-model path (all env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path
from merlin.common.paths import ext_path


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
DEFAULT_LLVM_INSTALL = Path(__file__).resolve().parents[4] / "third_party" / "llvm-install"


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
    """clang able to target both x86-64 and riscv64 (the IREE build's clang-23; else PATH)."""
    env = _env("MERLIN_CLANG")
    if env:
        return Path(env)
    b = _iree_bin()
    return (b / "clang-23") if b else Path("clang-23")


def mlir_translate() -> Path:
    """Standalone LLVM-23 ``mlir-translate`` (handles OpenMP -> LLVM-IR; the in-process
    torch-mlir bridge crashes on whole-model omp). Env-overridable."""
    env = _env("MERLIN_MLIR_TRANSLATE")
    return Path(env) if env else Path(DEFAULT_LLVM_INSTALL) / "bin" / "mlir-translate"


def available() -> bool:
    return m2m_python().is_file() and clang().is_file()
