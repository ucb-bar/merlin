"""Toolchain resolution for the whole-model path (all env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path
from merlin.common.paths import ext_path

DEFAULT_M2M_DIR = "/scratch/agustin/projects/model2MLIR"
DEFAULT_IREE_BIN = f"{ext_path("merlin_iree")}/build/host-merlin-release/install/bin"
# Standalone LLVM-23 install (mlir-opt/mlir-translate) used where the torch-mlir wheel's
# in-process translate bridge is unreliable (its OpenMPIRBuilder segfaults on whole-model
# omp IR, whereas this build's mlir-translate handles it cleanly).
DEFAULT_LLVM_INSTALL = Path(__file__).resolve().parents[4] / "third_party" / "llvm-install"


def m2m_dir() -> Path:
    return Path(os.environ.get("MERLIN_M2M_DIR", DEFAULT_M2M_DIR))


def m2m_python() -> Path:
    """Python of the model2MLIR venv (has the torch-mlir wheel: LLVM 23 passes)."""
    env = os.environ.get("MERLIN_M2M_VENV")
    base = Path(env) if env else m2m_dir() / ".venv"
    return base / "bin" / "python"


def clang() -> Path:
    """clang able to target both x86-64 and riscv64 (the IREE install's clang-23)."""
    env = os.environ.get("MERLIN_CLANG")
    return Path(env) if env else Path(DEFAULT_IREE_BIN) / "clang-23"


def mlir_translate() -> Path:
    """Standalone LLVM-23 ``mlir-translate`` (handles OpenMP -> LLVM-IR; the in-process
    torch-mlir bridge crashes on whole-model omp). Env-overridable."""
    env = os.environ.get("MERLIN_MLIR_TRANSLATE")
    return Path(env) if env else Path(DEFAULT_LLVM_INSTALL) / "bin" / "mlir-translate"


def available() -> bool:
    return m2m_python().is_file() and clang().is_file()
