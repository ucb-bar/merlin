"""Toolchain resolution for the whole-model path (all env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_M2M_DIR = "/scratch/agustin/projects/model2MLIR"
DEFAULT_IREE_BIN = "/scratch2/agustin/merlin/build/host-merlin-release/install/bin"


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


def available() -> bool:
    return m2m_python().is_file() and clang().is_file()
