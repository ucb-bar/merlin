"""MLIR toolchain resolution for the experiment ABI (env-overridable).

Phase 0 builds a standalone LLVM/MLIR 23 install (from ``third_party/llvm-project``) into
``third_party/llvm-install``. Out-of-tree C++ packages (``gemmini-opt``) link against it via
``find_package(MLIR REQUIRED CONFIG)`` with ``MLIR_DIR`` pointing at its cmake export; the runner
uses ``mlir-translate`` from it to take a package's lowered LLVM-dialect MLIR to ``.ll``.

All paths are overridable by ``MERLIN_MLIR_INSTALL``. The matching ``clang-23`` (same LLVM 23) is
reused from :mod:`merlin.llvmlower.toolchain`.
"""
from __future__ import annotations

import os
from pathlib import Path
from merlin.common.paths import repo_root

# repo root = .../merlin (this file: merlin/python/merlin/targetgen/contract/toolchain.py)
_REPO = repo_root()
DEFAULT_MLIR_INSTALL = _REPO / "third_party" / "llvm-install"

# LLVM/MLIR source pin (third_party/llvm-project); recorded into run manifests + the contract.
LLVM_VERSION = "23.0.0git"
LLVM_COMMIT = "a47bddccec30"


def mlir_install() -> Path:
    """Prefix of the standalone MLIR install (Phase 0 output)."""
    env = os.environ.get("MERLIN_MLIR_INSTALL")
    return Path(env) if env else DEFAULT_MLIR_INSTALL


def mlir_bin(tool: str) -> Path:
    """Path to a tool under the install's bin/ (mlir-opt, mlir-tblgen, mlir-translate)."""
    return mlir_install() / "bin" / tool


def mlir_cmake_dir() -> Path:
    """``MLIR_DIR`` for ``find_package(MLIR CONFIG)`` in out-of-tree package builds."""
    return mlir_install() / "lib" / "cmake" / "mlir"


def available() -> bool:
    """True iff the install exposes the tools + cmake export an OOT C++ package needs."""
    return (mlir_bin("mlir-opt").is_file()
            and mlir_bin("mlir-translate").is_file()
            and mlir_bin("mlir-tblgen").is_file()
            and (mlir_cmake_dir() / "MLIRConfig.cmake").is_file())


def require() -> Path:
    """Return the install prefix or raise a clear error naming the override env var."""
    if not available():
        raise RuntimeError(
            f"MLIR install not found/usable at {mlir_install()} "
            f"(expected bin/mlir-opt|mlir-translate|mlir-tblgen + lib/cmake/mlir/MLIRConfig.cmake). "
            f"Build it (Phase 0) or set MERLIN_MLIR_INSTALL.")
    return mlir_install()
