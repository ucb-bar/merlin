"""model2MLIR checkout and shared capture-interpreter selection.

This adapter does not import frameworks or perform capture. Per-workload capture.toml
selection remains in capture.bundle; LLVM Python selection remains in llvmlower.toolchain.
"""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common.paths import env


def root(*, default: str | Path = "/path/to/model2MLIR") -> Path:
    """Keep the historical capture-bundle alias precedence and unconfigured placeholder.

    Either process alias precedes either .env alias. Within each source,
    MERLIN_MODEL2MLIR precedes MERLIN_M2M_DIR. Resolution does not validate existence
    or fall through an explicit invalid path to a different checkout.
    """
    value = (
        os.environ.get("MERLIN_MODEL2MLIR")
        or os.environ.get("MERLIN_M2M_DIR")
        or env("MERLIN_MODEL2MLIR")
        or env("MERLIN_M2M_DIR")
        or default
    )
    return Path(value)


def capture_python_path(*, checkout: Path | None = None) -> Path:
    """Select capture Python without requiring an installed framework environment.

    MERLIN_M2M_PYTHON precedes MERLIN_M2M_VENV, then the checkout's .venv. Each setting
    honors process environment before .env. Selection never falls through an explicit
    invalid path. Keep its spelling: resolving a venv symlink loses its environment.
    This is not a replacement for a workload's explicitly pinned capture.toml environment.
    Lazy availability probes use this function; execution edges use capture_python.
    """
    executable = env("MERLIN_M2M_PYTHON")
    if executable:
        selected = Path(executable).expanduser()
    else:
        venv = env("MERLIN_M2M_VENV")
        base = Path(venv).expanduser() if venv else (checkout if checkout is not None else root()) / ".venv"
        selected = base / "bin" / "python"
    return selected


def capture_python(*, checkout: Path | None = None) -> Path:
    """Select and validate shared capture Python, never the compiler interpreter.

    Unlike capture_python_path, missing/nonexecutable choices raise immediately.
    Validation never falls back to a different checkout or environment.
    """
    selected = capture_python_path(checkout=checkout)
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise RuntimeError(
            f"model2MLIR capture Python is missing or not executable: {selected}; "
            "configure MERLIN_M2M_PYTHON or MERLIN_M2M_VENV (compiler Python is separate)"
        )
    return selected
