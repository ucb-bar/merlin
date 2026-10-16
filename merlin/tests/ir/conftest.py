"""Explicit prerequisites for tests that invoke the external host compiler or the Lean checker."""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.llvmlower import toolchain


@pytest.fixture
def upstream_host_tools():
    """Skip only an unavailable tool; compilation and runtime failures still fail."""
    tools = {
        "compiler interpreter": toolchain.compiler_python(),
        "Clang": toolchain.clang(),
        "MLIR translator": toolchain.mlir_translate(),
    }
    missing = [name for name, path in tools.items() if shutil.which(str(path)) is None]
    if missing:
        pytest.skip("external host compiler tools unavailable: " + ", ".join(missing))


@pytest.fixture
def lean_checker():
    """The built Lean checker alone, for tests independent of the torch_mlir interpreter."""
    from merlin.verify.lean.check import lean_exe

    exe = lean_exe()
    if not exe.is_file():
        reason = f"Lean checker unavailable: {exe}"
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    return exe


@pytest.fixture
def lean_tools(lean_checker):
    """The Lean checker and the interpreter of the authoritative MLIR exporter (design D9). A missing
    tool skips the test, or fails it when ``MERLIN_LEAN_REQUIRED`` is set: a run that has to qualify
    the Lean claims must not pass by skipping them."""
    default_m2m = Path.home() / "Projects" / "model2MLIR" / ".venv" / "bin" / "python"
    tools = SimpleNamespace(lean_exe=lean_checker, m2m_python=Path(os.environ.get("MERLIN_M2M_PYTHON", default_m2m)))
    missing = [f"{name} ({path})" for name, path in vars(tools).items() if not Path(path).exists()]
    if missing:
        reason = "Lean verification tools unavailable: " + ", ".join(missing)
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    return tools
