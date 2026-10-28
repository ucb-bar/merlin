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
    """The Lean checker and the interpreter of the authoritative MLIR exporter. A missing tool skips
    the test, or fails it when ``MERLIN_LEAN_REQUIRED`` is set."""
    selected = os.environ.get("MERLIN_M2M_PYTHON")
    tools = SimpleNamespace(
        lean_exe=lean_checker, m2m_python=Path(selected) if selected else toolchain.compiler_python()
    )
    missing = [f"{name} ({path})" for name, path in vars(tools).items() if not Path(path).exists()]
    if missing:
        reason = "Lean verification tools unavailable: " + ", ".join(missing)
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    return tools


@pytest.fixture
def pass_gate_tools():
    """The exact tools the Lean pass gate uses; skips without them, or fails under MERLIN_LEAN_REQUIRED."""
    from merlin.verify import pass_validation
    from merlin.verify.lean.check import lean_exe

    sdk_bin = os.environ.get(pass_validation.ENV_SDK_BIN)
    python = os.environ.get(pass_validation.ENV_COMPILER_PYTHON)
    missing = [
        name
        for name, ok in (
            (pass_validation.ENV_SDK_BIN, sdk_bin and (Path(sdk_bin) / "lake").is_file()),
            (pass_validation.ENV_COMPILER_PYTHON, python and Path(python).is_file()),
            ("built merlin-lean", lean_exe().is_file()),
        )
        if not ok
    ]
    if missing:
        reason = "Lean pass-validation tools unavailable: " + ", ".join(missing)
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    return pass_validation.CheckerTools(Path(sdk_bin), Path(python))
