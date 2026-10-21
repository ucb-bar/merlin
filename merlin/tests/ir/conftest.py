"""Explicit prerequisites for tests that invoke the external host compiler or the Lean checker."""

import json
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
    # Use the compiler's configured interpreter, rather than a developer's model capture checkout.
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


@pytest.fixture(scope="module")
def lean_oracle():
    """Build the verified library and oracle imports without rebuilding the receipt checker."""
    from merlin.common.paths import repo_root
    from merlin.verify.lean import differential as diff
    from merlin.verify.lean.check import lean_exe

    lake = Path(shutil.which("lake") or Path.home() / ".elan/bin/lake")
    missing = [str(path) for path in (lake, lean_exe()) if not path.is_file()]
    if missing:
        reason = "Lean oracle prerequisites missing: " + ", ".join(missing)
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    diff._run([str(lake), "build", "MerlinVerify"], cwd=repo_root() / "merlin/lean")
    return lake


@pytest.fixture(scope="module")
def fidelity_tools(tmp_path_factory, lean_oracle):
    """Required Lean and native MLIR tools, with a retained independent API-contract probe."""
    from merlin.verify.lean import differential as diff
    from merlin.verify.lean.check import lean_exe

    compiler = Path(os.environ.get("MERLIN_M2M_PYTHON", toolchain.compiler_python()))
    lake = lean_oracle
    runner = Path(
        os.environ.get("MERLIN_MLIR_RUNNER_UTILS", toolchain.llvm_install() / "lib/libmlir_c_runner_utils.so")
    )
    tools = diff.Tools(compiler, lean_exe(), lake, runner)
    missing = [str(p) for p in (compiler, tools.lean_exe, lake, runner) if not p.is_file()]
    if missing:
        reason = "Lean differential prerequisites missing: " + ", ".join(missing)
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    diff._run([str(compiler), "-c", "from torch_mlir.execution_engine import ExecutionEngine"])
    # An installed but incompatible compiler fails here, rather than resembling a semantic mismatch.
    work = tmp_path_factory.mktemp("mlir-map-contract")
    request = work / "request.json"
    request.write_text(json.dumps({"map_contract": True}))
    result = diff._run([str(compiler), str(Path(diff.__file__).with_name("mlir_execute.py")), str(request)])
    (work / "contract.json").write_text(result.stdout)
    return tools
