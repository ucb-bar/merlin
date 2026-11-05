"""The chip examples' shared setup and single-operator helpers (examples/lib), without native work."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def _load(name: str):
    path = repo_root() / "examples" / "lib" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples_lib_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_builtin_operator_has_its_loader():
    flow = _load("chip_flow")
    for op in flow.OPS:
        assert (flow.WORKLOADS / op / "loader.py").is_file(), op


def test_an_operator_shape_must_match_its_loader(monkeypatch, capsys):
    flow = _load("chip_flow")
    monkeypatch.setattr(flow, "capture", lambda *a, **k: pytest.fail("captured despite a bad shape"))
    with pytest.raises(SystemExit, match="takes 3 dims"):
        flow.main(["op", "gemm", "64", "64", "--board", "b"])
    with pytest.raises(SystemExit):
        flow.main(["op", "gemm", "--board", "b", "--matrix-unit", "u"])  # matrix flags come together
    assert "given together" in capsys.readouterr().err


def test_setup_names_each_missing_prerequisite_with_its_fix(tmp_path, monkeypatch):
    setup = _load("chip_setup")
    for key in ("MERLIN_CLANG", "MERLIN_CHIPYARD", "ZEPHYR_BASE", "ZEPHYR_SDK_INSTALL_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(setup, "REPO", tmp_path)  # no checkout-local LLVM install
    steps = setup.missing_steps({})
    assert len(steps) == 4
    assert "llvm_toolchain.md" in steps[0] and "OuterProductUnit" not in steps[0]
    assert "ea373800169ce7bbb9ca27ececf74101a7caf9d5" in steps[1]
    assert "west update" in steps[2] and "zephyr-sdk-0.17.0" in steps[3]


def test_setup_writes_only_the_keys_it_was_given(tmp_path):
    setup = _load("chip_setup")
    env = tmp_path / ".env"
    env.write_text("# mine\nKEEP=1\nMERLIN_CHIPYARD=/old\n")
    setup._upsert_env(env, {"MERLIN_CHIPYARD": "/new", "ZEPHYR_BASE": "/z"})
    assert env.read_text() == "# mine\nKEEP=1\nMERLIN_CHIPYARD=/new\nZEPHYR_BASE=/z\n"


def test_the_pinned_torch_mlir_is_documented_against_the_llvm_pin():
    setup = _load("chip_setup")
    guide = (repo_root() / "docs/guides/llvm_toolchain.md").read_text()
    assert "a47bddccec30" in guide  # the pin TORCH_MLIR is chosen against
    assert setup.TORCH_MLIR.startswith("torch-mlir==")
    assert Path(setup.__file__).read_text().count("a47bddcc") >= 1
