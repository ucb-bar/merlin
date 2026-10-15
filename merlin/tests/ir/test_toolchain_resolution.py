"""``toolchain.clang_for`` answers for another checkout what ``toolchain.clang`` answers for this one.

A launcher checks the compiler a child engine will use before starting it; the child resolves from
its own environment, its checkout's ``.env`` and its checkout's install, not the launcher's.
"""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.llvmlower import toolchain


def test_clang_for_another_checkout_reads_that_checkout_not_this_process(tmp_path, monkeypatch):
    for name in ("MERLIN_CLANG", "MERLIN_IREE_BIN", "MERLIN_EXT_MERLIN_IREE", "MERLIN_MLIR_INSTALL"):
        monkeypatch.delenv(name, raising=False)
    # This process's answer and the parameterised one agree about this checkout.
    assert toolchain.clang_for(repo_root(), os.environ) == toolchain.clang()
    # Another checkout: its own install, then its own .env, then an explicit environment.
    assert toolchain.clang_for(tmp_path, {}) == Path("clang-23")
    install = tmp_path / "third_party" / "llvm-install" / "bin" / "clang-23"
    install.parent.mkdir(parents=True)
    install.write_text("")
    assert toolchain.clang_for(tmp_path, {}) == install
    (tmp_path / ".env").write_text("MERLIN_CLANG=/from/dotenv/clang\n")
    assert toolchain.clang_for(tmp_path, {}) == Path("/from/dotenv/clang")
    assert toolchain.clang_for(tmp_path, {"MERLIN_CLANG": "/from/env/clang"}) == Path("/from/env/clang")


def test_selected_mlir_install_is_used_consistently(tmp_path, monkeypatch):
    install = tmp_path / "llvm"
    monkeypatch.setenv("MERLIN_MLIR_INSTALL", str(install))
    for name in (
        "MERLIN_CLANG",
        "MERLIN_MLIR_TRANSLATE",
        "MERLIN_OBJDUMP",
        "MERLIN_OBJCOPY",
        "MERLIN_NM",
        "MERLIN_READELF",
    ):
        monkeypatch.delenv(name, raising=False)
    bin_dir = install / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("clang-23", "mlir-translate", "llvm-objdump", "llvm-objcopy", "llvm-nm", "llvm-readelf"):
        (bin_dir / name).touch()
    assert toolchain.llvm_install() == install
    assert toolchain.clang() == bin_dir / "clang-23"
    assert toolchain.clang_for(tmp_path, {"MERLIN_MLIR_INSTALL": str(install)}) == bin_dir / "clang-23"
    assert toolchain.mlir_translate() == bin_dir / "mlir-translate"
    assert toolchain.objdump() == bin_dir / "llvm-objdump"
    assert toolchain.objcopy() == bin_dir / "llvm-objcopy"
    assert toolchain.nm() == bin_dir / "llvm-nm"
    assert toolchain.readelf() == bin_dir / "llvm-readelf"
    monkeypatch.setenv("MERLIN_MLIR_TRANSLATE", "/explicit/mlir-translate")
    assert toolchain.mlir_translate() == Path("/explicit/mlir-translate")


def test_missing_selected_clang_does_not_use_an_unrelated_install(tmp_path):
    install = tmp_path / "selected"
    legacy = tmp_path / "legacy" / "bin"
    legacy.mkdir(parents=True)
    (legacy / "clang-23").touch()
    selection = {"MERLIN_MLIR_INSTALL": str(install)}

    assert toolchain._resolve_clang(install, selection.get, lambda: legacy) == install / "bin" / "clang-23"
    assert toolchain.clang_for(tmp_path, {**selection, "MERLIN_IREE_BIN": str(legacy)}) == install / "bin" / "clang-23"


def test_contract_and_host_use_the_same_dotenv_install(tmp_path, monkeypatch):
    from merlin.common import paths
    from merlin.targetgen.contract import toolchain as contract_toolchain

    install = tmp_path / "selected"
    monkeypatch.delenv("MERLIN_MLIR_INSTALL", raising=False)
    monkeypatch.setattr(paths, "_dotenv", lambda: {"MERLIN_MLIR_INSTALL": str(install)})

    assert toolchain.llvm_install() == install
    assert contract_toolchain.mlir_install() == install
    assert contract_toolchain.mlir_bin("mlir-translate") == install / "bin" / "mlir-translate"


def test_stock_driver_readiness_requires_its_tools_without_python_bindings(tmp_path, monkeypatch):
    monkeypatch.setattr(toolchain, "_env", {"MERLIN_COMPILER_DRIVER": "stock-cli"}.get)
    monkeypatch.setattr(toolchain, "llvm_install", lambda: tmp_path)
    monkeypatch.setattr(toolchain, "clang", lambda: tmp_path / "bin" / "clang-23")
    monkeypatch.setattr(toolchain, "compiler_python", lambda: tmp_path / "absent-python")
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("mlir-opt", "mlir-translate", "clang-23"):
        (binary / name).touch()
    assert toolchain.available()
    (binary / "mlir-translate").unlink()
    assert not toolchain.available()
