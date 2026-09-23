"""Tool locations resolve at runtime, with an explicit universal-path alternative."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from merlin.common import paths as common_paths
from merlin.targetgen.sandbox import toolchain as TC


def test_import_does_not_resolve_checkout_or_configuration(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("import-time path discovery")

    for name in ("repo_root", "compat_lib_dir", "env", "ext_path"):
        monkeypatch.setattr(common_paths, name, forbidden)
    name = "isolated_toolchain_import_test"
    spec = importlib.util.spec_from_file_location(name, TC.__file__)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module.SIM_TOOLCHAINS.data["chipyard"] is module._chipyard


def test_checkout_defaults_preserve_selected_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(TC, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(TC, "compat_lib_dir", lambda: tmp_path / "compat")
    observed = []

    def env(name, default):
        observed.append((name, default))
        return str(tmp_path / "selected-clang")

    monkeypatch.setattr(TC, "env", env)
    value = TC.ToolchainPaths.from_checkout()
    assert value.venv == str(tmp_path / ".venv")
    assert value.llvm == str(tmp_path / "third_party/llvm-install")
    assert value.compat_lib == str(tmp_path / "compat")
    assert value.merlin_clang == str(tmp_path / "selected-clang/bin/clang-23")
    assert observed == [("MERLIN_CLANG_INSTALL", str(tmp_path / "build/host-merlin-release/install"))]


def test_explicit_paths_drive_binds_environment_and_probes(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("explicit paths must not choose checkout defaults")

    monkeypatch.setattr(TC.ToolchainPaths, "from_checkout", forbidden)
    answer_surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    monkeypatch.setattr(answer_surfaces, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    paths = TC.ToolchainPaths(
        tmp_path,
        str(tmp_path / "venv"),
        str(tmp_path / "llvm"),
        str(tmp_path / "compat"),
        str(tmp_path / "clang"),
        str(tmp_path / "uv"),
    )
    for path in (paths.venv, paths.llvm, paths.compat_lib, paths.clang_bin, paths.clang_resource, paths.uv_python):
        Path(path).mkdir(parents=True, exist_ok=True)
    target = SimpleNamespace(sim_via="", curated_harness=None, target="synthetic")
    argv = TC.toolchain_binds(target, paths=paths)
    for path in (paths.venv, paths.llvm, paths.compat_lib, paths.clang_bin, paths.clang_resource, paths.uv_python):
        assert any(argv[i : i + 3] == ["--ro-bind", path, path] for i in range(len(argv)))
    exports = TC.sandbox_env(target, tmp_path, paths=paths)
    assert f"export MERLIN_CLANG={paths.merlin_clang};" in exports
    assert f"export PYTHONPATH={tmp_path}/merlin/python" in exports
    assert "PYTHONDONTWRITEBYTECODE=1" in exports
    assert [probe.bind for probe in TC.required_tool_probes(target, paths=paths)] == [
        paths.venv,
        paths.llvm,
        paths.clang_bin,
    ]
