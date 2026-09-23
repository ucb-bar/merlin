"""Compatibility identities and dependency direction survive the source migration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common import paths
from merlin.llvmlower import toolchain


def test_capture_bundle_import_does_not_import_research():
    script = """
import sys
from merlin.capture.bundle import CaptureBundle
from merlin.baselines.bundle import CaptureBundle as LegacyBundle
from merlin.capture import rewrite
from merlin.baselines import bundle_rewrite
assert CaptureBundle is LegacyBundle
assert rewrite is bundle_rewrite
assert not any(name.startswith(("merlin.dse", "merlin.design_pressure")) for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_modelir_import_path_is_scoped_even_on_failure(tmp_path, monkeypatch):
    from merlin.integrations import modelir

    monkeypatch.setattr(modelir.importlib.util, "find_spec", lambda name: None)
    before = list(sys.path)
    try:
        with modelir.importable(tmp_path):
            assert sys.path[0] == str(tmp_path)
            raise RuntimeError("consumer failed")
    except RuntimeError:
        pass
    assert sys.path == before


def test_installed_work_root_never_defaults_to_site_packages(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_REPO_ROOT", raising=False)
    monkeypatch.delenv("MERLIN_WORK_DIR", raising=False)
    monkeypatch.delenv("MERLIN_OUT_ROOT", raising=False)
    monkeypatch.setattr(paths, "checkout_root", lambda: None)
    monkeypatch.chdir(tmp_path)
    assert paths.repo_root() == tmp_path
    assert paths.out_dir() == tmp_path / "out"


def test_explicit_compiler_python_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(
        toolchain,
        "_env",
        lambda name, default=None: {
            "MERLIN_COMPILER_PYTHON": "/missing/qualified/python",
            "MERLIN_COMPILER_VENV": "/other/compiler",
            "MERLIN_M2M_VENV": "/capture",
        }.get(name, default),
    )
    assert toolchain.compiler_python() == Path("/missing/qualified/python")
    assert toolchain.m2m_python() == toolchain.compiler_python()


def test_wheel_installed_inside_checkout_does_not_discover_source_extensions(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "build_tools").mkdir()
    (tmp_path / "build_tools/package_resources.json").write_text("{}")
    source = tmp_path / "src/merlin/common/paths.py"
    source.parent.mkdir(parents=True)
    source.write_text("")
    installed = tmp_path / "out/build/venv/lib/site-packages/merlin/common/paths.py"
    monkeypatch.setattr(paths, "__file__", str(installed))
    assert paths.checkout_root() is None
    assert paths.python_import_roots() == (installed.parents[2],)
    monkeypatch.setattr(paths, "__file__", str(source))
    assert paths.checkout_root() == tmp_path


def test_compiler_venv_and_legacy_fallback(monkeypatch):
    values = {"MERLIN_COMPILER_VENV": "/compiler", "MERLIN_M2M_VENV": "/capture"}
    monkeypatch.setattr(toolchain, "_env", lambda name, default=None: values.get(name, default))
    assert toolchain.compiler_python() == Path("/compiler/bin/python")
    values.pop("MERLIN_COMPILER_VENV")
    assert toolchain.compiler_python() == Path("/capture/bin/python")


def test_lazy_root_cli_does_not_import_engines():
    script = """
import sys
from merlin.cli import main
assert main(["--help"]) == 0
assert not any(name.startswith(("aet", "torch", "merlin.targetgen")) for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=os.environ)
    assert result.returncode == 0, result.stderr


def test_missing_experiment_distribution_has_actionable_install_hint(monkeypatch, capsys):
    from merlin import cli

    def missing(name):
        raise ModuleNotFoundError(name="merlin_experiments")

    monkeypatch.setattr(cli.importlib, "import_module", missing)
    with pytest.raises(SystemExit) as error:
        cli.main(["experiment", "list"])
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "uv pip install -e packages/merlin-experiments" in diagnostic
    assert "experiments extra" not in diagnostic


def test_missing_engine_dependency_is_not_misreported_as_missing_extension(monkeypatch):
    from merlin import cli

    def missing(name):
        raise ModuleNotFoundError(name="engine_dependency")

    monkeypatch.setattr(cli.importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError) as error:
        cli.main(["experiment", "list"])
    assert error.value.name == "engine_dependency"
