"""Exact archived test origins never admit an external production module."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def probe(monkeypatch, tmp_path):
    root = (
        Path(os.environ["MERLIN_TEST_SOURCE_INPUTS_ROOT"])
        if "MERLIN_TEST_SOURCE_INPUTS_ROOT" in os.environ
        else repo_root()
    )
    path = root / "build_tools/scripts/installed_qualification_probe.py"
    spec = importlib.util.spec_from_file_location("archived_test_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.setattr(module.sysconfig, "get_path", lambda _: str(site))
    monkeypatch.setattr(module, "sys", SimpleNamespace(modules={}))
    monkeypatch.delenv("MERLIN_TEST_ARCHIVED_SOURCES", raising=False)
    return module


def roster(monkeypatch, tmp_path, *, rows=None):
    path = tmp_path / "test_example.py"
    path.write_text("def test_example():\n    pass\n")
    row = {
        "module": "merlin.tests.targetgen.test_example",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    receipt = tmp_path / "archived-tests.json"
    receipt.write_text(
        json.dumps({"schema": "merlin.installed_test_sources.v1", "files": [row] if rows is None else rows})
    )
    monkeypatch.setenv("MERLIN_TEST_ARCHIVED_SOURCES", str(receipt))
    return path, row, receipt


def start(probe):
    # The old guard has no start hook; finishing still reaches its real failure.
    if hasattr(probe, "pytest_sessionstart"):
        probe.pytest_sessionstart(None)


def test_actual_finish_hook_accepts_only_the_exact_archived_test(monkeypatch, tmp_path, probe):
    path, row, _ = roster(monkeypatch, tmp_path)
    probe.sys.modules[row["module"]] = SimpleNamespace(__file__=str(path))
    start(probe)
    probe.pytest_sessionfinish(None, 0)
    # The ordinary wheel probe does not inherit a pytest exception.
    with pytest.raises(AssertionError):
        probe.assert_installed_origins()


@pytest.mark.parametrize(
    "name", ["merlin.codegen", "merlin.tests.targetgen.test_other", "merlin_experiments.phase0.driver"]
)
def test_pinned_test_file_cannot_disguise_an_external_production_or_other_test(monkeypatch, tmp_path, probe, name):
    path, _, _ = roster(monkeypatch, tmp_path)
    probe.sys.modules[name] = SimpleNamespace(__file__=str(path))
    with pytest.raises(AssertionError):
        start(probe)
        probe.pytest_sessionfinish(None, 0)


@pytest.mark.parametrize("defect", ["test_bytes", "receipt_bytes", "receipt_environment"])
def test_actual_finish_hook_reopens_the_original_roster_and_test_bytes(monkeypatch, tmp_path, probe, defect):
    path, row, receipt = roster(monkeypatch, tmp_path)
    probe.sys.modules[row["module"]] = SimpleNamespace(__file__=str(path))
    start(probe)
    if defect == "test_bytes":
        path.write_text("changed\n")
    elif defect == "receipt_bytes":
        receipt.write_text("{}")
    else:
        monkeypatch.delenv("MERLIN_TEST_ARCHIVED_SOURCES")
    with pytest.raises((AssertionError, ValueError)):
        probe.pytest_sessionfinish(None, 0)


def test_absent_roster_keeps_the_original_installed_only_guard(probe, tmp_path):
    probe.sys.modules["merlin.tests.targetgen.test_example"] = SimpleNamespace(
        __file__=str(tmp_path / "test_example.py")
    )
    with pytest.raises(AssertionError):
        start(probe)
        probe.pytest_sessionfinish(None, 0)


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_field",
        "duplicate_row",
        "production_name",
        "changed_module",
        "relative_path",
        "changed_digest",
        "missing_file",
        "nonstring",
        "duplicate_key",
    ],
)
def test_incomplete_or_substituted_archived_rosters_refuse_before_collection(monkeypatch, tmp_path, probe, defect):
    path, row, receipt = roster(monkeypatch, tmp_path)
    value = json.loads(receipt.read_text())
    if defect == "unknown_field":
        value["trusted"] = True
    elif defect == "duplicate_row":
        value["files"].append(row)
    elif defect == "production_name":
        value["files"][0]["module"] = "merlin.codegen.test_example"
    elif defect == "changed_module":
        value["files"][0]["module"] = "merlin.tests.targetgen.test_other"
    elif defect == "relative_path":
        value["files"][0]["path"] = path.name
    elif defect == "changed_digest":
        value["files"][0]["sha256"] = "0" * 64
    elif defect == "missing_file":
        path.unlink()
    elif defect == "nonstring":
        value["files"][0]["module"] = False
    if defect == "duplicate_key":
        receipt.write_text('{"schema":"merlin.installed_test_sources.v1","files":[],"files":[]}')
    else:
        receipt.write_text(json.dumps(value))
    with pytest.raises((AssertionError, ValueError)):
        probe.pytest_sessionstart(None)


@pytest.mark.parametrize("defect", ["none", "external_core", "missing_core", "external_optional"])
def test_actual_pytest_archive_import_preserves_the_installed_parent(monkeypatch, tmp_path, probe, defect):
    from _pytest.pathlib import ImportMode, import_path

    site = Path(probe.sysconfig.get_path("purelib"))
    installed = site / "merlin"
    installed.mkdir()
    (installed / "__init__.py").write_text("")
    (installed / "probe_marker.py").write_text("value = 'installed'\n")
    archive = tmp_path / "archive"
    path = archive / "merlin/tests/infra/test_import.py"
    path.parent.mkdir(parents=True)
    path.write_text("from merlin import probe_marker\n")
    row = {
        "module": "merlin.tests.infra.test_import",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if defect == "external_optional":
        path = archive / "merlin_experiments/tests/infra/test_import.py"
        path.parent.mkdir(parents=True)
        path.write_text("pass\n")
        row = {
            "module": "merlin_experiments.tests.infra.test_import",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt = tmp_path / "archive.json"
    receipt.write_text(json.dumps({"schema": "merlin.installed_test_sources.v1", "files": [row]}))
    monkeypatch.setenv("MERLIN_TEST_ARCHIVED_SOURCES", str(receipt))
    monkeypatch.setattr(probe, "sys", sys)
    saved = {
        name: module for name, module in sys.modules.items() if name.split(".")[0] in {"merlin", "merlin_experiments"}
    }
    for name in saved:
        del sys.modules[name]
    monkeypatch.setattr(sys, "path", [str(site)])
    if defect in {"external_core", "external_optional"}:
        external = tmp_path / "external"
        package = external / ("merlin" if defect == "external_core" else "merlin_experiments")
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        sys.path.insert(0, str(external))
    elif defect == "missing_core":
        (installed / "__init__.py").unlink()
        (installed / "probe_marker.py").unlink()
        installed.rmdir()
    try:
        if defect != "none":
            with pytest.raises((AssertionError, ModuleNotFoundError)):
                probe.pytest_sessionstart(None)
            return
        probe.pytest_sessionstart(None)
        imported = import_path(path, mode=ImportMode.importlib, root=archive, consider_namespace_packages=False)
        assert imported.probe_marker.value == "installed"
        assert Path(imported.probe_marker.__file__).is_relative_to(site)
        probe.pytest_sessionfinish(None, 0)
    finally:
        for name in tuple(sys.modules):
            if name.split(".")[0] in {"merlin", "merlin_experiments"}:
                del sys.modules[name]
        sys.modules.update(saved)
