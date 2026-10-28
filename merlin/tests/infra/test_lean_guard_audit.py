"""Audit every standalone driver without combining conflicting root entry points."""

import importlib.util
import json
import subprocess

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def guard(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("lean_guard", repo_root() / "build_tools/scripts/check_lean_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lean = tmp_path / "lean"
    for name in ["Check/Module", "Rules/Cse", "Driver/Qinner", "Driver/Overrank"]:
        path = lean / "MerlinVerify" / f"{name}.lean"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("-- source roster fixture\n")
    monkeypatch.setattr(module, "LEAN", lean)
    return module


def test_conflicting_driver_entrypoints_are_all_audited(guard, monkeypatch):
    imported = []

    def auditor(argv, **kwargs):
        arguments = argv[3:]
        selected = arguments[1:2] if arguments[:1] == ["--defined-in"] else []
        modules = arguments[2:] if selected else arguments
        imported.extend(modules)
        drivers = [name for name in modules if name.startswith("MerlinVerify.Driver.")]
        if len(drivers) > 1:
            return subprocess.CompletedProcess(argv, 1, "", "environment already contains 'main'")
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"theorems_audited": 1, "defined_in": selected, "violations": []}), ""
        )

    monkeypatch.setattr(guard.subprocess, "run", auditor)
    assert guard.run_audit()
    assert sorted(imported) == [
        "MerlinVerify.Check.Module",
        "MerlinVerify.Driver.Overrank",
        "MerlinVerify.Driver.Qinner",
        "MerlinVerify.Rules.Cse",
    ]


@pytest.mark.parametrize("stdout", ["", "not JSON", "[]", "{}"])
def test_missing_audit_reports_fail_without_masking_diagnostics(guard, monkeypatch, capsys, stdout):
    def auditor(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout, "original import diagnostic")

    monkeypatch.setattr(guard.subprocess, "run", auditor)
    assert not guard.run_audit()
    if stdout != "{}":
        assert "original import diagnostic" in capsys.readouterr().out


def test_a_violation_in_one_standalone_driver_fails_the_audit(guard, monkeypatch):
    def auditor(argv, **kwargs):
        violations = ["unapproved axiom"] if "MerlinVerify.Driver.Qinner" in argv else []
        selected = argv[4:5] if argv[3:4] == ["--defined-in"] else []
        return subprocess.CompletedProcess(
            argv,
            bool(violations),
            json.dumps({"theorems_audited": 1, "defined_in": selected, "violations": violations}),
            "",
        )

    monkeypatch.setattr(guard.subprocess, "run", auditor)
    assert not guard.run_audit()
