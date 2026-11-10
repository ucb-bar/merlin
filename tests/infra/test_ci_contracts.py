"""Fail-closed global collection and qualification report contracts, no native qualification claim."""

import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def load_tool(name):
    selected = os.environ.get("MERLIN_TEST_BUDGET_CHECKER") if name == "check_test_budget" else None
    group = "packaging" if name == "qualify_installed" else "checks"
    path = Path(selected) if selected else repo_root() / "build_tools" / group / (name + ".py")
    spec = importlib.util.spec_from_file_location("compact_contract_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qualifier(monkeypatch):
    tool = load_tool("qualify_installed")
    monkeypatch.setitem(
        tool.SUITES,
        "compact_contract",
        {
            "tests": ("test_contract.py",),
            "native_test_files": ("test_contract.py",),
            "native_test_cases": (("test_contract.py", "native_contract"),),
        },
    )
    return tool


def test_global_budget_counts_expanded_cases_not_functions():
    budget = load_tool("check_test_budget")
    report = dict(
        exit_code=0,
        collection_errors=[],
        collection_skips=[],
        deselected=[],
        nodeids=["test_contract.py::case[" + str(i) + "]" for i in range(150)],
    )
    assert budget.validate(report) == []
    report["nodeids"].append("test_contract.py::case[150]")
    assert any("151 cases" in error for error in budget.validate(report))


def test_global_budget_discovers_nested_active_package_collector(tmp_path):
    budget = load_tool("check_test_budget")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "packages/optional/tests/test_contract.py"
    source.parent.mkdir(parents=True)
    source.write_text("def test_contract(): pass\n")
    assert budget.collectors(tmp_path) == ["packages/optional/tests/test_contract.py"]


def test_qualification_junit_admits_exact_declared_member(qualifier, tmp_path):
    document = ET.Element("testsuite")
    ET.SubElement(document, "testcase", file="test_contract.py", classname="test_contract", name="native_contract")
    path = tmp_path / "tests.xml"
    ET.ElementTree(document).write(path)
    report = {}
    qualifier.check_native_test_report("compact_contract", path, report)
    assert report["native_test_counts"] == {"tests": 1, "skipped": 0}
    assert report["missing_native_test_files"] == report["missing_native_test_cases"] == []


def test_qualification_junit_skip_never_counts_as_execution(qualifier, tmp_path):
    document = ET.Element("testsuite")
    member = ET.SubElement(
        document, "testcase", file="test_contract.py", classname="test_contract", name="native_contract"
    )
    ET.SubElement(member, "skipped", message="native prerequisite unavailable")
    path = tmp_path / "tests.xml"
    ET.ElementTree(document).write(path)
    with pytest.raises(qualifier.QualificationFailed, match="zero skips"):
        qualifier.check_native_test_report("compact_contract", path, {})


def test_qualification_empty_junit_refuses_missing_execution(qualifier, tmp_path):
    path = tmp_path / "tests.xml"
    ET.ElementTree(ET.Element("testsuite")).write(path)
    with pytest.raises(qualifier.QualificationFailed, match="every declared native test file"):
        qualifier.check_native_test_report("compact_contract", path, {})


def test_failed_child_preserves_terminal_record_and_diagnostics(tmp_path):
    tool = load_tool("qualify_installed")
    recorder = tool.Recorder(tmp_path, {"commands": []}, 5)
    with pytest.raises(tool.QualificationFailed):
        recorder.run("failed", [sys.executable, "-c", "print('failure evidence'); raise SystemExit(7)"], tmp_path)
    saved = json.loads((tmp_path / "report.json").read_text())["commands"][0]
    assert saved["returncode"] == 7 and saved["status"] == "failed"
    assert saved["timeout_s"] == 5 and saved["elapsed_s"] >= 0
    assert "failure evidence" in (tmp_path / "failed.log").read_text()
