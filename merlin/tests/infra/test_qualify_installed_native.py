"""Actual pytest report and executable-pin controls, without compiler authority.

The tiny archived test modules and shell files are private diagnostic fixtures.
They are not stock compilers, a Phase-1 seed or a native qualification receipt.
"""

import importlib.util
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from merlin.common.paths import repo_root

_SPEC = importlib.util.spec_from_file_location(
    "qualify_installed_native_controls", repo_root() / "build_tools/scripts/qualify_installed.py"
)
Q = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(Q)


def actual_report(tmp_path, *, native="pass"):
    tests = tmp_path / "archived-tests"
    tests.mkdir()
    if native != "missing":
        content = (
            "import pytest\ndef test_native_one():\n    pytest.skip('selected translator absent')\n"
            if native == "skip"
            else "def test_native_one(): pass\ndef test_native_two(): pass\n"
        )
        for filename in Q.SUITES["component-convergence"]["native_test_files"]:
            (tests / filename).write_text(content)
    (tests / "test_component_generation.py").write_text(
        "import pytest\ndef test_other_scope():\n    pytest.skip('independent source selection absent')\n"
    )
    evidence = tmp_path / "actual-child"
    evidence.mkdir()
    xml = evidence / "tests.xml"
    recorder = Q.Recorder(evidence, {"commands": []}, 30)
    recorder.run(
        "pytest-report-control",
        [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "-q",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
            "--rootdir",
            tests,
            "-o",
            "junit_family=xunit1",
            "--junitxml",
            xml,
            "--basetemp",
            evidence / "test-tmp",
            tests,
        ],
        evidence,
    )
    return xml


def test_actual_pytest_native_subset_executes_while_other_skip_is_separately_retained(tmp_path):
    xml, report = actual_report(tmp_path), {}
    Q.check_native_test_report("component-convergence", xml, report)
    assert report["suite_test_counts"] == {"tests": 7, "skipped": 1}
    assert report["native_test_counts"] == {"tests": 6, "skipped": 0}
    assert report["other_test_counts"] == {"tests": 1, "skipped": 1}
    assert report["missing_native_test_files"] == []
    assert report["test_skips"]["native"] == []
    (other,) = report["test_skips"]["other"]
    assert other["file"] == "test_component_generation.py"
    assert "independent source selection absent" in other["message"]
    assert report["native_zero_skip_scope"] == "declared_native_test_files"
    assert report["native_test_report"] == {"path": str(xml), "sha256": Q.digest(xml)}


@pytest.mark.parametrize("native,reason", [("skip", "zero skips"), ("missing", "every declared native test file")])
def test_actual_pytest_skipped_or_omitted_native_subset_refuses_and_retains_denominator(tmp_path, native, reason):
    xml, report = actual_report(tmp_path, native=native), {}
    with pytest.raises(Q.QualificationFailed, match=reason):
        Q.check_native_test_report("component-convergence", xml, report)
    assert report["native_test_report"]["sha256"] == Q.digest(xml)
    assert report["other_test_counts"] == {"tests": 1, "skipped": 1}
    if native == "missing":
        assert report["missing_native_test_files"] == [
            "test_component_compile_role_transport.py",
            "test_component_native_deadline.py",
            "test_component_pointer_entry.py",
        ]
        assert report["native_test_counts"] == {"tests": 0, "skipped": 0}
    else:
        assert report["native_test_counts"] == {"tests": 3, "skipped": 3}
        assert "selected translator absent" in report["test_skips"]["native"][0]["message"]


@pytest.mark.parametrize("defect", ["foreign-file", "missing-file", "wrong-module", "duplicate", "failure", "error"])
def test_mutating_actual_child_report_cannot_substitute_a_native_identity(tmp_path, defect):
    xml, report = actual_report(tmp_path), {}
    tree = ET.parse(xml)
    suite = tree.getroot().find("testsuite")
    case = next(case for case in suite if case.get("file") == "test_component_compile_role_transport.py")
    if defect == "foreign-file":
        case.set("file", "unarchived/test_foreign.py")
    elif defect == "missing-file":
        case.attrib.pop("file")
    elif defect == "wrong-module":
        case.set("classname", "test_component_generation")
    elif defect == "duplicate":
        suite.append(ET.fromstring(ET.tostring(case)))
    else:
        ET.SubElement(case, defect, {"message": "explicit private report mutation"})
    tree.write(xml)
    with pytest.raises(Q.QualificationFailed):
        Q.check_native_test_report("component-convergence", xml, report)
    assert report["native_test_report"]["sha256"] == Q.digest(xml)


def test_compile_only_retains_zero_skip_scope_for_every_original_declared_test_file(tmp_path):
    configured = Q.SUITES["compile-only"]
    assert set(configured["native_test_files"]) < set(configured["tests"])
    assert configured["native_test_files"] == (
        "targetgen/test_compile_only_transport.py",
        "targetgen/test_shared_execution_deadline.py",
        "targetgen/test_frontend_use_def.py",
        "targetgen/test_stack_frame_preflight.py",
        "targetgen/test_explicit_execution_service.py",
        "infra/test_build_only_service.py",
        "targetgen/test_zero_input_abi.py",
    )
    xml, report = tmp_path / "synthetic.xml", {}
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for filename in configured["tests"]:
        ET.SubElement(
            suite,
            "testcase",
            {
                "file": filename,
                "classname": Path(filename).with_suffix("").as_posix().replace("/", "."),
                "name": "test_private_diagnostic",
            },
        )
    ET.ElementTree(root).write(xml)
    Q.check_native_test_report("compile-only", xml, report)
    assert report["native_test_counts"] == {"tests": 7, "skipped": 0}
    assert report["other_test_counts"] == {"tests": 5, "skipped": 0}
    native_case = next(case for case in suite if case.get("file") == "targetgen/test_zero_input_abi.py")
    ET.SubElement(native_case, "skipped", {"message": "unit prerequisite removed"})
    ET.ElementTree(root).write(xml)
    with pytest.raises(Q.QualificationFailed, match="zero skips"):
        Q.check_native_test_report("compile-only", xml, {})


def test_component_roster_imports_and_archives_only_explicit_new_owners():
    configured = Q.SUITES["component-convergence"]
    assert configured["native_tools"] == ("clang", "mlir-translate", "riscv-gcc")
    assert configured["native_test_files"] == (
        "test_component_compile_role_transport.py",
        "test_component_native_deadline.py",
        "test_component_pointer_entry.py",
    )
    assert {
        "test_component_compile_admission.py",
        "test_component_compile_role_transport.py",
        "test_component_compile_graphs.py",
        "test_feedback_guardian.py",
        "test_component_automatic.py",
        "test_component_native_deadline.py",
        "test_component_container_context.py",
        "test_fresh_author_tools.py",
        "test_fresh_author_compiler_tools.py",
        "test_runtime_dependency_projection.py",
        "test_component_operator_schemas.py",
        "test_rtl_state_control.py",
        "test_rtl_native_memory.py",
    } <= set(configured["tests"])
    assert {
        "merlin_experiments.phase1.component_generation_admission",
        "merlin_experiments.phase1.component_compile_admission",
        "merlin_experiments.phase1.component_compile_roles",
        "merlin_experiments.phase0.component_compile_graphs",
        "merlin_experiments.phase2.feedback_guardian",
        "merlin_experiments.execution.owned_children",
        "merlin_experiments.phase0.component_automatic",
        "merlin_experiments.phase0.component_automatic_plan",
        "merlin_experiments.phase1.component_tool_readiness",
        "merlin_experiments.phase2.component_stage",
        "merlin_experiments.phase2.component_final_qualification",
        "merlin_experiments.phase2.component_launch_probe",
        "merlin_experiments.phase2.protected_final_observation",
        "merlin_experiments.phase2.physical_final_admission",
        "merlin_experiments.phase0.operator_schema_intake",
        "merlin.targetgen.frontend_operator_effects",
        "merlin.targetgen.torch_schema_observer",
        "merlin_experiments.phase2.rtl_state_control",
    } <= set(configured["probe_modules"])
    assert configured["support_files"] == (
        "test_phase2_broker.py",
        "test_phase0_freeze.py",
        "component_baseline_fixture.py",
        "test_edit_authority.py",
        "reviewed_corpus_fixtures.py",
        "component_launch_fixture.py",
        "rtl_native_control_cases.py",
    )
    for filename in (*configured["tests"], *configured["support_files"]):
        assert (repo_root() / "packages/merlin-experiments/tests" / filename).is_file()


def test_native_tool_changed_by_actual_child_is_refused_after_command_completion(tmp_path):
    selections = []
    for name in Q.SUITES["component-convergence"]["native_tools"]:
        tool = tmp_path / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o700)
        selections.append(name + "=" + str(tool))
    selected = Q.capture_native_tools("component-convergence", selections)
    report = {"commands": [], "native_tools": selected}
    recorder = Q.Recorder(tmp_path, report, 5)
    recorder.run(
        "actual-byte-mutation",
        [
            sys.executable,
            "-I",
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('changed')",
            selected["clang"]["path"],
        ],
        tmp_path,
    )
    assert report["commands"][0]["status"] == "passed"
    with pytest.raises(Q.QualificationFailed, match="changed"):
        Q.verify_native_tools(selected)
    with pytest.raises(Q.QualificationFailed, match="changed"):
        recorder.run("must-not-launch", [sys.executable, "-c", "pass"], tmp_path)
    assert len(report["commands"]) == 1
