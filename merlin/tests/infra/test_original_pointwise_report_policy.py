"""Actual child-report controls, never compiler or release qualification."""

import importlib.util
import shutil
import sys
from xml.etree import ElementTree as ET

import pytest

from merlin.common.paths import repo_root

_SPEC = importlib.util.spec_from_file_location(
    "pointwise_report_qualification", repo_root() / "build_tools/scripts/qualify_installed.py"
)
Q = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(Q)
FILE = "test_component_original_pointwise_execution.py"
METHOD = "test_original_pointwise_ordinary_host_values_and_source_applicability"
MEMBERS = (
    "scalar_relu",
    "scalar_round",
    "scalar_integer",
    "scalar_integer_clamp",
    "round_tail",
    "clamp_rectangle",
    "integer_rectangle",
)


def _child(tmp_path, *, original=False, omit=False):
    tests = tmp_path / "archived-tests"
    tests.mkdir()
    if original:
        for name in Q.SUITES["original-pointwise-host"]["tests"]:
            shutil.copyfile(repo_root() / "packages/merlin-experiments/tests" / name, tests / name)
    else:
        # These bodies only exercise packaging report identity, not compilation.
        if not omit:
            (tests / FILE).write_text(
                "import pytest\n@pytest.mark.parametrize('case', "
                + repr(MEMBERS)
                + ")\ndef "
                + METHOD
                + "(case):\n    assert isinstance(case,str)\n"
            )
        (tests / "test_component_source_applicability.py").write_text("def test_private_report_control(): pass\n")
    evidence = tmp_path / "child"
    evidence.mkdir()
    xml = evidence / "tests.xml"
    recorder = Q.Recorder(evidence, {"commands": [], "native_tools": {}}, 30)
    recorder.run(
        "actual-pytest-report",
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; sys.path[:0]="
            + repr([str(repo_root() / "src"), str(repo_root() / "packages/merlin-experiments/src")])
            + "; import pytest; raise SystemExit(pytest.main(sys.argv[1:]))",
            "-q",
            "-c",
            "/dev/null",
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


def test_new_suite_requires_report_without_native_tools_and_preserves_optional_suites():
    assert Q.native_test_report_required("original-pointwise-host", {"native_tools": {}})
    for name, configured in Q.SUITES.items():
        if name == "original-pointwise-host":
            continue
        assert not Q.native_test_report_required(name, {"native_tools": {}})
        if configured.get("native_tools"):
            assert Q.native_test_report_required(name, {"native_tools": {"selected": {}}})


def test_actual_original_source_tests_cannot_qualify_with_all_execution_members_skipped(tmp_path):
    xml = _child(tmp_path, original=True)
    report = {"native_tools": {}}
    assert Q.native_test_report_required("original-pointwise-host", report)
    with pytest.raises(Q.QualificationFailed, match="zero skips"):
        Q.check_native_test_report("original-pointwise-host", xml, report)
    assert report["suite_test_counts"] == {"tests": 19, "skipped": 7}
    assert report["native_test_counts"] == {"tests": 7, "skipped": 7}
    assert report["missing_native_test_cases"] == []
    assert all("explicit compiler Python" in case["message"] for case in report["test_skips"]["native"])


def test_actual_report_requires_all_seven_original_member_identities(tmp_path):
    xml, report = _child(tmp_path), {}
    Q.check_native_test_report("original-pointwise-host", xml, report)
    assert report["native_test_counts"] == {"tests": 7, "skipped": 0}
    assert report["missing_native_test_cases"] == report["unexpected_native_test_cases"] == []
    assert report["native_test_cases"] == [[FILE, METHOD + "[" + name + "]"] for name in MEMBERS]


def test_actual_child_omitting_native_file_cannot_qualify(tmp_path):
    xml, report = _child(tmp_path, omit=True), {}
    with pytest.raises(Q.QualificationFailed, match="every declared native test file"):
        Q.check_native_test_report("original-pointwise-host", xml, report)
    assert report["missing_native_test_files"] == [FILE]
    assert len(report["missing_native_test_cases"]) == 7


@pytest.mark.parametrize(
    "defect", ("missing-member", "changed-member", "foreign-file", "subclass", "duplicate", "failure", "error")
)
def test_mutated_actual_child_report_cannot_replace_original_member(tmp_path, defect):
    xml, report = _child(tmp_path), {}
    tree = ET.parse(xml)
    suite = tree.getroot().find("testsuite")
    case = next(case for case in suite if case.get("file") == FILE)
    if defect == "missing-member":
        suite.remove(case)
    elif defect == "changed-member":
        case.set("name", METHOD + "[foreign]")
    elif defect == "foreign-file":
        case.set("file", "test_unarchived.py")
    elif defect == "subclass":
        case.set("classname", "test_component_original_pointwise_execution.SubstitutedOwner")
    elif defect == "duplicate":
        suite.append(ET.fromstring(ET.tostring(case)))
    else:
        ET.SubElement(case, defect, {"message": "owned negative report mutation"})
    tree.write(xml)
    with pytest.raises(Q.QualificationFailed):
        Q.check_native_test_report("original-pointwise-host", xml, report)
    assert report["native_test_report"] == {"path": str(xml), "sha256": Q.digest(xml)}
