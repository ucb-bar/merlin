"""Actual packaging report refusals; no descriptor execution or authority."""

import importlib.util
import shutil
import sys
from xml.etree import ElementTree as ET

import pytest

from merlin.common.paths import repo_root

_SPEC = importlib.util.spec_from_file_location(
    "descriptor_report_qualification", repo_root() / "build_tools/scripts/qualify_installed.py"
)
Q = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(Q)
SUITE = "host-ranked-descriptors"


@pytest.fixture(scope="module")
def actual_report(tmp_path_factory):
    owner = tmp_path_factory.mktemp("descriptor-report-policy")
    tests, child = owner / "archived-tests", owner / "child"
    tests.mkdir()
    child.mkdir()
    for name in Q.SUITES[SUITE]["tests"]:
        shutil.copyfile(repo_root() / "merlin/tests/ir" / name, tests / name)
    xml = child / "tests.xml"
    recorder = Q.Recorder(child, {"commands": [], "native_tools": {}}, 30)
    recorder.environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    recorder.run(
        "actual-descriptor-pytest-report",
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; sys.path.insert(0,"
            + repr(str(repo_root() / "src"))
            + "); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))",
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
            child / "test-tmp",
            tests,
        ],
        child,
    )
    return xml


def test_explicit_suite_closes_complete_original_members_and_tool_source_route():
    selected = Q.SUITES[SUITE]
    assert not selected["include_experiments"]
    assert selected["native_tools"] == ("compiler-python", "llvm-llc", "clang")
    assert selected["native_python_entries"] == ("compiler-python",)
    assert selected["native_sources"] == {"m2m": {"package": "m2m", "environment_key": "MERLIN_M2M_DIR"}}
    assert Q.native_test_report_required(SUITE, {"native_tools": {}})
    members = selected["native_test_cases"]
    assert len(members) == len(set(members)) == 48
    assert sum(filename == "test_host_descriptor_selection.py" for filename, _ in members) == 46
    assert {name for filename, name in members if filename == "test_host_descriptor_native.py"} == {
        "test_ordinary_original_complete_outputs_guards_and_actual_descriptor_call[rank_zero_signed_zero]",
        "test_ordinary_original_complete_outputs_guards_and_actual_descriptor_call[tail_wide_repeated_results]",
    }


def test_actual_original_report_with_missing_tools_refuses_both_native_skips(actual_report):
    report = {"native_tools": {}}
    with pytest.raises(Q.QualificationFailed, match="zero skips"):
        Q.check_native_test_report(SUITE, actual_report, report)
    assert report["native_test_counts"] == {"tests": 48, "skipped": 2}
    assert report["missing_native_test_cases"] == report["unexpected_native_test_cases"] == []
    assert all(
        "explicit ordinary public compiler/source selectors" in row["message"] for row in report["test_skips"]["native"]
    )


@pytest.mark.parametrize("defect", ("omitted-pure", "omitted-native", "renamed-native", "duplicate-native"))
def test_actual_report_cannot_drop_or_substitute_original_member(actual_report, tmp_path, defect):
    tree = ET.parse(actual_report)
    suite = tree.getroot().find("testsuite")
    case = next(
        row
        for row in suite
        if row.get("file")
        == ("test_host_descriptor_selection.py" if defect == "omitted-pure" else "test_host_descriptor_native.py")
    )
    if defect.startswith("omitted"):
        suite.remove(case)
    elif defect == "renamed-native":
        case.set("name", "test_substituted_original")
    else:
        suite.append(ET.fromstring(ET.tostring(case)))
    changed = tmp_path / "mutated-child.xml"
    tree.write(changed)
    with pytest.raises(Q.QualificationFailed, match="member|unique"):
        Q.check_native_test_report(SUITE, changed, {})
