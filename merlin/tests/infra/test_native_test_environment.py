"""Exact declared native test environment and refusal before launching children."""

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common.paths import repo_root

SPEC = importlib.util.spec_from_file_location(
    "selected_qualification_tool", repo_root() / "build_tools/scripts/qualify_installed.py"
)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def test_declared_environment_matches_actual_child_and_changed_file_blocks_launch(tmp_path):
    report = {"commands": [], "native_tools": {}}
    selected = Q.retain_test_environment("coherent-measurement", tmp_path, ENV, report)
    pin = report["native_test_environment"]
    path = Path(pin["path"])
    assert json.loads(path.read_bytes()) == selected
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert selected["MERLIN_TARGET_PATH"] == ""
    assert selected[pin["environment_key"]] == str(path)
    result = I.run(
        [sys.executable, "-I", "-c", "import os;print(os.environ['MERLIN_TARGET_PATH']=='')"],
        directory=tmp_path / "child",
        stage="declared_native_test_environment",
        env=selected,
        cwd=tmp_path,
        inputs=(path,),
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0 and result.stdout == b"True\n"
    record = next((tmp_path / "child/invocations").glob("*/invocation.json"))
    I.require_environment(record, environment=selected)
    Q.verify_native_inputs(report)
    path.write_bytes(path.read_bytes() + b" ")
    recorder = Q.Recorder(tmp_path, report, 5)
    recorder.environment = selected
    with pytest.raises(Q.QualificationFailed, match="environment changed"):
        recorder.run("must-not-launch", [sys.executable, "-c", "raise SystemExit(0)"], tmp_path)
    assert report["commands"] == []


def test_absent_environment_selection_preserves_mapping_without_io(tmp_path):
    report = {}
    assert Q.retain_test_environment("serial-llvm-products", tmp_path, ENV, report) == ENV
    assert not report and not list(tmp_path.iterdir())


@pytest.mark.parametrize("defect", ["duplicate", "conflict", "non_string", "existing", "alias"])
def test_environment_selection_refuses_without_replacing_existing_files(tmp_path, defect):
    selected, report = dict(ENV), {}
    output = tmp_path / "native-test-environment.json"
    protected = tmp_path / "protected"
    if defect == "duplicate":
        selected["MERLIN_TEST_MEASUREMENT_ENV_SELECTION"] = "preexisting"
    elif defect == "conflict":
        selected["MERLIN_TARGET_PATH"] = "undeclared-provider"
    elif defect == "non_string":
        selected["count"] = 2
    elif defect == "existing":
        output.write_bytes(b"retained original selection")
    else:
        protected.write_bytes(b"protected original")
        output.symlink_to(protected)
    before = output.read_bytes() if output.exists() else None
    with pytest.raises((Q.QualificationFailed, FileExistsError)):
        Q.retain_test_environment("coherent-measurement", tmp_path, selected, report)
    assert report == {}
    if before is not None:
        assert output.read_bytes() == before
    else:
        assert not output.exists()


def test_registered_coherent_suite_archives_complete_fixture_and_tool_rosters():
    suite = Q.SUITES["coherent-measurement"]
    assert suite["mandatory_test_report"] == "merlin.installed_mandatory_tests.v1"
    assert set(suite["native_tools"]) == {"clang", "mlir-translate", "riscv-gcc", "readelf", "cpu-simulator"}
    assert set(suite["support_files"]) == {
        "coherent_measurement_control.py",
        "coherent_measurement_runner.py",
        "measurement_execution_control.py",
    }
    assert set(suite["native_test_files"]) <= set(suite["tests"])
    assert Q.NATIVE_TOOL_ENVIRONMENT["cpu-simulator"] == "MERLIN_TEST_STOCK_CPU_SIMULATOR"
