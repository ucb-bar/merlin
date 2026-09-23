"""Analysis-owned reporting with explicit roots, cold imports and installed-shaped help."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

from merlin.agentreport.phase1 import by_model, by_treatment, runs
from merlin.common.paths import merlin_dir, module_source_path, python_import_roots, python_source_dir


def _run(root, name, *, cost=None, notional=None, provider="unknown"):
    directory = root / "raw_baseline" / name
    directory.mkdir(parents=True)
    for filename, content in {
        "environment.yaml": {"bundle_id": "raw_baseline_public_v0", "model": "fixture", "provider": provider},
        "cost_time_toolcalls.yaml": {
            "wall_time_seconds": 90,
            "estimated_cost_usd": cost,
            "subscription_notional_usd": notional,
            "tokens_total": 17,
        },
        "qa_loop_summary.yaml": {"converged": True},
    }.items():
        (directory / filename).write_text(yaml.safe_dump(content))
    return directory


def test_interleaved_targets_keep_run_audit_and_bridge_inputs_separate(tmp_path):
    first = runs.ReportInputs("first", tmp_path / "first-runs", tmp_path / "first-reports")
    second = runs.ReportInputs("second", tmp_path / "second-runs", tmp_path / "second-reports")
    first_run = _run(first.runs_root, "same", cost=None)
    _run(second.runs_root, "same", cost=7)
    (first_run / "bridge.json").write_text('{"bridged": true}')
    for inputs, passed in ((first, "1/2"), (second, "2/2")):
        inputs.reports_root.mkdir()
        (inputs.reports_root / "full_suite_audit.json").write_text(
            json.dumps({"backends": {"same": {"passed": passed}}})
        )
    for inputs, expected_cost, expected_passed, bridged in (
        (first, None, 1, True),
        (second, 7, 2, False),
        (first, None, 1, True),
    ):
        cell = by_treatment.collect(None, inputs=inputs)[("baseline", "kernels")][0]
        assert cell["cost_usd"] == expected_cost
        assert cell["fullsuite"]["all"]["passed"] == expected_passed
        rows = by_model.collect(None, None, inputs=inputs)
        assert rows[0]["cost_usd"] == expected_cost
        assert next(iter(by_model.by_cell(rows, runs_root=inputs.runs_root).values()))["bridged"] is bridged
        assert f"target `{inputs.target}`" in by_model.markdown(
            by_model.by_model(rows), rows, None, target=inputs.target
        )


def test_model_report_keeps_unpriced_notional_and_lower_bound_qualifications(tmp_path):
    inputs = runs.ReportInputs("fixture", tmp_path / "runs", tmp_path / "reports")
    _run(inputs.runs_root, "unpriced")
    _run(inputs.runs_root, "seat", notional=5, provider="subscription")
    rows = by_model.collect(None, None, inputs=inputs)
    seat = next(row for row in rows if row["run_id"] == "seat")
    seat["billing_mode"] = "subscription_notional"
    seat["codex"] = {"tokens_are_lower_bound": True}
    summary = by_model.by_model(rows)["fixture"]
    assert summary["unpriced_runs"] == 1
    assert summary["metered_cost_usd"] == 0
    assert summary["notional_cost_usd"] == 5
    assert summary["lower_bound_token_runs"] == 1
    text = by_model.markdown({"fixture": summary}, rows, None, target=inputs.target)
    assert "unpriced" in text and "notional" in text and "lower bound" in text


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase1-report-installed")
    destination = root / "site"
    shutil.copytree(
        python_source_dir() / "merlin", destination / "merlin", ignore=shutil.ignore_patterns("__pycache__")
    )
    owner = module_source_path("merlin.agentreport.phase1.runs").parent
    shutil.copytree(owner, destination / "merlin/agentreport/phase1", ignore=shutil.ignore_patterns("__pycache__"))
    dependencies = sorted({str(Path(module.__file__).resolve().parent.parent) for module in (yaml, jsonschema)})
    return root, destination, dependencies


@pytest.mark.parametrize("module", ["runs", "by_treatment", "by_model"])
def test_installed_shaped_help_and_import_are_target_inert(installed, module):
    root, site, dependencies = installed
    code = r"""
import importlib, importlib.abc, pathlib, runpy, sys
sys.path[:0] = __PATHS__
class RefuseNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == '_common' or fullname.startswith(('agg_', 'merlin_experiments')):
            raise AssertionError('native/experiments dependency: ' + fullname)
sys.meta_path.insert(0, RefuseNative())
name = 'merlin.agentreport.phase1.' + sys.argv[1]
owner = importlib.import_module(name)
assert pathlib.Path(owner.__file__).is_relative_to(pathlib.Path(sys.argv[2]))
assert not hasattr(owner, 'C')
for loaded_name, loaded in list(sys.modules.items()):
    if loaded_name.startswith('merlin.') and getattr(loaded, '__file__', None):
        assert pathlib.Path(loaded.__file__).is_relative_to(pathlib.Path(sys.argv[2])), loaded_name
sys.argv = [name, '--help']
try:
    owner.main()
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('help did not exit')
""".replace("__PATHS__", repr([str(site), *dependencies]))
    environment = {key: value for key, value in os.environ.items() if not key.startswith("MERLIN_")}
    environment["MERLIN_TARGET_EXPERIMENT"] = str(root / "must-not-read.yaml")
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, module, str(site)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--runs-root" in result.stdout and "--reports-root" in result.stdout
    assert not (root / "out").exists()


def test_cli_requires_explicit_context_and_retains_output_override(tmp_path):
    with pytest.raises(SystemExit) as missing:
        by_model.main([])
    assert missing.value.code == 2
    run_root = tmp_path / "runs"
    _run(run_root, "one")
    reports, output = tmp_path / "reports", tmp_path / "output"
    assert (
        by_model.main(
            [
                "--target",
                "fixture",
                "--runs-root",
                str(run_root),
                "--reports-root",
                str(reports),
                "--out-dir",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads((output / "by_model.json").read_text())
    assert payload["target"] == "fixture" and payload["n_runs"] == 1
    assert not reports.exists()


def test_descriptor_resolution_and_read_only_cohort(tmp_path):
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: fixture\n")
    inputs = [
        "--descriptor",
        str(descriptor),
        "--runs-root",
        str(tmp_path / "empty"),
        "--reports-root",
        str(tmp_path / "reports"),
    ]
    assert runs.main([*inputs, "--cohort"]) == 1
    assert not (tmp_path / "reports").exists()
    with pytest.raises(SystemExit) as mismatch:
        runs.main([*inputs, "--target", "another", "--cohort"])
    assert mismatch.value.code == 2


@pytest.mark.parametrize("target", ["..", ".", "../escape", "/absolute", "nested/device", "nested\\device"])
def test_target_cannot_escape_default_output_layout(tmp_path, target):
    with pytest.raises(SystemExit) as invalid:
        by_model.main(["--target", target, "--runs-root", str(tmp_path / "runs")])
    assert invalid.value.code == 2
    assert not (tmp_path / "runs").exists()


_NATIVE_REPORTS = (
    ("agg_agentic_results.py", "agentic_results.json"),
    ("agg_ab_results.py", "ab_results.json"),
    ("agg_by_model.py", "by_model.json"),
)


def _native_report(tmp_path, script, arguments, *, descriptor=None, direct=False):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("MERLIN_")}
    environment["PYTHONPATH"] = os.pathsep.join(map(str, python_import_roots()))
    environment["MERLIN_OUT_ROOT"] = str(tmp_path / "out")
    environment["MERLIN_TARGET_EXPERIMENT"] = str(descriptor or tmp_path / "unreadable-ambient.yaml")
    environment["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), environment.get("PATH", "")])
    command = [str(merlin_dir() / "experiments/capsule_bench/harness" / script), *arguments]
    return subprocess.run(
        command if direct else [sys.executable, *command],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("script,filename", _NATIVE_REPORTS)
@pytest.mark.parametrize("selection", ["target", "target_equals", "descriptor"])
def test_native_report_explicit_context_ignores_invalid_ambient_descriptor(tmp_path, script, filename, selection):
    descriptor = tmp_path / "explicit.yaml"
    descriptor.write_text("target: fixture\n")
    context = {
        "target": ["--target", "fixture"],
        "target_equals": ["--target=fixture"],
        "descriptor": [f"--descriptor={descriptor}"],
    }[selection]
    output = tmp_path / "reports"
    result = _native_report(
        tmp_path,
        script,
        [*context, "--runs-root", str(tmp_path / "runs"), "--reports-root", str(output)],
    )
    assert result.returncode == 0, result.stderr
    assert (output / filename).is_file()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("script,filename", _NATIVE_REPORTS)
def test_native_report_retains_ambient_legacy_defaults(tmp_path, script, filename):
    descriptor = tmp_path / "legacy.yaml"
    descriptor.write_text("target: legacyfixture\n")
    result = _native_report(tmp_path, script, [], descriptor=descriptor)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/artifacts/capsule-bench/legacyfixture" / filename).is_file()


@pytest.mark.parametrize("script,filename", _NATIVE_REPORTS)
@pytest.mark.parametrize("option", ["--target", "--descriptor"])
def test_native_report_missing_explicit_value_is_parser_error(tmp_path, script, filename, option):
    result = _native_report(tmp_path, script, [option])
    assert result.returncode == 2
    assert f"argument {option}: expected one argument" in result.stderr
    assert "unreadable-ambient" not in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("script,filename", _NATIVE_REPORTS)
def test_native_report_help_does_not_initialize_ambient_descriptor(tmp_path, script, filename):
    result = _native_report(tmp_path, script, ["--help"])
    assert result.returncode == 0, result.stderr
    assert "--target" in result.stdout and "--reports-root" in result.stdout
    assert not (tmp_path / "out").exists()


def test_model_report_preserves_direct_executable_entrypoint(tmp_path):
    result = _native_report(tmp_path, "agg_by_model.py", ["--help"], direct=True)
    assert result.returncode == 0, result.stderr
    assert "--target" in result.stdout
    assert not (tmp_path / "out").exists()
