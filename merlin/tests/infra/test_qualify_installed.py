"""Hermetic qualification-tool regressions; no builds, downloads or network services."""

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def load(name):
    path = repo_root() / "build_tools/scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q = load("qualify_installed")


def test_guarded_test_entrypoint_blocks_execution_before_pytest_main(monkeypatch):
    import runpy
    import socket
    import subprocess

    # Register restoration before the script deliberately assigns its tripwires.
    monkeypatch.setattr(subprocess, "Popen", subprocess.Popen)
    monkeypatch.setattr(socket.socket, "bind", socket.socket.bind)
    observed = []

    def fake_main(arguments):
        observed.append(arguments)
        with pytest.raises(AssertionError, match="cannot launch"):
            subprocess.Popen(["never-executed"])
        with pytest.raises(AssertionError, match="cannot launch"):
            socket.socket.bind(None, ("127.0.0.1", 0))
        return 17

    monkeypatch.setitem(sys.modules, "pytest", SimpleNamespace(main=fake_main))
    monkeypatch.setattr(sys, "argv", ["probe", "--guarded-tests", "--collect-only"])
    original = list(sys.meta_path)
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(
                str(repo_root() / "build_tools/scripts/installed_qualification_probe.py"), run_name="__main__"
            )
        assert exc.value.code == 17
        assert observed == [["--collect-only"]]
    finally:
        sys.meta_path[:] = original


@pytest.fixture
def probe():
    original = list(sys.meta_path)
    module = load("installed_qualification_probe")
    try:
        yield module
    finally:
        sys.meta_path[:] = original


@pytest.mark.parametrize("label", ["../escape", "/absolute", ".", "..", "a/b", "-bad", ""])
def test_output_refuses_unsafe_labels(tmp_path, label):
    with pytest.raises(ValueError):
        Q.reserve_output(tmp_path / "outputs", label)


def test_output_refuses_existing_even_empty_and_linked_ancestors(tmp_path):
    base = tmp_path / "outputs"
    first = Q.reserve_output(base, "first")
    with pytest.raises(FileExistsError):
        Q.reserve_output(base, "first")
    assert list(first.iterdir()) == []
    linked = tmp_path / "linked"
    linked.symlink_to(base, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        Q.reserve_output(linked / "child", "run")


def test_environment_removes_source_and_provider_overrides(monkeypatch):
    for key in ("PYTHONPATH", "PYTHONHOME", "MERLIN_TARGET_PATH", "AET_CONFIG", "CHIA_CONFIG"):
        monkeypatch.setenv(key, "must not leak")
    environment = Q.clean_environment()
    assert not any(k.startswith(("PYTHON", "MERLIN", "AET_", "CHIA_")) for k in environment)


def test_versions_and_assisted_extra_come_from_projects(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="custom-core"\nversion="9.8.7"\n[project.optional-dependencies]\nxdsl=["xdsl>=1"]\n'
    )
    extension = tmp_path / "packages/merlin-experiments"
    extension.mkdir(parents=True)
    (extension / "pyproject.toml").write_text('[project]\nname="custom-extension"\nversion="6.5.4"\n')
    projects = Q.projects(tmp_path, ("xdsl",))
    assert [p["version"] for p in projects] == ["9.8.7", "6.5.4"]
    assert projects[0]["extras"] == ["xdsl"] and projects[1]["extras"] == []
    with pytest.raises(Q.QualificationFailed, match="does not declare"):
        Q.projects(tmp_path, ("undeclared",))


def test_failed_child_keeps_terminal_record_and_log(tmp_path):
    report = {"commands": []}
    recorder = Q.Recorder(tmp_path, report, 5)
    with pytest.raises(Q.QualificationFailed):
        recorder.run("failed", [sys.executable, "-c", "print('failure evidence'); raise SystemExit(7)"], tmp_path)
    saved = json.loads((tmp_path / "report.json").read_text())["commands"][0]
    assert saved["returncode"] == 7 and saved["status"] == "failed"
    assert saved["timeout_s"] == 5 and saved["elapsed_s"] >= 0
    assert "failure evidence" in (tmp_path / "failed.log").read_text()


def test_timeout_is_recorded_and_child_is_reaped(tmp_path):
    recorder = Q.Recorder(tmp_path, {"commands": []}, 0.05)
    with pytest.raises(Q.QualificationFailed, match="timeout"):
        recorder.run("slow", [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path)
    saved = json.loads((tmp_path / "report.json").read_text())["commands"][0]
    assert saved["status"] == "timeout" and saved["returncode"] < 0
    assert (tmp_path / "slow.log").is_file()


def test_absent_program_is_recorded(tmp_path):
    recorder = Q.Recorder(tmp_path, {"commands": []}, 1)
    with pytest.raises(Q.QualificationFailed, match="launch_failed"):
        recorder.run("absent", [tmp_path / "nonexistent-program"], tmp_path)
    assert json.loads((tmp_path / "report.json").read_text())["commands"][0]["status"] == "launch_failed"


def test_native_guard_blocks_submodules(probe):
    for name in (
        "chia",
        "ray.worker",
        "run_baseline_qa_loop",
        "_pbcommon",
        "run_agentic_perf_experiment",
        "run_paired_perf_bench",
        "run_global_perf_experiment",
        "perf_agent_stage",
        "chia_agentic_perf_experiment",
    ):
        with pytest.raises(AssertionError, match="unexpected native"):
            probe.NoNative().find_spec(name)
    assert probe.NoNative().find_spec("merlin_experiments.phase1.controller") is None


def test_origin_check_rejects_checkout_import(probe, monkeypatch, tmp_path):
    site = tmp_path / "site"
    monkeypatch.setattr(probe.sysconfig, "get_path", lambda _: str(site))
    monkeypatch.setattr(
        probe,
        "sys",
        SimpleNamespace(
            modules={
                "merlin_experiments.phase1.controller": SimpleNamespace(
                    __file__=str(tmp_path / "checkout/controller.py")
                ),
            }
        ),
    )
    with pytest.raises(AssertionError):
        probe.assert_installed_origins()
    probe.sys.modules["merlin_experiments.phase1.controller"].__file__ = str(site / "controller.py")
    assert probe.assert_installed_origins() == ["merlin_experiments.phase1.controller"]


def test_probe_compares_actual_installed_bytes(probe, monkeypatch, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    leaf = site / "payload.py"
    leaf.write_bytes(b"original")
    wheel = tmp_path / "fixture-1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("payload.py", b"original")
    monkeypatch.setattr(probe.sysconfig, "get_path", lambda _: str(site))
    monkeypatch.setattr(probe, "sys", SimpleNamespace(modules={}))
    imported = []
    monkeypatch.setattr(probe.importlib, "import_module", imported.append)
    monkeypatch.setattr(probe.importlib.util, "find_spec", lambda _: object())
    assert probe.probe([wheel], modules=("fixture.module",), required_modules=("fixture",))["verified_payloads"] == 1
    assert imported == ["fixture.module"]
    monkeypatch.setattr(probe.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(AssertionError, match="suite requires module: fixture"):
        probe.probe([wheel], required_modules=("fixture",))
    leaf.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="payload.py"):
        probe.probe([wheel])


@pytest.mark.parametrize("missing_extra", [False, True])
@pytest.mark.parametrize("suite", list(Q.SUITES))
def test_pipeline_uses_archived_versions_extra_and_probe_before_pytest(monkeypatch, tmp_path, missing_extra, suite):
    output = tmp_path / "evidence"
    output.mkdir()
    calls = []
    monkeypatch.setattr(Q, "resolve_ref", lambda *_: "a" * 40)
    external = tmp_path / "external-venv"
    external.mkdir()
    monkeypatch.setattr(Q.tempfile, "mkdtemp", lambda **_: str(external))

    def run(self, label, argv, cwd, *, stdout=None):
        calls.append((label, list(map(str, argv))))
        if label == "resource-manifest":
            Path(stdout).write_text('{"files": []}')
        elif label == "source-archive":
            core = '[project]\nname="merlin"\nversion="7.8.9"\n'
            if not missing_extra:
                core += '[project.optional-dependencies]\nxdsl=["xdsl>=0.68"]\n'
            files = {
                "pyproject.toml": core,
                "packages/merlin-experiments/pyproject.toml": '[project]\nname="merlin-experiments"\nversion="9.8.7"\n',
            }
            for name in (*Q.SUITES[suite]["tests"], *Q.SUITES[suite].get("support_files", ())):
                tests_root = Q.SUITES[suite].get("tests_root", "packages/merlin-experiments/tests")
                files[tests_root + "/" + name] = "# committed synthetic test\n"
            with tarfile.open(stdout, "w") as archive:
                for name, text in files.items():
                    data = text.encode()
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
        elif label.endswith(("-sdist", "-wheel")):
            directory = Path(argv[argv.index("--out-dir") + 1])
            (directory / ("fixture.tar.gz" if label.endswith("-sdist") else "fixture.whl")).write_bytes(b"artifact")
        elif label == "freeze":
            Path(stdout).write_text("pytest==synthetic\n")

    monkeypatch.setattr(Q.Recorder, "run", run)
    success = Q.qualify(repo_root(), output, "b" * 40, suite, 5, requested_ref="named-ref", invocation=["synthetic"])
    report = json.loads((output / "report.json").read_text())
    assert report["requested_ref"] == "named-ref" and report["ref"] == "b" * 40
    if missing_extra and Q.SUITES[suite]["core_extras"]:
        assert not success and report["status"] == "failed"
        assert not any(name.endswith("-sdist") for name, _ in calls)
        return
    assert success
    versions = ["7.8.9", "9.8.7"] if Q.SUITES[suite].get("include_experiments", True) else ["7.8.9"]
    assert [p["version"] for p in report["projects"]] == versions
    labels = [name for name, _ in calls]
    assert (
        labels.index("install")
        < labels.index("payload-probe")
        < labels.index("pytest-install")
        < labels.index("freeze")
    )
    install = dict(calls)["install"]
    assert "pytest" not in install
    assert any(item.endswith(".whl[xdsl]") for item in install) == bool(Q.SUITES[suite]["core_extras"])
    assert report["selected_tests"] == list(Q.SUITES[suite]["tests"])
    assert report["support_files"] == list(Q.SUITES[suite].get("support_files", ()))
    assert report["probe_modules"] == list(Q.SUITES[suite]["probe_modules"])
    probe_command = dict(calls)["payload-probe"]
    for module in Q.SUITES[suite]["probe_modules"]:
        assert probe_command[probe_command.index(module) - 1] == "--module"
    assert ("--require-module" in probe_command) == bool(Q.SUITES[suite]["required_modules"])
    assert report["tests_root"] == Q.SUITES[suite].get("tests_root", "packages/merlin-experiments/tests")
    archive_command = dict(calls)["source-archive"]
    assert archive_command[:4] == ["git", "archive", "--format=tar", "b" * 40]
    for name in (*Q.SUITES[suite]["tests"], *Q.SUITES[suite].get("support_files", ())):
        assert report["tests_root"] + "/" + name in archive_command
        assert report["tests_root"] + "/" + name in report["source_files"]
        # Tests are copied from the selected commit archive, never the live checkout.
        assert (external / "qualification-tests" / name).read_text() == "# committed synthetic test\n"
    assert dict(calls)["tests"][-1] == str(external / "qualification-tests")
    test_command = dict(calls)["tests"]
    guarded = bool(Q.SUITES[suite].get("guarded_tests"))
    assert ("--guarded-tests" in test_command) is guarded
    assert report["test_process_policy"] == ("deny_processes_and_listeners" if guarded else "suite_defined")
    assert test_command[test_command.index("--basetemp") + 1] == str(external / "test-tmp")
    if not Q.SUITES[suite].get("include_experiments", True):
        assert len(report["projects"]) == 1
        assert not any("experiments" in label for label in labels)
        assert not any(module.startswith("merlin_experiments") for module in report["probe_modules"])
    assert "--source-root" in dict(calls)["layout"]
    assert (external / "qualification-tests/conftest.py").read_bytes() == (
        output / "installed_qualification_probe.py"
    ).read_bytes()


def test_phase2_policy_qualification_includes_extracted_owners():
    suite = Q.SUITES["phase2-policy"]
    assert suite["tests"] == (
        "test_phase2_workflow_policy.py",
        "test_phase2_broker_evidence.py",
        "test_phase2_transcript_audit.py",
        "test_phase2_functional_inputs.py",
    )
    assert suite["probe_modules"] == tuple(
        "merlin_experiments.phase2." + owner
        for owner in (
            "broker",
            "broker_policy",
            "broker_evidence",
            "corpus_feedback",
            "whole_model",
            "transcript_audit",
            "functional_inputs",
        )
    )


def test_target_fetch_qualification_is_core_only():
    suite = Q.SUITES["target-fetch"]
    assert suite["include_experiments"] is False
    assert suite["tests_root"] == "merlin/tests/targetgen"
    assert suite["tests"] == ("test_oot_fetch.py",)
    assert suite["core_extras"] == ()
    assert suite["required_modules"] == ()
    assert suite["probe_modules"] == ("merlin.targetgen.oot_fetch",)


def test_tensor_inspection_qualification_is_core_only():
    suite = Q.SUITES["tensor-inspection"]
    assert suite["include_experiments"] is False
    assert suite["tests_root"] == "merlin/tests/ir"
    assert suite["tests"] == ("test_ir_audit_tensors.py", "test_inspection_tensor_payloads.py")
    assert suite["core_extras"] == ("xdsl",)
    assert suite["required_modules"] == ("xdsl",)
    assert suite["probe_modules"] == ("merlin.common.ir_audit", "merlin.xdsl_dialects.ir_inspection")


def test_interrupted_pipeline_records_terminal_status(monkeypatch, tmp_path):
    monkeypatch.setattr(Q, "resolve_ref", lambda *_: "a" * 40)

    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(Q.Recorder, "run", interrupted)
    assert not Q.qualify(repo_root(), tmp_path, "b" * 40, "phase1", 5)
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "interrupted"
