"""Hermetic qualification-tool regressions; no builds, downloads or network services."""

import importlib.util
import io
import json
import subprocess
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


def test_packing_memory_suite_keeps_the_complete_original_control_roster():
    suite = Q.SUITES["packing-memory-intake"]
    files = (
        "merlin/tests/targetgen/test_hw_partition_memory_bindings.py",
        "packages/merlin-experiments/tests/test_declared_phase0_run.py",
        "packages/merlin-experiments/tests/test_packing_memory_intake.py",
        "packages/merlin-experiments/tests/test_packing_memory_intake_native.py",
    )
    members = suite["native_test_cases"]
    assert suite["tests"] == suite["native_test_files"] == files
    assert len(members) == len(set(members)) == 129
    assert [sum(filename == selected for filename, _ in members) for selected in files] == [42, 36, 29, 22]
    native_members = tuple(name for filename, name in members if filename == files[-1])
    assert [
        sum(name.endswith("[" + era + "]") or "[" + era + "-" in name for name in native_members)
        for era in (
            "legacy",
            "modern",
        )
    ] == [11, 11]
    assert suite["tests_root"] == "." and suite["test_fixture_imports"] is True
    assert suite["collect_selected_tests"] is True
    assert suite["mandatory_test_report"] == "merlin.installed_mandatory_tests.v1"
    assert suite["required_modules"] == ("xdsl", "jsonschema", "numpy")
    assert suite["native_tools"] == ("firtool", "circt-opt", "firtool-modern", "circt-opt-modern")


@pytest.mark.parametrize("changed", ["firtool", "circt-opt", "firtool-modern", "circt-opt-modern"])
def test_packing_memory_both_explicit_pairs_retain_actual_tool_bytes(tmp_path, monkeypatch, changed):
    suite = "packing-memory-intake"
    supplied = []
    expected_environment = {}
    for name in Q.SUITES[suite]["native_tools"]:
        tool = tmp_path / name
        tool.write_text("#!/bin/sh\n# " + name + "\nexit 0\n")
        tool.chmod(0o700)
        supplied.append(name + "=" + str(tool))
        expected_environment[Q.NATIVE_TOOL_ENVIRONMENT[name]] = str(tool)
    monkeypatch.setenv("MERLIN_TEST_FIRTOOL_MODERN", "/unselected/modern/compiler")
    selected = Q.capture_native_tools(suite, supplied)
    assert Q.native_environment({"native_tools": selected}, suite=suite) == expected_environment
    assert set(expected_environment) == {
        "MERLIN_TEST_FIRTOOL",
        "MERLIN_TEST_CIRCT_OPT",
        "MERLIN_TEST_FIRTOOL_MODERN",
        "MERLIN_TEST_CIRCT_OPT_MODERN",
    }
    Q.verify_native_tools(selected)
    Path(selected[changed]["path"]).write_text("changed explicitly selected compiler")
    with pytest.raises(Q.QualificationFailed, match="changed"):
        Q.verify_native_tools(selected)
    with pytest.raises(Q.QualificationFailed, match="complete explicit tool roster"):
        Q.capture_native_tools(suite, supplied[:-1])


@pytest.mark.parametrize(
    "change",
    [None, "pure_missing", "native_missing", "era", "extra", "duplicate", "skip_pure", "skip_native", "class", "error"],
)
def test_packing_memory_qualification_requires_every_pure_and_native_original_member(tmp_path, change):
    from xml.etree import ElementTree as ET

    suite = "packing-memory-intake"
    policy = Q.SUITES[suite]
    xml = ET.Element("testsuites")
    cases = ET.SubElement(xml, "testsuite")
    for filename, name in policy["native_test_cases"]:
        ET.SubElement(
            cases,
            "testcase",
            {"file": filename, "classname": Path(filename).with_suffix("").as_posix().replace("/", "."), "name": name},
        )
    pure, native = cases[0], cases[-1]
    if change == "pure_missing":
        cases.remove(pure)
    elif change == "native_missing":
        cases.remove(native)
    elif change == "era":
        native.set("name", native.get("name").replace("modern", "unselected"))
    elif change in {"extra", "duplicate"}:
        attributes = dict(pure.attrib)
        if change == "extra":
            attributes["name"] = "test_unselected_substitute"
        ET.SubElement(cases, "testcase", attributes)
    elif change in {"skip_pure", "skip_native"}:
        ET.SubElement(pure if change == "skip_pure" else native, "skipped", {"message": "actual unavailable input"})
    elif change == "class":
        native.set("classname", native.get("classname") + ".Alias")
    elif change == "error":
        ET.SubElement(native, "error", {"message": "actual native failure"})
    path = tmp_path / "actual-report.xml"
    ET.ElementTree(xml).write(path)
    report = {"native_tools": {}}
    assert Q.native_test_report_required(suite, report)
    if change is not None:
        with pytest.raises(Q.QualificationFailed):
            Q.check_native_test_report(suite, path, report)
        return
    Q.check_native_test_report(suite, path, report)
    assert report["suite_test_counts"] == report["native_test_counts"] == {"tests": 129, "skipped": 0}
    assert report["missing_native_test_cases"] == report["unexpected_native_test_cases"] == []


def test_source_input_patterns_are_target_neutral_and_archive_bound(tmp_path):
    pattern = "examples/*/target/descriptor.yaml"
    for target in ("neutral_a", "neutral_b"):
        member = tmp_path / "examples" / target / "target" / "descriptor.yaml"
        member.parent.mkdir(parents=True)
        member.write_text("selected committed descriptor")
    assert Q.source_input_archive_roots((pattern,)) == ("examples",)
    assert Q.selected_source_inputs(tmp_path, (pattern,)) == (
        "examples/neutral_a/target/descriptor.yaml",
        "examples/neutral_b/target/descriptor.yaml",
    )
    for unsafe in ("../escape", "/absolute", "*.yaml"):
        with pytest.raises(Q.QualificationFailed):
            Q.source_input_archive_roots((unsafe,))
    with pytest.raises(Q.QualificationFailed, match="missing"):
        Q.selected_source_inputs(tmp_path, ("missing/*.yaml",))
    (tmp_path / "examples/neutral_a/target/descriptor.yaml").unlink()
    (tmp_path / "examples/neutral_a/target/descriptor.yaml").symlink_to(
        tmp_path / "examples/neutral_b/target/descriptor.yaml"
    )
    with pytest.raises(Q.QualificationFailed, match="unsafe"):
        Q.selected_source_inputs(tmp_path, (pattern,))


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
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "MERLIN_TARGET_PATH",
        "AET_CONFIG",
        "CHIA_CONFIG",
        "UV_OVERRIDE",
        "UV_EXCLUDE",
        "UV_CONSTRAINT",
        "UV_BUILD_CONSTRAINT",
    ):
        monkeypatch.setenv(key, "must not leak")
    environment = Q.clean_environment()
    assert not any(k.startswith(("PYTHON", "MERLIN", "AET_", "CHIA_")) for k in environment)
    assert not {"UV_OVERRIDE", "UV_EXCLUDE", "UV_CONSTRAINT", "UV_BUILD_CONSTRAINT"} & environment.keys()


@pytest.mark.parametrize("suite", ["compile-only", "component-convergence", "original-candidate-members"])
def test_explicit_native_roster_is_closed_complete_and_bound_to_actual_bytes(tmp_path, monkeypatch, suite):
    selections = []
    for name in Q.SUITES[suite]["native_tools"]:
        tool = tmp_path / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o700)
        selections.append(name + "=" + str(tool))
    monkeypatch.setenv("MERLIN_TARGET_PATH", "must not leak")
    monkeypatch.setenv("MERLIN_CLANG", "/unrecorded/compiler")
    selected = Q.capture_native_tools(suite, selections)
    recorder = Q.Recorder(tmp_path, {"commands": [], "native_tools": selected}, 5)
    assert "MERLIN_TARGET_PATH" not in recorder.environment
    assert recorder.environment["MERLIN_CLANG"] == selected["clang"]["path"]
    Q.verify_native_tools(selected)
    Path(selected["clang"]["path"]).write_text("changed selected compiler")
    with pytest.raises(Q.QualificationFailed, match="changed"):
        recorder.run("must-not-launch", [sys.executable, "-c", "raise SystemExit(0)"], tmp_path)
    assert recorder.report["commands"] == []


@pytest.mark.parametrize("defect", ["unknown", "duplicate", "incomplete", "relative", "non-executable", "wrong-suite"])
def test_explicit_native_selection_refuses_bad_rosters(tmp_path, defect):
    tool = tmp_path / "tool"
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o700)
    selections = [name + "=" + str(tool) for name in Q.SUITES["compile-only"]["native_tools"]]
    suite = "compile-only"
    if defect == "unknown":
        selections[0] = "provider=" + str(tool)
    elif defect == "duplicate":
        selections.append(selections[0])
    elif defect == "incomplete":
        selections.pop()
    elif defect == "relative":
        selections[0] = "clang=relative/compiler"
    elif defect == "non-executable":
        tool.chmod(0o600)
    else:
        suite = "phase1"
    with pytest.raises(Q.QualificationFailed):
        Q.capture_native_tools(suite, selections)


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


def test_analysis_project_is_explicit_and_uses_declared_distribution(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="core"\nversion="1"\n')
    analysis = tmp_path / "packages/merlin-analysis"
    analysis.mkdir(parents=True)
    (analysis / "pyproject.toml").write_text('[project]\nname="analysis"\nversion="2"\n')
    assert [row["name"] for row in Q.projects(tmp_path, (), include_experiments=False)] == ["core"]
    assert [row["name"] for row in Q.projects(tmp_path, (), include_experiments=False, include_analysis=True)] == [
        "core",
        "analysis",
    ]
    suite = Q.SUITES["capture-contraction-formats"]
    assert suite["include_analysis"] is True and suite["include_experiments"] is False
    assert suite["tests"] == (
        "targetgen/test_contraction_formats.py",
        "dse/test_capture_format_freezing.py",
        "dse/test_capture_format_policy.py",
    )
    assert "merlin.compare.freeze" in suite["probe_modules"]


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
@pytest.mark.parametrize(
    "suite,native",
    [
        *((suite, False) for suite in Q.SUITES),
        ("compile-only", True),
        ("component-convergence", True),
        ("original-pointwise-host", True),
        ("original-candidate-members", True),
        ("host-ranked-descriptors", True),
        ("packing-memory-intake", True),
    ],
)
def test_pipeline_uses_archived_versions_extra_and_probe_before_pytest(
    monkeypatch,
    tmp_path,
    missing_extra,
    suite,
    native,
):
    output = tmp_path / "evidence"
    output.mkdir()
    calls = []
    monkeypatch.setattr(Q, "resolve_ref", lambda *_: "a" * 40)
    external = tmp_path / "external-venv"
    external.mkdir()
    monkeypatch.setattr(Q.tempfile, "mkdtemp", lambda **_: str(external))

    def run(self, label, argv, cwd, *, stdout=None):
        calls.append((label, list(map(str, argv))))
        if suite == "packing-memory-intake" and native:
            for name in Q.SUITES[suite]["native_tools"]:
                assert self.environment[Q.NATIVE_TOOL_ENVIRONMENT[name]] == str(tmp_path / name)
        if suite in ("original-pointwise-host", "original-candidate-members", "host-ranked-descriptors") and native:
            assert self.environment["MERLIN_COMPILER_PYTHON"] == str(tmp_path / "compiler-python")
            assert self.environment["MERLIN_LLVM_LLC"] == str(tmp_path / "llvm-llc")
            assert self.environment["MERLIN_M2M_DIR"] == str(frontend)
            if suite == "host-ranked-descriptors":
                assert self.environment["MERLIN_CLANG"] == str(tmp_path / "clang")
                assert "MERLIN_MLIR_TRANSLATE" not in self.environment
            else:
                assert self.environment["MERLIN_MLIR_TRANSLATE"] == str(tmp_path / "mlir-translate")
            if suite == "original-candidate-members":
                assert self.environment["MERLIN_TEST_BWRAP"] == str(tmp_path / "bwrap")
                assert self.environment["MERLIN_CLANG"] == str(tmp_path / "clang")
        if label == "resource-manifest":
            Path(stdout).write_text('{"files": []}')
        elif label == "source-archive":
            core = '[project]\nname="merlin"\nversion="7.8.9"\n'
            if not missing_extra:
                core += '[project.optional-dependencies]\nxdsl=["xdsl>=0.68"]\ntargetgen=["jsonschema>=4"]\n'
            files = {
                "pyproject.toml": core,
                "packages/merlin-experiments/pyproject.toml": '[project]\nname="merlin-experiments"\nversion="9.8.7"\n',
                "packages/merlin-analysis/pyproject.toml": '[project]\nname="merlin-analysis"\nversion="2.3.4"\n',
            }
            for name in (*Q.SUITES[suite]["tests"], *Q.SUITES[suite].get("support_files", ())):
                tests_root = Q.SUITES[suite].get("tests_root", "packages/merlin-experiments/tests")
                files[tests_root + "/" + name] = "# committed synthetic test\n"
            for pattern in Q.SUITES[suite].get("source_inputs", ()):
                files[pattern.replace("*", "neutral")] = "# committed synthetic input\n"
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
        elif label == "tests" and (native or Q.SUITES[suite].get("mandatory_test_report")):
            from xml.etree import ElementTree as ET

            # Synthetic orchestration fixture, never a native qualification.
            xml = ET.Element("testsuites")
            selected = ET.SubElement(xml, "testsuite")
            members = Q.SUITES[suite].get("native_test_cases") or tuple(
                (name, "test_synthetic_pipeline") for name in Q.SUITES[suite]["native_test_files"]
            )
            for name, method in members:
                ET.SubElement(
                    selected,
                    "testcase",
                    {
                        "file": name,
                        "classname": Path(name).with_suffix("").as_posix().replace("/", "."),
                        "name": method,
                    },
                )
            if suite == "component-convergence":
                other = ET.SubElement(
                    selected,
                    "testcase",
                    {
                        "file": "test_component_generation.py",
                        "classname": "test_component_generation",
                        "name": "test_synthetic_unselected_prerequisite",
                    },
                )
                ET.SubElement(other, "skipped", {"message": "separate synthetic prerequisite"})
            ET.ElementTree(xml).write(argv[argv.index("--junitxml") + 1])

    monkeypatch.setattr(Q.Recorder, "run", run)
    selections = []
    sources = []
    if native:
        for name in Q.SUITES[suite]["native_tools"]:
            tool = tmp_path / name
            tool.write_text("#!/bin/sh\nexit 0\n")
            tool.chmod(0o700)
            selections.append(name + "=" + str(tool))
        if suite in ("original-pointwise-host", "original-candidate-members", "host-ranked-descriptors"):
            frontend = tmp_path / "frontend"
            (frontend / "m2m").mkdir(parents=True)
            (frontend / "m2m/__init__.py").write_text("# owned source identity fixture\n")
            (frontend / "pyproject.toml").write_text('[project]\nname="owned-control"\nversion="0"\n')
            environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
            for argv in (
                ["git", "init", "-q", frontend],
                ["git", "-C", frontend, "add", "."],
                [
                    "git",
                    "-C",
                    frontend,
                    "-c",
                    "user.name=Control",
                    "-c",
                    "user.email=control@example.invalid",
                    "commit",
                    "-qm",
                    "source",
                ],
            ):
                subprocess.run(argv, env=environment, check=True, capture_output=True)
            selected_commit = (
                subprocess.check_output(["git", "-C", frontend, "rev-parse", "HEAD"], env=environment).decode().strip()
            )
            sources.append("m2m=" + str(frontend) + "@" + selected_commit)
    success = Q.qualify(
        repo_root(),
        output,
        "b" * 40,
        suite,
        5,
        requested_ref="named-ref",
        invocation=["synthetic"],
        native_tools=selections,
        native_sources=sources,
    )
    report = json.loads((output / "report.json").read_text())
    assert report["requested_ref"] == "named-ref" and report["ref"] == "b" * 40
    if missing_extra and Q.SUITES[suite]["core_extras"]:
        assert not success and report["status"] == "failed"
        assert not any(name.endswith("-sdist") for name, _ in calls)
        return
    assert success
    versions = ["7.8.9", "9.8.7"] if Q.SUITES[suite].get("include_experiments", True) else ["7.8.9"]
    if Q.SUITES[suite].get("include_analysis"):
        versions.append("2.3.4")
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
    extra_suffix = ".whl[" + ",".join(Q.SUITES[suite]["core_extras"]) + "]"
    assert any(item.endswith(extra_suffix) for item in install) == bool(Q.SUITES[suite]["core_extras"])
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
        member = (Path(report["tests_root"]) / name).as_posix()
        assert member in archive_command
        assert member in report["source_files"]
        # Tests are copied from the selected commit archive, never the live checkout.
        assert (external / "qualification-tests" / name).read_text() == "# committed synthetic test\n"
    expected_tests = (
        [str(external / "qualification-tests" / name) for name in Q.SUITES[suite]["tests"]]
        if Q.SUITES[suite].get("collect_selected_tests")
        else [str(external / "qualification-tests")]
    )
    assert dict(calls)["tests"][-len(expected_tests) :] == expected_tests
    test_command = dict(calls)["tests"]
    mandatory_report = native or bool(Q.SUITES[suite].get("mandatory_test_report"))
    assert ("--junitxml" in test_command) is mandatory_report
    if mandatory_report:
        assert test_command[test_command.index("--rootdir") + 1] == str(external / "qualification-tests")
        assert "junit_family=xunit1" in test_command
        assert report["native_test_counts"] == {
            "tests": len(Q.SUITES[suite].get("native_test_cases") or Q.SUITES[suite]["native_test_files"]),
            "skipped": 0,
        }
        assert report["missing_native_test_files"] == []
        assert report["native_zero_skip_scope"] == "declared_native_test_files"
        assert report["other_test_counts"]["skipped"] == int(suite == "component-convergence")
        assert report["native_test_report"]["sha256"] == Q.digest(output / "tests.xml")
    guarded = bool(Q.SUITES[suite].get("guarded_tests"))
    assert ("--guarded-tests" in test_command) is guarded
    assert report["test_process_policy"] == ("deny_processes_and_listeners" if guarded else "suite_defined")
    assert test_command[test_command.index("--basetemp") + 1] == str(external / "test-tmp")
    if not Q.SUITES[suite].get("include_experiments", True):
        assert len(report["projects"]) == (2 if Q.SUITES[suite].get("include_analysis") else 1)
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


def _native_test_input_file(tmp_path):
    path = tmp_path / "native-inputs.json"
    keys = Q.SUITES["original-transpose-sources"]["test_input_environment_keys"]
    path.write_text(json.dumps({key: "explicit original input" for key in keys}))
    return path


def test_native_test_inputs_are_explicit_closed_and_reopened(tmp_path):
    path = _native_test_input_file(tmp_path)
    selected = Q.capture_native_test_inputs("original-transpose-sources", str(path))
    report = {"native_test_inputs": selected}
    assert Q.native_environment(report) == json.loads(path.read_bytes())
    Q.verify_native_inputs(report)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(Q.QualificationFailed, match="test inputs changed"):
        Q.verify_native_inputs(report)
    assert Q.capture_native_test_inputs("phase1", None) == {}


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate", "boolean", "null", "empty", "nul"])
def test_native_test_inputs_reject_ambiguous_or_undeclared_mappings(tmp_path, defect):
    path = _native_test_input_file(tmp_path)
    mapping = json.loads(path.read_bytes())
    key = next(iter(mapping))
    if defect == "missing":
        mapping.pop(key)
    elif defect == "extra":
        mapping["UNDECLARED_OVERRIDE"] = "unselected"
    elif defect == "duplicate":
        path.write_text(path.read_text()[:-1] + "," + json.dumps(key) + ':"repeated"}')
    else:
        mapping[key] = {"boolean": True, "null": None, "empty": "", "nul": "a\0b"}[defect]
    if defect != "duplicate":
        path.write_text(json.dumps(mapping))
    with pytest.raises(Q.QualificationFailed, match="complete closed suite mapping"):
        Q.capture_native_test_inputs("original-transpose-sources", str(path))


@pytest.mark.parametrize("defect", ["relative", "symlink", "oversized", "wrong_suite"])
def test_native_test_input_selection_requires_bounded_regular_admitted_file(tmp_path, defect):
    path = _native_test_input_file(tmp_path)
    suite = "original-transpose-sources"
    if defect == "relative":
        path = Path("unselected.json")
    elif defect == "symlink":
        link = tmp_path / "alias.json"
        link.symlink_to(path)
        path = link
    elif defect == "oversized":
        path.write_bytes(b" " * 65537)
    else:
        suite = "phase1"
    with pytest.raises(Q.QualificationFailed):
        Q.capture_native_test_inputs(suite, str(path))


def test_native_test_inputs_cannot_replace_tool_or_source_environment(tmp_path):
    path = _native_test_input_file(tmp_path)
    selected = Q.capture_native_test_inputs("original-transpose-sources", str(path))
    key = next(iter(selected["environment"]))
    with pytest.raises(Q.QualificationFailed, match="conflict"):
        Q.native_environment(
            {"native_test_inputs": selected, "native_tools": {"owned": {"environment_key": key, "path": "/tool"}}}
        )


def test_actual_child_mutation_is_rejected_after_qualification_command(tmp_path):
    path = _native_test_input_file(tmp_path)
    report = {
        "commands": [],
        "native_test_inputs": Q.capture_native_test_inputs("original-transpose-sources", str(path)),
    }
    recorder = Q.Recorder(tmp_path, report, 5)
    script = "from pathlib import Path; Path(" + repr(str(path)) + ").write_text('{}')"
    with pytest.raises(Q.QualificationFailed, match="test inputs"):
        recorder.run("owned-input-mutation", [sys.executable, "-I", "-B", "-c", script], tmp_path)
    assert report["commands"][-1]["status"] == "inputs_changed"


def test_native_test_input_cli_forwards_explicit_selection(monkeypatch, tmp_path):
    path = _native_test_input_file(tmp_path)
    seen = {}
    monkeypatch.setattr(Q, "resolve_ref", lambda *_: "a" * 40)
    monkeypatch.setattr(Q, "reserve_output", lambda *_: tmp_path)

    def qualify(*args, **kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(Q, "qualify", qualify)
    assert Q.main(["--ref", "HEAD", "--suite", "original-transpose-sources", "--test-inputs", str(path)]) == 0
    assert seen["test_inputs"] == str(path)
