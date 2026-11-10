"""Qualify the committed Merlin software distribution outside its checkout.

Usage: python build_tools/packaging/qualify_installed.py --ref HEAD --suite compact-current

The closed current suite checks public compile/run services, package transport and explicit
provider composition using the installed wheel. Every selected member must run with zero
skips. Its Linux x86_64 process cases require a working system C compiler. Dependency
downloads may be needed; no simulator, hardware, paid agent or external service is launched.
These are software integration observations, not accelerator/numerical certification or
security isolation. Historical suites require the qualification script and inputs from the
same selected immutable historical Git revision; current suite names are never aliases.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

_INPUT_SPEC = importlib.util.spec_from_file_location(
    "installed_native_inputs", Path(__file__).with_name("installed_native_inputs.py")
)
_INPUTS = importlib.util.module_from_spec(_INPUT_SPEC)
_INPUT_SPEC.loader.exec_module(_INPUTS)

# Current software integration only. Every declared member must execute with zero skips.
# Linux x86_64 and a working system C compiler are required by the real process tests;
# these are software runtime checks, not accelerator or numerical qualification.
SUITES = {
    "compact-current": {
        "include_experiments": False,
        "tests_root": "tests/infra",
        "collect_selected_tests": True,
        "mandatory_test_report": "merlin.installed_mandatory_tests.v1",
        "native_test_files": ("test_public_services.py", "test_provider_contracts.py"),
        "native_test_cases": (
            ("test_public_services.py", "test_native_linux_arguments_affinity_streams_and_cli_status"),
            ("test_public_services.py", "test_native_linux_timeout_terminates_its_spawned_process_group"),
            ("test_public_services.py", "test_multi_product_compilation_and_cli_share_request_and_refuse_bad_options"),
            ("test_public_services.py", "test_external_runtime_contract_harts_and_no_compile"),
            (
                "test_public_services.py",
                "test_process_package_transport_retains_actual_invocations_and_opaque_products",
            ),
            ("test_provider_contracts.py", "test_missing_ambiguous_and_wrong_revision_bases_refuse"),
            ("test_provider_contracts.py", "test_backend_replacements_require_explicit_preserved_contract"),
            ("test_provider_contracts.py", "test_external_plugin_alias_refuses_before_import"),
        ),
        "tests": ("test_public_services.py", "test_provider_contracts.py"),
        "support_files": (),
        "source_inputs": (),
        "core_extras": ("semantic-search",),
        "probe_modules": ("merlin.api", "merlin.compiler.backend.composition", "merlin.runtime.service"),
        "required_modules": ("numpy", "yaml", "xdsl", "jsonschema", "z3"),
        "guarded_tests": False,
    }
}
LIMITATIONS = [
    "Tool source is assumed static while Python initially imports it; observed hashes are rechecked at completion.",
    "Packaging and selected functional regressions only; no numerical/hardware certification.",
    "External venv and import-origin checks are not a security isolation guarantee.",
    "Dependency resolution uses configured indexes; exact resulting freeze is retained, not a lockfile replay.",
    "Unrecorded dependency overrides are removed from child environments; "
    "a local diagnostic override is not qualification.",
    "External venv is deliberately retained on success or failure; this command never deletes evidence.",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retain_archived_test_sources(tests, names, output):
    """Pin exact copied test identities; production origin checks stay strict."""
    rows = []
    for name in dict.fromkeys(names):
        relative = Path(name.split("::", 1)[0])
        if relative.suffix != ".py" or not relative.stem.startswith("test_"):
            continue
        path = (tests / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(tests.resolve()):
            raise QualificationFailed("archived test identity escapes its selected root")
        rows.append({"module": ".".join(relative.with_suffix("").parts), "path": str(path), "sha256": digest(path)})
    path = output / "archived-test-sources.json"
    path.write_text(
        json.dumps({"schema": "merlin.installed_test_sources.v1", "files": rows}, sort_keys=True, indent=2) + "\n"
    )
    return path


def clean_environment():
    excluded = {"UV_OVERRIDE", "UV_EXCLUDE", "UV_CONSTRAINT", "UV_BUILD_CONSTRAINT"}
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MERLIN", "PYTHON", "AET_", "CHIA_")) and k not in excluded
    }


NATIVE_TOOL_ENVIRONMENT = {
    "bwrap": "MERLIN_TEST_BWRAP",
    "circt-opt": "MERLIN_TEST_CIRCT_OPT",
    "circt-opt-modern": "MERLIN_TEST_CIRCT_OPT_MODERN",
    "clang": "MERLIN_CLANG",
    "compiler-python": "MERLIN_COMPILER_PYTHON",
    "cpu-simulator": "MERLIN_TEST_STOCK_CPU_SIMULATOR",
    "firtool": "MERLIN_TEST_FIRTOOL",
    "firtool-modern": "MERLIN_TEST_FIRTOOL_MODERN",
    "iverilog": "MERLIN_TEST_IVERILOG",
    "mlir-opt": "MERLIN_TEST_MLIR_OPT",
    "mlir-translate": "MERLIN_MLIR_TRANSLATE",
    "llvm-llc": "MERLIN_LLVM_LLC",
    "operator-python": "MERLIN_TEST_TORCH_PYTHON",
    "readelf": "MERLIN_TEST_READELF",
    "riscv-gcc": "MERLIN_TEST_RISCV_GCC",
    "tensor-argument-compiler": "MERLIN_TEST_TENSOR_ARGUMENT_COMPILER",
    "vvp": "MERLIN_TEST_VVP",
}


def capture_native_tools(suite, selections):
    """Explicit executable selections, never ambient provider/env overrides."""
    selected = {}
    admitted = SUITES[suite].get("native_tools", ())
    for selection in selections:
        name, separator, supplied = selection.partition("=")
        if not separator or name not in admitted or name in selected:
            raise QualificationFailed("unknown, duplicate or suite-inadmissible native tool")
        path = Path(supplied)
        if not path.is_absolute():
            raise QualificationFailed("native tool must have an explicit absolute path")
        actual = path.resolve(strict=True)
        if not actual.is_file() or not os.access(actual, os.X_OK):
            raise QualificationFailed("native tool must resolve to an executable file")
        selected[name] = {
            "selected_path": str(path),
            "path": str(actual),
            "sha256": digest(actual),
            "environment_key": NATIVE_TOOL_ENVIRONMENT[name],
        }
        if name in SUITES[suite].get("native_python_entries", ()):
            try:
                selected[name]["python_entry"] = _INPUTS.python_entry(path)
            except (OSError, ValueError) as exc:
                raise QualificationFailed(f"interpreter entry unavailable: {exc}") from exc
    if selected and set(selected) != set(admitted):
        raise QualificationFailed("selected native suite needs its complete explicit tool roster")
    return selected


def verify_native_tools(selected):
    try:
        for tool in selected.values():
            if digest(tool["path"]) != tool["sha256"]:
                raise QualificationFailed("selected native executable changed")
            if "python_entry" in tool and _INPUTS.python_entry(tool["selected_path"]) != tool["python_entry"]:
                raise QualificationFailed("selected interpreter entry or prefix changed")
    except (OSError, ValueError) as exc:
        raise QualificationFailed(f"selected native executable unavailable: {exc}") from exc


def capture_native_sources(suite, selections):
    """Closed caller-selected public source identities, not dependency authority."""
    admitted, selected = SUITES[suite].get("native_sources", {}), {}
    for selection in selections:
        name, separator, supplied = selection.partition("=")
        if not separator or name not in admitted or name in selected:
            raise QualificationFailed("unknown, duplicate or suite-inadmissible native source")
        path, separator, commit = supplied.rpartition("@")
        if not separator:
            raise QualificationFailed("native source needs an explicit path and full commit")
        try:
            identity = _INPUTS.source_record(path, commit, admitted[name]["package"], check_committed=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            raise QualificationFailed(f"native source unavailable: {exc}") from exc
        selected[name] = {"identity": identity, "environment_key": admitted[name]["environment_key"]}
    if selected and set(selected) != set(admitted):
        raise QualificationFailed("selected native suite needs its complete explicit source roster")
    return selected


def verify_native_inputs(report):
    verify_native_tools(report.get("native_tools", {}))
    selected_environment = report.get("native_test_environment")
    if selected_environment is not None:
        try:
            if digest(selected_environment["path"]) != selected_environment["sha256"]:
                raise QualificationFailed("selected native test environment changed")
        except OSError as exc:
            raise QualificationFailed("selected native test environment unavailable") from exc
    try:
        _INPUTS.verify_sources(report.get("native_sources", {}))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise QualificationFailed(f"selected native source changed or unavailable: {exc}") from exc
    selected_inputs = report.get("native_test_inputs")
    if selected_inputs:
        try:
            current = _INPUTS.test_inputs_record(selected_inputs["path"], selected_inputs["environment"])
        except (OSError, ValueError) as exc:
            raise QualificationFailed("selected native test inputs unavailable") from exc
        if current != selected_inputs:
            raise QualificationFailed("selected native test inputs changed")


def capture_native_test_inputs(suite, path):
    if path is None:
        return {}
    names = SUITES[suite].get("test_input_environment_keys", ())
    if not names or len(names) != len(set(names)):
        raise QualificationFailed("suite does not admit the selected native test inputs")
    try:
        return _INPUTS.test_inputs_record(path, names)
    except (OSError, ValueError) as exc:
        raise QualificationFailed("native test inputs require their complete closed suite mapping") from exc


def native_environment(report, *, suite=None):
    environment = {
        tool["environment_key"]: tool["selected_path"] if "python_entry" in tool else tool["path"]
        for tool in report.get("native_tools", {}).values()
    }
    environment.update(
        (source["environment_key"], source["identity"]["path"]) for source in report.get("native_sources", {}).values()
    )
    selected_inputs = report.get("native_test_inputs", {}).get("environment", {})
    if set(selected_inputs) & set(environment):
        raise QualificationFailed("native test inputs conflict with selected tools or sources")
    environment.update(selected_inputs)
    if suite is not None and report.get("native_tools"):
        selected = report["native_tools"]
        for key, names in SUITES[suite].get("native_tool_path_lists", {}).items():
            if key in environment or any(name not in selected for name in names):
                raise QualificationFailed("declared native tool list lacks its complete selections")
            environment[key] = json.dumps([selected[name]["path"] for name in names])
    return environment


def retain_test_environment(suite, output, environment, report):
    """Retain a declared child mapping for selected native replay, never recover one."""
    key = SUITES[suite].get("test_environment_record")
    if key is None:
        return dict(environment)
    if key in environment or "native_test_environment" in report:
        raise QualificationFailed("native test environment was already selected")
    selected = dict(environment)
    for name, value in SUITES[suite].get("test_environment_defaults", {}).items():
        if name in selected and selected[name] != value:
            raise QualificationFailed("native test environment conflicts with its declared default")
        selected[name] = value
    if any(type(name) is not str or type(value) is not str for name, value in selected.items()):
        raise QualificationFailed("native test environment requires declared string values")
    path = Path(output).resolve(strict=True) / "native-test-environment.json"
    selected[key] = str(path)
    # The caller owns this private output directory. The file contains actual
    # values, stays outside candidate inputs, and must not inherit broad modes.
    with path.open("xb") as stream:
        path.chmod(0o600)
        stream.write((json.dumps(selected, sort_keys=True) + "\n").encode())
    report["native_test_environment"] = {"path": str(path), "sha256": digest(path), "environment_key": key}
    return selected


def native_test_report_required(suite, report):
    """A declared mandatory packaging roster is independent of tool selection."""
    policy = SUITES[suite].get("mandatory_test_report")
    if policy is not None and policy != "merlin.installed_mandatory_tests.v1":
        raise QualificationFailed("unknown mandatory test report policy")
    return policy is not None or bool(report["native_tools"])


def check_native_test_report(suite, path, report):
    """Require every declared native file to execute; distinguish other skips.

    The actual pytest child supplies xunit1 file identities relative to the
    explicit archived test root. This is packaging evidence, not native tool or
    compiler qualification. The retained XML preserves complete skip messages.
    """
    configured = SUITES[suite]
    required = configured.get("native_test_files", ())
    admitted = (*configured["tests"], *configured.get("support_files", ()))
    if not required or len(set(required)) != len(required) or not set(required) <= set(configured["tests"]):
        raise QualificationFailed("native suite has no closed declared test subset")
    required_cases = configured.get("native_test_cases", ())
    if required_cases and (
        len(set(required_cases)) != len(required_cases)
        or {filename for filename, _ in required_cases} != set(required)
        or any(not name for _, name in required_cases)
    ):
        raise QualificationFailed("native suite has no closed declared member roster")
    cases = list(ET.parse(path).getroot().iter("testcase"))
    report["native_test_report"] = {"path": str(path), "sha256": digest(path)}
    report["native_test_files"] = list(required)
    report["native_zero_skip_scope"] = "declared_native_test_files"
    report["suite_test_counts"] = {"tests": len(cases), "skipped": 0}
    report["native_test_counts"] = {"tests": 0, "skipped": 0}
    report["other_test_counts"] = {"tests": 0, "skipped": 0}
    report["test_skips"] = {"native": [], "other": []}
    modules = {filename: Path(filename).with_suffix("").as_posix().replace("/", ".") for filename in admitted}
    observed, identities, observed_cases = set(), set(), set()
    errors = []
    for case in cases:
        name, filename, classname = (case.get(key, "") for key in ("name", "file", "classname"))
        module = modules.get(filename)
        identity = (filename, classname, name)
        if (
            module is None
            or not name
            or identity in identities
            or not (classname == module or classname.startswith(module + "."))
        ):
            errors.append("testcase lacks a unique admitted archived file identity")
        identities.add(identity)
        category = "native" if filename in required else "other"
        report[category + "_test_counts"]["tests"] += 1
        if category == "native":
            observed.add(filename)
            if required_cases and (classname != module or (filename, name) in observed_cases):
                errors.append("testcase changed or repeated a declared top-level member identity")
            observed_cases.add((filename, name))
        skipped = case.find("skipped")
        if skipped is not None:
            report["suite_test_counts"]["skipped"] += 1
            report[category + "_test_counts"]["skipped"] += 1
            report["test_skips"][category].append(
                {
                    "file": filename,
                    "classname": classname,
                    "name": name,
                    "type": skipped.get("type", ""),
                    "message": skipped.get("message", ""),
                }
            )
        if case.find("failure") is not None or case.find("error") is not None:
            errors.append("testcase has a failure or error")
    report["missing_native_test_files"] = sorted(set(required) - observed)
    if required_cases:
        report["native_test_cases"] = [list(member) for member in required_cases]
        report["missing_native_test_cases"] = [list(member) for member in sorted(set(required_cases) - observed_cases)]
        report["unexpected_native_test_cases"] = [
            list(member) for member in sorted(observed_cases - set(required_cases))
        ]
    if errors:
        raise QualificationFailed(errors[0])
    if report["missing_native_test_files"]:
        raise QualificationFailed("explicit native qualification did not execute every declared native test file")
    if required_cases and (report["missing_native_test_cases"] or report["unexpected_native_test_cases"]):
        raise QualificationFailed("explicit native qualification did not execute the exact original member roster")
    if report["native_test_counts"]["skipped"]:
        raise QualificationFailed("explicit native qualification requires declared native tests with zero skips")


def resolve_ref(root, ref):
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=root,
        text=True,
        timeout=30,
    ).strip()


def reserve_output(base, label):
    if (
        not label
        or len(label) > 120
        or not label.isascii()
        or not label[0].isalnum()
        or any(not (character.isalnum() or character in "_.-") for character in label)
    ):
        raise ValueError("output label must be one safe path component")
    base = Path(base)
    if any(p.is_symlink() for p in (base, *base.parents)):
        raise ValueError("output ancestors may not be symlinks")
    base.mkdir(parents=True, exist_ok=True)
    output = base / label
    output.mkdir()  # Existing even-empty directories and symlinks are refused.
    return output


class QualificationFailed(RuntimeError):
    pass


class Recorder:
    def __init__(self, output, report, timeout):
        self.output, self.report, self.timeout = output, report, timeout
        self.environment = clean_environment()
        self.environment.update(native_environment(report, suite=report.get("suite")))

    def save(self):
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def run(self, label, argv, cwd, *, stdout=None):
        verify_native_inputs(self.report)
        argv = list(map(str, argv))
        log = self.output / (label + ".log")
        record = {
            "step": label,
            "argv": argv,
            "cwd": str(cwd),
            "timeout_s": self.timeout,
            "log": str(log),
            "returncode": None,
            "status": "running",
        }
        self.report["commands"].append(record)
        self.save()
        start = time.monotonic()
        try:
            with log.open("wb") as errors:
                stream = Path(stdout).open("wb") if stdout is not None else errors
                try:
                    with subprocess.Popen(
                        argv, cwd=cwd, env=self.environment, stdout=stream, stderr=errors, start_new_session=True
                    ) as child:
                        try:
                            record["returncode"] = child.wait(timeout=self.timeout)
                            record["status"] = "passed" if child.returncode == 0 else "failed"
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            record.update(status="timeout", returncode=child.returncode)
                        except KeyboardInterrupt:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            record.update(status="interrupted", returncode=child.returncode)
                            raise
                finally:
                    if stream is not errors:
                        stream.close()
        except OSError as exc:
            record.update(status="launch_failed", error=str(exc))
        finally:
            record["elapsed_s"] = time.monotonic() - start
            self.save()
        if (
            self.report.get("native_sources")
            or self.report.get("native_test_inputs")
            or any("python_entry" in tool for tool in self.report.get("native_tools", {}).values())
        ):
            try:
                verify_native_inputs(self.report)
            except QualificationFailed as exc:
                record.update(status="inputs_changed", error=str(exc))
                self.save()
                raise
        if record["status"] != "passed":
            raise QualificationFailed(f"{label}: {record['status']}; see {log}")


def projects(snapshot, extras, *, include_experiments=True, include_analysis=False):
    """Select dependency extras for one shipping distribution, never extra wheels."""
    directory = snapshot
    project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
    selected = tuple(
        dict.fromkeys(
            (*extras, *(("experiments",) if include_experiments else ()), *(("analysis",) if include_analysis else ()))
        )
    )
    for extra in selected:
        if extra not in project.get("optional-dependencies", {}):
            raise QualificationFailed(f"{project['name']} does not declare extra {extra}")
    return [{"name": project["name"], "version": project["version"], "path": str(directory), "extras": list(selected)}]


def only_artifact(directory, pattern):
    files = list(directory.glob(pattern))
    if len(files) != 1:
        raise QualificationFailed(f"expected one {pattern} in {directory}; got {files}")
    return files[0]


def source_input_archive_roots(patterns):
    """Archive literal owners; expand patterns only inside the selected commit."""
    roots = []
    for pattern in patterns:
        member = Path(pattern)
        if member.is_absolute() or ".." in member.parts or not member.parts:
            raise QualificationFailed(f"unsafe source input pattern: {pattern}")
        literal = []
        for part in member.parts:
            if any(char in part for char in "*?["):
                break
            literal.append(part)
        if not literal:
            raise QualificationFailed(f"source input pattern has no literal owner: {pattern}")
        roots.append(Path(*literal).as_posix())
    return tuple(dict.fromkeys(roots))


def selected_source_inputs(snapshot, patterns):
    source_input_archive_roots(patterns)
    names = set()
    for pattern in patterns:
        matched = sorted(snapshot.glob(pattern))
        if not matched:
            raise QualificationFailed(f"selected committed source input is missing: {pattern}")
        for source_path in matched:
            member = source_path.relative_to(snapshot)
            symlinked = any(
                snapshot.joinpath(*member.parts[:index]).is_symlink() for index in range(1, len(member.parts) + 1)
            )
            if not source_path.is_file() or symlinked or not source_path.resolve().is_relative_to(snapshot.resolve()):
                raise QualificationFailed(f"selected committed source input is unsafe: {member}")
            names.add(member.as_posix())
    return tuple(sorted(names))


def qualify(
    root,
    output,
    commit,
    suite,
    timeout,
    *,
    requested_ref=None,
    invocation=None,
    native_tools=(),
    native_sources=(),
    test_inputs=None,
):
    own = Path(__file__).resolve()
    helper = own.with_name("installed_qualification_probe.py")
    input_helper = own.with_name("installed_native_inputs.py")
    tests_root = Path(SUITES[suite].get("tests_root", "packages/merlin-experiments/tests"))
    support_files = SUITES[suite].get("support_files", ())
    source_inputs = SUITES[suite].get("source_inputs", ())
    test_files = (*SUITES[suite]["tests"], *support_files)
    report = {
        "schema": "merlin.installed_qualification.v1",
        "ref": commit,
        "requested_ref": requested_ref,
        "invocation": invocation,
        "suite": suite,
        "status": "running",
        "commands": [],
        "limitations": LIMITATIONS,
        "tooling_revision": resolve_ref(root, "HEAD"),
        "tool_sources": {str(p.relative_to(root)): digest(p) for p in (own, helper, input_helper)},
        "tool_source_note": "Actual executing bytes; hashes may include working-tree edits not in tooling_revision.",
        "selected_tests": list(SUITES[suite]["tests"]),
        "support_files": list(support_files),
        "test_fixture_imports": bool(SUITES[suite].get("test_fixture_imports")),
        "test_collection_policy": "selected_roster" if SUITES[suite].get("collect_selected_tests") else "copied_tree",
        "source_inputs": {},
        "tests_root": tests_root.as_posix(),
        "core_extras": list(SUITES[suite]["core_extras"]),
        "workflow_dependencies": {
            "experiments": SUITES[suite].get("include_experiments", True),
            "analysis": SUITES[suite].get("include_analysis", False),
        },
        "shipping_policy": "one main distribution; workflow flags select dependencies and probes only",
        "probe_modules": list(SUITES[suite]["probe_modules"]),
        "required_modules": list(SUITES[suite]["required_modules"]),
        "required_entry_points": list(SUITES[suite].get("required_entry_points", ())),
        "test_process_policy": (
            "deny_processes_and_listeners" if SUITES[suite].get("guarded_tests") else "suite_defined"
        ),
        "native_tools": {},
        "requested_native_tools": list(native_tools),
        "native_sources": {},
        "requested_native_sources": list(native_sources),
        "native_test_inputs": {},
        "native_test_input_policy": "Explicit closed suite input strings; no source, dependency or runtime authority.",
        "native_source_policy": (
            "Explicit package bytes and checkout identity only; import/dependency closure unproved."
        ),
        "native_test_files": list(SUITES[suite].get("native_test_files", ())),
        "native_tool_policy": "Explicit executable bytes only; system dependencies are not a frozen toolchain closure.",
        "test_fixture_retention": "all; unique qualification-owned external temp root",
    }
    runner = Recorder(output, report, timeout)
    runner.save()
    try:
        report["native_tools"] = capture_native_tools(suite, native_tools)
        report["native_sources"] = capture_native_sources(suite, native_sources)
        report["native_test_inputs"] = capture_native_test_inputs(suite, test_inputs)
        runner.environment.update(native_environment(report, suite=suite))
        report["native_environment"] = native_environment(report, suite=suite)
        runner.save()
        copied_helper = output / "installed_qualification_probe.py"
        shutil.copyfile(helper, copied_helper)
        if digest(copied_helper) != report["tool_sources"][str(helper.relative_to(root))]:
            raise QualificationFailed("qualification helper changed during capture")
        runner.run(
            "resource-manifest",
            ["git", "show", commit + ":build_tools/package_resources.json"],
            root,
            stdout=output / "resources.json",
        )
        resources = json.loads((output / "resources.json").read_text())["files"]
        selected = [
            "src",
            "pyproject.toml",
            "setup.py",
            "README.md",
            "LICENSE",
            "MANIFEST.in",
            "build_tools/package_resources.json",
            "build_tools/checks/check_distribution_layout.py",
            *[(tests_root / n).as_posix() for n in test_files],
            *source_input_archive_roots(source_inputs),
            *resources,
        ]
        archive = output / "source.tar"
        runner.run("source-archive", ["git", "archive", "--format=tar", commit, "--", *selected], root, stdout=archive)
        snapshot = output / "snapshot"
        snapshot.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(snapshot, filter="data")
        for name in test_files:
            if not (snapshot / tests_root / name).is_file():
                raise QualificationFailed(f"selected committed test/support file is missing: {name}")
        source_inputs = selected_source_inputs(snapshot, source_inputs)
        report["source_archive_sha256"] = digest(archive)
        report["source_files"] = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob("*") if p.is_file()}
        report["projects"] = projects(
            snapshot,
            SUITES[suite]["core_extras"],
            include_experiments=SUITES[suite].get("include_experiments", True),
            include_analysis=SUITES[suite].get("include_analysis", False),
        )
        runner.save()
        wheels, install = [], []
        for project in report["projects"]:
            directory = output / "dist" / project["name"]
            directory.mkdir(parents=True)
            runner.run(
                project["name"] + "-sdist", ["uv", "build", "--sdist", "--out-dir", directory, project["path"]], output
            )
            sdist = only_artifact(directory, "*.tar.gz")
            runner.run(project["name"] + "-wheel", ["uv", "build", "--wheel", "--out-dir", directory, sdist], output)
            wheel = only_artifact(directory, "*.whl")
            wheels.append(wheel)
            install.append(str(wheel) + ("[" + ",".join(project["extras"]) + "]" if project["extras"] else ""))
            report.setdefault("artifacts", {}).update({str(p.relative_to(output)): digest(p) for p in (sdist, wheel)})
            runner.save()
        runner.run(
            "layout",
            [
                sys.executable,
                snapshot / "build_tools/checks/check_distribution_layout.py",
                "--source-root",
                snapshot,
                *wheels,
            ],
            output,
        )
        external = Path(tempfile.mkdtemp(prefix="merlin-qualified-venv-"))
        report["external_venv"] = str(external)
        runner.save()
        if external.resolve().is_relative_to(root.resolve()):
            raise QualificationFailed("temporary venv must be outside the checkout; configure TMPDIR")
        runner.run("venv", ["uv", "venv", "--python", sys.executable, external], output)
        python = external / "bin/python"
        runner.run("install", ["uv", "pip", "install", "--python", python, *install], external)
        runner.run("dependencies", ["uv", "pip", "check", "--python", python], external)
        probe_args = [python, "-I", copied_helper]
        for module in SUITES[suite]["probe_modules"]:
            probe_args.extend(("--module", module))
        for module in SUITES[suite]["required_modules"]:
            probe_args.extend(("--require-module", module))
        for entry_point in SUITES[suite].get("required_entry_points", ()):
            probe_args.extend(("--require-entry-point", entry_point))
        runner.run("payload-probe", [*probe_args, *wheels], external)
        runner.run("pytest-install", ["uv", "pip", "install", "--python", python, "pytest"], external)
        runner.run("freeze", ["uv", "pip", "freeze", "--python", python], external, stdout=output / "dependencies.txt")
        report["dependency_freeze_sha256"] = digest(output / "dependencies.txt")
        tests = external / "qualification-tests"
        tests.mkdir()
        for name in test_files:
            (tests / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / tests_root / name, tests / name)
        for name in source_inputs:
            retained = tests / "source-inputs" / name
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / name, retained)
            source_sha256 = digest(snapshot / name)
            if digest(retained) != source_sha256:
                raise QualificationFailed(f"copied source input differs from committed archive: {name}")
            report["source_inputs"][name] = {"path": str(retained), "sha256": source_sha256}
        if source_inputs:
            input_root = str(tests / "source-inputs")
            runner.environment["MERLIN_TEST_SOURCE_INPUTS_ROOT"] = input_root
            report["source_input_root"] = input_root
        runner.save()
        shutil.copyfile(copied_helper, tests / "conftest.py")
        archived_tests = retain_archived_test_sources(tests, test_files, output)
        runner.environment["MERLIN_TEST_ARCHIVED_SOURCES"] = str(archived_tests)
        report["archived_test_sources"] = {"path": str(archived_tests), "sha256": digest(archived_tests)}
        runner.environment = retain_test_environment(suite, output, runner.environment, report)
        runner.save()
        runner.run(
            "tests",
            [
                python,
                "-I",
                *([copied_helper, "--guarded-tests"] if SUITES[suite].get("guarded_tests") else ["-m", "pytest"]),
                "-q",
                "-c",
                "/dev/null",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",
                "-o",
                "tmp_path_retention_policy=all",
                *(
                    [
                        "-o",
                        "pythonpath="
                        + " ".join(dict.fromkeys((str(tests), *(str((tests / name).parent) for name in test_files)))),
                    ]
                    if SUITES[suite].get("test_fixture_imports")
                    else []
                ),
                # This venv is fresh and unique; pytest must not clean shared user temp roots.
                "--basetemp",
                external / "test-tmp",
                *(
                    [
                        "--rootdir",
                        tests,
                        "-o",
                        "junit_family=xunit1",
                        "--junitxml",
                        output / "tests.xml",
                    ]
                    if native_test_report_required(suite, report)
                    else []
                ),
                *(
                    [tests / name for name in SUITES[suite]["tests"]]
                    if SUITES[suite].get("collect_selected_tests")
                    else [tests]
                ),
            ],
            external,
        )
        verify_native_inputs(report)
        if native_test_report_required(suite, report):
            check_native_test_report(suite, output / "tests.xml", report)
        if any(digest(root / name) != expected for name, expected in report["tool_sources"].items()):
            raise QualificationFailed("qualification tooling changed during execution")
        report["status"] = "passed"
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        runner.save()
    return report["status"] == "passed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Git commit/ref to archive; never the dirty worktree")
    parser.add_argument("--suite", default="compact-current", help="Current software suite: compact-current")
    parser.add_argument("--label", help="New output directory name beneath build_dir()/python/qualified-installs")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds per child command")
    parser.add_argument(
        "--native-tool",
        action="append",
        default=[],
        metavar="NAME=ABSOLUTE_PATH",
        help="Explicit suite-admitted executable; pin bytes and require zero skips in declared native test files",
    )
    parser.add_argument(
        "--native-source",
        action="append",
        default=[],
        metavar="NAME=ABSOLUTE_CHECKOUT@FULL_COMMIT",
        help="Explicit suite-admitted package checkout; pin tracked bytes and reject untracked package files",
    )
    parser.add_argument(
        "--test-inputs",
        metavar="ABSOLUTE_JSON_FILE",
        help="Explicit closed suite-declared native test input strings; pin and recheck before/after commands",
    )
    args = parser.parse_args(argv)
    if args.suite not in SUITES:
        parser.error(
            "retired or unknown suite; use compact-current for this revision, or run the qualification "
            "script and its declared inputs from the same immutable historical Git ref"
        )
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    from merlin.common.paths import build_dir, repo_root

    root = repo_root()
    commit = resolve_ref(root, args.ref)
    label = args.label or (
        datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + commit[:12] + "-" + uuid.uuid4().hex[:8]
    )
    output = reserve_output(build_dir() / "python/qualified-installs", label)
    print(f"Qualification evidence: {output}", flush=True)
    return (
        0
        if qualify(
            root,
            output,
            commit,
            args.suite,
            args.timeout,
            requested_ref=args.ref,
            invocation=[sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
            native_tools=args.native_tool,
            native_sources=args.native_source,
            test_inputs=args.test_inputs,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
