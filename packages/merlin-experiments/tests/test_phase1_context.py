"""Explicit Phase-1 setup and the narrow native compatibility boundary."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import context

from merlin.common.paths import repo_root

_HARNESS = repo_root() / "merlin/experiments/capsule_bench/harness"


def _load_native(name):
    spec = importlib.util.spec_from_file_location(f"_phase1_test_{name}", _HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _descriptor(tmp_path, body="target: synthetic\n"):
    descriptor = tmp_path / "chosen" / "custom-name.yaml"
    descriptor.parent.mkdir()
    descriptor.write_text(body)
    return descriptor


def test_package_help_and_toolchain_import_do_not_initialize_invocation(tmp_path):
    descriptor = _descriptor(tmp_path)
    (descriptor.parent / "experiment.env").write_text("MERLIN_CONTEXT_IMPORT_SENTINEL=forbidden\n")
    script = """
import os,pathlib,subprocess,sys
sys.path.insert(0,sys.argv[1])
before=dict(os.environ)
def forbidden(*args,**kwargs):
    raise AssertionError('import/help launched a subprocess')
subprocess.run=forbidden
original_read=pathlib.Path.read_text
def read(path,*args,**kwargs):
    assert path.name!='experiment.env', 'import/help sourced experiment.env'
    return original_read(path,*args,**kwargs)
pathlib.Path.read_text=read
import merlin_experiments.phase1.context
import sandbox_toolchain
from merlin_experiments.cli import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code==0
assert '_common' not in sys.modules
assert 'merlin.targetgen.sandbox.toolchain' not in sys.modules
assert dict(os.environ)==before
"""
    env = dict(os.environ, MERLIN_TARGET_EXPERIMENT=str(descriptor))
    result = subprocess.run(
        [sys.executable, "-c", script, str(_HARNESS)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_explicit_context_normalizes_paths_and_preserves_process_precedence(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path)
    output = tmp_path / "chosen-output"
    (descriptor.parent / "experiment.env").write_text(
        f"# comment\nMERLIN_CONTEXT_EXISTING=file\nMERLIN_CONTEXT_EMPTY=file\n"
        f"MERLIN_CONTEXT_ADDED='first'\nMERLIN_CONTEXT_ADDED=second\ninvalid\n=ignored\n"
        f'MERLIN_OUT_ROOT="{output}"\nMERLIN_REPO_ROOT=wrong\nMERLIN_TARGET_EXPERIMENT=wrong\n'
    )
    monkeypatch.setattr(os, "environ", {"MERLIN_CONTEXT_EXISTING": "process", "MERLIN_CONTEXT_EMPTY": ""})
    monkeypatch.chdir(tmp_path)
    invocation = context.load_context("chosen/custom-name.yaml", repo="work")
    assert invocation.descriptor == descriptor
    assert invocation.repo == tmp_path / "work"
    assert invocation.target == "synthetic"
    assert invocation.experiment == descriptor.parent
    assert invocation.bundles == descriptor.parent / "input_bundles"
    assert invocation.harness is None
    assert invocation.runs == output / "runs" / "synthetic/capsule-bench"
    assert invocation.reports == output / "artifacts" / "capsule-bench/synthetic"
    assert invocation.sourced_environment == ("MERLIN_CONTEXT_ADDED", "MERLIN_OUT_ROOT")
    assert os.environ["MERLIN_CONTEXT_EXISTING"] == "process"
    assert os.environ["MERLIN_CONTEXT_EMPTY"] == ""
    assert os.environ["MERLIN_CONTEXT_ADDED"] == "first"
    assert os.environ["MERLIN_TARGET_EXPERIMENT"] == str(descriptor)
    assert os.environ["MERLIN_REPO_ROOT"] == str(tmp_path / "work")


def test_missing_descriptor_fails_before_sidecar_or_environment_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"MERLIN_TARGET_EXPERIMENT": "original"})
    (tmp_path / "experiment.env").write_text("MUST_NOT_SOURCE=1\n")
    with pytest.raises(FileNotFoundError, match="phase-1 descriptor"):
        context.load_context(tmp_path / "missing.yaml", repo=tmp_path)
    assert os.environ == {"MERLIN_TARGET_EXPERIMENT": "original"}


def test_parsed_context_uses_descriptor_resource_selection(tmp_path, monkeypatch):
    from merlin.targetgen.target_experiment import TargetExperiment

    descriptor = _descriptor(tmp_path)
    monkeypatch.setattr(os, "environ", {})
    selected = []
    original = TargetExperiment.resource_path

    def resource_path(self, relative):
        selected.append(relative)
        return original(self, relative)

    monkeypatch.setattr(TargetExperiment, "resource_path", resource_path)
    invocation = context.load_context(descriptor, repo=tmp_path)
    assert selected == [".", "input_bundles"]
    assert invocation.experiment == descriptor.parent
    assert invocation.bundles == descriptor.parent / "input_bundles"


def test_sidecar_is_loaded_before_descriptor_validation(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path, "not_a_target: true\n")
    (descriptor.parent / "experiment.env").write_text("MERLIN_CONTEXT_READY=yes\n")
    monkeypatch.setattr(os, "environ", {})
    with pytest.raises(ValueError, match="missing 'target'"):
        context.load_context(descriptor, repo=tmp_path)
    assert os.environ["MERLIN_CONTEXT_READY"] == "yes"
    assert os.environ["MERLIN_TARGET_EXPERIMENT"] == str(descriptor)


@pytest.mark.parametrize("absolute", [False, True])
def test_explicit_resource_root_owns_environment_tasks_and_bundles(tmp_path, monkeypatch, absolute):
    root = tmp_path / "repository"
    selected = tmp_path / "external" if absolute else root / "authored/phase1"
    selected.mkdir(parents=True)
    (selected / "task").mkdir()
    (selected / "input_bundles").mkdir()
    (selected / "experiment.env").write_text("MERLIN_SELECTED_RESOURCE=selected\n")
    descriptor = _descriptor(
        tmp_path,
        yaml.safe_dump(
            {
                "target": "synthetic",
                "resources_root": str(selected) if absolute else "authored/phase1",
            }
        ),
    )
    (descriptor.parent / "experiment.env").write_text("MERLIN_SELECTED_RESOURCE=decoy\nMERLIN_DECOY=forbidden\n")
    monkeypatch.setattr(os, "environ", {})
    invocation = context.load_context(descriptor, repo=root)
    assert invocation.experiment == selected
    assert invocation.bundles == selected / "input_bundles"
    context.require_scaffolding(invocation.experiment, invocation.target)
    assert invocation.sourced_environment == ("MERLIN_SELECTED_RESOURCE",)
    assert os.environ["MERLIN_SELECTED_RESOURCE"] == "selected"
    assert "MERLIN_DECOY" not in os.environ


def test_generated_prompt_startup_requires_bundles_but_not_authored_tasks(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path)
    monkeypatch.setattr(os, "environ", {})
    invocation = context.load_context(descriptor, repo=tmp_path)
    assert not (invocation.experiment / "task").exists()
    with pytest.raises(SystemExit, match="input_bundles"):
        context.require_scaffolding(invocation.experiment, invocation.target)
    invocation.bundles.mkdir()
    context.require_scaffolding(invocation.experiment, invocation.target)
    assert not (invocation.experiment / "task").exists()


def test_selected_sidecar_still_precedes_full_descriptor_validation(tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "experiment.env").write_text("MERLIN_CONTEXT_READY=yes\n")
    descriptor = _descriptor(tmp_path, yaml.safe_dump({"resources_root": str(selected)}))
    (descriptor.parent / "experiment.env").write_text("MERLIN_DECOY=forbidden\n")
    monkeypatch.setattr(os, "environ", {})
    with pytest.raises(ValueError, match="missing 'target'"):
        context.load_context(descriptor, repo=tmp_path)
    assert os.environ["MERLIN_CONTEXT_READY"] == "yes"
    assert "MERLIN_DECOY" not in os.environ


@pytest.mark.parametrize("body", ["broken: [\n", "- not\n- mapping\n", "null\n"])
def test_legacy_unparseable_descriptor_still_sources_sibling_environment(tmp_path, monkeypatch, body):
    from merlin.targetgen.corpora import source_experiment_env

    descriptor = _descriptor(tmp_path, body)
    (descriptor.parent / "experiment.env").write_text("MERLIN_CONTEXT_LEGACY=yes\n")
    monkeypatch.setattr(os, "environ", {})
    assert source_experiment_env(descriptor=descriptor) == ["MERLIN_CONTEXT_LEGACY"]
    assert os.environ["MERLIN_CONTEXT_LEGACY"] == "yes"


@pytest.mark.parametrize("value", [None, True, 1, "", " ", "../escape", "nested/../escape", "a\x00b"])
def test_invalid_resource_selection_never_sources_decoy_environment(tmp_path, monkeypatch, value):
    descriptor = _descriptor(tmp_path, yaml.safe_dump({"target": "synthetic", "resources_root": value}))
    (descriptor.parent / "experiment.env").write_text("MERLIN_DECOY=forbidden\n")
    monkeypatch.setattr(os, "environ", {})
    with pytest.raises(ValueError, match="resources_root"):
        context.load_context(descriptor, repo=tmp_path)
    assert "MERLIN_DECOY" not in os.environ


def test_native_fallback_and_patchable_globals_remain_at_the_edge(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path, "broken: [\n")
    (descriptor.parent / "experiment.env").write_text("MERLIN_CONTEXT_NATIVE=yes\n")
    monkeypatch.setattr(os, "environ", {"MERLIN_TARGET_EXPERIMENT": str(descriptor), "MERLIN_REPO_ROOT": str(tmp_path)})
    native = _load_native("_common")
    assert native.TARGET == "chosen"
    assert native.CONTEXT.descriptor == descriptor
    assert native.SOURCED_EXPERIMENT_ENV == ["MERLIN_CONTEXT_NATIVE"]
    assert native.EXP == descriptor.parent
    native.EXP = tmp_path / "patched"
    native.TARGET = "patched"
    with pytest.raises(SystemExit, match="target=patched"):
        native.require_scaffolding()
    native.BUNDLES = tmp_path / "patched-bundles"
    bundle = native.BUNDLES / "arm_hwbringup_custom"
    bundle.mkdir(parents=True)
    (bundle / "input_bundle_manifest.yaml").touch()
    assert native.experiment_conditions() == ["hwbringup_custom"]


@pytest.mark.parametrize("selection", ["relative", "absolute", "legacy"])
def test_native_globals_and_scaffolding_follow_selected_resource_root(tmp_path, monkeypatch, selection):
    document = {"target": "synthetic"}
    if selection != "legacy":
        document["resources_root"] = "authored" if selection == "relative" else str(tmp_path / "external")
    descriptor = _descriptor(tmp_path, yaml.safe_dump(document))
    selected = (
        descriptor.parent
        if selection == "legacy"
        else tmp_path / ("authored" if selection == "relative" else "external")
    )
    bundles = selected / "input_bundles"
    bundles.mkdir(parents=True)
    (selected / "experiment.env").write_text("MERLIN_NATIVE_SELECTED=yes\n")
    if selection != "legacy":
        (descriptor.parent / "experiment.env").write_text("MERLIN_NATIVE_DECOY=forbidden\n")
    monkeypatch.setattr(
        os,
        "environ",
        {
            "MERLIN_TARGET_EXPERIMENT": str(descriptor),
            "MERLIN_REPO_ROOT": str(tmp_path),
        },
    )
    native = _load_native("_common")
    assert native.EXP == native.CONTEXT.experiment == selected
    assert native.BUNDLES == native.CONTEXT.bundles == bundles
    assert native.CONTEXT.descriptor == descriptor
    assert native.SOURCED_EXPERIMENT_ENV == ["MERLIN_NATIVE_SELECTED"]
    assert "MERLIN_NATIVE_DECOY" not in os.environ
    native.require_scaffolding()
    bundles.rename(selected / "retained-bundles")
    if selection != "legacy":
        (descriptor.parent / "input_bundles").mkdir()
    with pytest.raises(SystemExit, match="input_bundles"):
        native.require_scaffolding()


def test_native_launcher_reads_selected_descriptor_not_resource_sibling(tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    resources.mkdir()
    descriptor = _descriptor(
        tmp_path,
        yaml.safe_dump(
            {
                "target": "synthetic",
                "resources_root": str(resources),
                "toolchain": {"sim_via": "selected-simulator"},
            }
        ),
    )
    (resources / "target_experiment.yaml").write_text(
        yaml.safe_dump({"target": "decoy", "toolchain": {"sim_via": "wrong-simulator"}})
    )
    monkeypatch.setattr(
        os,
        "environ",
        {
            "MERLIN_TARGET_EXPERIMENT": str(descriptor),
            "MERLIN_REPO_ROOT": str(tmp_path),
        },
    )
    native = _load_native("_common")
    monkeypatch.setitem(sys.modules, "_common", native)
    # Native modules adjust their import search path; keep that adjustment invocation-local.
    monkeypatch.setattr(sys, "path", list(sys.path))
    launcher = _load_native("launch_ab_batch")
    assert launcher._sim_via() == "selected-simulator"


@pytest.mark.parametrize("active", [True, False])
def test_readiness_uses_declared_target_and_bundle_ownership(tmp_path, monkeypatch, active):
    from merlin.targetgen import target_experiment

    descriptor = tmp_path / "examples" / "unrelated-folder-name" / "target" / "descriptor.yaml"
    descriptor.parent.mkdir(parents=True)
    resources = tmp_path / "authored-resources"
    descriptor.write_text(yaml.safe_dump({"target": "synthetic", "resources_root": str(resources)}))
    for arm in ("merlin_assisted", "merlin_assisted_rtlchecks"):
        bundle = resources / "input_bundles" / f"{arm}_hwbringup_v0"
        bundle.mkdir(parents=True)
        (bundle / "input_bundle_manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "allowed": [{"path": "shared-tool"}],
                }
            )
        )
    monkeypatch.setattr(target_experiment, "repo_root", lambda: tmp_path)
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT", raising=False)
    # No EXP: neither discovery nor resource selection may reconstruct a sibling path.
    monkeypatch.setitem(
        sys.modules,
        "_common",
        SimpleNamespace(
            TARGET="synthetic" if active else "another-target",
            DESCRIPTOR=descriptor if active else tmp_path / "other.yaml",
        ),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    readiness = _load_native("tooling_readiness")
    assert readiness._target_experiment("synthetic").path == descriptor
    assert readiness._arm_superset_check("synthetic")[0]["ok"]
    (resources / "input_bundles/merlin_assisted_rtlchecks_hwbringup_v0/input_bundle_manifest.yaml").write_text(
        "allowed: []\n"
    )
    check = readiness._arm_superset_check("synthetic")[0]
    assert not check["ok"]
    assert "shared-tool" in check["evidence"]


@pytest.mark.parametrize("present", [False, True])
def test_readiness_refuses_missing_or_mismatched_selected_descriptor(tmp_path, monkeypatch, present):
    descriptor = tmp_path / "selected.yaml"
    if present:
        descriptor.write_text("target: wrong-target\n")
    monkeypatch.setitem(sys.modules, "_common", SimpleNamespace(TARGET="synthetic", DESCRIPTOR=descriptor))
    monkeypatch.setattr(sys, "path", list(sys.path))
    readiness = _load_native("tooling_readiness")
    with pytest.raises(ValueError if present else FileNotFoundError):
        readiness._target_experiment("synthetic")
    assert not readiness._arm_superset_check("synthetic")[0]["ok"]


def test_rtl_readiness_reads_selected_corpus(tmp_path, monkeypatch):
    from merlin.targetgen import rtl_check_compiler, rtl_check_runner

    capsule = tmp_path / "selected-corpus" / "sample" / "capsule.yaml"
    capsule.parent.mkdir(parents=True)
    capsule.write_text("id: selected-sample\n")
    descriptor = _descriptor(
        tmp_path,
        yaml.safe_dump(
            {
                "target": "synthetic",
                "capsule_corpus": str(capsule.parent.parent),
            }
        ),
    )
    monkeypatch.setitem(sys.modules, "_common", SimpleNamespace(TARGET="synthetic", DESCRIPTOR=descriptor))
    monkeypatch.setattr(sys, "path", list(sys.path))
    readiness = _load_native("tooling_readiness")
    observed = []
    monkeypatch.setattr(rtl_check_runner, "find_filecheck", lambda: "synthetic-filecheck")
    monkeypatch.setattr(rtl_check_runner, "load_facts", lambda target: {"selected": target})

    def compile_checks(facts, selected, target):
        observed.append((facts, selected, target))
        return {"kernel": "synthetic-check"}

    monkeypatch.setattr(rtl_check_compiler, "compile_checks", compile_checks)
    checks = readiness._rtl_check_checks("synthetic")
    assert all(check["ok"] for check in checks)
    assert observed == [({"selected": "synthetic"}, {"id": "selected-sample"}, "synthetic")]


def test_explicit_toolchain_context_never_needs_native_common(tmp_path, monkeypatch):
    from merlin.targetgen import sandbox, target_experiment

    descriptor = _descriptor(tmp_path)
    invocation = SimpleNamespace(descriptor=descriptor)
    resolved = object()
    observed = []
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: observed.append(path) or resolved)
    toolchain = SimpleNamespace(
        toolchain_binds=lambda target: ["bind", target],
        sandbox_env=lambda target, ws: (target, ws),
        curated_harness_dir=lambda target: ("curated", target),
    )
    monkeypatch.setattr(sandbox, "toolchain", toolchain, raising=False)
    monkeypatch.delitem(sys.modules, "_common", raising=False)
    native = _load_native("sandbox_toolchain")
    assert native.toolchain_binds(context=invocation) == ["bind", resolved]
    assert native.sandbox_env(tmp_path, context=invocation) == (resolved, tmp_path)
    assert native.curated_harness(context=invocation) == ("curated", resolved)
    assert observed == [descriptor] * 3
    assert "_common" not in sys.modules


def test_legacy_toolchain_resource_is_lazy_and_uses_actual_descriptor(tmp_path, monkeypatch):
    from merlin.targetgen import sandbox, target_experiment

    descriptor = _descriptor(tmp_path)
    monkeypatch.setitem(sys.modules, "_common", SimpleNamespace(DESCRIPTOR=descriptor))
    observed = []
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: observed.append(path) or path)
    monkeypatch.setattr(
        sandbox, "toolchain", SimpleNamespace(curated_harness_dir=lambda target: str(target)), raising=False
    )
    native = _load_native("sandbox_toolchain")
    assert observed == []
    assert native.CURATED_HARNESS == str(descriptor)
    assert observed == [descriptor]
    assert native.CURATED_HARNESS == str(descriptor)
    assert observed == [descriptor]  # legacy constant retains one initialized value
