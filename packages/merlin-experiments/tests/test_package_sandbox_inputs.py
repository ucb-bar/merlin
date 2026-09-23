"""Explicit package sandbox selections survive ambient resolver changes without launching tools."""

import importlib
import shlex
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen.sandbox import toolchain as TC


def _explicit_selection(tmp_path):
    paths = TC.ToolchainPaths(
        tmp_path / "checkout",
        str(tmp_path / "venv"),
        str(tmp_path / "llvm"),
        str(tmp_path / "compat"),
        str(tmp_path / "clang"),
        str(tmp_path / "uv"),
    )
    sim_dir = str(tmp_path / "sim")
    sim = TC.SimToolchain(
        bind_paths=(sim_dir,),
        path_dirs=(sim_dir + "/bin",),
        ld_dirs=(sim_dir + "/lib",),
        env_extra={"SELECTED_SIM": "explicit"},
        probes=(TC.ToolProbe("selected-sim", "selected-sim --version", sim_dir),),
    )
    for path in (
        paths.venv,
        paths.llvm,
        paths.compat_lib,
        paths.clang_bin,
        paths.clang_resource,
        paths.uv_python,
        sim_dir,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
    return paths, sim


def _forbid_discovery(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit sandbox selection rediscovered ambient configuration")

    for name in ("repo_root", "compat_lib_dir", "env", "ext_path", "_sim", "curated_harness_dir"):
        monkeypatch.setattr(TC, name, forbidden)
    monkeypatch.setattr(TC.ToolchainPaths, "from_checkout", forbidden)
    return forbidden


@pytest.mark.parametrize("roots", [(), ("/installed/site-packages",), ("/installed/a space", "/installed/a'quote")])
def test_explicit_python_roots_replace_ambient_path_without_granting_mounts(tmp_path, monkeypatch, roots):
    paths, sim = _explicit_selection(tmp_path)
    selected = replace(paths, python_import_roots=roots)
    _forbid_discovery(monkeypatch)
    monkeypatch.setenv("PYTHONPATH", "/ambient/private-modules")
    target = SimpleNamespace(target="synthetic")
    exports = TC.sandbox_env(target, tmp_path, paths=selected, sim=sim, harness="")
    assignment = exports.split("export PYTHONPATH=", 1)[1].split(";", 1)[0]
    assert shlex.split("PYTHONPATH=" + assignment) == ["PYTHONPATH=" + ":".join(roots)]
    assert "$PYTHONPATH" not in exports
    assert str(paths.repo / "merlin/python") not in exports
    assert "/ambient/private-modules" not in exports
    # Import search configuration does not grant access to entire package trees.
    assert TC.toolchain_binds(target, paths=selected, sim=sim, harness="", memory_dir="") == TC.toolchain_binds(
        target, paths=paths, sim=sim, harness="", memory_dir=""
    )


@pytest.mark.parametrize("root", ["", "relative", "/a:/b", "/a\x00b"])
def test_explicit_python_roots_reject_ambiguous_search_paths(tmp_path, root):
    paths, _ = _explicit_selection(tmp_path)
    with pytest.raises(ValueError):
        replace(paths, python_import_roots=(root,))


def test_omitted_python_roots_preserve_legacy_environment(tmp_path, monkeypatch):
    paths, sim = _explicit_selection(tmp_path)
    _forbid_discovery(monkeypatch)
    exports = TC.sandbox_env(SimpleNamespace(target="synthetic"), tmp_path, paths=paths, sim=sim, harness="")
    assert f"export PYTHONPATH={paths.repo}/merlin/python${{PYTHONPATH:+:$PYTHONPATH}}; " in exports


@pytest.mark.parametrize("has_harness", [False, True])
def test_explicit_sim_and_harness_bypass_descriptor_discovery(tmp_path, monkeypatch, has_harness):
    paths, sim = _explicit_selection(tmp_path)
    harness = str(tmp_path / "harness") if has_harness else ""
    if harness:
        Path(harness).mkdir()
    target = SimpleNamespace(target="synthetic")
    forbidden = _forbid_discovery(monkeypatch)
    answer_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    monkeypatch.setattr(answer_module, "experimenter_memory_dir", forbidden)
    selection = dict(paths=paths, sim=sim, harness=harness)
    argv = TC.toolchain_binds(target, **selection, memory_dir="")
    exports = TC.sandbox_env(target, tmp_path, **selection)
    probes = TC.required_tool_probes(target, paths=paths, sim=sim)
    assert sim.probes[0] in probes
    assert sim.bind_paths[0] in argv
    assert sim.path_dirs[0] in exports
    assert sim.ld_dirs[0] in exports
    assert "export SELECTED_SIM=explicit;" in exports
    assert ("MERLIN_HWBRINGUP_HARNESS_DIR=" in exports) is has_harness
    assert (str(tmp_path / "harness") in argv) is has_harness
    if harness:
        assert f"export MERLIN_SYNTHETIC_HARNESS_DIR={harness};" in exports


@pytest.mark.parametrize("has_harness", [False, True])
def test_policy_retains_selection_for_probes_and_boxed_execution(tmp_path, monkeypatch, has_harness):
    from merlin_experiments.phase2 import campaign

    from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface

    paths, sim = _explicit_selection(tmp_path)
    workspace, package = tmp_path / "workspace", tmp_path / "package"
    workspace.mkdir()
    package.mkdir()
    harness = str(tmp_path / "harness") if has_harness else ""
    if harness:
        Path(harness).mkdir()
    secret = Path(sim.bind_paths[0]) / "answers"
    secret.mkdir()
    surfaces = (AnswerSurface("private answers", secret, "dir", "oracle"),)
    target = SimpleNamespace(target="synthetic")
    forbidden = _forbid_discovery(monkeypatch)
    monkeypatch.setattr(campaign, "answer_surfaces", forbidden)
    base_calls = []

    def base_argv(ws, bundle, **kwargs):
        base_calls.append((ws, bundle, kwargs))
        return ["bwrap", "--bind", str(ws), str(ws)]

    monkeypatch.setattr(campaign.BW, "base_argv", base_argv)
    inputs = campaign.PackageSandboxInputs(paths, sim, harness, surfaces)
    policy = campaign.package_sandbox_policy(target, workspace, package, inputs=inputs)
    assert base_calls == [(workspace, {}, {"repo": paths.repo, "_policy_test_live_inputs": True})]
    assert policy.coverage_gap == ()
    # The selected toolchain exposes this surface before the real masking pass.
    raw = ["bwrap", "--ro-bind", sim.bind_paths[0], sim.bind_paths[0]]
    assert campaign.BW.coverage_gap(raw, surfaces) == list(surfaces)
    assert campaign.BW.coverage_gap(list(policy.argv), surfaces) == []
    assert ["--tmpfs", str(secret)] == list(policy.argv)[list(policy.argv).index("--tmpfs") :][:2]
    assert paths.merlin_clang in policy.env_prefix
    assert "SELECTED_SIM=explicit" in policy.env_prefix
    assert ("MERLIN_HWBRINGUP_HARNESS_DIR=" in policy.env_prefix) is has_harness

    # Even replacing the entire renderer must not affect an already selected policy.
    monkeypatch.setattr(TC, "sandbox_env", forbidden)
    sim.env_extra["SELECTED_SIM"] = "changed-after-policy"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="selected tool available", stderr="")

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    rows = campaign.run_tool_probes(policy, timeout=17)
    assert [row["label"] for row in rows] == [probe.label for probe in policy.required_tools]
    for (argv, kwargs), probe in zip(calls, policy.required_tools, strict=True):
        assert argv == [*policy.argv, "bash", "-c", policy.env_prefix + probe.cmd]
        assert kwargs == {"capture_output": True, "text": True, "timeout": 17}
    pkg = SimpleNamespace(
        directory=package,
        tool=package / "tool",
        manifest={"commands": {"compile": {"argv": ["{tool}", "{input_mlir}", "{output_json}"]}}},
    )
    input_mlir, output_json = workspace / "input.mlir", workspace / "output.json"
    with campaign.boxed_entrypoints(policy):
        result = campaign.oot_runner.run_entrypoint(pkg, "compile", input_mlir, output_json, timeout=23)
    assert result.returncode == 0
    assert calls[-1] == (
        [
            *policy.argv,
            "--chdir",
            str(package),
            "bash",
            "-c",
            policy.env_prefix + 'exec "$@"',
            "perf-package",
            str(pkg.tool),
            str(input_mlir),
            str(output_json),
        ],
        {"capture_output": True, "text": True, "timeout": 23},
    )
    assert "changed-after-policy" not in calls[-1][0][len(policy.argv) + 4]


def test_explicit_policy_rejects_package_bind_that_reopens_answer_surface(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import campaign

    from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface

    paths, sim = _explicit_selection(tmp_path)
    workspace, package = tmp_path / "workspace", tmp_path / "package"
    workspace.mkdir()
    package.mkdir()
    surfaces = (AnswerSurface("protected package", package, "dir", "oracle"),)
    monkeypatch.setattr(campaign.BW, "base_argv", lambda *args, **kwargs: ["bwrap"])
    inputs = campaign.PackageSandboxInputs(paths, sim, "", surfaces)
    with pytest.raises(campaign.CampaignGateError, match="exposes derived answer surfaces"):
        campaign.package_sandbox_policy(SimpleNamespace(target="synthetic"), workspace, package, inputs=inputs)
