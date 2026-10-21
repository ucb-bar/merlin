"""Explicit package sandbox selections survive ambient resolver changes without launching tools."""

import importlib
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen.sandbox import toolchain as TC


def test_cross_checkout_sandbox_searches_frozen_source_closure(tmp_path, monkeypatch):
    repo = tmp_path / "frozen-repo"
    legacy = repo / "merlin/python"
    legacy.mkdir(parents=True)
    (repo / "src").mkdir()
    implementation = tmp_path / "current-repo/src"
    implementation.mkdir(parents=True)
    monkeypatch.setattr(TC, "repo_root", lambda: repo)
    monkeypatch.setattr(TC, "python_source_dir", lambda: implementation)

    paths = TC.ToolchainPaths.from_checkout()
    assert paths.python_import_roots == (str(legacy), str(repo / "src"), str(implementation))


@pytest.mark.parametrize("installed", [False, True])
def test_installed_interpreter_not_shadowed_by_work_root_venv(tmp_path, monkeypatch, installed):
    repo = tmp_path / "work-root"
    (repo / ".venv").mkdir(parents=True)
    active = tmp_path / "qualified-venv"
    purelib = active / "lib/python3.12/site-packages"
    purelib.mkdir(parents=True)
    monkeypatch.setattr(TC, "repo_root", lambda: repo)
    monkeypatch.setattr(TC, "python_source_dir", lambda: purelib if installed else repo / "src")
    monkeypatch.setattr(TC, "sys", SimpleNamespace(prefix=str(active), base_prefix=str(tmp_path / "base")))
    monkeypatch.setattr(TC, "sysconfig", SimpleNamespace(get_path=lambda name: str(purelib)))

    paths = TC.ToolchainPaths.from_checkout()
    assert paths.venv == str(active if installed else repo / ".venv")


def test_installed_public_tool_imports_in_real_bwrap_with_private_mask(tmp_path, monkeypatch):
    """The selected interpreter, exact public grants and deny overlay must work together."""
    from merlin.common.paths import python_source_dir
    from merlin.targetgen import tool_registry as TR
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox import preflight
    from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface

    if sys.prefix == sys.base_prefix:
        pytest.skip("requires bwrap and an active virtual environment")
    if not preflight.probe_sandbox().usable:
        pytest.skip("this host cannot enter bwrap")

    source = python_source_dir()
    repo = tmp_path / "operator-work-root"
    (repo / ".venv").mkdir(parents=True)  # unrelated and missing PyYAML
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    public = tmp_path / "public"
    public.mkdir()
    secret = public / "answer.txt"
    secret.write_text("private answer", encoding="utf-8")
    allowed = []
    for name in TR.ARM_TOOLS["merlin_rtlchecks"]:
        for logical in TR.spec(name).bundle_paths:
            if logical.startswith("merlin/python/merlin/"):
                path = source / "merlin" / logical.removeprefix("merlin/python/merlin/")
                assert path.exists(), logical
                allowed.append({"path": str(path), "mode": "ro"})
    bundle = {"allowed": [{"path": str(public), "mode": "ro"}, *allowed], "denied": [{"path": str(secret)}]}

    # Simulate an installed distribution without replacing the public source
    # modules this checkout test is exercising.
    from sysconfig import get_path

    monkeypatch.setattr(TC, "repo_root", lambda: repo)
    monkeypatch.setattr(TC, "python_source_dir", lambda: Path(get_path("purelib")))
    paths = replace(TC.ToolchainPaths.from_checkout(), python_import_roots=(str(source),))
    assert paths.venv == str(Path(sys.prefix).resolve())
    target = SimpleNamespace(target="synthetic")
    sim = TC.SimToolchain()
    argv = BW.base_argv(workspace, bundle, repo=repo, _policy_test_live_inputs=True, include_claude_home=False)
    argv += TC.toolchain_binds(target, paths=paths, sim=sim, harness="", memory_dir="")
    argv += BW._bundle_mount_args(workspace, bundle, repo, _policy_test_live_inputs=True)
    assert BW.coverage_gap(argv, (AnswerSurface("answer", secret, "file", "golden"),)) == []
    imports = (
        "import yaml, numpy, xdsl; "
        "from merlin.targetgen.evidence.store import Evidence; "
        "from merlin.targetgen import synthesize, generate, rtl_backend; "
        "from merlin.targetgen.rtl import facts, gen_numeric_facts, gen_isa_module, gen_rtl_digest; "
        "from merlin.kernels import cca_contract, action_catalog; "
        "from merlin.targetgen.contract.interface_emit import emit_interface_mlir; "
        "from merlin.runtime.commandbuffer import pool_params; "
        "from merlin.runtime.tensor import pool_out_dims; "
        "print('PUBLIC_IMPORTS_OK')"
    )
    command = (
        TC.sandbox_env(target, workspace, paths=paths, sim=sim, harness="")
        + "python3 -P -c "
        + shlex.quote(imports)
        + " && test ! -s "
        + shlex.quote(str(secret))
    )
    result = subprocess.run([*argv, "bash", "-c", command], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "PUBLIC_IMPORTS_OK" in result.stdout


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
    assert shlex.split("PYTHONPATH=" + assignment) == ["PYTHONPATH=" + ":".join((*roots, str(tmp_path)))]
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
    assert f"export PYTHONPATH={paths.repo}/merlin/python:{tmp_path}${{PYTHONPATH:+:$PYTHONPATH}}; " in exports


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


@pytest.mark.parametrize("cfg,checkout", [(True, False), (False, False), (True, True), (False, True)])
def test_no_site_interpreter_preserves_only_explicit_venv_and_checkout_precedence(tmp_path, monkeypatch, cfg, checkout):
    repo, base, selected = (tmp_path / name for name in ("repo", "base-python", "selected-venv"))
    (selected / "bin").mkdir(parents=True)
    binary = selected / "bin/python3"
    binary.write_text("# explicit interpreter fixture\n")
    if cfg:
        (selected / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n")
    if checkout:
        (repo / ".venv").mkdir(parents=True)
    monkeypatch.setattr(TC, "repo_root", lambda: repo)
    monkeypatch.setattr(TC, "python_source_dir", lambda: repo / "src")
    monkeypatch.setattr(TC, "sys", SimpleNamespace(prefix=str(base), base_prefix=str(base), executable=str(binary)))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "unselected-ambient-venv"))
    paths = TC.ToolchainPaths.from_checkout()
    expected = selected if cfg and not checkout else repo / ".venv"
    assert paths.venv == str(expected)


def test_explicit_venv_symlink_is_resolved_after_environment_selection(tmp_path, monkeypatch):
    base, selected = tmp_path / "base", tmp_path / "venv"
    (base / "bin").mkdir(parents=True)
    (base / "bin/python3").write_text("# base interpreter\n")
    (selected / "bin").mkdir(parents=True)
    (selected / "bin/python3").symlink_to(base / "bin/python3")
    (selected / "pyvenv.cfg").write_text(f"home = {base / 'bin'}\n")
    monkeypatch.setattr(
        TC, "sys", SimpleNamespace(prefix=str(base), base_prefix=str(base), executable=str(selected / "bin/python3"))
    )
    assert TC._interpreter_prefix() == selected


def test_real_no_site_python_selects_same_environment_for_sandbox(tmp_path):
    """Run the actual bootstrap mode; neither site nor editable imports run."""
    import json

    from merlin.common.paths import python_import_roots

    binary = Path(sys.executable).absolute()
    selected = binary.parent.parent
    if not (selected / "pyvenv.cfg").is_file():
        pytest.skip("the explicitly selected test interpreter is not a virtual environment")
    program = (
        "import json,sys; from pathlib import Path; "
        "from merlin.targetgen.sandbox import toolchain as T; "
        f"T.repo_root=lambda:Path({str(tmp_path / 'no-checkout-venv')!r}); "
        "print(json.dumps({'venv':T.ToolchainPaths.from_checkout().venv,'prefix':sys.prefix,'base':sys.base_prefix}))"
    )
    # -I -S excludes all ambient paths. The fixed bootstrap adds only explicit
    # source/dependency roots, matching the sealed worker's import selection.
    roots = [str(path) for path in python_import_roots()]
    from merlin_experiments.frozen_python import _interpreter_dependencies

    roots += list(_interpreter_dependencies(str(binary)))
    bootstrap = f"import sys; sys.path[:0]={roots!r}; " + program
    result = subprocess.run(
        [str(binary), "-I", "-S", "-B", "-c", bootstrap], capture_output=True, text=True, timeout=20, check=True
    )
    assert json.loads(result.stdout)["venv"] == str(selected.resolve())
