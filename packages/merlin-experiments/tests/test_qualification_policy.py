"""Frozen execution resources survive live-path drift and keep private aliases masked."""

import copy
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import campaign
from merlin_experiments.phase2 import qualification_policy as policy

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


@pytest.fixture(autouse=True)
def refuse_processes(monkeypatch, tmp_path):
    def refused(*args, **kwargs):
        pytest.fail("resource-policy tests must not launch a process")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "content-store"))


def write(path, payload=b"synthetic resource\n", mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


@pytest.fixture
def resources(tmp_path):
    live = tmp_path / "live"
    repo = live / "repo"
    repo.mkdir(parents=True)
    real_venv = live / "real-venv"
    executable = write(real_venv / "bin/python3", b"synthetic interpreter\n", 0o751)
    private = write(real_venv / "backend/answer.py", b"private answer\n")
    alias = repo / ".venv"
    alias.symlink_to(real_venv, target_is_directory=True)
    paths = TC.ToolchainPaths(
        repo=repo,
        venv=str(alias),
        llvm=str(live / "llvm"),
        compat_lib=str(live / "compat"),
        clang_install=str(live / "clang"),
        uv_python=str(live / "uv"),
    )
    write(Path(paths.llvm) / "bin/mlir-opt", mode=0o755)
    write(Path(paths.compat_lib) / "compat.so")
    write(Path(paths.clang_bin) / "clang-23", mode=0o755)
    write(Path(paths.clang_resource) / "include/stddef.h")
    write(Path(paths.uv_python) / "python/bin/python3", mode=0o755)
    simulator = live / "simulator"
    write(simulator / "bin/sim", b"synthetic simulator\n", 0o711)
    write(simulator / "lib/sim.so")
    harness = live / "harness"
    write(harness / "harness.c", b"synthetic harness\n")
    sim = TC.SimToolchain(
        bind_paths=(str(simulator),),
        path_dirs=(str(simulator / "bin"),),
        ld_dirs=(str(simulator / "lib"),),
        env_extra={"FIXTURE_SIM": str(simulator / "bin/sim")},
        probes=(TC.ToolProbe("fixture simulator", "sim --version", str(simulator)),),
    )
    surface = AnswerSurface("private backend", private.parent, "dir", "backend")
    inputs = campaign.PackageSandboxInputs(paths, sim, str(harness), (surface,))
    root = tmp_path / "qualification"
    root.mkdir()
    return SimpleNamespace(
        root=root,
        live=live,
        target=SimpleNamespace(target="fixture"),
        inputs=inputs,
        executable=executable,
        private=private,
        alias=alias,
        real_venv=real_venv,
    )


def sandbox(resources, frozen):
    workspace = resources.root / "candidate-workspace"
    package = resources.root / "candidate-package"
    workspace.mkdir(exist_ok=True)
    package.mkdir(exist_ok=True)
    return campaign.package_sandbox_policy(resources.target, workspace, package, inputs=frozen)


def bind_source(argv, destination):
    matches = [
        Path(argv[index + 1])
        for index, argument in enumerate(argv[:-2])
        if argument == "--ro-bind" and argv[index + 2] == str(destination)
    ]
    assert matches, f"missing frozen bind to {destination}"
    return matches[-1]


def test_frozen_environment_retains_explicit_import_roots(resources, monkeypatch):
    import_root = resources.real_venv / "lib/site-packages"
    write(import_root / "synthetic.py", b"VALUE = 1\n")
    inputs = replace(resources.inputs, paths=replace(resources.inputs.paths, python_import_roots=(str(import_root),)))
    record = policy.freeze(resources.root, resources.target, inputs)
    monkeypatch.setenv("PYTHONPATH", "/ambient/replacement")

    def refused(*args, **kwargs):
        pytest.fail("frozen execution must not reconstruct its environment")

    monkeypatch.setattr(TC, "sandbox_env", refused)
    frozen = policy.restore(resources.root, record)
    selected = sandbox(resources, frozen)
    assert f"export PYTHONPATH={import_root}; " in selected.env_prefix
    assert "$PYTHONPATH" not in selected.env_prefix
    assert "/ambient/replacement" not in selected.env_prefix
    source = bind_source(selected.argv, resources.real_venv)
    assert (source / "lib/site-packages/synthetic.py").read_bytes() == b"VALUE = 1\n"


@pytest.mark.parametrize("drift", ["removed", "retargeted"])
def test_restore_uses_frozen_bytes_at_original_destinations(resources, monkeypatch, drift):
    record = policy.freeze(resources.root, resources.target, resources.inputs)
    resources.live.rename(resources.live.with_name("retired-live"))
    if drift == "retargeted":
        replacement = resources.live / "replacement"
        write(replacement / "bin/python3", b"replacement interpreter\n", 0o755)
        resources.alias.parent.mkdir(parents=True, exist_ok=True)
        resources.alias.symlink_to(replacement, target_is_directory=True)

    def ambient(*args, **kwargs):
        pytest.fail("restoration must not rediscover live toolchain policy")

    monkeypatch.setattr(TC.ToolchainPaths, "from_checkout", ambient)
    monkeypatch.setattr(TC, "_sim", ambient)
    monkeypatch.setattr(TC, "curated_harness_dir", ambient)
    monkeypatch.setattr(campaign, "answer_surfaces", ambient)
    frozen = policy.restore(resources.root, record)
    selected = sandbox(resources, frozen)
    assert TC.RESOLVE_DIR not in selected.argv
    for destination in (resources.alias, resources.real_venv):
        source = bind_source(selected.argv, destination)
        assert source.is_relative_to(resources.root)
        assert (source / "bin/python3").read_bytes() == b"synthetic interpreter\n"
        executable = source / "bin/python3"
        mode = stat.S_IMODE(executable.stat().st_mode)
        assert mode & 0o111 and not mode & 0o222
        member = executable.relative_to(Path(record["snapshot"]["path"])).as_posix()
        assert next(row["mode"] for row in record["inventory"] if row["path"] == member) == mode
    sim_source = bind_source(selected.argv, resources.inputs.sim.bind_paths[0])
    assert (sim_source / "bin/sim").read_bytes() == b"synthetic simulator\n"
    assert stat.S_IMODE((sim_source / "bin/sim").stat().st_mode) == 0o555
    harness_source = bind_source(selected.argv, resources.inputs.harness)
    assert (harness_source / "harness.c").read_bytes() == b"synthetic harness\n"
    assert stat.S_IMODE((harness_source / "harness.c").stat().st_mode) == 0o444
    assert str(resources.alias / "bin") in selected.env_prefix
    assert str(resources.inputs.sim.path_dirs[0]) in selected.env_prefix
    assert selected.required_tools[-1] == resources.inputs.sim.probes[0]


def test_actual_campaign_masks_original_and_frozen_private_aliases(resources):
    record = policy.freeze(resources.root, resources.target, resources.inputs)
    frozen = policy.restore(resources.root, record)
    selected = sandbox(resources, frozen)
    private_paths = {resources.private, resources.alias / "backend/answer.py"}
    for destination in (resources.alias, resources.real_venv):
        source = bind_source(selected.argv, destination)
        assert (source / "backend/answer.py").read_bytes() == b"private answer\n"
        private_paths.add(source / "backend/answer.py")
        # A real copied answer sits beneath a useful tool bind: masking is non-vacuous.
        unmasked = ["--ro-bind", str(source), str(destination)]
        assert BW.is_exposed(unmasked, destination / "backend/answer.py")
    assert len(private_paths) >= 3
    for path in private_paths:
        assert not BW.is_exposed(list(selected.argv), path), path
    assert not selected.coverage_gap
    assert BW.coverage_gap(list(selected.argv), list(frozen.surfaces)) == []
    assert BW.is_exposed(list(selected.argv), resources.alias / "bin/python3")


def test_private_tool_descendants_never_enter_shared_content_store(resources, tmp_path):
    record = policy.freeze(resources.root, resources.target, resources.inputs)
    selected = sandbox(resources, policy.restore(resources.root, record))
    objects = [path for path in (tmp_path / "content-store").rglob("*") if path.is_file()]
    assert objects, "disjoint public tools should still use the content store"
    assert all(path.read_bytes() != b"private answer\n" for path in objects)
    for destination in (resources.alias, resources.real_venv):
        source = bind_source(selected.argv, destination)
        private = source / "backend/answer.py"
        assert private.read_bytes() == b"private answer\n"
        assert private.stat().st_nlink == 1
        assert not any(private.samefile(path) for path in objects)
    public_tool = bind_source(selected.argv, resources.inputs.paths.llvm) / "bin/mlir-opt"
    assert any(public_tool.samefile(path) for path in objects)
    assert public_tool.stat().st_nlink > 1


def test_explicit_missing_harness_refuses_freeze(resources):
    inputs = replace(resources.inputs, harness=str(resources.live / "missing-harness"))
    with pytest.raises(campaign.CampaignGateError):
        policy.freeze(resources.root, resources.target, inputs)


def test_explicit_public_contracts_survive_while_nested_private_alias_stays_masked(resources):
    backend = resources.private.parent
    contracts = backend / "contracts"
    write(contracts / "interface.h", b"public interface\n")
    (contracts / "secret.py").symlink_to(resources.private)
    surface = replace(resources.inputs.surfaces[0], grantable=("contracts",))
    inputs = replace(resources.inputs, harness=str(contracts), surfaces=(surface,))
    record = policy.freeze(resources.root, resources.target, inputs)

    resources.live.rename(resources.live.with_name("retired-live"))
    replacement = resources.live / "replacement"
    write(replacement / "backend/contracts/interface.h", b"changed public interface\n")
    write(replacement / "backend/contracts/secret.py", b"changed private answer\n")
    resources.real_venv.symlink_to(replacement, target_is_directory=True)
    resources.alias.parent.mkdir(parents=True)
    resources.alias.symlink_to(replacement, target_is_directory=True)

    frozen = policy.restore(resources.root, record)
    selected = sandbox(resources, frozen)
    frozen_contracts = bind_source(selected.argv, contracts)
    assert (frozen_contracts / "interface.h").read_bytes() == b"public interface\n"
    assert (frozen_contracts / "secret.py").read_bytes() == b"private answer\n"
    assert BW.is_exposed(list(selected.argv), contracts / "interface.h")
    for private in (
        resources.private,
        resources.alias / "backend/answer.py",
        resources.alias / "backend/contracts/secret.py",
        contracts / "secret.py",
        frozen_contracts / "secret.py",
    ):
        assert not BW.is_exposed(list(selected.argv), private), private
    assert BW.coverage_gap(list(selected.argv), list(frozen.surfaces)) == []
    assert TC.RESOLVE_DIR not in selected.argv


@pytest.mark.parametrize("mutation", ["bytes", "membership", "writable", "directory-writable", "executable", "linked"])
def test_frozen_resource_tamper_refused(resources, mutation):
    record = policy.freeze(resources.root, resources.target, resources.inputs)
    frozen = policy.restore(resources.root, record)
    selected = sandbox(resources, frozen)
    directory = bind_source(selected.argv, resources.alias)
    executable = directory / "bin/python3"
    if mutation == "bytes":
        mode = stat.S_IMODE(executable.stat().st_mode)
        executable.chmod(0o751)
        executable.write_bytes(b"changed interpreter\n")
        executable.chmod(mode)
    elif mutation == "membership":
        directory.chmod(0o755)
        write(directory / "unrecorded", mode=0o444)
        directory.chmod(0o555)
    elif mutation == "writable":
        executable.chmod(0o751)
    elif mutation == "directory-writable":
        directory.chmod(0o755)
    elif mutation == "linked":
        replacement = write(resources.root / "replacement-python", executable.read_bytes(), 0o551)
        executable.parent.chmod(0o755)
        executable.unlink()
        executable.symlink_to(replacement)
        executable.parent.chmod(0o555)
    else:
        executable.chmod(0o444)
    with pytest.raises(campaign.CampaignGateError):
        policy.restore(resources.root, record)


@pytest.mark.parametrize("field", ["argv", "mounts", "inventory", "env_prefix", "surfaces", "probes"])
def test_resource_record_tamper_refused(resources, field):
    record = policy.freeze(resources.root, resources.target, resources.inputs)
    changed = copy.deepcopy(record)
    if field == "env_prefix":
        changed[field] += "export INJECTED=yes; "
    else:
        assert changed[field]
        changed[field] = changed[field][:-1]
    with pytest.raises(campaign.CampaignGateError):
        policy.restore(resources.root, changed)
