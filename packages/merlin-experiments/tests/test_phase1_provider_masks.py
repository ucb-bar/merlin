"""Provider mount-policy regression, not kernel or operating-system isolation qualification."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.providers import execution as E

from merlin.targetgen import target_experiment
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    repo = tmp_path / "repo"
    hidden = repo / "inputs/hidden"
    hidden.mkdir(parents=True)
    (hidden / "secret.bin").write_bytes(b"synthetic private payload")
    (repo / "inputs/public.txt").write_bytes(b"synthetic public input")
    workspace = tmp_path / "run/workspace"
    workspace.mkdir(parents=True)
    bundle = {"allowed": [{"path": "inputs"}], "host_inputs": [{"path": "inputs/hidden"}]}
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)
    context = InvocationContext(repo, repo / "descriptor", repo, "fixture", repo, repo, repo, ())
    target = SimpleNamespace(target="fixture")
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: target)
    monkeypatch.setattr(BW, "repo_root", lambda: repo)
    monkeypatch.setattr(BW, "claude_runtime_binds", lambda: [])
    monkeypatch.setattr(toolchain, "toolchain_binds", lambda te: [])
    monkeypatch.setattr(toolchain, "sandbox_env", lambda te, ws: "true;")
    monkeypatch.setattr(BW, "answer_surfaces", lambda te: [])
    surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    monkeypatch.setattr(surfaces, "answer_surfaces", lambda te: [])
    # Capture the actual composed argv at the final shell-transport boundary.
    monkeypatch.setattr(BW, "compose_command", lambda argv, payload, ws: argv)
    yield SimpleNamespace(
        repo=repo, ws=workspace, bundle=bundle, hidden=hidden, frozen=frozen, context=context, target=target
    )
    BW.remove_bundle_snapshot(workspace)


def _command(prepared, extra):
    return E.sandbox_command("true", prepared.ws, prepared.bundle, extra, context=prepared.context)


def test_candidate_cannot_inherit_or_reintroduce_frozen_host_context(prepared, monkeypatch):
    variable = "MERLIN_FROZEN_PYTHON_CONTEXT"
    monkeypatch.setenv(variable, "host-only-pinned-context")
    argv = _command(prepared, ["--setenv", variable, "provider-reintroduced-value"])
    assert argv[-2:] == ["--unsetenv", variable]
    assert argv.index("provider-reintroduced-value") < len(argv) - 2
    assert os.environ[variable] == "host-only-pinned-context"


def _assert_masked(prepared, extra, exposed, *, command_extra=None):
    assert BW.is_exposed(extra, exposed), "negative control must expose actual private bytes"
    argv = _command(prepared, extra if command_extra is None else command_extra)
    assert not BW.is_exposed(argv, exposed)
    assert not BW.is_exposed(argv, prepared.hidden / "secret.bin")
    assert BW.is_exposed(argv, prepared.repo / "inputs/public.txt")
    assert BW.coverage_gap(argv, BW.host_input_surfaces(argv, prepared.ws, prepared.bundle, repo=prepared.repo)) == []
    return argv


@pytest.mark.parametrize("bind_owner", ["provider", "toolchain"])
@pytest.mark.parametrize("location", ["live", "frozen"])
@pytest.mark.parametrize("scope", ["directory", "parent", "file"])
def test_actual_provider_private_alias_masks(prepared, monkeypatch, bind_owner, location, scope):
    source = prepared.hidden if location == "live" else prepared.frozen
    tail = Path("secret.bin")
    if scope == "parent":
        tail = source.name / tail
        source = source.parent
    elif scope == "file":
        source = source / tail
        tail = Path(".")
    destination = prepared.repo / "runtime-view"
    extra = ["--ro-bind", str(source), str(destination)]
    if bind_owner == "toolchain":
        monkeypatch.setattr(toolchain, "toolchain_binds", lambda te: extra)
        assert BW.is_exposed(extra, destination / tail)
        argv = _assert_masked(prepared, extra, destination / tail, command_extra=[])
        # The common complete policy must agree with the provider-specific prefix.
        assert not BW.is_exposed(BW.full_argv(prepared.target, prepared.ws, prepared.bundle), destination / tail)
    else:
        argv = _assert_masked(prepared, extra, destination / tail)
    assert (argv.index("MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT") < argv.index(str(destination))) == (
        bind_owner == "provider"
    )


@pytest.mark.parametrize("scope", ["parent", "root", "file"])
def test_actual_provider_masks_private_snapshot_inventory_alias(prepared, scope):
    snapshot = BW.bundle_snapshot_root(prepared.ws)
    marker = snapshot / "snapshot.json"
    source = {"parent": snapshot.parent, "root": snapshot, "file": marker}[scope]
    destination = prepared.repo / "metadata-view"
    exposed = destination / marker.relative_to(source) if source != marker else destination
    _assert_masked(prepared, ["--ro-bind", str(source), str(destination)], exposed)


@pytest.mark.parametrize("location", ["live", "frozen"])
@pytest.mark.parametrize("scope", ["directory", "file"])
def test_actual_provider_masks_runtime_hardlink(prepared, tmp_path, location, scope):
    runtime = tmp_path / "external-runtime"
    runtime.mkdir()
    source = prepared.hidden if location == "live" else prepared.frozen
    os.link(source / "secret.bin", runtime / "linked.bin")
    source = runtime if scope == "directory" else runtime / "linked.bin"
    destination = prepared.repo / "runtime-view"
    exposed = destination / "linked.bin" if scope == "directory" else destination
    _assert_masked(prepared, ["--ro-bind", str(source), str(destination)], exposed)


@pytest.mark.parametrize("damage", ["missing_marker", "missing_host_mapping", "changed_host_membership"])
def test_actual_provider_refuses_unverified_private_mapping(prepared, damage):
    marker = BW.bundle_snapshot_root(prepared.ws) / "snapshot.json"
    if damage == "missing_marker":
        marker.parent.chmod(0o700)
        marker.unlink()
    else:
        manifest = json.loads(marker.read_text())
        if damage == "missing_host_mapping":
            manifest["host_records"] = []
        else:
            manifest["host_inputs"] = []
        marker.chmod(0o600)
        marker.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        _command(prepared, [])


@pytest.mark.parametrize("grants", [False, True])
def test_no_private_inputs_preserve_exact_existing_argv_and_workspace_rebind(prepared, grants):
    BW.remove_bundle_snapshot(prepared.ws)
    prepared.bundle.clear()
    if grants:
        prepared.bundle["allowed"] = [{"path": "inputs/public.txt"}]
    BW.materialize_bundle_inputs(prepared.ws, prepared.bundle, repo=prepared.repo)
    extra = ["--ro-bind", str(prepared.repo / "inputs/public.txt"), str(prepared.repo / "public-alias")]
    # Historical provider composition, including its unconditional workspace
    # rebind for empty bundles. Shared full_argv's conditional rebind is distinct.
    before = BW.base_argv(prepared.ws, prepared.bundle, repo=prepared.repo)
    before += BW.claude_runtime_binds() + toolchain.toolchain_binds(prepared.target)
    before += [
        "--unsetenv",
        "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT",
        "--unsetenv",
        "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED",
        "--unsetenv",
        "MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD",
        *extra,
    ]
    before = BW.reapply_bundle_snapshot(before, prepared.ws, prepared.bundle, repo=prepared.repo)
    expected = BW.apply_answer_masks(before, BW.answer_surfaces(prepared.target))
    expected += ["--unsetenv", "MERLIN_FROZEN_PYTHON_CONTEXT"]
    assert _command(prepared, extra) == expected
    assert expected[-5:-2] == ["--bind", str(prepared.ws), str(prepared.ws)]
    assert expected[-2:] == ["--unsetenv", "MERLIN_FROZEN_PYTHON_CONTEXT"]

    common = BW.base_argv(prepared.ws, prepared.bundle, repo=prepared.repo)
    common += BW.claude_runtime_binds() + toolchain.toolchain_binds(prepared.target)
    if prepared.bundle:
        common = BW.reapply_bundle_snapshot(common, prepared.ws, prepared.bundle, repo=prepared.repo)
    assert BW.full_argv(prepared.target, prepared.ws, prepared.bundle) == (
        BW.apply_answer_masks(common, BW.answer_surfaces(prepared.target))
        + ["--unsetenv", "MERLIN_FROZEN_PYTHON_CONTEXT"]
    )
