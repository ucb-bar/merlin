"""Neutral runtime/mount controls; no client or target qualification claim."""

import dataclasses
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import codex_author_runtime as R
from merlin_experiments.phase2.contracts import StageGateError


@pytest.fixture
def selected(tmp_path, monkeypatch):
    client = tmp_path / "versioned-client" / "bin" / "codex"
    client.parent.mkdir(parents=True)
    client.write_text("synthetic client, never executed")
    client.chmod(0o700)
    client.with_name("codex-code-mode-host").write_text("synthetic companion")
    auth_root = tmp_path / "private-control"
    auth_root.mkdir()
    (auth_root / "auth.json").write_text("synthetic credential; never read")
    sandbox = tmp_path / "bwrap"
    sandbox.write_text("synthetic sandbox; never executed")
    monkeypatch.setattr(R.CA, "real_codex_home", lambda: auth_root)
    monkeypatch.setattr(R.CR.shutil, "which", lambda name: str(sandbox))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    public = tmp_path / "public"
    public.mkdir()
    argv = (
        str(sandbox),
        "--ro-bind",
        str(public),
        "/perf-corpus",
        "--ro-bind",
        str(public),
        "/perf-control",
        "--ro-bind",
        str(public),
        "/hidden-then-masked",
        "--tmpfs",
        "/hidden-then-masked",
        "--bind",
        str(workspace),
        str(workspace),
    )
    policy = AW.AgentSandboxPolicy(argv, (), "available_not_an_isolation_claim", True, True, True)
    return dict(binary=client, home_root=tmp_path / "client-homes", workspace=workspace, policy=policy)


def test_explicit_nonlegacy_client_and_public_mounts_use_original_runtime(selected):
    runtime = R.select_runtime(**selected)
    assert type(runtime) is R.CR.SelectedCodexRuntime
    assert "/perf-corpus" in runtime.read_paths and "/perf-control" in runtime.read_paths
    assert "/hidden-then-masked" not in runtime.read_paths
    assert str(selected["workspace"]) not in runtime.read_paths
    assert str(selected["binary"]) in runtime.read_paths
    assert str(selected["binary"].with_name("codex-code-mode-host")) in runtime.read_paths
    kwargs = runtime.round_kwargs()
    assert kwargs["require_fresh_home"] is True
    assert kwargs["candidate_read_paths"] == runtime.read_paths
    home = selected["home_root"] / "round"
    home.mkdir(parents=True)
    mounts = kwargs["runtime_binds"](home)
    assert ["--ro-bind", str(selected["binary"]), str(selected["binary"])] == mounts[
        mounts.index(str(selected["binary"])) - 1 : mounts.index(str(selected["binary"])) + 2
    ]
    assert str(kwargs["auth_source"]) not in runtime.read_paths
    R.verify_policy(runtime, workspace=selected["workspace"], policy=selected["policy"])


def test_later_writable_mount_does_not_become_a_native_read_grant(selected):
    policy = selected["policy"]
    selected["policy"] = dataclasses.replace(
        policy, argv=(*policy.argv, "--bind", str(selected["workspace"]), "/perf-control")
    )
    runtime = R.select_runtime(**selected)
    assert "/perf-control" not in runtime.read_paths
    assert "/perf-corpus" in runtime.read_paths


@pytest.mark.parametrize("fault", ["binary", "companion", "public_mount", "root_grant", "credential_grant"])
def test_runtime_or_public_policy_drift_refuses(selected, fault):
    runtime = R.select_runtime(**selected)
    if fault in {"binary", "companion"}:
        leaf = selected["binary"] if fault == "binary" else selected["binary"].with_name("codex-code-mode-host")
        leaf.write_text("changed runtime bytes")
    else:
        policy = selected["policy"]
        destination = {
            "public_mount": "/changed-public",
            "root_grant": "/",
            "credential_grant": str(runtime.round_kwargs()["auth_source"].parent),
        }[fault]
        selected["policy"] = dataclasses.replace(
            policy, argv=(*policy.argv, "--ro-bind", str(selected["workspace"]), destination)
        )
    with pytest.raises((ValueError, StageGateError)):
        R.verify_policy(runtime, workspace=selected["workspace"], policy=selected["policy"])


@pytest.mark.parametrize("automatic", [False, True])
def test_author_transport_forwards_explicit_continuation_only(selected, monkeypatch, automatic):
    from merlin_experiments.phase2 import search_session
    from merlin_experiments.phase2.campaign import PackageSandboxInputs

    policy = selected["policy"]
    sentinel = object()
    chosen = dict(continue_session=True, continuation_prompt="selected search", continuation_stop=sentinel)
    seen = {}
    def options(broker, **kwargs):
        assert broker is sentinel
        return chosen
    monkeypatch.setattr(search_session, "driver_options", options)
    monkeypatch.setattr(R, "prepare_dispatch", lambda **kwargs: (
        SimpleNamespace(round_kwargs=lambda: {}), sentinel, dict(policy=policy)))
    def run(*args, **kwargs):
        seen.update(kwargs)
        return 0, selected["workspace"]/'transcript.jsonl'
    inputs = PackageSandboxInputs(None, None, "", ())
    result = R.run_author_round(
        selected["workspace"], selected["workspace"], SimpleNamespace(text="frozen prompt"),
        None, (), None, None, None, None, sandbox_inputs=inputs, model="m", resolved_model="m",
        effort="high", round_index=0, timeout_s=600, codex_binary=selected["binary"],
        driver=SimpleNamespace(run_round=run), session_control=sentinel if automatic else None)
    assert result[0] == 0 and result[2] is policy
    for key in chosen:
        assert (key in seen) is automatic
        if automatic:
            assert seen[key] == chosen[key]
