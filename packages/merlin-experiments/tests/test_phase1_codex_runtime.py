"""Explicit selection controls with synthetic files; no client/native isolation claim."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import session as S
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase1.providers import codex_runtime as R
from merlin_experiments.phase1.providers import execution as E


def _options(tmp_path, **changes):
    binary, auth = tmp_path / "client", tmp_path / "credential"
    binary.write_text("synthetic nonexecuted executable\n")
    binary.chmod(0o700)
    auth.write_text("synthetic credential; no reader is authorized\n")
    options = parse_options(["--run-id", "selected", "--driver", "codex", "--model", "explicit"], environ={})
    return dataclasses.replace(
        options,
        codex_binary=str(binary),
        codex_auth_source=str(auth),
        codex_home_root=str(tmp_path / "homes"),
        **changes,
    )


def _select(tmp_path, monkeypatch, options):
    from merlin.targetgen import target_experiment as TE
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox import toolchain as TC

    ws, private = tmp_path / "workspace", tmp_path / "private-run"
    ws.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    public = tmp_path / "public.txt"
    public.write_text("public synthetic bytes")
    monkeypatch.setattr(TE, "load_target_experiment", lambda path: object())
    monkeypatch.setattr(BW, "_snapshot_grants", lambda *args: ({}, [("public.txt", public, public)]))
    monkeypatch.setattr(
        TC,
        "toolchain_binds",
        lambda target: ["--ro-bind", "/tool", "/tool", "--unsetenv", "TOKEN", "--tmpfs", "/masked"],
    )
    monkeypatch.setattr(TC, "sandbox_env", lambda target, workspace: "fixed tool environment")
    monkeypatch.setattr(
        TC.ToolchainPaths,
        "from_checkout",
        lambda: SimpleNamespace(repo=tmp_path, python_import_roots=("/public/python",)),
    )
    context = SimpleNamespace(repo=tmp_path, descriptor=tmp_path / "descriptor")
    return R.select(options, context, {}, ws, private), context, ws, private


def test_legacy_selection_has_no_runtime_owner(tmp_path):
    options = parse_options(["--run-id", "legacy"], environ={})
    assert R.selection_record(options) is None
    assert (
        "codex_runtime"
        not in S.RunRequest(
            SimpleNamespace(readback_policy=None),
            options,
            None,
            Path("manifest"),
            (),
            Path("entry"),
            False,
            {},
            lambda: (),
        ).run_config
    )


@pytest.mark.parametrize(
    "change",
    [
        {"codex_binary": ""},
        {"codex_auth_source": ""},
        {"codex_home_root": ""},
        {"driver": "auto"},
        {"driver": "converse"},
        {"sandbox": "none"},
        {"allow_unsandboxed": True},
        {"codex_auth_source": "relative"},
    ],
)
def test_partial_or_wrong_route_refuses_before_native_probe(tmp_path, monkeypatch, change):
    from merlin.targetgen.sandbox import preflight

    options = dataclasses.replace(_options(tmp_path), **change)
    monkeypatch.setattr(preflight, "require_working_sandbox", lambda **kwargs: pytest.fail("native probe started"))
    with pytest.raises(ValueError):
        S.validate_options(options)


@pytest.mark.parametrize("alias", ["same", "hardlink"])
def test_credential_alias_refuses_before_any_byte_read(tmp_path, monkeypatch, alias):
    options = _options(tmp_path)
    binary = Path(options.codex_binary)
    auth = binary
    if alias == "hardlink":
        auth = tmp_path / "credential-alias"
        auth.hardlink_to(binary)
    options = dataclasses.replace(options, codex_auth_source=str(auth))
    monkeypatch.setattr(Path, "open", lambda *args, **kw: pytest.fail("selected credential alias was read"))
    with pytest.raises(ValueError, match="distinct"):
        R.selection_record(options)


def test_selection_never_reads_credential_and_existing_parent_is_supported(tmp_path, monkeypatch):
    options = _options(tmp_path)
    Path(options.codex_home_root).mkdir()
    original = Path.open

    def guarded(path, *args, **kw):
        if path == Path(options.codex_auth_source):
            pytest.fail("credential bytes read")
        return original(path, *args, **kw)

    monkeypatch.setattr(Path, "open", guarded)
    record = R.selection_record(options)
    assert record["auth_source"] == options.codex_auth_source
    assert not any(key in record for key in ("auth_sha256", "auth_bytes", "token"))


@pytest.mark.parametrize("changed", ["binary", "missing_auth", "symlink_auth", "private_read", "mount", "environment"])
def test_live_selection_and_dispatch_drift_refuse(tmp_path, monkeypatch, changed):
    from merlin.targetgen.sandbox import toolchain as TC

    options = _options(tmp_path)
    runtime, context, ws, private = _select(tmp_path, monkeypatch, options)
    if changed == "binary":
        Path(options.codex_binary).write_text("changed executable")
    elif changed == "missing_auth":
        Path(options.codex_auth_source).unlink()
    elif changed == "symlink_auth":
        Path(options.codex_auth_source).unlink()
        Path(options.codex_auth_source).symlink_to(options.codex_binary)
    elif changed == "private_read":
        monkeypatch.setattr(
            TC, "toolchain_binds", lambda target: ["--ro-bind", options.codex_auth_source, options.codex_auth_source]
        )
        with pytest.raises(ValueError, match="private"):
            R.select(options, context, {}, ws, private)
        return
    elif changed == "mount":
        monkeypatch.setattr(TC, "toolchain_binds", lambda target: [])
    else:
        monkeypatch.setattr(TC, "sandbox_env", lambda target, workspace: "changed environment")
    monkeypatch.setattr(
        E, "sandbox_command", lambda *args, **kw: pytest.fail("changed dispatch reached sandbox composer")
    )
    with pytest.raises(ValueError):
        E.codex_sandbox_command("issued", ws, {}, context=context, private_run_dir=private, runtime=runtime)


def test_resume_record_binds_complete_base_selection(tmp_path):
    options = _options(tmp_path)
    environment = {"run_config": {"codex_runtime": R.selection_record(options)}}
    S._verify_client_selection(dataclasses.replace(options, resume=True), environment)
    other = tmp_path / "other-auth"
    other.write_text("synthetic unobserved credential")
    with pytest.raises(RuntimeError, match="selection changed"):
        S._verify_client_selection(dataclasses.replace(options, codex_auth_source=str(other)), environment)
    Path(options.codex_binary).write_text("changed executable")
    with pytest.raises(RuntimeError, match="selection changed"):
        S._verify_client_selection(options, environment)


def test_prepared_replay_binds_read_paths_and_environment(tmp_path, monkeypatch):
    options = _options(tmp_path)
    runtime, context, ws, private = _select(tmp_path, monkeypatch, options)
    selected = SimpleNamespace(
        request=SimpleNamespace(options=options, context=context),
        bundle={},
        workspace=ws,
        run_dir=private,
        environment={"codex_runtime_policy": runtime.record()},
    )
    assert S.PreparedRun.selected_codex_runtime.fget(selected).record() == runtime.record()
    selected.environment["codex_runtime_policy"]["read_paths"].append("/substituted")
    with pytest.raises(RuntimeError, match="permission/runtime selection changed"):
        S.PreparedRun.selected_codex_runtime.fget(selected)


def test_ordinary_launch_consumes_the_same_live_runtime(tmp_path, monkeypatch):
    from merlin_experiments.phase1.providers import agent_bridge

    from merlin.targetgen import target_experiment as TE

    options = _options(tmp_path)
    runtime, context, ws, private = _select(tmp_path, monkeypatch, options)
    (ws / "TASK.md").write_text("sealed synthetic task")
    monkeypatch.setattr(agent_bridge, "bridged_name", lambda *args: None)
    monkeypatch.setattr(TE, "load_target_experiment", lambda path: "target")
    monkeypatch.setattr(E.FL, "start_brokers", lambda *args: "owned")
    monkeypatch.setattr(E.FL, "stop_brokers", lambda *args, **kw: None)
    seen = []

    def run(*args, **kw):
        seen.append(kw)
        assert kw["auth_source"] == Path(options.codex_auth_source)
        assert kw["codex_home_root"] == Path(options.codex_home_root)
        assert kw["runtime_binds"].__self__ is runtime
        assert kw["sandbox_command"].func is E.codex_sandbox_command
        assert kw["require_fresh_home"] is True
        assert "effective_model" not in kw  # Preserve the provider's existing alias resolution.
        return 0, private / "transcript"

    monkeypatch.setattr(CA, "run_round", run)
    provider = E.ProviderConfig("codex", codex_runtime=runtime)
    config = E.ExecutionConfig(context, provider, lambda: (), tmp_path / "timing")
    assert E.launch(ws, private, "explicit", "high", "bwrap", {}, 1, 60, config=config)[0] == 0
    assert len(seen) == 1
    for rnd in (0, 1):
        home = Path(options.codex_home_root) / f"selected_r{rnd:02d}"
        mounts = runtime.runtime_binds(home)
        assert mounts[-2:] == ["CODEX_HOME", str(home)]


def test_explicit_auth_does_not_discover_real_home(tmp_path, monkeypatch):
    options = _options(tmp_path)
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(CA, "real_codex_home", lambda: pytest.fail("ambient home discovered"))
    result = CA.prepare_codex_home(
        tmp_path / "home",
        model="explicit",
        effort="high",
        workspace=ws,
        candidate_read_paths=("/public",),
        auth_source=Path(options.codex_auth_source),
    )
    assert result["auth_source"] == options.codex_auth_source and result["auth_copied"] is False
    assert not (tmp_path / "home/auth.json").exists()


def test_partial_provider_auth_selection_stops_before_client_version(tmp_path, monkeypatch):
    options = _options(tmp_path)
    monkeypatch.setattr(CA, "cli_version", lambda *args: pytest.fail("client launched"))
    with pytest.raises(ValueError, match="complete isolated"):
        CA.run_round(
            tmp_path, tmp_path, "explicit", {}, None, "bwrap", 0, 60, auth_source=Path(options.codex_auth_source)
        )


def test_new_host_private_sources_are_registered():
    from merlin.common import access

    for module in ("merlin_experiments.phase1.canary", "merlin_experiments.phase1.providers.codex_runtime"):
        assert any(item.origin == "grader" and module in item.modules for item in access.MODULE_ACCESS)


def test_existing_per_round_home_refuses_before_client_version(tmp_path, monkeypatch):
    options = _options(tmp_path)
    runtime, context, ws, private = _select(tmp_path, monkeypatch, options)
    home = Path(options.codex_home_root) / f"{private.name}_r00"
    home.mkdir(parents=True)
    monkeypatch.setattr(CA, "cli_version", lambda *a: pytest.fail("stale client launched"))
    with pytest.raises(ValueError, match="existing Codex home"):
        CA.run_round(
            ws,
            private,
            "explicit",
            {},
            None,
            "bwrap",
            0,
            60,
            **E.codex_call_kwargs(runtime, context=context, run_dir=private, model="explicit"),
        )


def test_preflight_accepts_the_complete_same_client_selection_without_provider(tmp_path, monkeypatch):
    from merlin.targetgen.sandbox import preflight

    options = _options(tmp_path, preflight_only=True)
    seal = tmp_path / "seal"
    seal.write_text("synthetic seal")
    options = dataclasses.replace(options, corpus_seal=str(seal), bundle="selected")
    monkeypatch.setattr(preflight, "require_working_sandbox", lambda **kw: None)
    monkeypatch.setattr(CA, "run_round", lambda *a, **kw: pytest.fail("provider started"))
    assert S.validate_options(options) is None
    assert R.selection_record(options)["auth_source"] == options.codex_auth_source
