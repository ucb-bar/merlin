"""Fake-client continuation controls only, never real client/isolation qualification."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shlex
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import canary as C
from merlin_experiments.phase1 import session as S
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import codex_runtime as R


def _options(tmp_path):
    seal, binary, auth = tmp_path / "seal", tmp_path / "client", tmp_path / "credential"
    seal.write_text("synthetic nonauthority seal")
    binary.write_text("nonexecuted client")
    binary.chmod(0o700)
    auth.write_text("synthetic credential, not read")
    options = parse_options(
        ["--run-id", "canary", "--bundle", "selected", "--driver", "codex", "--model", "explicit"], environ={}
    )
    return dataclasses.replace(
        options,
        codex_canary=True,
        corpus_seal=str(seal),
        codex_binary=str(binary),
        codex_auth_source=str(auth),
        codex_home_root=str(tmp_path / "homes"),
        round_timeout=60,
    )


def _events(command):
    item = {"id": "owned-command", "type": "command_execution", "command": command}
    return [
        {"type": "item.started", "item": {**item, "status": "in_progress"}},
        {"type": "item.completed", "item": {**item, "status": "completed", "exit_code": 0}},
    ]


@pytest.mark.parametrize("wrapper", [None, "/bin/bash -lc", "bash -c", "/bin/sh -c"])
def test_raw_exact_command_and_documented_wrapper_complete(wrapper):
    command = "python3 -I -B -c " + shlex.quote("print('owned literal')")
    actual = command if wrapper is None else wrapper + " " + shlex.quote(command)
    assert C.completed_probe(_events(actual), command) == "owned-command"


@pytest.mark.parametrize(
    "change",
    [
        "missing_start",
        "missing_end",
        "status",
        "exit_absent",
        "exit_bool",
        "exit_nonzero",
        "id",
        "duplicate_start",
        "duplicate_end",
        "reordered",
        "prefix",
        "suffix",
        "changed_literal",
        "extra_command",
        "file_change",
        "other_tool",
        "quote_expansion",
    ],
)
def test_raw_missing_failed_counterfeit_or_extra_tool_refuses(change):
    command = "python3 -I -B -c " + shlex.quote("print('$PUBLIC')")
    rows = _events(command)
    if change == "missing_start":
        rows.pop(0)
    elif change == "missing_end":
        rows.pop()
    elif change == "status":
        rows[1]["item"]["status"] = "failed"
    elif change == "exit_absent":
        rows[1]["item"].pop("exit_code")
    elif change == "exit_bool":
        rows[1]["item"]["exit_code"] = False
    elif change == "exit_nonzero":
        rows[1]["item"]["exit_code"] = 1
    elif change == "id":
        rows[1]["item"]["id"] = "substituted"
    elif change == "duplicate_start":
        rows.insert(0, rows[0])
    elif change == "duplicate_end":
        rows.append(rows[1])
    elif change == "reordered":
        rows.reverse()
    elif change in {"prefix", "suffix", "changed_literal", "quote_expansion"}:
        actual = {
            "prefix": "true; " + command,
            "suffix": command + "; true",
            "changed_literal": command.replace("$PUBLIC", "changed"),
            "quote_expansion": "python3 -I -B -c \"print('$PUBLIC')\"",
        }[change]
        rows = _events(actual)
    elif change == "extra_command":
        rows = _events("touch sitecustomize.py") + rows
    else:
        rows.insert(
            0,
            {"type": "item.completed", "item": {"type": "file_change" if change == "file_change" else "mcp_tool_call"}},
        )
    with pytest.raises(ValueError):
        C.completed_probe(rows, command)


@pytest.mark.parametrize(
    "change",
    [
        {"preflight_only": True},
        {"resume": True},
        {"no_oracle": True},
        {"skip_hidden": True},
        {"sandbox": "none"},
        {"seed_submission": "/candidate"},
        {"qualify_submission": "/candidate"},
        {"seal_current": True},
        {"continuous": True},
    ],
)
def test_canary_conflicting_modes_refuse_before_any_launch(tmp_path, monkeypatch, change):
    from merlin.targetgen.sandbox import preflight

    options = dataclasses.replace(_options(tmp_path), **change)
    monkeypatch.setattr(preflight, "require_working_sandbox", lambda **kw: pytest.fail("native launched"))
    with pytest.raises(ValueError):
        S.validate_options(options)


def test_literal_body_preserves_isolated_startup_and_fixed_selected_import_roots():
    body = C._probe(
        ("/public/interface", "public digest"),
        (Path("/synthetic/answer"), Path("/synthetic/history")),
        "print('public tool control')",
        ("/granted/python", "/workspace"),
    )
    assert b"sys.path[:0] = ('/granted/python', '/workspace')" in body
    assert b"/synthetic/answer" in body and b"/synthetic/history" in body
    assert b"public tool control" in body and b"auth.json" in body
    assert b"expected_output" not in body


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    options = _options(tmp_path)
    ws, run = tmp_path / "workspace", tmp_path / "private-run"
    ws.mkdir()
    run.mkdir()
    (ws / "submission").mkdir()
    (ws / "TASK.md").write_text("sealed ordinary task remains unchanged")
    runtime = R.SelectedCodexRuntime(
        tuple(sorted(R.selection_record(options).items())), ("/public",), (), ("/public/python", str(ws)), "fixed"
    )
    calls = []
    value = SimpleNamespace(
        request=SimpleNamespace(
            options=options,
            context=SimpleNamespace(repo=tmp_path, descriptor=tmp_path / "descriptor"),
            resolved_tools=lambda: ("selected-tool",),
        ),
        workspace=ws,
        run_dir=run,
        bundle={"bundle_id": "selected"},
        selected_codex_runtime=runtime,
    )
    monkeypatch.setattr(
        C.preflight, "verify_prepared_inputs", lambda actual: calls.append("replay") or "manifest digest"
    )
    monkeypatch.setattr(C, "_public_control", lambda actual: ("/public/interface", "digest"))
    monkeypatch.setattr(
        C.tooling_readiness, "public_probe_session", lambda *a, **kw: nullcontext("print('selected public tool')")
    )
    return value, calls


def _fake_provider(prepared, monkeypatch, *, mutate=None):
    value, calls = prepared

    def run(*args, **kwargs):
        calls.append("provider")
        assert args[:2] == (value.workspace, value.run_dir)
        assert kwargs["codex_binary"] == value.request.options.codex_binary
        assert kwargs["runtime_binds"].__self__ is value.selected_codex_runtime
        assert kwargs["continue_session"] is False and "effective_model" not in kwargs
        probe = (value.run_dir / "client_canary_probe.py").read_bytes()
        command = "python3 -I -B -c " + shlex.quote(probe.decode())
        assert command in kwargs["prompt"]
        assert not (value.workspace / "client_canary_probe.py").exists()
        rounds = value.run_dir / "rounds"
        rounds.mkdir()
        rows = _events(command)
        summary = {"usage_complete": True, "turns_usage_reported": 1, "exit_code": 0, "timed_out": False}
        (value.workspace / C._REPORT).write_text(C._PUBLIC + C._DONE)
        if mutate is not None:
            mutate(value, rows, summary)
        (rounds / "round_00.codex_events.raw.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        (rounds / "round_00.codex_summary.json").write_text(json.dumps(summary))
        return 0, rounds / "transcript"

    monkeypatch.setattr(C.codex_agent, "run_round", run)


def test_selected_fake_client_reaches_ordinary_complete_replays_without_authoring(prepared, monkeypatch):
    value, calls = prepared
    original = (value.workspace / "TASK.md").read_bytes()
    _fake_provider(prepared, monkeypatch)
    assert C.execute(value) == 0
    assert calls == ["replay", "replay", "provider", "replay", "replay"]
    assert (value.workspace / "TASK.md").read_bytes() == original
    result = json.loads((value.run_dir / "client_canary_result.json").read_bytes())
    assert result["observed_ok"] is True and result["formal_complete"] is False
    assert result["completed_command_item"] == "owned-command"
    assert "compiler_qualified" not in result and "all_pass" not in result
    assert (value.run_dir / "client_canary_result.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "change", ["markers_only", "failed_tool", "usage", "missing_exit", "task", "submission", "probe", "post_replay"]
)
def test_fake_marker_or_partial_observation_cannot_succeed(prepared, monkeypatch, change):
    value, calls = prepared

    def mutate(actual, rows, summary):
        if change == "markers_only":
            rows.clear()
        elif change == "failed_tool":
            rows[1]["item"]["exit_code"] = 1
        elif change == "usage":
            summary["usage_complete"] = False
        elif change == "missing_exit":
            summary.pop("exit_code")
        elif change == "task":
            (actual.workspace / "TASK.md").write_text("changed task")
        elif change == "submission":
            (actual.workspace / "submission/candidate.py").write_text("changed")
        elif change == "probe":
            (actual.run_dir / "client_canary_probe.py").write_text("changed")
        else:
            monkeypatch.setattr(
                C.preflight,
                "verify_prepared_inputs",
                lambda *a: (_ for _ in ()).throw(ValueError("private fake path /secret")),
            )

    _fake_provider(prepared, monkeypatch, mutate=mutate)
    assert C.execute(value) == 1
    result = (value.run_dir / "client_canary_result.json").read_text()
    assert json.loads(result)["observed_ok"] is False
    assert "/secret" not in result


@pytest.mark.parametrize("change", [None, "source", "missing_owner", "duplicate", "ungranted"])
def test_public_control_joins_exact_original_root_not_duplicate_granted_answers(tmp_path, monkeypatch, change):
    from merlin_experiments.phase1 import corpus_inputs as CI

    from merlin.targetgen.sandbox import bwrap as BW

    public, policy, frozen, original = (
        tmp_path / "public",
        tmp_path / "policy/0",
        tmp_path / "frozen",
        tmp_path / "original",
    )
    for root in (public, policy, frozen):
        (root / "case").mkdir(parents=True)
        (root / "case/capsule.interface.mlir").write_text("public interface bytes")
    (tmp_path / "source_commitments.json").write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "descriptor_cohort",
                "sources": [{"role": "corpus", "original": str(original), "staged": "policy/0"}],
            }
        )
    )

    def caps(root, **kw):
        chosen = public if root == public else policy
        rows = [{"name": "case", "__dir__": str(chosen / "case"), "interface_mlir": "capsule.interface.mlir"}]
        if root != public and change == "missing_owner":
            return []
        if root != public and change == "duplicate":
            return rows + rows
        return rows

    monkeypatch.setattr(CI, "discover_capsules", caps)
    # A masked same-byte copy in an unrelated grant must never select the path.
    grants = [("other", Path("/masked/duplicate"), tmp_path / "answer-copy")]
    if change != "ungranted":
        grants.append(("original", original, frozen))
    monkeypatch.setattr(BW, "_snapshot_grants", lambda *a: ({}, grants))
    calls = []
    monkeypatch.setattr(
        BW,
        "snapshot_input_paths",
        lambda ws, bundle, members, **kw: calls.extend(members) or [frozen / "case/capsule.interface.mlir"],
    )
    if change == "source":
        (frozen / "case/capsule.interface.mlir").write_text("changed source")
    value = SimpleNamespace(
        public_root=public,
        contract_root=tmp_path / "contract",
        workspace=tmp_path / "workspace",
        bundle={},
        request=SimpleNamespace(context=SimpleNamespace(repo=tmp_path)),
    )
    if change is not None:
        with pytest.raises(ValueError):
            C._public_control(value)
    else:
        assert C._public_control(value) == (
            str(original / "case/capsule.interface.mlir"),
            hashlib.sha256(b"public interface bytes").hexdigest(),
        )
        assert calls == [original / "case/capsule.interface.mlir"]


def test_ordinary_tooling_probe_keeps_original_frozen_facts_workspace(tmp_path, monkeypatch):
    from merlin_experiments import frozen_python
    from merlin_experiments.phase1 import tooling_readiness as T

    from merlin.targetgen.sandbox import toolchain as TC

    ws = tmp_path / "workspace"
    ws.mkdir()
    facts = tmp_path / "selected-facts"
    context = SimpleNamespace(repo=tmp_path, descriptor=tmp_path / "descriptor", target="synthetic")
    observations = []
    monkeypatch.setattr(
        T.BW, "frozen_selected_rtl_facts", lambda actual, *a, **kw: observations.append(actual) or facts
    )
    spec = SimpleNamespace(channel="channel", shims=(), module="synthetic.isa_tools", log="log")
    monkeypatch.setattr(T.TR, "brokers_for", lambda tools: (spec,))
    monkeypatch.setattr(T, "_asm_probe", lambda context: "explicit-public-mnemonic")
    monkeypatch.setattr(frozen_python, "inherited_python_command", lambda argv: argv)
    processes = []
    monkeypatch.setattr(
        T.subprocess, "Popen", lambda argv, **kw: processes.append(argv) or SimpleNamespace(wait=lambda **kw: 0)
    )
    monkeypatch.setattr(
        T.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="AUTHORING_AND_BROKER_ROUNDTRIPS_OK", stderr=""),
    )
    monkeypatch.setattr(TC, "sandbox_env", lambda *a: "fixed")
    monkeypatch.setattr(T.BW, "full_argv", lambda *a: [])
    assert T._live_probe(context, object(), ws, {}, ("isa_tools",))["ok"] is True
    assert observations == [ws]
    assert processes[0][-2:] == ["--rtl-facts", str(facts)]


@pytest.mark.parametrize("refusal", [None, 3])
def test_controller_canary_dispatch_stops_before_ordinary_authoring(tmp_path, monkeypatch, refusal):
    from merlin_experiments.phase1 import authoring, controller, runtime_environment, task_staging, timing

    options = _options(tmp_path)
    context = SimpleNamespace(readback_policy=None, descriptor=tmp_path / "descriptor", target="synthetic")
    marker = object()
    calls = []
    monkeypatch.setattr(S, "validate_options", lambda options: None)
    monkeypatch.setattr(timing, "requires_chipyard_timing", lambda *a: False)
    monkeypatch.setattr(
        task_staging, "callbacks", lambda config: SimpleNamespace(resolved_tools=lambda: (), stage_task=lambda *a: None)
    )
    monkeypatch.setattr(
        runtime_environment, "prepare_runtime_environment", lambda *a, **kw: SimpleNamespace(refusal=None, account={})
    )
    monkeypatch.setattr(runtime_environment, "applied_environment", lambda *a: nullcontext())
    monkeypatch.setattr(
        S,
        "prepare",
        lambda request, *a, **kw: calls.append(request.options) or (marker if refusal is None else refusal),
    )
    monkeypatch.setattr(C, "execute", lambda actual: calls.append(actual) or 0)
    monkeypatch.setattr(authoring, "execute", lambda *a, **kw: pytest.fail("authoring reached"))
    result = controller.run(
        context,
        options,
        bundle_manifest=tmp_path / "input_bundle_manifest.yaml",
        bundle_id="selected",
        oracle_timing=tmp_path / "timing",
        base_environment={},
    )
    assert result == (0 if refusal is None else refusal)
    assert calls == ([options, marker] if refusal is None else [options])


@pytest.mark.parametrize("canary", [True, False])
def test_installed_cli_forwards_same_explicit_client_selection(tmp_path, monkeypatch, canary):
    from merlin_experiments.phase1 import __main__ as CLI
    from merlin_experiments.phase1 import context, controller

    options = _options(tmp_path)
    manifest = tmp_path / "input_bundle_manifest.yaml"
    manifest.write_text("bundle_id: selected")
    seen = []
    monkeypatch.setattr(context, "load_context", lambda *a, **kw: "context")
    monkeypatch.setattr(controller, "run", lambda actual, selected, **kw: seen.append(selected) or 3)
    argv = [
        "--run-id",
        "canary" if canary else "ordinary",
        "--model",
        "explicit",
        "--driver",
        "codex",
        "--descriptor",
        str(tmp_path / "descriptor"),
        "--repo",
        str(tmp_path),
        "--bundle-manifest",
        str(manifest),
        "--bundle",
        "selected",
        "--oracle-timing",
        str(tmp_path / "timing"),
        "--corpus-seal",
        options.corpus_seal,
        "--codex-binary",
        options.codex_binary,
        "--codex-auth-source",
        options.codex_auth_source,
        "--codex-home-root",
        options.codex_home_root,
    ]
    if canary:
        argv += ["--codex-canary", "--round-timeout", "60"]
    assert CLI.main(argv) == 3
    assert seen[0].codex_canary is canary
    assert (seen[0].codex_binary, seen[0].codex_auth_source, seen[0].codex_home_root) == (
        options.codex_binary,
        options.codex_auth_source,
        options.codex_home_root,
    )
    with pytest.raises(SystemExit):
        CLI.main([arg for i, arg in enumerate(argv) if arg != "--model" and (i == 0 or argv[i - 1] != "--model")])


@pytest.mark.parametrize("change", ["public_hash", "protected_read", "tool_failure"])
def test_actual_owned_probe_logic_refuses_before_marker(tmp_path, monkeypatch, change):
    public = tmp_path / "public.interface"
    public.write_bytes(b"synthetic public interface")
    home = tmp_path / "synthetic-client-home"
    home.mkdir()
    protected = tmp_path / "synthetic-protected-answer"
    if change == "protected_read":
        protected.write_bytes(b"synthetic protected control")
    expected = hashlib.sha256(public.read_bytes()).hexdigest()
    if change == "public_hash":
        expected = "0" * 64
    tool = "raise AssertionError('synthetic public tool failure')" if change == "tool_failure" else "pass"
    body = C._probe((str(public), expected), (protected,), tool, ())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setattr(C.os, "access", lambda *a: False)
    with pytest.raises(AssertionError):
        exec(compile(body, "<owned-probe-control>", "exec"), {})
    assert not (tmp_path / C._REPORT).exists()
