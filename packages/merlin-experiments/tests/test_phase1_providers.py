"""Installed provider protocols: real local processes, never a paid provider."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.providers import agent_bridge as bridge
from merlin_experiments.phase1.providers import bedrock_agent as bedrock

from merlin.common.paths import module_source_path, python_source_dir


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _system(target):
    tree = ast.parse(inspect.getsource(bedrock.run_round))
    value = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "system" for t in node.targets)
    )
    return eval(compile(ast.Expression(value), "provider-system", "eval"), {"te": SimpleNamespace(target=target)})


def test_legacy_bedrock_prompt_and_tools_exact_parity():
    # SHA256 of canonical JSON evaluated from checkpoint 627b4d96f's unchanged
    # legacy tool/schema and system expression. No network or model is involved.
    assert _digest(bedrock._tools("atlas")) == "2124ea33985ef61d766d1f3c8a378450b3a70d8d8e0966bd4a82165e42682b17"
    assert _digest(_system("atlas")) == "f096c2b75432b78dcdf4fbf297152201715749a4cab883db631d8d5d8623c02e"


@pytest.mark.parametrize("target", ["fixture_alpha", "fixture_beta"])
def test_bedrock_prompt_uses_invocation_target_without_shared_mutation(target):
    tools = bedrock._tools(target)
    text = json.dumps([tools, _system(target)])
    assert f"{target}_opt.py" in text
    assert f"{target}-SPECIFIC lowering" in text
    assert "atlas" not in text
    tools["tools"].clear()
    assert bedrock._tools(target)["tools"]


def test_proxy_defaults_are_physical_checkout_only(monkeypatch, tmp_path):
    from merlin.common import paths

    monkeypatch.setattr(paths, "checkout_root", lambda: None)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    for variable, getter in (
        ("MERLIN_PROXY_CONFIG", bridge.proxy_config_path),
        ("MERLIN_PROXY_EXECUTABLE", bridge.proxy_venv_python),
    ):
        monkeypatch.delenv(variable, raising=False)
        with pytest.raises(RuntimeError, match=variable):
            getter()
        monkeypatch.setenv(variable, str(tmp_path / variable))
        assert getter() == tmp_path / variable
    monkeypatch.setattr(paths, "checkout_root", lambda: tmp_path / "physical")
    monkeypatch.delenv("MERLIN_PROXY_CONFIG")
    monkeypatch.delenv("MERLIN_PROXY_EXECUTABLE")
    assert (
        bridge.proxy_config_path() == tmp_path / "physical/merlin/experiments/capsule_bench/proxy/litellm_config.yaml"
    )
    assert bridge.proxy_venv_python() == tmp_path / "physical/build/proxy-venv/bin/litellm"


def test_verified_proxy_resource_and_explicit_override_remain_pinned(tmp_path, monkeypatch):
    relative = "merlin/experiments/capsule_bench/proxy/litellm_config.yaml"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("model_list: []\n")
    receipt = {"files": {relative: hashlib.sha256(path.read_bytes()).hexdigest()}}
    env = {}
    record = bridge.bind_frozen_proxy_config(tmp_path, receipt, env)
    assert record == {"path": str(path), "sha256": receipt["files"][relative], "role": "verified_snapshot"}
    assert bridge.bind_frozen_proxy_config(tmp_path, receipt, env) == record
    assert bridge.bind_frozen_proxy_config(tmp_path, {"files": {}}, {}) is None
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert bridge.proxy_config_path() == path
    path.write_text("changed: true\n")
    with pytest.raises(RuntimeError, match="changed"):
        bridge.proxy_config_path()
    with pytest.raises(RuntimeError, match="changed"):
        bridge._known_vendors()
    with pytest.raises(RuntimeError, match="changed"):
        bridge.bind_frozen_proxy_config(tmp_path, receipt, {})
    with pytest.raises(RuntimeError, match="changed"):
        bridge.bind_frozen_proxy_config(tmp_path, receipt, env)
    override = tmp_path / "operator.yaml"
    override.write_text("model_list: []\n")
    env = {"MERLIN_PROXY_CONFIG": str(override)}
    record = bridge.bind_frozen_proxy_config(tmp_path, receipt, env)
    assert record["role"] == "operator_override"
    assert record["path"] == str(override)
    override.write_text("operator changed\n")
    with pytest.raises(RuntimeError, match="changed"):
        bridge.bind_frozen_proxy_config(tmp_path, receipt, env)


@pytest.mark.parametrize("name", ["codex_agent", "opencode_agent"])
def test_bwrap_requires_explicit_caller_policy(tmp_path, monkeypatch, name):
    import importlib

    driver = importlib.import_module("merlin_experiments.phase1.providers." + name)
    if name == "codex_agent":
        monkeypatch.setattr(driver, "cli_version", lambda *args: "offline")
    with pytest.raises(ValueError, match="sandbox_command"):
        driver.run_round(tmp_path, tmp_path / "run", "fixture", {}, SimpleNamespace(target="fixture"), "bwrap", 0, 1)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("ambient", [None, "/ambient/codex"])
def test_codex_invocation_configuration_is_local(tmp_path, monkeypatch, explicit, ambient):
    from merlin_experiments.phase1.providers import codex_agent as driver

    from merlin.common import artifacts

    if ambient is None:
        monkeypatch.delenv("CODEX_BIN", raising=False)
    else:
        monkeypatch.setenv("CODEX_BIN", ambient)
    before = dict(os.environ)
    selected = tmp_path / "selected codex"
    root = tmp_path / "stage-homes"
    cache = tmp_path / "global-cache"
    expected_binary = str(selected) if explicit else (ambient or "codex")
    homes, versions, cache_calls = [], [], []

    def cache_dir(namespace):
        assert not explicit, "explicit home root must not consult global cache"
        cache_calls.append(namespace)
        return cache

    def prepare(home, **kwargs):
        homes.append(home)
        assert kwargs == {"model": "offline", "effort": ""}
        return {}

    class PolicyReached(Exception):
        pass

    def sandbox_command(inner, ws, bundle, *, extra_binds):
        assert dict(os.environ) == before
        assert shlex.split(inner)[0] == expected_binary
        assert extra_binds == ["synthetic-bind"]
        raise PolicyReached

    monkeypatch.setattr(artifacts, "cache_dir", cache_dir)
    monkeypatch.setattr(driver, "cli_version", lambda binary: versions.append(binary) or "offline")
    monkeypatch.setattr(driver, "prepare_codex_home", prepare)
    monkeypatch.setattr(driver, "codex_runtime_binds", lambda home: ["synthetic-bind"])
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch a provider"))
    kwargs = {"codex_binary": selected, "codex_home_root": root} if explicit else {}
    with pytest.raises(PolicyReached):
        driver.run_round(
            tmp_path / "workspace",
            tmp_path / "run",
            "offline",
            {},
            None,
            "bwrap",
            3,
            1,
            effective_model="offline",
            sandbox_command=sandbox_command,
            **kwargs,
        )
    assert homes == [(root if explicit else cache) / "run_r03"]
    assert cache_calls == ([] if explicit else ["codex_home"])
    assert versions == [expected_binary]
    assert dict(os.environ) == before


def test_codex_continuation_retains_explicit_binary_and_home(tmp_path, monkeypatch):
    from merlin_experiments.phase1.providers import codex_agent as driver

    from merlin.common import artifacts

    fake = tmp_path / "offline-cli"
    fake.write_text(
        "#!" + sys.executable + "\nimport json,sys\nsys.stdin.read()\n"
        "for event in [{'type':'thread.started','thread_id':'offline'},"
        "{'type':'turn.started'},{'type':'turn.completed','usage':"
        "{'input_tokens':1,'cached_input_tokens':0,'output_tokens':1}}]:\n"
        " print(json.dumps(event),flush=True)\n"
    )
    fake.chmod(0o700)
    monkeypatch.setenv("CODEX_BIN", "/must-not-run/ambient-codex")
    before = dict(os.environ)
    homes, binds, commands = [], [], []
    monkeypatch.setattr(driver, "cli_version", lambda binary: "offline")
    monkeypatch.setattr(driver, "_CONTINUE_MAX_TURNS", 2)
    monkeypatch.setattr(driver, "_CONTINUE_MIN_S", 0)
    monkeypatch.setattr(driver, "prepare_codex_home", lambda home, **kw: homes.append(home) or {})
    monkeypatch.setattr(driver, "codex_runtime_binds", lambda home: binds.append(home) or [])
    monkeypatch.setattr(artifacts, "cache_dir", lambda *a: pytest.fail("must not consult global cache"))
    original_popen = subprocess.Popen

    def offline_only(argv, **kwargs):
        if "--pid" in argv:
            raise OSError("resource sampler intentionally disabled in fixture")
        assert argv[0] == "bash" and str(argv[1]).endswith(".sh")
        return original_popen(argv, **kwargs)

    def synthetic_policy(inner, ws, bundle, *, extra_binds):
        command = shlex.split(inner)
        assert command[0] == str(fake)
        assert dict(os.environ) == before
        commands.append(command)
        # Execute only the harmless fixture, not a real sandbox or provider.
        return inner

    monkeypatch.setattr(driver.subprocess, "Popen", offline_only)
    rc, transcript = driver.run_round(
        tmp_path / "workspace",
        tmp_path / "run",
        "offline",
        {},
        None,
        "bwrap",
        0,
        10,
        effective_model="offline",
        continue_session=True,
        codex_binary=fake,
        codex_home_root=tmp_path / "local-homes",
        sandbox_command=synthetic_policy,
    )
    assert rc == 0
    assert len(commands) == 2
    assert commands[0][1:3] == ["exec", "--json"]
    assert commands[1][1:3] == ["exec", "resume"]
    assert commands[1][-2:] == ["offline", "-"]
    assert homes == [tmp_path / "local-homes/run_r00"]
    assert binds == homes * 2
    decisions = [
        row
        for line in transcript.read_text().splitlines()
        if (row := json.loads(line)).get("type") == "codex_session_turn"
    ]
    assert [row["continuing"] for row in decisions] == [True, False]
    assert decisions[-1]["stopped_because"] == "turn cap 2"
    assert dict(os.environ) == before


def test_real_snapshot_config_resource_survives_live_source_change(tmp_path, monkeypatch):
    from merlin_experiments import source_snapshot

    source, snapshot, output = (tmp_path / name for name in ("source", "snapshot", "output"))
    relative = "merlin/experiments/capsule_bench/proxy"
    source_config = source / relative / "litellm_config.yaml"
    source_config.parent.mkdir(parents=True)
    source_config.write_text("model_list: []\n")
    output.mkdir()
    try:
        source_snapshot.create(
            source, snapshot, output_root=output, source_roots=(relative,), python_roots=(), legacy_roots=()
        )
        receipt = source_snapshot.verify(snapshot)
        source_config.write_text("changed_live: true\n")
        env = {}
        record = bridge.bind_frozen_proxy_config(snapshot, receipt, env)
        assert record["role"] == "verified_snapshot"
        assert record["sha256"] == hashlib.sha256(b"model_list: []\n").hexdigest()
        assert env["MERLIN_PROXY_CONFIG"] == str(snapshot / relative / "litellm_config.yaml")
        assert source_snapshot.verify(snapshot) == receipt
    finally:
        if snapshot.exists():
            snapshot.chmod(0o700)
            for path in snapshot.rglob("*"):
                if not path.is_symlink():
                    path.chmod(0o700 if path.is_dir() else 0o600)


def test_installed_provider_runs_real_fake_cli_without_native_controller(tmp_path):
    installed = tmp_path / "installed"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(module_source_path("merlin_experiments").parent, installed / "merlin_experiments", ignore=ignore)
    # Core dependency closure, not the source checkout or any native harness.
    for relative in (
        "merlin/__init__.py",
        "merlin/common/__init__.py",
        "merlin/common/paths.py",
        "merlin/common/arrival_stamp.py",
    ):
        destination = installed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(python_source_dir() / relative, destination)
    fake = tmp_path / "fake-cli"
    fake.write_text(
        "#!" + sys.executable + "\nimport json,sys\n"
        "if '--version' in sys.argv: print('fake offline cli'); raise SystemExit\n"
        "sys.stdin.read()\n"
        "events=[{'type':'thread.started','thread_id':'offline'}, {'type':'turn.started'},"
        "{'type':'turn.completed','usage':{'input_tokens':10,'cached_input_tokens':3,'output_tokens':2}}]\n"
        "for event in events: print(json.dumps(event),flush=True)\n"
    )
    fake.chmod(0o700)
    program = r"""
import importlib.abc,json,os,pathlib,sys
sys.path[:0] = [sys.argv[1], *json.loads(sys.argv[2])]
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in {'_common','run_baseline_qa_loop','codex_agent','opencode_agent'}, fullname
sys.meta_path.insert(0,NoNative())
before = dict(os.environ)
from merlin_experiments.phase1.providers import codex_agent, opencode_agent, bedrock_agent, agent_bridge, model_tiers
from merlin.common.paths import checkout_root
assert checkout_root() is None
assert dict(os.environ) == before
assert 'boto3' not in sys.modules
root = pathlib.Path.cwd(); ws=root/'workspace'; ws.mkdir(); (ws/'TASK.md').write_text('offline task')
os.environ['CODEX_BIN'] = sys.argv[3]
rc, transcript = codex_agent.run_round(ws,root/'run','offline',{},None,'none',0,10,effective_model='offline')
assert rc == 0
rows = [json.loads(line) for line in transcript.read_text().splitlines()]
usage = next(row['message']['usage'] for row in rows
             if row.get('type')=='assistant' and row.get('message',{}).get('usage'))
assert usage['input_tokens'] == 7 and usage['cache_read_input_tokens'] == 3
rounds = transcript.parent
assert (rounds/'round_00.codex_events.raw.jsonl').read_text().count('\n') == 3
assert (rounds/'round_00.codex_events.timestamped.jsonl').is_file()
assert pathlib.Path(codex_agent.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            str(installed),
            json.dumps(sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})),
            str(fake),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
