"""Explicit provider execution; synthetic local CLI/processes, no paid or hardware calls."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.providers import execution as E

from merlin.common.paths import module_source_path, python_import_roots


def _config(root, driver="codex", tools=lambda: ()):
    context = InvocationContext(root, root / "descriptor.yaml", root, "fixture", root, root, root, ())
    return E.ExecutionConfig(
        context, E.ProviderConfig(driver, "bedrock", "sub", "background"), tools, root / "timing.json", 7
    )


@pytest.mark.parametrize("timeout", [False, True])
def test_cold_execution_real_fake_cli_and_cleanup_outside_checkout(tmp_path, timeout):
    # Actual bash -> fake Claude -> descendant process, real stream supervisor and
    # broker lifetime. No provider SDK, descriptor, native controller or bwrap is run.
    program = r"""
import importlib.abc, json, os, pathlib, subprocess, sys
sys.path[:0] = json.loads(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'_common', 'run_baseline_qa_loop', 'sandbox_toolchain', 'run_agent_experiment'}:
            raise AssertionError('native import: ' + fullname)
sys.meta_path.insert(0, NoNative())
before = dict(os.environ)
from merlin_experiments.phase1.providers import execution as E
from merlin_experiments.phase1.context import InvocationContext
assert dict(os.environ) == before
assert 'merlin_experiments.phase1.providers.bedrock_agent' not in sys.modules
from merlin_experiments.phase1.providers import agent_bridge as bridge
root = pathlib.Path.cwd()
ws, run = root / 'workspace', root / 'run'
ws.mkdir(); run.mkdir()
(ws / 'TASK.md').write_text('sealed local task\n')
slow = json.loads(sys.argv[2])
fake = root / 'claude'
fake.write_text('#!' + sys.executable + '\nimport os,sys,time,json\n'
    "print(json.dumps({'type':'assistant','cwd':os.getcwd(),'argv':sys.argv[1:]}),flush=True)\n"
    + ('time.sleep(300)\n' if slow else 'sys.exit(7)\n'))
fake.chmod(0o755)
os.environ['PATH'] = str(root) + os.pathsep + os.environ['PATH']
bridge.bridged_name = lambda *a: None
bridge.claude_env = lambda *a: {}
bridge.claude_model_name = lambda model, **kw: model
context = InvocationContext(root, root/'missing-descriptor', root, 'fixture', root, root, root, ())
config = E.ExecutionConfig(context, E.ProviderConfig('claudecode'), lambda: (), root/'timing')
children=[]
def start(ws, config):
    (ws/'.qa_channel').mkdir()
    p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])
    children.append(p)
    return [p]
E.FL.start_brokers=start
E.FL.BROKER_GRACE_SECONDS=0.02
E.sandbox_command=lambda inner, *a, **kw: inner
try:
    result = E.launch(ws, run, 'fixture', 'high', 'bwrap', {}, 2, 1 if slow else 10, config=config)
except subprocess.TimeoutExpired:
    assert slow
else:
    assert not slow and result == (7, run/'rounds/round_02.transcript.jsonl')
assert children and all(p.poll() is not None for p in children)
raw=(run/'rounds/round_02.stream.raw.jsonl').read_text()
stamped=json.loads((run/'rounds/round_02.transcript.jsonl').read_text())
assert stamped.pop('arrived_at')
assert stamped == json.loads(raw)
assert stamped['cwd'] == str(ws)
assert stamped['argv'][:4] == ['--print','--model','fixture','--effort']
assert (ws/'.qa_channel/STOP').exists()
print('provider-execution-qualified')
"""
    dependencies = [p for p in sys.path if p and Path(p).is_dir() and "site-packages" in p]
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            json.dumps([*map(str, python_import_roots()), *dependencies]),
            json.dumps(timeout),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "provider-execution-qualified"


@pytest.mark.parametrize("driver", ["codex", "opencode", "converse"])
@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt, SystemExit])
def test_actual_dispatch_configuration_order_and_primary_cleanup(tmp_path, monkeypatch, driver, failure):
    from merlin_experiments.phase1.providers import agent_bridge

    from merlin.targetgen import target_experiment

    events = []
    config = _config(tmp_path, driver, lambda: events.append("tools") or ("cca_tools",))
    ws = tmp_path / "workspace"
    ws.mkdir()
    public, policy, contract = (tmp_path / p for p in ("public", "policy", "contract"))
    kwargs = dict(config=config, capsules_root=public, policy_root=policy, contract=contract)
    with pytest.raises(RuntimeError, match="sealed task is missing"):
        E.launch(ws, tmp_path, "fixture", "high", "bwrap", {}, 3, 7, **kwargs)
    assert events == []
    (ws / "TASK.md").write_text("sealed")
    target = object()
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: target)
    monkeypatch.setattr(agent_bridge, "bridged_name", lambda *args: None)
    owned = object()

    def start(workspace, observed):
        events.append("start")
        assert workspace == ws and observed.context == config.context
        assert observed.tools == ("cca_tools",) and observed.sim_max_jobs == 7
        assert (observed.capsules_root, observed.policy_root, observed.contract) == (public, policy, contract)
        return owned

    primary = failure("provider failed") if failure else None

    def run(*args, **kw):
        events.append("provider")
        assert args == (ws, tmp_path, "fixture", {}, target, "bwrap", 3, 7)
        assert kw["subagent_model"] == "sub" and kw["background_model"] == "background"
        if driver != "converse":
            assert kw["effort"] == "high"
            assert kw["sandbox_command"].func is E.sandbox_command
            assert kw["sandbox_command"].keywords == {"context": config.context}
        if driver == "codex":
            assert kw["continue_session"] is True
        if primary:
            raise primary
        return 13, tmp_path / "transcript"

    def stop(workspace, brokers, *, primary_error):
        events.append("stop")
        assert workspace == ws and brokers is owned and primary_error is primary

    module = importlib.import_module(E._DRIVER_MODULES[driver])
    monkeypatch.setattr(module, "run_round", run)
    monkeypatch.setattr(E.FL, "start_brokers", start)
    monkeypatch.setattr(E.FL, "stop_brokers", stop)
    if failure:
        with pytest.raises(failure) as raised:
            E.launch(ws, tmp_path, "fixture", "high", "bwrap", {}, 3, 7, continuous=True, **kwargs)
        assert raised.value is primary
    else:
        assert E.launch(ws, tmp_path, "fixture", "high", "bwrap", {}, 3, 7, continuous=True, **kwargs) == (
            13,
            tmp_path / "transcript",
        )
    assert events == ["tools", "start", "provider", "stop"]


def test_provider_owner_is_private_and_byte_drift_invalidates_existing_source_receipt(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import source_inputs as SI
    from merlin_experiments.spec import SpecError

    from merlin.common import access

    package = tmp_path / "phase1"
    shutil.copytree(
        module_source_path("merlin_experiments.phase1").parent, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    original = SI._source

    def copied_source(module):
        if module == "merlin_experiments.phase1":
            return package / "__init__.py"
        if module.startswith("merlin_experiments.phase1."):
            path = package.joinpath(*module.split(".")[2:])
            return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        return original(module)

    monkeypatch.setattr(SI, "_source", copied_source)
    arguments = dict(repo=tmp_path, entrypoint=tmp_path / "transport.py")
    record = SI.record(**arguments)
    assert "phase1:source:providers/execution.py" in record["inputs"]
    for key in ("arrival_stamp", "sandbox_bwrap", "sandbox_toolchain", "answer_surface_policy"):
        assert f"phase1:startup:{key}" in record["inputs"]
    SI.verify(record, **arguments)
    member = package / "providers/execution.py"
    member.write_text(member.read_text() + "\n# changed provider policy\n")
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **arguments)
    assert any(
        item.origin == "grader"
        and item.directory
        and access.module_matches("merlin_experiments.phase1.providers.execution", item.identity)
        for item in access.MODULE_ACCESS
    )


def test_sandbox_composition_preserves_original_order_and_payload(tmp_path, monkeypatch):
    from merlin.targetgen import target_experiment
    from merlin.targetgen.sandbox import bwrap, toolchain

    events = []
    config = _config(tmp_path)
    workspace, bundle = tmp_path / "workspace", {"allowed": []}
    target = object()

    def observe(name, result):
        def call(*args, **kwargs):
            events.append((name, args, kwargs))
            return result

        return call

    monkeypatch.setattr(bwrap, "base_argv", observe("base", ["bwrap", "base"]))
    monkeypatch.setattr(bwrap, "claude_runtime_binds", observe("runtime", ["runtime"]))
    monkeypatch.setattr(target_experiment, "load_target_experiment", observe("target", target))
    monkeypatch.setattr(toolchain, "toolchain_binds", observe("toolchain", ["tools"]))
    monkeypatch.setattr(bwrap, "reapply_bundle_snapshot", observe("snapshot", ["frozen"]))
    monkeypatch.setattr(bwrap, "apply_final_answer_masks", observe("mask", ["masked"]))
    monkeypatch.setattr(toolchain, "sandbox_env", observe("environment", "export X=1;"))
    monkeypatch.setattr(bwrap, "compose_command", observe("compose", "composed"))
    assert (
        E.sandbox_command("printf '%s' '(value)'", workspace, bundle, ["extra"], context=config.context) == "composed"
    )
    assert [event[0] for event in events] == [
        "base",
        "runtime",
        "target",
        "toolchain",
        "snapshot",
        "target",
        "mask",
        "target",
        "environment",
        "compose",
    ]
    assert events[0][1:] == ((workspace, bundle), {"repo": tmp_path})
    expected = [
        "bwrap",
        "base",
        "runtime",
        "tools",
        "--unsetenv",
        "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT",
        "--unsetenv",
        "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED",
        "--unsetenv",
        "MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD",
        "extra",
    ]
    assert events[4][1:] == ((expected, workspace, bundle), {"repo": tmp_path})
    assert events[6][1:] == ((["frozen"], target, workspace, bundle), {"repo": tmp_path})
    assert events[8][1] == (target, workspace)
    payload = "export X=1; printf '%s' '(value)'"
    assert events[-1][1] == (
        ["masked"],
        " bash -c '" + payload.replace("'", "'\\''") + "'",
        workspace,
    )


def test_execution_implementation_mask_covers_broad_candidate_grant(tmp_path, monkeypatch):
    from merlin.common import access
    from merlin.targetgen.sandbox import bwrap

    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    private = tmp_path / "packages/merlin-experiments/src/merlin_experiments/phase1/providers/execution.py"
    private.parent.mkdir(parents=True)
    private.write_text("# trusted provider execution\n")
    monkeypatch.setattr(access, "sys", SimpleNamespace(path=[], prefix=str(tmp_path / "python"), modules={}))
    monkeypatch.setattr(surfaces_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(surfaces_module, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(surfaces_module, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(surfaces_module, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    policy = SimpleNamespace(
        target="fixture",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    surfaces = surfaces_module.answer_surfaces(policy)
    grants = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert any(private.is_relative_to(surface.path) for surface in bwrap.coverage_gap(grants, surfaces))
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(grants, surfaces), surfaces) == []
