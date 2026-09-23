"""Installed-shaped staging and real stdlib IPC; fake brokers supply no oracle grades."""

from __future__ import annotations

import ast
import importlib
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import yaml

from merlin.common.paths import module_source_path, python_source_dir
from merlin.targetgen.tool_registry import COMMON_CLIENTS, brokers_for, public_client_modules


def test_public_catalog_uses_phase_tools_with_unchanged_workspace_names():
    clients = {
        *COMMON_CLIENTS,
        *(client for broker in brokers_for(("isa_tools", "cca_tools")) for client in broker.shims),
    }
    prefix = "merlin_experiments.phase1.tools."
    assert clients == {
        (prefix + "selfcheck", "agent_selfcheck.py"),
        (prefix + "simjob", "simjob.py"),
        (prefix + "await_verdict", "await_verdict.py"),
        (prefix + "isa", "isa_tools.py"),
        (prefix + "cca", "cca_contract.py"),
        (prefix + "cca", "action_catalog.py"),
    }
    from merlin.common.access import is_harness_module

    for module in public_client_modules():
        # Public file staging is not permission to import the host package.
        assert is_harness_module(module)
        source = module_source_path(module)
        assert source.parent == module_source_path(prefix.rstrip(".")).parent
        assert source.name == module.rsplit(".", 1)[-1] + ".py"


@pytest.fixture
def staged(tmp_path):
    installed, workspace = tmp_path / "installed", tmp_path / "workspace"
    workspace.mkdir()
    # Actual startup files, not substitute packages. No site initialization or
    # editable checkout root is available to the fresh staging interpreter.
    for relative in (
        "merlin/__init__.py",
        "merlin/common/__init__.py",
        "merlin/common/paths.py",
        "merlin/common/tree_hash.py",
    ):
        destination = installed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(python_source_dir() / relative, destination)
    for module in ("merlin_experiments", "merlin_experiments.phase1", "merlin_experiments.phase1.tools"):
        destination = installed / module.replace(".", "/") / "__init__.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(module_source_path(module), destination)
    shutil.copyfile(module_source_path("merlin_experiments.spec"), installed / "merlin_experiments/spec.py")
    # Host-side package discovery requires the experiments distribution's declared
    # YAML/schema dependencies. No editable initialization runs; copied clients
    # below still execute with -I -S and no package search roots whatsoever.
    dependency_roots = sorted({str(Path(owner.__file__).resolve().parent.parent) for owner in (jsonschema, yaml)})
    names = public_client_modules()
    for module in names:
        shutil.copyfile(module_source_path(module), destination.parent / (module.rsplit(".", 1)[-1] + ".py"))
    private = installed / "private_host_evaluator.py"
    private.write_text("PRIVATE_HOST_SENTINEL = 'must not stage'\n")
    clients = [
        *COMMON_CLIENTS,
        *(client for broker in brokers_for(("isa_tools", "cca_tools")) for client in broker.shims),
    ]
    owner = module_source_path("merlin_experiments.phase1.feedback.lifecycle")
    helper = next(
        node
        for node in ast.parse(owner.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "stage_client"
    )
    program = """
import json, pathlib, shutil, sys
sys.path.insert(0, sys.argv[1])
sys.path.extend(json.loads(sys.argv[5]))
from merlin.common.paths import checkout_root, module_source_path
assert checkout_root() is None
Path = pathlib.Path
exec(sys.argv[4])
workspace = Path(sys.argv[2])
for module, staged_as in json.loads(sys.argv[3]):
    source = module_source_path(module)
    assert source.is_relative_to(Path(sys.argv[1])), source
    stage_client(workspace, source, staged_as)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            str(installed),
            str(workspace),
            json.dumps(clients),
            ast.unparse(helper),
            json.dumps(dependency_roots),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert {path.name for path in workspace.iterdir()} == {staged_as for _, staged_as in clients}
    for module, staged_as in clients:
        assert (workspace / staged_as).read_bytes() == module_source_path(module).read_bytes()
        assert b"PRIVATE_HOST_SENTINEL" not in (workspace / staged_as).read_bytes()
    return workspace


def _run(workspace, name, *arguments):
    return subprocess.run(
        [sys.executable, "-I", "-S", str(workspace / name), *arguments],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("module", public_client_modules())
def test_all_public_clients_import_only_standard_library(module):
    tree = ast.parse(module_source_path(module).read_text())
    imports = {
        name.split(".")[0]
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert imports <= sys.stdlib_module_names


def test_installed_staged_simjob_submit_poll_wait(staged):
    result = _run(staged, "simjob.py", "submit", "--sim", "contract", "--capsules", "one,two", "--workers", "3")
    assert result.returncode == 0, result.stderr
    submitted = json.loads(result.stdout)
    job = submitted["job_id"]
    channel = staged / ".qa_channel"
    request = json.loads((channel / f"simreq_{job}.json").read_text())
    assert request["sim"] == "contract" and request["capsules"] == "one,two" and request["workers"] == 3
    assert submitted["n_capsules"] == 2 and submitted["state"] == "queued"
    assert json.loads(_run(staged, "simjob.py", "poll", "--job-id", job).stdout)["state"] == "queued"
    # A fake broker returns a nonpassing transport result, never hardware success.
    (channel / f"simresp_{job}.json").write_text(json.dumps({"all_pass": False, "error": "synthetic refusal"}))
    (channel / f"simdone_{job}").touch()
    result = _run(staged, "simjob.py", "wait", "--job-id", job, "--timeout", "1")
    assert result.returncode == 1
    assert json.loads(result.stdout)["result"] == {"all_pass": False, "error": "synthetic refusal"}


@pytest.mark.parametrize(
    ("client", "arguments", "channel_name", "expected"),
    [
        (
            "isa_tools.py",
            ["asm", ".word 0x1234", "--timeout", "2"],
            ".isa_channel",
            {"cmd": "asm", "text": ".word 0x1234"},
        ),
        (
            "cca_contract.py",
            ["check-bijection", "fixture"],
            ".cca_channel",
            {"cmd": "check_bijection", "target": "fixture"},
        ),
        (
            "action_catalog.py",
            ["escalation-ladder", "axis", "fixture"],
            ".cca_channel",
            {"cmd": "escalation_ladder", "axis": "axis", "target": "fixture"},
        ),
    ],
)
def test_installed_staged_broker_clients_preserve_requests_and_failures(
    staged, client, arguments, channel_name, expected
):
    channel = staged / channel_name
    channel.mkdir()
    requests = []
    stop = threading.Event()
    response = {"error": "synthetic transport-only refusal"}

    def broker():
        while not stop.wait(0.01):
            for request in channel.glob("req_*.json"):
                try:
                    payload = json.loads(request.read_text())
                except ValueError:
                    continue
                requests.append(payload)
                identity = request.stem.removeprefix("req_")
                (channel / f"resp_{identity}.json").write_text(json.dumps(response))
                (channel / f"done_{identity}").touch()
                return

    thread = threading.Thread(target=broker)
    thread.start()
    try:
        result = _run(staged, client, *arguments)
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(requests) == 1
    assert all(requests[0][key] == value for key, value in expected.items())
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout) == response


def test_installed_staged_wait_reads_only_public_verdict(staged):
    (staged / "qa").mkdir()
    (staged / "qa/verdict.json").write_text(
        json.dumps({"n_passed": 0, "n_capsules": 1, "all_pass": False, "per_capsule": {"fixture": "fail"}})
    )
    result = _run(staged, "await_verdict.py", "--since-ns", "1", "--timeout", "1")
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["waited"] == "graded" and response["all_pass"] is False and response["failing"] == ["fixture"]
    assert not any(path.name.startswith(".") for path in staged.iterdir())


def test_private_phase1_inputs_mask_preserves_every_public_client(tmp_path, monkeypatch):
    from merlin.common import access
    from merlin.targetgen.sandbox import bwrap

    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    monkeypatch.setattr(access, "sys", SimpleNamespace(path=[], prefix=str(tmp_path / "python"), modules={}))
    monkeypatch.setattr(surfaces_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(surfaces_module, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(surfaces_module, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(surfaces_module, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    private = tmp_path / "packages/merlin-experiments/src/merlin_experiments/phase1/run_inputs.py"
    private.parent.mkdir(parents=True)
    private.write_text("private host-only input handling\n")
    public = []
    for module in public_client_modules():
        path = tmp_path / "packages/merlin-experiments/src" / (module.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(module_source_path(module).read_bytes())
        public.append(path)
    policy = SimpleNamespace(
        target="test_device",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    surfaces = surfaces_module.answer_surfaces(policy)
    assert any(surface.path == private and surface.origin == "grader" for surface in surfaces)
    assert all(
        surface.path != client and surface.path not in client.parents for surface in surfaces for client in public
    )
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in bwrap.coverage_gap(unmasked, surfaces)} == {private}
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(unmasked, surfaces), surfaces) == []
