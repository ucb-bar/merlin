"""Captured two-plane policies, without a process, listener or OS isolation claim."""

import hashlib
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import campaign as C
from merlin_experiments.phase2 import qualification_policy as Q
from merlin_experiments.phase2.contracts import StageGateError, canonical_json

from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


@pytest.fixture(autouse=True)
def no_execution(monkeypatch, tmp_path):
    def refused(*args, **kwargs):
        pytest.fail("sandbox policy tests cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))


@pytest.fixture
def case(tmp_path, monkeypatch):
    def write(path, payload=b"synthetic", mode=0o644):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(mode)
        return path

    repo = tmp_path / "selected-checkout"
    repo.mkdir()
    paths = TC.ToolchainPaths(
        repo,
        *(str(tmp_path / name) for name in ("venv", "llvm", "compat", "clang", "uv")),
        python_import_roots=(str(tmp_path / "venv/site-packages"),),
    )
    for name in (paths.venv, paths.llvm, paths.compat_lib, paths.clang_bin, paths.clang_resource, paths.uv_python):
        write(Path(name) / "synthetic-tool", mode=0o755)
    simulator = tmp_path / "sim"
    write(simulator / "bin/sim", mode=0o755)
    private = write(simulator / "answers/secret", b"private")
    sim = TC.SimToolchain(
        bind_paths=(str(simulator),),
        path_dirs=(str(simulator / "bin"),),
        ld_dirs=(),
        env_extra={"SELECTED": "original"},
        probes=(TC.ToolProbe("selected simulator", "sim --version", str(simulator)),),
    )
    inputs = C.PackageSandboxInputs(paths, sim, "", (AnswerSurface("answers", private.parent, "dir", "oracle"),))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    root = tmp_path / "agent-inputs"
    payload = b"public interface"
    write(root / "interface", payload)
    digest = hashlib.sha256(payload).hexdigest()
    aggregate = hashlib.sha256(f"interface\0{digest}\0{len(payload)}\n".encode()).hexdigest()
    manifest = write(
        root / "agent_input_manifest.json",
        canonical_json({"files": [{"path": "interface", "sha256": digest, "n_bytes": len(payload)}]}),
    )
    snapshot = AW.AgentInputSnapshot(
        root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest(), aggregate, 1, len(payload)
    )
    base_calls = []

    def base(workspace, bundle, **kwargs):
        base_calls.append((workspace, bundle, kwargs))
        return ["bwrap", "--bind", str(workspace), str(workspace)]

    monkeypatch.setattr(AW.BW, "base_argv", base)
    return SimpleNamespace(
        root=tmp_path,
        inputs=inputs,
        target=SimpleNamespace(target="synthetic"),
        candidate=candidate,
        snapshot=snapshot,
        private=private,
        base_calls=base_calls,
    )


def forbid_discovery(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("captured configuration rediscovered ambient inputs")

    for name in ("_sim", "curated_harness_dir", "repo_root", "compat_lib_dir", "env", "ext_path"):
        monkeypatch.setattr(TC, name, refused)
    monkeypatch.setattr(TC.ToolchainPaths, "from_checkout", refused)
    monkeypatch.setattr(C, "answer_surfaces", refused)
    return refused


def inner(case, selected=None):
    return AW.inner_execution_policy(
        case.target, case.candidate, case.snapshot, inputs=case.inputs if selected is None else selected
    )


@pytest.mark.parametrize("frozen", [False, True])
def test_broker_and_probes_share_captured_selection_after_ambient_drift(case, monkeypatch, frozen):
    selected = case.inputs
    if frozen:
        selected = Q.restore(case.root, Q.freeze(case.root, case.target, selected))
    refused = forbid_discovery(monkeypatch)
    policy = inner(case, selected)
    assert policy.process_cwd == case.inputs.paths.repo
    assert policy.required_tools[-1] == case.inputs.sim.probes[0]
    assert "SELECTED=original" in policy.env_prefix
    assert case.base_calls[-1][2]["repo"] == case.inputs.paths.repo
    monkeypatch.setattr(TC, "sandbox_env", refused)
    monkeypatch.setattr(TC, "required_tool_probes", refused)
    monkeypatch.setenv("PYTHONPATH", "/ambient/private")
    case.inputs.sim.env_extra["SELECTED"] = "changed"
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="synthetic observation", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    rows = AW.run_required_tool_probes(policy, case.target, case.candidate, timeout_s=7)
    assert [row["label"] for row in rows] == [probe.label for probe in policy.required_tools]
    for (argv, options), probe in zip(calls, policy.required_tools, strict=True):
        assert argv[-1] == policy.env_prefix + probe.cmd
        assert options["cwd"] == str(policy.process_cwd)
        assert options["timeout"] == 7
    receipt = case.root / "receipts.jsonl"
    workflow = BP.select_workflow(
        BP.CORPUS_FEEDBACK_V1, candidate=case.candidate, target_experiment=case.target, receipt_path=receipt
    )
    broker = B.Broker(
        policy,
        case.target,
        case.candidate,
        (B.BrokerAction("synthetic-tool", ("tool", "--version"), (), "test", False),),
        receipt,
        deadline=time.monotonic() + 30,
        workflow=workflow,
        max_calls=2,
        max_tool_seconds=7,
    )
    assert broker.execute({"action": "synthetic-tool"})["returncode"] == 0
    assert calls[-1][1]["cwd"] == str(policy.process_cwd)
    assert policy.env_prefix + 'exec "$@"' in calls[-1][0]
    assert "changed" not in " ".join(calls[-1][0])


@pytest.mark.parametrize("frozen", [False, True])
def test_outer_does_not_gain_inner_toolchain_and_masks_after_runtime_overlay(case, monkeypatch, frozen):
    selected = case.inputs
    if frozen:
        selected = Q.restore(case.root, Q.freeze(case.root, case.target, selected))
    forbid_discovery(monkeypatch)
    inside = inner(case, selected)
    outside = AW.outer_codex_policy(
        case.candidate,
        case.snapshot,
        ("--ro-bind", str(case.private.parent), str(case.private.parent)),
        case.target,
        inputs=selected,
    )
    assert AW.BW.is_exposed(list(inside.argv), Path(case.inputs.sim.bind_paths[0]) / "bin/sim")
    assert not AW.BW.is_exposed(list(outside.argv), Path(case.inputs.sim.bind_paths[0]) / "bin/sim")
    for policy in (inside, outside):
        assert not AW.BW.is_exposed(list(policy.argv), case.private)
        assert policy.answer_surface_gap == ()
    with pytest.raises(StageGateError):
        outside.verify_execution()


@pytest.mark.parametrize("route", ["probe", "command"])
def test_frozen_tamper_refuses_before_execution(case, monkeypatch, route):
    selected = Q.restore(case.root, Q.freeze(case.root, case.target, case.inputs))
    policy = inner(case, selected)
    snapshot_root = Path(selected.record["snapshot"]["path"])
    file = next(path for path in snapshot_root.rglob("*") if path.is_file())
    file.chmod(0o644)
    file.write_bytes(b"tampered")
    file.chmod(0o444)
    with pytest.raises((StageGateError, C.CampaignGateError)):
        if route == "probe":
            AW.run_required_tool_probes(policy, case.target, case.candidate)
        else:
            B.inner_command(policy, case.target, case.candidate, ["tool"], 3)


@pytest.mark.parametrize("field,value", [("required_tools", ()), ("env_prefix", None), ("process_cwd", None)])
def test_incomplete_policy_refuses_execution(case, field, value):
    policy = replace(inner(case), **{field: value})
    with pytest.raises(StageGateError):
        AW.run_required_tool_probes(policy, case.target, case.candidate)
    with pytest.raises(StageGateError):
        B.inner_command(policy, case.target, case.candidate, ["tool"], 3)


@pytest.mark.parametrize("mutation", ["environment", "probes", "cwd", "grants", "answer-overlay", "record"])
def test_frozen_policy_cannot_change_captured_authority(case, mutation):
    selected = Q.restore(case.root, Q.freeze(case.root, case.target, case.inputs))
    policy = inner(case, selected)
    if mutation == "environment":
        policy = replace(policy, env_prefix=policy.env_prefix + "export INJECTED=yes; ")
    elif mutation == "probes":
        policy = replace(policy, required_tools=(TC.ToolProbe("other", "other --version", "/other"),))
    elif mutation == "cwd":
        policy = replace(policy, process_cwd=case.candidate)
    elif mutation == "grants":
        policy = replace(policy, argv=("bwrap",))
    elif mutation == "answer-overlay":
        policy = replace(policy, argv=(*policy.argv, "--ro-bind", str(case.private.parent), str(case.private.parent)))
    else:
        policy.frozen_inputs.record["env_prefix"] += "export INJECTED=yes; "
    with pytest.raises(StageGateError):
        B.inner_command(policy, case.target, case.candidate, ["tool"], 3)


def test_zero_tool_selection_refuses_execution(case, monkeypatch):
    monkeypatch.setattr(TC, "required_tool_probes", lambda *_args, **_kwargs: ())
    policy = inner(case)
    with pytest.raises(StageGateError):
        AW.run_required_tool_probes(policy, case.target, case.candidate)


@pytest.mark.parametrize("option", ["--bind", "--ro-bind"])
@pytest.mark.parametrize("shadow", ["exact", "executable", "ancestor", "parent-alias"])
def test_appended_mount_cannot_shadow_frozen_tools(case, option, shadow):
    selected = Q.restore(case.root, Q.freeze(case.root, case.target, case.inputs))
    policy = inner(case, selected)
    tool_root = Path(case.inputs.sim.bind_paths[0])
    destinations = {
        "exact": str(tool_root),
        "executable": str(tool_root / "bin/sim"),
        "ancestor": str(tool_root.parent),
        "parent-alias": str(tool_root / "bin/../bin/sim"),
    }
    # Original frozen grants remain intact. This later exposing mount is the attack.
    changed = replace(policy, argv=(*policy.argv, option, str(case.candidate), destinations[shadow]))
    with pytest.raises(StageGateError):
        B.inner_command(changed, case.target, case.candidate, ["tool"], 3)
    with pytest.raises(StageGateError):
        AW.run_required_tool_probes(changed, case.target, case.candidate)


def test_disjoint_compiler_overlay_and_repeated_answer_masks_remain_admitted(case):
    selected = Q.restore(case.root, Q.freeze(case.root, case.target, case.inputs))
    policy = inner(case, selected)
    compiler = case.root / "compiler-overlay"
    compiler.mkdir()
    (compiler / "compiler.py").write_text("SYNTHETIC = True\n")
    changed = replace(
        policy,
        argv=(*policy.argv, "--ro-bind", str(compiler), "/selected-compiler", "--tmpfs", str(case.private.parent)),
    )
    command = B.inner_command(changed, case.target, case.candidate, ["tool"], 3)
    assert command[: len(changed.argv)] == list(changed.argv)
    assert not AW.BW.is_exposed(list(changed.argv), case.private)
