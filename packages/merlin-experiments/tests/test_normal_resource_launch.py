"""Routing/negative controls only; spies cannot issue a real kernel boundary.

No test starts bwrap, an author, simulator, container or resource load. Actual
kernel refusal controls remain in test_external_resources.
"""

import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import authoring as A
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import codex_author_runtime as CLIENT
from merlin_experiments.phase2 import normal_resource_launch as N
from merlin_experiments.phase2.contracts import StageGateError


def ordinary_prefix(candidate):
    return (
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind-try",
        "/etc",
        "/etc",
        "--tmpfs",
        "/scratch",
        "--tmpfs",
        "/scratch2",
        "--tmpfs",
        "/tmp",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--dir",
        "/tmp/.xdg",
        "--unsetenv",
        "ORIGINAL_DROP",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--bind",
        str(candidate),
        str(candidate),
        "--tmpfs",
        "/masked-answer",
    )


def scratch_sources(argv):
    return [Path(argv[i + 1]) for i, flag in enumerate(argv) if flag == "--bind" and "bounded-normal-" in argv[i + 1]]


@pytest.fixture
def routing_spy(monkeypatch):
    # This bypasses issuance ONLY to observe trusted argument/context routing.
    # It is never evidence that any process has hard resource enforcement.
    state = SimpleNamespace(verifications=[], active=[], drift=False)

    def verify(**kwargs):
        state.verifications.append((threading.get_ident(), kwargs))
        if state.drift:
            raise StageGateError("instrumented resource drift")

    state.owner = SimpleNamespace(verify=verify)

    @contextmanager
    def selected(boundary, *, writable_roots=()):
        assert boundary is state.owner
        boundary.verify(writable_roots=writable_roots)
        token = N.resources._ACTIVE.set(boundary)
        state.active.append(threading.get_ident())
        try:
            yield
            boundary.verify(writable_roots=writable_roots)
        finally:
            N.resources._ACTIVE.reset(token)

    monkeypatch.setattr(N, "selected_boundary", selected)
    return state


def test_ordinary_plan_and_default_path_have_no_issuance_or_filesystem_effect(tmp_path):
    argv = ordinary_prefix(tmp_path / "candidate")
    planned, masks, writable = N._scratch_plan(argv)
    assert planned == list(argv)
    assert [planned[i + 1] for i in masks] == ["/scratch", "/scratch2", "/tmp", "/masked-answer"]
    assert writable == (tmp_path / "candidate",)
    owner = tmp_path / "unused"
    with N.bounded_normal_scratch(argv, owner=owner) as actual:
        assert actual is argv and not owner.exists()


@pytest.mark.parametrize(
    "extra",
    [("--args", "3"), ("--dev-bind", "/dev", "/else"), ("--share-net",), ("--tmpfs", "/a/../b"), ("--", "true")],
)
def test_ordinary_plan_refuses_fd_device_payload_or_new_option_languages(tmp_path, extra):
    with pytest.raises(StageGateError):
        N._scratch_plan((*ordinary_prefix(tmp_path / "candidate"), *extra))


def test_unissued_stage_owner_refuses_before_any_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(A, "_require_stage_containment", lambda *a, **k: pytest.fail("unissued owner reached probe"))
    with pytest.raises(StageGateError, match="exact issued"):
        A.run_stage(
            suite="routing",
            contract_root=tmp_path,
            source_root=tmp_path,
            functional_runs_root=tmp_path,
            functional_run_id="unselected",
            functional_submission_sha256="0" * 64,
            target_experiment=SimpleNamespace(),
            sandbox_inputs=SimpleNamespace(),
            stage_root=tmp_path / "stage",
            model="offline",
            effort="",
            wall_budget_seconds=1,
            rounds=1,
            round_timeout_seconds=1,
            max_tool_calls=1,
            tool_timeout_seconds=1,
            external_resource_boundary={"memory.max": 1},
        )


def test_instrumented_adapter_keeps_every_mask_and_scratch_alive_until_return(tmp_path, routing_spy):
    original = ordinary_prefix(tmp_path / "candidate")
    with N.bounded_normal_scratch(original, owner=tmp_path / "private", boundary=routing_spy.owner) as actual:
        sources = scratch_sources(actual)
        assert len(sources) == 5 and len(set(sources)) == 5
        assert all(path.is_dir() and not list(path.iterdir()) for path in sources)
        assert "--tmpfs" not in actual
        assert actual[-3:] == ["--unshare-user", "--disable-userns", "--assert-userns-disabled"]
        assert N.resources.active_resource_boundary() is routing_spy.owner
        assert original == ordinary_prefix(tmp_path / "candidate")
    assert all(not path.exists() for path in sources)
    assert N.resources.active_resource_boundary() is None


@pytest.mark.parametrize("continuous", [False, True])
def test_outer_callback_scratch_survives_entire_instrumented_author_round(tmp_path, monkeypatch, routing_spy, continuous):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = AW.AgentSandboxPolicy(ordinary_prefix(workspace), (), "available_not_an_isolation_claim", True, True, True)
    monkeypatch.setattr(A.AW, "outer_codex_policy", lambda *a, **k: policy)
    binary, sandbox = tmp_path / "codex", tmp_path / "bwrap"
    binary.write_text("synthetic client, never executed")
    binary.chmod(0o700)
    sandbox.write_text("synthetic sandbox, never executed")
    auth = tmp_path / "private-client"
    auth.mkdir()
    (auth / "auth.json").write_text("synthetic credential, never read")
    monkeypatch.setattr(CLIENT.CA, "real_codex_home", lambda: auth)
    monkeypatch.setattr(CLIENT.CR.shutil, "which", lambda name: str(sandbox))
    sources = []
    stop = lambda: "declared performance target reached"

    def compose(argv, tail, ws):
        sources.extend(scratch_sources(argv))
        return "instrumented original command"

    def run_round(*args, **kwargs):
        if continuous:
            assert kwargs["continue_session"] is True and kwargs["continuation_stop"] is stop
            assert kwargs["continuation_prompt"] == A.SP.QUEUED_AUTHOR_CONTINUATION
        else:
            assert "continue_session" not in kwargs and "continuation_stop" not in kwargs
        assert kwargs["codex_binary"] == str(binary)
        assert kwargs["require_fresh_home"] is True
        assert "/usr" in kwargs["candidate_read_paths"]
        assert str(auth) not in kwargs["candidate_read_paths"]
        for _ in range(2):  # Original preflight and actual author callbacks.
            kwargs["sandbox_command"]("fixed offline payload", workspace, {}, extra_binds=[])
        assert all(path.is_dir() for path in sources)
        assert N.resources.active_resource_boundary() is routing_spy.owner
        return 0, tmp_path / "transcript"

    monkeypatch.setattr(CLIENT.BW, "compose_command", compose)
    monkeypatch.setattr(A, "_import_codex_driver", lambda: SimpleNamespace(run_round=run_round))
    selection = A.PC.PackageSandboxInputs(
        A.AW.TC.ToolchainPaths(tmp_path, *[str(tmp_path / str(i)) for i in range(5)]),
        A.AW.TC.SimToolchain(),
        "",
        (),
    )
    result = A._codex_round(
        workspace,
        tmp_path / "stage",
        SimpleNamespace(text="offline"),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        tmp_path / "base",
        tmp_path / "corpus",
        tmp_path / "control",
        sandbox_inputs=selection,
        model="offline",
        resolved_model="offline",
        effort="",
        round_index=0,
        timeout_s=1,
        codex_binary=tmp_path / "codex",
        external_boundary=routing_spy.owner,
        continue_session=continuous,
        continuation_stop=stop if continuous else None,
    )
    assert result[2] is policy and sources and all(not path.exists() for path in sources)
    assert N.resources.active_resource_boundary() is None


@pytest.mark.parametrize("drift", [False, True])
def test_broker_thread_binds_owner_and_refuses_drift_before_completion(tmp_path, monkeypatch, routing_spy, drift):
    candidate, receipts = tmp_path / "candidate", tmp_path / "control/receipts.jsonl"
    candidate.mkdir()
    policy = SimpleNamespace(
        argv=ordinary_prefix(candidate),
        env_prefix="",
        process_cwd=tmp_path,
        network="available_not_an_isolation_claim",
        clear_environment=True,
        verify_execution=lambda: None,
    )
    target = SimpleNamespace()
    workflow = SimpleNamespace(
        candidate=candidate,
        receipt_path=receipts,
        target_experiment=target,
        workflow_id="routing-only",
        validate_actions=lambda actions: None,
        admission=lambda name: SimpleNamespace(limit=None, reserved=False),
        execute=lambda *args: None,
    )
    broker = B.Broker(
        policy,
        target,
        candidate,
        (B.BrokerAction("fixed", ("true",), (), "routing", False),),
        receipts,
        deadline=time.monotonic() + 10,
        workflow=workflow,
        max_calls=1,
        max_tool_seconds=2,
        external_boundary=routing_spy.owner,
    )
    observed, outcome = [], []

    def run(command, **kwargs):
        assert N.resources.active_resource_boundary() is routing_spy.owner
        paths = scratch_sources(command)
        assert len(paths) == 5 and all(path.is_dir() for path in paths)
        observed.extend(paths)
        routing_spy.drift = drift
        return subprocess.CompletedProcess(command, 0, "fixed observation", "")

    def execute():
        try:
            outcome.append(broker.execute({"action": "fixed"}))
        except StageGateError as exc:
            outcome.append(exc)

    monkeypatch.setattr(B.subprocess, "run", run)
    thread = threading.Thread(target=execute)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive() and observed and all(not path.exists() for path in observed)
    assert thread.ident in routing_spy.active and thread.ident != threading.get_ident()
    if drift:
        assert isinstance(outcome[0], StageGateError) and broker.calls[0]["state"] == "rejected"
    else:
        assert outcome[0]["returncode"] == 0 and broker.calls[0]["state"] == "complete"


def test_required_tool_probe_uses_bounded_prefix_without_changing_policy(tmp_path, monkeypatch, routing_spy):
    candidate = tmp_path / "candidate"
    original = ordinary_prefix(candidate)
    policy = SimpleNamespace(
        argv=original,
        required_tools=(AW.TC.ToolProbe("fixed", "true"),),
        env_prefix="",
        process_cwd=tmp_path,
        verify_execution=lambda: None,
    )
    sources = []

    def run(command, **kwargs):
        sources.extend(scratch_sources(command))
        assert sources and all(path.is_dir() for path in sources)
        assert N.resources.active_resource_boundary() is routing_spy.owner
        return subprocess.CompletedProcess(command, 0, "fixed", "")

    monkeypatch.setattr(AW.subprocess, "run", run)
    rows = AW.run_required_tool_probes(policy, SimpleNamespace(), candidate, external_boundary=routing_spy.owner)
    assert rows[0]["returncode"] == 0 and policy.argv is original
    assert all(not path.exists() for path in sources)
