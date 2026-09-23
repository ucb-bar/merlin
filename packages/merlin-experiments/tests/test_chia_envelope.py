"""Installed envelope evidence with synthetic assignment/transport, never a service."""

import hashlib
import json
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from merlin_experiments.phase2 import chia_envelope as E
from merlin_experiments.phase2 import chia_envelope_cli as CLI
from merlin_experiments.phase2 import chia_launch as LAUNCH


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("envelope test launched a process or listener")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)


def test_installed_cli_explicit_module_plan(tmp_path, capsys):
    assert (
        CLI.main(
            [
                "--driver-python",
                sys.executable,
                "--cwd",
                str(tmp_path),
                "--suite",
                "synthetic",
                "--target",
                "sample",
                "--orchestration-run-id",
                "run",
                "--dry-run",
                "--",
                "--experiment-id",
                "case",
                "--suite",
                "coordinator-suite",
                "--target",
                "coordinator-target",
                "--chia-wrapper",
                str(Path(E.__file__).resolve()),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["command"] == [
        sys.executable,
        "-m",
        "merlin_experiments.phase2.checkpoint_cli",
        "--experiment-id",
        "case",
        "--suite",
        "coordinator-suite",
        "--target",
        "coordinator-target",
        "--chia-wrapper",
        str(Path(E.__file__).resolve()),
    ]
    assert plan["cwd"] == str(tmp_path)
    assert plan["driver_parity_claim"] is False


def test_installed_cli_has_no_implicit_layout():
    with pytest.raises(SystemExit):
        CLI.main(["--orchestration-run-id", "run", "--dry-run"])


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_stub_refuses_before_execution(tmp_path, value):
    with pytest.raises(SystemExit):
        CLI.main(
            [
                "--driver-python",
                sys.executable,
                "--cwd",
                str(tmp_path),
                "--suite",
                "test",
                "--target",
                "test",
                "--orchestration-run-id",
                "test",
                f"--stub-seconds={value}",
                "--dry-run",
            ]
        )


@pytest.mark.parametrize("field", ["suite", "target"])
def test_context_refuses_nonstring_labels(tmp_path, field):
    values = dict(
        python=Path(sys.executable),
        cwd=tmp_path,
        wrapper_source=Path(E.__file__).resolve(),
        coordinator_prefix=("-m", "merlin_experiments.phase2.checkpoint_cli"),
        suite="test",
        target="test",
    )
    values[field] = 7
    with pytest.raises(ValueError, match="explicit suite and target"):
        E.EnvelopeContext(**values)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1", None, 0])
def test_assignment_rejects_invalid_resources(value):
    with pytest.raises(RuntimeError):
        E.validate_assigned_resources({"codex_slots": value, "gsim_slots": 1})


@pytest.fixture
def assigned(tmp_path, monkeypatch):
    trace_file = tmp_path / "trace.py"
    trace_file.write_text("# synthetic trace identity\n")
    chia = ModuleType("chia")
    trace = ModuleType("chia.trace")
    trace.__file__ = str(trace_file)
    chia.trace = trace
    monkeypatch.setitem(sys.modules, "chia", chia)
    monkeypatch.setitem(sys.modules, "chia.trace", trace)
    resources = {"codex_slots": 1, "gsim_slots": 1}
    ray = ModuleType("ray")
    ray.get_runtime_context = lambda: SimpleNamespace(get_assigned_resources=lambda: resources)
    monkeypatch.setitem(sys.modules, "ray", ray)
    wrapper = Path(E.__file__).resolve()
    command = [sys.executable, "-m", "merlin_experiments.phase2.checkpoint_cli", "--experiment-id", "fixture"]
    from merlin.common.paths import module_source_path

    for name, path in (
        ("core-package-root", module_source_path("merlin").parent),
        ("experiments-package-root", module_source_path("merlin_experiments").parent),
        ("experiments-namespace-root", module_source_path("merlin.targetgen.capsule_runner").parent.parent),
    ):
        command += ["--" + name, str(path)]
    plan = {
        "command": command,
        "cwd": str(tmp_path),
        "wrapper": {"path": str(wrapper), "sha256": E._sha_file(wrapper)},
        "envelope_owner": E._owner_identity(),
        "chia_trace": {"path": str(trace_file), "sha256": E._sha_file(trace_file)},
        "command_artifacts": LAUNCH.command_artifacts(command),
        "launch_policy": LAUNCH.policy_identity(),
        "resources": {"codex_slots": 1, "gsim_slots": 1},
    }
    plan["python_sources"], plan["python_source_environment"] = E._python_selection(command)

    def seal():
        plan.pop("sha256", None)
        plan["sha256"] = hashlib.sha256(E._canonical(plan)).hexdigest()

    seal()
    calls = []
    from merlin_experiments.execution import chia_native

    def transport(command, *, cwd, env):
        # Admission must independently accept exactly the receipt the real worker wrote.
        receipt = LAUNCH.verify_launch_receipt(command=command, wrapper=wrapper, environment=env)
        for key, value in plan["python_source_environment"].items():
            assert env.get(key) == value
            assert (key in env) is (value is not None)
        calls.append((command, cwd, receipt))
        return 0

    monkeypatch.setattr(chia_native, "run", transport)
    return SimpleNamespace(
        root=tmp_path, wrapper=wrapper, command=command, plan=plan, resources=resources, calls=calls, seal=seal
    )


def execute(case, **changes):
    return E.execute_coordinator(
        changes.get("command", case.command),
        changes.get("cwd", str(case.root)),
        case.plan,
        str(case.root / "receipts"),
        str(case.wrapper),
    )


def test_assigned_worker_publishes_admitted_launch_and_completion(assigned):
    result = execute(assigned)
    assert result["returncode"] == 0
    assert len(assigned.calls) == 1
    for key in ("launch_receipt", "completion_receipt"):
        path = Path(result[key]["path"])
        assert E._sha_file(path) == result[key]["sha256"]
        assert path.stat().st_mode & 0o222 == 0
    completion = json.loads(Path(result["completion_receipt"]["path"]).read_text())
    assert completion["status"] == "complete"
    assert completion["launch_receipt"] == result["launch_receipt"]


@pytest.mark.parametrize(
    "change", ["command", "cwd", "owner", "wrapper", "trace", "policy", "artifacts", "digest", "assignment"]
)
def test_worker_refuses_drift_before_transport_or_receipts(assigned, change):
    kw = {}
    if change in ("command", "cwd"):
        kw[change] = [*assigned.command, "--other"] if change == "command" else str(assigned.root / "other")
    elif change == "assignment":
        assigned.resources["gsim_slots"] = float("nan")
    elif change == "digest":
        assigned.plan["sha256"] = "0" * 64
    else:
        field = {
            "owner": "envelope_owner",
            "trace": "chia_trace",
            "policy": "launch_policy",
            "artifacts": "command_artifacts",
        }.get(change, change)
        assigned.plan[field] = {}
        assigned.seal()
    with pytest.raises(RuntimeError):
        execute(assigned, **kw)
    assert not assigned.calls
    assert not (assigned.root / "receipts").exists()


def test_worker_ignores_ambient_source_environment(assigned, monkeypatch):
    for key in LAUNCH.PYTHON_SOURCE_ENVIRONMENT_KEYS:
        monkeypatch.setenv(key, "/unexpected/worker/source")
    assert execute(assigned)["returncode"] == 0
    assert len(assigned.calls) == 1


@pytest.mark.parametrize("change", ["missing_key", "extra_key", "value", "source_bytes"])
def test_worker_refuses_incomplete_or_changed_source_selection(assigned, change):
    if change == "missing_key":
        assigned.plan["python_source_environment"].pop("PYTHONHOME")
    elif change == "extra_key":
        assigned.plan["python_source_environment"]["SECRET_TOKEN"] = "must-not-be-serialized"
    elif change == "value":
        assigned.plan["python_source_environment"]["PYTHONSAFEPATH"] = "0"
    else:
        next(iter(assigned.plan["python_sources"].values()))["sha256"] = "0" * 64
    assigned.seal()
    with pytest.raises(RuntimeError, match="source selection"):
        execute(assigned)
    assert not assigned.calls


@pytest.mark.parametrize("returncode", [0, 7])
def test_envelope_composes_existing_supervisors_and_accounts_failure(assigned, monkeypatch, returncode):
    from merlin_experiments import frozen_python
    from merlin_experiments.execution import chia_group, chia_native

    from merlin.benchharness import chia_bridge, chia_tasks

    events = []
    run = SimpleNamespace(
        run_dir=assigned.root / "run",
        profile_path=assigned.root / "profile",
        mark_failed=lambda: events.append("failed"),
        metrics=SimpleNamespace(log_scalar=lambda *args: events.append(args)),
    )

    @contextmanager
    def accounting(**kwargs):
        assert kwargs["suite"] == "explicit-suite"
        assert kwargs["target"] == "explicit-target"
        assert kwargs["accounting"] == "child-ledgers"
        yield run

    @contextmanager
    def tasks(owner):
        assert owner is run
        yield SimpleNamespace(get=lambda result: result)

    class Group:
        def __init__(self, owner, session):
            assert owner is run
            assert session == "synthetic-session"

        def __enter__(self):
            events.append("group-enter")
            return self

        def __exit__(self, *args):
            events.append("group-exit")

        def submit(self, tasks, launch, *args, **kwargs):
            events.append("submit")
            return launch(*args)

        def returned(self, ref, result):
            assert ref is result
            events.append("returned")

    monkeypatch.setattr(E, "_HAVE_CHIA", True)
    monkeypatch.setattr(chia_bridge, "chia_run", accounting)
    monkeypatch.setattr(chia_tasks, "chia_tasks", tasks)
    monkeypatch.setattr(chia_group, "NativeTaskGroup", Group)
    monkeypatch.setattr(chia_group, "local_options", lambda resources: {"resources": resources})
    monkeypatch.setattr(frozen_python, "inherited_python_command", lambda command: list(command))
    monkeypatch.setattr(chia_native, "run", lambda *args, **kwargs: returncode)
    monkeypatch.setattr(
        E,
        "run_coordinator",
        SimpleNamespace(options=lambda **kwargs: SimpleNamespace(chia_remote=E.execute_coordinator)),
    )
    context = E.EnvelopeContext(
        python=Path(sys.executable),
        cwd=assigned.root,
        wrapper_source=assigned.wrapper,
        coordinator_prefix=("-m", "merlin_experiments.phase2.checkpoint_cli"),
        suite="explicit-suite",
        target="explicit-target",
    )
    args = SimpleNamespace(orchestration_run_id="fixture", codex_slots=1, gsim_slots=1)
    assigned.plan["protocol"] = "unchanged_sequential_resume_safe_coordinator"
    assert E._execute(args, assigned.command, assigned.plan, "synthetic-session", context=context) == returncode
    assert events[:3] == ["group-enter", "submit", "returned"]
    assert ("failed" in events) is bool(returncode)
    assert events[-1] == "group-exit"
    assert run.summary["result"]["returncode"] == returncode
    assert (run.run_dir / "chia/agentic_perf_plan.json").is_file()
