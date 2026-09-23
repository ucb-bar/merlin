"""Catalog-to-managed-checkpoint transport, not scientific qualification."""

import json
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from merlin_experiments import SpecError, load_spec
from merlin_experiments import measured_launch as ML
from merlin_experiments import runner as R
from merlin_experiments.phase2 import checkpoint_cli as CLI
from merlin_experiments.phase2 import chia_envelope as E
from merlin_experiments.phase2 import chia_envelope_cli as EDGE
from merlin_experiments.phase2 import chia_launch as LAUNCH

from merlin.common.paths import module_source_path


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("adapter test cannot launch real processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


@pytest.fixture
def case(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(source))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "output"))
    paths = {}
    for name in ("contract_root", "functional_runs_root", "stage_root", "measurement_root"):
        paths[name] = str(tmp_path / name)
        Path(paths[name]).mkdir()
    paths.update(
        source_root=str(source),
        core_package_root=str(module_source_path("merlin").parent),
        experiments_package_root=str(module_source_path("merlin_experiments").parent),
        experiments_namespace_root=str(module_source_path("merlin.targetgen.capsule_runner").parent.parent),
        managed_native_endpoint=str(tmp_path / "managed.sock"),
    )
    for name in ("descriptor", "rtl_facts", "perf_profile", "gsim_certificate", "holdout_catalog"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump({"target": "fixture"} if name == "descriptor" else {}))
        paths[name] = str(path)
    config = {
        **paths,
        "functional_run_id": "deliberately-unqualified",
        "functional_submission_sha256": "a" * 64,
        "gsim_certificate_sha256": "b" * 64,
        "suite": "synthetic-measured",
        "model": "synthetic",
        "effort": "high",
        "wall_budget_seconds": 60,
        "rounds": 1,
        "round_timeout_seconds": 30,
        "max_tool_calls": 2,
        "tool_timeout_seconds": 10,
    }
    definition = tmp_path / "experiment.yaml"
    document = {
        "schema_version": 1,
        "id": "measured-fixture",
        "target": "fixture",
        "phases": {"2": {"adapter": "measured_claims", "mode": "measured_claims", "config": config}},
    }
    definition.write_text(yaml.safe_dump(document))
    destination = tmp_path / "orchestration-run"
    return SimpleNamespace(
        root=tmp_path, paths=paths, config=config, document=document, definition=definition, destination=destination
    )


def plan(case):
    return R.resolve_plan(load_spec(case.definition), run_dir=case.destination)


@pytest.fixture
def managed(case, monkeypatch):
    from merlin_experiments.execution import chia_group, chia_native

    from merlin.benchharness import chia_bridge, chia_tasks

    trace_file = case.root / "trace.py"
    trace_file.write_text("# inert assignment trace\n")
    chia, trace, ray = ModuleType("chia"), ModuleType("chia.trace"), ModuleType("ray")
    trace.__file__ = str(trace_file)
    chia.trace = trace
    ray.get_runtime_context = lambda: SimpleNamespace(
        get_assigned_resources=lambda: {"codex_slots": 1, "gsim_slots": 1}
    )
    for name, module in (("chia", chia), ("chia.trace", trace), ("ray", ray)):
        monkeypatch.setitem(sys.modules, name, module)
    observed = SimpleNamespace(commands=[], receipts=[], configs=[], sessions=[], accounting=[], failures=[])
    run = SimpleNamespace(
        run_dir=case.root / "accounting",
        profile_path=case.root / "profile",
        mark_failed=lambda: observed.failures.append(True),
        metrics=SimpleNamespace(log_scalar=lambda *args: None),
    )

    @contextmanager
    def accounting(**kwargs):
        observed.accounting.append(kwargs)
        yield run

    @contextmanager
    def tasks(owner):
        yield SimpleNamespace(get=lambda result: result)

    class Session:
        def __init__(self, endpoint):
            observed.sessions.append(endpoint)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Group:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, tasks, launch, *args, **kwargs):
            return launch(*args)

        def returned(self, ref, result):
            assert ref is result

    original_controller = CLI.CTRL.run

    def controller(config, **kwargs):
        observed.configs.append(config)
        return original_controller(config, **kwargs)

    def transport(command, *, cwd, env):
        observed.commands.append((list(command), cwd, dict(env)))
        receipt = LAUNCH.verify_launch_receipt(command=command, wrapper=module_source_path(E.__name__), environment=env)
        observed.receipts.append(receipt)
        assert command[:3] == [sys.executable, "-m", "merlin_experiments.phase2.checkpoint_cli"]
        with monkeypatch.context() as child:
            for key in set(os.environ) - set(env):
                child.delenv(key)
            for key, value in env.items():
                child.setenv(key, value)
            return CLI.main(command[3:], invocation=tuple(command))

    monkeypatch.setattr(E, "_HAVE_CHIA", True)
    monkeypatch.setattr(chia_native, "Session", Session)
    monkeypatch.setattr(chia_native, "run", transport)
    monkeypatch.setattr(chia_bridge, "chia_run", accounting)
    monkeypatch.setattr(chia_tasks, "chia_tasks", tasks)
    monkeypatch.setattr(chia_group, "NativeTaskGroup", Group)
    monkeypatch.setattr(chia_group, "local_options", lambda resources: {"resources": resources})
    monkeypatch.setattr(
        E,
        "run_coordinator",
        SimpleNamespace(options=lambda **kwargs: SimpleNamespace(chia_remote=E.execute_coordinator)),
    )
    monkeypatch.setattr(CLI.CTRL, "run", controller)
    return observed


def test_real_managed_route_admits_receipt_then_honestly_refuses_unqualified_science(case, managed):
    resolved = plan(case)
    command = resolved["phases"]["2"]
    assert R.preflight(resolved)["configuration_ready"]
    assert command["argv"][:3] == [sys.executable, "-m", EDGE.__name__]
    assert command["cwd"] == case.paths["source_root"]
    split = command["argv"].index("--")
    child_args = command["argv"][split + 1 :]
    assert child_args[child_args.index("--chia-wrapper") + 1] == str(module_source_path(E.__name__))
    assert EDGE.main(command["argv"][3:]) == 2
    assert len(managed.receipts) == len(managed.configs) == 1
    assert managed.sessions == [case.paths["managed_native_endpoint"]]
    config = managed.configs[0]
    assert config.context.invocation == tuple(managed.commands[0][0])
    assert config.context.chia_wrapper == module_source_path(E.__name__)
    assert config.context.source_root == Path(case.paths["source_root"])
    assert config.context.contract_root == Path(case.paths["contract_root"])
    assert config.root == case.destination / "phase2"
    assert config.context.suite == case.config["suite"]
    assert managed.accounting[0]["run_id"] == case.destination.name
    assert managed.accounting[0]["target"] == "fixture"
    assert managed.failures == [True]


def test_worker_ambient_python_selection_cannot_replace_captured_roots(case, managed, monkeypatch):
    resolved = plan(case)
    command = resolved["phases"]["2"]

    def remote(plan_record, *args):
        with monkeypatch.context() as worker:
            worker.setenv("PYTHONPATH", "/untrusted/worker/modules")
            worker.setenv("PYTHONHOME", "/untrusted/worker/home")
            worker.setenv("PYTHONUSERBASE", "/untrusted/worker/user")
            worker.setenv("PYTHONSAFEPATH", "0")
            worker.setenv("PYTHONNOUSERSITE", "0")
            return E.execute_coordinator(plan_record, *args)

    monkeypatch.setattr(
        E, "run_coordinator", SimpleNamespace(options=lambda **kwargs: SimpleNamespace(chia_remote=remote))
    )
    monkeypatch.setenv("SYNTHETIC_SECRET", "must-not-be-serialized")
    assert EDGE.main(command["argv"][3:]) == 2
    environment = managed.commands[0][2]
    assert environment["PYTHONSAFEPATH"] == environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONHOME" not in environment
    assert "PYTHONUSERBASE" not in environment
    roots = environment["PYTHONPATH"].split(os.pathsep)
    for key in ("core_package_root", "experiments_package_root", "experiments_namespace_root"):
        assert str(Path(case.paths[key]).parent) in roots
    assert "/untrusted/worker/modules" not in roots
    assert "must-not-be-serialized" not in json.dumps(resolved)
    assert "must-not-be-serialized" not in json.dumps(managed.receipts)


def test_missing_managed_endpoint_refuses_before_transport(case):
    case.document["phases"]["2"]["config"].pop("managed_native_endpoint")
    case.definition.write_text(yaml.safe_dump(case.document))
    with pytest.raises(SpecError, match="managed_native_endpoint"):
        plan(case)


def fake_driver(monkeypatch, command, codes):
    observed = []

    class Process:
        pid = 2**30

        def __init__(self, argv, **kwargs):
            assert argv == command["argv"]
            assert kwargs["cwd"] == command["cwd"]
            observed.append(list(argv))

        def wait(self):
            return codes.pop(0)

    monkeypatch.setattr(R.subprocess, "Popen", Process)
    return observed


def test_interrupted_driver_resumes_same_frozen_root_and_exact_command(case, monkeypatch):
    resolved = plan(case)
    observed = fake_driver(monkeypatch, resolved["phases"]["2"], [130, 0])
    assert R.run(resolved) == 130
    assert R.resume(case.destination) == 0
    assert observed[0] == observed[1]
    assert observed[0][observed[0].index("--root") + 1] == str(case.destination / "phase2")


def test_changed_immutable_input_refuses_resume_before_transport(case, monkeypatch):
    resolved = plan(case)
    observed = fake_driver(monkeypatch, resolved["phases"]["2"], [130])
    assert R.run(resolved) == 130
    Path(case.paths["rtl_facts"]).write_text("changed: true\n")
    with pytest.raises(SpecError, match="changed"):
        R.resume(case.destination)
    assert len(observed) == 1


def test_changed_loaded_source_owner_refuses_resume_before_transport(case, monkeypatch):
    resolved = plan(case)
    observed = fake_driver(monkeypatch, resolved["phases"]["2"], [130])
    assert R.run(resolved) == 130
    replacement = case.root / "shadow-envelope.py"
    replacement.write_bytes(Path(E.__file__).read_bytes())
    monkeypatch.setattr(E, "__file__", str(replacement))
    with pytest.raises(SpecError, match="source|selection"):
        R.resume(case.destination)
    assert len(observed) == 1


def test_historical_native_command_is_not_rewritten_on_resume(case, monkeypatch):
    resolved = plan(case)
    script = case.root / "historical_native.py"
    script.write_text("# historical receipt owner, never executed\n")
    command = resolved["phases"]["2"]
    command["argv"] = [sys.executable, str(script), "--root", str(case.destination / "phase2")]
    command.update(module=None, entrypoint=str(script))
    command.pop("measured_launch")
    resolved["input_paths"]["phase2:entrypoint"] = str(script)
    observed = fake_driver(monkeypatch, command, [130, 0])
    assert R.run(resolved) == 130
    assert R.resume(case.destination) == 0
    assert observed == [command["argv"], command["argv"]]


@pytest.mark.parametrize("change", ["addition", "removal"])
def test_synthetic_source_discovery_membership_drift_refuses_resume(case, monkeypatch, change):
    resolved = plan(case)
    observed = fake_driver(monkeypatch, resolved["phases"]["2"], [130])
    assert R.run(resolved) == 130
    original = ML.python_members
    extra = case.root / "synthetic_extra.py"
    extra.write_text("# synthetic discovered source, never imported\n")

    def changed(root, **kwargs):
        members = dict(original(root, **kwargs))
        if root == Path(case.paths["experiments_package_root"]):
            if change == "addition":
                members["synthetic_extra.py"] = extra
            else:
                members.pop("measured_launch.py")
        return members

    monkeypatch.setattr(ML, "python_members", changed)
    with pytest.raises(SpecError, match="source|ownership|membership"):
        R.resume(case.destination)
    assert len(observed) == 1


@pytest.mark.parametrize("mutable", ["stage_root", "measurement_root", "managed_native_endpoint"])
@pytest.mark.parametrize("immutable", ["rtl_facts", "core_package_root"])
def test_mutable_selection_cannot_overlap_immutable_inputs_or_source_roots(case, mutable, immutable):
    config = case.document["phases"]["2"]["config"]
    selected = Path(case.paths[immutable])
    # No source tree is changed: this nonexistent child tests owner-root coverage
    # independently of whether any currently discovered source lives underneath.
    config[mutable] = str(selected / "empty-child") if immutable == "core_package_root" else str(selected)
    case.definition.write_text(yaml.safe_dump(case.document))
    with pytest.raises(SpecError, match="overlaps frozen input"):
        plan(case)


@pytest.mark.parametrize("selection", ["stage_root", "measurement_root", "source_root", "managed_native_endpoint"])
def test_post_freeze_symlink_redirection_refuses_before_resume_transport(case, monkeypatch, selection):
    if selection == "managed_native_endpoint":
        selected = case.root / "endpoint-parent"
        selected.mkdir()
        case.document["phases"]["2"]["config"][selection] = str(selected / "managed.sock")
        case.definition.write_text(yaml.safe_dump(case.document))
    else:
        selected = Path(case.paths[selection])
    resolved = plan(case)
    observed = fake_driver(monkeypatch, resolved["phases"]["2"], [130])
    assert R.run(resolved) == 130
    # Preserve the original fixture-owned directory; redirect only its old name.
    selected.rename(selected.with_name(selected.name + "-original"))
    selected.symlink_to(Path(case.paths["contract_root"]), target_is_directory=True)
    with pytest.raises(SpecError, match="canonical|location|overlap|changed|selection"):
        R.resume(case.destination)
    assert len(observed) == 1
