"""Exact launch-source admission without Chia, Ray, or executable dispatch."""

import hashlib
import json
import subprocess
import sys

import pytest
from merlin_experiments.phase2 import chia_launch as CL


@pytest.fixture(autouse=True)
def no_launch(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("process launch"))


@pytest.fixture(params=["script", "module"])
def launch(tmp_path, monkeypatch, request):
    source = tmp_path / "entry.py"
    wrapper = tmp_path / "wrapper.py"
    trace = tmp_path / "trace.py"
    for path in (source, wrapper, trace):
        path.write_text("# inert fixture\n")
    resolver = CL.module_source_path
    monkeypatch.setattr(CL, "module_source_path", lambda name: source if name == "fixture.entry" else resolver(name))
    command = [
        sys.executable,
        *([str(source)] if request.param == "script" else ["-m", "fixture.entry"]),
        "--input",
        "unchanged",
    ]
    plan = {
        "command": command,
        "command_artifacts": CL.command_artifacts(command),
        "wrapper": {"path": str(wrapper), "sha256": CL._sha_file(wrapper)},
        "chia_trace": {"path": str(trace), "sha256": CL._sha_file(trace)},
        "launch_policy": CL.policy_identity(),
    }
    plan["sha256"] = hashlib.sha256(CL._canonical(plan)).hexdigest()
    receipt = {
        **plan,
        "schema": "merlin.chia-agentic-perf-launch.v2",
        "status": "assigned_before_coordinator",
        "plan": plan,
        "plan_sha256": plan["sha256"],
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "assigned_resources": {"codex_slots": 1, "gsim_slots": 1},
    }
    path = tmp_path / "receipt.json"
    environment = {"MERLIN_CHIA_ENVELOPE_PLAN_SHA256": plan["sha256"], "MERLIN_CHIA_LAUNCH_RECEIPT": str(path)}

    def save():
        if path.exists():
            path.chmod(0o600)
        path.write_text(json.dumps(receipt))
        path.chmod(0o444)
        environment["MERLIN_CHIA_LAUNCH_RECEIPT_SHA256"] = CL._sha_file(path)

    save()
    return command, wrapper, environment, receipt, source, save


def test_exact_script_and_module_are_admitted(launch):
    command, wrapper, environment, receipt, source, _ = launch
    result = CL.verify_launch_receipt(command=command, wrapper=wrapper, environment=environment)
    assert result["command_artifacts"][-1]["path"] == str(source)
    assert result["command_artifacts"][-1]["index"] == (2 if command[1] == "-m" else 1)
    assert result["launch_policy"] == CL.policy_identity()


@pytest.mark.parametrize("change", ["entry", "argument", "old_receipt", "policy"])
def test_new_execution_refuses_changed_authority(launch, change):
    command, wrapper, environment, receipt, source, save = launch
    if change == "entry":
        source.write_text("# changed source\n")
    elif change == "argument":
        command = [*command[:-1], "changed"]
    elif change == "old_receipt":
        receipt["schema"] = "merlin.chia-agentic-perf-launch.v1"
        save()
    else:
        receipt["launch_policy"] = {"path": str(source), "sha256": CL._sha_file(source)}
        save()
    with pytest.raises(CL.ExperimentError, match="exact assigned invocation"):
        CL.verify_launch_receipt(command=command, wrapper=wrapper, environment=environment)


@pytest.mark.parametrize(
    "command",
    [[], ["python"], [sys.executable, "-c"], [sys.executable, "-m"], [sys.executable, "-m", "../module"]],
)
def test_unsupported_command_forms_refuse(command):
    with pytest.raises(CL.ExperimentError):
        CL.command_artifacts(command)


def test_module_package_pins_main_not_initializer(tmp_path, monkeypatch):
    initializer, main = tmp_path / "__init__.py", tmp_path / "__main__.py"
    initializer.write_text("# initializer\n")
    main.write_text("# executable module\n")
    monkeypatch.setattr(CL, "module_source_path", lambda name: main if name.endswith(".__main__") else initializer)
    assert CL.command_artifacts([sys.executable, "-m", "fixture"])[1]["sha256"] == CL._sha_file(main)


def test_inline_smoke_pins_exact_code_without_launch():
    first = CL.command_artifacts([sys.executable, "-c", "print('one')"])
    second = CL.command_artifacts([sys.executable, "-c", "print('two')"])
    assert first[0] == second[0]
    assert first[1] == {"index": 2, "kind": "inline_python", "sha256": hashlib.sha256(b"print('one')").hexdigest()}
    assert first[1] != second[1]
