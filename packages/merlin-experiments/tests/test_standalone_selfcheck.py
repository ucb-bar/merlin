"""The public workspace client must run without installed Merlin or private graders."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys

from merlin_experiments.phase1.brokers import selfcheck as selfcheck_broker
from merlin_experiments.phase1.feedback import lifecycle, selfcheck

from merlin.common.paths import module_source_path


def test_staged_client_runs_with_site_packages_disabled(tmp_path):
    source = module_source_path("merlin_experiments.phase1.tools.selfcheck")
    client = tmp_path / "agent_selfcheck.py"
    shutil.copyfile(source, client)
    assert client.read_bytes() == source.read_bytes()
    imports = {
        name.split(".")[0]
        for node in ast.walk(ast.parse(client.read_text()))
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert imports <= sys.stdlib_module_names
    env = dict(os.environ, MERLIN_REQUIRED_RTL_ENGINE="gsim", PYTHONPATH="")
    help_result = subprocess.run(
        [sys.executable, "-I", "-S", str(client), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--shape-coverage" in help_result.stdout
    refused = subprocess.run(
        [sys.executable, "-I", "-S", str(client), "--sim", "verilator"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert refused.returncode == 2, refused.stderr
    assert json.loads(refused.stdout)["all_pass"] is False
    assert not (tmp_path / ".qa_channel").exists()


def test_native_staging_copies_package_bytes_without_clobbering_bound_source(tmp_path):
    # Execute the canonical staging owner without importing an agent/hardware driver.
    bound_source = tmp_path / "original.py"
    bound_source.write_text("original bound source\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    staged = workspace / "agent_selfcheck.py"
    staged.symlink_to(bound_source)
    source = module_source_path("merlin_experiments.phase1.tools.selfcheck")
    lifecycle.stage_client(workspace, source, staged.name)
    assert not staged.is_symlink()
    assert staged.read_bytes() == source.read_bytes()
    assert bound_source.read_text() == "original bound source\n"


def test_unflagged_selfcheck_selects_the_grade_adapters_in_worker_and_broker(monkeypatch):
    """No engine named: the broker forwards the grade's own selection, which the worker resolves to
    exactly the adapters the official grade uses -- never one engine, pinned or discovered."""
    for pin in ("", "gsim"):
        monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", pin)
        assert selfcheck_broker._default_sim() == selfcheck.REQUIRED_SIM
        assert selfcheck.REQUIRED_SIM in selfcheck_broker._allowed_sims()
    official = {"L2": object(), "L3": object()}
    calls = []
    monkeypatch.setattr(
        selfcheck.CR, "qa_checkpoint_adapters",
        lambda target, sim_via, **kwargs: calls.append((target, sim_via, kwargs)) or official,
    )
    monkeypatch.setattr(selfcheck.CR, "simulator_adapter", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the grade's selection must not construct a narrower engine adapter")))
    adapters, sim = selfcheck._adapters(selfcheck.REQUIRED_SIM, "synthetic", "chipyard", readback_policy="full")
    assert sim == selfcheck.REQUIRED_SIM
    assert adapters == official and all(adapters[t] is official[t] for t in official)
    assert calls == [("synthetic", "chipyard", {"readback_policy": "full"})]


def test_legacy_capsule_fallback_names_itself_and_its_cause(tmp_path, capsys):
    # A silent fallback to the legacy set reads as the package failing capsules that are merely
    # missing from it ("unknown capsule(s)"); the operator has to see that it happened, and why.
    from merlin_experiments.phase1.context import InvocationContext

    missing = tmp_path / "absent" / "target_experiment.yaml"
    context = InvocationContext(
        repo=tmp_path,
        descriptor=missing,
        experiment=missing.parent,
        target="stub",
        runs=tmp_path,
        reports=tmp_path,
        bundles=tmp_path,
        sourced_environment=(),
        harness=tmp_path / "harness",
    )
    assert selfcheck._public_capsules(context) == tmp_path / "harness" / "full_public_capsules"
    err = capsys.readouterr().err
    assert "falling back to the legacy committed set" in err
    assert str(missing) in err and "Error" in err
