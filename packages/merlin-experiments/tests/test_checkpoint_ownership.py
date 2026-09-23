"""Installed checkpoint owners require explicit execution resources."""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import checkpoint_cli as CLI
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT

from merlin.common.paths import python_import_roots


def context(root):
    return AD.ExecutionContext(
        source_root=root,
        contract_root=root / "contract",
        functional_runs_root=root / "functional",
        stage_root=root / "stages",
        measurement_root=root / "measurements",
        holdout_sources=HOLDOUT.HoldoutSourceContext(
            source_root=root,
            catalog_path=root / "catalog.yaml",
            core_package_root=root / "core",
            experiments_package_root=root / "experiments",
            experiments_namespace_root=root / "namespace",
        ),
        chia_wrapper=root / "wrapper.py",
        invocation=(sys.executable, "-m", "merlin_experiments.phase2.checkpoint_cli", "--dry-run"),
        suite="synthetic-checkpoint",
    )


def config(root):
    return AD.Config(
        context=context(root),
        experiment_id="test",
        root=root / "run",
        functional_run_id="functional",
        functional_submission_sha256="a" * 64,
        descriptor=root / "descriptor.yaml",
        rtl_facts=root / "facts.json",
        perf_profile=root / "profile.json",
        gsim_certificate=root / "certificate.json",
        gsim_certificate_sha256="b" * 64,
        model="synthetic",
        effort="high",
        wall_budget_seconds=60,
        rounds=1,
        round_timeout_seconds=60,
        max_tool_calls=1,
        tool_timeout_seconds=60,
        smoke_replicates=1,
        holdout_count=4,
        measurement_timeout=60,
    )


def test_installed_checkpoint_import_has_no_checkout_or_native_dependency(tmp_path):
    program = """
import importlib.abc, json, subprocess, sys
sys.path[:0] = json.loads(sys.argv[1])
from merlin.common import paths
def forbidden(*args, **kwargs):
    raise AssertionError('checkpoint import discovered a checkout or launched a process')
paths.repo_root = paths.merlin_dir = forbidden
subprocess.Popen = forbidden
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name in {'run_agentic_perf_experiment', 'run_paired_perf_bench', '_pbcommon',
                    '_common', 'perf_agent_stage', 'chia_agentic_perf_experiment'}:
            raise AssertionError('native dependency: ' + name)
sys.meta_path.insert(0, NoNative())
from merlin_experiments.phase2 import checkpoint_admission, checkpoint_controller, checkpoint_cli, paired_cli
assert callable(checkpoint_controller.run)
assert callable(checkpoint_cli.main)
assert callable(paired_cli.main)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, json.dumps([str(path) for path in python_import_roots()])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_cli_requires_explicit_roots_before_controller(monkeypatch, capsys):
    monkeypatch.setattr(CLI.CTRL, "run", lambda *a, **kw: pytest.fail("controller launched"))
    with pytest.raises(SystemExit) as failure:
        CLI.main([])
    assert failure.value.code == 2
    error = capsys.readouterr().err
    for flag in (
        "--source-root",
        "--contract-root",
        "--functional-runs-root",
        "--stage-root",
        "--measurement-root",
        "--holdout-catalog",
        "--core-package-root",
        "--experiments-package-root",
        "--experiments-namespace-root",
        "--chia-wrapper",
        "--suite",
    ):
        assert flag in error


@pytest.mark.parametrize("field", [field.name for field in dataclasses.fields(AD.ExecutionContext)])
def test_every_execution_context_field_is_bound_to_checkpoint_identity(tmp_path, field):
    original = config(tmp_path)
    document = AD._config_document(original)
    assert json.loads(AD._canonical(document))["context"] == document["context"]
    state = AD.Checkpoints(tmp_path / "state", AD._sha_bytes(AD._canonical(document)))
    state.append("predeclared", {})
    value = getattr(original.context, field)
    changes = {}
    if isinstance(value, Path):
        changed = value / "different"
        if field == "source_root":
            changes["holdout_sources"] = dataclasses.replace(original.context.holdout_sources, source_root=changed)
    elif field == "holdout_sources":
        changed = dataclasses.replace(value, catalog_path=tmp_path / "other-catalog.yaml")
    elif field == "invocation":
        changed = (*value, "--different")
    else:
        changed = value + "-different"
    changes[field] = changed
    updated = dataclasses.replace(original, context=dataclasses.replace(original.context, **changes))
    updated_sha = AD._sha_bytes(AD._canonical(AD._config_document(updated)))
    assert updated_sha != state.config_sha256
    with pytest.raises(AD.ExperimentError, match="checkpoint chain is invalid"):
        AD.Checkpoints(state.root, updated_sha).load()


@pytest.mark.parametrize(
    "field", [field.name for field in dataclasses.fields(HOLDOUT.HoldoutSourceContext) if field.name != "source_root"]
)
def test_each_holdout_resource_is_bound_to_configuration(tmp_path, field):
    original = config(tmp_path)
    changed = dataclasses.replace(original.context.holdout_sources, **{field: tmp_path / (field + "-other")})
    updated = dataclasses.replace(original, context=dataclasses.replace(original.context, holdout_sources=changed))
    assert AD._canonical(AD._config_document(original)) != AD._canonical(AD._config_document(updated))


def test_relative_resources_are_refused(tmp_path):
    with pytest.raises(AD.ExperimentError, match="absolute Path"):
        dataclasses.replace(context(tmp_path), contract_root=Path("relative"))


def test_canary_uses_all_explicit_package_owners(monkeypatch, tmp_path):
    from merlin_experiments import frozen_python

    selected = context(tmp_path)
    selected = dataclasses.replace(
        selected,
        holdout_sources=dataclasses.replace(
            selected.holdout_sources,
            core_package_root=tmp_path / "core-install/merlin",
            experiments_package_root=tmp_path / "workflow-install/merlin_experiments",
            experiments_namespace_root=tmp_path / "namespace-install/merlin",
        ),
    )
    python = tmp_path / "python"
    bridge = tmp_path / "bridge.py"
    trace = tmp_path / "trace.py"
    for path in (python, bridge, trace, selected.chia_wrapper):
        path.write_text("synthetic canary resource")
    monkeypatch.setattr(AD, "_resolve_chia_python", lambda explicit: python)
    monkeypatch.setattr(AD, "module_source_path", lambda name: bridge)
    monkeypatch.setattr(frozen_python, "inherited_python_command", lambda argv: argv)
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"chia_trace": str(trace), "ray": "synthetic"}))

    monkeypatch.setattr(AD.subprocess, "run", fake_run)
    result = AD._chia_canary(python, context=selected)
    assert observed["cwd"] == tmp_path
    assert observed["env"]["PYTHONPATH"].split(AD.os.pathsep)[:3] == [
        str(tmp_path / "core-install"),
        str(tmp_path / "workflow-install"),
        str(tmp_path / "namespace-install"),
    ]
    assert result["required_entrypoint"] == str(selected.chia_wrapper)
