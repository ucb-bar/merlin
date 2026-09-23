"""The actual no-spend batch banner must describe the parsed account and driver."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import python_import_roots, repo_root


@pytest.mark.parametrize(
    "provider,driver,model,billing",
    [
        ("subscription", "claudecode", "claude-opus-4-8", "subscription_notional"),
        ("bedrock", "claudecode", "claude-opus-4-8", "metered"),
        ("subscription", "codex", "gpt-5.6-sol", "subscription_notional"),
        ("bedrock", "codex", "gpt-5.6-sol", "subscription_notional"),
        ("subscription", "codex", "nemotron", "metered"),
        ("bedrock", "codex", "nemotron", "metered"),
    ],
)
def test_real_dry_run_banner_matches_account_and_runtime_routing(tmp_path, provider, driver, model, billing):
    # Only native target/output context is synthetic. Parse real CLI flags, construct
    # real child argv, and execute the same package routing/accounting used by runs.
    program = r"""
import importlib.abc, importlib.util, json, pathlib, socket, subprocess, sys, types
sys.dont_write_bytecode = True
sys.path[:0] = json.loads(sys.argv[1])
class NoController(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'run_baseline_qa_loop':
            raise AssertionError('the banner must not import the native controller')
sys.meta_path.insert(0, NoController())
root=pathlib.Path.cwd()
context=types.ModuleType('_common')
context.EXP=root/'experiment'; context.RUNS=root/'runs'; context.REPO=root
sys.modules['_common']=context
spec=importlib.util.spec_from_file_location('batch_under_test',sys.argv[2])
batch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
def forbidden(*args, **kwargs):
    raise AssertionError('dry-run attempted a process or network operation')
subprocess.Popen=forbidden
socket.socket.connect=forbidden
provider, driver, model = sys.argv[3:]
assert batch.main(['--tag','billing-probe','--arms','baseline','--dry-run',
    '--provider',provider,'--driver',driver,'--model',model]) == 0
assert 'run_baseline_qa_loop' not in sys.modules
assert not list(root.iterdir()), 'dry-run wrote run state'
"""
    dependencies = [p for p in sys.path if p and Path(p).is_dir() and "site-packages" in p]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            program,
            json.dumps([*map(str, python_import_roots()), *dependencies]),
            str(repo_root() / "merlin/experiments/capsule_bench/harness/launch_ab_batch.py"),
            provider,
            driver,
            model,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    banner = result.stdout.splitlines()[0]
    assert f"driver={driver} -> resolved={driver} model={model} billing={billing}" in banner
    assert "nothing launched" in result.stdout
    command = next(line for line in result.stdout.splitlines() if "$ " in line)
    if provider == "bedrock":
        assert "--provider bedrock" in command
    else:
        assert "--provider" not in command  # The runtime's unchanged subscription default.
