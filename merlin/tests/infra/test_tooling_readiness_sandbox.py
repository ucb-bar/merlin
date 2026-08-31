"""Launch-critical checks for the promised Arm4 authoring surface."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

import pytest

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.sandbox import toolchain
from merlin.targetgen.target_experiment import load_target_experiment


def _module():
    harness = str(merlin_dir() / "experiments/capsule_bench/harness")
    if harness not in sys.path:
        sys.path.insert(0, harness)
    return importlib.import_module("tooling_readiness")


def _bwrap_works() -> bool:
    if not shutil.which("bwrap"):
        return False
    run = subprocess.run(
        ["bwrap", "--ro-bind", "/", "/", "--", "true"],
        capture_output=True, timeout=15)
    return run.returncode == 0


def test_tool_only_bundle_refuses_a_missing_promised_grant():
    readiness = _module()
    exp = merlin_dir() / "experiments/capsule_bench/targets/gemmini"
    te = load_target_experiment(exp / "target_experiment.yaml")
    _, bundle = readiness._public_bundle(te, "merlin_assisted_rtlchecks")
    promised, _ = readiness._promised_paths(te, "merlin_assisted_rtlchecks")
    missing = promised[0]
    stale = dict(bundle)
    stale["allowed"] = [entry for entry in bundle["allowed"] if entry.get("path") != missing]

    with pytest.raises(RuntimeError, match="missing promised authoring grant"):
        readiness._tool_only_bundle(te, "merlin_assisted_rtlchecks", stale)


def test_sandbox_pythonpath_names_the_frozen_checkout_not_an_unbuilt_workspace_tree(tmp_path):
    exp = merlin_dir() / "experiments/capsule_bench/targets/gemmini"
    te = load_target_experiment(exp / "target_experiment.yaml")
    value = toolchain.sandbox_env(te, tmp_path / "workspace")
    assert f"export PYTHONPATH={repo_root()}/merlin/python" in value
    assert f"{tmp_path}/workspace/merlin/python" not in value


def test_gemmini_promised_tools_run_in_frozen_bwrap_with_live_brokers():
    if not _bwrap_works():
        pytest.skip("unprivileged bwrap namespaces unavailable")
    check = _module().sandbox_authoring_readiness("gemmini", "merlin_assisted_rtlchecks")
    assert check["ok"], check["evidence"]
    assert "AUTHORING_IMPORTS_AND_OUTPUTS_OK" in check["evidence"]
    assert "BROKER_ROUNDTRIPS_OK" in check["evidence"]
