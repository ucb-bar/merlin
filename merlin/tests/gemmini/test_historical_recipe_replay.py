"""The retired recipe study must not mistake an ignored local copy for a replayable package."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.target("gemmini")

ROOT = Path(__file__).resolve().parents[3]
CHECK = ROOT / "merlin/experiments/agent_recipe_select_v0/scripts/_track.py"


def _check(root: str | None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "packages/merlin-experiments/src"), str(ROOT / "src"), env.get("PYTHONPATH", ""))
    )
    if root is None:
        env.pop("MERLIN_RECIPE_FROZEN_PACKAGE_ROOT", None)
    else:
        env["MERLIN_RECIPE_FROZEN_PACKAGE_ROOT"] = root
    return subprocess.run([sys.executable, str(CHECK)], env=env, capture_output=True, text=True, check=False)


def test_historical_replay_requires_explicit_external_root():
    run = _check(None)
    assert run.returncode != 0
    assert "MERLIN_RECIPE_FROZEN_PACKAGE_ROOT" in run.stderr


def test_historical_replay_refuses_relative_and_unpinned_package(tmp_path: Path):
    relative = _check("relative/compiler")
    assert relative.returncode != 0
    assert "absolute package root" in relative.stderr

    (tmp_path / "SHA256SUMS").write_text("0" * 64 + "  ./payload.py\n", encoding="utf-8")
    unpinned = _check(str(tmp_path))
    assert unpinned.returncode != 0
    assert "differs from the historical pinned manifest" in unpinned.stderr


def test_census_capability_report_does_not_require_caller_arguments_or_emit_workloads(monkeypatch, tmp_path):
    import importlib.util

    monkeypatch.syspath_prepend(str(CHECK.parent))
    spec = importlib.util.spec_from_file_location("recipe_census_report", CHECK.with_name("census_workloads.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {
        "model_id": "source_model",
        "M": 1,
        "fits_without_cutting": "no",
        "expressible": "yes",
        "mac_share": 1.0,
    }
    report = module._capability_report([row], tmp_path / "census.json", tmp_path / "compiler", "a" * 64, 300)
    assert "| source_model | 1 | 0/1 | 1/1 | 100.0% |" in report
    assert "K is preserved on every row" in report
    assert list(tmp_path.iterdir()) == []
