"""Durability tests for the three-root artifact/run convention.

Covers the shared helper (start_run/new_product/cache_dir), the layout linter, and the
PreToolUse guard hook. These assert the convention "holds for the future": a fresh run lands
in the right place, products carry a manifest + relative `latest`, the linter flags a planted
stray, and the hook denies a bad write while allowing source edits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common import artifacts as A
from merlin.common.paths import repo_root

REPO = repo_root()
HOOK = REPO / ".claude" / "hooks" / "guard_artifact_writes.py"
LINT = REPO / "build_tools" / "scripts" / "check_artifact_layout.py"


# ----------------------------------------------------------------- helper API


def test_utc_stamp_is_sortable_and_colon_free():
    ts = A.utc_stamp()
    assert ts.endswith("Z") and "T" in ts and ":" not in ts and len(ts) == 16


def test_start_run_lands_under_target_suite(tmp_path):
    h = A.start_run(suite="perf-bench", method="m0", seed=2, target="gemmini",
                    project_root=tmp_path)
    rel = h.run_dir.relative_to(tmp_path).parts
    assert rel[:3] == ("runs", "gemmini", "perf-bench")          # target at folder level
    assert h.run_dir.name.endswith(f"_m0_seed002_{h.git_sha}")    # naming convention
    assert (h.run_dir / "logs").is_dir() and (h.run_dir / "generated").is_dir()
    assert any(h.run_dir.glob("run_record.json"))                 # provenance written
    A.finish_run(h, "completed", summary={"n": 1})


def test_new_product_manifest_and_relative_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    p = A.new_product("dse", version=1, target="bitvla", notes="t")
    rel = p.path.relative_to(tmp_path).parts
    assert rel[:4] == ("artifacts", "dse", "bitvla", "v1")        # topic + target + version
    p.add_artifact("findings.csv").write_text("a,b\n1,2\n")
    mp = p.write_manifest()
    man = mp.read_text()
    assert "git_sha" in man and "timestamp" in man and "findings.csv" in man
    latest = p.path.parent / "latest"
    assert latest.is_symlink() and not os.path.isabs(os.readlink(latest))   # relative (bwrap-safe)
    assert (p.path.parent / os.readlink(latest)).exists()                   # not dangling


def test_cache_dir_under_artifacts_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    d = A.cache_dir("kc")
    assert d.relative_to(tmp_path).parts == ("artifacts", "cache", "kc") and d.is_dir()


# ----------------------------------------------------------------- linter


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def test_linter_passes_on_clean_tree():
    r = _run([sys.executable, str(LINT)], cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr


def test_linter_flags_planted_stray(tmp_path):
    # init a throwaway git repo with a tracked generated file outside artifacts/
    _run(["git", "init", "-q"], cwd=tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "stray.png").write_bytes(b"x")
    _run(["git", "add", "-A", "-f"], cwd=tmp_path)
    r = _run([sys.executable, str(LINT)], cwd=tmp_path)
    assert r.returncode == 1 and "stray.png" in (r.stdout + r.stderr)


# ----------------------------------------------------------------- guard hook


def _hook(tool, path):
    payload = json.dumps({"tool_name": tool, "tool_input": {"file_path": path}})
    return subprocess.run([sys.executable, str(HOOK)], input=payload,
                          capture_output=True, text=True, cwd=REPO)


@pytest.mark.parametrize("path", [
    "output/foo.png", "results/x_dse_analysis/r.csv", "selfcheck_out/log.jsonl",
    "merlin/x.svg",
])
def test_hook_denies_generated_outside_roots(path):
    assert _hook("Write", path).returncode == 2


@pytest.mark.parametrize("path", [
    "artifacts/plots/foo.png", "runs/gemmini/s/r/perf_results.json",
    "merlin/python/merlin/x.py", "output/AGENT.md", "build/x.o",
])
def test_hook_allows_sanctioned_and_source(path):
    assert _hook("Write", path).returncode == 0


def test_hook_ignores_non_write_tools():
    assert _hook("Read", "output/foo.png").returncode == 0
