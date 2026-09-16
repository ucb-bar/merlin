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
    assert rel[:5] == ("out", "artifacts", "dse", "bitvla", "v1")  # single out/ root; topic+target+version
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
    assert d.relative_to(tmp_path).parts == ("out", "artifacts", "cache", "kc") and d.is_dir()


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


def test_linter_flags_generated_output_inside_merlin(tmp_path):
    # generated output dirs must not live in the source tree; curated inputs may.
    _run(["git", "init", "-q"], cwd=tmp_path)
    for rel in ("merlin/experiments/e/reports/r.md",
                "merlin/benchmarks/b/case_study/c.yaml"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    # an INPUT corpus path that must NOT be flagged
    ok = tmp_path / "merlin/benchmarks/b/recaptures/m/model.mlir"
    ok.parent.mkdir(parents=True, exist_ok=True)
    ok.write_text("mlir")
    _run(["git", "add", "-A", "-f"], cwd=tmp_path)
    r = _run([sys.executable, str(LINT)], cwd=tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 1
    assert "reports/r.md" in out and "case_study/c.yaml" in out
    assert "recaptures" not in out  # curated input corpus is allowed to stay


def _init_out_repo(tmp_path):
    _run(["git", "init", "-q"], cwd=tmp_path)
    _run(["git", "config", "user.email", "t@t"], cwd=tmp_path)
    _run(["git", "config", "user.name", "t"], cwd=tmp_path)
    (tmp_path / ".gitignore").write_text(
        "/out/**\n!/out/\n!/out/**/\n!/out/artifacts/concern/*\n/out/artifacts/concern/*/**\n")
    _run(["git", "add", ".gitignore"], cwd=tmp_path)
    _run(["git", "commit", "-qm", "ignore"], cwd=tmp_path)


def test_linter_staged_rejects_force_added_file_under_ignored_out(tmp_path):
    # A gitignore rule cannot stop `git add -f`; the staged gate must. The dump lives one level
    # below the concern root, exactly the shape of the ~1,500-file leak this guards against.
    _init_out_repo(tmp_path)
    dump = tmp_path / "out/artifacts/concern/run_1/receipt.json"
    dump.parent.mkdir(parents=True)
    dump.write_text("{}")
    _run(["git", "add", "-f", str(dump)], cwd=tmp_path)
    r = _run([sys.executable, str(LINT), "--staged"], cwd=tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "force-added" in out and "run_1/receipt.json" in out


def test_linter_staged_accepts_negated_curated_file(tmp_path):
    # The concern's own top-level file is negated: tracking it is the reviewed path, not a force.
    _init_out_repo(tmp_path)
    story = tmp_path / "out/artifacts/concern/summary.md"
    story.parent.mkdir(parents=True)
    story.write_text("# story\n")
    _run(["git", "add", str(story)], cwd=tmp_path)
    r = _run([sys.executable, str(LINT), "--staged"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


def test_linter_staged_still_flags_modified_tracked_dump_only_as_stale_literal_free(tmp_path):
    # A MODIFICATION to an already-tracked ignored file is not a force-add (the debt is untracking,
    # handled separately); the guard must look only at additions so it cannot block every edit
    # to legacy tracked dumps.
    _init_out_repo(tmp_path)
    dump = tmp_path / "out/artifacts/concern/run_1/receipt.json"
    dump.parent.mkdir(parents=True)
    dump.write_text("{}")
    _run(["git", "add", "-f", str(dump)], cwd=tmp_path)
    _run(["git", "commit", "-qm", "legacy"], cwd=tmp_path)
    dump.write_text('{"n": 1}')
    _run(["git", "add", str(dump)], cwd=tmp_path)
    r = _run([sys.executable, str(LINT), "--staged"], cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


# ----------------------------------------------------------------- guard hook


def _hook(tool, path):
    payload = json.dumps({"tool_name": tool, "tool_input": {"file_path": path}})
    return subprocess.run([sys.executable, str(HOOK)], input=payload,
                          capture_output=True, text=True, cwd=REPO)


@pytest.mark.parametrize("path", [
    "output/foo.png", "results/x_dse_analysis/r.csv", "selfcheck_out/log.jsonl",
    "merlin/x.svg", "artifacts/plots/foo.png",   # retired top-level roots now denied (out/ only)
])
def test_hook_denies_generated_outside_roots(path):
    assert _hook("Write", path).returncode == 2


@pytest.mark.parametrize("path", [
    "out/artifacts/presentation/set-a/foo.png", "out/runs/gemmini/s/r/perf_results.json",
    "merlin/python/merlin/x.py", "output/AGENT.md", "out/build/x.o",
])
def test_hook_allows_sanctioned_and_source(path):
    assert _hook("Write", path).returncode == 0


def test_hook_ignores_non_write_tools():
    assert _hook("Read", "output/foo.png").returncode == 0


# The out/ root's own shape. The guard blocked writes OUTSIDE out/, so nothing was watching what
# happened inside it: four undeclared top-level roots and 52 undeclared concerns accumulated against
# 16 declared ones, none of which the tracked-file linter can see because generated output is
# gitignored by design. The roster the guard reads is merlin/contract/storage.yaml, so a new concern
# costs a reviewed line in that file at the moment it is created.


@pytest.mark.parametrize("path", [
    "out/artifacts/a-concern-nobody-declared/axis/run/result.json",
    "out/a-root-nobody-declared/some-study/report.md",
    "out/scratch/whatever.json",
])
def test_hook_denies_undeclared_destinations_inside_out(path):
    # These names are deliberately ones no fold has created. The hook resolves the path first, so a
    # write through a symlink an `organize` fold left behind lands in the DECLARED concern it points
    # at and is allowed -- which is the behaviour that makes the folds safe, not a hole in the rule.
    result = _hook("Write", path)
    assert result.returncode == 2
    assert "storage.yaml" in result.stderr


def test_hook_allows_every_concern_the_contract_declares():
    """Derived from the roster rather than a second list of names: a copy would drift from it, which
    is the exact failure this rule exists to stop."""
    from merlin.common import storage_cli as SC

    for concern in SC.declared_concerns():
        path = f"out/artifacts/{concern}/axis/unit/result.json"
        assert _hook("Write", path).returncode == 0, f"{concern} is declared but blocked"


def test_hook_still_allows_writes_when_the_roster_cannot_be_read(tmp_path):
    """The guard's standing contract is to never block on input it cannot read. A checkout without
    the contract -- or with one this small parser does not understand -- must not become unwritable."""
    payload = json.dumps({"tool_name": "Write",
                          "tool_input": {"file_path": "out/artifacts/anything/a/b.json"}})
    env = {**os.environ, "MERLIN_REPO_ROOT": str(tmp_path)}
    result = subprocess.run([sys.executable, str(HOOK)], input=payload, capture_output=True,
                            text=True, cwd=REPO, env=env)
    assert result.returncode == 0
