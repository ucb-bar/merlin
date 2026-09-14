"""Regression guard for the docs anti-drift system (Phase-4/5 of the docs restructure).

Asserts the committed tree is self-consistent: generated docs are fresh, front-matter is
schema-valid, the hub is in sync, no retired paths, and the drift detector actually fires.
"""
from __future__ import annotations

import subprocess
import sys

from merlin.common.paths import repo_root

ROOT = repo_root()
SCRIPTS = ROOT / "build_tools" / "scripts"


def _run(script: str, *args: str):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                          capture_output=True, text=True)


def test_docs_aggregator_clean():
    r = _run("check_docs.py")
    assert r.returncode == 0, r.stderr + r.stdout


def test_hub_and_generated_docs_fresh():
    for script in ("gen_docs_index.py", "gen_cli_docs.py",
                   "gen_package_docs.py", "gen_schema_docs.py"):
        r = _run(script, "--check")
        assert r.returncode == 0, f"{script} stale:\n{r.stderr}{r.stdout}"


def test_front_matter_schema_valid():
    r = _run("check_docs_freshness.py", "--check")
    assert r.returncode == 0, r.stderr + r.stdout


def test_no_retired_paths():
    r = _run("check_doc_paths.py", "--check")
    assert r.returncode == 0, r.stderr + r.stdout


def test_drift_detector_reports_json():
    """--json must return a well-formed worklist (the docs-doctor backbone)."""
    import json
    r = _run("check_docs_freshness.py", "--json")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert "drift" in data and "uncategorized" in data
    assert isinstance(data["drift"], list)


def test_freshness_ratchet_fails_on_new_drift_and_passes_when_ratcheted(tmp_path):
    """--ratchet must fail on a drifted doc that is not in the ledger, pass once it is, and say so
    when a ratcheted doc has healed. Built in a throwaway repo so real-tree drift cannot mask it."""
    import shutil
    scripts = tmp_path / "build_tools" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPTS / "check_docs_freshness.py", scripts / "check_docs_freshness.py")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    doc = tmp_path / "docs" / "guides" / "g.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("---\ntitle: g\nkind: guide\nstatus: current\nowner: core\n"
                   "last_verified: 2000-01-01\ncode_refs: [src/a.py]\n---\n# g\n")
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["add", "-A"], ["commit", "-qm", "c"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    gate = [sys.executable, str(scripts / "check_docs_freshness.py"), "--ratchet"]
    r = subprocess.run(gate, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 1 and "guides/g.md" in r.stderr, r.stdout + r.stderr
    (scripts / "docs_freshness_ratchet.txt").write_text("# debt\nguides/g.md\n")
    r = subprocess.run(gate, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    doc.write_text(doc.read_text().replace("2000-01-01", "2999-01-01"))
    r = subprocess.run(gate, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and "no longer drift" in r.stdout, r.stdout + r.stderr
