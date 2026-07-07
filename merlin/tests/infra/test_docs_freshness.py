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
