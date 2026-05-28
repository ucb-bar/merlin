"""Smoke test for the paper baseline script.

The script wraps ``score_against_history`` over every target in
``.bring_up_commits.yaml``. The numbers themselves are tested in
``tests/integration/test_retrospective_accuracy.py`` (which runs only
under the ``integration`` marker, since it shells out to ``git``); this
test pins the CLI surface — argparse, output paths, JSON shape — without
re-running the git work.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "eval" / "paper" / "baselines" / "score_baselines.py"
BRING_UP = REPO_ROOT / "tests" / "integration" / ".bring_up_commits.yaml"


@pytest.mark.skipif(not (REPO_ROOT / ".git").is_dir(), reason="needs a git checkout")
def test_score_baselines_script_runs(tmp_path: Path) -> None:
    out_dir = tmp_path / "results"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--bring-up",
            str(BRING_UP),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert (
        result.returncode == 0
    ), f"script failed (rc={result.returncode}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    json_path = out_dir / "baseline_scores.json"
    md_path = out_dir / "baseline_scores.md"
    assert json_path.is_file()
    assert md_path.is_file()
    payload = json.loads(json_path.read_text())
    assert "targets" in payload
    assert payload["targets"], "expected at least one target row"
    for row in payload["targets"]:
        assert {"target", "precision", "recall", "f1", "primary_integration"} <= set(row)
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0
