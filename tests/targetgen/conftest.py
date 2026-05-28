"""Shared fixtures and helpers for TargetGen tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


@pytest.fixture(scope="session")
def merlin_repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def capability_examples_dir() -> Path:
    return REPO_ROOT / "target_specs" / "examples"


def all_capability_specs() -> list[Path]:
    """All capability.yaml files under target_specs/examples/, sorted."""
    examples = REPO_ROOT / "target_specs" / "examples"
    return sorted(examples.glob("*/capability.yaml"))


@pytest.fixture(scope="session")
def all_capability_paths() -> list[Path]:
    return all_capability_specs()


def assert_text_equals(
    actual: str,
    expected_path: Path,
    *,
    label: str = "snapshot",
) -> None:
    """Compare ``actual`` text against the snapshot at ``expected_path``.

    Set ``UPDATE_SNAPSHOTS=1`` to (re)generate the file. The snapshot is
    written exactly as ``actual`` (no normalisation), so callers must
    normalise before passing in.
    """
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(actual, encoding="utf-8")
        return
    if not expected_path.exists():
        pytest.fail(f"{label} snapshot missing: {expected_path}.\n" f"Run with UPDATE_SNAPSHOTS=1 to create it.")
    expected = expected_path.read_text(encoding="utf-8")
    if actual != expected:
        # Build a compact diff for the assertion message.
        import difflib

        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=str(expected_path),
                tofile=f"{label}-actual",
                lineterm="",
                n=3,
            )
        )
        pytest.fail(
            f"{label} drift detected vs {expected_path}.\n"
            f"Run with UPDATE_SNAPSHOTS=1 to refresh after intentional changes.\n\n"
            f"{diff}"
        )


def normalise_json(obj: Any) -> str:
    """Stable JSON dump for snapshot comparisons (sorted keys, 2-space indent)."""
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"
