"""The incremental format gate: Python a commit adds or changes must already be ruff-formatted.

~410 files were left unformatted on purpose because other work had them checked out, so the rule is
"whatever you touch, you format". These tests pin the two ways such a gate goes wrong: judging the
working tree instead of the bytes being committed, and passing when the formatter could not run.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_format.py"


def _ruff_available() -> bool:
    try:
        return subprocess.run(["uvx", "ruff@0.16.8", "--version"], capture_output=True, timeout=120).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(not _ruff_available(), reason="pinned ruff not runnable here")


def _repo(tmp_path):
    scripts = tmp_path / "build_tools" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(GATE, scripts / GATE.name)
    shutil.copy(repo_root() / "pyproject.toml", tmp_path / "pyproject.toml")
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def _gate(root):
    return subprocess.run(
        [sys.executable, str(root / "build_tools" / "scripts" / GATE.name), "--staged"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_unformatted_staged_python_is_refused(tmp_path):
    root = _repo(tmp_path)
    (root / "m.py").write_text("x=[1,2,\n3]\n")
    subprocess.run(["git", "add", "m.py"], cwd=root, check=True)
    r = _gate(root)
    assert r.returncode == 1 and "m.py" in r.stderr, r.stdout + r.stderr


def test_formatted_staged_python_passes(tmp_path):
    root = _repo(tmp_path)
    (root / "m.py").write_text("x = [1, 2, 3]\n")
    subprocess.run(["git", "add", "m.py"], cwd=root, check=True)
    r = _gate(root)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_staged_bytes_are_judged_not_the_working_copy(tmp_path):
    """Stage a formatted file, then scribble on the working copy: the commit is still clean."""
    root = _repo(tmp_path)
    (root / "m.py").write_text("x = [1, 2, 3]\n")
    subprocess.run(["git", "add", "m.py"], cwd=root, check=True)
    (root / "m.py").write_text("x=[1,2,\n3]\n")
    assert _gate(root).returncode == 0

    (root / "n.py").write_text("y=[1,2,\n3]\n")
    subprocess.run(["git", "add", "n.py"], cwd=root, check=True)
    (root / "n.py").write_text("y = [1, 2, 3]\n")  # tidy on disk, NOT what is staged
    r = _gate(root)
    assert r.returncode == 1 and "n.py" in r.stderr, r.stdout + r.stderr


def test_an_excluded_file_is_not_judged(tmp_path):
    """pyproject.toml's format exclusions (e.g. capsule loaders, a benchmark input) apply here too."""
    root = _repo(tmp_path)
    p = root / "merlin" / "contract" / "capsules" / "isa" / "X" / "capsule.pytorch.py"
    p.parent.mkdir(parents=True)
    p.write_text("x=[1,2,\n3]\n")
    subprocess.run(["git", "add", str(p.relative_to(root))], cwd=root, check=True)
    assert _gate(root).returncode == 0
