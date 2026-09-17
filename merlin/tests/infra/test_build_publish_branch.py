"""The publish-branch builder's one claim: a different history to the SAME tree.

It exists so a many-thousand-commit working branch can be reviewed as a handful of subsystem commits
without rewriting that branch. If its final tree ever differed from the tip's, the published branch
would ship content nobody reviewed on the working branch -- so that equality is asserted, along with
the properties that make it safe to run beside other sessions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from merlin.common.paths import repo_root

BUILDER = repo_root() / "build_tools" / "scripts" / "build_publish_branch.py"


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        _git(root, *args)
    (root / "lib").mkdir()
    (root / "lib" / "a.py").write_text("a = 1\n")
    (root / "doc.md").write_text("old\n")
    (root / "gone.txt").write_text("bye\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    base = _git(root, "rev-parse", "HEAD")
    (root / "lib" / "a.py").write_text("a = 2\n")
    (root / "lib" / "b.py").write_text("b = 1\n")
    (root / "doc.md").write_text("new\n")
    (root / "run.sh").write_text("#!/bin/sh\n")
    os.chmod(root / "run.sh", 0o755)
    os.symlink("lib/a.py", root / "link.py")
    (root / "gone.txt").unlink()
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "tip")
    return root, base, _git(root, "rev-parse", "HEAD")


def _plan(tmp_path, groups):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"groups": groups}))
    return path


def _run(root, *args):
    return subprocess.run(
        [sys.executable, str(BUILDER), *args],
        cwd=root,
        capture_output=True,
        text=True,
        env=dict(os.environ, TMPDIR=str(root.parent)),
    )


def test_the_series_ends_on_the_tips_exact_tree(tmp_path):
    root, base, tip = _repo(tmp_path)
    plan = _plan(
        tmp_path,
        [
            {"id": "lib", "paths": ["lib", "link.py", "run.sh"], "subject": "feat(lib): lib", "body": "b"},
            {"id": "rest", "paths": ["doc.md", "gone.txt"], "subject": "docs(doc): doc", "body": "b"},
        ],
    )
    r = _run(root, base, tip, str(plan), "refs/heads/publish")
    assert r.returncode == 0, r.stdout + r.stderr
    assert _git(root, "rev-parse", "publish^{tree}") == _git(root, "rev-parse", f"{tip}^{{tree}}")
    subjects = _git(root, "log", "--format=%s", f"{base}..publish").splitlines()
    assert subjects == ["docs(doc): doc", "feat(lib): lib"]
    # Mode and symlink carried exactly, not re-derived from a checkout.
    assert _git(root, "ls-tree", "publish", "run.sh").split()[0] == "100755"
    assert _git(root, "ls-tree", "publish", "link.py").split()[0] == "120000"


def test_a_path_no_group_claims_is_an_error_not_an_omission(tmp_path):
    root, base, tip = _repo(tmp_path)
    plan = _plan(tmp_path, [{"id": "lib", "paths": ["lib"], "subject": "feat(lib): lib", "body": "b"}])
    r = _run(root, base, tip, str(plan))
    assert r.returncode != 0 and "UNGROUPED" in r.stderr


def test_an_existing_branch_is_never_overwritten(tmp_path):
    root, base, tip = _repo(tmp_path)
    _git(root, "branch", "publish", base)
    plan = _plan(
        tmp_path,
        [
            {
                "id": "all",
                "paths": ["lib", "link.py", "run.sh", "doc.md", "gone.txt"],
                "subject": "feat(x): x",
                "body": "b",
            },
        ],
    )
    r = _run(root, base, tip, str(plan), "refs/heads/publish")
    assert r.returncode != 0
    assert _git(root, "rev-parse", "publish") == base


def test_the_working_tree_and_index_are_untouched(tmp_path):
    root, base, tip = _repo(tmp_path)
    (root / "doc.md").write_text("uncommitted edit\n")
    _git(root, "add", "doc.md")
    before = _git(root, "status", "--porcelain")
    plan = _plan(
        tmp_path,
        [
            {
                "id": "all",
                "paths": ["lib", "link.py", "run.sh", "doc.md", "gone.txt"],
                "subject": "feat(x): x",
                "body": "b",
            },
        ],
    )
    assert _run(root, base, tip, str(plan), "refs/heads/publish").returncode == 0
    assert _git(root, "status", "--porcelain") == before
