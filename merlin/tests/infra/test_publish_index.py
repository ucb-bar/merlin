"""The published repo's landing page (``merlin-target-publish index``).

Branch-per-version publishing puts every package on its own ``stable/<pkg>`` (or ``baseline``)
branch, which leaves the repo's DEFAULT branch empty — `git clone` with no `-b` yields a repo
containing nothing and no hint that the content is elsewhere. That was the real state of
ucb-bar/rvv-mlir. These tests pin the two properties that make the fix trustworthy:

  * the page lists ONLY branches that exist on the remote (never advertises a dead branch), and
  * it is a directory page, not a package tree (a consumer must not build the default branch).

Everything runs against a LOCAL bare remote (``file://``) — the harness never pushes to GitHub.
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.targetgen import publish as P


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          timeout=120, check=True)


@pytest.fixture()
def bare_remote(tmp_path):
    """A local bare repo with one package branch already published, plus an empty default."""
    bare = tmp_path / "rvv-mlir-bare.git"
    _git("init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    seed.mkdir()
    _git("init", "-b", "main", str(seed))
    (seed / "LICENSE").write_text("seed\n")
    _git("add", "--", ".", cwd=seed)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "seed", cwd=seed)
    _git("remote", "add", "origin", f"file://{bare}", cwd=seed)
    _git("push", "origin", "main", cwd=seed)
    # one package branch, as a real publish would leave it
    _git("checkout", "--orphan", "stable/impr_tuned_wholemodel_vf_int8", cwd=seed)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty",
         "-m", "pkg", cwd=seed)
    _git("push", "origin", "stable/impr_tuned_wholemodel_vf_int8", cwd=seed)
    return f"file://{bare}"


def test_index_lists_only_certified_packages():
    """Uncertified packages are never published, so the page must not point at them."""
    entries = P.index_entries("rvv")
    assert entries, "no certified rvv packages found"
    for e in entries:
        sel = P.select_champion("rvv", package_id=e["package_id"])
        ok, _ = P._check_gate(sel)
        assert ok, f"{e['package_id']} is listed but would not pass the publish gate"
    # the int8 and fp32 families are distinguished, so an int8 consumer is not sent to fp32
    assert {e["dtype"] for e in entries} >= {"fp32", "int8_w8a8"}


def test_index_tree_is_a_landing_page_not_a_package(tmp_path):
    dest = tmp_path / "repo"
    info = P.assemble_index_tree("rvv", dest)
    readme = (dest / "README.md").read_text()
    assert (dest / "LICENSE").is_file()
    # a directory, not something to build
    assert "branch-per-version" in readme
    assert "git clone -b <branch>" in readme
    for name in ("CMakeLists.txt", "manifest.yaml", "payload"):
        assert not (dest / name).exists(), f"{name} would make the default branch look buildable"
    for e in info["entries"]:
        assert e["branch"] in readme and e["package_id"] in readme


def test_index_never_advertises_a_branch_absent_from_the_remote(tmp_path):
    """The whole point: a listed branch a consumer checks out must exist."""
    dest = tmp_path / "repo"
    info = P.assemble_index_tree("rvv", dest, only_branches={"stable/impr_tuned_wholemodel_vf_int8"})
    assert [e["branch"] for e in info["entries"]] == ["stable/impr_tuned_wholemodel_vf_int8"]
    readme = (dest / "README.md").read_text()
    assert "stable/rvv_tuned_v1_d1_vfmacc_outerproduct" not in readme


def test_publish_index_dry_run_touches_no_remote(bare_remote):
    res = P.publish_index("rvv", remote=bare_remote, dry_run=True)
    assert res["dry_run"] and not res.get("commit_sha")
    assert any("dry-run" in a for a in res["actions"])


def test_publish_index_execute_writes_default_branch(bare_remote, tmp_path):
    res = P.publish_index("rvv", remote=bare_remote, dry_run=False)
    assert res.get("commit_sha"), f"index not pushed: {res['actions']}"
    # only the branch that actually exists on this remote is listed
    assert [e["branch"] for e in res["entries"]] == ["stable/impr_tuned_wholemodel_vf_int8"]

    clone = tmp_path / "clone"
    _git("clone", bare_remote, str(clone))
    readme = (clone / "README.md").read_text()
    assert "stable/impr_tuned_wholemodel_vf_int8" in readme
    assert "stable/rvv_tuned_v1_d1_vfmacc_outerproduct" not in readme

    # idempotent: republishing the same state is a no-op
    again = P.publish_index("rvv", remote=bare_remote, dry_run=False)
    assert again["noop"], f"second publish was not a no-op: {again['actions']}"


def test_publish_index_refuses_unconfirmed_push_to_github():
    """A non-local remote needs the explicit fingerprint, same as `publish`."""
    assert P._needs_push_confirmation("git@github.com:ucb-bar/rvv-mlir.git")
    assert not P._needs_push_confirmation("file:///tmp/whatever.git")


def test_baseline_branch_is_per_datatype():
    """The frozen controls must not share one branch.

    `_BASELINE_PACKAGE_IDS` holds one control per datatype. Mapping them all to `baseline` made
    them overwrite each other on publish, so "the" control became whichever went last -- and a
    speedup claimed against `baseline` could be measured against the wrong datatype's schedule.
    fp32 keeps the bare historical name so existing published branches stay valid.
    """
    from merlin.targetgen import publish as P

    fp32 = P.select_champion("rvv", package_id="hand_v0")
    int8 = P.select_champion("rvv", package_id="hand_v0_int8")
    assert P._is_baseline(fp32) and P._is_baseline(int8)

    b32, b8 = P.resolve_branch(fp32), P.resolve_branch(int8)
    assert b32 == P.BASELINE_BRANCH == "baseline"
    assert b8 == "baseline-int8_w8a8"
    assert b32 != b8
