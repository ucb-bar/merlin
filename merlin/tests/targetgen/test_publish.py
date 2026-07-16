"""Tests for the WS-E target-publishing bridge (merlin.targetgen.publish).

The heavy real champion packages are NOT depended on: each test builds a minimal fixture package
under an isolated ``MERLIN_OUT_ROOT`` tmp tree and verifies the publish flow against a LOCAL bare
git remote (``file://…``) — never an external/GitHub remote. The clone+cmake build proof skips
cleanly if a cmake / C++ toolchain is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common import paths
from merlin.common.yaml import dump_yaml
from merlin.targetgen import publish as pub


# --------------------------------------------------------------------------- fixtures


def _cxx_toolchain_available() -> bool:
    if not shutil.which("cmake"):
        return False
    return any(shutil.which(c) for c in ("c++", "g++", "clang++"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_rvv_package(targets_root: Path, package_id: str = "hand_v0") -> Path:
    """A minimal rvv (vector_schedule) fixture package satisfying load_rvv_package."""
    d = targets_root / "rvv" / package_id
    manifest = {
        "target": "rvv",
        "run_id": package_id,
        "package_id": package_id,
        "family": "vector_schedule",
        "status": "spike_verified",
        "version": 0,
        "authoring": {"mode": "hand_curated", "author": "fixture", "generated_by_agent": False},
        "outputs": {"schedule": "schedule.mlir", "knobs": "knobs.yaml"},
    }
    knobs = {
        "schedule_file": "schedule.mlir",
        "op_match": [{"op": "linalg.matmul", "tile": [4, 8, 1], "vector": [4, 8, 1]}],
        "lowering_patterns": ["lower_contraction"],
        "lmul_policy": "m1",
        "cflags": ["-march=rv64gcv", "-fno-vectorize"],
        "dtype_strategy": "fp32",
        "expected_instructions": ["vsetivli", "vle32.v"],
    }
    _write(d / "manifest.yaml", dump_yaml(manifest))
    _write(d / "knobs.yaml", dump_yaml(knobs))
    _write(d / "schedule.mlir", "module attributes {transform.with_named_sequence} {\n}\n")
    _write(d / "baseline_runs" / "matmul_f32_64" / "results.yaml",
           dump_yaml({"status": "pass", "workload": "matmul_f32_64"}))
    return d


def _make_gemmini_package(targets_root: Path, package_id: str = "hand_v0") -> Path:
    """A minimal gemmini (tensor_resident, non-contract-shaped) fixture package."""
    d = targets_root / "gemmini" / package_id
    manifest = {
        "target": "gemmini",
        "run_id": package_id,
        "package_id": package_id,
        "family": "tensor_resident",
        "status": "rtl_certified",
        "version": 0,
        "authoring": {"mode": "hand_curated", "author": "fixture", "generated_by_agent": False},
        "outputs": {"dialect_module": "dialect.py", "lowering": "lowering.yaml"},
    }
    _write(d / "manifest.yaml", dump_yaml(manifest))
    _write(d / "dialect.py", "DIALECT_NAME = 'gemmini_fixture'\nSPEC_OPS = {}\n")
    _write(d / "lowering.yaml", dump_yaml({"interface_to_target": {}, "target_to_opcode": {}}))
    _write(d / "contracts" / "target_contract.yaml", dump_yaml({"target": "gemmini"}))
    _write(d / "inputs" / "rtl_facts.yaml", dump_yaml({"facts": []}))
    return d


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    """Isolate all generated output (targets, staging, remotes, product events) under tmp."""
    root = tmp_path / "out"
    (root / "artifacts" / "targets").mkdir(parents=True)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root))
    return root


def _bare_remote(out_root: Path, target: str, monkeypatch) -> str:
    bare = out_root / "build" / "publish" / "_fake_remotes" / f"{target}.git"
    bare.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    remote = f"file://{bare}"
    monkeypatch.setenv(f"MERLIN_PUBLISH_REMOTE_{target.upper()}", remote)
    return remote


# --------------------------------------------------------------------------- selection & remote


def test_select_champion_ranks_and_honors_package_id(out_root):
    troot = out_root / "artifacts" / "targets"
    _make_rvv_package(troot, "hand_v0")
    _make_rvv_package(troot, "cand_b")
    sel = pub.select_champion("rvv")
    assert sel.target == "rvv"
    assert sel.family == "vector_schedule"
    # explicit package_id overrides the ranking
    sel2 = pub.select_champion("rvv", package_id="cand_b")
    assert sel2.package_id == "cand_b"


def test_resolve_remote_precedence(out_root, monkeypatch):
    # override wins
    assert pub.resolve_remote("rvv", override="file:///x") == "file:///x"
    # env beats config file
    monkeypatch.setenv("MERLIN_PUBLISH_REMOTE_RVV", "file:///from-env")
    assert pub.resolve_remote("rvv") == "file:///from-env"
    monkeypatch.delenv("MERLIN_PUBLISH_REMOTE_RVV")
    # falls back to merlin/targets/publish.yaml
    assert pub.resolve_remote("rvv").endswith("rvv-mlir.git")


# --------------------------------------------------------------------------- (b) dry-run


def test_dry_run_makes_no_network_calls(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    _make_rvv_package(troot)
    remote = _bare_remote(out_root, "rvv", monkeypatch)
    res = pub.publish("rvv", dry_run=True)
    assert res.dry_run and not res.committed and res.commit_sha is None
    assert res.remote == remote
    assert any("DRY-RUN" in a for a in res.actions)
    # the assembled tree exists, but the remote received nothing
    assert (res.repo_dir / "manifest.yaml").is_file()
    bare = remote[len("file://"):]
    log = subprocess.run(["git", "-C", bare, "log", "--oneline"], capture_output=True, text=True)
    assert log.returncode != 0 or not log.stdout.strip()  # empty bare repo


# --------------------------------------------------------------------------- (a)+(d) publish


def test_publish_commits_tags_and_is_idempotent(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    _make_rvv_package(troot)
    remote = _bare_remote(out_root, "rvv", monkeypatch)
    bare = remote[len("file://"):]

    res = pub.publish("rvv", dry_run=False)
    assert res.committed and not res.noop and res.commit_sha
    # commit present on the remote
    log = subprocess.run(["git", "-C", bare, "log", "--oneline"], capture_output=True, text=True)
    assert res.commit_sha[:7] in log.stdout
    # tag present
    tags = subprocess.run(["git", "-C", bare, "tag"], capture_output=True, text=True).stdout.split()
    assert res.tag in tags
    # fingerprint trailer embedded in the commit message
    body = subprocess.run(["git", "-C", bare, "log", "-1", "--format=%B"],
                          capture_output=True, text=True).stdout
    assert f"Merlin-Publish-Fingerprint: {res.fingerprint}" in body
    # a publish event was recorded as a versioned product
    assert res.product_dir is not None and (res.product_dir / "manifest.yaml").is_file()

    # (d) idempotent re-publish -> no-op
    res2 = pub.publish("rvv", dry_run=False)
    assert res2.noop and not res2.committed
    count = subprocess.run(["git", "-C", bare, "rev-list", "--count", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    assert count == "1"


# --------------------------------------------------------------------------- (e) family parity


def test_family_parity_same_top_level_skeleton(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    _make_rvv_package(troot)
    _make_gemmini_package(troot)
    _bare_remote(out_root, "rvv", monkeypatch)
    _bare_remote(out_root, "gemmini", monkeypatch)

    rvv = pub.publish("rvv", dry_run=True)
    gem = pub.publish("gemmini", dry_run=True)

    def top(repo: Path) -> set[str]:
        return {p.name for p in repo.iterdir()}

    rvv_top, gem_top = top(rvv.repo_dir), top(gem.repo_dir)
    assert rvv_top == gem_top
    for required in {"CMakeLists.txt", "README.md", "include", "lib", "tools", "test",
                     "payload", "manifest.yaml", ".merlin"}:
        assert required in rvv_top
    # family-specific payloads differ
    assert (rvv.repo_dir / "payload" / "schedule.mlir").is_file()
    assert (gem.repo_dir / "payload" / "dialect.py").is_file()
    # both manifests validate against the contract manifest schema (repo root == {package})
    from merlin.targetgen.contract import schemas
    from merlin.common.yaml import load_yaml
    for r in (rvv, gem):
        schemas.validate_manifest(load_yaml(r.repo_dir / "manifest.yaml"))


# --------------------------------------------------------------------------- promote


def test_promote_sets_single_champion(out_root):
    troot = out_root / "artifacts" / "targets"
    _make_rvv_package(troot, "hand_v0")
    _make_rvv_package(troot, "cand_b")
    pub.promote("rvv", "hand_v0")
    from merlin.common.yaml import load_yaml
    m1 = load_yaml(troot / "rvv" / "hand_v0" / "manifest.yaml")
    assert m1["publication"]["champion"] is True
    assert m1["publication"]["fingerprint"]
    # select now returns the flagged champion deterministically
    assert pub.select_champion("rvv").package_id == "hand_v0"
    # re-promote the other clears the first (single-champion invariant)
    pub.promote("rvv", "cand_b")
    m1b = load_yaml(troot / "rvv" / "hand_v0" / "manifest.yaml")
    m2 = load_yaml(troot / "rvv" / "cand_b" / "manifest.yaml")
    assert m1b["publication"]["champion"] is False
    assert m2["publication"]["champion"] is True
    assert pub.select_champion("rvv").package_id == "cand_b"


def test_gate_refuses_uncertified(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    d = _make_rvv_package(troot, "wip")
    # downgrade status below the gate
    from merlin.common.yaml import load_yaml, write_yaml
    man = load_yaml(d / "manifest.yaml")
    man["status"] = "draft"
    write_yaml(d / "manifest.yaml", man)
    _bare_remote(out_root, "rvv", monkeypatch)
    with pytest.raises(pub.PublishError):
        pub.publish("rvv", dry_run=False, package_id="wip")
    # --no-gate overrides (dry-run so no git needed)
    res = pub.publish("rvv", dry_run=True, package_id="wip", gate=False)
    assert not res.gate_ok


# --------------------------------------------------------------------------- (c) clone + build


@pytest.mark.skipif(not _cxx_toolchain_available(), reason="cmake / C++ toolchain unavailable")
@pytest.mark.parametrize("target,maker", [("rvv", _make_rvv_package), ("gemmini", _make_gemmini_package)])
def test_fresh_clone_builds_tool(out_root, monkeypatch, target, maker):
    from merlin.targetgen import oot_runner
    troot = out_root / "artifacts" / "targets"
    maker(troot)
    remote = _bare_remote(out_root, target, monkeypatch)
    bare = remote[len("file://"):]
    res = pub.publish(target, dry_run=False)
    assert res.committed

    clone = out_root / "build" / f"_clone_{target}"
    subprocess.run(["git", "clone", "-q", f"file://{bare}", str(clone)], check=True)
    # the clone's manifest.yaml == the committed .merlin/manifest.yaml (provenance copy)
    assert (clone / ".merlin" / "manifest.yaml").read_text() == (clone / "manifest.yaml").read_text()

    pkg = oot_runner.load_package(clone)
    assert pkg.manifest["build"]["tool_output"] == f"build/bin/{target}-opt"
    oot_runner.build_package(pkg)
    assert pkg.tool.exists(), f"expected built tool at {pkg.tool}"
