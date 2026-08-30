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
from merlin.common.yaml import dump_yaml, load_yaml
from merlin.targetgen import publish as pub


# --------------------------------------------------------------------------- fixtures


def _cxx_toolchain_available() -> bool:
    # Probe cmake by RUNNING it, not by finding it: sourcing the chipyard/Vitis environment puts
    # a cmake 3.3.2 on PATH that is linked against a libidn.so.11 no current distro ships, so
    # shutil.which succeeds and every configure step then dies with a loader error.
    from merlin.targetgen.oot_runner import usable_cmake
    if usable_cmake() == "cmake" and not shutil.which("cmake"):
        return False
    if subprocess.run([usable_cmake(), "--version"], capture_output=True).returncode != 0:
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


def test_resolve_branch_policy_and_precedence(out_root, monkeypatch):
    troot = out_root / "artifacts" / "targets"
    _make_rvv_package(troot, "hand_v0")          # a frozen baseline id
    _make_rvv_package(troot, "impr_tuned_a")     # a champion id
    base = pub.select_champion("rvv", package_id="hand_v0")
    champ = pub.select_champion("rvv", package_id="impr_tuned_a")
    # default policy: frozen baseline -> shared `baseline`; champion -> `stable/<pkg>`
    assert pub.resolve_branch(base) == "baseline"
    assert pub.resolve_branch(champ) == "stable/impr_tuned_a"
    # a manifest publication.role: baseline opts a renamed control into the baseline branch
    from merlin.common.yaml import load_yaml, write_yaml
    man = load_yaml(troot / "rvv" / "impr_tuned_a" / "manifest.yaml")
    man["publication"] = {"role": "baseline"}
    write_yaml(troot / "rvv" / "impr_tuned_a" / "manifest.yaml", man)
    assert pub.resolve_branch(pub.select_champion("rvv", package_id="impr_tuned_a")) == "baseline"
    # override wins over everything
    assert pub.resolve_branch(base, override="custom/x") == "custom/x"
    # env beats the default policy
    monkeypatch.setenv("MERLIN_PUBLISH_BRANCH_RVV", "from-env")
    assert pub.resolve_branch(champ) == "from-env"


def test_resolve_remote_precedence(out_root, monkeypatch):
    # override wins
    assert pub.resolve_remote("rvv", override="file:///x") == "file:///x"
    # env beats config file
    monkeypatch.setenv("MERLIN_PUBLISH_REMOTE_RVV", "file:///from-env")
    assert pub.resolve_remote("rvv") == "file:///from-env"
    monkeypatch.delenv("MERLIN_PUBLISH_REMOTE_RVV")
    # falls back to merlin/targets/publish.yaml. The repo holds ALL host codegen -- scalar is the
    # case where no vector unit is declared, not a separate target -- so there is no second scalar
    # repo. An earlier plan called it `host-mlir` for that reason; the repo that actually exists is
    # `rvv-mlir`, which is also the `<target>-mlir` default, so no name override is configured.
    assert pub.resolve_remote("rvv").endswith("rvv-mlir.git")
    assert pub.resolve_repo_name("rvv") == "rvv-mlir"


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
    # BB0: the frozen hand_v0 baseline publishes to the shared `baseline` branch (not the default HEAD)
    assert res.branch == "baseline"
    # commit present on the remote BRANCH (bare HEAD is unset — publishing targets refs/heads/<branch>)
    log = subprocess.run(["git", "-C", bare, "log", res.branch, "--oneline"], capture_output=True, text=True)
    assert res.commit_sha[:7] in log.stdout
    # tag present
    tags = subprocess.run(["git", "-C", bare, "tag"], capture_output=True, text=True).stdout.split()
    assert res.tag in tags
    # fingerprint trailer embedded in the commit message
    body = subprocess.run(["git", "-C", bare, "log", "-1", "--format=%B", res.branch],
                          capture_output=True, text=True).stdout
    assert f"Merlin-Publish-Fingerprint: {res.fingerprint}" in body
    # a publish event was recorded as a versioned product
    assert res.product_dir is not None and (res.product_dir / "manifest.yaml").is_file()

    # (d) idempotent re-publish -> no-op (per-branch fingerprint match)
    res2 = pub.publish("rvv", dry_run=False)
    assert res2.noop and not res2.committed
    count = subprocess.run(["git", "-C", bare, "rev-list", "--count", res.branch],
                           capture_output=True, text=True).stdout.strip()
    assert count == "1"


# --------------------------------------------------------------------------- diff-confirm gate


def test_needs_push_confirmation_classifies_remotes(tmp_path):
    # non-local network remotes require confirmation
    assert pub._needs_push_confirmation("git@github.com:ucb-bar/rvv-mlir.git")
    assert pub._needs_push_confirmation("https://github.com/ucb-bar/rvv-mlir.git")
    assert pub._needs_push_confirmation("ssh://git@host/repo.git")
    # local / file remotes (the verification + test path) are exempt
    assert not pub._needs_push_confirmation("file:///tmp/rvv-mlir.git")
    assert not pub._needs_push_confirmation(str(tmp_path))
    assert not pub._needs_push_confirmation("/srv/git/rvv-mlir.git")


def test_require_push_confirmation_token_must_match_fingerprint(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "manifest.yaml").write_text("x: 1\n")
    fp = "abc123"
    # non-local + wrong/absent token -> refuse (and the message lists the assembled tree)
    with pytest.raises(pub.PublishError, match="REFUSED"):
        pub._require_push_confirmation("git@github.com:o/r.git", repo, "stable/x", fp, None)
    with pytest.raises(pub.PublishError):
        pub._require_push_confirmation("git@github.com:o/r.git", repo, "stable/x", fp, "wrong")
    # matching token -> passes; and a local remote never needs a token
    pub._require_push_confirmation("git@github.com:o/r.git", repo, "stable/x", fp, fp)
    pub._require_push_confirmation("file:///tmp/r.git", repo, "stable/x", fp, None)


def test_real_push_to_github_refused_without_confirmation(out_root):
    """End-to-end: a real (non-dry-run) publish to a github-style remote is refused BEFORE any git
    clone/push, so no network is touched. dry-run to the same remote is unaffected (returns first)."""
    troot = out_root / "artifacts" / "targets"
    _make_rvv_package(troot)
    remote = "git@github.com:ucb-bar/rvv-mlir.git"
    with pytest.raises(pub.PublishError, match="REFUSED|confirm"):
        pub.publish("rvv", dry_run=False, remote=remote)
    # dry-run still plans fine (no confirmation needed, no network)
    res = pub.publish("rvv", dry_run=True, remote=remote)
    assert res.dry_run and res.remote == remote


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
    # BB0: publishing targets refs/heads/<branch>, so a fresh clone must ask for that branch.
    subprocess.run(["git", "clone", "-q", "-b", res.branch, f"file://{bare}", str(clone)], check=True)
    # the clone's manifest.yaml == the committed .merlin/manifest.yaml (provenance copy)
    assert (clone / ".merlin" / "manifest.yaml").read_text() == (clone / "manifest.yaml").read_text()

    pkg = oot_runner.load_package(clone)
    assert pkg.manifest["build"]["tool_output"] == f"build/bin/{target}-opt"
    oot_runner.build_package(pkg)
    assert pkg.tool.exists(), f"expected built tool at {pkg.tool}"


# --------------------------------------------------------------- repo name & recorded certification


def test_repo_name_defaults_to_target_and_is_overridable(out_root, monkeypatch, tmp_path):
    """The public repo name is config, not the target key.

    They are deliberately separable: the host target is keyed `rvv` because its payload is a vector
    schedule, but the repo holds all host codegen (scalar is the no-vector case, not a second
    target), so it publishes as `host-mlir`. The tool name stays `<target>-opt` either way, because
    that is what the build contract names.
    """
    assert pub.resolve_repo_name("gemmini") == "gemmini-mlir"
    cfg = tmp_path / "publish.yaml"
    cfg.write_text(dump_yaml({"targets": {"rvv": "git@example:x.git"},
                              "repo_names": {"rvv": "host-mlir"}}))
    assert pub.resolve_repo_name("rvv", config=cfg) == "host-mlir"
    assert pub.resolve_repo_name("rvv", config=cfg, override="other") == "other"
    monkeypatch.setenv("MERLIN_PUBLISH_REPO_NAME_RVV", "env-mlir")
    assert pub.resolve_repo_name("rvv", config=cfg) == "env-mlir"


def _cert_results(d: Path, run_id: str, *, status="pass", rtl=True, cycles=100) -> Path:
    _write(d / "results.yaml", dump_yaml({
        "status": status, "rung": run_id, "run_id": run_id,
        "oracle": {"kind": "rtl_verilator" if rtl else "spike_gemmini_functional",
                   "derived_from_rtl": rtl, "cycle_accurate": rtl,
                   "result": status, "cycles": cycles}}))
    return d


def test_record_certification_breaks_the_promote_gate_circularity(out_root, tmp_path):
    """A certify verdict has to reach the manifest, or nothing can ever be promoted.

    `promote` writes `publication.certification` only when the gate already passes, and the gate
    asks for that same field -- so a package carrying a real out-of-tree dialect could never be
    promoted, and the only publishable champion was a hand baseline whose repo builds a stub.
    """
    troot = out_root / "artifacts" / "targets"
    _make_gemmini_package(troot, "oot_pkg")
    # strip the fixture's pre-baked status so only a recorded certification can open the gate
    man_path = troot / "gemmini" / "oot_pkg" / "manifest.yaml"
    man = load_yaml(man_path); man["status"] = ""; _write(man_path, dump_yaml(man))

    with pytest.raises(pub.PublishError):
        pub.promote("gemmini", "oot_pkg")

    pub.record_certification("gemmini", "oot_pkg",
                             [_cert_results(tmp_path / "r1", "g3"), _cert_results(tmp_path / "r2", "g4")])
    pub.promote("gemmini", "oot_pkg")        # no longer refused
    sel = pub.select_champion("gemmini", package_id="oot_pkg")
    assert sel.cert_status == "pass"


def test_recorded_certification_keeps_its_tier(out_root, tmp_path):
    """RTL and functional passes are both `pass` to the gate but are not the same claim."""
    troot = out_root / "artifacts" / "targets"
    _make_gemmini_package(troot, "p")
    got = pub.record_certification("gemmini", "p", [_cert_results(tmp_path / "r", "g3", rtl=True)])
    assert got["certification_tier"] == {"derived_from_rtl": True, "cycle_accurate": True,
                                         "oracles": ["rtl_verilator"]}
    assert "cycle-accurate RTL" in pub._tier_phrase(got["certification_tier"])


def test_a_mixed_tier_records_as_the_weakest_rung(out_root, tmp_path):
    """A package is only as certified as its least-certified rung; quoting the best one overclaims."""
    troot = out_root / "artifacts" / "targets"
    _make_gemmini_package(troot, "p")
    got = pub.record_certification("gemmini", "p", [
        _cert_results(tmp_path / "a", "g3", rtl=True),
        _cert_results(tmp_path / "b", "g4", rtl=False)])
    assert got["certification"] == "pass"
    assert got["certification_tier"]["derived_from_rtl"] is False
    assert "not** an RTL" in pub._tier_phrase(got["certification_tier"])


def test_a_failing_rung_fails_the_whole_certification(out_root, tmp_path):
    troot = out_root / "artifacts" / "targets"
    _make_gemmini_package(troot, "p")
    got = pub.record_certification("gemmini", "p", [
        _cert_results(tmp_path / "a", "g3"),
        _cert_results(tmp_path / "b", "g4", status="fail")])
    assert got["certification"] == "fail"


def test_an_unnamed_oracle_records_unknown_not_a_benign_default(out_root, tmp_path):
    """Fail closed: a results file with no oracle block must not read as a passing RTL tier."""
    troot = out_root / "artifacts" / "targets"
    _make_gemmini_package(troot, "p")
    _write(tmp_path / "r" / "results.yaml", dump_yaml({"status": "pass", "run_id": "g3"}))
    got = pub.record_certification("gemmini", "p", [tmp_path / "r"])
    assert got["certification_tier"]["oracles"] == ["UNKNOWN"]
    assert got["certification_tier"]["derived_from_rtl"] is False


def test_tier_phrase_refuses_to_imply_rtl_when_nothing_was_recorded():
    assert "not recorded" in pub._tier_phrase(None)
    assert "not recorded" in pub._tier_phrase({})
