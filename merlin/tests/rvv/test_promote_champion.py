"""Tests for GAP 3: promoting a beam-verified RVV champion into the publishable package structure.

Mirrors ``tests/targetgen/test_publish.py``: everything runs under an isolated ``MERLIN_OUT_ROOT``
tmp tree and against a LOCAL bare ``file://`` remote -- never GitHub. A fixture beam run dir
(``beam_tree.yaml`` + minted fork packages) stands in for a real K1 beam run; the clone+cmake build
proof skips cleanly when no C++ toolchain is available.
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.yaml import dump_yaml, load_yaml, write_yaml
from merlin.rvvgen import promote_champion as pc
from merlin.rvvgen.registry import load_rvv_package
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


def _fork_pkg(targets_rvv: Path, run_id: str, *, version: int, depth: int,
              parent: str | None) -> Path:
    """Write a minimal beam fork package (mirrors merlin.rvvgen.fork.write_fork)."""
    d = targets_rvv / run_id
    manifest = {
        "target": "rvv",
        "run_id": run_id,
        "family": "vector_schedule",
        "schedule_format": "transform_dialect_mlir",
        "status": "proposed",
        "authoring": {"mode": "deterministic_generated_from_spec",
                      "generated_by_agent": False, "author": "rvvgen.from_strategy"},
        "lineage": {"parent_run_id": parent, "version": version, "depth": depth,
                    "lever": "feature", "source_evidence": ["census:byte-traffic"]},
        "outputs": {"schedule": "schedule.mlir", "knobs": "knobs.yaml"},
    }
    knobs = {
        "schedule_file": "schedule.mlir",
        "op_match": [{"op": "linalg.matmul", "tile": [4, 8, 1], "vector": [4, 8, 1]}],
        "lowering_patterns": ["lower_contraction"],
        "lmul_policy": "m1",
        "cflags": ["-march=rv64gcv", "-fno-vectorize"],
        "dtype_strategy": "fp32",
        "compiler_features": ["accumulator_resident_wholemodel_vf_mrpad"],
        "expected_instructions": ["vsetivli", "vle32.v"],
    }
    _write(d / "manifest.yaml", dump_yaml(manifest))
    _write(d / "knobs.yaml", dump_yaml(knobs))
    _write(d / "schedule.mlir", "module attributes {transform.with_named_sequence} {\n}\n")
    return d


def _beam_run(out_root: Path, *, best_gate_ok: bool = True, best_inert: bool = False,
              best_speedup: float | None = 18.713, best_k1_wall_ns: int | None = 133141607,
              noise_margin: float | None = 0.02, run_name: str = "20260720T214407Z_cca_beam_seed000_test") -> Path:
    """Build a fixture beam run dir: beam_tree.yaml + two minted fork packages under targets/rvv/."""
    run_dir = out_root / "runs" / "rvv" / "beam" / "matmul" / run_name
    targets_rvv = run_dir / "targets" / "rvv"

    loser = _fork_pkg(targets_rvv, "rvv_tuned_v1_d1_beam_1", version=1, depth=1, parent="hand_v0__beam")
    winner = _fork_pkg(targets_rvv, "rvv_tuned_v2_d2_beam_11", version=2, depth=2,
                       parent="rvv_tuned_v1_d1_beam_1")

    tree = {
        "target": "rvv",
        "depth": 2,
        "noise_margin": noise_margin,
        "expert_wall_ns": 168988634.0,
        "op_key": {"dtype": "int8", "op": "matmul", "shape_regime": "square"},
        "best": {"run_id": "rvv_tuned_v2_d2_beam_11", "speedup": best_speedup,
                 "attainment_vs_expert": 1.269, "lever": "feature"},
        "nodes": [
            {"run_id": "rvv_tuned_v1_d1_beam_1", "depth": 1, "gate_ok": True, "inert": False,
             "speedup": 1.005, "k1_wall_ns": 2479067251, "attainment_vs_expert": 0.068,
             "lever": "feature", "package_dir": str(loser)},
            {"run_id": "rvv_tuned_v2_d2_beam_11", "depth": 2, "gate_ok": best_gate_ok,
             "inert": best_inert, "speedup": best_speedup, "k1_wall_ns": best_k1_wall_ns,
             "attainment_vs_expert": 1.269, "lever": "feature", "package_dir": str(winner)},
        ],
    }
    _write(run_dir / "beam_tree.yaml", dump_yaml(tree))
    return run_dir


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
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


# --------------------------------------------------------------------------- read + verify


def test_read_beam_champion_locates_best_node_and_package(out_root):
    run_dir = _beam_run(out_root)
    champ = pc.read_beam_champion(run_dir)
    assert champ.run_id == "rvv_tuned_v2_d2_beam_11"
    assert champ.beam_run_id == run_dir.name
    assert champ.package_dir.is_dir()
    assert champ.speedup == 18.713
    assert champ.k1_wall_ns == 133141607
    assert champ.noise_margin == 0.02
    assert champ.gate_ok and not champ.inert


def test_verify_fail_closed_on_not_gate_ok(out_root):
    champ = pc.read_beam_champion(_beam_run(out_root, best_gate_ok=False))
    v = pc.verify_champion(champ)
    assert not v.ok and any("gate_ok" in r for r in v.reasons)


def test_verify_fail_closed_on_inert(out_root):
    champ = pc.read_beam_champion(_beam_run(out_root, best_inert=True))
    v = pc.verify_champion(champ)
    assert not v.ok and any("inert" in r for r in v.reasons)


def test_verify_fail_closed_on_missing_board_measurement(out_root):
    champ = pc.read_beam_champion(_beam_run(out_root, best_k1_wall_ns=None))
    v = pc.verify_champion(champ)
    assert not v.ok and any("k1_wall_ns" in r for r in v.reasons)


def test_verify_fail_closed_within_noise_floor(out_root):
    # speedup just 1% above baseline, noise floor 2% -> refused
    champ = pc.read_beam_champion(_beam_run(out_root, best_speedup=1.01, noise_margin=0.02))
    v = pc.verify_champion(champ)
    assert not v.ok and any("noise" in r for r in v.reasons)


def test_verify_passes_real_champion(out_root):
    champ = pc.read_beam_champion(_beam_run(out_root))
    assert pc.verify_champion(champ).ok


# --------------------------------------------------------------------------- stamp (honest cert)


def test_stamp_champion_honest_certification(out_root):
    run_dir = _beam_run(out_root)
    champ = pc.read_beam_champion(run_dir)
    stamp = pc.stamp_champion(champ)

    man = load_yaml(champ.package_dir / "manifest.yaml")
    # honest recorded status: k1_verified, NEVER a false spike_verified
    assert man["status"] == pc.K1_VERIFIED_STATUS == "k1_verified"
    assert man["status"] != "spike_verified"
    assert man["version"] == 2
    assert man["package_id"] == "rvv_tuned_v2_d2_beam_11"
    p = man["publication"]
    assert p["champion"] is True
    assert p["certification"] == "pass"
    assert p["certified_by"] == "k1_board"
    assert p["certified_by_run"] == run_dir.name
    assert p["measured"]["k1_wall_ns"] == 133141607
    assert p["measured"]["speedup"] == 18.713
    assert stamp.branch_hint == "stable/rvv_tuned_v2_d2_beam_11"

    # the stamped package still loads as a valid rvv package (schema + integrity intact)
    pkg = load_rvv_package(champ.package_dir)
    assert pkg.run_id == "rvv_tuned_v2_d2_beam_11"

    # publish selects it by the stamped package_id
    sel = pub.select_champion("rvv", artifacts_root=str(run_dir), package_id="rvv_tuned_v2_d2_beam_11")
    assert sel.package_id == "rvv_tuned_v2_d2_beam_11"
    # and resolves to its own stable/<pkg> branch
    assert pub.resolve_branch(sel) == "stable/rvv_tuned_v2_d2_beam_11"


def test_stamp_refuses_unverified_without_force(out_root):
    champ = pc.read_beam_champion(_beam_run(out_root, best_gate_ok=False))
    with pytest.raises(pc.PromoteError):
        pc.stamp_champion(champ)
    # --force stamps anyway (loud), for a human-driven override
    stamp = pc.stamp_champion(champ, force=True)
    assert stamp.status == "k1_verified"


def test_rvv_gate_accepts_k1_verified(out_root):
    """The rvv gate now ACCEPTS a k1_verified champion: a live-board measurement is a stronger
    certification than the spike simulator for a physical target, so publish runs through the REAL
    gate (gate=True) with no --no-gate bypass and no false spike_verified claim."""
    run_dir = _beam_run(out_root)
    champ = pc.read_beam_champion(run_dir)
    pc.stamp_champion(champ)
    sel = pub.select_champion("rvv", artifacts_root=str(run_dir),
                              package_id="rvv_tuned_v2_d2_beam_11")
    gate_ok, detail = pub._check_gate(sel)
    assert gate_ok is True            # k1_verified is accepted alongside spike_verified/rtl_certified
    assert "k1_verified" in detail
    # ...but an unverified/proposed status is still refused (the gate is not a rubber stamp)
    sel_unverified = dataclasses.replace(sel, status="proposed")
    refused, _ = pub._check_gate(sel_unverified)
    assert refused is False


# --------------------------------------------------------------------------- publish (dry + real)


def test_promote_and_publish_dry_run(out_root, monkeypatch):
    run_dir = _beam_run(out_root)
    remote = _bare_remote(out_root, "rvv", monkeypatch)
    stamp, res = pc.promote_and_publish(run_dir, execute=False)
    assert res.dry_run and not res.committed
    assert res.remote == remote
    assert res.branch == "stable/rvv_tuned_v2_d2_beam_11"
    assert res.gate_ok is True        # real gate now accepts the stamped k1_verified champion
    # the assembled tree exists locally; the bare remote got nothing
    assert (res.repo_dir / "manifest.yaml").is_file()


def test_promote_and_publish_execute_yields_stable_branch(out_root, monkeypatch):
    run_dir = _beam_run(out_root)
    remote = _bare_remote(out_root, "rvv", monkeypatch)
    bare = remote[len("file://"):]

    stamp, res = pc.promote_and_publish(run_dir, execute=True)
    assert res.committed and res.commit_sha
    assert res.branch == "stable/rvv_tuned_v2_d2_beam_11"

    # commit + tag present on the stable branch of the bare remote
    log = subprocess.run(["git", "-C", bare, "log", res.branch, "--oneline"],
                         capture_output=True, text=True)
    assert res.commit_sha[:7] in log.stdout
    tags = subprocess.run(["git", "-C", bare, "tag"], capture_output=True, text=True).stdout.split()
    assert res.tag in tags and res.tag == "v2-rvv_tuned_v2_d2_beam_11"

    # the truthful K1 certification rides along in the committed manifest
    show = subprocess.run(["git", "-C", bare, "show", f"{res.branch}:.merlin/certification.yaml"],
                          capture_output=True, text=True).stdout
    assert "rvv_tuned_v2_d2_beam_11" in show


def test_published_payload_round_trip_nuance(out_root, monkeypatch):
    """The published payload/ has schedule.mlir + knobs.yaml but NO rvv_package manifest, so
    load_rvv_package can't read it back directly -- write_payload_manifest is the opt-in fix."""
    run_dir = _beam_run(out_root)
    _bare_remote(out_root, "rvv", monkeypatch)
    stamp, res = pc.promote_and_publish(run_dir, execute=False)  # dry-run assembles the tree

    payload = res.repo_dir / "payload"
    assert (payload / "schedule.mlir").is_file()
    assert (payload / "knobs.yaml").is_file()
    assert not (payload / "manifest.yaml").exists()             # <-- the nuance
    with pytest.raises(Exception):
        load_rvv_package(payload)

    # opt-in fix makes the published payload directly loadable
    pc.write_payload_manifest(payload, package_id=stamp.package_id)
    pkg = load_rvv_package(payload)
    assert pkg.run_id == stamp.package_id


@pytest.mark.skipif(not _cxx_toolchain_available(), reason="cmake / C++ toolchain unavailable")
def test_fresh_clone_builds_champion(out_root, monkeypatch):
    from merlin.targetgen import oot_runner
    run_dir = _beam_run(out_root)
    remote = _bare_remote(out_root, "rvv", monkeypatch)
    bare = remote[len("file://"):]
    stamp, res = pc.promote_and_publish(run_dir, execute=True)
    assert res.committed

    clone = out_root / "build" / "_clone_champion"
    subprocess.run(["git", "clone", "-q", "-b", res.branch, f"file://{bare}", str(clone)], check=True)
    pkg = oot_runner.load_package(clone)
    assert pkg.manifest["build"]["tool_output"] == "build/bin/rvv-opt"
    oot_runner.build_package(pkg)
    assert pkg.tool.exists()
