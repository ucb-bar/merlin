"""Whole-model grading binds the scalar/RVV lane to the target descriptor.

The accelerator submission and the host package are intentionally different artifacts.  The former
arrives as ``mesh_package``; the latter is frozen experiment infrastructure declared by
``target_experiment.yaml``.  Neither may silently select the other's default.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

import merlin.compile_cli as compile_cli
from merlin.common.paths import repo_root
from merlin.targetgen import capsule_runner
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.target_experiment import load_target_experiment


#: A REAL frozen model capsule, not a synthetic dict.
#:
#: These tests are about host-lane package selection, and they monkeypatch `compile_model` so the
#: accelerator side never runs. But `_grade_model_capsule_inline` materializes the capsule from its
#: frozen source directory FIRST, and a dict has no `__dir__`, so production correctly refused with
#: "model capsule has no frozen source directory" and these tests failed on the wrong assertion.
#: Production is right to fail closed there -- a whole-model grade with no frozen source cannot be
#: attributed to anything -- so the fixture supplies a real capsule rather than the guard being
#: relaxed. M3 is the smallest capsule that satisfies materialization end to end: declared
#: interface, loader, external weights, independent golden, and an arg_order its golden agrees with.
_FROZEN_MODEL_DIR = "merlin/contract/capsules/model/M3_host_island_seam_gemmini"

#: Materialization reads the capsule's independent golden, and `golden.yaml` is UNTRACKED by design
#: (answer surfaces never enter the public repo). A fresh git worktree therefore has the capsule but
#: not its golden, and without this guard these tests would fail there for a reason that has nothing
#: to do with the host lane. Skip and name the absent asset -- "we cannot tell", never a verdict.
_REQUIRED_ASSETS = ("capsule.yaml", "capsule.interface.mlir", "capsule.pytorch.py",
                    "capsule.weights.safetensors", "golden.yaml")


@pytest.fixture
def frozen_capsule() -> dict:
    root = repo_root() / _FROZEN_MODEL_DIR
    missing = [n for n in _REQUIRED_ASSETS if not (root / n).is_file()]
    if missing:
        pytest.skip(f"frozen model capsule at {_FROZEN_MODEL_DIR} is missing {missing} "
                    f"(golden.yaml and other answer surfaces are untracked by design, so a "
                    f"fresh worktree has none)")
    return capsule_runner.load_capsule(root)

def _gemmini_descriptor():
    return (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/"
            "target_experiment.yaml")


def _snapshot_package(tmp_path, package_rel: str):
    """Create the minimal valid v2 snapshot shape consumed by model grading."""
    root = tmp_path / "bundle_inputs"
    package = root / "repo" / package_rel
    package.parent.mkdir(parents=True)
    shutil.copytree(repo_root() / package_rel, package)
    digest, n_files, n_bytes = bwrap._snapshot_content(root)
    (root / "snapshot.json").write_text(json.dumps({
        "version": 2,
        "repo": str(repo_root()),
        "content_sha256": digest,
        "n_files": n_files,
        "n_bytes": n_bytes,
    }), encoding="utf-8")
    return root, package, digest


def test_target_experiment_owns_and_resolves_the_frozen_host_lane():
    te = load_target_experiment(_gemmini_descriptor())

    package, identity = te.resolve_host_lane()

    assert package == (repo_root() /
                       "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8").resolve()
    assert identity["package"] == te.host_lane.package
    assert identity["package_sha256"] == \
        "32d265324cba85abc6760a151d56b03bdc3e95c79e8ebf0bc392207c0a041d8b"
    assert identity["run_id"] == "impr_tuned_wholemodel_vf_int8"
    assert identity["dtype_strategy"] == "int8_w8a8"


def test_gemmini_model_dtypes_match_the_descriptor_package():
    te = load_target_experiment(_gemmini_descriptor())
    _, identity = te.resolve_host_lane()
    model_dtypes = set()
    for root in te.graded_roots():
        for capsule_path in root.glob("*/capsule.yaml"):
            capsule = yaml.safe_load(capsule_path.read_text(encoding="utf-8")) or {}
            if capsule.get("kind") == "model":
                attrs = (capsule.get("operation") or {}).get("attributes") or {}
                model_dtypes.add(attrs.get("compile_dtype"))

    assert model_dtypes == {"int8"}, "the experiment's host package selection must cover every capstone"
    assert identity["dtype_strategy"] == compile_cli._DTYPE_STRATEGY["int8"]


def test_materialized_gemmini_bundles_lock_the_same_read_only_package():
    te = load_target_experiment(_gemmini_descriptor())
    _, identity = te.resolve_host_lane()
    package = te.host_lane.package
    bundles = _gemmini_descriptor().parent / "input_bundles"
    checked = []
    for manifest_path in sorted(bundles.glob("*/input_bundle_manifest.yaml")):
        if manifest_path.parent.name == "grader_private_v0":
            continue
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        grants = [entry for entry in manifest.get("allowed", [])
                  if str(entry.get("path", "")).rstrip("/") == package.rstrip("/")]
        assert len(grants) == 1 and grants[0].get("mode") == "ro", manifest_path
        lock = yaml.safe_load((manifest_path.parent / "bundle_lock.yaml").read_text(
            encoding="utf-8")) or {}
        assert (lock.get("allowed_tree_sha256") or {}).get(package.rstrip("/")) == \
            identity["package_sha256"]
        checked.append(manifest_path.parent.name)
    assert checked, "no agent input bundles were checked"


def test_targeted_model_grade_passes_exact_descriptor_package_and_records_it(
        monkeypatch, frozen_capsule):
    seen = {}

    def fake_compile_model(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "verified", "verify": {"gate_ok": True}}

    monkeypatch.setattr(compile_cli, "compile_model", fake_compile_model)
    # Keep this focused on package selection. A host diagnostic with a named target still has to use
    # that target's descriptor, but returns before mesh-tier accounting.
    monkeypatch.setenv("MERLIN_MODEL_GRADE_RUN", "host")

    result = capsule_runner._grade_model_capsule_inline(
        frozen_capsule, target="gemmini", timeout=1, package_dir="submission-under-test")

    expected = (repo_root() /
                "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8").resolve()
    assert seen["package"] == str(expected)
    assert seen["mesh_package"] == "submission-under-test"
    assert result["host_lane"]["package_sha256"] == \
        "32d265324cba85abc6760a151d56b03bdc3e95c79e8ebf0bc392207c0a041d8b"


def test_bwrap_model_grade_executes_run_snapshot_not_live_package(
        monkeypatch, tmp_path, frozen_capsule):
    package_rel = "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8"
    snapshot_root, snapshot_package, snapshot_digest = _snapshot_package(tmp_path, package_rel)

    # A separate mutable repo models the operator worktree after launch. It drifts after the snapshot,
    # but compile_model must still receive and read the agent-visible snapshot bytes.
    live_root = tmp_path / "live-repo"
    live_package = live_root / package_rel
    live_package.parent.mkdir(parents=True)
    shutil.copytree(repo_root() / package_rel, live_package)
    live_schedule = live_package / "schedule.mlir"
    live_schedule.write_text("LIVE WORKTREE DRIFT\n", encoding="utf-8")
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(_gemmini_descriptor().read_text(encoding="utf-8"), encoding="utf-8")

    from merlin.targetgen import target_experiment
    monkeypatch.setattr(target_experiment, "repo_root", lambda: live_root)
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))
    monkeypatch.setenv("MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT", str(snapshot_root))
    monkeypatch.setenv("MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED", "1")
    monkeypatch.setenv("MERLIN_MODEL_GRADE_RUN", "host")
    seen = {}

    def fake_compile_model(*args, **kwargs):
        package = Path(kwargs["package"])
        seen["package"] = package
        seen["schedule"] = (package / "schedule.mlir").read_text(encoding="utf-8")
        return {"status": "verified", "verify": {"gate_ok": True}}

    monkeypatch.setattr(compile_cli, "compile_model", fake_compile_model)
    result = capsule_runner._grade_model_capsule_inline(
        frozen_capsule, target="gemmini", timeout=1, package_dir="submission-under-test")

    assert seen["package"] == snapshot_package.resolve()
    assert seen["schedule"] != "LIVE WORKTREE DRIFT\n"
    assert result["host_lane"]["run_snapshot"]["content_sha256"] == snapshot_digest
    assert result["host_lane"]["resolved_package"] == str(snapshot_package.resolve())


@pytest.mark.parametrize("malformed", [False, True], ids=["missing-pointer", "bad-snapshot"])
def test_bwrap_model_grade_never_falls_back_when_snapshot_is_missing_or_malformed(
        monkeypatch, tmp_path, malformed, frozen_capsule):
    called = False

    def fake_compile_model(*args, **kwargs):
        nonlocal called
        called = True
        return {"status": "verified", "verify": {"gate_ok": True}}

    monkeypatch.setattr(compile_cli, "compile_model", fake_compile_model)
    monkeypatch.setenv("MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED", "1")
    if malformed:
        root = tmp_path / "bundle_inputs"
        (root / "repo").mkdir(parents=True)
        (root / "snapshot.json").write_text('{"version": 1}', encoding="utf-8")
        monkeypatch.setenv("MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT", str(root))
    else:
        monkeypatch.delenv("MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT", raising=False)

    result = capsule_runner._grade_model_capsule_inline(
        frozen_capsule, target="gemmini", timeout=1, package_dir="submission-under-test")

    assert result["status"] == "incomplete"
    assert "snapshot" in result["failure"]["detail"]
    assert not called


def test_qa_loop_exports_verified_snapshot_only_to_host_grading():
    loop = (repo_root() / "merlin/experiments/capsule_bench/harness/"
            "run_baseline_qa_loop.py").read_text(encoding="utf-8")

    assert "_BWS.verify_bundle_snapshot(ws, bundle, repo=C.REPO)" in loop
    assert '"model_host_lane_snapshot": _model_host_lane_snapshot' in loop
    assert 'os.environ[_MODEL_HOST_SNAPSHOT_ROOT_ENV] = str(_snapshot_root)' in loop
    assert 'os.environ[_MODEL_HOST_SNAPSHOT_REQUIRED_ENV] = "1"' in loop
    assert 'parts += ["--unsetenv", _MODEL_HOST_SNAPSHOT_ROOT_ENV' in loop


def test_model_grade_refuses_descriptor_package_of_the_wrong_datatype(
        monkeypatch, tmp_path, frozen_capsule):
    called = False

    def fake_compile_model(*args, **kwargs):
        nonlocal called
        called = True
        return {"status": "verified", "verify": {"gate_ok": True}}

    monkeypatch.setattr(compile_cli, "compile_model", fake_compile_model)
    doc = yaml.safe_load(_gemmini_descriptor().read_text(encoding="utf-8"))
    old = "out/artifacts/targets/rvv/rvv_tuned_v1_d1_vfmacc_outerproduct"
    doc["host_lane"]["package"] = old
    doc["host_lane"]["read_only"] = [old + "/"]
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(descriptor))
    result = capsule_runner._grade_model_capsule_inline(
        frozen_capsule, target="gemmini", timeout=1, package_dir="submission-under-test")

    assert result["status"] == "incomplete"
    # The refusal now happens EARLIER -- at host-lane resolution, where the descriptor's declared
    # dtype_strategy is compared against the package's own knobs -- rather than after resolution when
    # the loaded strategy met the capsule's compile dtype. Both are the same protection; asserting the
    # substance rather than one layer's wording keeps this test about the guarantee. What must not
    # change is that BOTH precisions are named, so a reader can see which two things disagreed.
    detail = result["failure"]["detail"]
    assert "int8_w8a8" in detail and "fp32" in detail, detail
    assert not called, "a cross-datatype host package must be rejected before compilation"


def test_model_grade_refuses_host_package_drift_during_compile(monkeypatch, frozen_capsule):
    before = {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"}
    after = {"package_sha256": "b" * 64, "dtype_strategy": "int8_w8a8"}

    class Experiment:
        def resolve_host_lane(self, *, root=None, dtype=None):
            return repo_root() / "frozen-test-host", after

    monkeypatch.setattr(capsule_runner, "_resolve_model_host_lane", lambda target, dtype: (
        Experiment(), repo_root() / "frozen-test-host", before))
    monkeypatch.setattr(compile_cli, "compile_model", lambda *args, **kwargs: {
        "status": "verified", "verify": {"gate_ok": True},
    })

    result = capsule_runner._grade_model_capsule_inline(
        frozen_capsule, target="gemmini", timeout=1, package_dir="submission-under-test")

    assert result["status"] == "incomplete"
    assert "changed during grading" in result["failure"]["detail"]
    assert result["host_lane_after"]["package_sha256"] == "b" * 64


def test_host_lane_resolution_fails_closed_when_required_content_is_missing(tmp_path):
    doc = yaml.safe_load(_gemmini_descriptor().read_text(encoding="utf-8"))
    package = tmp_path / "host-package"
    package.mkdir()
    (package / "manifest.yaml").write_text("target: rvv\n", encoding="utf-8")
    doc["host_lane"]["package"] = "host-package"
    doc["host_lane"]["read_only"] = ["host-package"]
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    te = load_target_experiment(descriptor)
    with pytest.raises(ValueError, match="missing required path"):
        te.resolve_host_lane(root=tmp_path)


def test_host_package_must_be_inside_a_read_only_grant(tmp_path):
    doc = yaml.safe_load(_gemmini_descriptor().read_text(encoding="utf-8"))
    doc["host_lane"]["package"] = "out/artifacts/targets/rvv/rvv_tuned_v1_d1_vfmacc_outerproduct"
    doc["host_lane"]["read_only"] = ["merlin/contract/"]
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    te = load_target_experiment(descriptor)
    with pytest.raises(ValueError, match="read-only grant"):
        te.resolve_host_lane()
