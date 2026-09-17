"""Focused tests for the fail-closed non-ExecuTorch package-set producer."""
from __future__ import annotations

import copy
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from merlin.common.artifacts import ProductDir
from merlin.common.paths import bench_dir
from merlin.common.yaml import write_yaml
from merlin.compare import paper_merlin_packages as packages
from merlin.compare.freeze import sha256_paths
from merlin.compare.paper import PaperStudySpec

STUDY = bench_dir() / "rvv_paper" / "study_v2.yaml"


def _product(path: Path) -> ProductDir:
    path.mkdir(parents=True)
    return ProductDir(
        path=path, manifest_path=path / "manifest.yaml", run_id=path.name,
        topic="paper-merlin-packages", version=1, git_sha="abcdef0",
        timestamp="20260831T000000Z", target="k1", sources=[], _artifacts=[])


def _staged(tmp_path: Path) -> tuple[Path, Path]:
    original = PaperStudySpec.from_yaml(STUDY)
    raw = copy.deepcopy(original.canonical_dict())
    captures = []
    for model in raw["models"]:
        for precision, artifact in model["artifacts"].items():
            root = tmp_path / "captures" / model["name"] / precision
            root.mkdir(parents=True)
            (root / "capture-byte").write_text(
                f"{model['name']}/{precision}\n", encoding="utf-8")
            digest = sha256_paths([root])
            artifact.update({"path": str(root), "sha256": digest})
            captures.append({
                "model": model["name"], "precision": precision, "path": str(root),
                "sha256": digest, "status": "validated",
            })
    staged = tmp_path / "staged-study.yaml"
    write_yaml(staged, raw)
    registration = tmp_path / "capture-registration.json"
    registration.write_text(json.dumps({
        "version": 1, "complete": True, "study": str(staged),
        "study_sha256": packages._sha(staged), "captures": captures,
    }), encoding="utf-8")
    return staged, registration


def test_curated_non_executorch_inventory_is_25_packages_and_50_templates():
    cells = packages._required_cells(PaperStudySpec.from_yaml(STUDY))
    assert len(cells) == 50
    assert len({(cell.backend.name, cell.model.name, cell.precision)
                for cell in cells}) == 25
    assert Counter(cell.backend.name for cell in cells) == {
        "hand_v0_int8": 10, "merlin_frozen": 20,
        "merlin_xnnpack": 10, "merlin_openblas": 10,
    }


def test_explicit_authority_command_emits_closed_receipt_and_rejects_tampering(tmp_path):
    authority, receipt = packages.issue_toolchain_authority(
        output=tmp_path / "authority.json", authority_id="reviewed-k1-compiler",
        target="k1", build_tool="/bin/true")
    parsed = packages._verify_authority_receipt(authority, receipt, target="k1")
    assert parsed["tool"]["path"] == str(Path("/bin/true").resolve())
    assert json.loads(receipt.read_text())["environment_sources"] == []
    value = json.loads(receipt.read_text())
    value["target"] = "not-k1"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the authority"):
        packages._verify_authority_receipt(authority, receipt, target="k1")


def test_promoted_compiler_must_equal_capture_authorizing_campaign(tmp_path):
    from merlin.benchharness.host_agent import (
        _submission_package_digest,
        _submission_source_digest,
    )
    promoted = tmp_path / "compiler-submission"
    promoted.mkdir()
    (promoted / "manifest.yaml").write_text(
        "version: 1\nbuild: {}\ncompiler: {}\npolicy: policy.yaml\n", encoding="utf-8")
    policy = promoted / "policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")
    (promoted / "source.cpp").write_text("// exact compiler source\n", encoding="utf-8")
    registration = {"host_campaign_freeze": {
        "selected_compiler_package": str(promoted),
        "selected_policy_sha256": packages._sha(policy),
        "runtime_sha256": _submission_package_digest(promoted),
        "compiler_sha256": _submission_source_digest(promoted),
    }}
    packages._validate_promoted_campaign_binding(registration, promoted)
    policy.write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        packages._validate_promoted_campaign_binding(registration, promoted)


def test_missing_backend_producers_block_all_publication_and_report_every_package(
        tmp_path, monkeypatch):
    staged, registration = _staged(tmp_path)
    authority, receipt = packages.issue_toolchain_authority(
        output=tmp_path / "authority.json", authority_id="reviewed-k1-compiler",
        target="k1", build_tool="/bin/true")
    runtime = tmp_path / "runtime.a"
    runtime.write_bytes(b"runtime")
    promoted = tmp_path / "promoted"
    promoted.mkdir()
    producer_inputs = tmp_path / "producer-inputs"
    producer_inputs.mkdir()
    monkeypatch.setattr(packages, "_backend_identity", lambda cell, _promoted: {
        "package_path": "/bound", "package_sha256": "a" * 64,
        "kernel_source_sha256": None, "run_id": "bound",
        "dtype_strategy": "int8_w8a8" if cell.precision == "w8a8" else "fp32",
        "kernel_backend": cell.backend.options.get("kernel_backend"),
        "promoted_compiler_sha256": None,
        "promoted_compiler_source_sha256": None,
    })
    monkeypatch.setattr(packages, "_validate_promoted_campaign_binding", lambda *_args: None)
    product = _product(tmp_path / "product")
    with pytest.raises(packages.MerlinPackageSetNotReady) as caught:
        packages.materialize(
            staged, registration, promoted, producer_inputs, runtime, authority, receipt,
            execute=True, product=product)
    plan = json.loads((product.path / "package-plan.json").read_text())
    assert plan["required_packages"] == 25
    assert plan["required_templates"] == 50
    assert plan["producer_inputs_validated"] == 0
    assert len(plan["errors"]) == 25
    assert all("producer input is absent or unsafe" in error for error in plan["errors"])
    assert not (product.path / "package-set").exists()
    assert caught.value.output_dir == product.path


def test_capture_registration_mutation_blocks_before_producer_inputs(tmp_path, monkeypatch):
    staged, registration = _staged(tmp_path)
    authority, receipt = packages.issue_toolchain_authority(
        output=tmp_path / "authority.json", authority_id="reviewed-k1-compiler",
        target="k1", build_tool="/bin/true")
    runtime = tmp_path / "runtime.a"
    runtime.write_bytes(b"runtime")
    promoted = tmp_path / "promoted"
    promoted.mkdir()
    producer_inputs = tmp_path / "producer-inputs"
    producer_inputs.mkdir()
    raw = json.loads(registration.read_text())
    raw["captures"][0]["sha256"] = "0" * 64
    registration.write_text(json.dumps(raw), encoding="utf-8")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("producer input must not be opened")

    monkeypatch.setattr(packages, "_producer_input", forbidden)
    monkeypatch.setattr(packages, "_validate_promoted_campaign_binding", lambda *_args: None)
    with pytest.raises(packages.MerlinPackageSetNotReady, match="bytes differ"):
        packages.materialize(
            staged, registration, promoted, producer_inputs, runtime, authority, receipt,
            product=_product(tmp_path / "product"))
    assert called is False


def test_register_producer_input_deep_stages_and_never_overwrites(tmp_path, monkeypatch):
    staged, _registration = _staged(tmp_path)
    promoted = tmp_path / "promoted"
    promoted.mkdir()
    runtime = tmp_path / "runtime.a"
    runtime.write_bytes(b"runtime")
    source = tmp_path / "source-compiler-input.json"
    source.write_text("{}\n", encoding="ascii")
    producer_root = tmp_path / "producer-inputs"
    staged_destinations = []

    def fake_stage(raw, destination, *, recipe):
        assert Path(raw) == source
        assert recipe == packages.MERLIN_RECIPE
        Path(destination).parent.mkdir(parents=True)
        shutil.copy2(raw, destination)
        staged_destinations.append(Path(destination))
        return Path(destination)

    monkeypatch.setattr(packages, "stage_compiler_input", fake_stage)
    monkeypatch.setattr(packages, "merlin_session_resources", lambda _path: object())
    monkeypatch.setattr(packages, "_backend_identity", lambda _cell, _promoted: {
        "package_path": "/bound", "package_sha256": "a" * 64,
        "kernel_source_sha256": None, "run_id": "bound", "dtype_strategy": "fp32",
        "kernel_backend": None, "promoted_compiler_sha256": "b" * 64,
        "promoted_compiler_source_sha256": "c" * 64,
    })
    monkeypatch.setattr(packages, "_producer_input", lambda *_args, **_kwargs: source)
    output = packages.register_backend_producer_input(
        study_path=staged, promoted_compiler=promoted, runtime_artifact=runtime,
        producer_root=producer_root, backend="merlin_frozen", model="gemma2_2b",
        precision="fp32", compiler_input=source)
    value = json.loads(output.read_text())
    assert value["kind"] == packages._PRODUCER_INPUT_KIND
    assert value["promoted_compiler_sha256"] == "b" * 64
    assert staged_destinations == [output.parent / "compiler-input/compiler-input.json"]
    with pytest.raises(FileExistsError, match="already exists"):
        packages.register_backend_producer_input(
            study_path=staged, promoted_compiler=promoted, runtime_artifact=runtime,
            producer_root=producer_root, backend="merlin_frozen", model="gemma2_2b",
            precision="fp32", compiler_input=source)
