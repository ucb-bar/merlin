"""Opening a native snapshot binds the installed compiler's actual source bytes, and native compile takes its boundary ABI
from a file or, for `--abi auto`, from the provider."""

import hashlib
import json
import shutil

import pytest
from corefixtures.row_table import run_rows

from merlin.common.paths import module_source_path
from merlin.compiler.semantic import target_binding as snapshots


def test_snapshot_opens_with_current_sources_and_refuses_source_drift(tmp_path, monkeypatch):
    run_rows(tmp_path, monkeypatch, {
        "source_drift": _check_snapshot_source_drift,
        "nested_publication": _check_nested_publication,
        "boundary_abi": _check_boundary_abi,
    })


def _check_boundary_abi(tmp_path, patch):
    from types import SimpleNamespace

    from merlin.compiler.build.native import boundary_abi

    planned = {"fixed_inputs": {"x": 64}, "fixed_outputs": [128]}
    requests = []
    provider = SimpleNamespace(plan_boundary_abi=lambda request: requests.append(request) or planned)
    assert boundary_abi("auto", provider, "request") == planned and requests == ["request"]
    with pytest.raises(ValueError, match="needs a provider that plans its boundary ABI"):
        boundary_abi("auto", SimpleNamespace(), "request")
    (tmp_path / "abi.json").write_text(json.dumps(planned))
    assert boundary_abi(str(tmp_path / "abi.json"), SimpleNamespace(), "request") == planned
    for wrong in ({"fixed_inputs": {}}, {"fixed_inputs": {"x": "64"}, "fixed_outputs": []}, ["auto"]):
        with pytest.raises(ValueError, match="native compile ABI"):
            boundary_abi("auto", SimpleNamespace(plan_boundary_abi=lambda request, wrong=wrong: wrong), "request")


def _check_nested_publication(tmp_path, patch):
    _check_nested_program_leaves_preserve_authenticated_parent_metadata(tmp_path / "nested-valid")
    for mutation in ("binary", "source", "extra_directory", "symlink_ancestor", "overlap"):
        _check_nested_program_publication_refuses_changed_or_unaccounted_members(
            tmp_path / ("nested-" + mutation), mutation
        )


def _check_snapshot_source_drift(tmp_path, monkeypatch):
    # Copy only the declared source closure, as an installed package would ship it.
    package = module_source_path("merlin").parent
    sources = snapshots._compiler_sources()
    for name in sources:
        destination = tmp_path / "installed" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, destination)
    monkeypatch.setattr(snapshots, "__file__", str(tmp_path / "installed/compiler/semantic/target_binding.py"))
    profile = snapshots.NativeTargetProfile(
        "fixture",
        (
            snapshots.InstructionDescriptor(
                "add",
                "f32_add",
                ("ram", "ram"),
                "ram",
                "f32",
                "exact",
                (1,),
                input_dtypes=("f32", "f32"),
                input_numerical_policies=("exact", "exact"),
            ),
        ),
        (snapshots.StorageBank("ram", "memory", 1024, "bytes"),),
    )
    root = tmp_path / "snapshot"
    (root / "bin").mkdir(parents=True)
    (root / "profile.json").write_text(json.dumps(profile.record()))
    # Opening validates bridge bytes; execution/building the Rust bridge is a separate boundary.
    (root / "bin/merlin-egg-bridge").write_bytes(b"test bridge identity")
    manifest = {
        "schema": "merlin.native_target_snapshot.v6",
        "status": "selection_only",
        "target_identity": profile.target_identity,
        "profile_sha256": hashlib.sha256((root / "profile.json").read_bytes()).hexdigest(),
        "bridge_sha256": hashlib.sha256((root / "bin/merlin-egg-bridge").read_bytes()).hexdigest(),
        "compiler_sources": sources,
        "runtime_dependencies": snapshots._runtime_dependencies(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert snapshots.open_native_snapshot(root).profile == profile

    # Board memory bindings and numerical/input implementations remain bound.
    for name in (
        "compiler/semantic/search/allocation.py",
        "compiler/frontend/input_types.py",
        "compiler/build/memory_budget.py",
        "compiler/build/memory_placement.py",
        "target/soc.py",
        "target/system.py",
        "compiler/quantization/contraction/scaled_fp8.py",
        "compiler/semantic/source_preparation.py",
    ):
        implementation = tmp_path / "installed" / name
        original = implementation.read_bytes()
        implementation.write_bytes(original + b"\n# changed implementation\n")
        with pytest.raises(ValueError, match="sources differ"):
            snapshots.open_native_snapshot(root)
        implementation.unlink()
        with pytest.raises(ValueError, match="sources are unavailable"):
            snapshots.open_native_snapshot(root)
        implementation.write_bytes(original)
    assert snapshots.open_native_snapshot(root).profile == profile

def _nested_publication(root):
    leaf = root / "segments/region_000/segments/body_000"
    leaf.mkdir(parents=True)
    (leaf / "program.bin").write_bytes(b"native program identity")
    detail = {
        "engine": "fixture",
        "request_digest": "leaf-request",
        "target_identity": "fixture-target",
        "artifact_kind": "program",
        "binary_sha256": hashlib.sha256((leaf / "program.bin").read_bytes()).hexdigest(),
    }
    (leaf / "manifest.json").write_text(json.dumps(detail))
    source = root / "segments/region_000/source.mlir"
    source.write_text("module {}")
    manifest = {
        "engine": "fixture",
        "request_digest": "application-request",
        "target_identity": "fixture-target",
        "artifact_kind": "program_set",
        "stage_artifacts": {
            source.relative_to(root).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "segments": [{
            "index": 0,
            "path": leaf.relative_to(root).as_posix(),
            "request_digest": detail["request_digest"],
            "binary_sha256": detail["binary_sha256"],
            "manifest_sha256": hashlib.sha256((leaf / "manifest.json").read_bytes()).hexdigest(),
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _verify_publication(root, manifest):
    snapshots.verify_native_publication(
        root, manifest, engine="fixture", request_digest="application-request", target_identity="fixture-target"
    )


def _check_nested_program_leaves_preserve_authenticated_parent_metadata(tmp_path):
    manifest = _nested_publication(tmp_path)
    _verify_publication(tmp_path, manifest)


def _check_nested_program_publication_refuses_changed_or_unaccounted_members(tmp_path, mutation):
    manifest = _nested_publication(tmp_path)
    region = tmp_path / "segments/region_000"
    if mutation == "binary":
        (region / "segments/body_000/program.bin").write_bytes(b"different native program")
    elif mutation == "source":
        (region / "source.mlir").write_text("module { changed }")
    elif mutation == "extra_directory":
        (region / "unaccounted").mkdir()
    elif mutation == "symlink_ancestor":
        actual = tmp_path / "actual-region"
        region.rename(actual)
        region.symlink_to(actual, target_is_directory=True)
    elif mutation == "overlap":
        manifest["segments"].append({**manifest["segments"][0], "index": 1, "path": "segments/region_000"})
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _verify_publication(tmp_path, manifest)
