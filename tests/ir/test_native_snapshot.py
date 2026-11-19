"""Opening a native snapshot binds the installed compiler's actual source bytes."""

import hashlib
import json
import shutil

import pytest

from merlin.common.paths import module_source_path
from merlin.compiler.semantic import target_binding as snapshots


def test_snapshot_opens_with_current_sources_and_refuses_source_drift(tmp_path, monkeypatch):
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

    # A moved implementation and the new shared input record must both remain bound.
    for name in ("compiler/semantic/search/allocation.py", "compiler/frontend/input_types.py"):
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
