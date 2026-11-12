from __future__ import annotations

import pytest

from merlin.common.digest import sha256_file
from merlin.development.targetgen.artifact_bundle import BundleError, SelectedArtifactProfile, verify_bundle


def _profile(root):
    path = root / "profile.yaml"
    path.write_text("name: independent-profile\n")
    return SelectedArtifactProfile.capture(path)


def _bundle(root):
    profile = _profile(root)
    image = root / "commands.bin"
    image.write_bytes(b"independent compiler artifact")
    document = {
        "schema": "merlin.artifact_bundle.v1",
        "profile": {"name": profile.name, "sha256": profile.sha256},
        "artifacts": [
            {
                "id": "producer",
                "kind": "command_stream",
                "path": image.name,
                "sha256": sha256_file(image),
                "unit": "first",
                "issuer": "host",
            },
            {
                "id": "consumer",
                "kind": "native_image",
                "path": image.name,
                "sha256": sha256_file(image),
                "unit": "second",
                "issuer": "runtime",
            },
        ],
        "buffers": [
            {"id": "input", "memory_space": "input-space", "initialized": True},
            {"id": "shared", "memory_space": "shared-space", "initialized": False},
            {"id": "out", "memory_space": "output-space", "initialized": False},
        ],
        "steps": [
            {"id": "write", "artifact": "producer", "after": [], "reads": ["input"], "writes": ["shared"]},
            {"id": "barrier", "artifact": "producer", "after": ["write"], "reads": [], "writes": []},
            {"id": "read", "artifact": "consumer", "after": ["barrier"], "reads": ["shared"], "writes": ["out"]},
        ],
    }
    return (document, profile)


@pytest.mark.parametrize("mutation", ["digest"])
def test_artifact_paths_and_bytes_are_bound(tmp_path, mutation):
    root = tmp_path / "bundle"
    root.mkdir()
    document, profile = _bundle(root)
    row = document["artifacts"][0]
    if mutation == "absolute":
        row["path"] = str(root / row["path"])
    elif mutation == "parent":
        row["path"] = "../commands.bin"
    elif mutation in ("escape", "broken"):
        outside = tmp_path / "outside.bin"
        if mutation == "escape":
            outside.write_bytes((root / row["path"]).read_bytes())
        (root / "alias.bin").symlink_to(outside)
        row["path"] = "alias.bin"
    elif mutation == "digest":
        (root / row["path"]).write_bytes(b"changed")
    else:
        row["path"] = "missing.bin"
    with pytest.raises(BundleError):
        verify_bundle(document, root, profile=profile)
