"""Pinned reference examples remain separate readonly release inputs."""
import hashlib
import json
import stat

import pytest

from merlin_experiments.corpus import public_references as R
from merlin_experiments.spec import SpecError


def source(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "example.c").write_text("int example(void) { return 0; }\n")
    manifest = root / "reference-manifest.json"
    manifest.write_text(json.dumps({"schema": R.SCHEMA, "members": [
        {"path": "example.c", "sha256": hashlib.sha256((root / "example.c").read_bytes()).hexdigest()}
    ]}))
    return manifest


def test_pinned_leaf_copy_and_readonly_grant(tmp_path):
    manifest = source(tmp_path)
    (manifest.parent / "not-selected.py").write_text("private = 1")
    destination = tmp_path / "payload" / "reference-kernels"
    record = R.stage(manifest, destination)
    assert record["members"] == 1
    assert (destination / "example.c").read_bytes() == (manifest.parent / "example.c").read_bytes()
    assert not (destination / "not-selected.py").exists()
    assert stat.S_IMODE((destination / "example.c").stat().st_mode) == 0o444
    assert (destination / "example.c").stat().st_ino != (manifest.parent / "example.c").stat().st_ino
    document = {}
    R.declare(document, destination=destination, variants=("public_v0",), source_sha256=record["sha256"], repo=tmp_path)
    grant = document["information_sets"]["public_v0"]["allowed"][0]
    assert grant == {"path": "payload/reference-kernels/", "as": "reference-kernels", "mode": "ro",
                     "note": "explicit pinned examples; compiler starts empty"}


@pytest.mark.parametrize("mutation", ["drift", "symlink", "traversal", "duplicate", "binary"])
def test_unpinned_or_unsafe_members_refuse(tmp_path, mutation):
    manifest = source(tmp_path)
    doc = json.loads(manifest.read_text())
    leaf = manifest.parent / "example.c"
    if mutation == "drift":
        leaf.write_text("changed")
    elif mutation == "symlink":
        original = manifest.parent / "original.c"; leaf.rename(original); leaf.symlink_to(original)
    elif mutation == "traversal":
        doc["members"][0]["path"] = "../example.c"
    elif mutation == "duplicate":
        doc["members"].append(dict(doc["members"][0]))
    else:
        leaf.write_bytes(b"ELF\x00")
        doc["members"][0]["sha256"] = hashlib.sha256(leaf.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(doc))
    with pytest.raises(SpecError):
        R.stage(manifest, tmp_path / "payload")
