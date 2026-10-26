"""Pinned reference examples remain separate readonly release inputs."""
import hashlib
import json
import stat
import zipfile

import pytest
from merlin_experiments.corpus import public_references as R
from merlin_experiments.spec import SpecError


def source(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "example.c").write_text("int example(void) { return 0; }\n")
    manifest = root / "reference-manifest.json"
    manifest.write_text(json.dumps({"schema": R.SCHEMA, "review_id": "test-positive-review", "origins": [{
        "name": "public_example", "kind": "public_repository", "container": ".", "identity": "1" * 40,
        "license": "LICENSE", "public_url": "https://example.org/compiler",
        "allowed_purpose": "reference_reading", "files": [
            {"member": "example.c", "destination": "example.c",
             "sha256": hashlib.sha256((root / "example.c").read_bytes()).hexdigest()}]}]}))
    return manifest


def test_pinned_leaf_copy_and_readonly_grant(tmp_path):
    manifest = source(tmp_path)
    (manifest.parent / "not-selected.py").write_text("private = 1")
    destination = tmp_path / "payload" / "reference-kernels"
    record = R.stage(manifest, destination)
    assert record["members"] == 1
    assert (destination / "public_example/example.c").read_bytes() == (manifest.parent / "example.c").read_bytes()
    assert not (destination / "not-selected.py").exists()
    assert stat.S_IMODE((destination / "public_example/example.c").stat().st_mode) == 0o444
    assert (destination / "public_example/example.c").stat().st_ino != (manifest.parent / "example.c").stat().st_ino
    assert {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} == {
        "public_example/example.c", "reference-manifest.json", "reference-selection.json"}
    assert record["origins"][0]["allowed_purpose"] == "reference_reading"
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
        original = manifest.parent / "original.c"
        leaf.rename(original)
        leaf.symlink_to(original)
    elif mutation == "traversal":
        doc["origins"][0]["files"][0]["member"] = "../example.c"
    elif mutation == "duplicate":
        doc["origins"][0]["files"].append(dict(doc["origins"][0]["files"][0]))
    else:
        leaf.write_bytes(b"ELF\x00")
        doc["origins"][0]["files"][0]["sha256"] = hashlib.sha256(leaf.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(doc))
    with pytest.raises(SpecError):
        R.stage(manifest, tmp_path / "payload")


def test_frozen_v1_is_not_silently_reclassified_as_public(tmp_path):
    manifest = source(tmp_path)
    original = json.dumps({"schema": "operator.allowed_reference_sources.v1", "members": [
        {"path": "example.c", "sha256": hashlib.sha256((manifest.parent / "example.c").read_bytes()).hexdigest()}]})
    manifest.write_text(original)
    with pytest.raises(SpecError, match="v1 cannot be restaged"):
        R.stage(manifest, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()
    assert manifest.read_text() == original


@pytest.mark.parametrize("field,value", [("kind", "coworker_private_archive"),
    ("allowed_purpose", "compiler_seed"), ("public_url", "https://example.org/with space")])
def test_missing_or_nonpositive_origin_refuses_before_writing(tmp_path, field, value):
    manifest = source(tmp_path)
    document = json.loads(manifest.read_text())
    document["origins"][0][field] = value
    manifest.write_text(json.dumps(document))
    with pytest.raises(SpecError):
        R.stage(manifest, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()


def test_user_permitted_archive_requires_exact_container_and_member_hash(tmp_path):
    manifest = source(tmp_path)
    archive = manifest.parent / "permitted.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("selected/example.c", b"int allowed;\n")
        stream.writestr("not_selected.c", b"not exported\n")
    (manifest.parent / "coworker-private.zip").write_bytes(b"never selected or read")
    document = {"schema": R.SCHEMA, "review_id": "explicit-user-review", "origins": [{
        "name": "authorized_archive", "kind": "user_permitted_archive", "container": "permitted.zip",
        "identity": hashlib.sha256(archive.read_bytes()).hexdigest(), "license": "LICENSE",
        "user_authorization": "explicit-user-permission-record", "allowed_purpose": "reference_reading",
        "files": [{"member": "selected/example.c", "destination": "example.c",
                   "sha256": hashlib.sha256(b"int allowed;\n").hexdigest()}]}]}
    manifest.write_text(json.dumps(document))
    destination = tmp_path / "stage"
    R.stage(manifest, destination)
    assert (destination / "authorized_archive/example.c").read_bytes() == b"int allowed;\n"
    assert {p.name for p in destination.rglob("*") if p.is_file()} == {
        "example.c", "reference-manifest.json", "reference-selection.json"}
    document["origins"][0]["identity"] = "0" * 64
    manifest.write_text(json.dumps(document))
    with pytest.raises(SpecError, match="authorized identity"):
        R.stage(manifest, tmp_path / "changed")
