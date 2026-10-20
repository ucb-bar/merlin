"""Fresh release dependencies retain capture-byte authority and old source inodes."""

import hashlib
import json
import os
import shutil

import pytest
import yaml
from merlin_experiments.corpus.preparation import publish_loader_profile
from merlin_experiments.spec import SpecError


def fixture(tmp_path):
    source, capture, snapshot = (tmp_path / name for name in ("source", "capture", "snapshot"))
    for directory in (source, capture, snapshot):
        directory.mkdir()
    loader = b"def get_model_and_inputs(): pass\n"
    profile = b'{"declared_width": 4}\n'
    digest = hashlib.sha256(profile).hexdigest()
    loader_sha = hashlib.sha256(loader).hexdigest()
    (source / "capsule.pytorch.py").write_bytes(loader)
    (snapshot / "loader.py").write_bytes(loader)
    (snapshot / "profile.json").write_bytes(profile)
    metadata = json.dumps({"loader_provenance": {"profile_sha256": digest}}).encode()
    (capture / "meta.json").write_bytes(metadata)
    receipt = json.dumps({"source": {"sha256": loader_sha}, "artifacts": {"meta.json": {
        "sha256": hashlib.sha256(metadata).hexdigest(), "bytes": len(metadata)}}}).encode()
    (source / "source-capture-receipt.json").write_bytes(receipt)
    document = {
        "label": "public", "pytorch_ref": {"loader": "capsule.pytorch.py"},
        "operation": {"attributes": {}}, "materialized_capture": {
            "path": str(capture / "model.mlir"), "loader_sha256": loader_sha,
            "receipt_sha256": hashlib.sha256(receipt).hexdigest()},
    }
    (source / "capsule.yaml").write_text(yaml.safe_dump(document))
    target = tmp_path / "release"
    shutil.copytree(source, target, copy_function=os.link)
    return source, target, snapshot / "profile.json", document


def test_profile_publication_atomically_replaces_only_new_release_binding(tmp_path):
    source, target, profile, document = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    assert (source / "capsule.yaml").stat().st_ino == (target / "capsule.yaml").stat().st_ino
    dependency = publish_loader_profile(source, target, document, profile_sources=(profile,))
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert not (source / "profile.json").exists()
    assert (source / "capsule.yaml").stat().st_ino != (target / "capsule.yaml").stat().st_ino
    assert (target / "profile.json").read_bytes() == profile.read_bytes()
    attrs = yaml.safe_load((target / "capsule.yaml").read_text())["operation"]["attributes"]
    assert attrs == {"loader_profile": "profile.json", "loader_profile_sha256": dependency["profile"]["sha256"]}


@pytest.mark.parametrize("fault", ["profile", "owner", "metadata", "ambiguous"])
def test_profile_dependency_refuses_changed_or_ambiguous_authority(tmp_path, fault):
    source, target, profile, document = fixture(tmp_path)
    profiles = (profile,)
    if fault == "profile":
        profile.write_bytes(b"changed")
    elif fault == "owner":
        profile.with_name("loader.py").write_bytes(b"other owner")
    elif fault == "metadata":
        (tmp_path / "capture/meta.json").write_bytes(b"changed")
    else:
        other = tmp_path / "other"
        shutil.copytree(profile.parent, other)
        profiles += (other / "profile.json",)
    before = (source / "capsule.yaml").read_bytes()
    with pytest.raises(SpecError):
        publish_loader_profile(source, target, document, profile_sources=profiles)
    assert (source / "capsule.yaml").read_bytes() == before
