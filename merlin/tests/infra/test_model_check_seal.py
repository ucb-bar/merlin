"""A sealed bundle carries a sha256 of every file; verify finds any change made after the seal."""

from __future__ import annotations

import json

import pytest

from merlin.model_check import seal as SL
from merlin.model_check import binding as BD
from merlin.model_check.bundle import Bundle, BundleError


def _bundle(tmp_path) -> Bundle:
    root = tmp_path / "bundle"
    program = root / "m" / "programs" / "p"
    program.mkdir(parents=True)
    (program / "catalog.json").write_text("{}")
    (program / "census.json").write_text("{}")
    (root / "m" / "blobs").mkdir()
    (root / "m" / "blobs" / "a.npy").write_bytes(b"\x00\x01")
    (root / "m" / ".fp_work").mkdir()  # a builder's work directory is not part of the bundle
    (root / "m" / ".fp_work" / "scratch").write_text("x")
    manifest = {"schema": "merlin.model_check.bundle.v1", "models": {"m": {"programs": {"p": {}}}}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    return Bundle(root)


def test_a_sealed_bundle_verifies_and_work_directories_are_ignored(tmp_path):
    bundle = _bundle(tmp_path)
    record = SL.seal(bundle, jobs=2)
    assert record["files"] == 3
    (bundle.root / "m" / ".fp_work" / "scratch").write_text("changed later, not part of the bundle")
    assert SL.verify(bundle, jobs=2)["status"] == "pass"


@pytest.mark.parametrize("kind", ["changed", "missing", "extra"])
def test_any_change_after_the_seal_fails(tmp_path, kind):
    bundle = _bundle(tmp_path)
    SL.seal(bundle)
    blob = bundle.root / "m" / "blobs" / "a.npy"
    if kind == "changed":
        blob.write_bytes(b"\x00\x02")
    elif kind == "missing":
        blob.unlink()
    else:
        (bundle.root / "m" / "blobs" / "b.npy").write_bytes(b"new")
    result = SL.verify(bundle)
    assert result["status"] == "fail" and result["counts"][kind] == 1


def test_a_missing_catalog_is_a_structure_failure(tmp_path):
    bundle = _bundle(tmp_path)
    (bundle.root / "m" / "programs" / "p" / "census.json").unlink()
    SL.seal(bundle)
    result = SL.verify(bundle)
    assert result["status"] == "fail" and result["structure"] == ["m/p: no census.json"]


def test_an_unsealed_bundle_is_refused(tmp_path):
    with pytest.raises(BundleError, match="not sealed"):
        SL.verify(_bundle(tmp_path))


def test_raw_binding_receipt_matches_manifest_digest_and_status(tmp_path):
    bundle = _bundle(tmp_path)
    receipt = {"status": "pass", "checks": []}
    (bundle.root / "binding.json").write_bytes(BD.serialize(receipt))
    manifest = bundle.manifest()
    manifest["binding"] = {"receipt": "binding.json", "receipt_sha256": BD.digest(receipt), "status": "pass"}
    (bundle.root / "MANIFEST.json").write_text(json.dumps(manifest))
    SL.seal(bundle)
    assert SL.verify(bundle)["status"] == "pass"

    manifest = bundle.manifest()
    manifest["binding"]["status"] = "fail"
    (bundle.root / "MANIFEST.json").write_text(json.dumps(manifest))
    assert SL.verify(bundle)["structure"] == ["binding receipt digest differs from the manifest's"]


def test_legacy_binding_report_still_verifies(tmp_path):
    bundle = _bundle(tmp_path)
    receipt = {"status": "pass", "checks": []}
    (bundle.root / "binding.json").write_text(json.dumps({"binding": {"receipt": receipt}}))
    manifest = bundle.manifest()
    manifest["binding"] = {"receipt": "binding.json", "receipt_sha256": BD.digest(receipt), "status": "pass"}
    (bundle.root / "MANIFEST.json").write_text(json.dumps(manifest))
    SL.seal(bundle)
    assert SL.verify(bundle)["status"] == "pass"
