"""A materialized capture must bind its external quantization manifest end to end."""

import hashlib
import json

import pytest

from merlin.targetgen.application_inventory import verify_capture_receipt


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_capture(root, *, change=None, schema="m2m.quantization_manifest.v1", overrides=None):
    manifest = {"schema": schema, "sites": [{"site_id": "one", "status": "host"}]}
    manifest.update(overrides or {})
    digest = _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    mlir = f'module attributes {{prov.quantization_manifest_sha256 = "{digest}"}} {{}}\n'.encode()
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    pointer = {"path": "quantization-manifest.json", "sha256": _sha(manifest_bytes), "manifest_sha256": digest}
    if change == "pointer":
        pointer["manifest_sha256"] = "0" * 64
    if change == "mlir":
        mlir = mlir.replace(digest.encode(), b"0" * 64)
    if change == "manifest":
        manifest_bytes = (json.dumps({"schema": schema, "sites": []}) + "\n").encode()
        pointer["sha256"] = _sha(manifest_bytes)
    metadata = {"quantization_manifest": pointer}
    contents = {
        "model.mlir": mlir,
        "weights.safetensors": b"weights",
        "weights.safetensors.manifest.json": b"{}",
        "meta.json": json.dumps(metadata).encode(),
        "quantization-manifest.json": manifest_bytes,
    }
    if change == "missing":
        contents.pop("quantization-manifest.json")
        metadata.pop("quantization_manifest")
        contents["meta.json"] = json.dumps(metadata).encode()
    for name, raw in contents.items():
        (root / name).write_bytes(raw)
    (root / "capture_receipt.json").write_text(json.dumps({
        "schema": "m2m.capture-receipt.v1",
        "materialized_abi": {"complete": True},
        "artifacts": {name: {"bytes": len(raw), "sha256": _sha(raw)} for name, raw in contents.items()},
    }))


@pytest.mark.parametrize("schema", ["m2m.quantization_manifest.v1", "m2m.quantization_manifest.v2"])
@pytest.mark.parametrize("change", [None, "pointer", "mlir", "manifest", "missing"])
def test_external_manifest_binding(change, schema, tmp_path):
    _write_capture(tmp_path, change=change, schema=schema)
    observed = verify_capture_receipt(tmp_path / "model.mlir")
    assert observed["status"] == ("verified_materialized" if change is None else "unverified")
    assert observed["source_closure_verified"] is False


@pytest.mark.parametrize("status", ["quantized", "host", "skipped", "preserved"])
def test_v2_generic_site_statuses_are_admitted(tmp_path, status):
    site = {"site_id": "one", "status": status}
    if status == "preserved":
        site["execution_route"] = "fixture.accelerator_float"
    _write_capture(tmp_path, schema="m2m.quantization_manifest.v2", overrides={"sites": [site]})
    observed = verify_capture_receipt(tmp_path / "model.mlir")
    assert observed["status"] == "verified_materialized", observed["errors"]
    assert observed["source_closure_verified"] is False


@pytest.mark.parametrize("overrides,error", [
    ({"schema": "m2m.quantization_manifest.v3"}, "unsupported manifest schema"),
    ({"sites": []}, "nonempty site census"),
    ({"sites": {}}, "nonempty site census"),
    ({"sites": [None]}, "stable site_id"),
    ({"sites": [{"site_id": 1, "status": "host"}]}, "stable site_id"),
    ({"sites": [{"site_id": "one", "status": "unknown"}]}, "explicit status"),
    ({"sites": [{"site_id": "one", "status": "preserved"}]}, "explicit execution route"),
    ({"sites": [{"site_id": "one", "status": "preserved", "execution_route": ""}]}, "explicit execution route"),
    ({"sites": [{"site_id": "one", "status": "preserved", "execution_route": 1}]}, "explicit execution route"),
    ({"sites": [{"site_id": "one", "status": "host"}, {"site_id": "one", "status": "quantized"}]}, "duplicate site IDs"),
])
def test_fully_hash_bound_v2_manifest_still_requires_generic_site_contract(tmp_path, overrides, error):
    # Every artifact, pointer, and MLIR hash agrees with this malformed manifest.
    _write_capture(tmp_path, schema="m2m.quantization_manifest.v2", overrides=overrides)
    observed = verify_capture_receipt(tmp_path / "model.mlir")
    assert observed["status"] == "unverified"
    assert any(error in message for message in observed["errors"])
    assert observed["source_closure_verified"] is False


def test_v1_binding_keeps_legacy_census_compatibility(tmp_path):
    _write_capture(tmp_path, overrides={"sites": []})
    observed = verify_capture_receipt(tmp_path / "model.mlir")
    assert observed["status"] == "verified_materialized", observed["errors"]
    assert observed["source_closure_verified"] is False
