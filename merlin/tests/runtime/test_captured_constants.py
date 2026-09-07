"""Pinned capture reader refuses ambiguous or malformed state; no model runtime."""
from dataclasses import FrozenInstanceError
import hashlib
import json
import struct

import pytest

from merlin.runtime.captured_constants import verify_capture_constant


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _files(tmp_path, *, manifest=None, header=None, payload=b"\x01\xff\x02\x80",
           raw_manifest=None, raw_header=None):
    manifest = manifest if manifest is not None else {
        "0": {"kind": "input", "name": "activation"},
        "1": {"kind": "buffer", "weight": "state", "shape": [2, 2], "dtype": "int8"}}
    header = header if header is not None else {
        "state": {"dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]}}
    manifest_bytes = raw_manifest if raw_manifest is not None else json.dumps(manifest).encode()
    header_bytes = raw_header if raw_header is not None else json.dumps(header).encode()
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + payload
    mp, bp = tmp_path / "state.manifest.json", tmp_path / "state.safetensors"
    mp.write_bytes(manifest_bytes)
    bp.write_bytes(blob)
    return {"manifest_path": mp, "manifest_sha256": _sha(manifest_bytes),
            "safetensors_path": bp, "safetensors_sha256": _sha(blob),
            "entry_argument_index": 1, "source_shape": [2, 2],
            "source_dtype": "i8", "max_payload_bytes": 4}


def test_exact_bytes_and_evidence_not_authorization(tmp_path):
    args = _files(tmp_path)
    value = verify_capture_constant(**args)
    assert value.logical_payload == b"\x01\xff\x02\x80"
    assert value.payload_sha256 == _sha(value.logical_payload)
    evidence = value.to_evidence()
    assert evidence["prepack_authorized"] is False
    assert evidence["entry_argument_index"] == 1
    assert evidence["tensor_key"] == "state"
    assert evidence["payload_bytes"] == 4
    with pytest.raises(FrozenInstanceError):
        value.source_dtype = "f32"


@pytest.mark.parametrize("field", ["manifest_sha256", "safetensors_sha256"])
def test_stale_pins_refused(tmp_path, field):
    args = _files(tmp_path)
    args[field] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        verify_capture_constant(**args)


@pytest.mark.parametrize("entry", [
    {"kind": "input", "name": "state"},
    {"kind": "buffer", "weight": "state", "shape": [2, 2], "dtype": "int8", "stub": True},
    {"kind": "param", "weight": "state", "shape": [2, 2], "dtype": "int8", "error": "missing"},
    {"kind": "buffer", "shape": [2, 2], "dtype": "int8"},
])
def test_missing_dynamic_or_stub_constant_refused(tmp_path, entry):
    with pytest.raises(ValueError, match="concrete captured"):
        verify_capture_constant(**_files(tmp_path, manifest={"1": entry}))


@pytest.mark.parametrize("field,value", [
    ("entry_argument_index", 9), ("entry_argument_index", True),
    ("source_shape", [4]), ("source_shape", [True, 2]),
    ("source_dtype", "f32"), ("max_payload_bytes", 3),
])
def test_source_identity_or_budget_mismatch_refused(tmp_path, field, value):
    args = _files(tmp_path)
    args[field] = value
    with pytest.raises(ValueError):
        verify_capture_constant(**args)


@pytest.mark.parametrize("kind", ["manifest", "header"])
def test_duplicate_json_keys_refused(tmp_path, kind):
    raw = b'{"state":{},"state":{}}' if kind == "header" else b'{"1":{},"1":{}}'
    with pytest.raises(ValueError, match="duplicate"):
        verify_capture_constant(**_files(tmp_path, **{f"raw_{kind}": raw}))


def test_numeric_manifest_alias_refused(tmp_path):
    with pytest.raises(ValueError, match="canonical"):
        verify_capture_constant(**_files(tmp_path, manifest={"01": {}}))


@pytest.mark.parametrize("offsets", [[-1, 3], [0, 5], [3, 2], [0, 3], [True, 4], [0]])
def test_malformed_or_out_of_bounds_ranges_refused(tmp_path, offsets):
    header = {"state": {"dtype": "I8", "shape": [2, 2], "data_offsets": offsets}}
    with pytest.raises(ValueError):
        verify_capture_constant(**_files(tmp_path, header=header))


def test_overlapping_unselected_tensor_refused(tmp_path):
    header = {
        "state": {"dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]},
        "other": {"dtype": "I8", "shape": [2], "data_offsets": [2, 4]},
    }
    with pytest.raises(ValueError, match="overlap"):
        verify_capture_constant(**_files(tmp_path, header=header))


def test_unselected_payload_still_hash_verified(tmp_path):
    header = {
        "state": {"dtype": "I8", "shape": [2, 2], "data_offsets": [0, 4]},
        "other": {"dtype": "I8", "shape": [4], "data_offsets": [4, 8]},
    }
    args = _files(tmp_path, header=header, payload=b"12345678")
    assert verify_capture_constant(**args).logical_payload == b"1234"
    path = args["safetensors_path"]
    path.write_bytes(path.read_bytes()[:-1] + b"9")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_capture_constant(**args)


def test_scalar_f32_bit_pattern_preserved_without_numeric_conversion(tmp_path):
    payload = b"\x00\x00\x00\x80"  # Negative zero, copied as bits.
    args = _files(tmp_path,
        manifest={"1": {"kind": "param", "weight": "state", "shape": [], "dtype": "float32"}},
        header={"state": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}, payload=payload)
    args.update(source_shape=[], source_dtype="f32")
    assert verify_capture_constant(**args).logical_payload == payload


def test_oversized_header_rejected_before_allocation(tmp_path):
    args = _files(tmp_path)
    blob = struct.pack("<Q", 2**63)
    args["safetensors_path"].write_bytes(blob)
    args["safetensors_sha256"] = _sha(blob)
    with pytest.raises(ValueError, match="bounded"):
        verify_capture_constant(**args)


def test_missing_blob_key_and_unknown_dtype_refused(tmp_path):
    for header in ({"other": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}},
                   {"state": {"dtype": "UNKNOWN", "shape": [4], "data_offsets": [0, 4]}}):
        with pytest.raises(ValueError):
            verify_capture_constant(**_files(tmp_path, header=header))
