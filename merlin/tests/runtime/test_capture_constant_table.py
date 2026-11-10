"""Immutable compiler tables share the streaming reader's exact byte contract."""

from dataclasses import FrozenInstanceError
import hashlib
import json
import struct

import pytest

from merlin.runtime.captured_constants import CaptureConstantTable, verify_capture_constant


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(root, storage="BF16", shape=(3,)):
    types = {"BF16": ("bfloat16", "bf16", 2), "F16": ("float16", "f16", 2), "F32": ("float32", "f32", 4),
        "F64": ("float64", "f64", 8), "I8": ("int8", "i8", 1), "I16": ("int16", "i16", 2),
        "I32": ("int32", "i32", 4), "I64": ("int64", "i64", 8), "BOOL": ("bool", "i1", 1)}
    manifest_dtype, dtype, itemsize = types[storage]
    import math
    payload = bytes((i * 53 + 7) % 256 for i in range(math.prod(shape) * itemsize))
    manifest = {"0": {"kind": "constant", "weight": "literal", "name": "c_literal", "source_target": "literal",
                      "dtype": manifest_dtype, "shape": list(shape), "sha256": sha(payload)}}
    header = json.dumps({"literal": {"dtype": storage, "shape": list(shape), "data_offsets": [0, len(payload)]}}).encode()
    raw_manifest = json.dumps(manifest).encode()
    blob = struct.pack("<Q", len(header)) + header + payload
    mp, bp = root / "manifest.json", root / "state.safetensors"
    mp.write_bytes(raw_manifest)
    bp.write_bytes(blob)
    table = CaptureConstantTable(raw_manifest, blob, sha(raw_manifest), sha(blob), str(mp.resolve()), str(bp.resolve()))
    args = dict(entry_argument_index=0, source_shape=shape, source_dtype=dtype, max_payload_bytes=max(1, len(payload)))
    return table, args, mp, bp, payload


@pytest.mark.parametrize("storage", ["BF16", "F16", "F32", "F64", "I8", "I16", "I32", "I64", "BOOL"])
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
def test_table_matches_streaming_reader_without_numeric_interpretation(tmp_path, storage, shape):
    table, args, mp, bp, payload = fixture(tmp_path, storage, shape)
    streamed = verify_capture_constant(manifest_path=mp, manifest_sha256=table.manifest_sha256,
        safetensors_path=bp, safetensors_sha256=table.safetensors_sha256, **args)
    assert table.read(**args) == streamed
    assert streamed.logical_payload == payload and streamed.to_evidence()["prepack_authorized"] is False


def test_table_keeps_its_pinned_snapshot_after_external_files_change(tmp_path):
    table, args, mp, bp, payload = fixture(tmp_path)
    bp.write_bytes(b"changed")
    mp.write_text("{}")
    assert table.read(**args).logical_payload == payload
    with pytest.raises(FrozenInstanceError):
        table.safetensors_bytes = b"changed"


@pytest.mark.parametrize("change", ["index", "dtype", "shape", "budget", "manifest-pin", "blob-pin", "source-hash"])
def test_invalid_selection_or_pin_is_refused(tmp_path, change):
    table, args, *_ = fixture(tmp_path)
    if change == "index": args["entry_argument_index"] = 1
    elif change == "dtype": args["source_dtype"] = "f16"
    elif change == "shape": args["source_shape"] = (1, 3)
    elif change == "budget": args["max_payload_bytes"] = 1
    elif change in ("manifest-pin", "blob-pin"):
        with pytest.raises(ValueError, match="SHA-256"):
            CaptureConstantTable(table.manifest_bytes, table.safetensors_bytes,
                "0" * 64 if change == "manifest-pin" else table.manifest_sha256,
                "0" * 64 if change == "blob-pin" else table.safetensors_sha256)
        return
    elif change == "source-hash":
        manifest = json.loads(table.manifest_bytes)
        manifest["0"]["sha256"] = "0" * 64
        raw = json.dumps(manifest).encode()
        table = CaptureConstantTable(raw, table.safetensors_bytes, sha(raw), table.safetensors_sha256)
    with pytest.raises(ValueError):
        table.read(**args)
