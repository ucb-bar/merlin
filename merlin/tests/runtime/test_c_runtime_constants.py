"""Lifted source constants use the ordinary exact weight blob transport."""

import hashlib
import json
import struct

import numpy as np
import pytest

from merlin.llvmlower import c_runtime


@pytest.mark.parametrize("storage", ["streamed", "materialized"])
@pytest.mark.parametrize("dtype,storage_dtype,width", [("f16","F16",2), ("bf16","BF16",2), ("f32","F32",4),
    ("f64","F64",8), ("i1","BOOL",1), ("i8","I8",1), ("i16","I16",2), ("i32","I32",4), ("i64","I64",8)])
def test_constant_argument_uses_original_weight_bytes(tmp_path, monkeypatch, storage, dtype, storage_dtype, width):
    model, out = tmp_path / "source", tmp_path / "generated"
    model.mkdir()
    payload = bytes((i * 53 + 7) % 256 for i in range(width * 3)) if dtype != "i1" else b"\x00\x01\x01"
    header = json.dumps({"literal":{"dtype":storage_dtype,"shape":[3],"data_offsets":[0,len(payload)]}}).encode()
    (model / "weights.safetensors").write_bytes(struct.pack("<Q",len(header)) + header + payload)
    manifest = {"0":{"kind":"constant", "name":"c_literal", "source_target":"literal", "weight":"literal",
        "shape":[3], "dtype":dtype, "sha256":hashlib.sha256(payload).hexdigest()}}
    (model / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    (model / "model.mlir").write_text(f"module {{func.func @forward(%literal:tensor<3x{dtype}>)->tensor<3x{dtype}> {{return %literal:tensor<3x{dtype}>}}}}")
    np.savez(model / "inputs.npz")
    monkeypatch.setenv("MERLIN_WEIGHT_PAYLOAD_STORAGE",storage)
    assert c_runtime.manifest_requires_weight_blob(manifest)
    c_runtime.generate(model,out,model / "inputs.npz")
    assert (out / "weights.bin").read_bytes() == payload
    assert "{MERLIN_WEIGHT," in (out / "model_gen.h").read_text()
    manifest["0"]["stub"] = True
    (model / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="constants cannot use.*stubs"):
        c_runtime.generate(model,tmp_path / "stub",model / "inputs.npz")
