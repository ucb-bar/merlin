"""Compiler-only TTIR receipt custody, independent of frontend numerical admission."""

import hashlib
import json
import struct
from pathlib import Path

import pytest

from merlin.compiler.frontend.inputs import stage_compile_inputs
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text


def _capture(root, *, declared_extent=4):
    root.mkdir()
    model = b"""module attributes {prov.family = "pointwise"} {
      func.func @copy(%x: tensor<4xf32>) -> tensor<4xf32> {
        return %x : tensor<4xf32>
      }
    }"""
    source = b"""module {tt.func @copy(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
      %v = tt.load %x : !tt.ptr<f32>
      tt.store %y, %v : !tt.ptr<f32>
      tt.return
    }}"""
    spec = {
        "schema": "t2m.shared-tensor-abi.v2",
        "entry": "copy",
        "buffers": [
            {"name": "x", "shape": [declared_extent], "dtype": "f32", "effect": "read"},
            {"name": "y", "shape": [declared_extent], "dtype": "f32", "effect": "write"},
        ],
    }

    def digest(value):
        return hashlib.sha256(value).hexdigest()

    provenance = {
        "schema": "t2m.shared-tensor-provenance.v2",
        "spec": spec,
        "mlir_sha256": digest(model),
        "source_sha256": digest(source),
        "source_operations": [{"id": "load"}, {"id": "store"}],
        "normalized_operations": [{"id": "copy"}],
    }
    members = {
        "model.mlir": model,
        "source.ttir": source,
        "spec.json": json.dumps(spec).encode(),
        "provenance.json": json.dumps(provenance).encode(),
        "weights.safetensors": struct.pack("<Q", 8) + b"{}      ",
        "weights.safetensors.manifest.json": b'{"0": {"kind": "input", "name": "x"}}',
        "inputs.npz": b"evaluator input",
        "golden.npy": b"evaluator reference",
    }
    receipt = {
        "schema": "t2m.capture-receipt.v2",
        "profile": "shared-functional-tensor",
        "materialized_abi": {"complete": True, "inputs": 1, "outputs": 1},
        "artifacts": {name: {"sha256": digest(value), "bytes": len(value)} for name, value in members.items()},
    }
    for name, value in members.items():
        (root / name).write_bytes(value)
    (root / "capture_receipt.json").write_text(json.dumps(receipt))
    return model


def test_tensor_kernel_stage_retains_source_and_excludes_evaluator_data(tmp_path, monkeypatch):
    capture, target = tmp_path / "capture", tmp_path / "compiler"
    original = _capture(capture)
    read = Path.read_bytes

    def checked_read(path):
        if path.parent == capture and path.name in {"inputs.npz", "golden.npy"}:
            raise AssertionError("compiler staging opened evaluator data")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", checked_read)
    manifest = stage_compile_inputs(capture, target)
    assert manifest["source_profile"] == "standalone-tensor-kernel"
    assert {"inputs.npz", "golden.npy"}.isdisjoint(manifest["members"])
    assert (target / "source/model.mlir").read_bytes() == original
    staged = parse_mlir_text((target / "program.mlir").read_text())
    staged.verify()
    assert staged.attributes["prov.family"].data == "pointwise"
    signature = json.loads((target / "signature.json").read_text())
    assert signature["inputs"] == signature["outputs"] == [{"shape": [4], "dtype": "f32"}]
    assert signature["entry"] == "forward"


def test_rehashed_false_tensor_abi_cannot_replace_the_actual_entry_type(tmp_path):
    capture, target = tmp_path / "capture", tmp_path / "compiler"
    _capture(capture, declared_extent=5)
    with pytest.raises(ValueError):
        stage_compile_inputs(capture, target)
    assert not target.exists()
