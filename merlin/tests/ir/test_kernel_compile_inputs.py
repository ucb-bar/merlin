"""Standalone tensor-kernel byte custody and explicit functional ABI staging."""

import hashlib
import json
from pathlib import Path
import struct

import pytest

from merlin.frontends.compile_inputs import stage_compile_inputs
from merlin.frontends.linalg_mlir import parse_mlir_text


def record(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def capture_fixture(root):
    """Protocol-only fixture; no claimed Triton or numeric admission."""
    root.mkdir()
    model = b'''builtin.module {
      func.func @pair(%x: tensor<2xf32>, %old: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
        return %x, %old : tensor<2xf32>, tensor<2xf32>
      }
    }'''
    spec = {"schema": "t2m.shared-tensor-abi.v2", "entry": "pair", "buffers": [
        {"name": "x", "shape": [2], "effect": "read", "dtype": "f32"},
        {"name": "old", "shape": [2], "effect": "readwrite", "dtype": "f32"},
        {"name": "out", "shape": [2], "effect": "write", "dtype": "f32"}]}
    ttir = b"original TTIR protocol fixture"
    provenance = {"schema": "t2m.shared-tensor-provenance.v2", "spec": spec,
                  "mlir_sha256": record(model)["sha256"], "source_sha256": record(ttir)["sha256"],
                  "source_operations": [{"id": "original.0"}], "normalized_operations": [{"id": "normalized.0"}],
                  "example_numeric_validation": {"input_sha256": "EVALUATOR_DIGEST"}}
    files = {"model.mlir": model, "source.ttir": ttir, "spec.json": encoded(spec),
             "provenance.json": encoded(provenance), "weights.safetensors": struct.pack("<Q", 8) + b"{}      ",
             "weights.safetensors.manifest.json": encoded({"0": {"kind": "input", "name": "x"},
                                                           "1": {"kind": "input", "name": "old"}}),
             "inputs.npz": b"EVALUATOR INPUTS", "golden.npy": b"EVALUATOR GOLDEN",
             "extra.npz": b"EVALUATOR ADDITIONAL REFERENCES", "execution.json": b"EVALUATOR REPORT"}
    receipt = {"schema": "t2m.capture-receipt.v2", "profile": "shared-functional-tensor",
               "materialized_abi": {"complete": True, "inputs": 2, "outputs": 2},
               "artifacts": {name: record(raw) for name, raw in files.items()}}
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    (root / "capture_receipt.json").write_bytes(encoded(receipt))
    return receipt


def test_standalone_staging_preserves_order_source_inventory_and_readonly_inputs(tmp_path, monkeypatch):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    receipt = capture_fixture(source)
    original_read = Path.read_bytes

    def read(path):
        if path.name in ("inputs.npz", "golden.npy", "extra.npz", "execution.json"):
            raise AssertionError("compiler accessed evaluator values")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    manifest = stage_compile_inputs(source, target)
    assert manifest["source_profile"] == "standalone-tensor-kernel"
    module = parse_mlir_text((target / "program.mlir").read_text())
    function = next(iter(module.ops))
    assert function.sym_name.data == "forward"
    assert all(a.data["bufferization.writable"].value.data == 0 for a in function.arg_attrs)
    signature = json.loads((target / "signature.json").read_text())
    assert len(signature["inputs"]) == len(signature["outputs"]) == 2
    correspondence = json.loads((target / "source-correspondence.json").read_text())
    assert correspondence["source_entry"] == "pair"
    assert correspondence["staged_entry"] == "forward"
    compiler_provenance = json.loads((target / "source/compiler-provenance.json").read_text())
    assert compiler_provenance["source_operations"] == [{"id": "original.0"}]
    assert compiler_provenance["normalized_operations"] == [{"id": "normalized.0"}]
    data = b"".join((target / name).read_bytes() for name in manifest["members"])
    assert b"EVALUATOR" not in data
    for name in ("inputs.npz", "golden.npy", "extra.npz", "execution.json"):
        assert receipt["artifacts"][name]["sha256"].encode() not in data


@pytest.mark.parametrize("change", ["source_bytes", "source_relation", "shape", "effects", "manifest", "entry", "unresolved"])
def test_changed_or_rehashed_inconsistent_kernel_never_publishes(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    receipt = capture_fixture(source)
    if change == "source_bytes":
        (source / "source.ttir").write_bytes(b"different source")
    else:
        spec = json.loads((source / "spec.json").read_text())
        provenance = json.loads((source / "provenance.json").read_text())
        if change == "source_relation":
            provenance["source_sha256"] = "0" * 64
        elif change == "shape":
            spec["buffers"][0]["shape"] = [3]
        elif change == "effects":
            spec["buffers"][1]["effect"] = "write"
        elif change == "manifest":
            (source / "weights.safetensors.manifest.json").write_bytes(b'{}')
        elif change == "entry":
            spec["entry"] = "different"
        else:
            raw = (source / "model.mlir").read_bytes().replace(b"return %x", b'"unknown.effect"() : () -> ()\n return %x')
            (source / "model.mlir").write_bytes(raw)
            provenance["mlir_sha256"] = record(raw)["sha256"]
        provenance["spec"] = spec
        (source / "spec.json").write_bytes(encoded(spec))
        (source / "provenance.json").write_bytes(encoded(provenance))
        for name in receipt["artifacts"]:
            receipt["artifacts"][name] = record((source / name).read_bytes())
        (source / "capture_receipt.json").write_bytes(encoded(receipt))
    with pytest.raises(ValueError):
        stage_compile_inputs(source, target)
    assert not target.exists()
