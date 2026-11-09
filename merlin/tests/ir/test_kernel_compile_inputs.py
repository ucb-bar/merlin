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


def scalar_fixture(root):
    receipt = capture_fixture(root)
    model = (root / "model.mlir").read_bytes().replace(b"%old: tensor<2xf32>", b"%old: tensor<2xf32>, %alpha: f32")
    spec = json.loads((root / "spec.json").read_text())
    spec.update(schema="t2m.shared-tensor-abi.v3", runtime_scalars=[{"name": "alpha", "dtype": "f32", "source_arg": 1}])
    provenance = json.loads((root / "provenance.json").read_text())
    provenance.update(spec=spec, mlir_sha256=record(model)["sha256"])
    manifest = json.loads((root / "weights.safetensors.manifest.json").read_text())
    manifest["2"] = {"kind": "input", "name": "alpha"}
    for name, raw in {"model.mlir": model, "spec.json": encoded(spec), "provenance.json": encoded(provenance),
                      "weights.safetensors.manifest.json": encoded(manifest)}.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    receipt["materialized_abi"]["inputs"] = 3
    (root / "capture_receipt.json").write_bytes(encoded(receipt))
    return receipt


def test_plain_runtime_scalar_staging_retains_type_without_tensor_mutability(tmp_path):
    source, target = tmp_path / "source", tmp_path / "compiler"
    scalar_fixture(source)
    stage_compile_inputs(source, target)
    signature = json.loads((target / "signature.json").read_text())
    assert signature["inputs"][-1] == {"shape": [], "dtype": "f32", "kind": "scalar"}
    assert signature["scalar_abi"] == "native-by-value; invocation-data"
    entry = next(iter(parse_mlir_text((target / "program.mlir").read_text()).ops))
    assert "bufferization.writable" in entry.arg_attrs.data[0].data
    assert "bufferization.writable" not in entry.arg_attrs.data[-1].data


def dynamic_fixture(root, shape=("n",)):
    receipt = scalar_fixture(root)
    tensor_type = ("tensor<" + "x".join("?" if isinstance(d, str) else str(d) for d in shape) + "xf32>").encode()
    model = (root / "model.mlir").read_bytes().replace(b"tensor<2xf32>", tensor_type).replace(b"%alpha: f32", b"%n: i32")
    spec = json.loads((root / "spec.json").read_text())
    spec.update(schema="t2m.shared-tensor-abi.v5", grid=[{"dimension": "n", "divisor": 8}],
                runtime_scalars=[{"name": "n", "dtype": "i32", "source_arg": 1, "bounds": [0, 13]}])
    for buffer in spec["buffers"]:
        buffer["shape"] = list(shape)
    provenance = json.loads((root / "provenance.json").read_text())
    provenance.update(spec=spec, mlir_sha256=record(model)["sha256"])
    manifest = json.loads((root / "weights.safetensors.manifest.json").read_text())
    manifest["2"]["name"] = "n"
    for name, raw in {"model.mlir": model, "spec.json": encoded(spec), "provenance.json": encoded(provenance),
                      "weights.safetensors.manifest.json": encoded(manifest)}.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    (root / "capture_receipt.json").write_bytes(encoded(receipt))
    return receipt


@pytest.mark.parametrize("shape", [("n",), (2, "n", 3), ("n", "n")])
def test_runtime_shape_staging_retains_symbol_bindings_without_reading_examples(tmp_path, monkeypatch, shape):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    dynamic_fixture(source, shape)
    original_read = Path.read_bytes
    def read(path):
        if path.name in ("inputs.npz", "golden.npy", "extra.npz", "execution.json"):
            raise AssertionError("dynamic shape staging accessed evaluator data")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    stage_compile_inputs(source, target)
    signature = json.loads((target / "signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v2"
    assert signature["shape_scope"] == "runtime_scalar_bound_dimensions"
    assert signature["inputs"][0]["shape"] == signature["outputs"][0]["shape"] == list(shape)
    assert signature["inputs"][-1] == {"shape": [], "dtype": "i32", "kind": "scalar", "scalar": "n", "bounds": [0, 13]}


@pytest.mark.parametrize("change", ["unbounded", "negative", "unknown_dimension", "grid", "static_type", "old_schema", "rank", "invocation_value"])
def test_rehashed_invalid_runtime_dimension_binding_never_publishes(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    receipt = dynamic_fixture(source)
    spec = json.loads((source / "spec.json").read_text())
    provenance = json.loads((source / "provenance.json").read_text())
    model = (source / "model.mlir").read_bytes()
    if change == "unbounded":
        spec["runtime_scalars"][0].pop("bounds")
    elif change == "negative":
        spec["runtime_scalars"][0]["bounds"] = [-1, 13]
    elif change == "unknown_dimension":
        spec["buffers"][0]["shape"] = ["missing"]
    elif change == "grid":
        spec["grid"][0]["divisor"] = 0
    elif change == "static_type":
        model = model.replace(b"tensor<?xf32>", b"tensor<13xf32>")
    elif change == "old_schema":
        spec["schema"] = "t2m.shared-tensor-abi.v4"
    elif change == "rank":
        spec["buffers"][0]["shape"] = ["n", 2]
    else:
        spec["runtime_scalars"][0]["value"] = 7
    provenance.update(spec=spec, mlir_sha256=record(model)["sha256"])
    for name, raw in {"model.mlir": model, "spec.json": encoded(spec), "provenance.json": encoded(provenance)}.items():
        (source / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    (source / "capture_receipt.json").write_bytes(encoded(receipt))
    with pytest.raises(ValueError):
        stage_compile_inputs(source, target)
    assert not target.exists()


@pytest.mark.parametrize("change", ["tensor_box", "scalar_type", "position", "invocation_value"])
def test_rehashed_invalid_runtime_scalar_profile_refuses(tmp_path, change):
    source, target = tmp_path / "source", tmp_path / "compiler"
    receipt = scalar_fixture(source)
    spec = json.loads((source / "spec.json").read_text())
    if change in ("tensor_box", "scalar_type"):
        model = (source / "model.mlir").read_bytes().replace(b"%alpha: f32", b"%alpha: tensor<f32>" if change == "tensor_box" else b"%alpha: i32")
        (source / "model.mlir").write_bytes(model)
    elif change == "position":
        spec["runtime_scalars"][0]["source_arg"] = 4
    else:
        spec["runtime_scalars"][0]["value"] = 1.25
    provenance = json.loads((source / "provenance.json").read_text())
    provenance.update(spec=spec, mlir_sha256=record((source / "model.mlir").read_bytes())["sha256"])
    (source / "spec.json").write_bytes(encoded(spec))
    (source / "provenance.json").write_bytes(encoded(provenance))
    for name in ("model.mlir", "spec.json", "provenance.json"):
        receipt["artifacts"][name] = record((source / name).read_bytes())
    (source / "capture_receipt.json").write_bytes(encoded(receipt))
    with pytest.raises(ValueError):
        stage_compile_inputs(source, target)
    assert not target.exists()


@pytest.mark.parametrize("bounds", [[0, 13], [-1, 13], [0, 2**31], [7, 2], [False, 13]])
def test_guarded_integer_scalar_profile_has_explicit_ranges(tmp_path, bounds):
    source, target = tmp_path / "source", tmp_path / "compiler"
    receipt = scalar_fixture(source)
    model = (source / "model.mlir").read_bytes().replace(b"%alpha: f32", b"%alpha: i32")
    spec = json.loads((source / "spec.json").read_text())
    spec.update(schema="t2m.shared-tensor-abi.v4", runtime_scalars=[{"name": "alpha", "dtype": "i32", "source_arg": 1, "bounds": bounds}])
    provenance = json.loads((source / "provenance.json").read_text())
    provenance.update(spec=spec, mlir_sha256=record(model)["sha256"])
    for name, raw in {"model.mlir": model, "spec.json": encoded(spec), "provenance.json": encoded(provenance)}.items():
        (source / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    (source / "capture_receipt.json").write_bytes(encoded(receipt))
    if bounds[0] is False or bounds[0] > bounds[1] or bounds[1] >= 2**31:
        with pytest.raises(ValueError, match="range is outside"):
            stage_compile_inputs(source, target)
        assert not target.exists()
    else:
        stage_compile_inputs(source, target)
        signature = json.loads((target / "signature.json").read_text())
        assert signature["inputs"][-1]["bounds"] == bounds
