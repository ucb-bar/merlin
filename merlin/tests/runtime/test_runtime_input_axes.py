"""Input-axis invocation contracts retain actual body shapes and weight bytes."""

import json
import struct

import numpy as np
import pytest

from merlin.llvmlower.c_runtime import generate
from merlin.llvmlower.runtime_invocation import resolve


def example(root, length=7, kind="param"):
    root.mkdir()
    (root / "model.mlir").write_text("""module {
      func.func @forward(%weight:tensor<1xi32>,%x:tensor<?xi32>)->(tensor<?xi32>,tensor<1xi32>) {
        return %x,%weight:tensor<?xi32>,tensor<1xi32>
      }}""")
    signature = {
        "schema": "merlin.compile-signature.v3",
        "entry": "forward",
        "inputs": [{"shape": [1], "dtype": "i32"}, {"shape": ["rows"], "dtype": "i32"}],
        "outputs": [{"shape": ["rows"], "dtype": "i32"}, {"shape": [1], "dtype": "i32"}],
        "dimension_bindings": {"rows": {"argument": 1, "axis": 0}},
    }
    (root / "signature.json").write_text(json.dumps(signature))
    manifest = {"0": {"kind": kind, "weight": "weight"}, "1": {"kind": "input", "name": "x"}}
    (root / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    header = json.dumps({"weight": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    payload = np.array([17], np.int32).tobytes()
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    np.savez(root / "inputs.npz", in0=np.array([17], np.int32), in1=np.arange(length, dtype=np.int32))
    return root, signature


@pytest.mark.parametrize("length", [0, 1, 31])
def test_axis_dimensions_follow_actual_function_argument_and_preserve_program(tmp_path, length):
    root, _signature = example(tmp_path / "model", length)
    original = (root / "model.mlir").read_bytes()
    actual = resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")
    assert actual["input_axis_dimensions"] == {"rows": length}
    assert actual["outputs"] == [{"shape": [length], "dtype": "i32"}, {"shape": [1], "dtype": "i32"}]
    assert actual["scalars"] == [] and (root / "model.mlir").read_bytes() == original


@pytest.mark.parametrize(
    "damage",
    [
        "downgrade",
        "missing",
        "coordinate",
        "unknown",
        "extra",
        "bounds",
        "boolean",
        "output",
        "static",
        "dtype",
        "body",
    ],
)
def test_rehashed_contract_cannot_change_axis_types_result_sizes_or_source_body(tmp_path, damage):
    root, signature = example(tmp_path / "model")
    if damage == "downgrade":
        signature["schema"] = "merlin.compile-signature.v2"
    elif damage == "missing":
        signature.pop("dimension_bindings")
    elif damage == "coordinate":
        signature["dimension_bindings"]["rows"]["argument"] = 0
    elif damage == "unknown":
        signature["outputs"][0]["shape"] = ["unbound"]
    elif damage == "extra":
        signature["dimension_bindings"]["rows"]["source_permission"] = True
    elif damage == "bounds":
        signature["dimension_bindings"]["rows"]["bounds"] = [0, 6]
    elif damage == "boolean":
        signature["dimension_bindings"]["rows"]["axis"] = False
    elif damage == "output":
        signature["outputs"][0]["shape"] = [1]
    elif damage == "static":
        signature["inputs"][0]["shape"] = [2]
    elif damage == "dtype":
        signature["outputs"][0]["dtype"] = "i64"
    else:
        (root / "model.mlir").write_text("""module {
          func.func @forward(%weight:tensor<1xi32>,%x:tensor<?xi32>)->(tensor<?xi32>,tensor<1xi32>) {
            %one=arith.constant 1:index
            %out=tensor.empty(%one):tensor<?xi32>
            return %out,%weight:tensor<?xi32>,tensor<1xi32>
          }}""")
    (root / "signature.json").write_text(json.dumps(signature))
    with pytest.raises(ValueError):
        resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")


@pytest.mark.parametrize("kind", ["param", "buffer", "constant"])
def test_c_runtime_uses_original_weight_blob_with_positional_model_inputs(tmp_path, kind):
    root, _signature = example(tmp_path / "model", kind=kind)
    result = generate(root, tmp_path / "generated", root / "inputs.npz", runtime_signature=root / "signature.json")
    assert result["outputs"] == [([7], "i32"), ([1], "i32")]
    assert (tmp_path / "generated/weights.bin").read_bytes() == np.array([17], np.int32).tobytes()
    assert json.loads((tmp_path / "generated/runtime-abi.json").read_text())["input_axis_dimensions"] == {"rows": 7}


def test_c_runtime_cannot_silently_ignore_changed_parameter_argument(tmp_path):
    root, _signature = example(tmp_path / "model")
    np.savez(root / "inputs.npz", in0=np.array([18], np.int32), in1=np.arange(7, dtype=np.int32))
    with pytest.raises(ValueError, match="must match the captured weight container"):
        generate(root, tmp_path / "generated", root / "inputs.npz", runtime_signature=root / "signature.json")


def test_half_parameter_uses_float16_argument_storage_and_original_word_bytes(tmp_path):
    root, signature = example(tmp_path / "model")
    (root / "model.mlir").write_text((root / "model.mlir").read_text().replace("tensor<1xi32>", "tensor<1xf16>"))
    signature["inputs"][0]["dtype"] = signature["outputs"][1]["dtype"] = "f16"
    (root / "signature.json").write_text(json.dumps(signature))
    value = np.array([0x7C01], np.uint16).view(np.float16)
    header = json.dumps({"weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + value.tobytes())
    np.savez(root / "inputs.npz", in0=value, in1=np.arange(7, dtype=np.int32))
    generate(root, tmp_path / "generated", root / "inputs.npz", runtime_signature=root / "signature.json")
    assert (tmp_path / "generated/weights.bin").read_bytes() == value.tobytes()
