"""Inspection payloads preserve xDSL storage bits without changing executable IR."""

import hashlib
import struct

import pytest
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import ArrayAttr, BytesAttr, DenseIntOrFPElementsAttr, ModuleOp, TensorType, f32, i32

from merlin.common.ir_audit import IrAudit
from merlin.xdsl_dialects._common import text
from merlin.xdsl_dialects.ir_inspection import compact_text, reconstruct_tensor, record_stage


@pytest.fixture(autouse=True)
def no_processes_or_listeners(monkeypatch):
    import socket
    import subprocess

    def refused(*args, **kwargs):
        pytest.fail("tensor reconstruction must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.mark.parametrize("element_name", ["i8", "si16", "ui32", "i64", "f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("shape", [(), (0,), (4,), (2, 2)])
def test_reopened_dense_reconstruction_preserves_storage(tmp_path, element_name, shape):
    import json
    from math import prod

    from xdsl.dialects.builtin import IntegerType, Signedness, bf16, f16, f64

    elements = {
        "i8": IntegerType(8),
        "si16": IntegerType(16, Signedness.SIGNED),
        "ui32": IntegerType(32, Signedness.UNSIGNED),
        "i64": IntegerType(64),
        "f16": f16,
        "bf16": bf16,
        "f32": f32,
        "f64": f64,
    }
    element = elements[element_name]
    tensor_type = TensorType(element, shape)
    # Opaque storage includes arbitrary FP payload bits; no numeric unpack/repack.
    payload = bytes(range(element.compile_time_size)) * prod(shape)
    original = DenseIntOrFPElementsAttr(tensor_type, BytesAttr(payload))
    with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
        record = audit.tensor(payload, element_type=str(element), shape=shape)
        audit.stage("weights-bias", "exact", inspection="view", inspection_tensors=[record])
    index = json.loads((audit.directory / "index.json").read_text())
    descriptor = index["stages"][0]["inspection"]["tensors"][0]
    restored = reconstruct_tensor(audit.directory, descriptor, tensor_type=tensor_type)
    assert restored == original
    assert restored.data.data == payload


def test_dense_nan_and_negative_zero_reconstruction(tmp_path):
    module, payload = dense_module()
    with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
        record_stage(audit, "weight", module)
    descriptor = audit.record["stages"][0]["inspection"]["tensors"][0]
    restored = reconstruct_tensor(audit.directory, descriptor, tensor_type=TensorType(f32, [128]))
    assert restored.data.data == payload
    assert text(ModuleOp([ConstantOp(restored)])) == text(module)


def test_shared_storage_reconstructs_weight_and_bias_shapes_independently(tmp_path):
    payload = bytes(range(16))
    with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
        weight = audit.tensor(payload, element_type="i32", shape=(2, 2))
        bias = audit.tensor(payload, element_type="i32", shape=(4,))
    assert weight["file"] == bias["file"]
    for descriptor in (weight, bias):
        expected = TensorType(i32, descriptor["shape"])
        restored = reconstruct_tensor(audit.directory, descriptor, tensor_type=expected)
        assert restored.get_type() == expected
        assert restored.data.data == payload


@pytest.mark.parametrize("kind", ["vector", "encoded", "dynamic", "i1", "shape", "element", "storage"])
def test_reconstruction_refuses_unsupported_or_mismatched_types(tmp_path, kind):
    from xdsl.dialects.builtin import IntegerType, StringAttr, VectorType

    tensor_type = TensorType(i32, [4])
    element = "i32"
    shape = (4,)
    payload = bytes(16)
    if kind == "vector":
        tensor_type = VectorType(i32, [4])
    elif kind == "encoded":
        tensor_type = TensorType(i32, [4], StringAttr("unsupported"))
    elif kind == "dynamic":
        tensor_type = TensorType(i32, [-1])
    elif kind == "i1":
        tensor_type = TensorType(IntegerType(1), [4])
        element = "i1"
    elif kind == "shape":
        shape = (2, 2)
    elif kind == "element":
        element = "f32"
    else:
        payload = bytes(4)
    with IrAudit(tmp_path, enabled="both", producer="fixture", source=__file__) as audit:
        descriptor = audit.tensor(payload, element_type=element, shape=shape)
    with pytest.raises(ValueError):
        reconstruct_tensor(audit.directory, descriptor, tensor_type=tensor_type)


def dense_module():
    # Negative zero and two different NaN payloads must not pass through float conversion.
    payload = struct.pack("<IIII", 0x80000000, 0x7FC00001, 0x7FC00123, 0x3F800000) * 32
    dense = DenseIntOrFPElementsAttr(TensorType(f32, [128]), BytesAttr(payload))
    return ModuleOp([ConstantOp(dense)]), payload


@pytest.mark.parametrize("mode", ["compact", "both"])
def test_dense_storage_bits_and_stage_deduplication(tmp_path, mode):
    module, payload = dense_module()
    before = text(module)
    with IrAudit(tmp_path, enabled=mode, producer="synthetic", source=__file__) as audit:
        record_stage(audit, "first", module)
        record_stage(audit, "second", module)
    assert text(module) == before
    descriptors = [stage["inspection"]["tensors"][0] for stage in audit.record["stages"]]
    assert descriptors[0] == descriptors[1]
    descriptor = descriptors[0]
    assert descriptor["format"] == "xdsl-dense-bytes"
    assert descriptor["element_type"] == "f32"
    assert list(descriptor["shape"]) == [128]
    assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
    assert descriptor["bytes"] == len(payload)
    assert (audit.directory / descriptor["file"]).read_bytes() == payload
    assert len(list((audit.directory / descriptor["file"]).parent.iterdir())) == 1
    for stage in audit.record["stages"]:
        view = (audit.directory / stage["inspection"]["file"]).read_text()
        assert descriptor["file"] in view and descriptor["sha256"] in view
        assert "inspection_tensor<" in view
        if mode == "both":
            assert (audit.directory / stage["file"]).read_text() == before


def test_small_dense_remains_inline_and_nested_large_attribute_uses_sink():
    small = DenseIntOrFPElementsAttr.from_list(TensorType(i32, [2]), [1, 2])
    large = DenseIntOrFPElementsAttr.from_list(TensorType(i32, [128]), list(range(128)))
    module = ModuleOp([])
    module.attributes["nested"] = ArrayAttr([small, ArrayAttr([large])])
    before = text(module)
    observed = []

    def sink(payload, **metadata):
        observed.append((payload, metadata))
        return {"file": "tensors/synthetic.bin", "sha256": hashlib.sha256(payload).hexdigest()}

    view = compact_text(module, tensor_sink=sink)
    assert observed == [(large.data.data, {"element_type": "i32", "shape": (128,)})]
    assert "dense<[1, 2]>" in view
    assert "tensors/synthetic.bin" in view
    assert text(module) == before
    assert "dense<...>" in compact_text(module)


def test_exact_audit_never_exports_inspection_tensors(tmp_path, monkeypatch):
    module, _ = dense_module()
    with IrAudit(tmp_path, enabled="exact", producer="synthetic", source=__file__) as audit:
        monkeypatch.setattr(audit, "tensor", lambda *args, **kwargs: pytest.fail("exact mode exported tensor"))
        record_stage(audit, "exact", module)
    stage = audit.record["stages"][0]
    assert "inspection" not in stage
    assert (audit.directory / stage["file"]).read_text() == text(module)
