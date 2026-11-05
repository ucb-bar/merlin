"""Standalone tensor storage binding never invents whole-program authority."""

import copy
import json

import numpy as np
import pytest
import yaml

from merlin.targetgen import native_pointer_source_binding as P

SOURCE = """builtin.module {
  func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x2xi8>) -> tensor<2x2xi32> {
    %z = arith.constant 0 : i32
    %init = tensor.splat %z : tensor<2x2xi32>
    %y = linalg.generic {indexing_maps = [
      affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>,
      affine_map<(d0,d1,d2)->(d0,d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<2x3xi8>, tensor<3x2xi8>) outs(%init : tensor<2x2xi32>) {
    ^bb0(%x: i8, %w: i8, %s: i32):
      %xx = arith.extsi %x : i8 to i32
      %ww = arith.extsi %w : i8 to i32
      %p = arith.muli %xx, %ww : i32
      %v = arith.addi %s, %p : i32
      linalg.yield %v : i32
    } -> tensor<2x2xi32>
    func.return %y : tensor<2x2xi32>
  }
}"""


@pytest.fixture
def frozen(tmp_path):
    root, capture = tmp_path / "capsule", tmp_path / "capture"
    root.mkdir()
    capture.mkdir()
    a = np.asarray([[1, -2, 3], [-4, 5, 2]], dtype=np.int8)
    b = np.asarray([[-2, 1], [3, -1], [4, 2]], dtype=np.int8)
    order = ["I0", "I1", "answer"]
    declaration = dict(
        interface_mlir="source.mlir",
        inputs=[
            dict(name="I0", role="input", shape=[2, 3], dtype="i8"),
            dict(name="I1", role="input", shape=[3, 2], dtype="i8"),
        ],
        operation=dict(op="model", attributes=dict(out="answer", arg_order=order)),
        numeric_policy=dict(compare="exact_int", dtype="i32"),
    )
    golden = dict(
        golden_source="independent_fixture",
        outputs=dict(answer=(a.astype(np.int32) @ b.astype(np.int32)).tolist()),
        oracle_provenance=dict(
            interface="linalg_positional",
            linalg_mlir="source.mlir",
            arg_order=order,
            inputs={
                "I0": dict(shape=[2, 3], decoded=a.reshape(-1).tolist()),
                "I1": dict(shape=[3, 2], decoded=b.reshape(-1).tolist()),
            },
        ),
    )
    (root / "source.mlir").write_text(SOURCE)
    (root / "capsule.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    (root / "golden.yaml").write_text(yaml.safe_dump(golden, sort_keys=False))
    (capture / "model.mlir").write_text(SOURCE)
    (capture / "weights.safetensors.manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "left"}, "1": {"kind": "input", "name": "right"}})
    )
    (capture / "input_order.json").write_text(json.dumps({"left": 0, "right": 1}))
    (capture / "weights.safetensors").write_bytes((8).to_bytes(8, "little") + b"{}      ")
    np.savez(capture / "inputs.npz", in0=a, in1=b)
    cb = dict(
        abi_version="0.1",
        target="neutral",
        tensors=dict(
            left=dict(shape=[2, 3], dtype="i8", role="input"),
            right=dict(shape=[3, 2], dtype="i8", role="input"),
            result=dict(shape=[2, 2], dtype="i32", role="output"),
        ),
        commands=[],
        compiler_pointer_abi=dict(version=1, arguments=["left", "right", "result"]),
    )
    return root, capture, cb, a, b


def test_complete_typed_positional_storage_does_not_change_candidate(frozen):
    root, capture, cb, a, b = frozen
    original = copy.deepcopy(cb)
    values, record = P.bind_pointer_source(cb, root, capture)
    assert values == {
        "left": dict(shape=[2, 3], values=a.reshape(-1).tolist()),
        "right": dict(shape=[3, 2], values=b.reshape(-1).tolist()),
    }
    assert [(x["source_name"], x["compiled_name"]) for x in record["inputs"]] == [("I0", "left"), ("I1", "right")]
    assert record["returns"][0]["source_name"] == "answer"
    assert record["returns"][0]["compiled_name"] == "result"
    assert record["protocol"] == P.PROTOCOL
    assert cb == original and "kernel_abi" not in cb


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cb: cb["compiler_pointer_abi"].update(version=True),
        lambda cb: cb["compiler_pointer_abi"].update(arguments=["right", "left", "result"]),
        lambda cb: cb["compiler_pointer_abi"].update(arguments=["left", "left", "result"]),
        lambda cb: cb["compiler_pointer_abi"].update(extra="unreviewed"),
        lambda cb: cb.update(kernel_abi={"kind": "whole_program"}),
        lambda cb: cb.update(params={"storage_encodings": {}}),
        lambda cb: cb["tensors"]["left"].update(role="weight"),
        lambda cb: cb["tensors"]["left"].update(layout="blocked"),
        lambda cb: cb["tensors"]["left"].update(dtype="u8"),
        lambda cb: cb["tensors"]["result"].update(dtype="i8"),
        lambda cb: cb["tensors"]["result"].update(shape=[4, 1]),
        lambda cb: cb["tensors"].update(scratch={"role": "scratch", "dtype": "i32", "shape": [1]}),
    ],
)
def test_unknown_or_changed_pointer_storage_refuses(frozen, mutation):
    root, capture, cb, *_ = frozen
    mutation(cb)
    with pytest.raises(ValueError):
        P.bind_pointer_source(cb, root, capture)


def test_complete_independent_inputs_are_not_replaced_by_captured_values(frozen):
    root, capture, cb, a, b = frozen
    a[0, 1] += 1
    np.savez(capture / "inputs.npz", in0=a, in1=b)
    with pytest.raises(ValueError, match="independent values"):
        P.bind_pointer_source(cb, root, capture)


def test_same_shape_different_runtime_source_refuses(frozen):
    root, capture, cb, *_ = frozen
    (capture / "model.mlir").write_text(SOURCE.replace("arith.addi", "arith.subi"))
    with pytest.raises(ValueError, match="interfaces differ"):
        P.bind_pointer_source(cb, root, capture)


@pytest.mark.parametrize(
    "field,value", [("arg_order", ["I1", "I0", "answer"]), ("interface", "named"), ("linalg_mlir", "other.mlir")]
)
def test_original_reference_provenance_is_required(frozen, field, value):
    root, capture, cb, *_ = frozen
    p = root / "golden.yaml"
    golden = yaml.safe_load(p.read_text())
    golden["oracle_provenance"][field] = value
    p.write_text(yaml.safe_dump(golden))
    with pytest.raises(ValueError, match="provenance"):
        P.bind_pointer_source(cb, root, capture)


def test_unknown_constant_ownership_refuses(frozen):
    root, capture, cb, *_ = frozen
    (capture / "weights.safetensors.manifest.json").write_text(json.dumps({"weight": {"shape": [3, 2], "dtype": "i8"}}))
    with pytest.raises(ValueError, match="constant-weight ownership"):
        P.bind_pointer_source(cb, root, capture)


def test_payload_bound_precedes_golden_or_archive_loading(frozen, monkeypatch):
    root, capture, cb, *_ = frozen
    with (root / "rogue").open("wb") as stream:
        stream.truncate(P.MAX_PAYLOAD_BYTES + 1)
    monkeypatch.setattr(
        "merlin.targetgen.golden_store.load_golden", lambda *_: pytest.fail("must refuse before reading reference")
    )
    with pytest.raises(ValueError, match="payload bound"):
        P.bind_pointer_source(cb, root, capture)


def test_source_changed_while_loading_archive_refuses(frozen, monkeypatch):
    root, capture, cb, *_ = frozen
    load = np.load

    def mutate(*args, **kwargs):
        result = load(*args, **kwargs)
        (root / "source.mlir").write_text(SOURCE + "\n")
        return result

    monkeypatch.setattr(np, "load", mutate)
    with pytest.raises(ValueError, match="namespace changed"):
        P.bind_pointer_source(cb, root, capture)


def test_symlink_and_extra_archive_member_refuse(frozen):
    root, capture, cb, a, b = frozen
    (root / "indirect").symlink_to(capture / "model.mlir")
    with pytest.raises(ValueError, match="canonical"):
        P.bind_pointer_source(cb, root, capture)
    (root / "indirect").unlink()
    np.savez(capture / "inputs.npz", in0=a, in1=b, unowned=a)
    with pytest.raises(ValueError, match="input roster"):
        P.bind_pointer_source(cb, root, capture)


def test_compressed_array_allocation_is_bounded_before_reference_load(frozen, monkeypatch):
    root, capture, cb, *_ = frozen
    np.savez_compressed(capture / "inputs.npz", in0=np.zeros((P.MAX_ELEMENTS + 1,), dtype=np.int8))
    monkeypatch.setattr(
        "merlin.targetgen.golden_store.load_golden",
        lambda *_: pytest.fail("must refuse before reference/array loading"),
    )
    with pytest.raises(ValueError, match="decoded array shape/storage bound"):
        P.bind_pointer_source(cb, root, capture)


def test_object_archive_refuses_without_unpickling(frozen):
    root, capture, cb, *_ = frozen
    np.savez(capture / "inputs.npz", in0=np.asarray([{"unowned": 1}], dtype=object))
    with pytest.raises(ValueError, match="decoded array shape/storage bound"):
        P.bind_pointer_source(cb, root, capture)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["1"].update(name="left"),
        lambda manifest: manifest["0"].update(kind="weight"),
        lambda manifest: manifest.update({"2": {"kind": "input", "name": "unowned"}}),
    ],
)
def test_captured_manifest_must_bind_every_original_input(frozen, mutation):
    root, capture, cb, *_ = frozen
    path = capture / "weights.safetensors.manifest.json"
    manifest = json.loads(path.read_bytes())
    mutation(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="constant-weight ownership"):
        P.bind_pointer_source(cb, root, capture)


def test_unindexed_constant_bytes_cannot_hide_behind_input_only_manifest(frozen):
    root, capture, cb, *_ = frozen
    path = capture / "weights.safetensors"
    path.write_bytes(path.read_bytes() + b"unowned tensor")
    with pytest.raises(ValueError, match="unbound constant"):
        P.bind_pointer_source(cb, root, capture)
