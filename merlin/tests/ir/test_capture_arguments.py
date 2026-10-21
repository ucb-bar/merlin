"""Capture constants enter compilation; mutable buffers retain their state role."""

import hashlib
import json
import struct
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from merlin.frontends.capture_arguments import CONSTANT_FILE, materialize_arguments
from merlin.frontends.compile_inputs import (
    _WEIGHT_DTYPES,
    load_compile_inputs,
    load_lifted_argument_bytes,
    stage_compile_inputs,
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _capture(root, *, dtype="f32", scalar=None, buffer=None, extra_members=None):
    root.mkdir()
    scalar = np.array(-0.0, dtype="float32") if scalar is None else scalar
    buffer = np.array([1, -2], dtype="float32") if buffer is None else buffer
    source = f'''builtin.module attributes {{prov.weights_file = "old.safetensors"}} {{
      func.func @forward(%w: tensor<2xf32>, %c: tensor<{dtype}>,
                         %b: tensor<2xf32>, %x: tensor<2xf32>) -> tensor<2xf32> {{
        func.return %x : tensor<2xf32>
      }}
    }}'''.encode()
    manifest = {
        "0": {"kind": "param", "weight": "linear.weight", "dtype": "float32", "shape": [2]},
        "1": {"kind": "input", "name": "c_lifted_tensor_0"},
        "2": {"kind": "buffer", "name": "b_state_value"},
        "3": {"kind": "input", "name": "x"},
    }
    header = json.dumps({"linear.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    trace = json.dumps({"status": "complete", "mlir": {"sha256": _sha(source)}}).encode()
    meta = {"ok": True, "opaque": 0, "weights": "old.safetensors", "func_name": "forward",
            "input_abi": [{"shape": [2], "dtype": "f32"}], "output_abi": [{"shape": [2], "dtype": "f32"}],
            "dtype": "f32", "frontend_trace": {"sha256": _sha(trace)}}
    stream = BytesIO()
    np.savez(stream, **{"c_lifted_tensor_0": scalar, "buf::state.value": buffer,
                       "unrelated_private_tensor": np.array([123]), **(extra_members or {})})
    members = {"model.mlir": source, "weights.safetensors": struct.pack("<Q", len(header)) + header + bytes(8),
               "weights.safetensors.manifest.json": json.dumps(manifest).encode(),
               "input_order.json": b'{"x": 0}', "extra.npz": stream.getvalue(),
               "frontend-trace.json": trace, "meta.json": json.dumps(meta).encode(),
               "inputs.npz": b"private runtime data", "golden.npy": b"private reference data"}
    for name, raw in members.items():
        (root / name).write_bytes(raw)
    receipt = {"schema": "m2m.capture-receipt.v1",
               "materialized_abi": {"complete": True, "inputs": 1, "lifted_constants": ["c_lifted_tensor_0"]},
               "lifted_constants": {"c_lifted_tensor_0": {"shape": list(scalar.shape), "dtype": str(scalar.dtype),
                                                          "sha256": _sha(scalar.tobytes())}},
               "artifacts": {name: {"bytes": len(raw), "sha256": _sha(raw)} for name, raw in members.items()}}
    (root / "capture_receipt.json").write_text(json.dumps(receipt))


def _replace(root, name, raw):
    (root / name).write_bytes(raw)
    path = root / "capture_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["artifacts"][name] = {"bytes": len(raw), "sha256": _sha(raw)}
    path.write_text(json.dumps(receipt))


def test_declared_leaves_only_and_buffer_remains_state(tmp_path, monkeypatch):
    from merlin.compile.application_state import StateDeclaration
    from merlin.compile.c_parameters import freeze_c_arguments
    from merlin.semantic_compiler.linalg_bridge import translate_linalg_text

    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    reads, opens = [], []
    read_bytes, zip_open = Path.read_bytes, zipfile.ZipFile.open

    def checked_read(path):
        assert path.name not in {"inputs.npz", "golden.npy"}
        reads.append(path.name)
        return read_bytes(path)

    def checked_open(archive, member, *args, **kwargs):
        assert member != "unrelated_private_tensor.npy"
        opens.append(member)
        return zip_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", checked_read)
    monkeypatch.setattr(zipfile.ZipFile, "open", checked_open)
    result = stage_compile_inputs(capture, stage)
    staged = load_compile_inputs(stage)
    payloads = load_lifted_argument_bytes(stage, staged)
    assert [row["kind"] for row in staged.argument_bindings] == ["param", "constant", "buffer", "input"]
    assert staged.argument_bindings[1]["shape"] == []
    assert staged.argument_bindings[3]["input_index"] == 0
    assert payloads == {0: bytes(8), 1: struct.pack("<I", 0x80000000), 2: struct.pack("<2f", 1, -2)}
    assert opens == ["c_lifted_tensor_0.npy", "buf::state.value.npy"]
    assert not {"extra.npz", "golden.npy", "inputs.npz"} & {p.name for p in stage.iterdir()}
    assert (capture / "weights.safetensors").read_bytes() == (stage / "weights.safetensors").read_bytes()
    assert result["source"]["argument_materialization"]["entries"][1]["role"] == "buffer"
    state = StateDeclaration.from_record({"schema": "merlin.application_state.v1",
                                         "source_program_sha256": staged.program_sha256,
                                         "ports": [{"argument_index": 2, "name": "b_state_value",
                                                    "mode": "read_only", "output_index": None,
                                                    "initialization": "captured"}]}, staged)
    assert state.ports[0].argument_index == 2
    request = translate_linalg_text(
        staged.program, entry=staged.entry, target_identity="synthetic",
        argument_constants={0: payloads[0], 1: payloads[1]},
    ).request
    assert request.node("arg1").op == "constant" and request.node("arg2").op == "input"
    assert {key for key, _ in request.input_storages} == {"arg2", "arg3"}
    with pytest.raises(ValueError, match="state interface"):
        freeze_c_arguments(stage, name="model", source_sha256=staged.program_sha256)


@pytest.mark.parametrize("case, expected", [
    ("buffer_collision", "unique initializer"), ("wrong_dtype", "dtype differs"),
    ("wrong_shape", "shape or numeric storage"), ("hash", "capture receipt"),
    ("runtime_constant", "runtime input positions"), ("zip", "valid NPZ"),
])
def test_rejects_invalid_materialization_atomically(tmp_path, case, expected):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    options = {"buffer_collision": {"extra_members": {"buf::state_value": np.ones(2, dtype="float32")}},
               "wrong_dtype": {"buffer": np.ones(2, dtype="float64")},
               "wrong_shape": {"buffer": np.ones(3, dtype="float32")}}
    _capture(capture, **options.get(case, {}))
    if case == "hash":
        path = capture / "capture_receipt.json"
        receipt = json.loads(path.read_text())
        receipt["lifted_constants"]["c_lifted_tensor_0"]["sha256"] = "0" * 64
        path.write_text(json.dumps(receipt))
    if case == "runtime_constant":
        _replace(capture, "input_order.json", b'{"c_lifted_tensor_0": 0}')
    if case == "zip":
        _replace(capture, "extra.npz", b"not a zip")
    with pytest.raises(ValueError, match=expected):
        stage_compile_inputs(capture, stage)
    assert not stage.exists()


def test_constant_archive_checked_again_after_load(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    stage_compile_inputs(capture, stage)
    staged = load_compile_inputs(stage)
    path = stage / CONSTANT_FILE
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="changed after"):
        load_lifted_argument_bytes(stage, staged)
    with pytest.raises(ValueError, match="differs from manifest"):
        load_compile_inputs(stage)


def test_streamed_archives_select_roles_and_check_budget_without_whole_file_reads(tmp_path, monkeypatch):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    read_bytes = Path.read_bytes

    def no_archive_copy(path):
        assert path.name not in {"weights.safetensors", CONSTANT_FILE, "inputs.npz", "golden.npy"}
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", no_archive_copy)
    stage_compile_inputs(capture, stage)
    staged = load_compile_inputs(stage)
    assert [row["kind"] for row in staged.argument_bindings] == ["param", "constant", "buffer", "input"]
    all_values = {0: bytes(8), 1: struct.pack("<I", 0x80000000), 2: struct.pack("<2f", 1, -2)}
    for kinds, indices, budget in (((), (), 0), (("constant",), (1,), 4), (("buffer",), (2,), 8),
                                   (("param",), (0,), 8), (("param", "constant"), (0, 1), 12)):
        assert load_lifted_argument_bytes(stage, staged, kinds=kinds, max_bytes=budget) == {
            index: all_values[index] for index in indices
        }
    with pytest.raises(ValueError, match="byte budget") as failure:
        load_lifted_argument_bytes(stage, staged, kinds=("constant",), max_bytes=3)
    assert failure.value.status == "resource_limit"
    for kinds in (["constant"], ("constant", "constant"), ("input",), ([],)):
        with pytest.raises(ValueError, match="declared roles"):
            load_lifted_argument_bytes(stage, staged, kinds=kinds)
    for budget in (True, -1, 1.5):
        with pytest.raises(ValueError, match="byte budget"):
            load_lifted_argument_bytes(stage, staged, max_bytes=budget)


def test_selecting_constants_still_rejects_changed_unselected_weights(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    stage_compile_inputs(capture, stage)
    staged = load_compile_inputs(stage)
    with (stage / "weights.safetensors").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    with pytest.raises(ValueError, match="staged weights changed"):
        load_lifted_argument_bytes(stage, staged, kinds=("constant",), max_bytes=4)


def test_streamed_staging_refuses_weights_changed_during_copy(tmp_path, monkeypatch):
    import merlin.frontends.compile_inputs as compiler_inputs

    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    copy = compiler_inputs._copy_archive

    def changed(source, destination, info):
        with source.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x01")
        copy(source, destination, info)

    monkeypatch.setattr(compiler_inputs, "_copy_archive", changed)
    with pytest.raises(ValueError, match="changed during staging"):
        stage_compile_inputs(capture, stage)
    assert not stage.exists()


def test_capture_without_entry_name_uses_the_only_parsed_function(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    meta = json.loads((capture / "meta.json").read_text())
    del meta["func_name"]
    _replace(capture, "meta.json", json.dumps(meta).encode())
    stage_compile_inputs(capture, stage)
    assert load_compile_inputs(stage).entry == "forward"


def test_capture_without_entry_name_does_not_choose_between_functions(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    source = (capture / "model.mlir").read_text().rsplit("}", 1)[0]
    source += "func.func private @helper() { func.return } }"
    _replace(capture, "model.mlir", source.encode())
    trace = {"status": "complete", "mlir": {"sha256": _sha(source.encode())}}
    trace_raw = json.dumps(trace).encode()
    _replace(capture, "frontend-trace.json", trace_raw)
    meta = json.loads((capture / "meta.json").read_text())
    del meta["func_name"]
    meta["frontend_trace"]["sha256"] = _sha(trace_raw)
    _replace(capture, "meta.json", json.dumps(meta).encode())
    with pytest.raises(ValueError, match="multiple functions"):
        stage_compile_inputs(capture, stage)
    assert not stage.exists()


@pytest.mark.parametrize("mutation", ["path", "downgrade", "constant_role"])
def test_resigned_invalid_signature_refuses(tmp_path, mutation):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    stage_compile_inputs(capture, stage)
    signature = json.loads((stage / "signature.json").read_text())
    rows = json.loads((stage / "weights.safetensors.manifest.json").read_text())
    if mutation == "path":
        rows["1"]["file"] = "../extra.npz"
    elif mutation == "constant_role":
        rows["1"]["kind"] = "param"
    else:
        signature["schema"] = "merlin.compile-signature.v2"
    (stage / "signature.json").write_text(json.dumps(signature))
    (stage / "weights.safetensors.manifest.json").write_text(json.dumps(rows))
    path = stage / "compile-inputs.json"
    record = json.loads(path.read_text())
    for name in ("signature.json", "weights.safetensors.manifest.json"):
        record["members"][name] = _sha((stage / name).read_bytes())
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="schema|shape or dtype"):
        load_compile_inputs(stage)


def test_bf16_storage_all_non_nan_encodings():
    from merlin.frontends.capture_arguments import _encode

    words = np.arange(65536, dtype="<u4")
    words = words[(words & 0x7FFF) <= 0x7F80]
    stored = (words << 16).view("<f4")
    _, _, raw, codec = _encode(stored, {"dtype": "bf16", "shape": [len(stored)]}, _WEIGHT_DTYPES)
    assert codec == "exact-bf16-from-f32-storage"
    assert len(stored) == 65282 and raw == words.astype("<u2").tobytes()


@pytest.mark.parametrize("dtype, bits", [("bf16", 0x3F800001), ("bf16", 0x7FC00000), ("f64", 0x3F800000)])
def test_no_lossy_buffer_reconstruction(dtype, bits):
    from merlin.frontends.capture_arguments import _encode

    with pytest.raises(ValueError, match="lossy|dtype differs"):
        _encode(np.array([bits], dtype="<u4").view("<f4"), {"shape": [1], "dtype": dtype}, _WEIGHT_DTYPES)


def test_duplicate_zip_members_refuse():
    stream = BytesIO()
    member = BytesIO()
    np.save(member, np.array([1], dtype="float32"))
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("c_lifted_tensor_0.npy", member.getvalue())
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("c_lifted_tensor_0.npy", member.getvalue())
    with pytest.raises(ValueError, match="duplicate members"):
        materialize_arguments({"0": {"kind": "input", "name": "c_lifted_tensor_0"}},
                              [{"shape": [1], "dtype": "f32"}],
                              {"lifted_constants": {"c_lifted_tensor_0": {}},
                               "materialized_abi": {"lifted_constants": ["c_lifted_tensor_0"]}},
                              stream.getvalue(), _WEIGHT_DTYPES)
