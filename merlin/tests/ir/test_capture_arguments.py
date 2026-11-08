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


@pytest.mark.parametrize("dtype,width", [("float32", 4), ("bfloat16", 2)])
@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("negative_zero", [False, True])
def test_real_archived_lifted_constant_preserves_source_archive(
        tmp_path, monkeypatch, dtype, width, depth, negative_zero):
    torch = pytest.importorskip("torch")
    pytest.importorskip("m2m")
    from m2m.capture.bundle import write_bundle

    class Literal(torch.nn.Module):
        def forward(self, x):
            return x + torch.tensor(-0.0 if negative_zero else 0.0, dtype=getattr(torch, dtype))

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    capture, stage = tmp_path / "capture", tmp_path / "stage"
    model = Literal()
    for _ in range(depth):
        model = Wrapper(model)
    write_bundle(model, (torch.ones(2, dtype=getattr(torch, dtype)),), capture,
                 capture_regions=False, capture_trace=True)
    members = ("weights.safetensors", "weights.safetensors.manifest.json", "model.mlir",
               "capture_receipt.json", "extra.npz", "inputs.npz", "golden.npy")
    before = {name: (capture / name).read_bytes() for name in members}
    original_read = Path.read_bytes
    reads = []

    def no_evaluator_or_extra(path):
        assert path.name not in {"inputs.npz", "golden.npy"}
        reads.append(path.name)
        return original_read(path)

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", no_evaluator_or_extra)
        result = stage_compile_inputs(capture, stage)
        loaded = load_compile_inputs(stage)
        constant, = [row for row in loaded.argument_bindings if row["kind"] == "constant"]
        assert constant == {"kind": "constant", "weight": "model." * depth + "lifted_tensor_0",
                            "name": "c_" + "model_" * depth + "lifted_tensor_0", "shape": [],
                            "dtype": "f32" if dtype == "float32" else "bf16"}
        payloads = load_lifted_argument_bytes(stage, loaded, kinds=("constant",), max_bytes=width)
        expected = bytes(width - 1) + (b"\x80" if negative_zero else b"\0")
        assert list(payloads.values()) == [expected]
        if dtype == "bfloat16" and not negative_zero:
            assert _sha(expected) == "96a296d224f285c67bee93c30f8a309157f0daa35dc5b87e410b78630a09cfc7"
        assert "extra.npz" not in reads
    assert (stage / "weights.safetensors").read_bytes() == before["weights.safetensors"]
    assert (stage / "weights.safetensors.manifest.json").read_bytes() == before["weights.safetensors.manifest.json"]
    assert {name: (capture / name).read_bytes() for name in members} == before
    assert "argument_materialization" not in result["source"]
    assert not (stage / CONSTANT_FILE).exists()


def _archived_capture(root, *, dtype="f32", scalar=None, encoding="F32", source_dtype="float32"):
    """Mix an archive-backed scalar with a legacy extra-only buffer initializer."""
    scalar = np.array(-0.0, dtype="float32") if scalar is None else scalar
    _capture(root, dtype=dtype, scalar=scalar)
    raw = (root / "weights.safetensors").read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + size])
    header["lifted_tensor_0"] = {"dtype": encoding, "shape": [],
                                 "data_offsets": [8, 8 + scalar.nbytes]}
    encoded = json.dumps(header).encode()
    _replace(root, "weights.safetensors", struct.pack("<Q", len(encoded)) + encoded
             + raw[8 + size:] + scalar.tobytes())
    manifest = json.loads((root / "weights.safetensors.manifest.json").read_text())
    manifest["1"] = {"kind": "constant", "name": "c_lifted_tensor_0",
                     "weight": "lifted_tensor_0", "dtype": source_dtype, "shape": []}
    _replace(root, "weights.safetensors.manifest.json", json.dumps(manifest).encode())


@pytest.mark.parametrize("legacy_prefix", [False, True])
def test_receipt_cannot_promote_a_real_named_runtime_input(tmp_path, legacy_prefix):
    torch = pytest.importorskip("torch")
    pytest.importorskip("m2m")
    from m2m.capture.bundle import write_bundle

    class NestedName(torch.nn.Module):
        def forward(self, c_model_model_lifted_tensor_input):
            return c_model_model_lifted_tensor_input

    class LegacyName(torch.nn.Module):
        def forward(self, c_lifted_tensor_input):
            return c_lifted_tensor_input

    capture, stage = tmp_path / "capture", tmp_path / "stage"
    value = torch.tensor([-0.0, 1.25])
    write_bundle(LegacyName() if legacy_prefix else NestedName(), (value,), capture,
                 capture_regions=False, capture_trace=True)
    name, = json.loads((capture / "input_order.json").read_text())
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["lifted_constants"] == {}
    receipt["lifted_constants"] = {name: {
        "shape": [2], "dtype": "float32", "sha256": _sha(value.numpy().tobytes())}}
    receipt["materialized_abi"]["lifted_constants"] = [name]
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="undeclared constant roles|runtime input positions"):
        stage_compile_inputs(capture, stage)
    assert not stage.exists()


def test_staged_archived_constant_rejects_invalid_placeholder_even_after_resigning(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _archived_capture(capture)
    stage_compile_inputs(capture, stage)
    path = stage / "weights.safetensors.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["1"]["name"] = "../outside"
    path.write_text(json.dumps(manifest))
    record_path = stage / "compile-inputs.json"
    record = json.loads(record_path.read_text())
    record["members"][path.name] = _sha(path.read_bytes())
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="shape or dtype"):
        load_compile_inputs(stage)


def test_archived_constant_and_legacy_buffer_share_checked_signature(tmp_path, monkeypatch):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _archived_capture(capture)
    original = (capture / "weights.safetensors").read_bytes()
    opened, zip_open = [], zipfile.ZipFile.open

    def checked_open(archive, member, *args, **kwargs):
        opened.append(member)
        return zip_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", checked_open)
    result = stage_compile_inputs(capture, stage)
    loaded = load_compile_inputs(stage)
    assert loaded.argument_bindings[1] == {
        "kind": "constant", "weight": "lifted_tensor_0", "name": "c_lifted_tensor_0",
        "shape": [], "dtype": "f32", "file": "weights.safetensors"}
    assert loaded.argument_bindings[2]["kind"] == "buffer"
    assert loaded.argument_bindings[2]["file"] == CONSTANT_FILE
    assert opened == ["buf::state.value.npy"]
    assert result["source"]["argument_materialization"]["entries"][0]["role"] == "buffer"
    assert load_lifted_argument_bytes(stage, loaded, kinds=("constant",), max_bytes=4) == {
        1: struct.pack("<I", 0x80000000)}
    assert (stage / "weights.safetensors").read_bytes() == original


@pytest.mark.parametrize("dtype,scalar,encoding,source_dtype", [
    ("i8", np.array(255, dtype="uint8"), "U8", "uint8"),
    ("i32", np.array(-127, dtype="int32"), "I32", "int32"),
])
def test_archived_integer_codes_and_exponents_keep_raw_bytes(tmp_path, dtype, scalar, encoding, source_dtype):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _archived_capture(capture, dtype=dtype, scalar=scalar, encoding=encoding, source_dtype=source_dtype)
    original = (capture / "weights.safetensors").read_bytes()
    stage_compile_inputs(capture, stage)
    loaded = load_compile_inputs(stage)
    assert loaded.argument_bindings[1]["dtype"] == dtype
    assert loaded.argument_bindings[1]["kind"] == "constant"
    assert load_lifted_argument_bytes(stage, loaded, kinds=("constant",), max_bytes=scalar.nbytes) == {
        1: scalar.tobytes()}
    assert (stage / "weights.safetensors").read_bytes() == original


def test_archived_and_extra_only_lifted_constants_coexist(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _archived_capture(capture)
    value = np.array([1, -2], dtype="float32")
    manifest = json.loads((capture / "weights.safetensors.manifest.json").read_text())
    manifest["2"] = {"kind": "constant", "name": "c_lifted_tensor_1"}
    _replace(capture, "weights.safetensors.manifest.json", json.dumps(manifest).encode())
    extra = BytesIO()
    np.savez(extra, c_lifted_tensor_0=np.array(-0.0, dtype="float32"), c_lifted_tensor_1=value)
    _replace(capture, "extra.npz", extra.getvalue())
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["lifted_constants"]["c_lifted_tensor_1"] = {
        "shape": [2], "dtype": "float32", "sha256": _sha(value.tobytes())}
    receipt["materialized_abi"]["lifted_constants"].append("c_lifted_tensor_1")
    receipt_path.write_text(json.dumps(receipt))
    original = (capture / "weights.safetensors").read_bytes()
    stage_compile_inputs(capture, stage)
    loaded = load_compile_inputs(stage)
    assert loaded.argument_bindings[1]["file"] == "weights.safetensors"
    assert loaded.argument_bindings[2]["file"] == CONSTANT_FILE
    assert loaded.argument_bindings[1]["kind"] == loaded.argument_bindings[2]["kind"] == "constant"
    assert load_lifted_argument_bytes(stage, loaded, kinds=("constant",), max_bytes=12) == {
        1: struct.pack("<I", 0x80000000), 2: value.tobytes()}
    assert (stage / "weights.safetensors").read_bytes() == original


@pytest.mark.parametrize("mutation,expected", [
    ("unknown_field", "unknown manifest fields"),
    ("missing_tensor", "safetensors shape or dtype"),
    ("wrong_dtype", "capture declaration"),
    ("wrong_shape", "capture declaration"),
    ("unknown_placeholder", "unbound or duplicate"),
    ("duplicate_placeholder", "unbound or duplicate"),
    ("header_dtype", "safetensors shape or dtype"),
    ("unbound_tensor", "stored tensor unbound"),
])
def test_invalid_archive_backed_constants_refuse_before_publication(tmp_path, mutation, expected):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _archived_capture(capture)
    path = capture / "weights.safetensors.manifest.json"
    rows = json.loads(path.read_text())
    if mutation == "unknown_field":
        rows["1"]["unknown"] = True
    elif mutation == "missing_tensor":
        rows["1"]["weight"] = "missing"
    elif mutation == "wrong_dtype":
        rows["1"]["dtype"] = "int32"
    elif mutation == "wrong_shape":
        rows["1"]["shape"] = [1]
    elif mutation == "unknown_placeholder":
        rows["1"]["name"] = "c_lifted_tensor_missing"
    elif mutation == "duplicate_placeholder":
        rows["2"] = dict(rows["1"])
    else:
        raw = (capture / "weights.safetensors").read_bytes()
        size = int.from_bytes(raw[:8], "little")
        header, payload = json.loads(raw[8:8 + size]), raw[8 + size:]
        if mutation == "header_dtype":
            header["lifted_tensor_0"]["dtype"] = "I32"
        else:
            header["unbound"] = {"dtype": "U8", "shape": [1], "data_offsets": [12, 13]}
            payload += b"\x01"
        encoded = json.dumps(header).encode()
        _replace(capture, "weights.safetensors", struct.pack("<Q", len(encoded)) + encoded + payload)
    _replace(capture, path.name, json.dumps(rows).encode())
    with pytest.raises(ValueError, match=expected):
        stage_compile_inputs(capture, stage)
    assert not stage.exists()


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


def test_producer_declared_readonly_buffer_is_an_immutable_constant(tmp_path):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    meta = json.loads((capture / "meta.json").read_text())
    meta["read_only_buffer_arguments"] = [2]
    _replace(capture, "meta.json", json.dumps(meta).encode())
    stage_compile_inputs(capture, stage)
    loaded = load_compile_inputs(stage)
    assert loaded.argument_bindings[2]["kind"] == "constant"
    assert load_lifted_argument_bytes(stage, loaded)[2] == struct.pack("<2f", 1, -2)


@pytest.mark.parametrize("indices", [[3], [99], [2, 2], [True], "2"])
def test_readonly_buffer_declaration_must_name_unique_buffer_arguments(tmp_path, indices):
    capture, stage = tmp_path / "capture", tmp_path / "stage"
    _capture(capture)
    meta = json.loads((capture / "meta.json").read_text())
    meta["read_only_buffer_arguments"] = indices
    _replace(capture, "meta.json", json.dumps(meta).encode())
    with pytest.raises(ValueError, match="read-only buffer declaration"):
        stage_compile_inputs(capture, stage)
    assert not stage.exists()


def test_logical_uint8_source_codes_use_signless_i8_without_conversion():
    from merlin.frontends.capture_arguments import _encode

    codes = np.array([0x80, 0xfe, 0x7e, 0xb8], dtype="uint8")
    dtype, encoding, payload, codec = _encode(codes, {"shape": [4], "dtype": "i8"}, _WEIGHT_DTYPES)
    assert (dtype, encoding, payload) == ("uint8", "U8", codes.tobytes())
    assert codec == "byte-preserving-little-endian-storage"
