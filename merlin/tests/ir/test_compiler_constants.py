"""Source-constant byte/ABI protocol fixtures, not claimed numeric captures."""

import hashlib
import json
from pathlib import Path
import struct

import pytest

from merlin.frontends.compile_inputs import stage_compile_inputs


def encoded(value):
    return json.dumps(value).encode()


def record(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(root, change=None):
    root.mkdir()
    source = b'''builtin.module attributes {prov.weights_file="original/weights.safetensors"} {
      func.func @forward(%literal:tensor<2xbf16>,%x:tensor<2xbf16>)->tensor<2xbf16> {
        return %x:tensor<2xbf16>
      }
    }'''
    payload = b"\x81\x7f\x00\x80"
    key = "__m2m_lifted__/literal"
    row = {"argument": 0, "name": "c_literal", "target": "literal", "weight": key,
        "dtype": "bfloat16", "shape": [2], "sha256": record(payload)["sha256"]}
    manifest = {"0": {"kind": "constant", "name": row["name"], "source_target": row["target"],
        **{k: row[k] for k in ("weight", "dtype", "shape", "sha256")}}, "1": {"kind": "input", "name": "x"}}
    nodes = [{"op": "placeholder", "graph_id": "g:prepared:root", "target": row["name"], "input_target": row["target"],
        "input_kind": "CONSTANT_TENSOR", "results": [{"kind": "tensor", "dtype": "bfloat16", "shape": [2]}]},
        {"op": "placeholder", "graph_id": "g:prepared:root", "target": "x", "input_target": None,
         "input_kind": "USER_INPUT", "results": [{"kind": "tensor", "dtype": "bfloat16", "shape": [2]}]}]
    lifted = {row["name"]: {"shape": [2], "dtype": "float32", "sha256": "legacy-constant-value-digest"}}
    profile = {"schema": "m2m.compiler-constants.v1", "storage": "weights.safetensors", "bindings": [row]}
    if change == "target": row["target"] = manifest["0"]["source_target"] = "different"
    elif change == "name":
        lifted["c_other"] = lifted.pop("c_literal")
        row["name"] = manifest["0"]["name"] = "c_other"
    elif change == "dtype": row["dtype"] = manifest["0"]["dtype"] = "float16"
    elif change == "shape":
        row["shape"] = manifest["0"]["shape"] = nodes[0]["results"][0]["shape"] = [1, 2]
        lifted["c_literal"]["shape"] = [1, 2]
    elif change == "payload": payload = b"\x83\x7f\x00\x80"
    elif change == "digest": row["sha256"] = manifest["0"]["sha256"] = "0" * 64
    elif change == "source-role": nodes[0]["input_kind"] = "USER_INPUT"
    elif change == "source-index": nodes.reverse()
    elif change == "abi": source = source.replace(b"%literal:tensor<2xbf16>", b"%literal:tensor<2xf32>")
    elif change == "duplicate": profile["bindings"].append(dict(row))
    elif change == "schema": profile["schema"] = "unknown"
    elif change == "missing-binding": profile["bindings"] = []
    elif change == "old": profile = None
    header = encoded({key: {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}})
    weights = struct.pack("<Q", len(header)) + header + payload
    trace = {"status": "complete", "mlir": record(source),
        "graphs": {"prepared": {"graph_id": "g:prepared:root", "nodes": nodes}}}
    files = {"model.mlir": source, "weights.safetensors": weights, "weights.safetensors.manifest.json": encoded(manifest),
        "frontend-trace.json": encoded(trace), "meta.json": encoded({"ok": True, "opaque": 0, "weights": "original/weights.safetensors",
            "func_name": "forward", "frontend_trace": record(encoded(trace)), "input_abi": [{"shape": [2], "dtype": "bf16"}],
            "output_abi": [{"shape": [2], "dtype": "bf16"}]}),
        **{name: b"EVALUATOR " + name.encode() for name in ("inputs.npz", "extra.npz", "golden.npy", "tensor_words.npz")}}
    receipt = {"schema": "m2m.capture-receipt.v1", "materialized_abi": {"complete": True, "inputs": 1},
        "lifted_constants": lifted, "artifacts": {name: record(raw) for name, raw in files.items()}}
    if profile is not None:
        receipt["compiler_constants"] = profile
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    (root / "capture_receipt.json").write_bytes(encoded(receipt))
    return receipt


def test_staging_only_reads_exact_compiler_constant_storage(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "compiler"
    receipt = fixture(source)
    read, opened = Path.read_bytes, Path.open
    def check(path):
        if path.name in ("inputs.npz", "extra.npz", "golden.npy", "tensor_words.npz"):
            raise AssertionError("compiler accessed evaluator data")
    def guarded_read(path):
        check(path)
        return read(path)
    def guarded_open(path, *args, **kwargs):
        check(path)
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    monkeypatch.setattr(Path, "open", guarded_open)
    staged = stage_compile_inputs(source, target)
    assert "compiler-constants.json" in staged["members"]
    projection = json.loads((target / "compiler-constants.json").read_text())
    assert projection["bindings"] == receipt["compiler_constants"]["bindings"]
    data = b"".join((target / name).read_bytes() for name in staged["members"])
    assert b"EVALUATOR" not in data
    for name in ("inputs.npz", "extra.npz", "golden.npy", "tensor_words.npz"):
        assert receipt["artifacts"][name]["sha256"].encode() not in data


@pytest.mark.parametrize("change", ["target", "name", "dtype", "shape", "payload", "digest", "source-role",
    "source-index", "abi", "duplicate", "schema", "missing-binding", "old"])
def test_rehashed_profiles_still_need_actual_source_types_and_bytes(tmp_path, change):
    source, target = tmp_path / "source", tmp_path / "compiler"
    fixture(source, change)
    with pytest.raises(ValueError):
        stage_compile_inputs(source, target)
    assert not target.exists()
