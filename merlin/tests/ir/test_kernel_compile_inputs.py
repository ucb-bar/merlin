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


def guarded_fixture(root, change=None):
    """Protocol fixture only: staging does not prove executable guard semantics."""
    receipt = capture_fixture(root)
    spec = json.loads((root / "spec.json").read_text())
    model = (root / "model.mlir").read_bytes().replace(b") {", b") attributes {prov.memory_guard_bytes_per_result_element = 16 : i64} {")
    if change == "storage_attribute":
        model = model.replace(b"16 : i64", b"8 : i64")
    memory_input, memory_output = b"guarded upstream input", b"standard memory output"
    admission = {"schema": "t2m.memory-admission.v1", "strategy": "runtime-guarded-private-output-state", "spec": spec,
        "internal_arguments": ([{"kind": "extent", "buffer": b["name"]} for b in spec["buffers"]]
            + [{"kind": k, "buffer": b["name"]} for b in spec["buffers"] if b["effect"] != "read" for k in ("owners", "visited")]
            + [{"kind": "program"}]),
        "guarded_input_sha256": record(memory_input)["sha256"], "continuation_output_sha256": record(memory_output)["sha256"],
        "guard_storage_bytes_per_output_element": 16, "writable_reads": "refused",
        "fresh_initialization": "post-grid-complete-write-coverage",
        "accesses": [{"buffer": "out", "opcode": "tts.scatter", "effect": "write", "source_ids": ["normalized.0"], "tile": [8]}]}
    if change == "layout":
        admission["internal_arguments"].reverse()
    elif change == "source":
        admission["accesses"][0]["source_ids"] = ["unknown.0"]
    elif change == "effect":
        admission["accesses"][0]["buffer"] = "x"
    elif change == "coverage":
        admission["fresh_initialization"] = "unchecked"
    elif change == "storage":
        admission["guard_storage_bytes_per_output_element"] = 0
    files = {"model.mlir": model, "memory-input.mlir": memory_input, "memory-stage.mlir": memory_output,
        "memory-admission.json": encoded(admission), "memory-stage.json": encoded({"schema": "t2m.triton-shared-stage.v1",
            "stage": "memory-ir", "stage_completed": True, "artifacts": {
                "source.ttir": record(memory_input), "lowered.mlir": record(memory_output)}})}
    provenance = json.loads((root / "provenance.json").read_text())
    provenance.update(profile="shared-guarded-buffer", mlir_sha256=record(model)["sha256"], memory={
        "admission_sha256": record(files["memory-admission.json"])["sha256"],
        "continuation_input_sha256": record(memory_input)["sha256"], "continuation_output_sha256": record(memory_output)["sha256"]})
    files["provenance.json"] = encoded(provenance)
    for name, raw in files.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    receipt["profile"] = "shared-guarded-buffer"
    (root / "capture_receipt.json").write_bytes(encoded(receipt))


def test_guarded_memory_staging_retains_evidence_and_scratch_without_evaluator(tmp_path, monkeypatch):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    guarded_fixture(source)
    original_read = Path.read_bytes
    def read(path):
        if path.name in ("inputs.npz", "golden.npy", "extra.npz", "execution.json"):
            raise AssertionError("guarded memory staging accessed evaluator data")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    manifest = stage_compile_inputs(source, target)
    assert manifest["payload_profile"] == "shared-guarded-buffer"
    signature = json.loads((target / "signature.json").read_text())
    assert signature["guard_storage_bytes_per_result_element"] == 16
    assert (target / "source/memory-admission.json").read_bytes() == (source / "memory-admission.json").read_bytes()
    assert b"EVALUATOR" not in b"".join((target / name).read_bytes() for name in manifest["members"])


@pytest.mark.parametrize("change", ["layout", "source", "effect", "coverage", "storage", "storage_attribute"])
def test_rehashed_inconsistent_guarded_contract_never_publishes(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    guarded_fixture(source, change)
    with pytest.raises(ValueError, match="guarded"):
        stage_compile_inputs(source, target)
    assert not target.exists()


def mutable_fixture(root, grid, words, change=None):
    guarded_fixture(root)
    receipt = json.loads((root / "capture_receipt.json").read_text())
    spec = json.loads((root / "spec.json").read_text())
    spec["grid"] = list(grid)
    admission = json.loads((root / "memory-admission.json").read_text())
    admission.update(schema="t2m.memory-admission.v2", spec=spec, owner_words_per_element=words,
        guard_storage_bytes_per_output_element=8 * (words + 1), writable_reads="shared-readers-or-single-writer-program",
        ownership_protocol="sticky-complemented-program-fragments")
    # A read of a writable source is a declared effect here, not a staging proof.
    admission["accesses"].append({"buffer": "old", "opcode": "tts.gather", "effect": "read", "source_ids": ["normalized.0"], "tile": [8]})
    if change == "words":
        admission["owner_words_per_element"] += 1
    elif change == "protocol":
        admission["ownership_protocol"] = "unchecked"
    elif change == "scope":
        admission["writable_reads"] = "unchecked"
    stage = json.loads((root / "memory-stage.json").read_text())
    stage["stage"] = "memory-ir" if change == "stage" else "memory-effects"
    model = (root / "model.mlir").read_bytes().replace(b"16 : i64", str(8 * (words + 1)).encode() + b" : i64")
    provenance = json.loads((root / "provenance.json").read_text())
    provenance.update(spec=spec, mlir_sha256=record(model)["sha256"])
    provenance["memory"]["admission_sha256"] = record(encoded(admission))["sha256"]
    files = {"spec.json": encoded(spec), "model.mlir": model, "memory-admission.json": encoded(admission),
             "memory-stage.json": encoded(stage), "provenance.json": encoded(provenance)}
    for name, raw in files.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    (root / "capture_receipt.json").write_bytes(encoded(receipt))


@pytest.mark.parametrize("grid,words,cost", [((1,), 1, 16), ((65536, 65536), 2, 24),
    ((2**31 - 1, 2**31 - 1, 2), 3, 32)])
def test_mutable_profile_retains_grid_derived_scratch_cost(tmp_path, grid, words, cost):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    mutable_fixture(source, grid, words)
    stage_compile_inputs(source, target)
    assert json.loads((target / "signature.json").read_text())["guard_storage_bytes_per_result_element"] == cost


@pytest.mark.parametrize("change", ["words", "protocol", "scope", "stage"])
def test_rehashed_invalid_mutable_contract_never_publishes(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    mutable_fixture(source, (65536, 65536), 2, change)
    with pytest.raises(ValueError):
        stage_compile_inputs(source, target)
    assert not target.exists()


def word_fixture(root, change=None, *, isolated=False):
    """Source-policy byte custody only; this is not an execution oracle."""
    from xdsl.dialects.builtin import StringAttr
    from merlin.xdsl_dialects._common import text
    mutable_fixture(root, (1,), 1)
    receipt = json.loads((root / "capture_receipt.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    admission = json.loads((root / "memory-admission.json").read_text())
    word_input, word_tensor = b"original typed integer word operations", b"signed i64 pointer offset stage"
    pipeline = ["--triton-to-unstructured=offset-bit-width=64",
        "--triton-arith-to-linalg=pids-to-func-args=false tt-to-func-func=false", "--remove-dead-values", "--canonicalize"]
    stage = {"schema": "t2m.triton-shared-stage.v1", "stage": "source-word-tensor", "stage_completed": True,
        "command": ["optimizer", *pipeline, "source.ttir"], "artifacts": {
            "source.ttir": record(word_input), "lowered.mlir": record(word_tensor)}}
    if isolated:
        event_input, event_output, compute_input = "typed event template", "one native gather", "parent with event in original region"
        def native(name, flags, source, output):
            return {"schema": "t2m.triton-shared-stage.v1", "stage": name, "stage_completed": True,
                "command": ["optimizer", *flags, "source.ttir"], "optimizer": {"sha256": "a" * 64},
                "artifacts": {"source.ttir": record(source.encode()), "lowered.mlir": record(output.encode())}}
        stage = {"schema": "t2m.source-word-stages.v2", "stage_completed": True,
            "input": record(word_input), "output": record(word_tensor), "reconstruction": "same-region-same-event-order",
            "events": {"input": event_input, "output": event_output,
                "native_stage": native("source-word-events", pipeline[:1], event_input, event_output),
                "bindings": [{"symbol": "event0", "opcode": "tt.load", "source_ids": admission["accesses"][0]["source_ids"],
                    "input_types": ["!tt.ptr<f32>", "tensor<2xi64>"], "result_types": ["tensor<2xf32>"], "fill": "unchanged"}]},
            "compute": {"input": compute_input, "native_stage": native("source-word-compute", pipeline[1:], compute_input, word_tensor.decode())}}
        if change == "event-bytes":
            stage["events"]["output"] += "changed"
        elif change == "compute-bytes":
            stage["compute"]["input"] += "changed"
        elif change == "pointer-pass-twice":
            stage["compute"]["native_stage"]["command"].insert(1, pipeline[0])
        elif change == "different-tools":
            stage["events"]["native_stage"]["optimizer"]["sha256"] = "b" * 64
        elif change == "missing-events":
            stage["events"]["bindings"] = []
        elif change == "unknown-source":
            stage["events"]["bindings"][0]["source_ids"] = ["unknown"]
        elif change == "lost-result":
            stage["events"]["bindings"][0]["result_types"] = []
        elif change == "duplicate-event":
            stage["events"]["bindings"] *= 2
        elif change == "unknown-fill":
            stage["events"]["bindings"][0]["fill"] = "discard"
    if change == "pipeline":
        stage["command"].insert(1, "--canonicalize")
    elif change == "width":
        stage["command"][1] = "--triton-to-unstructured=offset-bit-width=32"
    elif change == "stage":
        stage["stage"] = "tensor-ir"
    elif change == "untyped_command":
        stage["command"] = None
    admission.update(address_policy="structured" if change == "scope" else "source-words-offset64",
        word_input_sha256=record(word_input)["sha256"], word_tensor_sha256=record(word_tensor)["sha256"])
    module = parse_mlir_text((root / "model.mlir").read_text())
    entry = next(iter(module.ops))
    if change != "policy_attribute":
        entry.attributes["prov.address_policy"] = StringAttr("source-words-offset64")
    entry.attributes["prov.source_accesses"] = StringAttr(json.dumps([] if change == "intent" else admission["accesses"]))
    model = text(module, generic=True).encode()
    provenance.update(profile="shared-source-word-buffer", mlir_sha256=record(model)["sha256"])
    provenance["memory"].update(address_policy="source-words-offset64", word_stage_sha256=record(encoded(stage))["sha256"],
        admission_sha256=record(encoded(admission))["sha256"])
    if isolated and change != "undeclared-version":
        provenance["memory"]["word_conversion"] = "native-isolated-events-v2"
    files = {"word-input.mlir": word_input + (b"changed" if change == "input" else b""),
        "word-tensor.mlir": word_tensor, "word-stage.json": encoded(stage), "model.mlir": model,
        "memory-admission.json": encoded(admission), "provenance.json": encoded(provenance)}
    for name, raw in files.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    receipt["profile"] = "shared-source-word-buffer"
    (root / "capture_receipt.json").write_bytes(encoded(receipt))


def test_source_word_staging_keeps_policy_evidence_and_original_access_intent(tmp_path, monkeypatch):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    word_fixture(source)
    original = Path.read_bytes
    def read(path):
        if path.name in ("inputs.npz", "golden.npy", "extra.npz", "execution.json"):
            raise AssertionError("source-word compiler accessed evaluator values")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    manifest = stage_compile_inputs(source, target)
    assert manifest["payload_profile"] == "shared-source-word-buffer"
    for name in ("word-input.mlir", "word-tensor.mlir", "word-stage.json"):
        assert (target / ("source/" + name)).read_bytes() == (source / name).read_bytes()
    entry = next(iter(parse_mlir_text((target / "program.mlir").read_text()).ops))
    assert entry.attributes["prov.address_policy"].data == "source-words-offset64"


@pytest.mark.parametrize("change", ["input", "pipeline", "width", "stage", "scope", "intent", "policy_attribute", "untyped_command"])
def test_rehashed_inconsistent_source_word_policy_never_publishes(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    word_fixture(source, change)
    with pytest.raises(ValueError, match="source-word"):
        stage_compile_inputs(source, target)
    assert not target.exists()


def test_isolated_word_stages_retain_intermediates_without_evaluator_values(tmp_path, monkeypatch):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    word_fixture(source, isolated=True)
    original = Path.read_bytes
    def read(path):
        if path.name in ("inputs.npz", "golden.npy", "extra.npz", "execution.json"):
            raise AssertionError("isolated-event compiler accessed evaluator values")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    manifest = stage_compile_inputs(source, target)
    raw = (target / "source/word-stage.json").read_bytes()
    assert raw == (source / "word-stage.json").read_bytes()
    assert json.loads(raw)["schema"] == "t2m.source-word-stages.v2"
    staged = b"".join((target / name).read_bytes() for name in manifest["members"])
    assert b"EVALUATOR" not in staged


@pytest.mark.parametrize("change", ["event-bytes", "compute-bytes", "pointer-pass-twice", "different-tools",
    "missing-events", "unknown-source", "lost-result", "duplicate-event", "unknown-fill", "undeclared-version"])
def test_rehashed_invalid_isolated_word_chain_never_publishes(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    word_fixture(source, change, isolated=True)
    with pytest.raises(ValueError, match="source-word"):
        stage_compile_inputs(source, target)
    assert not target.exists()


def atomic_fixture(root, change=None, *, cas=False, floating=False, word=False):
    """Declaration custody only; the producer's real atomic tests own execution."""
    from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64
    from merlin.xdsl_dialects._common import text
    word_fixture(root)
    receipt = json.loads((root / "capture_receipt.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    admission = json.loads((root / "memory-admission.json").read_text())
    spec = provenance["spec"]
    spec["grid"] = [2**30 + 1]
    for buffer in spec["buffers"]:
        buffer["dtype"] = "f32" if floating or word else "i32"
    info = {"operation": "add", "standard_kind": 1, "word_bits": 32, "source_order": "relaxed", "source_scope": "gpu",
        "logical_scope": "all-logical-programs-in-private-invocation", "standard_order": "acq_rel",
        "external_participants": "excluded-by-private-buffer-ABI"}
    provenance["normalized_operations"][0].update(opcode="tt.atomic_rmw", atomic=json.loads(encoded(info)))
    access = {"buffer": "old", "opcode": "tts.unstructured_atomic_rmw", "effect": "atomic-readwrite",
        "source_ids": ["normalized.0"], "tile": [2], "mask": "source-per-lane", "atomic": info}
    payload = {**info, "buffer": "old", "source_ids": ["normalized.0"], "inactive_old_value": "zero-refinement-of-source-undefined-old-value",
        "effect_boundary": "standard-execute-region-no-inline", "payload": "standard-memref-atomic-rmw-with-original-returned-old-value"}
    effect = "private-invocation-integer-atomics-v1"
    admission.update(schema="t2m.memory-admission.v3", spec=spec, owner_fragment_bits=30, owner_words_per_element=2,
        guard_storage_bytes_per_output_element=24, ownership_protocol="sticky-complemented-atomic-participants-v1",
        effect_profile=effect, accesses=[access], atomic_adaptations=[payload])
    provenance["memory"].update(effect_profile=effect, atomic_payload=json.loads(encoded([payload])))
    if cas:
        info.update(operation="cas", standard_kind="compare-exchange")
        access.update(opcode="tt.atomic_cas", mask="unmasked")
        payload.update(info, inactive_old_value="no-inactive-lanes",
            payload="standard-memref-generic-compare-exchange-with-original-returned-old-value")
        provenance["normalized_operations"][0].update(opcode="tt.atomic_cas", atomic=json.loads(encoded(info)))
        provenance["memory"]["atomic_payload"] = json.loads(encoded([payload]))
    if floating:
        info.update(operation="exch", standard_kind=3, value_dtype="f32", numeric_contract="bitwise-exchange")
        payload.update(info)
        effect = "private-invocation-typed-atomics-v2"
        admission.update(schema="t2m.memory-admission.v4", effect_profile=effect)
        provenance["normalized_operations"][0]["atomic"] = json.loads(encoded(info))
        provenance["memory"].update(effect_profile=effect, atomic_payload=json.loads(encoded([payload])))
    if word:
        info.update(operation="max", standard_kind=6, word_view={"storage_dtype": "f32", "value_dtype": "i32",
            "element_bits": 32, "address_units": "equal-width-elements"})
        payload.update(info, payload="standard-memref-generic-integer-word-update-with-original-returned-old-value")
        effect = "private-invocation-typed-atomics-v2"
        admission.update(schema="t2m.memory-admission.v4", effect_profile=effect)
        provenance["normalized_operations"][0]["atomic"] = json.loads(encoded(info))
        provenance["memory"].update(effect_profile=effect, atomic_payload=json.loads(encoded([payload])))
    module = parse_mlir_text((root / "model.mlir").read_text().replace("f32", "f32" if floating or word else "i32"))
    entry = next(iter(module.ops))
    entry.attributes["prov.effect_profile"] = StringAttr(effect)
    entry.attributes["prov.memory_guard_bytes_per_result_element"] = IntegerAttr(24, i64)
    if change == "protocol": admission["ownership_protocol"] = "sticky-complemented-program-fragments"
    elif change == "fragments": admission["owner_fragment_bits"] = 31
    elif change == "cost": admission["guard_storage_bytes_per_output_element"] = 16
    elif change == "order": access["atomic"]["source_order"] = "unspecified"
    elif change == "scope": access["atomic"]["source_scope"] = "external"
    elif change == "kind": access["atomic"]["standard_kind"] = 15
    elif change == "old-value": payload["payload"] = "discarded"
    elif change == "once": payload["effect_boundary"] = "duplicable"
    elif change == "effect-intent": del entry.attributes["prov.effect_profile"]
    elif change == "profile": provenance["memory"]["effect_profile"] = "ordinary"
    elif change == "cas-mask": access["mask"] = "source-per-lane"
    elif change == "cas-source": provenance["normalized_operations"][0]["opcode"] = "tt.atomic_rmw"
    elif change == "cas-access": access["opcode"] = "tts.unstructured_atomic_rmw"
    entry.attributes["prov.source_accesses"] = StringAttr(json.dumps(admission["accesses"]))
    model = text(module, generic=True).encode()
    provenance.update(spec=spec, mlir_sha256=record(model)["sha256"])
    provenance["memory"]["admission_sha256"] = record(encoded(admission))["sha256"]
    for name, raw in {"spec.json": encoded(spec), "model.mlir": model, "provenance.json": encoded(provenance),
                      "memory-admission.json": encoded(admission)}.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    (root / "capture_receipt.json").write_bytes(encoded(receipt))


def test_atomic_staging_retains_v3_source_contract_and_thirty_bit_cost(tmp_path):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    atomic_fixture(source)
    manifest = stage_compile_inputs(source, target)
    assert manifest["payload_profile"] == "shared-source-word-buffer"
    signature = json.loads((target / "signature.json").read_text())
    assert signature["guard_storage_bytes_per_result_element"] == 24
    assert (target / "source/memory-admission.json").read_bytes() == (source / "memory-admission.json").read_bytes()


@pytest.mark.parametrize("opcode", ["tt.atomic_rmw", "tt.atomic_cas"])
@pytest.mark.parametrize("change", [None, "ordinary", "lost-old-value", "fill", "lost-value"])
def test_isolated_atomic_event_requires_effect_profile_value_and_old_result(tmp_path, change, opcode):
    from merlin.frontends.kernel_word_evidence import checked_word_stages
    source = tmp_path / "capture"
    word_fixture(source, isolated=True)
    stage = json.loads((source / "word-stage.json").read_text())
    provenance = json.loads((source / "provenance.json").read_text())
    memory = provenance["memory"]
    memory["effect_profile"] = "private-invocation-integer-atomics-v1"
    binding = stage["events"]["bindings"][0]
    binding.update(opcode=opcode, input_types=["!tt.ptr<i32>", "tensor<2xi64>", "tensor<2xi32>"] + (["tensor<2xi32>"] if opcode == "tt.atomic_cas" else []),
                   result_types=["tensor<2xi32>"])
    if change == "ordinary": memory.pop("effect_profile")
    elif change == "lost-old-value": binding["result_types"] = []
    elif change == "fill": binding["fill"] = "source-mask-select-after-zero-filled-load"
    elif change == "lost-value": binding["input_types"].pop()
    args = (stage, (source / "word-input.mlir").read_bytes(), (source / "word-tensor.mlir").read_bytes(), memory,
            {record["id"] for record in provenance["normalized_operations"]})
    if change is None:
        checked_word_stages(*args)
    else:
        with pytest.raises(ValueError, match="source/type inventory"):
            checked_word_stages(*args)


@pytest.mark.parametrize("change", [None, "cas-mask", "cas-source", "cas-access"])
def test_cas_staging_keeps_declared_comparison_operation_and_unmasked_access(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    atomic_fixture(source, change, cas=True)
    if change is None:
        assert stage_compile_inputs(source, target)["payload_profile"] == "shared-source-word-buffer"
    else:
        with pytest.raises(ValueError):
            stage_compile_inputs(source, target)
        assert not target.exists()


@pytest.mark.parametrize("change", ["protocol", "fragments", "cost", "order", "scope", "kind", "old-value", "once", "effect-intent", "profile"])
def test_rehashed_inconsistent_atomic_contract_never_publishes(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    atomic_fixture(source, change)
    with pytest.raises(ValueError):
        stage_compile_inputs(source, target)
    assert not target.exists()


@pytest.mark.parametrize("change", [None, "protocol", "fragments", "cost", "order", "scope", "kind", "old-value", "once", "effect-intent", "profile"])
def test_float_exchange_staging_binds_v4_contract_and_guard_cost(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    atomic_fixture(source, change, floating=True)
    if change is not None:
        with pytest.raises(ValueError):
            stage_compile_inputs(source, target)
        assert not target.exists()
        return
    manifest = stage_compile_inputs(source, target)
    assert manifest["payload_profile"] == "shared-source-word-buffer"
    signature = json.loads((target / "signature.json").read_text())
    assert signature["guard_storage_bytes_per_result_element"] == 24
    assert (target / "source/memory-admission.json").read_bytes() == (source / "memory-admission.json").read_bytes()


@pytest.mark.parametrize("change", [None, "protocol", "fragments", "cost", "order", "scope", "kind", "old-value", "once", "effect-intent", "profile"])
def test_atomic_word_staging_binds_integer_updates_on_float_storage(tmp_path, change):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    atomic_fixture(source, change, word=True)
    if change is not None:
        with pytest.raises(ValueError):
            stage_compile_inputs(source, target)
        assert not target.exists()
        return
    stage_compile_inputs(source, target)
    signature = json.loads((target / "signature.json").read_text())
    assert signature["guard_storage_bytes_per_result_element"] == 24
    assert (target / "source/memory-admission.json").read_bytes() == (source / "memory-admission.json").read_bytes()
