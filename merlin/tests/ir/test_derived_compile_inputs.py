"""Derived source lineage and compiler/evaluator separation, without numeric admission."""

import hashlib
import json
from pathlib import Path

import pytest

from merlin.frontends.compile_inputs import stage_compile_inputs
from merlin.frontends.linalg_mlir import parse_mlir_text


SOURCE = b'''builtin.module attributes {prov.weights_file = "weights.safetensors"} {
  func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {
    return %x : tensor<2xf32>
  }
}'''


def record(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def capture_fixture(root):
    """A byte-lineage protocol fixture, not a claimed capture/numeric oracle."""
    root.mkdir()
    parent_model, kernel_model, ttir = SOURCE, SOURCE.replace(b"@forward", b"@copy"), b"original TTIR protocol fixture"
    spec = {"schema": "t2m.shared-tensor-abi.v2", "entry": "copy"}
    digest = record(kernel_model)["sha256"]
    provenance = {"schema": "t2m.shared-tensor-provenance.v1", "spec": spec,
                  "mlir_sha256": digest, "source_sha256": record(ttir)["sha256"]}
    parent_files = {"model.mlir": parent_model, "frontend-trace.json": encoded({"status": "complete", "mlir": record(parent_model)}),
                    "weights.safetensors": b"declared parameter bytes", "weights.safetensors.manifest.json": b'{}',
                    "inputs.npz": b"EVALUATOR INPUTS", "golden.npy": b"EVALUATOR GOLDEN",
                    "goldens.npz": b"EVALUATOR TUPLE REFERENCES", "tensor_words.npz": b"EVALUATOR BF16 WORDS"}
    parent = {"schema": "m2m.capture-receipt.v1", "materialized_abi": {"complete": True, "inputs": 1},
              "artifacts": {name: record(raw) for name, raw in parent_files.items()}}
    files = {"model.mlir": SOURCE, "parent/capture_receipt.json": encoded(parent),
             "execution.json": b"EVALUATOR EXECUTION REPORT", "meta.json": b'{}'}
    mapping = {}
    for name, raw in parent_files.items():
        destination = "parent/" + name if name in ("model.mlir", "frontend-trace.json") else name
        files[destination], mapping[name] = raw, destination
    files.update({"kernels/0/model.mlir": kernel_model, "kernels/0/source.ttir": ttir,
                  "kernels/0/spec.json": encoded(spec), "kernels/0/provenance.json": encoded(provenance),
                  "kernel-bindings.json": encoded({"schema": "t2m.model-kernel-binding.v1", "mlir_sha256": record(SOURCE)["sha256"],
                      "input_model_sha256": record(parent_model)["sha256"], "kernels": {digest: provenance},
                      "calls": [{"id": "call.0", "implementation_sha256": digest, "source_sha256": record(ttir)["sha256"]}]})})
    receipt = {"schema": "t2m.bound-model-receipt.v1", "materialized_abi": "unchanged-from-parent",
               "parent_receipt": record(files["parent/capture_receipt.json"]), "parent_artifact_bindings": mapping,
               "kernel_artifacts": {digest: "kernels/0"}, "artifacts": {name: record(raw) for name, raw in files.items()}}
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (root / "capture_receipt.json").write_bytes(encoded(receipt))
    return receipt


def test_derived_staging_preserves_both_graphs_and_excludes_evaluator_values_and_digests(tmp_path, monkeypatch):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    receipt = capture_fixture(capture)
    read = Path.read_bytes

    def guarded(path):
        if path.name in ("inputs.npz", "golden.npy", "goldens.npz", "tensor_words.npz", "execution.json"):
            raise AssertionError("compiler accessed evaluator data")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    manifest = stage_compile_inputs(capture, staged)
    assert manifest["source_profile"] == "derived-tensor-model"
    assert "frontend-trace.json" not in manifest["members"]
    parsed = parse_mlir_text((staged / "program.mlir").read_text())
    parsed.verify()
    function = next(iter(parsed.ops))
    assert function.arg_attrs.data[0].data["bufferization.writable"].value.data == 0
    assert json.loads((staged / "signature.json").read_text())["inputs"] == [{"shape": [2], "dtype": "f32"}]
    for name, digest in manifest["members"].items():
        assert hashlib.sha256((staged / name).read_bytes()).hexdigest() == digest
    data = b"".join((staged / name).read_bytes() for name in manifest["members"])
    assert b"EVALUATOR" not in data
    assert "tensor_words.npz" in manifest["excluded_evaluator_members"]
    for name in ("inputs.npz", "golden.npy", "goldens.npz", "tensor_words.npz", "execution.json"):
        assert receipt["artifacts"][name]["sha256"].encode() not in data


@pytest.mark.parametrize("name", ["model.mlir", "kernel-bindings.json", "kernels/0/source.ttir", "parent/model.mlir"])
def test_changed_derived_compiler_input_never_publishes_a_stage(tmp_path, name):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    capture_fixture(capture)
    (capture / name).write_bytes((capture / name).read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs from receipt"):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()


def test_derived_identity_program_has_no_required_kernel_calls(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    receipt = capture_fixture(capture)
    name = "kernel-bindings.json"
    bindings = json.loads((capture / name).read_bytes())
    bindings.update(calls=[], kernels={})
    raw = encoded(bindings)
    (capture / name).write_bytes(raw)
    receipt["kernel_artifacts"] = {}
    receipt["artifacts"][name] = record(raw)
    (capture / "capture_receipt.json").write_bytes(encoded(receipt))
    stage_compile_inputs(capture, staged)
    assert not (staged / "source/kernels").exists()


@pytest.mark.parametrize("relation", ["model", "kernel", "parent", "evaluator_path"])
def test_rehashed_broken_lineage_or_evaluator_alias_is_still_refused(tmp_path, relation):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    receipt = capture_fixture(capture)
    name = "kernel-bindings.json"
    binding = json.loads((capture / name).read_bytes())
    if relation == "model":
        binding["mlir_sha256"] = "0" * 64
    elif relation == "kernel":
        binding["calls"][0]["source_sha256"] = "0" * 64
    elif relation == "parent":
        binding["input_model_sha256"] = "0" * 64
    else:
        receipt["parent_artifact_bindings"]["model.mlir"] = "golden.npy"
    raw = encoded(binding)
    (capture / name).write_bytes(raw)
    receipt["artifacts"][name] = record(raw)
    (capture / "capture_receipt.json").write_bytes(encoded(receipt))
    with pytest.raises(ValueError):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()


def guarded_fixture(root, *, scratch=64, typed_shape=2):
    """Declared lineage only; no executable memory or numerical claim."""
    receipt = capture_fixture(root)
    spec = {"schema": "t2m.shared-tensor-abi.v2", "entry": "copy", "buffers": [
        {"name": "x", "shape": [2], "dtype": "f32", "effect": "read"},
        {"name": "out", "shape": [2], "dtype": "f32", "effect": "write"}]}
    model = SOURCE.replace(b") -> tensor<2xf32> {", b") -> tensor<2xf32> attributes {prov.memory_guard_scratch_bytes = " + str(scratch).encode() + b" : i64} {")
    kernel = SOURCE.replace(b"@forward", b"@copy").replace(b"tensor<2xf32>", ("tensor<" + str(typed_shape) + "xf32>").encode()).replace(
        b"> {", b"> attributes {prov.memory_guard_bytes_per_result_element = 16 : i64} {")
    ttir = (root / "kernels/0/source.ttir").read_bytes()
    memory_input, memory_output = b"guarded input", b"memory output"
    admission = {"schema": "t2m.memory-admission.v1", "strategy": "runtime-guarded-private-output-state", "spec": spec,
        "internal_arguments": [{"kind": "extent", "buffer": "x"}, {"kind": "extent", "buffer": "out"},
            {"kind": "owners", "buffer": "out"}, {"kind": "visited", "buffer": "out"}, {"kind": "program"}],
        "guarded_input_sha256": record(memory_input)["sha256"], "continuation_output_sha256": record(memory_output)["sha256"],
        "guard_storage_bytes_per_output_element": 16, "writable_reads": "refused",
        "fresh_initialization": "post-grid-complete-write-coverage", "accesses": []}
    provenance = {"profile": "shared-guarded-buffer", "spec": spec, "mlir_sha256": record(kernel)["sha256"],
        "source_sha256": record(ttir)["sha256"], "memory": {"admission_sha256": record(encoded(admission))["sha256"],
            "continuation_input_sha256": record(memory_input)["sha256"], "continuation_output_sha256": record(memory_output)["sha256"]}}
    digest = record(kernel)["sha256"]
    bindings = json.loads((root / "kernel-bindings.json").read_text())
    bindings.update(mlir_sha256=record(model)["sha256"], kernels={digest: provenance}, guard_storage_bytes=64,
        calls=[{"id": "call." + str(i), "implementation_sha256": digest, "source_sha256": record(ttir)["sha256"]} for i in range(2)])
    files = {"model.mlir": model, "kernels/0/model.mlir": kernel, "kernels/0/spec.json": encoded(spec),
        "kernels/0/provenance.json": encoded(provenance), "kernel-bindings.json": encoded(bindings),
        "kernels/0/memory-input.mlir": memory_input, "kernels/0/memory-stage.mlir": memory_output,
        "kernels/0/memory-admission.json": encoded(admission), "kernels/0/memory-stage.json": encoded({
            "schema": "t2m.triton-shared-stage.v1", "stage": "memory-ir", "stage_completed": True,
            "artifacts": {"source.ttir": record(memory_input), "lowered.mlir": record(memory_output)}})}
    for name, raw in files.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    receipt["kernel_artifacts"] = {digest: "kernels/0"}
    (root / "capture_receipt.json").write_bytes(encoded(receipt))


def test_guarded_derived_scratch_counts_kernel_calls_and_copies_admission(tmp_path):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    guarded_fixture(source)
    stage_compile_inputs(source, target)
    signature = json.loads((target / "signature.json").read_text())
    assert signature["guard_storage_bytes"] == 64
    assert (target / "source/kernels/0/memory-admission.json").is_file()


@pytest.mark.parametrize("options", [{"scratch": 32}, {"scratch": 0}, {"typed_shape": 3}])
def test_rehashed_guarded_derived_scratch_mismatch_never_publishes(tmp_path, options):
    source, target = tmp_path / "capture", tmp_path / "compiler"
    guarded_fixture(source, **options)
    with pytest.raises(ValueError, match="guarded"):
        stage_compile_inputs(source, target)
    assert not target.exists()
