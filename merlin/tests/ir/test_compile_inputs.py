"""A compiler input stage never inherits the evaluator's runtime data."""

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.frontends.compile_inputs import load_compile_inputs, stage_compile_inputs
from merlin.frontends.linalg_mlir import parse_mlir_text

SOURCE = """builtin.module attributes {prov.weights_file = "origin/weights.safetensors"} {
  func.func @forward(%a: tensor<2xf32>) -> tensor<2xf32> {
    func.return %a : tensor<2xf32>
  }
}"""


def _write_capture(
    path: Path,
    *,
    source: str = SOURCE,
    dtype: str = "f32",
    input_shapes: tuple[tuple[int, ...], ...] = ((2,),),
    output_shapes: tuple[tuple[int, ...], ...] = ((2,),),
) -> None:
    path.mkdir()
    members = {
        "model.mlir": source.encode(),
        "weights.safetensors": b"declared weight bytes",
        "weights.safetensors.manifest.json": b"{}",
        "frontend-trace.json": json.dumps(
            {"status": "complete", "mlir": {"sha256": hashlib.sha256(source.encode()).hexdigest()}}
        ).encode(),
        "meta.json": json.dumps(
            {
                "ok": True,
                "opaque": 0,
                "weights": "origin/weights.safetensors",
                "frontend_trace": {},
                "func_name": "forward",
                "input_abi": [{"shape": list(shape), "dtype": dtype} for shape in input_shapes],
                "output_abi": [{"shape": list(shape), "dtype": dtype} for shape in output_shapes],
                "dtype": dtype,
            }
        ).encode(),
        "inputs.npz": b"PRIVATE RUNTIME INPUTS",
        "golden.npy": b"PRIVATE REFERENCE OUTPUTS",
    }
    meta = json.loads(members["meta.json"])
    meta["frontend_trace"] = {"sha256": hashlib.sha256(members["frontend-trace.json"]).hexdigest()}
    members["meta.json"] = json.dumps(meta).encode()
    for name, raw in members.items():
        (path / name).write_bytes(raw)
    (path / "capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "materialized_abi": {
                    "complete": True,
                    "inputs": len(input_shapes),
                    "lifted_constants": [],
                },
                "lifted_constants": {},
                "artifacts": {
                    name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                    for name, raw in members.items()
                },
            }
        )
    )


def _write_weighted_capture(
    path: Path,
    *,
    shared_weight: bool = False,
    return_weight: bool = False,
) -> None:
    source = """builtin.module attributes {prov.weights_file = "origin/weights.safetensors"} {
      func.func @forward(%weight: tensor<2x2xf32>, %input: tensor<2x2xf32>) -> tensor<2x2xf32> {
        func.return %input : tensor<2x2xf32>
      }
    }"""
    if return_weight:
        source = source.replace(
            "-> tensor<2x2xf32> {\n        func.return %input : tensor<2x2xf32>",
            "-> (tensor<2x2xf32>, tensor<2x2xf32>) {\n"
            "        func.return %weight, %input : tensor<2x2xf32>, tensor<2x2xf32>",
        )
    if shared_weight:
        source = source.replace(
            "%weight: tensor<2x2xf32>, %input:",
            "%weight: tensor<2x2xf32>, %shared: tensor<2x2xf32>, %input:",
        )
    outputs = ((2, 2), (2, 2)) if return_weight else ((2, 2),)
    _write_capture(path, source=source, input_shapes=((2, 2),), output_shapes=outputs)
    header = json.dumps(
        {
            "linear.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            }
        }
    ).encode()
    argument_manifest = {
        "0": {"kind": "param", "weight": "linear.weight", "dtype": "float32", "shape": [2, 2]},
        "1": {"kind": "input", "name": "input"},
    }
    if shared_weight:
        argument_manifest["1"] = dict(argument_manifest["0"])
        argument_manifest["2"] = {"kind": "input", "name": "input"}
    members = {
        "weights.safetensors": struct.pack("<Q", len(header)) + header + bytes(16),
        "weights.safetensors.manifest.json": json.dumps(argument_manifest).encode(),
        "input_order.json": b'{"input": 0}',
    }
    receipt_path = path / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for name, raw in members.items():
        (path / name).write_bytes(raw)
        receipt["artifacts"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    receipt_path.write_text(json.dumps(receipt))


def test_staging_preserves_lifted_weight_and_runtime_argument_order(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture)
    stage_compile_inputs(capture, staged)
    signature = json.loads((staged / "signature.json").read_text())
    assert signature["inputs"] == [{"shape": [2, 2], "dtype": "f32"}]
    assert signature["arguments"] == [
        {"kind": "param", "weight": "linear.weight", "shape": [2, 2], "dtype": "f32"},
        {"kind": "input", "name": "input", "input_index": 0, "shape": [2, 2], "dtype": "f32"},
    ]
    assert "input_order.json" in {member.name for member in staged.iterdir()}
    loaded = load_compile_inputs(staged)
    assert loaded.entry == "forward"
    assert loaded.argument_bindings == tuple(signature["arguments"])


def test_shared_weight_arguments_bind_one_stored_tensor(tmp_path):
    from merlin.frontends.compile_inputs import load_lifted_argument_bytes

    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture, shared_weight=True)
    stage_compile_inputs(capture, staged)
    bindings = load_compile_inputs(staged).argument_bindings
    assert [row["kind"] for row in bindings] == ["param", "param", "input"]
    assert bindings[0]["weight"] == bindings[1]["weight"] == "linear.weight"
    values = load_lifted_argument_bytes(staged, load_compile_inputs(staged), kinds=("param",), max_bytes=16)
    assert values == {0: bytes(16), 1: bytes(16)} and values[0] is values[1]
    with pytest.raises(ValueError, match="byte budget"):
        load_lifted_argument_bytes(staged, load_compile_inputs(staged), kinds=("param",), max_bytes=15)


def test_c_parameter_binding_embeds_shared_weights_and_keeps_runtime_inputs(tmp_path):
    import shutil

    from merlin.compile.c_parameters import freeze_c_arguments

    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture, shared_weight=True)
    # Exercise exact signed zero, a subnormal, and finite endpoint literals.
    weights = capture / "weights.safetensors"
    raw = weights.read_bytes()
    words = (0x80000000, 1, 0x3F800000, 0x7F7FFFFF)
    raw = raw[:-16] + struct.pack("<4I", *words)
    weights.write_bytes(raw)
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][weights.name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    receipt_path.write_text(json.dumps(receipt))
    stage_compile_inputs(capture, staged)
    selected = load_compile_inputs(staged)
    rendered = freeze_c_arguments(staged, name="model", source_sha256=selected.program_sha256)
    assert rendered.arguments == ("model_parameter_0", "model_parameter_0", "inputs[0]")
    assert len(rendered.parameters) == 1 and len(rendered.runtime_inputs) == 1
    assert rendered.weights_sha256 == hashlib.sha256(raw).hexdigest()
    compiler = shutil.which("cc")
    assert compiler, "a C compiler is required to check embedded parameter bytes"
    source = tmp_path / "check.c"
    source.write_text(
        "#include <stdint.h>\n#include <string.h>\n#include <assert.h>\n"
        + rendered.declarations
        + "\nint main(void) { uint32_t bits[4];\n"
        + "memcpy(bits, model_parameter_0, sizeof(bits));\n"
        + "\n".join(f"assert(bits[{i}] == {word}u);" for i, word in enumerate(words))
        + "return 0; }\n"
    )
    subprocess.run([compiler, "-std=c11", "-Wall", "-Werror", str(source), "-o", str(tmp_path / "check")], check=True)
    subprocess.run([str(tmp_path / "check")], check=True)
    with pytest.raises(ValueError, match="differs from compiled IR"):
        freeze_c_arguments(staged, name="model", source_sha256="0" * 64)
    with pytest.raises(ValueError, match="byte budget"):
        freeze_c_arguments(staged, name="model", source_sha256=selected.program_sha256, max_embedded_bytes=15)


def test_c_parameter_binding_does_not_freeze_mutable_state(tmp_path):
    from merlin.compile.c_parameters import freeze_c_arguments

    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture)
    manifest = capture / "weights.safetensors.manifest.json"
    rows = json.loads(manifest.read_text())
    rows["0"]["kind"] = "buffer"
    raw = json.dumps(rows).encode()
    manifest.write_bytes(raw)
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][manifest.name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    receipt_path.write_text(json.dumps(receipt))
    stage_compile_inputs(capture, staged)
    with pytest.raises(ValueError, match="state interface"):
        freeze_c_arguments(staged, name="model", source_sha256=load_compile_inputs(staged).program_sha256)


def test_native_request_binds_lifted_weights_as_constants(tmp_path):
    from argparse import Namespace

    from merlin.semantic_compiler.model import TensorType
    from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
    from merlin.targetgen.cli import _load_native_request

    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture, return_weight=True)
    stage_compile_inputs(capture, staged)
    request, source_kind, _ = _load_native_request(
        Namespace(compile_inputs=str(staged), linalg_entry=None),
        target_identity="synthetic",
        mode="strict-native",
    )
    assert source_kind == "staged_capture"
    assert [(node.id, node.op, node.effect) for node in request.nodes] == [
        ("arg0", "constant", "constant"),
        ("arg1", "input", "input"),
    ]
    assert request.input_storages == (("arg1", "external"),)
    assert request.constants[0].node_id == "arg0"
    assert request.constants[0].data_hex == bytes(16).hex()
    boundary = TensorValue(TensorType((2, 2), "f32", "raw-bit-pattern"), (1, 2, 3, 4))
    assert evaluate_graph(
        request,
        {"arg1": boundary},
        constants={"arg0": TensorValue(boundary.type, (0,) * 4)},
    ) == (TensorValue(boundary.type, (0,) * 4), boundary)


def test_changed_declared_weight_changes_native_request_identity(tmp_path):
    from argparse import Namespace

    from merlin.targetgen.cli import _load_native_request

    requests = []
    for index in range(2):
        capture, staged = tmp_path / f"capture-{index}", tmp_path / f"compiler-{index}"
        _write_weighted_capture(capture, return_weight=True)
        if index:
            weights = capture / "weights.safetensors"
            changed = weights.read_bytes()[:-16] + bytes([1]) * 16
            weights.write_bytes(changed)
            receipt_path = capture / "capture_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["artifacts"][weights.name] = {
                "bytes": len(changed),
                "sha256": hashlib.sha256(changed).hexdigest(),
            }
            receipt_path.write_text(json.dumps(receipt))
        stage_compile_inputs(capture, staged)
        request, _, _ = _load_native_request(
            Namespace(compile_inputs=str(staged), linalg_entry=None),
            target_identity="synthetic",
            mode="strict-native",
        )
        requests.append(request)
    assert requests[0].source_identity == requests[1].source_identity
    assert requests[0].digest() != requests[1].digest()
    assert requests[0].constants[0].data_hex != requests[1].constants[0].data_hex


def test_lifted_weight_reader_rejects_change_after_staging_load(tmp_path):
    from merlin.frontends.compile_inputs import load_lifted_argument_bytes

    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture)
    stage_compile_inputs(capture, staged)
    loaded = load_compile_inputs(staged)
    weights = staged / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1] + b"\x01")
    with pytest.raises(ValueError, match="weights changed"):
        load_lifted_argument_bytes(staged, loaded)


def test_staging_rejects_duplicate_safetensors_weight_keys(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture)
    entry = '"linear.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}'
    header = ("{" + entry + "," + entry + "}").encode()
    changed = struct.pack("<Q", len(header)) + header + bytes(16)
    weights = capture / "weights.safetensors"
    weights.write_bytes(changed)
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][weights.name] = {
        "bytes": len(changed),
        "sha256": hashlib.sha256(changed).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="duplicate safetensors header key"):
        stage_compile_inputs(capture, staged)


def test_staging_rejects_lifted_weight_shape_mismatch(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture)
    manifest_path = capture / "weights.safetensors.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["0"]["shape"] = [4]
    changed = json.dumps(manifest).encode()
    manifest_path.write_bytes(changed)
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"][manifest_path.name] = {
        "bytes": len(changed),
        "sha256": hashlib.sha256(changed).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="weight.*shape|argument.*shape"):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()


def test_staged_weight_mapping_is_rechecked_after_digest_rewrite(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_weighted_capture(capture)
    stage_compile_inputs(capture, staged)
    signature_path = staged / "signature.json"
    signature = json.loads(signature_path.read_text())
    signature["arguments"][0]["weight"] = "another.weight"
    changed = json.dumps(signature).encode()
    signature_path.write_bytes(changed)
    manifest_path = staged / "compile-inputs.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["members"]["signature.json"] = hashlib.sha256(changed).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="signature arguments differ"):
        load_compile_inputs(staged)


def test_staging_rewrites_only_declared_weight_reference_and_excludes_evaluator_data(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    manifest = stage_compile_inputs(capture, staged)
    assert set(path.name for path in staged.iterdir()) == {
        "program.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "frontend-trace.json",
        "signature.json",
        "compile-inputs.json",
    }
    assert manifest["members"]["program.mlir"] == hashlib.sha256((staged / "program.mlir").read_bytes()).hexdigest()
    parsed = parse_mlir_text((staged / "program.mlir").read_text())
    assert parsed.attributes["prov.weights_file"].data == "weights.safetensors"
    assert json.loads((staged / "signature.json").read_text())["inputs"] == [{"shape": [2], "dtype": "f32"}]
    assert "PRIVATE" not in b"".join(member.read_bytes() for member in staged.iterdir()).decode()


@pytest.mark.parametrize("direction", ["input_abi", "output_abi"])
def test_staging_rejects_capture_abi_that_disagrees_with_parsed_ir(tmp_path, direction):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    metadata_path = capture / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[direction][0]["shape"] = [3]
    changed = json.dumps(metadata).encode()
    metadata_path.write_bytes(changed)
    receipt_path = capture / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"]["meta.json"] = {
        "bytes": len(changed),
        "sha256": hashlib.sha256(changed).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="signature .* differ from parsed function"):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()


@pytest.mark.parametrize("name", ["model.mlir", "weights.safetensors", "meta.json", "frontend-trace.json"])
def test_changed_declared_input_refuses_without_publishing(tmp_path, name):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    (capture / name).write_bytes((capture / name).read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="differs from receipt"):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()


def test_missing_weight_attribute_or_symlink_refuses(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    (capture / "weights.safetensors").unlink()
    (capture / "weights.safetensors").symlink_to("golden.npy")
    with pytest.raises(ValueError, match="symlink"):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()


def test_unbound_lifted_constant_refuses(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    receipt = json.loads((capture / "capture_receipt.json").read_text())
    receipt["lifted_constants"] = {"missing": "not copied into compiler inputs"}
    (capture / "capture_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="unbound lifted constants"):
        stage_compile_inputs(capture, staged)
    assert not staged.exists()


def test_does_not_read_runtime_or_golden_members(tmp_path, monkeypatch):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    original = Path.read_bytes

    def guarded(path):
        if path.name in {"inputs.npz", "golden.npy"}:
            raise AssertionError("evaluator data accessed by compiler staging")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    stage_compile_inputs(capture, staged)


def test_selected_precision_policy_is_source_bound_and_staged_without_evaluator_data(tmp_path, monkeypatch):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    policy = tmp_path / "precision.json"
    raw = json.dumps(
        {
            "schema": "synthetic.precision.v1",
            "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            "transformation": "declared_candidate",
        }
    ).encode()
    policy.write_bytes(raw)
    original = Path.read_bytes

    def guarded(path):
        if path.name in {"inputs.npz", "golden.npy"}:
            raise AssertionError("evaluator data accessed by compiler staging")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    manifest = stage_compile_inputs(capture, staged, precision_policy=policy)
    assert (staged / "precision-policy.json").read_bytes() == raw
    assert manifest["members"]["precision-policy.json"] == hashlib.sha256(raw).hexdigest()
    assert "PRIVATE" not in b"".join(member.read_bytes() for member in staged.iterdir()).decode()
    loaded = load_compile_inputs(staged)
    assert loaded.entry == "forward" and loaded.precision_policy["transformation"] == "declared_candidate"
    assert loaded.precision_policy_document == raw
    assert 'prov.weights_file = "weights.safetensors"' in loaded.program
    assert loaded.original_source_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert loaded.program_sha256 == manifest["members"]["program.mlir"]
    assert loaded.precision_policy_sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("invalid", ["stale", "malformed", "symlink"])
def test_invalid_precision_policy_does_not_publish_compiler_inputs(tmp_path, invalid):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    policy = tmp_path / "precision.json"
    if invalid == "stale":
        policy.write_text(json.dumps({"schema": "synthetic.precision.v1", "source_sha256": "0" * 64}))
    elif invalid == "malformed":
        policy.write_text(json.dumps({"schema": "synthetic.precision.v1", "source_sha256": "INVALID"}))
    else:
        policy.symlink_to(capture / "golden.npy")
    with pytest.raises(ValueError, match="precision policy"):
        stage_compile_inputs(capture, staged, precision_policy=policy)
    assert not staged.exists()


def test_native_request_reads_staged_capture_and_rejects_unimplemented_precision_transform(tmp_path):
    from argparse import Namespace

    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.targetgen.cli import _load_native_request

    source = SOURCE.replace("f32", "bf16")
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture, source=source, dtype="bf16")
    stage_compile_inputs(capture, staged)
    args = Namespace(compile_inputs=str(staged), linalg=None, linalg_entry=None, request=None)
    request, kind, operations = _load_native_request(args, target_identity="synthetic-tile", mode="diagnostic")
    assert kind == "staged_capture" and operations == ("func.return",)
    assert request.target_identity == "synthetic-tile" and request.outputs == ("arg0",)
    policy = tmp_path / "precision.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "synthetic.precision.v1",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
    )
    selected = tmp_path / "selected"
    stage_compile_inputs(capture, selected, precision_policy=policy)
    args.compile_inputs = str(selected)
    with pytest.raises(LinalgBridgeError, match="no admitted native transformation"):
        _load_native_request(args, target_identity="synthetic-tile", mode="diagnostic")


def test_native_request_checks_source_bound_matmul_decisions_before_refusing_transform(tmp_path):
    from argparse import Namespace

    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.targetgen.cli import _load_native_request

    source = """builtin.module attributes {prov.weights_file = "origin/weights.safetensors"} {
      func.func @forward(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %zero = arith.constant 0.000000e+00 : f32
        %empty = tensor.empty() : tensor<2x2xf32>
        %init = linalg.fill ins(%zero : f32)
          outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
        %result = linalg.matmul {prov.region_id = "mm"}
          ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
          outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
        func.return %result : tensor<2x2xf32>
      }
    }"""
    capture = tmp_path / "capture"
    _write_capture(capture, source=source, input_shapes=((2, 2), (2, 2)), output_shapes=((2, 2),))
    policy = tmp_path / "policy.json"
    base = {"schema": "synthetic.precision.v1", "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    policy.write_text(json.dumps({**base, "matmul_regions": {"wrong": {}}}))
    staged = tmp_path / "wrong"
    stage_compile_inputs(capture, staged, precision_policy=policy)
    args = Namespace(compile_inputs=str(staged), linalg=None, linalg_entry=None, request=None)
    with pytest.raises(LinalgBridgeError, match="missing=\\['mm'\\], extra=\\['wrong'\\]"):
        _load_native_request(args, target_identity="synthetic-tile", mode="diagnostic")

    policy.write_text(json.dumps({**base, "matmul_regions": {"mm": {}}}))
    admitted = tmp_path / "matched"
    stage_compile_inputs(capture, admitted, precision_policy=policy)
    args.compile_inputs = str(admitted)
    with pytest.raises(LinalgBridgeError, match="no admitted native transformation"):
        _load_native_request(args, target_identity="synthetic-tile", mode="diagnostic")


def test_staged_compiler_inputs_reject_modified_or_undeclared_members(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    stage_compile_inputs(capture, staged)
    (staged / "program.mlir").write_text((staged / "program.mlir").read_text() + " ")
    with pytest.raises(ValueError, match="differs from manifest"):
        load_compile_inputs(staged)
    (staged / "program.mlir").write_text((staged / "program.mlir").read_text()[:-1])
    (staged / "golden.npy").write_bytes(b"private result")
    with pytest.raises(ValueError, match="undeclared members"):
        load_compile_inputs(staged)


def test_staged_compiler_inputs_recheck_signature_even_with_matching_member_digest(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    stage_compile_inputs(capture, staged)
    signature_path = staged / "signature.json"
    signature = json.loads(signature_path.read_text())
    signature["outputs"][0]["dtype"] = "bf16"
    changed = json.dumps(signature).encode()
    signature_path.write_bytes(changed)
    manifest_path = staged / "compile-inputs.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["members"]["signature.json"] = hashlib.sha256(changed).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="signature outputs differ from parsed function"):
        load_compile_inputs(staged)


def test_native_cli_refuses_staged_precision_policy_without_a_transform(tmp_path, monkeypatch):
    from merlin.semantic_compiler import snapshot as snapshot_module
    from merlin.targetgen import cli

    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    policy = tmp_path / "precision.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "synthetic.precision.v1",
                "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            }
        )
    )
    stage_compile_inputs(capture, staged, precision_policy=policy)

    class Profile:
        target_identity = "synthetic-target"

    class Snapshot:
        profile = Profile()

    monkeypatch.setattr(snapshot_module, "open_native_snapshot", lambda _: Snapshot())
    output = tmp_path / "selection.json"
    result = cli.main(
        [
            "native-select",
            "--engine",
            "merlin_native",
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--compile-inputs",
            str(staged),
            "--mode",
            "diagnostic",
            "--out",
            str(output),
        ]
    )
    assert result == 2
    report = json.loads(output.read_text())
    assert report["engine"] == "merlin_native"
    assert report["status"] == "unsupported_semantics"
    assert "no admitted native transformation" in report["reason"]


def test_installed_stage_capture_cli_reports_failure_without_stale_success(tmp_path):
    capture, staged, status = tmp_path / "capture", tmp_path / "compiler", tmp_path / "status.json"
    _write_capture(capture)
    policy = tmp_path / "precision.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "synthetic.precision.v1",
                "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            }
        )
    )
    command = Path(sys.executable).with_name("merlin-targetgen")
    assert command.is_file()
    argv = [
        str(command),
        "stage-capture",
        "--capture",
        str(capture),
        "--out",
        str(staged),
        "--precision-policy",
        str(policy),
        "--status-file",
        str(status),
    ]
    first = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)
    assert first.returncode == 0, first.stderr + first.stdout
    assert json.loads(status.read_text())["status"] == "staged"
    second = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)
    assert second.returncode == 2
    assert json.loads(status.read_text())["status"] == "compile_input_error"
    assert json.loads((staged / "compile-inputs.json").read_text())["schema"] == "merlin.compile-inputs.v1"
    assert (staged / "precision-policy.json").read_bytes() == policy.read_bytes()
    stale = tmp_path / "stale-precision.json"
    stale.write_text(json.dumps({"schema": "synthetic.precision.v1", "source_sha256": "0" * 64}))
    stale_output = tmp_path / "stale-compiler"
    rejected = subprocess.run(
        [
            str(command),
            "stage-capture",
            "--capture",
            str(capture),
            "--out",
            str(stale_output),
            "--precision-policy",
            str(stale),
            "--status-file",
            str(status),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert rejected.returncode == 2
    assert not stale_output.exists()
    assert json.loads(status.read_text())["status"] == "compile_input_error"
