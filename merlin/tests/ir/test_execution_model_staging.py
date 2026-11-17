"""Compiler-only executed model receipt protocol; no numeric oracle claim."""

import json
from pathlib import Path

import pytest
from test_derived_compile_inputs import capture_fixture, encoded, record
from test_execution_model_correspondence import printed
from test_execution_model_correspondence import selection as selection
from xdsl.dialects import arith, builtin

from merlin.frontends.compile_inputs import stage_compile_inputs
from merlin.frontends.linalg_mlir import parse_mlir_text


@pytest.fixture
def capture(tmp_path, selection):
    parent, bound, bindings, interfaces = selection
    root = tmp_path / "capture"
    receipt = capture_fixture(root)
    digest = next(iter(interfaces))
    spec, located = interfaces[digest]
    ttir = b"declared TTIR for compiler-only protocol fixture"
    source_hash = record(ttir)["sha256"]
    provenance = {
        "schema": "t2m.shared-tensor-provenance.v1",
        "spec": spec,
        "mlir_sha256": digest,
        "source_sha256": source_hash,
    }
    bindings["kernels"] = {digest: provenance}
    bindings["calls"][0]["source_sha256"] = source_hash
    parent_source, bound_source = printed(parent, locations=True).encode(), printed(bound, locations=True).encode()
    bindings.update(input_model_sha256=record(parent_source)["sha256"], mlir_sha256=record(bound_source)["sha256"])
    parent_receipt = json.loads((root / "parent/capture_receipt.json").read_bytes())
    changes = {
        "parent/model.mlir": parent_source,
        "parent/frontend-trace.json": encoded({"status": "complete", "mlir": record(parent_source)}),
        "model.mlir": bound_source,
        "kernels/0/model.mlir": printed(parse_mlir_text(located)).encode(),
        "kernels/0/located.mlir": located.encode(),
        "kernels/0/source.ttir": ttir,
        "kernels/0/spec.json": encoded(spec),
        "kernels/0/provenance.json": encoded(provenance),
        "kernel-bindings.json": encoded(bindings),
    }
    for name in ("model.mlir", "frontend-trace.json"):
        parent_receipt["artifacts"][name] = record(changes["parent/" + name])
    changes["parent/capture_receipt.json"] = encoded(parent_receipt)
    for name, raw in changes.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    receipt.update(
        schema="t2m.bound-model-receipt.v2",
        result_profile="merlin.execution-owned-results.v1",
        qualification="source-only-unqualified",
        kernel_artifacts={digest: "kernels/0"},
        parent_receipt=record(changes["parent/capture_receipt.json"]),
    )
    (root / "capture_receipt.json").write_bytes(encoded(receipt))
    return root, receipt


def test_source_only_staging_reconstructs_v4_without_opening_evaluator_members(capture, tmp_path, monkeypatch):
    root, receipt = capture
    original = Path.read_bytes

    def guarded(path):
        if path.name in ("inputs.npz", "golden.npy", "goldens.npz", "tensor_words.npz", "execution.json", "meta.json"):
            raise AssertionError("compiler opened an evaluator member")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    staged = tmp_path / "compiler"
    manifest = stage_compile_inputs(root, staged, result_mode="execution-owned")
    assert manifest["result_mode"] == receipt["result_profile"]
    signature = json.loads((staged / "signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v6"
    assert signature["input_constraints"]["inputs"] == [{"kind": "tensor", "dtype": "i32", "shape": [4]}]
    correspondence = json.loads((staged / "source-correspondence.json").read_text())
    assert correspondence["schema"] == "merlin.derived-source-correspondence.v2"
    assert correspondence["execution_binding"]["calls"] == 1
    assert (staged / "source/kernels/0/located.mlir").read_bytes() == original(root / "kernels/0/located.mlir")


@pytest.mark.parametrize(
    "damage", ["default-mode", "receipt-downgrade", "binding-downgrade", "missing-located", "body", "result-type"]
)
def test_rehashed_or_downgraded_profiles_cannot_bypass_executed_staging(capture, tmp_path, damage):
    root, receipt = capture
    mode = "execution-owned"
    bindings = json.loads((root / "kernel-bindings.json").read_text())
    if damage == "default-mode":
        mode = None
    elif damage == "receipt-downgrade":
        receipt["schema"] = "t2m.bound-model-receipt.v1"
    elif damage == "binding-downgrade":
        bindings["schema"] = "t2m.model-kernel-binding.v1"
    elif damage == "missing-located":
        receipt["artifacts"].pop("kernels/0/located.mlir")
    else:
        module = parse_mlir_text((root / "model.mlir").read_text())
        if damage == "body":
            constant = next(op for op in module.walk() if isinstance(op, arith.ConstantOp))
            constant.properties["value"] = builtin.IntegerAttr(8, builtin.i32)
        else:
            old = next(op for op in module.walk() if isinstance(op, arith.TruncIOp))
            new = arith.TruncIOp(old.operands[0], builtin.i8)
            old.parent.insert_op_before(new, old)
            old.parent.erase_op(old)
        module.verify()
        raw = printed(module, locations=True).encode()
        (root / "model.mlir").write_bytes(raw)
        receipt["artifacts"]["model.mlir"] = record(raw)
        bindings["mlir_sha256"] = record(raw)["sha256"]
    raw = encoded(bindings)
    (root / "kernel-bindings.json").write_bytes(raw)
    receipt["artifacts"]["kernel-bindings.json"] = record(raw)
    (root / "capture_receipt.json").write_bytes(encoded(receipt))
    staged = tmp_path / "compiler"
    with pytest.raises(ValueError):
        stage_compile_inputs(root, staged, result_mode=mode)
    assert not staged.exists()
