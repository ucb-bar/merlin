"""A compiler input stage never inherits the evaluator's runtime data."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.frontends.compile_inputs import stage_compile_inputs
from merlin.frontends.linalg_mlir import parse_mlir_text

SOURCE = '''builtin.module attributes {prov.weights_file = "origin/weights.safetensors"} {
  func.func @forward(%a: tensor<2xf32>) -> tensor<2xf32> {
    func.return %a : tensor<2xf32>
  }
}'''


def _write_capture(path: Path) -> None:
    path.mkdir()
    members = {
        "model.mlir": SOURCE.encode(),
        "weights.safetensors": b"declared weight bytes",
        "weights.safetensors.manifest.json": b'{}',
        "frontend-trace.json": json.dumps(
            {"status": "complete", "mlir": {"sha256": hashlib.sha256(SOURCE.encode()).hexdigest()}}
        ).encode(),
        "meta.json": json.dumps(
            {
                "ok": True,
                "opaque": 0,
                "weights": "origin/weights.safetensors",
                "frontend_trace": {},
                "func_name": "forward",
                "input_abi": [{"shape": [2], "dtype": "f32"}],
                "output_abi": [{"shape": [2], "dtype": "f32"}],
                "dtype": "fp32",
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
                "materialized_abi": {"complete": True, "inputs": 1, "lifted_constants": []},
                "lifted_constants": {},
                "artifacts": {
                    name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                    for name, raw in members.items()
                },
            }
        )
    )


def test_staging_rewrites_only_declared_weight_reference_and_excludes_evaluator_data(tmp_path):
    capture, staged = tmp_path / "capture", tmp_path / "compiler"
    _write_capture(capture)
    manifest = stage_compile_inputs(capture, staged)
    assert set(path.name for path in staged.iterdir()) == {
        "program.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
        "frontend-trace.json", "signature.json", "compile-inputs.json",
    }
    assert manifest["members"]["program.mlir"] == hashlib.sha256((staged / "program.mlir").read_bytes()).hexdigest()
    parsed = parse_mlir_text((staged / "program.mlir").read_text())
    assert parsed.attributes["prov.weights_file"].data == "weights.safetensors"
    assert json.loads((staged / "signature.json").read_text())["inputs"] == [{"shape": [2], "dtype": "f32"}]
    assert "PRIVATE" not in b"".join(member.read_bytes() for member in staged.iterdir()).decode()


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


def test_installed_stage_capture_cli_reports_failure_without_stale_success(tmp_path):
    capture, staged, status = tmp_path / "capture", tmp_path / "compiler", tmp_path / "status.json"
    _write_capture(capture)
    command = Path(sys.executable).with_name("merlin-targetgen")
    assert command.is_file()
    argv = [str(command), "stage-capture", "--capture", str(capture), "--out", str(staged),
            "--status-file", str(status)]
    first = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)
    assert first.returncode == 0, first.stderr + first.stdout
    assert json.loads(status.read_text())["status"] == "staged"
    second = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)
    assert second.returncode == 2
    assert json.loads(status.read_text())["status"] == "compile_input_error"
    assert json.loads((staged / "compile-inputs.json").read_text())["schema"] == "merlin.compile-inputs.v1"
