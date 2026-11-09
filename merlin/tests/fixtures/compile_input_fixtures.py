"""Source-bound capture fixtures shared by intake and runtime conformance."""
import hashlib
import json
import struct
from pathlib import Path

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


