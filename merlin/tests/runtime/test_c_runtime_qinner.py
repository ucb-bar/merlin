"""The generated argument table binds quant-inner tensors, and caps a session corpus on request.

`c_runtime.generate` reads the BUNDLE while the object is lowered from the prepared IR. These pin
the two halves of that seam: the table gains a row (and the blob gains the bytes) for exactly the
arguments `qinner.lift` appends, an absent tensor is refused rather than zero-filled, and a caller
that only measures a few steps is not forced to embed the whole corpus as C literals.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
import yaml

from merlin.llvmlower import c_runtime, qinner

_MODULE = """builtin.module {
  func.func @forward(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %e = tensor.empty() : tensor<4xf32>
    %d = tensor.empty() : tensor<4xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, \
affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} \
ins(%arg0, %e : tensor<4xf32>, tensor<4xf32>) outs(%d : tensor<4xf32>) \
attrs = {prov.quant_inner_1 = "fc.tensor_impl.scale"} {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
    } -> tensor<4xf32>
    func.return %r : tensor<4xf32>
  }
}
"""

SCALE = np.array([1.0, 2.0, 3.0, 4.0], np.float32)


def _bundle(tmp_path, *, with_scale: bool = True, steps: int = 0):
    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text(_MODULE, encoding="utf-8")
    header = json.dumps({}).encode("utf-8")
    (model / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (model / "weights.safetensors.manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "x"}}), encoding="utf-8")
    (model / "input_order.json").write_text(json.dumps({"x": 0}), encoding="utf-8")
    np.savez(model / "inputs.npz", in0=np.ones(4, np.float32))
    extra = {"qinner::fc.tensor_impl.scale": SCALE} if with_scale else {}
    np.savez(model / "extra.npz", **extra)
    if steps:
        np.savez(model / "session_inputs.npz",
                 frames=np.arange(steps * 4, dtype=np.float32).reshape(steps, 4))
        np.savez(model / "session_goldens.npz",
                 output0=np.zeros((steps, 4), np.float32))
        (model / "session_contract.yaml").write_text(yaml.safe_dump({
            "version": 1, "kind": "frames", "paper_ready": False, "stages": ["step"],
            "inputs": "session_inputs.npz", "states": [],
            "streams": [{"name": "x", "input_arg": 0, "key": "frames"}],
            "correctness": {"scope": "trajectory", "golden": "session_goldens.npz",
                            "key": "output0", "output_index": 0},
            "quality": {"scope": "trajectory", "golden": "session_goldens.npz",
                        "key": "output0", "output_index": 0},
        }), encoding="utf-8")
    return model


def test_table_and_blob_carry_the_lifted_quant_inner_tensor(tmp_path):
    model = _bundle(tmp_path)
    out = tmp_path / "generated"
    info = c_runtime.generate(model, out, model / "inputs.npz")

    assert info["n_qinner"] == 1
    gen = (out / "model_gen.h").read_text(encoding="utf-8")
    # one input, one lifted quant-inner argument, one output -- and the lifted one is a blob row
    assert "MERLIN_N_ARGS 3" in gen
    rows = [ln for ln in gen.splitlines() if ln.strip().startswith("{MERLIN_")]
    assert len(rows) == 3 and "MERLIN_WEIGHT" in rows[1]

    offset = int(rows[1].split(",")[1].strip().rstrip("L"))
    blob = (out / "weights.bin").read_bytes()
    stored = np.frombuffer(blob[offset:offset + SCALE.nbytes], dtype=np.float32)
    assert np.array_equal(stored, SCALE)
    assert info["weights_bytes"] == len(blob)

    # and the derivation the object is built from names exactly that argument
    assert [a.key for a in qinner.plan_for_bundle(model / "model.mlir")] == ["fc.tensor_impl.scale"]


def test_a_tagged_tensor_absent_from_extra_is_refused(tmp_path):
    model = _bundle(tmp_path, with_scale=False)
    with pytest.raises(qinner.QinnerError, match="absent from extra.npz"):
        c_runtime.generate(model, tmp_path / "generated", model / "inputs.npz")
