"""A session corpus is embedded as C literals, so a run that measures a few steps can cap it.

resnet50's 256-step, 154 MB `session_inputs.npz` becomes a 770 MB `model_io.h` and ~7 GB of RSS
at compile. Streams and the trajectory references must shrink TOGETHER, or step k grades against
reference k of a longer corpus.
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


def test_session_step_budget_truncates_streams_and_references_together(tmp_path):
    model = _bundle(tmp_path, steps=8)
    full = c_runtime.generate(model, tmp_path / "full", model / "inputs.npz")
    capped = c_runtime.generate(model, tmp_path / "capped", model / "inputs.npz",
                                max_session_steps=2)
    assert "MERLIN_SESSION_STEPS 8" in (tmp_path / "full" / "model_gen.h").read_text()
    assert "MERLIN_SESSION_STEPS 2" in (tmp_path / "capped" / "model_gen.h").read_text()
    assert full["has_session_correctness"] and capped["has_session_correctness"]

    def _literals(path, name):
        line = next(ln for ln in path.read_text().splitlines() if name in ln)
        return line.count(",") + 1

    io_full = tmp_path / "full" / "model_io.h"
    io_capped = tmp_path / "capped" / "model_io.h"
    # streams AND the trajectory reference shrink by the same factor, so step k still meets
    # reference k; a stream truncated on its own would grade against the wrong step.
    assert _literals(io_full, "merlin_stream_0[]") == 32
    assert _literals(io_capped, "merlin_stream_0[]") == 8
    assert _literals(io_full, "merlin_correctness_golden[]") == 32
    assert _literals(io_capped, "merlin_correctness_golden[]") == 8
