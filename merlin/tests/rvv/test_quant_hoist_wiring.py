"""Reachability tests for invariant-weight preprocessing.

Registration alone is not reachability: this lever was accepted by the feature registry while the
whole-model preparation seam never forwarded it to the pass that actually rewrites the IR.
"""
from __future__ import annotations

import inspect
import json
import struct

import numpy as np

from merlin.llvmlower import c_runtime, quant_hoist
from merlin.runtime.backends import zephyr_model


class _Seq:
    def __init__(self, data):
        self.data = data


class _Named:
    def __init__(self, name):
        self.name = name


class _Block:
    def __init__(self, names):
        self.ops = [_Named(name) for name in names]


class _Region:
    def __init__(self, names):
        self.blocks = [_Block(names)]


class _Generic:
    def __init__(self, body):
        self.name = "linalg.generic"
        self.properties = {"iterator_types": _Seq(["parallel"])}
        self.regions = [_Region([*body, "linalg.yield"])]
        self.operands = [object(), object()]


def test_quant_hoist_recognizes_the_current_guarded_scale_and_inverse_multiply():
    """The consumer must track both canonical quant formulas the producer legitimately emits."""
    guarded_scale = _Generic([
        "arith.divf", "arith.maximumf", "arith.cmpf", "arith.select",
    ])
    inverse_multiply = _Generic([
        "arith.mulf", "math.roundeven", "arith.minimumf", "arith.maximumf", "arith.fptosi",
    ])

    assert quant_hoist._is_scale(guarded_scale)
    assert quant_hoist._quantize_formula(inverse_multiply) == "multiply_inverse"


def test_prepare_for_lowering_forwards_quant_hoist_and_the_bundle_authority():
    src = inspect.getsource(zephyr_model.prepare_for_lowering)

    assert "hoist_weight_invariant_quantize=" in src
    assert "from ...llvmlower.quant_hoist import FEATURE" in src
    assert "_HOIST_WEIGHT_INVARIANT_QUANTIZE in _closed" in src
    assert "bundle_dir=" in src


def test_c_runtime_appends_the_quant_hoist_plan_to_the_forward_abi(tmp_path):
    """Rewriting @forward without binding its new arguments would run on garbage bytes."""
    model, prepared, out = (tmp_path / "capture", tmp_path / "prepared",
                            tmp_path / "generated")
    model.mkdir()
    (model / "model.mlir").write_text("""builtin.module {
  func.func @forward(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    func.return %arg0 : tensor<4xf32>
  }
}
""")
    header = json.dumps({}).encode()
    (model / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (model / "weights.safetensors.manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "x"}}))
    (model / "input_order.json").write_text(json.dumps({"x": 0}))
    np.savez(model / "inputs.npz", in0=np.ones(4, np.float32))
    prepared.mkdir()
    lifted = np.array([3.0, 1.0], np.float32)
    quant_hoist.write_plan(prepared, [quant_hoist.HoistedArg("folded", (2,), "f32")])
    quant_hoist.write_values(prepared, {"folded": lifted})

    info = c_runtime.generate(model, out, model / "inputs.npz", prepared_dir=prepared)

    assert info["n_quant_hoist"] == 1
    rows = [line for line in (out / "model_gen.h").read_text().splitlines()
            if line.strip().startswith("{MERLIN_")]
    assert len(rows) == 3 and "MERLIN_WEIGHT" in rows[1]
    offset = int(rows[1].split(",")[1].strip().rstrip("L"))
    got = np.frombuffer((out / "weights.bin").read_bytes(), dtype=np.float32,
                        count=2, offset=offset)
    assert np.array_equal(got, lifted)
