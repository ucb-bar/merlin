"""Each weight is stored once: device-layout copies live in the blob, originals only they replaced leave it.

`blob_weights.WeightArgs` binds a laid-out weight to a trailing `@forward` argument (one per distinct
tensor) and records which original arguments lost their last reader; `c_runtime.generate` appends the
bytes and the rows after qinner's, and drops the dead originals' bytes from the blob. These pin the
plan, the compaction and the argument table the C runtime is handed.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.llvmlower import blob_weights as BW
from merlin.llvmlower import c_runtime

_MODULE = """builtin.module {
  func.func @forward(%x: tensor<2x4xi8>, %w: tensor<4x3xi8>, %v: tensor<3xf32>) -> tensor<4x3xi8> {
    %e = tensor.empty() : tensor<3x4xi8>
    %t = linalg.transpose ins(%w : tensor<4x3xi8>) outs(%e : tensor<3x4xi8>) permutation = [1, 0]
    %e2 = tensor.empty() : tensor<4x3xi8>
    %u = linalg.transpose ins(%t : tensor<3x4xi8>) outs(%e2 : tensor<4x3xi8>) permutation = [1, 0]
    func.return %u : tensor<4x3xi8>
  }
}
"""

W = np.arange(12, dtype=np.int8).reshape(4, 3)
V = np.array([1.0, 2.0, 3.0], np.float32)


def _rebind(module, binder):
    """What the group rewrite does: the transpose's consumer reads a bound copy, the transpose goes."""
    from xdsl.dialects.builtin import IntegerType

    func = next(op for op in module.walk() if op.name == "func.func")
    transposes = [op for op in func.body.blocks[0].ops if op.name == "linalg.transpose"]
    laid = binder(W.T.copy(), IntegerType(8))
    assert binder(W.T.copy(), IntegerType(8)) is laid, "identical bytes share one argument"
    transposes[1].operands[0] = laid
    transposes[0].detach()
    transposes[0].erase()
    return func


def test_a_bound_weight_is_a_trailing_argument_and_the_replaced_original_is_dead(tmp_path) -> None:
    module = mq.parse(_MODULE)
    binder = BW.WeightArgs(module)
    func = _rebind(module, binder)
    binder.finish()
    assert len(func.body.blocks[0].args) == 4
    assert "tensor<3x4xi8>" in str(func.function_type.inputs.data[3])
    assert binder.dead_args == (1,), "only the argument the rewrite stopped reading is dropped"
    binder.write(tmp_path)
    plan = BW.read(tmp_path)
    assert plan.dead_args == frozenset({1})
    [(arg, data)] = list(plan.arrays())
    assert arg == {"key": "w0", "shape": [3, 4], "dtype": "i8"} and data == W.T.tobytes()


def test_a_weight_that_does_not_fit_its_element_type_is_refused() -> None:
    from xdsl.dialects.builtin import IntegerType

    binder = BW.WeightArgs(mq.parse(_MODULE))
    with pytest.raises(ValueError, match="does not fit"):
        binder(np.array([[300]]), IntegerType(8))


def test_compaction_drops_only_tensors_no_live_argument_reads() -> None:
    header = {
        "a": {"dtype": "I8", "shape": [3], "data_offsets": [0, 3]},
        "b": {"dtype": "I8", "shape": [5], "data_offsets": [3, 8]},
        "c": {"dtype": "I8", "shape": [2], "data_offsets": [8, 10]},
    }
    payload = bytes(range(10))
    manifest = {
        "0": {"kind": "param", "weight": "a"},
        "1": {"kind": "param", "weight": "b"},
        "2": {"kind": "param", "weight": "c"},
        "3": {"kind": "param", "weight": "c"},
    }
    out, blob = BW.compact(header, payload, manifest, frozenset({1, 2}))
    assert out["b"]["data_offsets"] == [0, 0], "read only by a dead argument: dropped"
    assert blob[:3] == payload[0:3]
    begin, end = out["c"]["data_offsets"]
    assert begin % BW.ALIGNMENT == 0 and blob[begin:end] == payload[8:10], "a tied tensor a live argument reads stays"
    assert len(blob) == BW.ALIGNMENT + 2


def _capture(tmp_path):
    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text(_MODULE, encoding="utf-8")
    header = {
        "w": {"dtype": "I8", "shape": [4, 3], "data_offsets": [0, 12]},
        "v": {"dtype": "F32", "shape": [3], "data_offsets": [12, 24]},
    }
    raw = json.dumps(header).encode()
    (model / "weights.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + W.tobytes() + V.tobytes())
    manifest = {
        "0": {"kind": "input", "name": "x"},
        "1": {"kind": "param", "weight": "w"},
        "2": {"kind": "param", "weight": "v"},
    }
    (model / "weights.safetensors.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (model / "input_order.json").write_text(json.dumps({"x": 0}), encoding="utf-8")
    np.savez(model / "inputs.npz", in0=np.ones((2, 4), np.int8))
    return model


def test_the_argument_table_carries_the_bound_weight_and_the_blob_stores_each_weight_once(tmp_path) -> None:
    model = _capture(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    module = mq.parse(_MODULE)
    binder = BW.WeightArgs(module)
    _rebind(module, binder)
    binder.finish()
    binder.write(prepared)
    info = c_runtime.generate(model, tmp_path / "cgen", model / "inputs.npz", prepared_dir=prepared)
    blob = (tmp_path / "cgen" / "weights.bin").read_bytes()
    assert W.tobytes() not in blob, "the original only the device read is gone"
    assert V.tobytes() in blob and W.T.tobytes() in blob
    gen = (tmp_path / "cgen" / "model_gen.h").read_text()
    assert "#define MERLIN_N_ARGS 5" in gen  # x, w (dead), v, the bound copy, the output
    assert "{MERLIN_WEIGHT, 0L, 2, {4,3}, 1}" in gen, "the dead argument's descriptor points at the base"
    offset = blob.index(W.T.tobytes())
    assert f"{{MERLIN_WEIGHT, {offset}L, 2, {{3,4}}, 1}}" in gen
    assert info["weights_bytes"] == len(blob)


def test_without_a_plan_the_table_and_blob_are_unchanged(tmp_path) -> None:
    model = _capture(tmp_path)
    c_runtime.generate(model, tmp_path / "cgen", model / "inputs.npz", prepared_dir=tmp_path)
    assert (tmp_path / "cgen" / "weights.bin").read_bytes() == W.tobytes() + V.tobytes()


def test_the_scope_requests_blob_weights_only_inside_it() -> None:
    assert BW.is_requested() is False
    with BW.requested(True):
        assert BW.is_requested() is True
    assert BW.is_requested() is False
    with pytest.raises(TypeError):
        with BW.requested(1):  # type: ignore[arg-type]
            pass


_CHAIN = """builtin.module {
  func.func @forward(%x: tensor<4xf32>, %w: tensor<4xf32>) -> tensor<4xf32> {
    %e = tensor.empty() : tensor<4xf32>
    %d = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], \
iterator_types = ["parallel"]} ins(%w : tensor<4xf32>) outs(%e : tensor<4xf32>) {
    ^bb0(%a: f32, %o: f32):
      %n = arith.negf %a : f32
      linalg.yield %n : f32
    } -> tensor<4xf32>
    func.return %d : tensor<4xf32>
  }
}
"""


def test_an_original_read_only_by_a_dead_value_chain_is_dropped() -> None:
    """The rewrite leaves a replaced weight's dequantize chain for canonicalization; it reads nothing live."""
    from xdsl.dialects.builtin import Float32Type

    module = mq.parse(_CHAIN)
    binder = BW.WeightArgs(module)
    bound = binder(np.arange(4, dtype=np.float32), Float32Type())
    ret = next(op for op in module.walk() if op.name == "func.return")
    ret.operands[0] = bound
    binder.finish()
    assert binder.dead_args == (1,), "the generic still reads %w, but nothing reads the generic"
