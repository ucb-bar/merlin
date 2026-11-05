"""The numpy kernel reference computes exactly what the integer reference engine computes."""

from __future__ import annotations

import numpy as np
import pytest

from merlin.model_check import kernel_reference as KR
from merlin.runtime import reference as REF
from merlin.targetgen.contract.interface_emit import parse_interface_mlir

TARGET = "gemmini"  # the target whose readout routes these epilogues
_BASE = {
    "kind": "op",
    "operand_dtype": "int8",
    "scale_granularity": "tensor",
    "cat": "layers",
    "label": "dev",
    "source_role": "model_derived",
    "source_reference": "test",
}
_ENTRIES = [
    {
        "name": "conv_bias_scale_relu",
        "op": "conv2d",
        "N": 8,
        "ci": 3,
        "Himg": 9,
        "Wimg": 7,
        "kh": 3,
        "kw": 3,
        "stride": [2, 1],
        "padding": [1, 1, 1, 1],
        "epilogue": ["bias_add", "acc_scale", "relu"],
        "acc_scale": 0.0137,
    },
    {"name": "matmul_scale", "op": "matmul", "M": 5, "K": 33, "N": 7, "epilogue": ["acc_scale"], "acc_scale": 0.21},
    {"name": "matmul_raw", "op": "matmul", "M": 4, "K": 50, "N": 6, "epilogue": []},
    {
        "name": "conv_scale_relu_pool",
        "op": "conv2d",
        "N": 16,
        "ci": 4,
        "Himg": 8,
        "Wimg": 8,
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "epilogue": ["acc_scale", "relu", "maxpool"],
        "acc_scale": 0.05,
        "pool_size": [3, 3],
        "pool_stride": [2, 2],
        "pool_padding": [1, 1, 1, 1],
        "pool_pad_value": 0,
    },
]


def _inputs(cb: dict, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = {}
    for name, spec in cb["tensors"].items():
        if spec.get("role") not in ("input", "weight", "bias"):
            continue
        shape = tuple(int(v) for v in spec["shape"])
        if spec.get("dtype") == "i32":
            values[name] = rng.integers(-5000, 5000, size=shape, dtype=np.int64)
        else:
            values[name] = rng.integers(-128, 128, size=shape, dtype=np.int64)
    return values


@pytest.mark.parametrize("entry", _ENTRIES, ids=lambda e: e["name"])
@pytest.mark.parametrize("seed", [0, 1])
def test_numpy_engine_equals_integer_reference(entry, seed):
    interface = KR.interface_for({**_BASE, **entry}, target=TARGET, operand_dtype="int8", accum_dtype="int32")
    cb = parse_interface_mlir(interface)
    inputs = _inputs(cb, seed)
    fast = KR.evaluate(cb, inputs)
    slow = REF.reference_outputs(cb, {k: v.tolist() for k, v in inputs.items()})
    assert set(fast) == set(slow) and fast
    for name in slow:
        assert np.array_equal(np.asarray(slow[name], dtype=np.int64).reshape(-1), fast[name].reshape(-1)), name


def test_acc_scale_rounds_ties_to_even_in_float32():
    x = np.array([5, 7, -5, -7, 3, 16777217], dtype=np.int64)
    got = KR._acc_scale(x, 0.5)  # 2.5 -> 2, 3.5 -> 4, -2.5 -> -2, -3.5 -> -4, 1.5 -> 2
    assert got[:5].tolist() == [2, 4, -2, -4, 2]
    assert got[5] == 8388608  # 16777217 is not representable in float32: it rounds before scaling


def test_call_outputs_maps_host_operands_by_role():
    entry = {**_BASE, **_ENTRIES[1]}
    interface = KR.interface_for(entry, target=TARGET, operand_dtype="int8", accum_dtype="int32")
    cb = parse_interface_mlir(interface)
    inputs = _inputs(cb, 3)
    expected = KR.evaluate(cb, inputs)["Y0"]
    lhs, weight = inputs["A0"], inputs["W"]
    out = np.zeros((5, 7), dtype=np.int8)
    got = KR.call_outputs(interface, [lhs, weight, out], roles=["input_0", "weight_0", "out_0"])
    assert len(got) == 1 and np.array_equal(got[0].astype(np.int64), expected.reshape(5, 7))
    with pytest.raises(KR.KernelReferenceError):
        KR.call_outputs(interface, [lhs, weight, out], roles=["input_0", "weight_0", "bias_0"])
