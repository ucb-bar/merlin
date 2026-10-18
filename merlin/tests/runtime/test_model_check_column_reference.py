"""The fast kernel oracle preserves each column's independently rounded readout."""

import numpy as np
import pytest

from merlin.model_check import kernel_reference as KR
from merlin.runtime import reference as REF


def _case(kind, width, *, scalar=False):
    lhs = np.array([[3, -5, 7], [-4, 2, 9]], dtype=np.int64)
    rhs = (np.arange(3 * width).reshape(3, width) % 11 - 5).astype(np.int64)
    bias = np.arange(width, dtype=np.int64) * 13 - 17
    scales = np.full(width, 0.5) if scalar else np.resize([0.5, 0.125, 0.75, -0.25], width)
    attrs = {"epilogue": ["acc_scale", "relu"], "output_dtype": "i8"}
    attrs.update({"acc_scale": 0.5} if scalar else {"acc_scale_by_column": scales.tolist()})
    tensors = {
        "A": {"shape": [2, 3], "dtype": "i8", "role": "input"},
        "W": {"shape": [3, width], "dtype": "i8", "role": "weight"},
        "Y": {"shape": [2, width], "dtype": "i8", "role": "output"},
    }
    inputs = {"A": lhs, "W": rhs}
    expected_acc = lhs @ rhs
    inputs["B"] = bias
    tensors["B"] = {"shape": [width], "dtype": "i32", "role": "bias"}
    attrs["epilogue"].insert(0, "bias_add")
    expected_acc = expected_acc + bias
    commands = [{"opcode": "RES_PACK", "operands": {"src": "W", "dst": "Wr"}, "attributes": {"layout": "packed_rhs"}}]
    if kind == "conv":
        inputs["A"] = lhs.reshape(1, 1, 2, 3)
        tensors["A"]["shape"] = [1, 1, 2, 3]
        attrs.update(kernel=[1, 1, 3, width], stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1], layout="nhwc")
        commands.append(
            {"opcode": "CONV2D", "operands": {"ifm": "A", "weight": "Wr", "bias": "B", "dst": "Y"}, "attributes": attrs}
        )
    else:
        commands.extend(
            [
                {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "Wr", "dst": "acc"}},
                {"opcode": "COMMIT", "operands": {"src": "acc", "bias": "B", "dst": "Y"}, "attributes": attrs},
            ]
        )
    # The oracle under test never constructs this integer accumulator or expected output.
    rounded = np.rint(expected_acc.astype(np.float32) * scales.astype(np.float32))
    expected = np.clip(np.maximum(rounded, 0), -128, 127).astype(np.int64)
    return {"abi_version": "0.1", "target": "neutral", "tensors": tensors, "commands": commands}, inputs, expected


@pytest.mark.parametrize("kind", ["commit", "conv"])
@pytest.mark.parametrize("width", [2, 19, 27])
@pytest.mark.parametrize("scalar", [False, True])
def test_column_readout_matches_integer_engine_and_independent_accumulator(kind, width, scalar):
    cb, inputs, expected = _case(kind, width, scalar=scalar)
    actual = KR.evaluate(cb, inputs)["Y"]
    integer = REF.reference_outputs(cb, {k: v.tolist() for k, v in inputs.items()})["Y"]
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, np.asarray(integer))


@pytest.mark.parametrize("kind", ["commit", "conv"])
@pytest.mark.parametrize(
    "change",
    [
        {},
        {"acc_scale": 0.5, "acc_scale_by_column": [0.5, 0.25]},
        {"acc_scale_by_column": [0.5]},
        {"acc_scale_by_column": [True, 0.5]},
        {"acc_scale_by_column": [float("inf"), 0.5]},
    ],
)
def test_missing_ambiguous_or_malformed_readouts_refuse(kind, change):
    cb, inputs, _ = _case(kind, 2)
    attrs = cb["commands"][-1]["attributes"]
    attrs.pop("acc_scale_by_column", None)
    attrs.update(change)
    with pytest.raises(ValueError):
        KR.evaluate(cb, inputs)


def test_column_readout_keeps_float32_accumulator_conversion_before_multiply():
    values = np.array([[16777217, -16777217, 5, 7], [3, -5, -7, 17]], dtype=np.int64)
    attrs = {"acc_scale_by_column": [0.5, 0.125, 0.5, 0.5]}
    actual = KR._epilogue(values, ["acc_scale"], attrs, {}, {}, 0, "COMMIT")
    assert actual.tolist() == [[8388608, -2097152, 2, 4], [2, -1, -4, 8]]
