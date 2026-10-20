"""Explicit per-column readouts retain float32 conversion and ties-to-even rounding."""

import numpy as np
import pytest

from merlin.runtime.commandbuffer import apply_declared_acc_scale
from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate
from merlin.runtime.tensor import Tensor


@pytest.mark.parametrize("width", [3, 19, 32])
def test_column_scales_match_an_independent_float32_reference(width):
    values = np.resize(np.array([-16777217, -17, -5, -3, -1, 0, 1, 3, 5, 17, 16777217]), (4, width))
    scales = np.resize(np.array([0.5, 0.125, 1.00000006, 0.03125], dtype=np.float64), width)
    expected = np.rint(values.astype(np.float32) * scales.astype(np.float32)).astype(np.int64)
    source = Tensor((4, width), values.reshape(-1).tolist(), "i32")
    result = apply_declared_acc_scale(source, {"acc_scale_by_column": scales.tolist()}, op="test")
    assert result.shape == source.shape and result.dtype == "i32"
    assert result.data == expected.reshape(-1).tolist()
    assert result.to_i8().data == np.clip(expected, -128, 127).reshape(-1).tolist()


def test_uniform_column_scales_preserve_the_scalar_result():
    value = Tensor((2, 5), [-511, -5, -3, -1, 0, 1, 3, 5, 127, 511], "i32")
    assert apply_declared_acc_scale(value, {"acc_scale_by_column": [0.5] * 5}, op="test").data == (
        apply_declared_acc_scale(value, {"acc_scale": 0.5}, op="test").data
    )


@pytest.mark.parametrize("attrs", [
    {}, {"acc_scale": 1.0, "acc_scale_by_column": [1.0, 2.0]},
    {"acc_scale_by_column": []}, {"acc_scale_by_column": 1.0},
    {"acc_scale_by_column": [1.0]}, {"acc_scale_by_column": [True, 1.0]},
    {"acc_scale_by_column": [float("nan"), 1.0]},
    {"acc_scale_by_column": [float("inf"), 1.0]},
    {"acc_scale_by_column": [1e39, 1.0]}, {"acc_scale_by_column": ["0.5", 1.0]},
    {"acc_scale_by_column": [10**1000, 1.0]},
])
def test_incomplete_ambiguous_or_nonfinite_column_scales_refuse(attrs):
    with pytest.raises(ValueError, match="acc_scale"):
        apply_declared_acc_scale(Tensor((1, 2), [3, 5], "i32"), attrs, op="test")


def _commit_case():
    width = 19
    lhs = np.array([[3, -5, 7], [-4, 2, 9]], dtype=np.int32)
    rhs = (np.arange(3 * width).reshape(3, width) % 7 - 3).astype(np.int32)
    scales = np.resize(np.array([0.5, 0.125, 0.75], dtype=np.float32), width)
    attrs = {"epilogue": ["acc_scale"], "output_dtype": "i8", "acc_scale_by_column": scales.tolist()}
    cb = {
        "abi_version": "0.1", "target": "t",
        "tensors": {
            "A": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "W": {"shape": [3, width], "dtype": "i8", "role": "weight"},
            "Y": {"shape": [2, width], "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "Wr"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "Wr", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y"}, "attributes": attrs},
        ],
    }
    inputs = {"A": lhs.tolist(), "W": rhs.tolist()}
    expected = np.clip(np.rint((lhs @ rhs).astype(np.float32) * scales), -128, 127).astype(np.int32).tolist()
    return cb, inputs, expected


def _conv_case():
    width = 19
    image = (np.arange(12).reshape(1, 2, 3, 2) % 9 - 4).astype(np.int32)
    weight = (np.arange(2 * width).reshape(2, width) % 7 - 3).astype(np.int32)
    scales = np.resize(np.array([0.5, 0.125, 0.75], dtype=np.float32), width)
    cb = {
        "abi_version": "0.1", "target": "t",
        "tensors": {
            "A": {"shape": [1, 2, 3, 2], "dtype": "i8", "role": "input"},
            "W": {"shape": [2, width], "dtype": "i8", "role": "weight"},
            "Y": {"shape": [6, width], "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "Wr"},
             "attributes": {"layout": "packed_conv_rhs"}},
            {"opcode": "CONV2D", "operands": {"ifm": "A", "weight": "Wr", "dst": "Y"},
             "attributes": {"kernel": [1, 1, 2, width], "stride": [1, 1], "padding": [0, 0, 0, 0],
                            "dilation": [1, 1], "layout": "nhwc", "epilogue": ["acc_scale"],
                            "output_dtype": "i8", "acc_scale_by_column": scales.tolist()}},
        ],
    }
    expected = np.clip(np.rint((image.reshape(6, 2) @ weight).astype(np.float32) * scales), -128, 127)
    return cb, {"A": image.tolist(), "W": weight.tolist()}, expected.astype(np.int32).tolist()


@pytest.mark.parametrize("case", [_commit_case, _conv_case])
def test_readout_applies_each_output_columns_own_declared_multiplier(case):
    cb, inputs, expected = case()
    assert reference_outputs(cb, inputs=inputs)["Y"] == expected
    assert simulate(cb, inputs=inputs)["outputs"]["Y"] == expected


@pytest.mark.parametrize("case", [_commit_case, _conv_case])
def test_column_readout_round_trips_as_typed_float32_attributes(case):
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir
    from merlin.targetgen.contract.schemas import validate_command_buffer

    cb, inputs, expected = case()
    validate_command_buffer(cb)
    text = emit_interface_mlir(cb)
    assert "acc_scale_by_column = [0.5 : f32, 0.125 : f32, 0.75 : f32" in text
    parsed = parse_interface_mlir(text)
    assert parsed["commands"][-1]["attributes"]["acc_scale_by_column"] == cb["commands"][-1]["attributes"][
        "acc_scale_by_column"
    ]
    assert reference_outputs(parsed, inputs=inputs)["Y"] == expected


@pytest.mark.parametrize("change", [
    {"epilogue": []}, {"acc_scale": 1.0}, {"acc_scale_by_column": [0.5]},
])
def test_command_validation_refuses_unused_ambiguous_or_wrong_width_column_scales(change):
    from merlin.runtime.commandbuffer import validate_command_buffer
    from merlin.targetgen.contract import schemas

    cb, _, _ = _commit_case()
    cb["commands"][-1]["attributes"].update(change)
    assert any("acc_scale" in problem for problem in validate_command_buffer(cb))
    with pytest.raises(schemas.ContractViolation, match="acc_scale_by_column"):
        schemas.validate_command_buffer(cb)


@pytest.mark.parametrize("case", [_commit_case, _conv_case])
def test_native_reference_instrument_applies_column_scales(case, tmp_path):
    import ctypes
    import hashlib
    import shutil
    import subprocess

    from merlin.targetgen.contract.harness_render import logical_abi, logical_interface
    from merlin.targetgen.contract.reference_kernel import render_reference_kernel

    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable for the independent reference instrument")
    cb, inputs, expected = case()
    text = render_reference_kernel(cb, symbol="column_readout_reference")
    identity = hashlib.sha256(text.encode()).hexdigest()[:16]
    source, library = tmp_path / f"reference-{identity}.c", tmp_path / f"reference-{identity}.so"
    source.write_text(text)
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", str(source), "-o", str(library), "-lm"], check=True,
                   capture_output=True)
    buffers = logical_interface(cb, logical_abi())
    arrays = {name: np.ascontiguousarray(value, dtype=np.int8) for name, value in inputs.items()}
    arrays["Y"] = np.full(cb["tensors"]["Y"]["shape"], -99, dtype=np.int8)
    owner = ctypes.CDLL(str(library))
    function = owner.column_readout_reference
    function.argtypes = [ctypes.c_void_p] * len(buffers)
    function.restype = None
    function(*(arrays[buffer.name].ctypes.data for buffer in buffers))
    assert arrays["Y"].tolist() == expected
