"""The emitted command buffer must preserve the input interface's logical tensor ABI."""
from __future__ import annotations

import pytest

from merlin.targetgen.capsule_common import validate_interface_tensor_dtypes
from merlin.targetgen.contract.schemas import ContractViolation


def _interface(*, src: str = "f8E4M3FN", dst: str = "f32") -> str:
    return f'''module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"}} {{
  %X = merlin_iface.tensor {{name = "X", role = "input"}} : tensor<4x8x{src}>
  %Y0 = merlin_iface.movement %X {{name = "Y0", semantic = "mvin_mvout", output_dtype = "{dst}"}} : (tensor<4x8x{src}>) -> tensor<4x8x{dst}>
}}'''


def _buffer(*, src: str = "fp8_e4m3", dst: str = "float32") -> dict:
    return {
        "tensors": {
            "X": {"shape": [4, 8], "dtype": src, "role": "input"},
            "Y0": {"shape": [4, 8], "dtype": dst, "role": "output"},
        }
    }


def test_interface_dtype_binding_accepts_registry_and_machine_aliases():
    validate_interface_tensor_dtypes(_buffer(), _interface())
    validate_interface_tensor_dtypes(
        _buffer(src="int8", dst="int32"), _interface(src="i8", dst="i32"))


def test_interface_dtype_binding_rejects_narrowed_output_despite_physical_hint():
    cb = _buffer(dst="bf16")
    cb["tensors"]["Y0"]["physical"] = {"logical_dtype": "f32"}

    with pytest.raises(ContractViolation, match="Y0.*f32.*bf16"):
        validate_interface_tensor_dtypes(cb, _interface())


def test_interface_dtype_binding_rejects_retyped_input():
    with pytest.raises(ContractViolation, match="X.*f8E4M3FN.*bf16"):
        validate_interface_tensor_dtypes(_buffer(src="bf16"), _interface())


def test_interface_dtype_binding_leaves_non_interface_frontends_alone():
    validate_interface_tensor_dtypes(
        _buffer(dst="bf16"),
        "module { func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> }",
    )
