"""The strict command-buffer schema admits every ABI-native whole operation.

These commands are public ABI operations, not target-private extensions.  The ABI document,
interface grammar, and execution engines already define them, so the validator must not reject a
well-formed buffer solely because its opcode enumeration drifted behind those definitions.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.contract import schemas


def _tensor(shape, *, role="input", dtype="i8"):
    return {"shape": shape, "dtype": dtype, "role": role}


def _cb(command, tensors):
    return {"abi_version": "0.1", "target": "gemmini", "commands": [command], "tensors": tensors}


def test_schema_accepts_abi_native_movement():
    schemas.validate_command_buffer(_cb(
        {"opcode": "MOVEMENT",
         "operands": {"src": "X", "dst": "Y0"},
         "attributes": {"semantic": "mvin_mvout", "output_dtype": "i32"}},
        {"X": _tensor([4, 4]), "Y0": _tensor([4, 4], role="output", dtype="i32")},
    ))


def test_schema_accepts_abi_native_conv2d():
    schemas.validate_command_buffer(_cb(
        {"opcode": "CONV2D",
         "operands": {"ifm": "X", "weight": "W_res", "dst": "Y0"},
         "attributes": {"kernel": [3, 3, 1, 2], "stride": [1, 1],
                        "padding": [0, 0, 0, 0], "dilation": [1, 1],
                        "layout": "nhwc", "epilogue": [], "output_dtype": "i32"}},
        {"X": _tensor([1, 4, 4, 1]), "W": _tensor([9, 2], role="weight"),
         "Y0": _tensor([4, 2], role="output", dtype="i32")},
    ))


def test_schema_accepts_abi_native_attention_pv():
    schemas.validate_command_buffer(_cb(
        {"opcode": "ATTENTION_PV",
         "operands": {"p": "P", "v": "V", "dst": "Y0"},
         "attributes": {"epilogue": [], "output_dtype": "i32"}},
        {"P": _tensor([2, 3]), "V": _tensor([3, 4]),
         "Y0": _tensor([2, 4], role="output", dtype="i32")},
    ))


def test_schema_remains_closed_to_undeclared_opcodes():
    with pytest.raises(schemas.ContractViolation, match="FROBNICATE.*is not one of"):
        schemas.validate_command_buffer(_cb(
            {"opcode": "FROBNICATE", "operands": {"src": "X", "dst": "Y0"}},
            {"X": _tensor([1]), "Y0": _tensor([1], role="output")},
        ))
