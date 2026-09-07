"""Scalar-memory semantics come from instruction behavior, never mnemonic tables."""
from __future__ import annotations

from merlin.targetgen.oracle_helpers.isa_introspect import scalar_memory_semantics


class _ScalarReg:
    reg_name = "scalar register"


class _Imm12(int):
    pass


class _LoadPattern:
    rd: _ScalarReg
    rs1: _ScalarReg
    imm: _Imm12


class _StorePattern:
    rs1: _ScalarReg
    rs2: _ScalarReg
    imm: _Imm12


def _sign_extend(value, bits):
    return value


def _to_bytes(value, width):
    return value


class _OddlyNamedRead(_LoadPattern):
    def exec(self, state):
        offset = _sign_extend(self.imm, 12)
        value = state.read_local_mem(state.read_xrf(self.rs1), offset, 1)
        state.write_xrf(self.rd, value)


class _OddlyNamedWrite(_StorePattern):
    def exec(self, state):
        offset = _sign_extend(self.imm, 12)
        state.write_local_mem(state.read_xrf(self.rs1), offset, _to_bytes(self.rs2, 2))


def test_load_semantics_are_derived_from_effect_and_typed_operands():
    got = scalar_memory_semantics(
        _OddlyNamedRead,
        _LoadPattern,
        {"rd": list(range(5)), "rs1": list(range(5)), "imm": list(range(12))},
    )

    assert got == {
        "direction": "load",
        "address_space": "local_mem",
        "width_bytes": 1,
        "address_unit_bytes": 1,
        "addressing": {
            "mode": "base_plus_immediate",
            "base_operand": "rs1",
            "offset_operand": "imm",
            "offset_bits": 12,
            "offset_signed": True,
            "offset_scale": 1,
        },
        "effect_method": "read_local_mem",
    }


def test_store_semantics_do_not_depend_on_a_store_like_class_name():
    got = scalar_memory_semantics(
        _OddlyNamedWrite,
        _StorePattern,
        {"rs1": list(range(5)), "rs2": list(range(5)), "imm": list(range(12))},
    )

    assert got["direction"] == "store"
    assert got["address_space"] == "local_mem"
    assert got["width_bytes"] == 2
    assert got["addressing"]["base_operand"] == "rs1"
    assert got["addressing"]["offset_operand"] == "imm"


class _TensorReg:
    reg_name = "matrix register"


class _TensorMovePattern:
    vd: _TensorReg
    rs1: _ScalarReg
    imm: _Imm12


class _TensorRead(_TensorMovePattern):
    def exec(self, state):
        value = state.read_local_mem(state.read_xrf(self.rs1), self.imm, 32)
        state.write_tensor(self.vd, value)


def test_tensor_memory_operation_is_not_misreported_as_scalar_memory():
    assert scalar_memory_semantics(
        _TensorRead,
        _TensorMovePattern,
        {"vd": list(range(6)), "rs1": list(range(5)), "imm": list(range(12))},
    ) is None
