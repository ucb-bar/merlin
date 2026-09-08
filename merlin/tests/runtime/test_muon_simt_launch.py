"""Fail-closed tests for the evaluator-owned Radiance SIMT launch wrapper."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.runtime.backends.base import get_backend
from merlin.targetgen.contract.schemas import ContractViolation, validate_command_buffer


@pytest.fixture(scope="module")
def mh():
    import importlib

    get_backend("muon")
    return importlib.import_module("merlin._oot_backends.muon.muon_harness")


def _model(*, xlen=32, opcode=11, funct3=0):
    return SimpleNamespace(
        target="radiance",
        runtime_abi={"xlen": xlen},
        base_isa_family=lambda: f"riscv{xlen}",
        sfu_op=lambda role: {"opcode": opcode, "funct3": funct3},
    )


def _facts():
    return {"facts": {"simt": {
        "cores": 1,
        "warps_per_core": 8,
        "lanes_per_warp": 16,
    }}}


def _resources():
    return {"cores": 1, "warps_per_core": 8, "lanes_per_warp": 16}


def _cb():
    return {
        "abi_version": "0.1",
        "target": "radiance",
        "backend": "radiance-xdsl",
        "tensors": {
            "A": {"shape": [2, 2], "dtype": "f32", "role": "input"},
            "W": {"shape": [2, 2], "dtype": "f32", "role": "weight"},
            "Y": {"shape": [2, 2], "dtype": "f32", "role": "output"},
        },
        "commands": [{"opcode": "MATMUL", "operands": {
            "lhs": "A", "rhs": "W", "dst": "Y"}}],
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": "A", "access": "read"},
                {"tensor": "W", "access": "read"},
                {"tensor": "Y", "access": "write"},
            ],
            "outputs": ["Y"],
            "launch": {"kind": "simt_single_warp"},
        },
        "resources": _resources(),
    }


def test_launch_contract_is_schema_valid():
    validate_command_buffer(_cb())


@pytest.mark.parametrize("launch", [
    {"kind": "simt_single_warp", "lanes": 16},
    {"kind": "simt_single_warp", "opcode": 11},
    {"kind": "arbitrary_asm"},
    {},
])
def test_launch_contract_rejects_submission_supplied_machine_fields(launch):
    cb = _cb()
    cb["kernel_abi"]["launch"] = launch
    with pytest.raises(ContractViolation):
        validate_command_buffer(cb)


def test_wrapper_publishes_state_before_tmc_and_restores_manager(mh, monkeypatch):
    import merlin.targetgen.rtl.facts as rtl_facts

    monkeypatch.setattr(rtl_facts, "load_facts", lambda target: _facts())
    source = mh._external_kernel_launch_wrapper(
        {"kind": "simt_single_warp"}, _resources(), kernel_symbol="radiance_kernel",
        ptrs="float*, const float*, float*", argument_count=3, model=_model())

    first_tmc = source.index(".insn r 11, 0, 0, x0, t1, x0")
    return_address = source.index("la ra, 1f")
    tail = source.index("tail radiance_kernel")
    epilogue = source.index('"1:\\n"')
    second_tmc = source.index(".insn r 11, 0, 0, x0, t1, x0", first_tmc + 1)
    assert source.index("sw a0, 0(t0)") < first_tmc
    assert source.index("sw a1, 4(t0)") < first_tmc
    assert source.index("sw a2, 8(t0)") < first_tmc
    assert source.index("sw ra, 12(t0)") < first_tmc
    assert source.index("fence rw, rw") < first_tmc
    assert source.index("li t1, 65535") < first_tmc
    assert first_tmc < source.index("lw a0, 0(t0)") < return_address < tail < epilogue < second_tmc
    assert second_tmc < source.index("lw ra, 12(t0)") < source.index("ret")


def test_wrapper_refuses_resource_facts_mismatch(mh, monkeypatch):
    import merlin.targetgen.rtl.facts as rtl_facts

    monkeypatch.setattr(rtl_facts, "load_facts", lambda target: _facts())
    resources = _resources()
    resources["lanes_per_warp"] = 32
    with pytest.raises(ValueError, match="disagree with target RTL facts"):
        mh._external_kernel_launch_wrapper(
            {"kind": "simt_single_warp"}, resources, kernel_symbol="radiance_kernel",
            ptrs="float*", argument_count=1, model=_model())


def test_wrapper_refuses_unrepresentable_or_unsupported_abi(mh, monkeypatch):
    import merlin.targetgen.rtl.facts as rtl_facts

    monkeypatch.setattr(rtl_facts, "load_facts", lambda target: _facts())
    with pytest.raises(ValueError, match="1..8"):
        mh._external_kernel_launch_wrapper(
            {"kind": "simt_single_warp"}, _resources(), kernel_symbol="radiance_kernel",
            ptrs=", ".join(["float*"] * 9), argument_count=9, model=_model())
    with pytest.raises(ValueError, match="invalid derived TMC encoding"):
        mh._external_kernel_launch_wrapper(
            {"kind": "simt_single_warp"}, _resources(), kernel_symbol="radiance_kernel",
            ptrs="float*", argument_count=1, model=_model(opcode=128))
