"""Public contract tests for runner-owned whole-program harnessing.

The command stream describes accelerator work; ``kernel_abi`` describes only the pointer boundary of
the submitted kernel.  This keeps scalar work in the submitted ELF and keeps all computation out of
the benchmark harness.
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends import base as bk
from merlin.runtime.commandbuffer import validate_command_buffer
from merlin.targetgen.contract import schemas


gem = bk.get_backend("gemmini")
CodegenError = gem.gemmini_codegen.CodegenError


def _whole_program_cb() -> dict:
    return {
        "abi_version": "0.1",
        "target": "gemmini",
        "backend": "mlir_oot_xdsl_gemmini",
        "tensors": {
            "scale": {"shape": [32], "dtype": "f32", "role": "input"},
            "W0": {"shape": [32, 32], "dtype": "i8", "role": "weight"},
            "W1": {"shape": [32, 32], "dtype": "i8", "role": "weight"},
            "A0": {"shape": [16, 32], "dtype": "i8", "role": "input"},
            "mid_mesh": {"shape": [16, 32], "dtype": "i8", "role": "intermediate"},
            "mid_host": {"shape": [16, 32], "dtype": "i8", "role": "intermediate"},
            "Y0": {"shape": [16, 32], "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W0", "dst": "W0_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT",
             "operands": {"lhs": "A0", "rhs": "W0_res", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "mid_mesh"},
             "attributes": {"epilogue": [], "output_dtype": "i8"}},
            {"opcode": "RES_PACK", "operands": {"src": "W1", "dst": "W1_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT",
             "operands": {"lhs": "mid_host", "rhs": "W1_res", "dst": "acc1"}},
            {"opcode": "COMMIT", "operands": {"src": "acc1", "dst": "Y0"},
             "attributes": {"epilogue": [], "output_dtype": "i8"}},
        ],
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": "scale", "access": "read"},
                {"tensor": "W0", "access": "read"},
                {"tensor": "W1", "access": "read"},
                {"tensor": "A0", "access": "read"},
                {"tensor": "mid_mesh", "access": "write"},
                {"tensor": "mid_host", "access": "write"},
                {"tensor": "Y0", "access": "write"},
            ],
            "outputs": ["Y0"],
        },
    }


def test_whole_program_abi_runs_only_the_submitted_kernel_in_one_warm_cycle_window(monkeypatch):
    monkeypatch.setenv("MERLIN_CACHE_STATE", "warm")
    source = gem.render_harness(_whole_program_cb(), target="gemmini")

    call = ("gemmini_kernel((void*)T_scale, (void*)T_W0, (void*)T_W1, (void*)T_A0, "
            "(void*)T_mid_mesh, (void*)T_mid_host, (void*)T_Y0);")
    assert source.count(call) == 2
    assert "merlin: warmup completed outside the measured/counter window" in source
    assert 'printf("METRIC cycle_window_gemmini_region 1\\n")' in source
    assert 'printf("OUT Y0 16 32")' in source
    assert "layernorm" not in source.lower()


def test_whole_program_abi_refuses_an_external_tensor_missing_from_the_pointer_boundary():
    cb = _whole_program_cb()
    cb["kernel_abi"]["args"] = cb["kernel_abi"]["args"][1:]

    problems = validate_command_buffer(cb)

    assert any("scale" in problem and "kernel_abi.args" in problem for problem in problems)


def test_whole_program_abi_schema_refuses_an_undefined_access_mode():
    cb = _whole_program_cb()
    cb["kernel_abi"]["args"][0]["access"] = "execute"

    with pytest.raises(schemas.ContractViolation, match="kernel_abi/args/0/access"):
        schemas.validate_command_buffer(cb)


def test_contract_gate_refuses_a_whole_program_with_an_unbound_external_tensor():
    cb = _whole_program_cb()
    cb["kernel_abi"]["args"] = cb["kernel_abi"]["args"][1:]

    with pytest.raises(schemas.ContractViolation, match="scale"):
        schemas.validate_command_buffer(cb)


def test_contract_gate_refuses_duplicate_pointer_slots_for_one_tensor():
    cb = _whole_program_cb()
    cb["kernel_abi"]["args"].append({"tensor": "scale", "access": "readwrite"})

    with pytest.raises(schemas.ContractViolation, match="duplicate.*scale"):
        schemas.validate_command_buffer(cb)


def test_contract_gate_refuses_a_pointer_slot_without_a_declared_buffer():
    cb = _whole_program_cb()
    cb["kernel_abi"]["args"].append({"tensor": "secret", "access": "read"})

    with pytest.raises(schemas.ContractViolation, match="secret.*no declared tensor"):
        schemas.validate_command_buffer(cb)


def test_contract_gate_refuses_reading_a_buffer_as_the_reported_result():
    cb = _whole_program_cb()
    cb["kernel_abi"]["outputs"] = ["scale"]

    with pytest.raises(schemas.ContractViolation, match="scale.*write access"):
        schemas.validate_command_buffer(cb)


def test_whole_program_schema_can_name_internal_cross_lane_buffers():
    cb = _whole_program_cb()

    schemas.validate_command_buffer(cb)


def test_contract_gate_refuses_an_unbound_internal_cross_lane_buffer():
    cb = _whole_program_cb()
    cb["kernel_abi"]["args"] = [arg for arg in cb["kernel_abi"]["args"]
                                  if arg["tensor"] != "mid_host"]

    with pytest.raises(schemas.ContractViolation, match="mid_host"):
        schemas.validate_command_buffer(cb)


def test_contract_gate_refuses_reporting_an_intermediate_instead_of_the_model_output():
    cb = _whole_program_cb()
    cb["kernel_abi"]["outputs"] = ["mid_mesh"]

    with pytest.raises(schemas.ContractViolation, match="Y0.*mid_mesh"):
        schemas.validate_command_buffer(cb)


def test_contract_gate_refuses_a_model_input_that_the_kernel_does_not_read():
    cb = _whole_program_cb()
    cb["kernel_abi"]["args"][0]["access"] = "write"

    with pytest.raises(schemas.ContractViolation, match="scale.*input.*read"):
        schemas.validate_command_buffer(cb)


def test_host_operands_cannot_be_smuggled_through_residency_commands():
    cb = _whole_program_cb()
    cb.pop("kernel_abi")
    cb["commands"].insert(0, {
        "opcode": "RES_PACK",
        "operands": {"src": "scale", "dst": "scale_res"},
        "attributes": {"layout": "host_lane_operand"},
    })

    with pytest.raises(CodegenError, match="no matrix or bias consumer"):
        gem.render_harness(cb, target="gemmini")


def test_whole_program_contract_refuses_harness_derived_model_work():
    cb = _whole_program_cb()
    cb["params"] = {"im2col_recipes": [{"source": "A0", "target": "mid_host",
                                          "kh": 3, "kw": 3, "ci": 1}]}

    with pytest.raises(schemas.ContractViolation, match="im2col_recipes.*submitted kernel"):
        schemas.validate_command_buffer(cb)
