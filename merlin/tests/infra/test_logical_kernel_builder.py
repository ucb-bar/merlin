"""Source roles are resolved by name even when declaration/call order differs."""
import pytest

from merlin.llvmlower.host_builder import FnBuilder, PTR
from merlin.targetgen.contract.harness_render import LogicalAbi, HarnessRenderError


def abi():
    return LogicalAbi(2, "kernel", ("declared_whole_program_args",),
        ("logical_inputs_in_declaration_order", "logical_outputs_in_result_order"),
        frozenset({"COMMIT"}), 64, 90, None)


def buffer():
    return {"tensors": {
        "weights": {"shape": [4, 4], "dtype": "i8", "role": "weight"},
        "activations": {"shape": [4, 4], "dtype": "i8", "role": "input"},
        "result": {"shape": [4, 4], "dtype": "i32", "role": "output"}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "weights", "dst": "resident"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {
                "lhs": "activations", "rhs": "resident", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "result"},
                "attributes": {"output_dtype": "i32"}}]}


def test_default_source_order_is_not_matmul_semantic_order():
    builder, pointers = FnBuilder.for_logical_kernel(buffer(), abi=abi())
    assert list(pointers) == ["weights", "activations", "result"]
    assert pointers["activations"] is builder.entry.args[1]
    assert pointers["weights"] is builder.entry.args[0]
    assert all(value.type == PTR for value in pointers.values())


def test_explicit_original_whole_program_order_remains_distinct():
    cb = buffer()
    cb["kernel_abi"] = {"kind": "whole_program", "args": [
        {"tensor": "activations", "access": "read"},
        {"tensor": "weights", "access": "read"},
        {"tensor": "result", "access": "write"}], "outputs": ["result"]}
    builder, pointers = FnBuilder.for_logical_kernel(cb, abi=abi())
    assert list(pointers) == ["activations", "weights", "result"]
    assert pointers["activations"] is builder.entry.args[0]


def test_duplicate_declared_whole_program_pointer_refuses():
    cb = buffer()
    cb["kernel_abi"] = {"kind": "whole_program", "args": [
        {"tensor": "activations", "access": "read"},
        {"tensor": "activations", "access": "read"}], "outputs": []}
    with pytest.raises(HarnessRenderError, match="distinct"):
        FnBuilder.for_logical_kernel(cb, abi=abi())
