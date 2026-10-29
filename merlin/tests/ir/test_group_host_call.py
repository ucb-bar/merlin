"""A logical-ABI group call binds each operand to the interface role the device states for it.

A contraction stated with a stationary ACTIVATION (an activation-major matmul such as a patch-embedding
convolution) holds its right operand in the interface's weight role, so the call must pass that
per-inference tensor as ``weight_0``; labelled ``input_1`` the device shim declines the kernel and the
whole-model image does not link. Every other two-input group with no stored operand (a residual add)
passes both operands as inputs.
"""

from types import SimpleNamespace

from merlin.llvmlower.group_host_call import _logical_group_call
from merlin.xdsl_dialects.lowering import group_command as GC

_GROUP = SimpleNamespace(root=None, members=[], window_mean=None, window_pool=None)
_NOTHING_STORED = SimpleNamespace(stored_operand=None)


def _roles(entry) -> list[str]:
    left, right = object(), object()
    operands, constants, roles, why = _logical_group_call(_GROUP, _NOTHING_STORED, entry, (left, right), None, {}, "")
    assert why == ""
    assert operands == [left, right] and list(constants) == []
    return list(roles)


def test_a_stationary_activation_is_passed_in_the_weight_slot():
    assert _roles({"op": "matmul", GC.STATIONARY_KEY: GC.STATIONARY_ACTIVATION}) == ["input_0", "weight_0"]


def test_two_activations_of_any_other_group_are_both_inputs():
    assert _roles({"op": "add"}) == ["input_0", "input_1"]
    assert _roles({"op": "matmul"}) == ["input_0", "input_1"]


def test_proved_bias_is_passed_even_without_a_stored_weight():
    import numpy as np

    left, right = object(), object()
    entry = {"op": "matmul", GC.STATIONARY_KEY: GC.STATIONARY_ACTIVATION, "epilogue": ["bias_add"]}
    row = {"bias": {"array": "offsets"}}
    arrays = {"offsets": np.array([17, -91, 0], dtype=np.int32)}
    operands, constants, roles, why = _logical_group_call(
        _GROUP, _NOTHING_STORED, entry, (left, right), row, arrays, ""
    )
    assert why == "" and roles == ["input_0", "weight_0", "bias_0"]
    (constant,) = constants
    assert operands == [left, right, constant.results[0]]
    assert list(constant.value.get_values()) == [17, -91, 0]
    assert str(constant.results[0].type) == "tensor<3xi32>"
