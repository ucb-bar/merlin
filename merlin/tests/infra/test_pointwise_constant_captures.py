"""Exact integer constants may be copied into an explicitly admitted scalar body."""

import pytest
from test_pointwise_source import source_program, source_text
from xdsl.dialects import builtin

from merlin.compile.scheduling.fusion_semantics import ScalarBudgetExceeded, UnsupportedElementwise
from merlin.compile.scheduling.pointwise_source import read_pointwise_source


def read(operation, **controls):
    return read_pointwise_source(
        operation,
        max_elements=1024,
        max_operations=controls.pop("max_operations", 128),
        max_ir_bytes=65536,
        **controls,
    )


def captured_source():
    text = source_text()
    constants = "%low=arith.constant -128:i32\n %high=arith.constant 127:i32\n "
    text = text.replace(constants, "")
    return source_program(text.replace('%out="linalg.generic"', constants + '%out="linalg.generic"'))


def test_explicit_constant_capture_has_identical_complete_scalar_graph():
    _, internal = source_program()
    module, external = captured_source()
    before = str(module)
    assert read(internal).graph.expression(("x", "bias")) == read(
        external, allow_constant_captures=True
    ).graph.expression(("x", "bias"))
    assert str(module) == before
    with pytest.raises(UnsupportedElementwise, match="captured state"):
        read(external)


@pytest.mark.parametrize("mutation", ["computed", "metadata", "initializer"])
def test_constant_permission_does_not_admit_other_captured_state(mutation):
    module, operation = captured_source()
    if mutation == "computed":
        text = source_text().replace("%high=arith.constant 127:i32", "%high=arith.addi %low,%low:i32")
        text = text.replace("%low=arith.constant -128:i32\n %high=arith.addi %low,%low:i32\n ", "")
        _, operation = source_program(
            text.replace(
                '%out="linalg.generic"',
                '%low=arith.constant -128:i32\n %high=arith.addi %low,%low:i32\n %out="linalg.generic"',
            )
        )
    elif mutation == "metadata":
        next(op for op in module.walk() if op.name == "arith.constant").attributes["unknown"] = builtin.UnitAttr()
    else:
        body = operation.body.block
        body.last_op.operands = (body.args[-1],)
    with pytest.raises(UnsupportedElementwise, match="unsupported scalar state|initializer"):
        read(operation, allow_constant_captures=True)


def test_cloned_constant_operations_still_count_against_scalar_budget():
    _, operation = captured_source()
    with pytest.raises(ScalarBudgetExceeded, match="operation budget"):
        read(operation, allow_constant_captures=True, max_operations=4)


def test_constant_admission_requires_an_explicit_boolean():
    _, operation = captured_source()
    with pytest.raises(ValueError, match="Boolean"):
        read(operation, allow_constant_captures=1)
