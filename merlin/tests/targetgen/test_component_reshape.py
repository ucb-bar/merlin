"""A component-program reshape moves elements under a new rank-two extent and computes nothing."""

from __future__ import annotations

import pytest

from merlin.perf.component_source_demand import SourceDemandLimits, _metadata
from merlin.targetgen.component_program import SCHEMA_V2, analyze, render


def _program(shape, *, source=(2, 6)):
    return {
        "schema": SCHEMA_V2,
        "inputs": [{"name": "X", "role": "input", "shape": list(source), "dtype": "operand"}],
        "nodes": [{"name": "Y", "op": "reshape", "inputs": ["X"], "shape": shape}],
        "outputs": [{"name": "Y0", "value": "Y"}],
    }


def test_reshape_keeps_the_element_type_under_the_declared_extent():
    typed = analyze(_program([3, 4]), operand_dtype="i8", accumulator_dtype="i32")
    assert typed["outputs"][0]["shape"] == [3, 4] and typed["outputs"][0]["dtype"] == "i8"
    _, source = render(_program([3, 4]), operand_dtype="i8", accumulator_dtype="i32")
    assert "tensor.collapse_shape %X [[0, 1]] : tensor<2x6xi8> into tensor<12xi8>" in source
    assert "output_shape [3, 4] : tensor<12xi8> into tensor<3x4xi8>" in source
    assert "tensor.empty" not in source


@pytest.mark.parametrize("shape", [[5, 2], [12], [3, 0], [3, "4"]])
def test_reshape_refuses_a_count_or_rank_change(shape):
    with pytest.raises(ValueError, match="reshape"):
        analyze(_program(shape), operand_dtype="i8", accumulator_dtype="i32")


def test_reshape_is_not_a_v1_operation():
    program = _program([3, 4])
    del program["schema"]
    with pytest.raises(ValueError, match="unsupported component node"):
        analyze(program, operand_dtype="i8", accumulator_dtype="i32")


def test_source_demand_admits_the_reshape_extent():
    limits = SourceDemandLimits(
        max_inputs=4, max_nodes=4, max_outputs=4, max_scalar_bits=64, max_extent_bits=16, max_payload_bits=256
    )
    _metadata(_program([3, 4]), "i8", "i32", limits)
    with pytest.raises(ValueError, match="rank-two"):
        _metadata(_program([12]), "i8", "i32", limits)
