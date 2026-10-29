"""Logical result names survive the ordinary public source-only compiler input."""
import copy

import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.component_program import LOGICAL_IO_V1, render


def program():
    return {
        "inputs": [{"name": "activation", "role": "input", "shape": [2, 3], "dtype": "i8"},
                   {"name": "weights", "role": "weight", "shape": [3, 4], "dtype": "i8"}],
        "nodes": [{"name": "projection", "op": "matmul", "inputs": ["activation", "weights"]}],
        "outputs": [{"name": "publishedResult", "value": "projection"}],
    }


def emit(value):
    return render(value, operand_dtype="i8", accumulator_dtype="i32")


def test_opt_in_logical_names_preserve_legacy_body_and_independent_alias():
    original = program()
    legacy, old = emit(original)
    selected = copy.deepcopy(original)
    selected["logical_io"] = LOGICAL_IO_V1
    typed, source = emit(selected)
    module = parse_mlir_text(source)
    module.verify()
    function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
    assert [v.data for v in function.attributes["merlin.logical_inputs"]] == ["activation", "weights"]
    assert [v.data for v in function.attributes["merlin.logical_outputs"]] == ["publishedResult"]
    assert function.attributes["merlin.logical_io"].data == LOGICAL_IO_V1
    assert "merlin.logical_io" not in old and "logical_io" not in legacy
    assert source.splitlines()[2:] == old.splitlines()[2:]
    assert typed["outputs"] == legacy["outputs"]
    assert typed["logical_io"] == LOGICAL_IO_V1
    assert emit(selected) == (typed, source)
    assert original == program()


@pytest.mark.parametrize("bad", [False, None, "other.version"])
def test_unknown_logical_io_refuses(bad):
    value = program()
    value["logical_io"] = bad
    with pytest.raises(ValueError):
        emit(value)


def test_logical_input_output_name_collision_refuses():
    value = program()
    value["logical_io"] = LOGICAL_IO_V1
    value["outputs"][0]["name"] = "activation"
    with pytest.raises(ValueError, match="distinct input and output"):
        emit(value)
