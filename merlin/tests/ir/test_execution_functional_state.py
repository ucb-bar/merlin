"""Original updated bases cannot be rebound to fresh, equally typed results."""

import json

import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.frontends.execution_functional_state import call, entry


def source():
    typ = builtin.TensorType(builtin.i64, [4])
    block = Block(arg_types=[typ])
    operation = func.CallOp("update", [block.args[0]], [typ, typ])

    def integer(value):
        return builtin.IntegerAttr(value, builtin.i64)

    operation.attributes.update(
        {
            "prov.kernel_operands": builtin.ArrayAttr([builtin.StringAttr("state")]),
            "prov.kernel_functionalization": builtin.StringAttr("m2m.functional-kernel-bases.v1"),
            "prov.kernel_source_schema": builtin.StringAttr("example::update(Tensor(a!) state) -> Tensor"),
            "prov.kernel_mutated": builtin.ArrayAttr([builtin.StringAttr("state")]),
            "prov.kernel_result_slots": builtin.ArrayAttr([integer(0), integer(1)]),
            "prov.kernel_tied": builtin.ArrayAttr(
                [
                    builtin.DictionaryAttr(
                        {
                            "argument": builtin.StringAttr("state"),
                            "base_index": integer(0),
                            "operand": integer(0),
                            "result": integer(1),
                        }
                    )
                ]
            ),
        }
    )
    block.add_ops([operation, func.ReturnOp(*operation.results)])
    function = func.FuncOp("forward", ([typ], [typ, typ]), Region(block))
    spec = {"buffers": [{"name": "root", "effect": "readwrite"}, {"name": "old", "effect": "write"}]}
    record = {"source_operand_bindings": {"root": "state"}, "source_result_order": ["old", "root"]}
    return function, operation, spec, record


def test_original_mutation_tie_preserves_old_and_updated_values():
    _function, operation, spec, record = source()
    assert call(operation, spec, record)
    operation.attributes["prov.kernel_result_slots"] = builtin.ArrayAttr(
        [builtin.IntegerAttr(-1, builtin.i64), builtin.IntegerAttr(0, builtin.i64), builtin.IntegerAttr(1, builtin.i64)]
    )
    with pytest.raises(ValueError, match="tuple slots"):
        call(operation, spec, record)


@pytest.mark.parametrize(
    "damage",
    (
        "fresh-output",
        "read-only",
        "missing-profile",
        "unknown-profile",
        "missing-tie",
        "changed-operand",
        "negative-base",
        "wrong-int-type",
        "duplicate-slot",
    ),
)
def test_forged_mutation_cannot_authorize_different_source_memory(damage):
    _function, operation, spec, record = source()
    if damage == "fresh-output":
        record["source_result_order"] = ["root", "old"]
    elif damage == "read-only":
        spec["buffers"][0]["effect"] = "read"
    elif damage == "missing-profile":
        operation.attributes.pop("prov.kernel_functionalization")
    elif damage == "unknown-profile":
        operation.attributes["prov.kernel_functionalization"] = builtin.StringAttr("unchecked")
    elif damage == "missing-tie":
        operation.attributes["prov.kernel_tied"] = builtin.ArrayAttr([])
    elif damage == "duplicate-slot":
        operation.attributes["prov.kernel_result_slots"] = builtin.ArrayAttr(
            [builtin.IntegerAttr(0, builtin.i64), builtin.IntegerAttr(0, builtin.i64)]
        )
    else:
        tie = dict(operation.attributes["prov.kernel_tied"].data[0].data)
        key = "operand" if damage == "changed-operand" else "base_index"
        tie[key] = builtin.IntegerAttr(
            1 if damage == "changed-operand" else -1, builtin.i32 if damage == "wrong-int-type" else builtin.i64
        )
        operation.attributes["prov.kernel_tied"] = builtin.ArrayAttr([builtin.DictionaryAttr(tie)])
    with pytest.raises(ValueError, match="functional kernel"):
        call(operation, spec, record)


def state_contract():
    return {
        "schema": "m2m.functional-state-results.v1",
        "invocation": "explicit-state-results",
        "output_count": 2,
        "state_outputs": [{"result": 0, "input": 0, "kind": "USER_INPUT_MUTATION", "target": "x"}],
        "user_outputs": [1],
    }


def test_explicit_state_entry_keeps_borrowed_inputs_and_complete_user_outputs():
    function, _operation, _spec, _record = source()
    assert not entry(function)
    function.attributes["prov.functional_state"] = builtin.StringAttr(json.dumps(state_contract()))
    assert entry(function)
    assert not function.arg_attrs


@pytest.mark.parametrize(
    "damage", ("missing-user", "overlapping-results", "wrong-input", "boolean-ordinal", "copyback")
)
def test_unknown_state_or_invocation_semantics_refuse(damage):
    function, _operation, _spec, _record = source()
    contract = state_contract()
    if damage == "missing-user":
        contract["user_outputs"] = []
    elif damage == "overlapping-results":
        contract["user_outputs"] = [0]
    elif damage == "wrong-input":
        contract["state_outputs"][0]["input"] = -1
    elif damage == "boolean-ordinal":
        contract["user_outputs"] = [True]
    else:
        contract["invocation"] = "in-place"
    function.attributes["prov.functional_state"] = builtin.StringAttr(json.dumps(contract))
    with pytest.raises(ValueError, match="functional state"):
        entry(function)
