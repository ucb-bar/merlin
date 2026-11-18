"""Independently check functional state ports and original kernel memory ties.

The reader uses original parent SSA and retained kernel ABI, never producer code
or evaluator data. State results do not make borrowed entry arguments writable
and do not grant a public alias, physical copyback or target effect implementation.
"""

import json

from xdsl.dialects import builtin

PROFILE = "t2m.functional-kernel-binding.v1"
_FIELDS = {
    "prov.kernel_functionalization",
    "prov.kernel_source_schema",
    "prov.kernel_mutated",
    "prov.kernel_tied",
    "prov.kernel_result_slots",
}


def entry(function):
    attr = function.attributes.get("prov.functional_state")
    if attr is None:
        return False
    if not isinstance(attr, builtin.StringAttr):
        raise ValueError("functional state entry lacks its source contract")
    record = json.loads(attr.data)
    if (
        not isinstance(record, dict)
        or set(record) != {"schema", "state_outputs", "user_outputs", "output_count", "invocation"}
        or record["schema"] != "m2m.functional-state-results.v1"
        or record["invocation"] != "explicit-state-results"
        or type(record["output_count"]) is not int
        or record["output_count"] != len(function.function_type.outputs)
        or not isinstance(record["state_outputs"], list)
        or not record["state_outputs"]
        or len(record["state_outputs"]) > record["output_count"]
        or not isinstance(record["user_outputs"], list)
        or any(type(value) is not int for value in record["user_outputs"])
        or record["user_outputs"] != list(range(len(record["state_outputs"]), record["output_count"]))
    ):
        raise ValueError("functional state entry differs from its complete original result roster")
    arguments = set()
    for result, state in enumerate(record["state_outputs"]):
        if (
            not isinstance(state, dict)
            or set(state) != {"result", "input", "kind", "target"}
            or type(state["result"]) is not int
            or state["result"] != result
            or type(state["input"]) is not int
            or not 0 <= state["input"] < len(function.function_type.inputs)
            or state["input"] in arguments
            or state["kind"] not in ("USER_INPUT_MUTATION", "BUFFER_MUTATION")
            or not isinstance(state["target"], str)
            or not state["target"]
            or function.function_type.outputs.data[result] != function.function_type.inputs.data[state["input"]]
        ):
            raise ValueError("functional state result changes its original input/type binding")
        arguments.add(state["input"])
    return True


def call(operation, spec, record):
    attributes = operation.attributes
    present = _FIELDS.intersection(attributes)
    if not present:
        return False
    if present != _FIELDS or attributes["prov.kernel_functionalization"] != builtin.StringAttr(
        "m2m.functional-kernel-bases.v1"
    ):
        raise ValueError("functional kernel lacks its known complete mutation contract")
    operands, mutated, ties, slots = (
        attributes[key]
        for key in ("prov.kernel_operands", "prov.kernel_mutated", "prov.kernel_tied", "prov.kernel_result_slots")
    )
    schema = attributes["prov.kernel_source_schema"]
    if (
        not isinstance(schema, builtin.StringAttr)
        or not schema.data
        or any(not isinstance(value, builtin.ArrayAttr) for value in (operands, mutated, ties, slots))
        or any(not isinstance(value, builtin.StringAttr) for value in (*operands, *mutated))
        or not mutated
        or len(set(mutated)) != len(mutated)
        or len(mutated) != len(ties)
        or any(not isinstance(value, builtin.IntegerAttr) or value.type != builtin.i64 for value in slots)
    ):
        raise ValueError("functional kernel has incomplete typed schema/base/result records")
    results = record["source_result_order"]
    ordinals = [value.value.data for value in slots]
    if (
        len(results) < len(ties)
        or ordinals not in (list(range(len(results))), [-1, *range(len(results))])
        or ordinals[0] == -1
        and len(results) != len(ties)
    ):
        raise ValueError("functional kernel changes its source tuple slots")
    mapping = record["source_operand_bindings"]
    buffers = {value["name"]: value for value in spec["buffers"]}
    expected = {argument: name for name, argument in mapping.items() if buffers[name]["effect"] == "readwrite"}
    seen = set()
    for index, tie in enumerate(ties):
        if not isinstance(tie, builtin.DictionaryAttr) or set(tie.data) != {
            "argument",
            "base_index",
            "operand",
            "result",
        }:
            raise ValueError("functional kernel loses an original updated-base field")
        values = tie.data
        name = values["argument"]
        if (
            not isinstance(name, builtin.StringAttr)
            or name not in mutated
            or name.data in seen
            or any(
                not isinstance(values[key], builtin.IntegerAttr) or values[key].type != builtin.i64
                for key in ("base_index", "operand", "result")
            )
        ):
            raise ValueError("functional kernel changes a typed original mutation tie")
        operand, result = values["operand"].value.data, values["result"].value.data
        source = expected.get(name.data)
        if (
            values["base_index"].value.data != index
            or not 0 <= operand < len(operands)
            or operands.data[operand] != name
            or result != len(results) - len(ties) + index
            or not 0 <= result < len(results)
            or source is None
            or results[result] != source
            or operation.operands[operand].type != operation.results[result].type
        ):
            raise ValueError("functional kernel mutation does not match its actual source readwrite buffer")
        seen.add(name.data)
    if seen != set(expected):
        raise ValueError("functional kernel loses an original source readwrite base")
    return True
