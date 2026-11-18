"""Distinct mutable views must reach the same real source storage and offsets."""

import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block

from merlin.frontends.execution_functional_state import call


def source():
    typ = builtin.TensorType(builtin.i64, [6])
    fields = {
        "root_n": "@base0.size0",
        "left_n": "@view0.size0",
        "left_s": "@view0.stride0",
        "left_o": "@view0.offset",
        "right_n": "@view1.size0",
        "right_s": "@view1.stride0",
        "right_o": "@view1.offset",
    }
    block = Block(arg_types=[typ, *([builtin.i64] * len(fields))])
    operation = func.CallOp("update", block.args, [typ, typ, typ])

    def integer(number):
        return builtin.IntegerAttr(number, builtin.i64)

    def names(*values):
        return builtin.ArrayAttr(builtin.StringAttr(value) for value in values)

    views = []
    for index, argument in enumerate(("left", "right")):
        views.append(
            builtin.DictionaryAttr(
                {
                    "argument": builtin.StringAttr(argument),
                    "base_index": integer(0),
                    "kind": builtin.StringAttr("as_strided"),
                    "shape": names(f"@view{index}.size0"),
                    "strides": names(f"@view{index}.stride0"),
                    "offset": builtin.StringAttr(f"@view{index}.offset"),
                }
            )
        )
    operation.attributes.update(
        {
            "prov.kernel_operands": names("@base0", *fields.values()),
            "prov.kernel_functionalization": builtin.StringAttr("m2m.functional-kernel-storage.v1"),
            "prov.kernel_source_schema": builtin.StringAttr(
                "example::update(Tensor(a!) left, Tensor(b!) right) -> (Tensor, Tensor)"
            ),
            "prov.kernel_mutated": names("left", "right"),
            "prov.kernel_result_slots": builtin.ArrayAttr(integer(index) for index in range(3)),
            "prov.kernel_tied": builtin.ArrayAttr(
                [builtin.DictionaryAttr({"base_index": integer(0), "operand": integer(0), "result": integer(2)})]
            ),
            "prov.kernel_storage_bases": builtin.ArrayAttr(
                [builtin.DictionaryAttr({"operand": builtin.StringAttr("@base0"), "shape": names("@base0.size0")})]
            ),
            "prov.kernel_storage_views": builtin.ArrayAttr(views),
        }
    )
    spec = {
        "buffers": [
            {"name": "root", "effect": "readwrite"},
            {"name": "old_left", "effect": "write"},
            {"name": "old_right", "effect": "write"},
        ],
        "runtime_scalars": [{"name": name, "dtype": "i64"} for name in fields],
    }
    record = {
        "source_operand_bindings": {"root": "@base0"},
        "source_result_order": ["old_left", "old_right", "root"],
        "mutation_pointer_arguments": {"left": 0, "right": 1},
        "runtime_operand_arguments": fields,
    }
    retained = {
        "source_argument_binding": {
            "schema": "t2m.source-allocation-bindings.v1",
            "bindings": [
                {"kind": "pointer", "allocation": "root", "offset": "left_o"},
                {"kind": "pointer", "allocation": "root", "offset": "right_o"},
            ],
        }
    }
    return operation, spec, record, retained


def test_distinct_original_view_pointers_keep_one_complete_updated_base():
    assert call(*source())


@pytest.mark.parametrize(
    "damage",
    (
        "offset",
        "split-root",
        "fresh-result",
        "duplicate-pointer",
        "missing-pointer",
        "coordinate-type",
        "coordinate-rank",
        "lost-base",
        "unknown-profile",
        "missing-fields",
    ),
)
def test_storage_annotations_cannot_authorize_changed_memory_or_coordinates(damage):
    operation, spec, record, retained = source()
    if damage == "offset":
        retained["source_argument_binding"]["bindings"][0]["offset"] = "right_o"
    elif damage == "split-root":
        retained["source_argument_binding"]["bindings"][1]["allocation"] = "old_right"
    elif damage == "fresh-result":
        record["source_result_order"] = ["root", "old_right", "old_left"]
    elif damage == "duplicate-pointer":
        record["mutation_pointer_arguments"]["right"] = 0
    elif damage == "missing-pointer":
        record.pop("mutation_pointer_arguments")
    elif damage == "coordinate-type":
        spec["runtime_scalars"][1]["dtype"] = "i32"
    elif damage == "coordinate-rank":
        view = dict(operation.attributes["prov.kernel_storage_views"].data[0].data)
        view["strides"] = builtin.ArrayAttr([])
        operation.attributes["prov.kernel_storage_views"] = builtin.ArrayAttr(
            [builtin.DictionaryAttr(view), operation.attributes["prov.kernel_storage_views"].data[1]]
        )
    elif damage == "lost-base":
        spec["buffers"][0]["effect"] = "read"
    elif damage == "unknown-profile":
        operation.attributes["prov.kernel_functionalization"] = builtin.StringAttr("m2m.functional-kernel-bases.v1")
    else:
        operation.attributes.pop("prov.kernel_storage_bases")
    with pytest.raises(ValueError):
        call(operation, spec, record, retained)
