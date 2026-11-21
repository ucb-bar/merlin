"""Check original mutable-view storage against retained executable TTIR pointers.

Each complete base is one readwrite allocation, even when several source
arguments overlap. Coordinate SSA must reach source scalar formals and the
actual pointer-offset bindings. No producer analysis or evaluator is imported.
"""

from xdsl.dialects import builtin

PROFILE = "t2m.functional-kernel-binding.v2"
SOURCE_PROFILE = "m2m.functional-kernel-storage.v1"
FIELDS = {"prov.kernel_storage_bases", "prov.kernel_storage_views"}


def check(operation, spec, record, retained):
    attributes = operation.attributes
    source = retained.get("source_argument_binding", {})
    pointers = source.get("bindings")
    arguments = record.get("mutation_pointer_arguments")
    bases, views, mutated, ties, slots, names = (
        attributes.get(key)
        for key in (
            "prov.kernel_storage_bases",
            "prov.kernel_storage_views",
            "prov.kernel_mutated",
            "prov.kernel_tied",
            "prov.kernel_result_slots",
            "prov.kernel_operands",
        )
    )
    if (
        any(not isinstance(value, builtin.ArrayAttr) for value in (bases, views, mutated, ties, slots, names))
        or not bases
        or len(bases) > 64
        or not views
        or len(views) != len(mutated)
        or len(ties) != len(bases)
        or any(not isinstance(value, builtin.StringAttr) for value in (*names, *mutated))
        or len(set(mutated)) != len(mutated)
        or any(not isinstance(value, builtin.IntegerAttr) or value.type != builtin.i64 for value in slots)
        or source.get("schema") != "t2m.source-allocation-bindings.v1"
        or not isinstance(pointers, list)
        or not isinstance(arguments, dict)
        or set(arguments) != {value.data for value in mutated}
    ):
        raise ValueError("mutable storage needs its complete original base/view/pointer contract")
    outputs = record["source_result_order"]
    indices = [value.value.data for value in slots]
    if (
        len(outputs) < len(bases)
        or indices not in (list(range(len(outputs))), [-1, *range(len(outputs))])
        or indices[0] == -1
        and len(outputs) != len(bases)
    ):
        raise ValueError("mutable storage changes its original result/base slots")
    operands = dict(zip((name.data for name in names), operation.operands, strict=True))
    buffers = {buffer["name"]: buffer for buffer in spec["buffers"]}
    source_names = {name: buffer for buffer, name in record["source_operand_bindings"].items()}
    scalar_names = record["runtime_operand_arguments"]
    scalars = {scalar["name"]: scalar for scalar in spec.get("runtime_scalars", ())}
    coordinates, root_names = set(), []

    def fields(value):
        if not isinstance(value, builtin.ArrayAttr) or any(not isinstance(name, builtin.StringAttr) for name in value):
            raise ValueError("mutable storage coordinate fields are not complete scalar names")
        for name in value:
            selected = [source for source, target in scalar_names.items() if target == name.data]
            if (
                name.data in coordinates
                or name.data not in operands
                or operands[name.data].type != builtin.i64
                or not selected
                or any(scalars[source]["dtype"] != "i64" for source in selected)
            ):
                raise ValueError("mutable storage coordinate does not reach its actual logical i64 source formal")
            coordinates.add(name.data)

    for index, (base, tie) in enumerate(zip(bases, ties, strict=True)):
        if (
            not isinstance(base, builtin.DictionaryAttr)
            or set(base.data) != {"operand", "shape"}
            or not isinstance(base.data["operand"], builtin.StringAttr)
            or not isinstance(tie, builtin.DictionaryAttr)
            or set(tie.data) != {"base_index", "operand", "result"}
            or any(
                not isinstance(value, builtin.IntegerAttr) or value.type != builtin.i64 for value in tie.data.values()
            )
        ):
            raise ValueError("mutable storage has a malformed original base/result tie")
        name = base.data["operand"].data
        tensor, buffer = operands.get(name), source_names.get(name)
        operand, result = tie.data["operand"].value.data, tie.data["result"].value.data
        if (
            tensor is None
            or not isinstance(tensor.type, builtin.TensorType)
            or buffer not in buffers
            or buffer in root_names
            or buffers[buffer]["effect"] != "readwrite"
            or tie.data["base_index"].value.data != index
            or not 0 <= operand < len(names)
            or names.data[operand] != base.data["operand"]
            or result != len(outputs) - len(bases) + index
            or not 0 <= result < len(outputs)
            or outputs[result] != buffer
            or tensor.type != operation.results[result].type
        ):
            raise ValueError("mutable storage loses its original complete readwrite allocation")
        fields(base.data["shape"])
        if len(base.data["shape"]) != len(tensor.type.get_shape()):
            raise ValueError("mutable storage base coordinates change its actual tensor rank")
        root_names.append(buffer)
    if set(root_names) != {name for name, buffer in buffers.items() if buffer["effect"] == "readwrite"}:
        raise ValueError("mutable storage has unaccounted source readwrite allocations")
    positions = set()
    for view, argument in zip(views, mutated, strict=True):
        if (
            not isinstance(view, builtin.DictionaryAttr)
            or set(view.data) != {"argument", "base_index", "kind", "shape", "strides", "offset"}
            or view.data["argument"] != argument
            or not isinstance(view.data["kind"], builtin.StringAttr)
            or view.data["kind"].data not in ("base", "alias", "slice", "as_strided")
            or not isinstance(view.data["base_index"], builtin.IntegerAttr)
            or view.data["base_index"].type != builtin.i64
            or not 0 <= view.data["base_index"].value.data < len(root_names)
            or not isinstance(view.data["offset"], builtin.StringAttr)
        ):
            raise ValueError("mutable storage view lacks its exact original geometry fields")
        fields(view.data["shape"])
        fields(view.data["strides"])
        fields(builtin.ArrayAttr([view.data["offset"]]))
        if len(view.data["shape"]) != len(view.data["strides"]):
            raise ValueError("mutable storage view has an incomplete stride rank")
        index = arguments[argument.data]
        if type(index) is not int or not 0 <= index < len(pointers) or index in positions:
            raise ValueError("mutable storage view lacks a distinct original TTIR pointer")
        positions.add(index)
        pointer = pointers[index]
        allocation = root_names[view.data["base_index"].value.data]
        if (
            not isinstance(pointer, dict)
            or set(pointer) != {"kind", "allocation", "offset"}
            or pointer["kind"] != "pointer"
            or pointer["allocation"] != allocation
            or scalar_names.get(pointer["offset"]) != view.data["offset"].data
        ):
            raise ValueError("mutable storage pointer does not use the original cloned allocation and offset")
    return True
