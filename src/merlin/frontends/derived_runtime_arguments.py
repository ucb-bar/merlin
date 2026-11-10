"""Check runtime kernel bindings against the original typed model call.

This rederives concrete shape/scratch declarations using compiler members only.
It grants no source-equivalence, bounds, ownership or numeric permission and
never imports the producer frontend or evaluator.
"""

import math

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, FloatAttr, IntegerAttr, StringAttr, TensorType, f64, i1, i64
from xdsl.dialects.func import CallOp, FuncOp

from .kernel_shape_abi import validate_dimensions


def checked_runtime_call_shapes(parent_module, spec, record):
    def refuse():
        raise ValueError("derived runtime kernel binding disagrees with its original typed call")

    scalars, buffers = spec.get("runtime_scalars", []), spec.get("buffers", [])
    if (not isinstance(scalars, list) or not isinstance(buffers, list)
            or any(not isinstance(s, dict) or not isinstance(s.get("name"), str) for s in scalars)
            or any(not isinstance(b, dict) or not isinstance(b.get("name"), str) for b in buffers)
            or len({s.get("name") for s in scalars}) != len(scalars)):
        refuse()
    count = len(scalars) + len(buffers) + len(spec.get("scalar_constants", []))
    positions = [s.get("source_arg") for s in scalars]
    names = [s.get("name") for s in scalars] + [b.get("name") for b in buffers]
    if (len(set(names)) != len(names)
            or any(type(position) is not int or not 0 <= position < count for position in positions)
            or len(set(positions)) != len(positions)):
        refuse()
    parent_calls = [op for op in parent_module.walk() if isinstance(op, CallOp)]
    ordinal = record.get("parent_call_ordinal")
    if type(ordinal) is not int or not 0 <= ordinal < len(parent_calls):
        refuse()
    call = parent_calls[ordinal]
    identity, attributes, operands = (call.attributes.get(name) for name in
        ("prov.kernel", "prov.kernel_args", "prov.kernel_operands"))
    declarations = {op.sym_name.data: op for op in parent_module.ops if isinstance(op, FuncOp)}
    callee = call.callee.root_reference.data
    declaration = declarations.get(callee)
    if (not isinstance(identity, StringAttr) or identity.data != record.get("kernel")
            or callee != record.get("callee") or declaration is None
            or declaration.attributes.get("prov.kernel") != identity or declaration.body.blocks
            or not isinstance(attributes, DictionaryAttr) or not isinstance(operands, ArrayAttr)
            or any(not isinstance(name, StringAttr) for name in operands)
            or len(operands) != len(call.arguments)
            or record.get("model_call_attributes") != {key: str(value) for key, value in call.attributes.items()}):
        refuse()
    schema_names = [name.data for name in operands]
    if len(set(schema_names)) != len(schema_names) or record.get("schema_operands") != schema_names:
        refuse()
    bindings = record.get("runtime_scalar_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {s["name"] for s in scalars}:
        refuse()
    values = {}
    for scalar in scalars:
        if (set(scalar) != {"name", "dtype", "source_arg"} | ({"bounds"} if "bounds" in scalar else set())
                or not isinstance(scalar["name"], str) or not scalar["name"].isidentifier()):
            refuse()
        binding = bindings[scalar["name"]]
        if not isinstance(binding, dict) or set(binding) != {"schema_argument", "attribute"}:
            refuse()
        schema = binding["schema_argument"]
        if not isinstance(schema, str) or schema not in attributes.data:
            refuse()
        attribute, dtype, bounds = attributes.data[schema], scalar["dtype"], scalar.get("bounds")
        if binding["attribute"] != str(attribute):
            refuse()
        if dtype in ("i8", "i16", "i32", "i64"):
            if not isinstance(attribute, IntegerAttr) or attribute.type != i64:
                refuse()
            value, width = attribute.value.data, int(dtype[1:])
            if not -(1 << (width-1)) <= value < 1 << (width-1):
                refuse()
            if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2
                    or any(type(n) is not int for n in bounds)
                    or not -(1 << (width-1)) <= bounds[0] <= value <= bounds[1] < 1 << (width-1)):
                refuse()
        elif dtype == "i1":
            if bounds is not None or not isinstance(attribute, IntegerAttr) or attribute.type != i1:
                refuse()
            value = bool(attribute.value.data)
        elif dtype in ("f32", "f64"):
            if (bounds is not None or not isinstance(attribute, FloatAttr) or attribute.type != f64
                    or not math.isfinite(attribute.value.data)):
                refuse()
            value = attribute.value.data
        else:
            refuse()
        values[scalar["name"]] = value
    validate_dimensions(spec, buffers, scalars)
    shapes = {buffer["name"]: [values[n] if isinstance(n, str) else n for n in buffer["shape"]]
              for buffer in buffers}
    inputs = [buffer for buffer in buffers if buffer["effect"] != "write"]
    outputs = {buffer["name"]: buffer for buffer in buffers if buffer["effect"] != "read"}
    mapping, result_order = record.get("source_operand_bindings"), record.get("source_result_order")
    if (not isinstance(mapping, dict) or set(mapping) != {b["name"] for b in inputs}
            or any(not isinstance(name, str) for name in mapping.values())
            or len(set(mapping.values())) != len(mapping) or set(mapping.values()) != set(schema_names)
            or not isinstance(result_order, list) or len(result_order) != len(outputs)
            or set(result_order) != set(outputs) or len(call.results) != len(outputs)):
        refuse()
    by_name = dict(zip(schema_names, call.arguments, strict=True))

    def matches(type_, buffer):
        return (isinstance(type_, TensorType) and list(type_.get_shape()) == shapes[buffer["name"]]
                and str(type_.get_element_type()) == buffer["dtype"])

    if (any(not matches(by_name[mapping[b["name"]]].type, b) for b in inputs)
            or any(not matches(value.type, outputs[name]) for value, name in zip(call.results, result_order, strict=True))):
        refuse()
    return shapes
