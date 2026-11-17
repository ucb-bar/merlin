"""Independently reopen live scalar bindings from compiler-owned parent SSA.

Singleton element reads and proved static tensor dimensions have standard MLIR
semantics. No producer import, evaluator input, weight value or provenance tag
supplies a scalar value, range or shape fact here.
Checked unsigned index adapters reopen original dominating source assertions;
their enclosures grant no concrete tensor dimension or scratch fact.
"""

from xdsl.dialects import arith, builtin, tensor

PROFILE = "t2m.live-runtime-arguments.v1"
RANGE_PROFILE = "t2m.checked-index-runtime-arguments.v1"
WORD_RANGE_PROFILE = "t2m.checked-word-runtime-arguments.v1"


def binding_profile(calls):
    live = []
    for call in calls:
        bindings = call.get("runtime_operand_bindings", {})
        if not isinstance(bindings, dict) or any(not isinstance(binding, dict) for binding in bindings.values()):
            raise ValueError("derived live runtime operand bindings must be dictionaries")
        live.extend(bindings.values())
    if any(binding.get("adapter") == "checked-size-scalar" for binding in live):
        return WORD_RANGE_PROFILE
    if any(binding.get("adapter") == "checked-index-castui" for binding in live):
        return RANGE_PROFILE
    return PROFILE if live else None


def _integer(value, depth=0):
    if depth >= 16 or not hasattr(value.owner, "name"):
        return None
    operation = value.owner
    if isinstance(operation, arith.ConstantOp) and isinstance(operation.value, builtin.IntegerAttr):
        return operation.value.value.data
    if isinstance(operation, tensor.DimOp):
        source, axis = operation.operands
        axis = _integer(axis, depth + 1)
        if (
            isinstance(source.type, builtin.TensorType)
            and type(axis) is int
            and 0 <= axis < len(source.type.get_shape())
        ):
            extent = source.type.get_shape()[axis]
            return extent if extent >= 0 else None
    if isinstance(operation, tensor.FromElementsOp) and len(operation.operands) == 1:
        return _integer(operation.operands[0], depth + 1)
    return None


def checked_live_operand(
    scalar, record, by_name, *, source_point=None, checked_indices=False, checked_words=False, shape_facts=None
):
    def refuse():
        raise ValueError("derived live runtime scalar disagrees with original parent SSA/types")

    dtype = scalar.get("dtype")
    if dtype not in ("i1", "i8", "i16", "i32", "i64", "f32", "f64"):
        refuse()
    target, bounds = getattr(builtin, dtype), scalar.get("bounds")
    if bounds is not None:
        if dtype not in ("i8", "i16", "i32", "i64") or not isinstance(bounds, list) or len(bounds) != 2:
            refuse()
        width = target.bitwidth
        if any(type(n) is not int for n in bounds) or not -(1 << (width - 1)) <= bounds[0] <= bounds[1] < 1 << (
            width - 1
        ):
            refuse()
    if not isinstance(record, dict) or not isinstance(record.get("schema_operand"), str):
        refuse()
    name = record["schema_operand"]
    if name not in by_name:
        refuse()
    value = by_name[name]
    fact = _integer(value)
    enclosure = None
    if isinstance(value.type, builtin.TensorType):
        if (
            not isinstance(value.type.encoding, builtin.NoneAttr)
            or value.type.element_type != target
            or any(extent != 1 for extent in value.type.get_shape())
        ):
            refuse()
        adapter = "singleton-tensor-extract"
    elif value.type == target:
        adapter = "exact-scalar"
        if (
            checked_words
            and shape_facts is not None
            and shape_facts.data_words
            and isinstance(target, builtin.IntegerType)
            and record.get("adapter") == "checked-size-scalar"
        ):
            enclosure = shape_facts.range_at(value, source_point)
            if enclosure is None or not 0 <= enclosure[0] <= enclosure[1] < 1 << (target.bitwidth - 1):
                refuse()
            adapter = "checked-size-scalar"
    elif isinstance(value.type, builtin.IndexType) and dtype in ("i8", "i16", "i32", "i64"):
        width = target.bitwidth
        if type(fact) is int and -(1 << (width - 1)) <= fact < 1 << (width - 1):
            adapter = "proved-index-cast"
        elif fact is None and checked_indices and source_point is not None:
            from .source_index_ranges import unsigned_index_range

            enclosure = unsigned_index_range(value, source_point)
            if enclosure is None and shape_facts is not None and shape_facts.expressions:
                enclosure = shape_facts.range_at(value, source_point)
            if enclosure is None or not 0 <= enclosure[0] <= enclosure[1] < 1 << (width - 1):
                refuse()
            adapter = "checked-index-castui"
        else:
            refuse()
    else:
        refuse()
    if bounds is not None and fact is not None and not bounds[0] <= fact <= bounds[1]:
        refuse()
    expected = {"schema_operand": name, "caller_type": str(value.type), "kernel_type": str(target), "adapter": adapter}
    if fact is not None:
        expected["constant_integer"] = fact
    if enclosure is not None:
        expected["integer_range" if adapter == "checked-size-scalar" else "unsigned_integer_range"] = list(enclosure)
    if record != expected:
        refuse()
    return fact
