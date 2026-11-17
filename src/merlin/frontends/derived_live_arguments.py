"""Independently reopen live scalar bindings from compiler-owned parent SSA.

Singleton element reads and proved static tensor dimensions have standard MLIR
semantics. No producer import, evaluator input, weight value or provenance tag
supplies a scalar value, range or shape fact here.
"""

from xdsl.dialects import arith, builtin, tensor

PROFILE = "t2m.live-runtime-arguments.v1"


def _integer(value, depth=0):
    if depth >= 16 or not hasattr(value.owner, "name"):
        return None
    operation = value.owner
    if isinstance(operation, arith.ConstantOp) and isinstance(operation.value, builtin.IntegerAttr):
        return operation.value.value.data
    if isinstance(operation, tensor.DimOp):
        source, axis = operation.operands
        axis = _integer(axis, depth + 1)
        if (isinstance(source.type, builtin.TensorType) and type(axis) is int
                and 0 <= axis < len(source.type.get_shape())):
            extent = source.type.get_shape()[axis]
            return extent if extent >= 0 else None
    if isinstance(operation, tensor.FromElementsOp) and len(operation.operands) == 1:
        return _integer(operation.operands[0], depth + 1)
    return None


def checked_live_operand(scalar, record, by_name):
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
        if (any(type(n) is not int for n in bounds)
                or not -(1 << (width - 1)) <= bounds[0] <= bounds[1] < 1 << (width - 1)):
            refuse()
    if not isinstance(record, dict) or not isinstance(record.get("schema_operand"), str):
        refuse()
    name = record["schema_operand"]
    if name not in by_name:
        refuse()
    value = by_name[name]
    fact = _integer(value)
    if isinstance(value.type, builtin.TensorType):
        if (not isinstance(value.type.encoding, builtin.NoneAttr) or value.type.element_type != target
                or any(extent != 1 for extent in value.type.get_shape())):
            refuse()
        adapter = "singleton-tensor-extract"
    elif value.type == target:
        adapter = "exact-scalar"
    elif isinstance(value.type, builtin.IndexType) and dtype in ("i8", "i16", "i32", "i64"):
        width = target.bitwidth
        if type(fact) is not int or not -(1 << (width - 1)) <= fact < 1 << (width - 1):
            refuse()
        adapter = "proved-index-cast"
    else:
        refuse()
    if bounds is not None and fact is not None and not bounds[0] <= fact <= bounds[1]:
        refuse()
    expected = {"schema_operand": name, "caller_type": str(value.type), "kernel_type": str(target), "adapter": adapter}
    if fact is not None:
        expected["constant_integer"] = fact
    if record != expected:
        refuse()
    return fact
