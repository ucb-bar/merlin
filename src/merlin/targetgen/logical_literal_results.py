"""A uniform opaque complex word can have unequal real/imag carrier fields."""


def uniform_logical_literal(value, abi):
    from math import prod

    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType
    from xdsl.ir.affine import AffineMap

    dtype = abi.get("dtype")
    component = {"complex<f32>": 4, "complex<f64>": 8}.get(dtype)
    if component is None or not isinstance(value.type, TensorType):
        return False
    logical = tuple(abi.get("shape", ()))
    if value.type.get_shape() != (*logical, 2):
        return False
    if str(value.type.element_type) not in ({"f32", "i32"} if component == 4 else {"f64", "i64"}):
        return False
    visited = set()

    def raw(current):
        if current in visited:
            return None
        visited.add(current)
        owner = current.owner
        name = getattr(owner, "name", None)
        if not isinstance(current.type, TensorType):
            return None
        count = prod(current.type.get_shape())
        token = str(current.type.element_type)
        width = {"f32": 4, "i32": 4, "f64": 8, "i64": 8}.get(token)
        if width is None:
            return None
        if name == "arith.constant":
            attr = owner.properties.get("value")
            if not isinstance(attr, DenseIntOrFPElementsAttr):
                return None
            data = attr.data.data
            if len(data) == width:
                data *= count
            return data if len(data) == width * count else None
        if name in {"tensor.reshape", "tensor.cast", "tensor.collapse_shape", "tensor.expand_shape"}:
            if name == "tensor.reshape":
                shape_owner = owner.operands[1].owner
                shape = (
                    shape_owner.properties.get("value")
                    if getattr(shape_owner, "name", None) == "arith.constant"
                    else None
                )
                if (
                    not isinstance(shape, DenseIntOrFPElementsAttr)
                    or tuple(shape.get_values()) != current.type.get_shape()
                ):
                    return None
            elif len(owner.operands) != 1:
                return None
            data = raw(owner.operands[0])
            return data if data is not None and len(data) == width * count else None
        if name == "linalg.generic" and len(owner.inputs) == 1 and len(owner.results) == 1:
            if any(getattr(seed.owner, "name", None) != "tensor.empty" for seed in owner.outputs):
                return None
            if any(it.data.value != "parallel" for it in owner.get_iterator_types()):
                return None
            maps = list(owner.get_indexing_maps())
            identity = AffineMap.identity(len(current.type.get_shape()))
            if any(m.data != identity for m in maps):
                return None
            body = list(owner.body.block.ops)
            argument = owner.body.block.args[0]
            if len(body) == 1 and body[0].name == "linalg.yield" and list(body[0].operands) == [argument]:
                return raw(owner.inputs[0])
            if (
                len(body) == 2
                and body[0].name == "arith.bitcast"
                and list(body[0].operands) == [argument]
                and body[1].name == "linalg.yield"
                and list(body[1].operands) == list(body[0].results)
            ):
                return raw(owner.inputs[0])
        return None

    data = raw(value)
    word = component * 2
    return (
        bool(data)
        and len(data) == prod(logical) * word
        and all(data[i : i + word] == data[:word] for i in range(0, len(data), word))
    )
