"""Resolve explicit public scalar storage words at invocation, never compilation."""


def resolve_words(bindings, inputs, arrays):
    if not isinstance(bindings, dict) or len(bindings) > 256:
        raise ValueError("data-derived shapes require a bounded public word binding roster")
    result = {}
    for name, binding in bindings.items():
        if (
            not isinstance(name, str)
            or not isinstance(binding, dict)
            or set(binding) != {"argument", "indices", "dtype"}
        ):
            raise ValueError("runtime shape word binding is incomplete")
        ordinal, indices, dtype = binding["argument"], binding["indices"], binding["dtype"]
        if (
            type(ordinal) is not int
            or not 0 <= ordinal < len(inputs)
            or ordinal >= len(arrays)
            or name != f"w{ordinal}"
            or dtype not in ("i8", "i16", "i32", "i64")
            or inputs[ordinal].get("dtype") != dtype
        ):
            raise ValueError("runtime shape word differs from its actual public argument/type")
        declared, array = inputs[ordinal], arrays[ordinal]
        if indices is None:
            valid = declared.get("kind") == "scalar" and declared.get("shape") == [] and array.shape == ()
        else:
            valid = (
                isinstance(indices, list)
                and len(indices) == len(declared["shape"])
                and all(type(index) is int and index == 0 for index in indices)
                and declared.get("kind") != "scalar"
                and all(type(size) is int and size == 1 for size in declared["shape"])
                and tuple(declared["shape"]) == tuple(array.shape)
            )
        width = int(dtype[1:])
        if not valid or array.dtype.kind != "i" or array.dtype.itemsize * 8 != width or not array.dtype.isnative:
            raise ValueError("runtime shape words need exact singleton integer source storage")
        value = array.item()
        if type(value) is not int:
            raise ValueError("runtime shape word cannot come from a floating or Boolean carrier")
        result[name] = value % (1 << width)
    return result
