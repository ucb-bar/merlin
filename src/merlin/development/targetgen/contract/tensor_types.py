"""Compare static source tensor declarations without materializing their values."""


def type_identity(dtype):
    from merlin.common import quant_formats as Q

    token = str(dtype)
    if Q.has(token):
        return ("format", Q.get(token).name)
    bits = Q.machine_bits(token)
    for prefix, family in (
        ("float", "float"),
        ("uint", "uint"),
        ("int", "int"),
        ("f", "float"),
        ("u", "uint"),
        ("i", "int"),
    ):
        if bits is not None and token.startswith(prefix) and token[len(prefix) :].isdigit():
            return (family, bits)
    raise ValueError(f"static source tensor dtype is unavailable: {dtype}")


def match_tensor_spec(spec, emitted):
    if (
        not isinstance(emitted, dict)
        or list(spec["shape"]) != emitted.get("shape")
        or type_identity(spec["dtype"]) != type_identity(emitted.get("dtype"))
    ):
        raise ValueError(f"emission changes declared shape or dtype: {spec['name']}")
