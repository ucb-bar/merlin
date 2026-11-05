"""Source-owned immutable column-panel preparation before device outlining.

The module attribute is an explicit layout request, never payload authority.
The ordinary group's captured immutable prepack must supply the exact matrix;
the existing blob/constant owner then binds its bit-preserving permutation.
"""

from merlin.compile.scheduling.tensor_storage import column_panel_storage

REQUEST = "merlin.constant_panel_columns"
MINIMUM_INPUT = "merlin.constant_panel_minimum_input_channels"


def request_constant_panels(module, columns, *, minimum_input_channels=1):
    from xdsl.dialects.builtin import IntegerAttr, i64

    if any(type(v) is not int or not 1 <= v <= 1 << 20 for v in (columns, minimum_input_channels)):
        raise ValueError("constant panel capabilities must be explicit bounded positive integers")
    if REQUEST in module.attributes or MINIMUM_INPUT in module.attributes:
        raise ValueError("source already contains a constant panel request")
    module.attributes[REQUEST] = IntegerAttr(columns, i64)
    module.attributes[MINIMUM_INPUT] = IntegerAttr(minimum_input_channels, i64)


def constant_panel_request(module):
    from xdsl.dialects.builtin import IntegerAttr, i64

    attr = module.attributes.get(REQUEST)
    if attr is None:
        return None
    if not isinstance(attr, IntegerAttr) or attr.type != i64 or not 1 <= attr.value.data <= 1 << 20:
        raise ValueError("source constant panel request is malformed")
    return attr.value.data


def constant_panel_minimum_input_channels(module):
    from xdsl.dialects.builtin import IntegerAttr, i64

    attr = module.attributes.get(MINIMUM_INPUT)
    if constant_panel_request(module) is None:
        if attr is not None:
            raise ValueError("source input-channel capability has no constant panel request")
        return 1
    if not isinstance(attr, IntegerAttr) or attr.type != i64 or not 1 <= attr.value.data <= 1 << 20:
        raise ValueError("source constant panel input-channel capability is malformed")
    return attr.value.data


def prepare_constant_panels(entry, array, columns, element_type, *, minimum_input_channels=1):
    """Prepare an eligible complete immutable multi-tap contraction matrix.

    Single-tap/dynamic/stationary-activation consumers retain their old source
    layout. No tensor role or request can create a missing captured payload.
    """
    if type(minimum_input_channels) is not int or not 1 <= minimum_input_channels <= 1 << 20:
        raise ValueError("constant panel input-channel capability must be explicitly bounded")
    if (columns is None or entry.get("op") != "conv2d"
            or entry.get("kh", 1) * entry.get("kw", 1) <= 1
            or entry.get("ci", 0) < minimum_input_channels):
        return array, None
    if array is None:
        raise ValueError("constant panels require the original immutable group prepack")
    import numpy as np

    shape = (int(entry["kh"]) * int(entry["kw"]) * int(entry["ci"]), int(entry["N"]))
    if tuple(array.shape) != shape:
        raise ValueError("constant panel payload differs from the complete original reduction matrix")
    encoding = column_panel_storage(*shape, columns, str(element_type))
    if encoding is None:
        return array, None
    data = np.ascontiguousarray(array)
    dtype = str(element_type)
    if (not dtype or dtype[0] not in "if" or not dtype[1:].isdigit()
            or data.dtype.kind != dtype[0] or data.dtype.itemsize * 8 != int(dtype[1:])):
        raise ValueError("constant panel payload storage width differs from its source type")
    raw = data.tobytes()
    packed = encoding.pack_bytes(raw, max_storage_bytes=len(raw))
    if encoding.unpack_bytes(packed, max_storage_bytes=len(raw)) != raw:
        raise ValueError("constant panel preparation failed its exact source-byte inverse")
    return np.frombuffer(packed, dtype=data.dtype).reshape(encoding.physical_shape), encoding
