"""Read complete byte-bound independently checked outputs, never output0 alone."""
from __future__ import annotations

import json


def complete_float_outputs(metadata, raw):
    """The existing floating-only selection, with its original strict scalar types."""
    return _complete_outputs(metadata, raw, integers=False)


def complete_outputs(metadata, raw):
    """Preserve each declared floating, signed-integer or boolean result exactly."""
    return _complete_outputs(metadata, raw, integers=True)


def _complete_outputs(metadata, raw, *, integers):
    import numpy as np

    document = json.loads(raw)
    abi = metadata.get("output_abi")
    if (not isinstance(document, dict) or not isinstance(abi, list) or not abi
            or document.get("schema") != "merlin.capture.integer_reference.v1"
            or document.get("output_abi") != abi or not isinstance(document.get("outputs"), list)
            or len(document["outputs"]) != len(abi)):
        raise ValueError("independent reference must retain the exact complete output ABI")
    outputs = document["outputs"]
    for value, row in zip(outputs, abi, strict=True):
        shape, dtype = row.get("shape"), row.get("dtype")
        integral = dtype in {"i1", "i8", "i16", "i32", "i64"}
        if (dtype not in {"f32", "f16", "bf16"} and not (integers and integral)
                or not isinstance(shape, list)
                or any(type(n) is not int or n < 1 for n in shape)):
            raise ValueError("complete reference needs a supported finite static ABI")
        pending = [value]
        while pending:
            scalar = pending.pop()
            if isinstance(scalar, list):
                pending.extend(scalar)
            elif integral:
                width = int(dtype[1:])
                if dtype == "i1":
                    valid = type(scalar) is bool
                else:
                    valid = type(scalar) is int and -(1 << (width - 1)) <= scalar < (1 << (width - 1))
                if not valid:
                    raise ValueError("integer reference contains a converted or out-of-range scalar")
            elif type(scalar) not in (int, float):
                raise ValueError("complete floating reference contains a converted nonnumeric scalar")
        array = np.asarray(value)
        if integral:
            if list(array.shape) != shape:
                raise ValueError("complete integer reference differs from its exact output shape")
            continue
        if (list(array.shape) != shape or array.dtype.kind not in "if" or not np.isfinite(array).all()
                or not np.array_equal(array, array.astype(np.float32).astype(array.dtype))):
            raise ValueError("complete floating reference does not match its exact output shape/values")
        f32 = array.astype(np.float32)
        if dtype == "f16" and not np.array_equal(f32, f32.astype(np.float16).astype(np.float32)):
            raise ValueError("complete output is not exactly representable in declared f16")
        if dtype == "bf16" and np.any(f32.view(np.uint32) & 0xFFFF):
            raise ValueError("complete output is not exactly representable in declared bf16")
    return outputs


def reference_members(identities):
    """Both legacy output0 and the complete independent result remain protected."""
    return ("golden.npy",) + (("integer-reference.json",) if "integer-reference.json" in identities else ())
