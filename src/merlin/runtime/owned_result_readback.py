"""Complete observed-shape/result readback for explicit signature v6.

The source supplies ranks/storage/static extents; execution supplies unknown
extents. Exact input observations remain a separate required check. Frame
checksums close the transport, never substitute for complete value comparison.
"""

import math

import numpy as np

from merlin.compiler.frontend.intake.execution_result_signature import validate_signature
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.lower.options.execution_result_inputs import resolve
from merlin.compiler.lower.options.owned_results import _NUMPY
from merlin.runtime.harness.output_decoding import parse_binary_console_payloads

PROFILE = "merlin.execution-owned-readback.v1"


def decode(source, signature, arguments, raw, *, max_console_bytes=(128 << 20) + (1 << 20)):
    """Read one complete successful call; no result shape comes from inputs."""
    signature = validate_signature(parse_mlir_text(source), signature)
    arguments = tuple(arguments)
    resolve(source, signature, arguments)
    if type(max_console_bytes) is not int or not 0 < max_console_bytes <= 256 << 20:
        raise ValueError("execution readback needs a bounded byte budget")
    if type(raw) is not bytes or len(raw) > max_console_bytes:
        raise ValueError("execution readback exceeds its console byte budget")
    values, metrics, frames = parse_binary_console_payloads(raw)
    names = {
        f"{prefix}{i}{suffix}"
        for prefix, rows in (("in", signature["inputs"]), ("out", signature["outputs"]))
        for i in range(len(rows))
        for suffix in ("", "_shape")
    }
    if set(values) != names or set(frames) != names:
        raise ValueError("execution readback omits or changes the complete input/result roster")
    observed, result_bytes = [], 0
    for prefix, rows in (("in", signature["inputs"]), ("out", signature["outputs"])):
        for i, row in enumerate(rows):
            name = f"{prefix}{i}"
            shape_frame = frames[name + "_shape"]
            rank = len(row["shape"])
            if (shape_frame.rows, shape_frame.cols, shape_frame.word_bytes, shape_frame.signed) != (1, rank, 8, False):
                raise ValueError("execution readback changes the actual source rank")
            shape = tuple(int(value) for value in np.frombuffer(values[name + "_shape"], dtype="<u8"))
            if len(shape) != rank or any(
                not 0 <= size < 1 << 63 or wanted is not None and size != wanted
                for size, wanted in zip(shape, row["shape"], strict=True)
            ):
                raise ValueError("execution readback changes a source static dimension")
            dtype = np.dtype(_NUMPY[row["dtype"]])
            count = math.prod(shape)
            nbytes = count * dtype.itemsize
            if prefix == "out":
                result_bytes += nbytes
                if result_bytes > signature["allocation_policy"]["max_output_bytes"]:
                    raise ValueError("execution readback exceeds the source output byte budget")
            elif shape != arguments[i].shape:
                raise ValueError("execution input observation changes actual invocation dimensions")
            signed = row["dtype"] in ("i8", "i16", "i32", "i64")
            frame = frames[name]
            if (frame.rows, frame.cols, frame.word_bytes, frame.signed) != (1, count, dtype.itemsize, signed):
                raise ValueError("execution readback format/geometry differs from source storage and observed extents")
            payload = values[name]
            if len(payload) != nbytes:
                raise ValueError("execution readback has an incomplete payload")
            if row["dtype"] == "i1" and any(value > 1 for value in payload):
                raise ValueError("execution readback Boolean words are noncanonical")
            if prefix == "in":
                expected = arguments[i].astype(arguments[i].dtype.newbyteorder("<"), copy=False)
                if payload != memoryview(expected).cast("B"):
                    raise ValueError("execution changed an original invocation input")
            else:
                observed.append(np.frombuffer(payload, dtype=dtype.newbyteorder("<")).reshape(shape).copy())
    return tuple(observed), {
        "profile": PROFILE,
        "shapes": [list(value.shape) for value in observed],
        "complete_result_bytes": result_bytes,
        "functional_inputs_preserved": True,
        "metrics": metrics,
        "scope": ("complete transport and input observations; numerical/target-equivalence-unverified"),
    }
