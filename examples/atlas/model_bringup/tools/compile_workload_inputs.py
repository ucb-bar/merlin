"""Prepare Atlas workload inputs without an arithmetic oracle or device execution.

The caller configures the pinned model assembler/FP8 codec and passes its
assembler callable. This function performs source admission, encoding, packing
and instruction assembly only; it never constructs a Core or computes matmul.
"""

from atlas_host_layout import ATLAS_DRAM_BASE


import io
import struct

import numpy as np

import kernel_library  # noqa: E402,F401
from atlas_program import assembly, dma_load_bytes
from run_model_tile import encode_operands
from run_workload_contraction import admit_program, physical_payload, resource_plan, validate_workload_arrays


def _physical_array(array, dtype, maximum_bytes):
    if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype) or array.ndim != 2
        or any(size <= 0 or size % 32 for size in array.shape) or array.nbytes > maximum_bytes):
        raise ValueError("require bounded physical rank-two array with the explicit dtype and whole 32x32 tiles")
    return array


def pack_activation(array_uint8):
    """Pack physical A in M-tile, K-tile, row-major tile order."""
    array = _physical_array(array_uint8, "uint8", 64 * 1024**2)
    m, k = array.shape
    return b"".join(array[row : row + 32, inner : inner + 32].tobytes() for row in range(0, m, 32) for inner in range(0, k,
        32))


def pack_weights(array_uint8):
    """Pack physical B in N-tile, K-tile, row-major tile order."""
    array = _physical_array(array_uint8, "uint8", 64 * 1024**2)
    k, n = array.shape
    return b"".join(array[inner : inner + 32, column : column + 32].tobytes() for column in range(0, n,
        32) for inner in range(0, k, 32))


def pack_output(array_uint16):
    """Pack physical BF16 C as low/high 16-column halves of each output tile."""
    return physical_payload(_physical_array(array_uint16, "uint16", 2 * 1048576))


def prepare(arrays, source_dtype, assemble):
    """Return raw instructions and guarded DRAM for one complete contraction."""
    a, b, _ = validate_workload_arrays(arrays, source_dtype)
    m, k = a.shape
    n = b.shape[1]
    plan = resource_plan(m, k, n)
    encoded, _, conversion = encode_operands(arrays, "tensor-pow2")
    exponent = sum(item["scale_exponent"] for item in conversion.values())
    if not -126 <= exponent <= 127:
        raise ValueError("VPU rescale factor must be normal finite BF16")
    activation = pack_activation(encoded["A"])
    weights = pack_weights(encoded["B"])
    inputs = activation + weights
    output_bytes = m * n * 2
    offset = len(inputs) + 256
    initial = inputs + b"\xa5" * 256 + b"\xcd" * output_bytes + b"\x5a" * 256
    source = assembly(m, k, n, offset, exponent, resident_b=plan["resident_b"], pretranspose_resident_b=plan["resident_b"],
        resident_a=plan["resident_a"])
    words = list(assemble(io.StringIO(source)).assemble())
    admit_program(words)
    plan["instruction_words"] = len(words)
    load_bytes = dma_load_bytes(m, k, n, resident_a=plan["resident_a"], resident_b=plan["resident_b"])
    return {"source": source, "program": struct.pack(f"<{len(words)}I", *words), "initial": initial,
        "activation": activation, "weights": weights, "shape": [m, k, n], "rescale_exponent": exponent,
        "conversion": conversion, "plan": plan,
        "memory": {"dram_base": ATLAS_DRAM_BASE, "input_bytes": len(inputs), "output_offset": offset,
        "output_bytes": output_bytes, "guard_bytes_each": 256, "total_dram_bytes": len(initial)},
        "requested_dma": {"load_bytes": load_bytes, "store_bytes": output_bytes}}
