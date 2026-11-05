"""SmolVLA full-policy padded projection recipe using existing Atlas arithmetic.

K and N tails are positive-zero physical padding. Logical rows and columns are
stripped only after the complete ordered MXU reduction and output rescale.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from atlas_host_layout import fp8_output_offset
import kernel_library  # noqa: E402,F401
from atlas_program import assembly, dma_load_bytes
from atlas_vector_arithmetic import ordered_matmul, rescale_bf16_array


def padded_extent(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Require a positive logical extent")
    return (value + 31) // 32 * 32


def prepare(encoded_a, encoded_b, exponent, *, workers=1):
    """Return physical operands, ordered reference, and unchanged-ISA assembly."""
    a, b = np.asarray(encoded_a), np.asarray(encoded_b)
    if (a.dtype != np.uint8 or b.dtype != np.uint8 or a.ndim != 2 or b.ndim != 2
            or a.shape[1] != b.shape[0] or not a.size or not b.size):
        raise ValueError("Require finite rank-two FP8 operands with matching K")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Require one to sixteen arithmetic workers")
    if type(exponent) is not int or not -126 <= exponent <= 127:
        raise ValueError("Output rescale outside finite normal BF16 domain")
    logical_m, logical_k, logical_n = a.shape[0], a.shape[1], b.shape[1]
    m, k, n = map(padded_extent, (logical_m, logical_k, logical_n))
    if k > 12288 or m * n > 1048576:
        raise ValueError("Projection requires row panels or exceeds SmolVLA K bound")
    output_offset = fp8_output_offset(m, k, n)
    dram_bytes = output_offset + m * n * 2 + 256
    if dram_bytes > 64 * 1024 * 1024:
        raise ValueError("Projection exceeds bounded Atlas DRAM window")
    if np.any((a & 127) == 127) or np.any((b & 127) == 127):
        raise ValueError("NaN FP8 outside admitted arithmetic")
    aa = np.pad(a, ((0, m - logical_m), (0, k - logical_k)))
    bb = np.pad(b, ((0, k - logical_k), (0, n - logical_n)))
    if workers == 1:
        expected = ordered_matmul(aa, bb)
    else:
        chunks = [aa[start:start + 32] for start in range(0, m, 32)]
        with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
            expected = np.concatenate(list(pool.map(lambda lhs: ordered_matmul(lhs, bb), chunks)))
    expected = rescale_bf16_array(expected, exponent)
    resident_b = k * n <= 256 * 1024
    resident_a = not resident_b and m * k <= 256 * 1024
    source = assembly(m, k, n, output_offset, exponent, resident_b=resident_b,
                      pretranspose_resident_b=resident_b, resident_a=resident_a)
    requested = {"load_bytes": dma_load_bytes(m, k, n, resident_a=resident_a, resident_b=resident_b),
                 "store_bytes": m * n * 2}
    return {"A": aa, "B": bb, "expected": expected,
            "logical_output": expected[:logical_m, :logical_n].copy(), "source": source,
            "logical_mkn": [logical_m, logical_k, logical_n], "physical_mkn": [m, k, n],
            "output_offset": output_offset, "dram_bytes": dram_bytes,
            "requested_dma": requested, "resident_a": resident_a, "resident_b": resident_b}
