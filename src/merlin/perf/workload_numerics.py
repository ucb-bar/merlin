"""Host numerical reference for generated layer-scale workloads.

This module handles accumulator-format rounding independently of instruction
emission. It does not select a target or infer an undeclared numeric policy.
"""

from __future__ import annotations

from .workload_errors import WorkloadError


def encode_operand_bytes(values, dtype: str) -> bytes:
    """Encode values using the registered format, refusing unknown packing."""
    import numpy as np

    from merlin.common import quant_formats as QF

    a = np.asarray(values, dtype=np.float64)
    if not QF.has(dtype):
        raise WorkloadError(f"unknown operand format {dtype!r}")
    f = QF.get(dtype)
    bits = int(f.element_bits or 0)
    if bits == 0 or bits % 8:
        raise WorkloadError(f"{dtype!r} is a sub-byte format; its packing is not derivable here")
    if f.kind == "int_affine":
        lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if f.signed else (0, 2**bits - 1)
        code = "<i" if f.signed else "<u"
        return np.clip(np.rint(a), lo, hi).astype(f"{code}{bits // 8}").tobytes()
    if f.kind == "float_ieee":
        if bits == 32:
            return a.astype("<f4").tobytes()
        if bits == 16 and int(f.exp_bits or 0) == 5:
            return a.astype("<f2").tobytes()
        if bits == 16 and int(f.exp_bits or 0) == 8:
            u = a.astype("<f4").view("<u4").astype(np.uint64)
            return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")).tobytes()
        raise WorkloadError(f"no byte encoding derivable for {dtype!r}")
    if f.kind == "fp_ocp":
        from merlin.targetgen.fp8_codec import ocp_encode

        eb, mb = int(f.exp_bits or 0), int(f.mant_bits or 0)
        if not eb or bits != 8:
            raise WorkloadError(f"no byte encoding derivable for {dtype!r}")
        return bytes(bytearray(ocp_encode(float(v), eb, mb, signed=bool(f.signed)) for v in a.reshape(-1)))
    raise WorkloadError(f"no byte encoding derivable for {dtype!r}")


def _accum_rounder(accum_dtype: str):
    """Round a running sum into the declared accumulator format, if needed."""
    import numpy as np

    from merlin.common import quant_formats as QF

    f = QF.get(accum_dtype)
    if f.kind != "float_ieee":
        return None
    mant, exp = int(f.mant_bits or 0), int(f.exp_bits or 0)
    if mant == 10 and exp == 5:
        return lambda x: x.astype("<f2").astype(np.float32)
    if mant == 7 and exp == 8:

        def rnd(x):
            u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
            return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16).astype(np.uint32).view(np.float32)

        return rnd
    return None


def accumulate_reference(
    A, W, *, accum_dtype: str, operand_dtype: str | None = None, subnormal_operand_flush: bool = False
):
    """Host golden rounded after each MAC in the declared accumulator format.

    Operands are flushed only when the selected profile declares that its
    multiplier flushes subnormals. Returns ``None`` for an accumulator that
    does not round partial sums; the plain product is then the reference.
    """
    import numpy as np

    A = np.asarray(A, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    if subnormal_operand_flush and operand_dtype:
        from merlin.runtime import fp8_formats as FF

        try:
            min_normal, _ = FF.normal_range(operand_dtype)
        except KeyError:
            return None
        A = np.where(np.abs(A) < min_normal, np.float32(0.0), A).astype(np.float32)
        W = np.where(np.abs(W) < min_normal, np.float32(0.0), W).astype(np.float32)
    rnd = _accum_rounder(accum_dtype)
    if rnd is None:
        return None
    acc = np.zeros((A.shape[0], W.shape[1]), dtype=np.float32)
    for i in range(A.shape[1]):
        acc = rnd(acc + np.outer(A[:, i], W[i, :]))
    return acc
