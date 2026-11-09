"""Atlas contraction boundaries, with scales applied only after accumulation."""
from dataclasses import dataclass
import numpy as np
from .arithmetic import matmul_codes, readout_bf16, bf16_values, bf16_bits


@dataclass
class Result:
    accumulator_bf16: np.ndarray
    readout_bf16: np.ndarray
    output_bf16: np.ndarray


def _codes(value):
    value = np.asarray(value)
    if value.dtype != np.uint8:
        raise ValueError("code operands must be raw uint8, with no dtype conversion")
    exp, mant = (value >> 3) & 15, value & 7
    if np.any(((exp == 0) & (mant != 0)) | ((exp == 15) & (mant == 7))):
        raise ValueError("graph contains forbidden E4M3 subnormal or reserved codes")
    return value


def _row_exponents(exp, shape):
    exp = np.asarray(exp)
    if exp.dtype.kind not in "iu":
        raise ValueError("exponents must be integers")
    if np.any(exp < -(2**31)) or np.any(exp > 2**31 - 1):
        raise ValueError("operand exponents must be representable in the source int32 format")
    try:
        return np.broadcast_to(exp, shape[:-1] + (1,)).astype(np.int64)
    except ValueError as e:
        raise ValueError("exponents must be scalar or constant over each contraction row") from e


def _add_bf16(a, b):
    # Atlas VADD widens to FP32, then truncates; it does not round to BF16.
    with np.errstate(invalid="ignore", over="ignore"):
        value = np.add(bf16_values(a), bf16_values(b), dtype=np.float32)
    bits = (value.view(np.uint32) >> 16).astype(np.uint16)
    return np.where(np.isnan(value), np.uint16(0x7fc0), bits).astype(np.uint16)


def contraction_nt(a_codes, a_exp, b_codes, b_exp, *, unit="mxu0", bias_bf16=None,
                   engine="numpy", threads=1):
    """[... M,K] x [... N,K], in logical ascending K order, BF16 device output."""
    a, b = _codes(a_codes), _codes(b_codes)
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-1]:
        raise ValueError("contraction requires [... M,K] and [... N,K] with matching K")
    ax, bx = _row_exponents(a_exp, a.shape), _row_exponents(b_exp, b.shape)
    exponent = ax + np.swapaxes(bx, -1, -2)
    if engine == 'numpy':
        method, options = matmul_codes, {}
    elif engine == 'native':
        from .native import matmul_codes as method
        options = {'threads': threads}
    else:
        raise ValueError('unknown independent arithmetic engine')
    accumulator = method(a, np.swapaxes(b, -1, -2), unit=unit, **options)
    readout = readout_bf16(accumulator, exponent)
    output = readout
    if bias_bf16 is not None:
        bias = np.asarray(bias_bf16)
        if bias.dtype != np.uint16:
            raise ValueError("bias must contain raw BF16 uint16 bits")
        output = _add_bf16(readout, bias)
    return Result(accumulator, readout, output)


def qlinear(x_codes, x_exp, w_codes, w_exp, *, unit="mxu0", bias_bf16=None,
            engine="numpy", threads=1):
    x = _codes(x_codes)
    w = _codes(w_codes)
    if x.ndim < 1 or w.ndim != 2:
        raise ValueError("qlinear requires [... K] and [N,K]")
    shape = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    ex = _row_exponents(x_exp, x.shape).reshape(-1, 1)
    result = contraction_nt(x2, ex, w, w_exp, unit=unit, bias_bf16=bias_bf16,
                            engine=engine, threads=threads)
    return Result(*(v.reshape(shape + (w.shape[0],)) for v in
                    (result.accumulator_bf16, result.readout_bf16, result.output_bf16)))
