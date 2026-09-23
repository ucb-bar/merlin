"""The expected output of a layer-bench program, computed off-device, so every run checks numerics too.

A layer-bench program fills its operands on the device from a seeded LCG, runs the layer, and prints a
digest of the output bytes. This module reproduces all three steps exactly: the same fill stream, the
exact integer accumulation, the declared numerics contract's readout, and the same digest. A run whose
printed digest differs from :func:`expected_digest` computed a different function than the contract
says, whatever its cycle count.

The fill and digest constants below ARE the protocol. A target's program renderer must emit them from
here (never retype them), which is why they are module constants.

Layouts (row-major): conv input ``[B][H][W][CI]``, weights ``[KH][KW][CI][CO]``, bias ``[CO]``, output
``[B][OH][OW][CO]``; matmul ``A[M][K]``, ``B[K][N]``, bias ``D[N]`` broadcast over rows, ``C[M][N]``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

#: The program's operand generator: ``state = state * LCG_MUL + LCG_INC (mod 2**64)``; each draw
#: returns the new state. int8 operands take the top byte; bias values take ``(draw >> 33) % (2s+1) - s``.
LCG_MUL = 6364136223846793005
LCG_INC = 1442695040888963407
#: FNV-1a 64 over the output bytes; the printed digest clears the top bit so it prints as a signed-safe int.
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
DIGEST_MASK = (1 << 63) - 1
DEFAULT_BIAS_SPAN = 1 << 12
_M64 = (1 << 64) - 1


class LcgStream:
    """The device fill stream, vectorized by jump-ahead doubling (exact mod 2**64)."""

    def __init__(self, seed: int):
        self.state = int(seed) & _M64

    def draws(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, dtype=np.uint64)
        first = (self.state * LCG_MUL + LCG_INC) & _M64
        out = np.array([first], dtype=np.uint64)
        a, c = LCG_MUL, LCG_INC  # F^m(x) = a*x + c, currently m = 1
        with np.errstate(over="ignore"):
            while out.size < n:
                out = np.concatenate([out, out * np.uint64(a) + np.uint64(c)])
                a, c = (a * a) & _M64, (a * c + c) & _M64
        out = out[:n]
        self.state = int(out[-1])
        return out

    def fill_i8(self, n: int) -> np.ndarray:
        return (self.draws(n) >> np.uint64(56)).astype(np.uint8).view(np.int8)

    def fill_i8_sparse_binary(self, n: int) -> np.ndarray:
        """``{0, 1}`` at density 1/8, drawn from THIS stream so the oracle stays reproducible.

        It exists because of the readout, not as a preference. A kernel that pins its accumulator
        readout to the identity scale (an int8 output equal to the int32 accumulator, clamped) turns a
        full-range int8 dot product over a reduction of 64 or more into a saturated extreme in
        essentially every element, and a digest over a saturated output checks only a sign pattern.
        At density 1/8 with {0, 1} values the exact accumulator stays inside int8 for every element of
        the shapes this bench measures, so the digest checks every value rather than its clamp.
        """
        return ((self.draws(n) >> np.uint64(56)) % np.uint64(8) == 0).astype(np.int8)

    def fill_acc(self, n: int, span: int) -> np.ndarray:
        raw = (self.draws(n) >> np.uint64(33)).astype(np.int64)
        return (raw % (2 * span + 1) - span).astype(np.int64)


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & _M64
    return h


def _exact_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """int8 x int8 products summed in float64 are exact while every partial sum stays below 2**53."""
    k = a.shape[-1]
    if k * 128 * 128 >= 1 << 53:
        raise ValueError("reduction too long for an exact float64 accumulation")
    return np.rint(a.astype(np.float64) @ b.astype(np.float64)).astype(np.int64)


def matmul_accumulator(a: np.ndarray, b: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """``bias[n] + sum_k a[m, k] * b[k, n]``, in exact integer arithmetic.

    The accumulator BEFORE any readout. :func:`expected_output` narrows it through the numerics
    contract; a full-width readout (a kernel whose declared output dtype is the accumulator's own)
    is checked against this directly, because there is no narrowing stage to apply.
    """
    acc = _exact_matmul(a, b)
    return acc if bias is None else acc + np.asarray(bias, dtype=np.int64)[None, :]


def conv2d_accumulator(x: np.ndarray, w: np.ndarray, bias: np.ndarray, *, stride: int, padding: int) -> np.ndarray:
    """``bias[co] + sum x[b, oy*s+kh-p, ox*s+kw-p, ci] * w[kh, kw, ci, co]`` with zero padding."""
    b, h, wd, ci = x.shape
    kh, kw, ci2, co = w.shape
    if ci2 != ci:
        raise ValueError("channel mismatch")
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (wd + 2 * padding - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)))
    acc = np.broadcast_to(bias.astype(np.int64), (b, oh, ow, co)).copy()
    for i in range(kh):
        for j in range(kw):
            window = xp[:, i : i + stride * oh : stride, j : j + stride * ow : stride, :]
            acc += _exact_matmul(window.reshape(-1, ci), w[i, j]).reshape(b, oh, ow, co)
    return acc


#: Every operand in a packed blob starts on this boundary (the widest DMA row the programs issue).
OPERAND_ALIGN = 64

#: How a spec's int8 operands are drawn from the LCG stream. ``i8_full`` is the default and what every
#: receipt written before this option existed was keyed on: the top byte of each draw, over the whole
#: int8 range. ``sparse_binary`` draws the 1/8-dense {0, 1} matrices described in
#: :meth:`LcgStream.fill_i8_sparse_binary`, which a spec whose readout scale is 1 needs if its digest is
#: to check values rather than a saturation pattern. The mode is part of the spec, so it is part of
#: every receipt key and two modes can never be mistaken for repeated measurements of one thing.
ELEM_MODES = ("i8_full", "sparse_binary")


def operand_arrays(spec: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
    """The layer's operands, in blob order, drawn from the protocol's LCG stream (int8 data, int64 bias)."""
    stream = LcgStream(int(spec.get("seed", 1)))
    span = int(spec.get("bias_span", DEFAULT_BIAS_SPAN))
    mode = str(spec.get("elem_mode", "i8_full"))
    if mode not in ELEM_MODES:
        raise ValueError(f"unknown elem_mode {mode!r} (known: {ELEM_MODES})")
    fill_i8 = stream.fill_i8 if mode == "i8_full" else stream.fill_i8_sparse_binary
    if spec["op"] == "conv2d":
        b, n, ci, co = int(spec["batch"]), int(spec["in_dim"]), int(spec["in_channels"]), int(spec["out_channels"])
        k = int(spec["kernel"])
        return [
            ("input", fill_i8(b * n * n * ci).reshape(b, n, n, ci)),
            ("weights", fill_i8(k * k * ci * co).reshape(k, k, ci, co)),
            ("bias", stream.fill_acc(co, span)),
        ]
    if spec["op"] == "matmul":
        m, nn, kk = int(spec["m"]), int(spec["n"]), int(spec["k"])
        return [
            ("a", fill_i8(m * kk).reshape(m, kk)),
            ("b", fill_i8(kk * nn).reshape(kk, nn)),
            ("d", stream.fill_acc(nn, span)),
        ]
    raise ValueError(f"no reference for op {spec['op']!r}")


def pack_operands(spec: Mapping[str, Any], *, accumulator_dtype: str) -> tuple[bytes, dict[str, int]]:
    """One little-endian blob holding every operand at an aligned offset, plus the offsets by name.

    The bias is stored at the accumulator width the contract declares (e.g. ``i32``); int8 data as is.
    The program embeds this blob in its image, so it carries no generator of its own.
    """
    bits = accumulator_dtype[1:]
    if not accumulator_dtype.startswith("i") or not bits.isdigit():
        raise ValueError(f"unsupported accumulator dtype {accumulator_dtype!r}")
    acc = np.dtype(f"<i{int(bits) // 8}")
    blob = bytearray()
    offsets: dict[str, int] = {}
    for name, arr in operand_arrays(spec):
        blob += b"\0" * ((-len(blob)) % OPERAND_ALIGN)
        offsets[name] = len(blob)
        data = arr.astype(acc) if arr.dtype != np.int8 else arr
        if data.dtype != np.int8 and np.any(data.astype(np.int64) != arr):
            raise ValueError(f"operand {name} does not fit {accumulator_dtype}")
        blob += np.ascontiguousarray(data).tobytes()
    return bytes(blob), offsets


def fnv1a64_words(data: bytes) -> int:
    """FNV-1a over little-endian 64-bit words (the tail zero-padded): 8x fewer steps on the device."""
    padded = data + b"\0" * ((-len(data)) % 8)
    h = FNV_OFFSET
    for word in np.frombuffer(padded, dtype="<u8").tolist():
        h ^= word
        h = (h * FNV_PRIME) & _M64
    return h


def expected_output(spec: Mapping[str, Any], contract) -> np.ndarray:
    """The output tensor a layer program must produce under ``contract`` (a NumericsContract)."""
    ops = dict(operand_arrays(spec))
    act = "relu" if spec.get("relu") else "none"
    if spec["op"] == "conv2d":
        acc = conv2d_accumulator(
            ops["input"], ops["weights"], ops["bias"], stride=int(spec["stride"]), padding=int(spec["padding"])
        )
    else:
        acc = matmul_accumulator(ops["a"], ops["b"], ops["d"])
    return contract.readout(acc, np.float32(spec["scale"]), activation=act)


def expected_digest(spec: Mapping[str, Any], contract) -> int:
    """What the program prints: the word digest of the output bytes, top bit cleared."""
    return fnv1a64_words(expected_output(spec, contract).tobytes()) & DIGEST_MASK
