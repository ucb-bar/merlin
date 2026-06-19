"""A dependency-free integer tensor with the ops the ToyNPU semantics need.

Pure Python (no numpy) so the runtime substrate has zero heavy dependencies. Tensors are
stored as a flat ``list[int]`` plus a shape; only the operations the simulator actually uses
are implemented: deterministic fill, 2-D matmul (i8 x i8 -> i32 accumulate), per-column bias
add, requantization (rounding arithmetic shift), relu, and saturating cast to int8.

This is real arithmetic, not a stub: the simulator's outputs are computed here and compared
against an independent reference recomputation.
"""
from __future__ import annotations

from dataclasses import dataclass

DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4}


def _i8_clamp(x: int) -> int:
    return -128 if x < -128 else (127 if x > 127 else x)


@dataclass
class Tensor:
    """A 1-D or 2-D integer tensor stored row-major in a flat list."""

    shape: tuple[int, ...]
    data: list[int]
    dtype: str = "i32"

    def __post_init__(self) -> None:
        n = 1
        for d in self.shape:
            n *= d
        if len(self.data) != n:
            raise ValueError(f"shape {self.shape} needs {n} elements, got {len(self.data)}")

    # -- construction -------------------------------------------------------
    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: str = "i32") -> "Tensor":
        n = 1
        for d in shape:
            n *= d
        return cls(tuple(shape), [0] * n, dtype)

    @classmethod
    def deterministic(cls, name: str, shape: tuple[int, ...], dtype: str = "i8",
                      lo: int = 0, hi: int = 3) -> "Tensor":
        """Fill deterministically from ``name`` (no RNG): stable across runs/machines."""
        n = 1
        for d in shape:
            n *= d
        span = hi - lo + 1
        seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) or 1
        data = [lo + ((seed * (k + 1) + (k * k)) % span) for k in range(n)]
        return cls(tuple(shape), data, dtype)

    # -- introspection ------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return len(self.data) * DTYPE_BYTES.get(self.dtype, 4)

    def to_list(self) -> list:
        """Return nested lists (1-D or 2-D) for JSON serialization."""
        if len(self.shape) == 1:
            return list(self.data)
        rows, cols = self.shape
        return [self.data[r * cols:(r + 1) * cols] for r in range(rows)]

    # -- ops ----------------------------------------------------------------
    def matmul(self, rhs: "Tensor") -> "Tensor":
        """2-D integer matmul: (m,k) x (k,n) -> (m,n) accumulated in i32."""
        m, k = self.shape
        k2, n = rhs.shape
        if k != k2:
            raise ValueError(f"matmul shape mismatch: {self.shape} x {rhs.shape}")
        out = [0] * (m * n)
        a, b = self.data, rhs.data
        for i in range(m):
            ai = i * k
            oi = i * n
            for p in range(k):
                aip = a[ai + p]
                if aip == 0:
                    continue
                bp = p * n
                for j in range(n):
                    out[oi + j] += aip * b[bp + j]
        return Tensor((m, n), out, "i32")

    def add_bias(self, bias: "Tensor") -> "Tensor":
        """Add a length-n bias vector to each row of an (m,n) tensor."""
        m, n = self.shape
        if bias.shape != (n,):
            raise ValueError(f"bias shape {bias.shape} != ({n},)")
        out = list(self.data)
        for i in range(m):
            oi = i * n
            for j in range(n):
                out[oi + j] += bias.data[j]
        return Tensor((m, n), out, self.dtype)

    def requant(self, shift: int) -> "Tensor":
        """Rounding arithmetic right shift by ``shift`` (a fixed-point requantization).

        merlin's native integer requant: round-half-UP. Distinct from Gemmini's float
        acc_scale (see ``requant_acc_scale``) — kept for the host/runtime-side path.
        """
        if shift <= 0:
            return Tensor(self.shape, list(self.data), self.dtype)
        half = 1 << (shift - 1)
        out = [(x + half) >> shift for x in self.data]
        return Tensor(self.shape, out, self.dtype)

    def requant_acc_scale(self, scale: float) -> "Tensor":
        """Gemmini-faithful float acc_scale: round-to-nearest-EVEN of ``x * scale`` in
        float32, matching ``gemmini_params.h`` ``ACC_SCALE``/``ROUND_NEAR_EVEN`` exactly.

        The i8 saturation that ACC_SCALE folds in is applied separately by ``to_i8`` (via
        ``output_dtype: i8``), so the composition is bit-identical to the macro. This is an
        ADDITIVE second requant format alongside the integer ``requant`` — Gemmini's i8
        accumulator-readout path uses this, not the round-half-up shift."""
        import struct

        def f32(v: float) -> float:                       # round a Python float to IEEE-754 single
            return struct.unpack("<f", struct.pack("<f", v))[0]

        s = f32(scale)
        out = []
        for x in self.data:
            prod = f32(f32(float(x)) * s)                 # float32 product, as the C macro computes it
            i = int(prod)                                 # trunc toward zero
            nxt = i - 1 if prod < 0 else i + 1
            rem = abs(prod - i)
            if rem < 0.5:
                y = i
            elif rem > 0.5:
                y = nxt
            else:                                         # exact tie -> round to even
                y = i if i % 2 == 0 else nxt
            out.append(int(y))
        return Tensor(self.shape, out, self.dtype)

    def relu(self) -> "Tensor":
        return Tensor(self.shape, [x if x > 0 else 0 for x in self.data], self.dtype)

    # -- vector-family (SIMD) ops: elementwise over equal-shape vectors + reduction --
    def ew_add(self, other: "Tensor") -> "Tensor":
        """Elementwise add of two equal-shape tensors."""
        if self.shape != other.shape:
            raise ValueError(f"ew_add shape mismatch: {self.shape} vs {other.shape}")
        return Tensor(self.shape, [a + b for a, b in zip(self.data, other.data)], self.dtype)

    def ew_mul(self, other: "Tensor") -> "Tensor":
        """Elementwise multiply of two equal-shape tensors."""
        if self.shape != other.shape:
            raise ValueError(f"ew_mul shape mismatch: {self.shape} vs {other.shape}")
        return Tensor(self.shape, [a * b for a, b in zip(self.data, other.data)], self.dtype)

    def reduce_sum(self) -> "Tensor":
        """Sum all elements -> a length-1 tensor."""
        return Tensor((1,), [sum(self.data)], self.dtype)

    def to_i8(self) -> "Tensor":
        return Tensor(self.shape, [_i8_clamp(x) for x in self.data], "i8")
