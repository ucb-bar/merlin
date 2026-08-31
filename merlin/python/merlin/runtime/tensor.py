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

from merlin.common import stimulus as STIM

DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4}


def _i8_clamp(x: int) -> int:
    return -128 if x < -128 else (127 if x > 127 else x)


def pool_out_dims(H: int, W: int, pool_size, pool_stride, pool_padding) -> tuple[int, int]:
    """Pooled spatial extent, the floor form (``(H + pt + pb - ph) // sh + 1``).

    Lives here rather than beside ``conv_out_dims`` in :mod:`merlin.runtime.commandbuffer` only because
    :meth:`Tensor.maxpool2d_rows` needs it and that module imports THIS one; ``commandbuffer`` re-exports
    it so callers of the conv geometry find the pool geometry in the same place. One definition either
    way -- an engine that computed its own output extent could disagree with the tensor the golden
    produced and the mismatch would read as a numeric defect rather than a geometry one.
    """
    ph, pw = int(pool_size[0]), int(pool_size[1])
    sh, sw = int(pool_stride[0]), int(pool_stride[1])
    pt, pl, pb, pr = (int(x) for x in pool_padding)
    return (H + pt + pb - ph) // sh + 1, (W + pl + pr - pw) // sw + 1


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
        """Fill deterministically from ``name`` (no RNG): stable across runs/machines.

        The fill is indexed by ``(row, col)`` rather than by flat position, so rows and columns
        differ from one another and a row-stride / offset / transpose bug changes the output.
        See :mod:`merlin.common.stimulus` for why that matters and for the C emitters that keep
        the baremetal reference programs filling byte-identical data.
        """
        return cls(tuple(shape), STIM.fill(name, tuple(shape), lo, hi), dtype)

    # -- introspection ------------------------------------------------------
    @property
    def nbytes(self) -> int:
        return len(self.data) * DTYPE_BYTES.get(self.dtype, 4)

    def to_list(self) -> list:
        """Return nested lists (any rank) for JSON serialization.

        Rank 1 and 2 are unchanged. Rank 3+ used to raise
        ``ValueError: too many values to unpack (expected 2)`` from the ``rows, cols = self.shape``
        below, which made every batched operand unserialisable. That surfaced far from here: the muon
        harness's operand derivation calls this through a helper written to be *rank-agnostic*
        (``_fl2``, added after an earlier rank-3 crash), so the caller believed it handled batching while
        the type underneath still did not -- and a batched capsule failed as an opaque
        "cyclotron invocation failed: too many values to unpack (expected 2)". Measured on
        RP10_gemv_batched_fp16_pt, operands ``[2,16,16] @ [2,16,1]``.
        """
        if len(self.shape) == 1:
            return list(self.data)
        if len(self.shape) == 2:
            rows, cols = self.shape
            return [self.data[r * cols:(r + 1) * cols] for r in range(rows)]

        def _nest(dims, flat):
            """Row-major split of ``flat`` into ``dims``; the data layout is unchanged, only the nesting."""
            if len(dims) == 1:
                return list(flat)
            stride = 1
            for x in dims[1:]:
                stride *= x
            return [_nest(dims[1:], flat[i * stride:(i + 1) * stride]) for i in range(dims[0])]

        return _nest(list(self.shape), self.data)

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

    def maxpool2d_rows(self, *, in_dims, pool_size, pool_stride,
                       pool_padding=(0, 0, 0, 0), pad_value: int | None = None) -> "Tensor":
        """Windowed MAX over the spatial axes of a 2-D ``[batch*H*W, C]`` tensor -> ``[batch*Ho*Wo, C]``.

        THE ONLY windowed-max primitive in the runtime, on purpose. ``reduce_sum`` above is a TOTAL
        reduction, so before this there was nothing an engine could call to pool -- and the golden, the
        reference and the simulator each grade the others' arithmetic. Had any one of them grown its own
        pooling loop the three would have been free to disagree about window order, the ragged tail, or
        the padded cell, and the numeric gate would have enforced whichever one it asked first. One
        implementation here is what makes "golden == reference == simulate" mean something for a pooled
        capsule.

        The tensor is 2-D because that is the shape pooling actually reaches on this substrate: the
        accumulator readout is ``[M, N]`` and a conv's im2col result is ``[N*Ho*Wo, Co]``. Neither
        carries its own spatial extent, so ``in_dims`` (the ``orows``/``ocols`` of the store-path ABI)
        is a REQUIRED parameter rather than something inferred from the row count -- ``rows = 25`` is
        5x5 or 25x1 or 1x25 and nothing in the buffer says which.

        ``pad_value`` is required whenever any pad is nonzero and has NO default. The identity element
        of a max over a padded cell is not derivable: mathematically it is -inf, a hardware store path
        typically feeds zero, and picking either silently would produce a full tensor of plausible wrong
        numbers -- the exact silent-wrong-answer this repo fails closed against.
        """
        if len(self.shape) != 2:
            raise ValueError(
                f"maxpool2d_rows expects a 2-D [rows, C] tensor, got shape {self.shape}")
        rows, channels = self.shape
        H, W = int(in_dims[0]), int(in_dims[1])
        ph, pw = int(pool_size[0]), int(pool_size[1])
        sh, sw = int(pool_stride[0]), int(pool_stride[1])
        pt, pl, pb, pr = (int(x) for x in pool_padding)
        if H <= 0 or W <= 0:
            raise ValueError(f"maxpool2d_rows in_dims must be positive, got {(H, W)}")
        if ph <= 0 or pw <= 0:
            raise ValueError(f"maxpool2d_rows pool_size must be positive, got {(ph, pw)}")
        if sh <= 0 or sw <= 0:
            raise ValueError(f"maxpool2d_rows pool_stride must be positive, got {(sh, sw)}")
        plane = H * W
        if rows % plane:
            # A row count that is not a whole number of HxW planes means the declared geometry does not
            # describe this tensor. Rounding down would pool a truncated image and still return a
            # well-formed result, which is the failure mode that is impossible to spot downstream.
            raise ValueError(
                f"maxpool2d_rows: {rows} rows is not a whole multiple of the declared plane "
                f"{H}x{W}={plane}; the pool geometry does not describe this tensor")
        if (pt or pl or pb or pr) and pad_value is None:
            raise ValueError(
                f"maxpool2d_rows: pool_padding {(pt, pl, pb, pr)} is nonzero but no pad_value was "
                f"declared; the identity element for a max over a padded cell is a property of the "
                f"datapath (-inf mathematically, commonly 0 in a store path) and is not derivable here")
        batch = rows // plane
        Ho, Wo = pool_out_dims(H, W, (ph, pw), (sh, sw), (pt, pl, pb, pr))
        if Ho < 1 or Wo < 1:
            raise ValueError(
                f"maxpool2d_rows: window {(ph, pw)} stride {(sh, sw)} padding {(pt, pl, pb, pr)} "
                f"leaves no output position over a {H}x{W} plane (got {Ho}x{Wo})")
        src = self.data
        out = [0] * (batch * Ho * Wo * channels)
        oi = 0
        for n in range(batch):
            base_n = n * plane * channels
            for oy in range(Ho):
                y0 = oy * sh - pt
                for ox in range(Wo):
                    x0 = ox * sw - pl
                    for c in range(channels):
                        best: int | None = None
                        for ky in range(ph):
                            y = y0 + ky
                            for kx in range(pw):
                                x = x0 + kx
                                if 0 <= y < H and 0 <= x < W:
                                    v = src[base_n + (y * W + x) * channels + c]
                                else:
                                    v = pad_value
                                if best is None or v > best:
                                    best = v
                        out[oi] = best
                        oi += 1
        return Tensor((batch * Ho * Wo, channels), out, self.dtype)

    def dequant_per_channel(self, scale: "Tensor", axis: int = 1) -> "Tensor":
        """Dequantize a 2-D integer weight by a per-channel float scale -> a float32 weight.

        This is model2MLIR's int8 weight-only idiom: ``W_f32[k, n] = W_i8[k, n] * scale[c]`` where
        ``c`` is ``n`` for ``axis == 1`` (per-output-channel, the transposed matmul-RHS layout) or
        ``k`` for ``axis == 0``. The result is a plain float weight the matmul consumes normally."""
        if len(self.shape) != 2:
            raise ValueError(f"dequant_per_channel expects a 2-D weight, got {self.shape}")
        k, n = self.shape
        s = scale.data
        if axis == 1:
            if len(s) != n:
                raise ValueError(f"scale length {len(s)} != n {n} for axis 1")
            out = [self.data[r * n + c] * s[c] for r in range(k) for c in range(n)]
        elif axis == 0:
            if len(s) != k:
                raise ValueError(f"scale length {len(s)} != k {k} for axis 0")
            out = [self.data[r * n + c] * s[r] for r in range(k) for c in range(n)]
        else:
            raise ValueError(f"dequant_per_channel axis must be 0 or 1, got {axis}")
        return Tensor((k, n), out, "f32")

    def to_i8(self) -> "Tensor":
        return Tensor(self.shape, [_i8_clamp(x) for x in self.data], "i8")
