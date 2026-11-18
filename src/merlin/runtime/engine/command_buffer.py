"""A dependency-free integer tensor with the ops the ToyNPU semantics need.

Pure Python, so the runtime substrate needs no heavy dependency; when numpy is importable, the
integer matmul uses it for large products (exact: see :meth:`Tensor.matmul`). Tensors are
stored as a flat ``list[int]`` plus a shape; only the operations the simulator actually uses
are implemented: deterministic fill, 2-D matmul (i8 x i8 -> i32 accumulate), per-column bias
add, requantization (rounding arithmetic shift), relu, and saturating cast to int8.

This is real arithmetic, not a stub: the simulator's outputs are computed here and compared
against an independent reference recomputation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from merlin.common import target_data as STIM
from merlin.compiler.ir.logical_abi import (
    DERIVATION_RECIPE_KEYS,
    EPILOGUE_STAGES,
    EPILOGUE_STAGE_SET,
    IM2COL_RECIPE_KEY,
    POOL_ATTR_ARITY,
    POOL_PAD_VALUE_ATTR,
    PRODUCING_KEYS,
    _produces,
    declared_output_dtypes,
    harness_derived_tensors,
    pool_out_dims,
    pool_params,
)

# ---- tensor ------------------------------------------------------------------------------------

try:
    import numpy as _np
except ImportError:  # the dependency-free path below stays complete
    _np = None

#: Products smaller than this many multiply-adds stay on the pure-Python loop (conversion dominates).
_ARRAY_MATMUL_MIN_MACS = 1 << 15

DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4}


def _i8_clamp(x: int) -> int:
    return -128 if x < -128 else (127 if x > 127 else x)


def _array_matmul(a: list[int], b: list[int], m: int, k: int, n: int) -> list[int] | None:
    """The loop matmul's exact result through int64 arrays, or ``None`` when that could overflow.

    The reference and the simulator evaluated large products element by element in Python -- a
    32x2048x32000 capsule is two billion multiply-adds -- and that held most of a grade's CPUs.
    Every partial sum is bounded by ``k * max|a| * max|b|``; below 2**62 int64 accumulation cannot
    wrap in any order, so the values equal the Python-integer loop's exactly.
    """
    try:
        left = _np.asarray(a, dtype=_np.int64).reshape(m, k)
        right = _np.asarray(b, dtype=_np.int64).reshape(k, n)
    except (OverflowError, TypeError, ValueError):
        return None  # an element outside int64, or a non-integer: the loop decides
    if left.size == 0 or right.size == 0:
        return None
    bound = k * int(_np.abs(left).max()) * int(_np.abs(right).max())
    if bound >= 1 << 62:
        return None
    return (left @ right).ravel().tolist()


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
    def deterministic(cls, name: str, shape: tuple[int, ...], dtype: str = "i8", lo: int = 0, hi: int = 3) -> "Tensor":
        """Fill deterministically from ``name`` (no RNG): stable across runs/machines.

        The fill is indexed by ``(row, col)`` rather than by flat position, so rows and columns
        differ from one another and a row-stride / offset / transpose bug changes the output.
        See :mod:`merlin.common.target_data` for why that matters and for the C emitters that keep
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
            return [self.data[r * cols : (r + 1) * cols] for r in range(rows)]

        def _nest(dims, flat):
            """Row-major split of ``flat`` into ``dims``; the data layout is unchanged, only the nesting."""
            if not dims:
                # Rank-0: a scalar tensor holds exactly one element and nests to that element, not
                # to a list. Reached whenever a whole-model entry point takes a scalar argument
                # (a per-tensor scale broadcast, say); `dims[0]` below would IndexError instead.
                return flat[0]
            if len(dims) == 1:
                return list(flat)
            stride = 1
            for x in dims[1:]:
                stride *= x
            return [_nest(dims[1:], flat[i * stride : (i + 1) * stride]) for i in range(dims[0])]

        return _nest(list(self.shape), self.data)

    # -- ops ----------------------------------------------------------------
    def matmul(self, rhs: "Tensor") -> "Tensor":
        """2-D integer matmul: (m,k) x (k,n) -> (m,n) accumulated in i32."""
        m, k = self.shape
        k2, n = rhs.shape
        if k != k2:
            raise ValueError(f"matmul shape mismatch: {self.shape} x {rhs.shape}")
        if _np is not None and m * k * n >= _ARRAY_MATMUL_MIN_MACS:
            exact = _array_matmul(self.data, rhs.data, m, k, n)
            if exact is not None:
                return Tensor((m, n), exact, "i32")
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

        Integer requantization rounds half up. A command buffer can instead declare
        a floating-point ``acc_scale`` stage (see ``requant_acc_scale``).
        """
        if shift <= 0:
            return Tensor(self.shape, list(self.data), self.dtype)
        half = 1 << (shift - 1)
        out = [(x + half) >> shift for x in self.data]
        return Tensor(self.shape, out, self.dtype)

    def requant_acc_scale(self, scale: float) -> "Tensor":
        """Scale in float32 and round to the nearest integer, ties to even.

        This is the declared ``acc_scale`` command-buffer arithmetic. An ``output_dtype: i8``
        stage applies saturation separately through :meth:`to_i8`; integer-shift
        :meth:`requant` remains a distinct operation.
        """
        import struct

        def f32(v: float) -> float:  # round a Python float to IEEE-754 single
            return struct.unpack("<f", struct.pack("<f", v))[0]

        s = f32(scale)
        out = []
        for x in self.data:
            prod = f32(f32(float(x)) * s)
            i = int(prod)  # trunc toward zero
            nxt = i - 1 if prod < 0 else i + 1
            rem = abs(prod - i)
            if rem < 0.5:
                y = i
            elif rem > 0.5:
                y = nxt
            else:  # exact tie -> round to even
                y = i if i % 2 == 0 else nxt
            out.append(int(y))
        return Tensor(self.shape, out, self.dtype)

    def requant_acc_scale_by_column(self, scales: list[float]) -> Tensor:
        """Apply one declared float32 multiplier per final-axis column.

        Every column retains the scalar operation's conversion, multiply and
        ties-to-even rounding. Narrowing/saturation is a separate operation.
        """
        from math import prod

        if not self.shape or not scales or len(scales) != self.shape[-1]:
            raise ValueError("acc_scale_by_column must name one multiplier per output column")
        if prod(self.shape) != len(self.data):
            raise ValueError("acc_scale_by_column tensor storage disagrees with its shape")
        width = len(scales)
        rows = len(self.data) // width
        columns = [
            Tensor((rows,), self.data[index::width], self.dtype).requant_acc_scale(scale).data
            for index, scale in enumerate(scales)
        ]
        return Tensor(self.shape, [columns[column][row] for row in range(rows) for column in range(width)], self.dtype)

    def residual_add_scaled(self, other: "Tensor", lhs_scale: float, rhs_scale: float) -> "Tensor":
        """``roundeven(self * lhs_scale + other * rhs_scale)``, elementwise, in float32.

        THE REFERENCE for a quantized residual add, and it rounds ONCE: both operands are scaled into
        the output's domain, summed, and the sum is rounded half-to-even. That is what a
        fake-quantized capture computes (dequantize, add in floating point, quantize). A unit that
        scales and rounds each operand on the way in and then adds integers computes a different
        function, up to one output step per operand; a capsule says how far a target may be from
        this reference (``bound_lsb``) instead of pretending the two are one.
        """
        import struct

        if self.shape != other.shape:
            raise ValueError(f"residual_add shape mismatch: {self.shape} vs {other.shape}")

        def f32(v: float) -> float:
            return struct.unpack("<f", struct.pack("<f", v))[0]

        sa, sb = f32(lhs_scale), f32(rhs_scale)
        out = []
        for a, b in zip(self.data, other.data):
            total = f32(f32(f32(float(a)) * sa) + f32(f32(float(b)) * sb))
            i = int(total)
            nxt = i - 1 if total < 0 else i + 1
            rem = abs(total - i)
            out.append(i if rem < 0.5 else (nxt if rem > 0.5 else (i if i % 2 == 0 else nxt)))
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

    def maxpool2d_rows(
        self, *, in_dims, pool_size, pool_stride, pool_padding=(0, 0, 0, 0), pad_value: int | None = None
    ) -> "Tensor":
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
            raise ValueError(f"maxpool2d_rows expects a 2-D [rows, C] tensor, got shape {self.shape}")
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
                f"{H}x{W}={plane}; the pool geometry does not describe this tensor"
            )
        if (pt or pl or pb or pr) and pad_value is None:
            raise ValueError(
                f"maxpool2d_rows: pool_padding {(pt, pl, pb, pr)} is nonzero but no pad_value was "
                f"declared; the identity element for a max over a padded cell is a property of the "
                f"datapath (-inf mathematically, commonly 0 in a store path) and is not derivable here"
            )
        batch = rows // plane
        Ho, Wo = pool_out_dims(H, W, (ph, pw), (sh, sw), (pt, pl, pb, pr))
        if Ho < 1 or Wo < 1:
            raise ValueError(
                f"maxpool2d_rows: window {(ph, pw)} stride {(sh, sw)} padding {(pt, pl, pb, pr)} "
                f"leaves no output position over a {H}x{W} plane (got {Ho}x{Wo})"
            )
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


# ---- commandbuffer -----------------------------------------------------------------------------
# Load/validate a Merlin command buffer and materialize its declared tensors.
#
# The execution-oriented command buffer carries a ``tensors`` table (name -> shape/dtype/role)
# in addition to the opcode list. Input/weight/bias tensors are materialized deterministically
# (see :func:`Tensor.deterministic`) so a run is reproducible without external input files; an
# explicit inputs mapping can override them.

#: The attribute names a POOLING epilogue stage reads, and the arity each one must have.
#:
#: They ride on the COMMAND's attributes because that is where the hardware carries them: this target
#: family fuses pooling into the STORE path (``config_st(..., pool_stride, pool_size, pool_out_dim,
#: porows, pocols, orows, ocols, upad, lpad)``) and into the fused conv loop, so the pool parameters are
#: configured on the same instruction that reads the accumulator out. By the time an engine reaches the
#: readout the tensor is a 2-D ``[rows, channels]`` and no conv geometry is in scope, so the extent the
#: rows unflatten to (``pool_in_dims`` = the ABI's ``orows``/``ocols``) must be declared, not guessed.
#: Optional companion: the value a padded cell contributes to the max. No default on purpose -- see
#: :meth:`Tensor.maxpool2d_rows`.


def apply_pool_stage(t: Tensor, stage: str, attrs: dict[str, Any], *, op: str) -> Tensor:
    """Apply the named pooling stage to ``t``. The single definition all three engines call."""
    if stage != "maxpool":  # pragma: no cover - callers dispatch on the name
        raise ValueError(f"{op}: {stage!r} is not a pooling stage this runtime defines")
    p = pool_params(attrs, op=op)
    return t.maxpool2d_rows(
        in_dims=p["pool_in_dims"],
        pool_size=p["pool_size"],
        pool_stride=p["pool_stride"],
        pool_padding=p["pool_padding"],
        pad_value=p["pad_value"],
    )


#: The epilogue stages the command-buffer ABI admits, in the order a COMMIT applies them. This is the
#: single definition the ABI, its JSON schema and the interface dialect all read, so a stage exists here
#: or it does not exist. It is NOT a claim that a given target implements one: an engine that does not
#: RAISES by name, and a target that cannot execute it fails at its oracle. Widening this tuple is a
#: contract change and needs both an engine that implements the stage and a target rung evidencing it.


#: Stages a readout applies with NO operand and NO parameter. Such a stage is pure configuration: a
#: unit is switched into it and stays there until something switches it back, so the command AFTER one
#: is where a backend that does not restore its configuration shows. A stage with a parameter or an
#: operand is re-stated by every command that uses it and cannot leak this way.
CONFIGURATION_ONLY_STAGES: tuple[str, ...] = ("relu",)

#: The subset of the epilogue that adds a per-column bias, which is the one stage needing an operand
#: NAME resolved rather than a flag. Kept as its own tuple because a bias dropped in silence leaves
#: every element off by exactly its column's bias, which reads as a plausible answer.
BIAS_STAGES = ("bias_add", "bias")


@dataclass(frozen=True)
class BatchedMatmulGeometry:
    """Shape facts for one rank-N batch of independent matrix contractions.

    ``batch_shape`` is deliberately retained rather than collapsed into one ABI dimension: the
    command preserves the source tensor's rank, while an engine that executes 2-D kernels may walk
    its row-major slices using ``batch_count``.
    """

    batch_shape: tuple[int, ...]
    batch_count: int
    m: int
    k: int
    n: int

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (*self.batch_shape, self.m, self.n)

    @property
    def lhs_slice_elements(self) -> int:
        return self.m * self.k

    @property
    def rhs_slice_elements(self) -> int:
        return self.k * self.n

    @property
    def output_slice_elements(self) -> int:
        return self.m * self.n

    @property
    def macs(self) -> int:
        return self.batch_count * self.m * self.k * self.n


def batched_matmul_geometry(
    lhs_shape: Sequence[int] | None,
    rhs_shape: Sequence[int] | None,
    dst_shape: Sequence[int] | None,
    *,
    op: str = "BATCHED_MATMUL",
) -> BatchedMatmulGeometry:
    """Validate ``A[*B,M,K] @ W[*B,K,N] -> Y[*B,M,N]`` and return its geometry.

    There is no broadcasting: both varying operands must carry the same non-empty batch prefix.
    Flattening that prefix is only an execution detail and follows row-major lexicographic order.
    Shapes are static and strictly positive so a zero/unknown extent cannot be priced as real work.
    """

    def normalized(shape: Sequence[int] | None, role: str) -> tuple[int, ...]:
        if (
            not isinstance(shape, Sequence)
            or isinstance(shape, (str, bytes))
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
        ):
            raise ValueError(f"{op}: {role} shape must contain only positive static integers, got {shape!r}")
        result = tuple(int(value) for value in shape)
        if len(result) < 3:
            raise ValueError(f"{op}: {role} must have rank at least 3, got shape {result}")
        return result

    lhs = normalized(lhs_shape, "operand a")
    rhs = normalized(rhs_shape, "operand w")
    lhs_batch, (m, k) = lhs[:-2], lhs[-2:]
    rhs_batch, (k2, n) = rhs[:-2], rhs[-2:]
    if lhs_batch != rhs_batch:
        raise ValueError(
            f"{op}: operand batch prefixes must match exactly (no broadcasting), got {lhs_batch} and {rhs_batch}"
        )
    if k != k2:
        raise ValueError(f"{op}: reduction dimensions do not match, got K={k} and K={k2}")
    batch_count = 1
    for extent in lhs_batch:
        batch_count *= extent
    geometry = BatchedMatmulGeometry(lhs_batch, batch_count, m, k, n)
    dst = normalized(dst_shape, "destination")
    if dst != geometry.output_shape:
        raise ValueError(f"{op}: destination shape must be {geometry.output_shape}, got {dst}")
    return geometry


#: A program whose command buffer declares ``params.console_value_cap`` prints an output LARGER than
#: the cap as one digest line instead of its values: ``OUTSUM <name> <rows> <cols> <16 hex digits>``.
#: A simulated UART moves a few tens of bytes a second. Measured: a whole-model program closed its
#: measured window, passed its own validation, then spent ninety minutes draining 188 KB of a
#: million-value output and was killed at its time limit, twice, on two designs.
CONSOLE_VALUE_CAP_PARAM = "console_value_cap"
OUTPUT_DIGEST_LINE = "OUTSUM"
# Opt-in full-value transport: only the wire representation changes. Unlike
# OUTSUM it reconstructs every integer container word before numeric checking.
CONSOLE_OUTPUT_TRANSPORT_PARAM = "console_output_transport"
CONSOLE_OUTPUT_TRANSPORT_B64_V1 = "out_b64_v1"
_FNV_OFFSET, _FNV_PRIME, _U64 = 1469598103934665603, 1099511628211, (1 << 64) - 1


def output_text_digest(values) -> str:
    """FNV-1a (64-bit) over exactly the text an ``OUT`` line would have carried for ``values``.

    Defined over the PRINTED form (`` <v>`` per value, row-major) and not over the stored bytes, so
    one definition serves every container a harness prints and the host needs nothing but the
    integers it would have parsed. Sixteen lower-case hex digits.
    """
    digest = _FNV_OFFSET
    for value in values:
        for byte in f" {int(value)}".encode("ascii"):
            digest = ((digest ^ byte) * _FNV_PRIME) & _U64
    return f"{digest:016x}"


def output_digest_line(name: str, rows: list) -> str:
    """The exact ``OUTSUM`` line a correct program prints for output ``name`` with these values."""
    flat = [v for row in rows for v in row] if rows and isinstance(rows[0], list) else list(rows)
    n_rows = len(rows) if rows and isinstance(rows[0], list) else 1
    n_cols = len(rows[0]) if rows and isinstance(rows[0], list) else len(flat)
    return f"{OUTPUT_DIGEST_LINE} {name} {n_rows} {n_cols} {output_text_digest(flat)}"


def parse_console_digests(text: str) -> dict[str, dict]:
    """``{name: {rows, cols, digest}}`` for every digest line of a console transcript."""
    found: dict[str, dict] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] == OUTPUT_DIGEST_LINE:
            found[parts[1]] = {"rows": int(parts[2]), "cols": int(parts[3]), "digest": parts[4]}
    return found


#: The stages a ``RESIDUAL_ADD`` may carry. The add's own rounding is its definition, so the only
#: thing left for a readout to do is the activation.
RESIDUAL_ADD_STAGES: tuple[str, ...] = ("relu",)


def residual_add_params(attrs: dict[str, Any], *, op: str) -> tuple[float, float, int]:
    """``(lhs_scale, rhs_scale, bound_lsb)`` of a residual add, FAILING CLOSED on any that is absent.

    The two multipliers carry each operand into the output's domain. ``bound_lsb`` is the DECLARED
    distance, in output steps, a target's result may lie from the single-rounding reference
    (:meth:`Tensor.residual_add_scaled`): 0 demands that arithmetic exactly, 1 admits a unit that
    rounds each operand before it adds. None of the three has a default. A bound nobody declared
    is a tolerance chosen after the fact, and a multiplier nobody declared is read as 1.0 by every
    engine at once.
    """
    values = []
    for key in ("lhs_scale", "rhs_scale"):
        value = (attrs or {}).get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{op}: a residual add needs `{key}` (a number); got {value!r}")
        values.append(float(value))
    bound = (attrs or {}).get("bound_lsb")
    if isinstance(bound, bool) or not isinstance(bound, int) or bound < 0:
        raise ValueError(f"{op}: a residual add needs `bound_lsb` (a non-negative integer); got {bound!r}")
    unknown = [str(s) for s in (attrs or {}).get("epilogue") or () if str(s) not in RESIDUAL_ADD_STAGES]
    if unknown:
        raise ValueError(f"{op}: a residual add carries only {list(RESIDUAL_ADD_STAGES)}; got {unknown}")
    return values[0], values[1], bound


def declared_acc_scale(attrs: dict[str, Any], *, op: str) -> float:
    """The multiplier an ``acc_scale`` stage applies, FAILING CLOSED when the command names none.

    A stage declared with no parameter is not a no-op. Reading an absent multiplier as ``1.0`` makes
    the golden, the reference and the simulator agree with each other on a saturating cast nobody
    asked for, while the backend is handed a stage with nothing to apply -- and the capsule then
    passes for a reason unrelated to what it tests.
    """
    value = (attrs or {}).get("acc_scale")
    if value is None:
        raise ValueError(
            f"{op}: an 'acc_scale' epilogue stage is declared and the command names no `acc_scale` "
            f"multiplier; the stage is not applied with a default"
        )
    return float(value)


def declared_acc_scale_by_column(attrs: dict[str, Any], columns: int, *, op: str) -> list[float]:
    """Read finite float32 multipliers, exactly one for every output column."""
    import math
    import struct

    if "acc_scale" in attrs:
        raise ValueError(f"{op}: select either acc_scale or acc_scale_by_column, not both")
    scales = attrs.get("acc_scale_by_column")
    if not isinstance(scales, list) or not scales:
        raise ValueError(f"{op}: acc_scale_by_column must be a nonempty numeric list")
    if len(scales) != columns:
        raise ValueError(f"{op}: acc_scale_by_column must name one multiplier per output column")
    rounded = []
    for scale in scales:
        if isinstance(scale, bool) or not isinstance(scale, int | float):
            raise ValueError(f"{op}: acc_scale_by_column needs finite numeric multipliers")
        try:
            number = struct.unpack("<f", struct.pack("<f", float(scale)))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{op}: acc_scale_by_column multiplier is not representable as float32") from error
        if not math.isfinite(number):
            raise ValueError(f"{op}: acc_scale_by_column multiplier is not finite float32")
        rounded.append(number)
    return rounded


def apply_declared_acc_scale(value: Tensor, attrs: dict[str, Any], *, op: str) -> Tensor:
    """Apply a declared scalar or column-dependent float32 readout.

    This states arithmetic, not hardware admission or a scale-change schedule.
    """
    if "acc_scale_by_column" not in attrs:
        return value.requant_acc_scale(declared_acc_scale(attrs, op=op))
    scales = declared_acc_scale_by_column(attrs, value.shape[-1] if value.shape else 0, op=op)
    return value.requant_acc_scale_by_column(scales)


def bias_tensor_name(operands: dict[str, Any], attrs: dict[str, Any], *, op: str) -> str:
    """Name of the bias tensor a ``bias_add``/``bias`` stage consumes. One definition, all engines.

    The name is carried in TWO places in this tree, and both are legitimate:

    * ``attributes["bias"]`` -- what a buffer that came through the interface grammar carries.
      ``merlin_iface.commit`` declares ``bias`` as a property, so the ingest lands it beside
      ``epilogue`` in the attributes (see ``interface_emit``/``ir_ingest``).
    * ``operands["bias"]`` -- what a buffer an emitter built directly carries, because there the
      bias is a named operand slot like any other (``linalg_lower``, ``runtime_lowering``, and the
      whole-op ``BIAS_ADD`` whose operand keys the grammar table itself spells ``["src", "bias"]``).

    Reading only one of them is how a bias gets DROPPED IN SILENCE. The reference/simulate engines
    read only the operands while the golden engine read only the attributes, so on a fused
    matmul+bias capsule the golden added the bias and the reference did not -- and the capsule
    failed L0 with every element off by exactly its column's bias, which reads like a rounding or
    ordering defect rather than a stage that never ran.

    FAIL CLOSED when neither carries a name: a declared stage with no operand is not a no-op, it is
    an unanswerable command. Skipping it would make two engines that both skip it AGREE on a value
    neither computed -- the same silent-pass hazard the unknown-stage branches guard against.
    """
    for source in (attrs, operands):
        name = (source or {}).get("bias")
        if name:
            return str(name)
    raise ValueError(
        f"{op} declares a bias epilogue stage but names no bias tensor in either its operands or "
        f"its attributes. The stage cannot be executed and is NOT skipped: a dropped bias makes "
        f"every output element differ from the golden by its own column's bias, which is "
        f"indistinguishable from an arithmetic defect"
    )


def load_command_buffer(path: str | Path) -> dict[str, Any]:
    """Load a command-buffer JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


REQUIRED_KEYS = ("abi_version", "target", "commands")

#: Opcodes whose readout NARROWS the i32 accumulator into a declared container. Kept beside the
#: engines that do the narrowing (``simulator``/``reference`` both route these through
#: ``_narrow_int_readout``; the golden routes them through ``_apply_epilogue``), and asserted equal to
#: that set by ``tests/ir/test_readout_dtype_divergence.py`` so the list cannot drift away from
#: the engines it describes.
NARROWING_OPCODES = frozenset({"COMMIT", "CONV2D", "BIAS_ADD", "RESIDUAL_ADD", "ATTENTION_QK", "ATTENTION_PV"})


def whole_program_entry_bindings(cb: Mapping[str, Any]) -> list[str] | None:
    """Return the source entry operands in their declared positional ABI order.

    A whole-program lowering may deliberately alias an entry argument with a result buffer.  In that
    case the physical tensor has an ``output`` role but still needs the source stimulus, so tensor roles
    cannot define the input boundary.  ``global_program_plan.entry_bindings`` is the compiler's explicit
    source-argument-to-buffer mapping and is therefore authoritative when present.

    ``None`` means the command buffer has no such mapping and a legacy caller may use its old role-based
    convention.  A present but malformed mapping fails closed: silently falling back would run the right
    kernel on operands assigned to the wrong pointer slots.
    """
    abi = cb.get("kernel_abi")
    if not isinstance(abi, Mapping) or abi.get("kind") != "whole_program":
        return None
    params = cb.get("params")
    plan = params.get("global_program_plan") if isinstance(params, Mapping) else None
    if not isinstance(plan, Mapping):
        return None
    bindings = plan.get("entry_bindings")
    if not isinstance(bindings, list) or any(not isinstance(name, str) or not name for name in bindings):
        raise ValueError(
            "whole-program global_program_plan.entry_bindings must be a list of non-empty tensor "
            "names in source entry-argument order"
        )
    duplicates = sorted({name for name in bindings if bindings.count(name) > 1})
    if duplicates:
        raise ValueError(
            "whole-program global_program_plan.entry_bindings repeats physical buffer(s) "
            f"{duplicates}; distinct source operands cannot be initialized through one pointer"
        )
    tensors = cb.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("whole-program global_program_plan.entry_bindings is present but tensors is not a mapping")
    args = abi.get("args")
    if not isinstance(args, list):
        raise ValueError(
            "whole-program global_program_plan.entry_bindings is present but kernel_abi.args is not a list"
        )
    access_by_name: dict[str, list[Any]] = {}
    for arg in args:
        if isinstance(arg, Mapping) and isinstance(arg.get("tensor"), str):
            access_by_name.setdefault(str(arg["tensor"]), []).append(arg.get("access"))
    for name in bindings:
        if name not in tensors:
            raise ValueError(f"whole-program entry binding {name!r} has no declared tensor buffer")
        accesses = access_by_name.get(name, [])
        if len(accesses) != 1:
            raise ValueError(f"whole-program entry binding {name!r} must occupy exactly one kernel_abi.args slot")
        if accesses[0] not in ("read", "readwrite"):
            raise ValueError(
                f"whole-program entry binding {name!r} must have read or readwrite ABI access, got {accesses[0]!r}"
            )
    return list(bindings)


def row_major_view_shape(cb, operands, attrs, source_shape, source_dtype, *, produced):
    """Validate a byte-preserving final view of one already written buffer.

    This metadata operation neither copies nor aliases distinct external slots.
    Positive static row-major shapes and identical element width/order preserve
    every byte. An input, fresh output, transpose or cast is not a view here.
    """
    from math import prod

    if set(operands) != {"src", "dst"} or operands["src"] != operands["dst"]:
        raise ValueError("ROW_MAJOR_VIEW must retain one physical buffer identity")
    name = operands["src"]
    if name not in produced or set(attrs) != {"source_shape", "shape"}:
        raise ValueError("ROW_MAJOR_VIEW requires an already produced value and exact shape fields")
    before, after = attrs["source_shape"], attrs["shape"]
    if any(
        type(shape) is not list or not shape or len(shape) > 32 or any(type(n) is not int or n <= 0 for n in shape)
        for shape in (before, after)
    ):
        raise ValueError("ROW_MAJOR_VIEW requires positive static shapes")
    if tuple(before) != tuple(source_shape) or prod(before) != prod(after):
        raise ValueError("ROW_MAJOR_VIEW changes the source extent or element count")
    declared = cb.get("tensors", {}).get(name, {})
    if declared.get("shape") != after or declared.get("dtype") != source_dtype or declared.get("role") != "output":
        raise ValueError("ROW_MAJOR_VIEW differs from complete output shape/type ownership")
    return tuple(after)


def validate_command_buffer(cb: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty == valid).

    Beyond the required keys, this catches the one structural error the JSON schema cannot express: an
    operand slot holds the NAME of a buffer, and the schema types it as a bare string, so a value that
    names nothing at all still validates. A submission that emitted commands referencing shape strings
    (``dst: "16x16"``) with no ``tensors`` declared passed schema validation, was told it was valid, and
    was then rejected downstream for a constraint the contract never stated -- so it spent its session
    guessing spellings instead of writing a compiler.

    The check is deliberately the weakest one that is certainly true: commands that reference operands
    need SOMETHING to reference. Names produced by earlier commands (an accumulator, a committed
    intermediate) are legitimately absent from ``tensors``, so per-name resolution is NOT asserted here --
    only that a computing command buffer declares at least one buffer to compute over."""
    problems: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in cb:
            problems.append(f"missing key '{k}'")
    for i, cmd in enumerate(cb.get("commands", [])):
        if "opcode" not in cmd:
            problems.append(f"command {i} missing 'opcode'")
    cmds = cb.get("commands") or []
    referenced = sorted({str(v) for c in cmds for v in (c.get("operands") or {}).values() if isinstance(v, str) and v})
    # A NARROWING COMMAND MUST DECLARE ITS OUTPUT CONTAINER, BY NAME.
    #
    # The attribute is optional in the JSON schema and every engine has a default, so omitting it has
    # always produced a well-formed buffer that computes SOMETHING. The trouble is which something: the
    # runtime engines defaulted to `i8` and narrowed on an exact match while the golden defaulted to
    # `i32` and narrowed any width, and the two disagreed for 77 days -- a correct backend that simply
    # left the attribute out failed L0 on 85 of 130 capsules, with a message blaming the backend. The
    # engines agree now, but agreement on a DEFAULT is a weaker guarantee than a declaration: it makes
    # the buffer's meaning depend on a convention the submission never stated, and the next divergence
    # would be just as silent as the last.
    #
    # So this is reported as a problem rather than defaulted. It names the command index and the
    # opcode, because the failure this replaces was a numeric mismatch hundreds of lines downstream
    # with nothing pointing back at the omission. All 421 narrowing ops in the shipped corpus already
    # declare it, so this states an invariant that already holds rather than imposing a new one.
    for i, cmd in enumerate(cmds):
        attrs = cmd.get("attributes") or {}
        if "acc_scale_by_column" in attrs:
            if "acc_scale" not in (attrs.get("epilogue") or []):
                problems.append(f"command {i}: acc_scale_by_column requires its acc_scale epilogue stage")
            scales = attrs.get("acc_scale_by_column")
            dst = (cmd.get("operands") or {}).get("dst")
            shape = ((cb.get("tensors") or {}).get(dst) or {}).get("shape")
            columns = shape[-1] if isinstance(shape, list) and shape else len(scales) if isinstance(scales, list) else 0
            try:
                declared_acc_scale_by_column(attrs, columns, op=f"command {i}")
            except ValueError as error:
                problems.append(str(error))
        if str(cmd.get("opcode")) in NARROWING_OPCODES:
            attrs = cmd.get("attributes") or {}
            if "output_dtype" not in attrs:
                problems.append(
                    f"command {i} ({cmd.get('opcode')}) narrows the accumulator but declares no "
                    f"'output_dtype'. The attribute decides the readout width, every engine has to "
                    f"agree about it, and a default is a convention this buffer never stated -- "
                    f'declare it (e.g. "i32" to read the accumulator out whole, "i8" to '
                    f"saturate to a byte)"
                )

    if referenced and not (cb.get("tensors") or {}) and not cb.get("declined"):
        problems.append(
            f"commands reference operand name(s) {referenced[:6]}"
            f"{' ...' if len(referenced) > 6 else ''} but the command buffer declares no 'tensors'. "
            f"An operand slot holds the NAME of a tensor declared in 'tensors' (e.g. \"Y0\"), not a "
            f"shape, a type, or a dimension list"
        )

    kernel_abi = cb.get("kernel_abi")
    if isinstance(kernel_abi, dict) and kernel_abi.get("kind") == "whole_program":
        args = kernel_abi.get("args")
        arg_names = [arg.get("tensor") for arg in args if isinstance(arg, dict)] if isinstance(args, list) else []
        declared = [name for name, spec in (cb.get("tensors") or {}).items() if isinstance(spec, dict)]
        missing = [name for name in declared if name not in arg_names]
        if missing:
            problems.append(
                f"whole-program kernel_abi.args omits declared tensor buffer(s) {missing}; the "
                f"runner cannot bind an interface buffer that is absent from the pointer boundary"
            )
        duplicates = sorted({name for name in arg_names if name and arg_names.count(name) > 1})
        if duplicates:
            problems.append(
                f"whole-program kernel_abi.args has duplicate pointer slot(s) for tensor(s) "
                f"{duplicates}; one declared buffer must occupy exactly one ABI position"
            )
        undeclared = [name for name in arg_names if name not in (cb.get("tensors") or {})]
        if undeclared:
            problems.append(
                f"whole-program kernel_abi.args names tensor(s) {undeclared} with no declared tensor "
                f"buffer; the runner never allocates an implicit pointer"
            )
        access_of = (
            {arg.get("tensor"): arg.get("access") for arg in args if isinstance(arg, dict)}
            if isinstance(args, list)
            else {}
        )
        unread_inputs = [
            name
            for name, spec in (cb.get("tensors") or {}).items()
            if isinstance(spec, dict)
            and spec.get("role") in ("input", "weight", "bias", "scale")
            and access_of.get(name) not in ("read", "readwrite")
        ]
        if unread_inputs:
            problems.append(
                f"whole-program kernel_abi tensor(s) {unread_inputs} have an input-like role but no "
                f"read access; the runner must supply the declared model stimulus to the kernel"
            )
        unwritable_results = [
            name
            for name, spec in (cb.get("tensors") or {}).items()
            if isinstance(spec, dict)
            and spec.get("role") in ("output", "intermediate")
            and access_of.get(name) not in ("write", "readwrite")
        ]
        if unwritable_results:
            problems.append(
                f"whole-program kernel_abi tensor(s) {unwritable_results} have an output-like role but "
                f"no write access; the submitted kernel must produce those buffers"
            )
        bad_outputs = [
            name for name in (kernel_abi.get("outputs") or []) if access_of.get(name) not in ("write", "readwrite")
        ]
        if bad_outputs:
            problems.append(
                f"whole-program kernel_abi output tensor(s) {bad_outputs} do not have write access in "
                f"kernel_abi.args; a reported result must be produced by the submitted kernel"
            )
        declared_outputs = [
            name
            for name, spec in (cb.get("tensors") or {}).items()
            if isinstance(spec, dict) and spec.get("role") == "output"
        ]
        reported_outputs = kernel_abi.get("outputs") or []
        if set(reported_outputs) != set(declared_outputs):
            problems.append(
                f"whole-program kernel_abi must report exactly the declared model outputs "
                f"{declared_outputs}, got {reported_outputs}; an intermediate is not a substitute for "
                f"the interface result"
            )
        derived = harness_derived_tensors(cb)
        if derived:
            problems.append(
                f"whole-program kernel_abi cannot use harness derivation recipe(s) "
                f"{sorted(set(derived.values()))} for {sorted(derived)}; all model work must execute "
                f"inside the measured submitted kernel"
            )
    return problems


def conv_out_dims(H: int, W: int, kh: int, kw: int, stride, padding, dilation) -> tuple[int, int]:
    sh, sw = stride
    pt, pl, pb, pr = padding
    dh, dw = dilation
    Ho = (H + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    Wo = (W + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    return Ho, Wo


def conv_im2col(ifm: Tensor, *, kh: int, kw: int, ci: int, stride, padding, dilation, layout: str = "nhwc") -> Tensor:
    """Build the [N*Ho*Wo, Kh*Kw*Ci] im2col matrix from an NHWC activation (zero-pad OOB taps).

    This is the single source of truth shared by the runner harness/reference/simulate (via
    :func:`materialize_inputs`) and the capsule golden. Column order = (kh, kw, ci), matching the
    weight packing [Kh*Kw*Ci, Co].
    """
    if layout != "nhwc":
        raise ValueError(f"conv_im2col layout {layout!r} unsupported (nhwc only)")
    N, H, W, C = ifm.shape
    if C != ci:
        raise ValueError(f"conv_im2col channel mismatch: ifm C={C} != ci={ci}")
    sh, sw = stride
    pt, pl, _, _ = padding
    dh, dw = dilation
    Ho, Wo = conv_out_dims(H, W, kh, kw, stride, padding, dilation)
    a = ifm.data
    rows: list[int] = []

    def at(n, y, x, c):
        if 0 <= y < H and 0 <= x < W:
            return a[((n * H + y) * W + x) * C + c]
        return 0

    for n in range(N):
        for oy in range(Ho):
            for ox in range(Wo):
                by, bx = oy * sh - pt, ox * sw - pl
                for ky in range(kh):
                    for kx in range(kw):
                        for c in range(ci):
                            rows.append(at(n, by + ky * dh, bx + kx * dw, c))
    return Tensor((N * Ho * Wo, kh * kw * ci), rows, ifm.dtype)


#: Every ``params`` key under which a command buffer may DECLARE that one of its tensors is derived
#: from another by the HARNESS rather than produced by a command. Each entry names its result in
#: ``target`` and its source in ``source``.
#:
#: This tuple exists so the two readers of that declaration cannot drift apart.
#: :func:`materialize_inputs` BUILDS the derived tensors from it (that is the mechanism's legitimate
#: job: the reference, the simulator and the device must be handed byte-identical stimulus, or the
#: numeric comparison between them compares three different inputs), and
#: :func:`harness_derived_tensors` REPORTS which tensors were built that way, which is what a caller
#: needs in order to ask whether the emitted program computed its own operands. A key added to the
#: materializer and not to this tuple would silently become invisible to that question.


def operand_flow(cb: dict[str, Any]) -> tuple[set[str], set[str]]:
    """``(names WRITTEN by a command, every name any command REFERENCES)``.

    Answers "did the emitted program compute this tensor, or was it handed to it?". Purely
    structural — no opcode, dtype or target fact is read.

    A write is recognised BY KEY SPELLING ONLY (:data:`PRODUCING_KEYS`), deliberately without
    :func:`_produces`' fallback to the named tensor's declared ``role``. That fallback is right for
    :func:`dataflow_operands`, which is trying to find a buffer's result and must tolerate a
    destination spelled ``result`` or ``y``. It is wrong here, because a caller asking this question is
    asking whether a COMMAND produced the tensor, and ``role: output`` on a tensor that appears only in
    read slots would answer "yes" on the strength of a label the buffer wrote about itself — which is
    exactly the claim under examination. Read positions and written positions are therefore separated
    by the ABI's own write-slot vocabulary and by nothing else.
    """
    written: set[str] = set()
    referenced: set[str] = set()
    for cmd in cb.get("commands") or []:
        for key, name in (cmd.get("operands") or {}).items():
            if not isinstance(name, str) or not name:
                continue
            referenced.add(name)
            if key in PRODUCING_KEYS:
                written.add(name)
    return written, referenced


#: Where a command buffer may declare the inclusive range its deterministic stimulus is drawn from.
#: ABSENT means the historical default, so every existing buffer materializes byte-identically.
STIMULUS_RANGE_KEY = "stimulus_range"

#: The historical default. Non-negative, which is exactly why a ReLU capsule could not detect whether
#: its activation was applied: with weights and inputs both in 0..3 the accumulator is never negative
#: and ``max(0, x)`` is the identity on every value such a program can produce.
DEFAULT_STIMULUS_RANGE = (0, 3)

#: The signed counterpart of the default: the same magnitude with its two's-complement negative
#: half. A capsule whose operation has any sign-sensitive stage should declare this (or wider), so
#: accumulators go negative and the stage's behaviour on a negative value is observable. Derived
#: from the default rather than chosen, so widening one widens the other.
SIGNED_STIMULUS_RANGE = (-(DEFAULT_STIMULUS_RANGE[1] + 1), DEFAULT_STIMULUS_RANGE[1])


def stimulus_range(cb: Mapping[str, Any]) -> tuple[int, int]:
    """The inclusive ``(lo, hi)`` this buffer's stimulus is drawn from.

    Declared under ``params.stimulus_range``; absent, the historical default. A malformed
    declaration RAISES rather than falling back, because a silently ignored range would fill one
    side of a comparison from a different distribution than the other and report the difference as a
    numeric failure of the datapath.
    """
    params = cb.get("params") if isinstance(cb.get("params"), Mapping) else {}
    declared = params.get(STIMULUS_RANGE_KEY)
    if declared is None:
        return DEFAULT_STIMULUS_RANGE
    if (
        not isinstance(declared, Sequence)
        or isinstance(declared, (str, bytes))
        or len(declared) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in declared)
    ):
        raise ValueError(
            f"{STIMULUS_RANGE_KEY} must be a two-element [lo, hi] of integers, got {declared!r}; a "
            f"malformed range is refused rather than defaulted, because filling one side of a "
            f"comparison from a different distribution reports a stimulus difference as a datapath "
            f"failure"
        )
    lo, hi = int(declared[0]), int(declared[1])
    if hi < lo:
        raise ValueError(f"empty {STIMULUS_RANGE_KEY} [{lo}, {hi}]")
    return lo, hi


def materialize_inputs(cb: dict[str, Any], inputs: dict[str, Any] | None = None) -> dict[str, Tensor]:
    """Create the leaf input tensors declared in the command buffer's ``tensors`` table.

    A tensor is a *leaf* (materialized here) when its role is input/weight/bias, i.e. it is
    not produced by a command. ``inputs`` may supply explicit nested-list data per name.

    If the command buffer declares ``params.im2col_recipes``, each derived activation is built by
    gathering conv windows from its source leaf (so a compiler-lowered conv2d's im2col activation is
    materialized identically for the reference, the simulator, and the device harness).
    """
    inputs = inputs or {}
    for command in cb.get("commands", []):
        if command.get("opcode") == "ROW_MAJOR_VIEW":
            source = command.get("operands", {}).get("src")
            if cb.get("tensors", {}).get(source, {}).get("role") != "output":
                raise ValueError("ROW_MAJOR_VIEW cannot mutate a leaf input")
    produced = set()
    for cmd in cb.get("commands", []):
        ops = cmd.get("operands", {})
        for key in ("dst",):
            if key in ops:
                produced.add(ops[key])
    lo, hi = stimulus_range(cb)
    env: dict[str, Tensor] = {}
    for name, spec in cb.get("tensors", {}).items():
        if name in produced:
            continue
        shape = tuple(spec["shape"])
        dtype = spec.get("dtype", "i8")
        if name in inputs:
            flat = _flatten(inputs[name])
            env[name] = Tensor(shape, flat, dtype)
        elif isinstance(spec.get("data"), (list, tuple)):
            # VALUES CARRIED ON THE DECLARATION WIN OVER ANY FILL. A caller that has already
            # materialized this leaf -- from a capsule's own declaration, say -- injects the exact
            # numbers here, so the device computes on the same bytes the reference did rather than on
            # a second fill that merely agrees by construction. Agreeing by construction is what
            # breaks the moment either side's fill parameters change.
            env[name] = Tensor(shape, _flatten(spec["data"]), dtype)
        else:
            env[name] = Tensor.deterministic(name, shape, dtype, lo, hi)
    # additive: overwrite derived im2col activations from their source leaf. Keyed off the shared
    # DERIVATION_RECIPE_KEYS so a recipe kind cannot be materialized here while staying invisible to
    # harness_derived_tensors (which is what decides whether the emitted program owes this gather).
    for r in cb.get("params", {}).get(IM2COL_RECIPE_KEY, []):
        src = env[r["source"]]
        env[r["target"]] = conv_im2col(
            src,
            kh=int(r["kh"]),
            kw=int(r["kw"]),
            ci=int(r["ci"]),
            stride=tuple(r.get("stride", [1, 1])),
            padding=tuple(r.get("padding", [0, 0, 0, 0])),
            dilation=tuple(r.get("dilation", [1, 1])),
            layout=r.get("layout", "nhwc"),
        )
    return env


#: Operand keys that PRODUCE a tensor. This is the command-buffer ABI's own vocabulary (the buffer
#: declares its ``abi_version``), not a fact about any target — every command-buffer endpoint names its
#: result with one of these, whatever its opcodes are called. Kept next to :func:`materialize_inputs`,
#: which uses the same notion of "produced" to decide what is a leaf.
#: Operand keys that name a WRITE by spelling. Not exhaustive by construction -- the schema lets a
#: buffer spell its destination `result`, `y`, or anything else -- so :func:`dataflow_operands` also
#: accepts a key whose tensor DECLARES `role: output`. See `_produces` there.


def dataflow_operands(cb: dict[str, Any]) -> tuple[list[str], str] | None:
    """``(leaf_input_names, final_output_name)`` of a command buffer, derived by DATAFLOW alone.

    Target-agnostic and opcode-agnostic by construction: a declared tensor that some command CONSUMES and
    no command PRODUCES is a leaf input; the last produced tensor that is itself declared is the output;
    a tensor that is both produced and consumed is an intermediate and is neither. Nothing here reads a
    target fact, an opcode meaning, or an op-specific shape rule, so it works for any endpoint that
    speaks this ABI — SIMT, systolic/NPU, or otherwise.

    Why dataflow rather than the declared ``role``: role cannot decide it. Measured on a real fused
    flash-attention buffer, THREE tensors declare ``role: output`` (``S``, ``P``, ``Y0``) because each is
    produced by some command, and two of them are intermediates feeding the next stage. Reading roles
    alone yields three candidate outputs and no way to choose; the dataflow yields exactly ``Q, K, V`` in
    and ``Y0`` out.

    This is the general case the single-op binders cannot express. A binder that pattern-matches "one
    matmul" returns ``None`` on a CHAIN, which is what left every fused capsule (attention_qk -> softmax
    -> matmul -> commit, rmsnorm -> matmul, chained matmuls) unbindable and therefore ungradeable.

    Returns ``None`` when the buffer carries no commands, no declared ``tensors``, or no produced tensor
    that is declared — i.e. when dataflow genuinely cannot answer, never a guess.

    NOTE ON ORDER: the returned inputs are in first-consumption order (the order the command stream
    first reads them), which is deterministic and reproducible. It is NOT authoritative for a kernel ABI
    — the emitted kernel's own signature is. A caller that must match a kernel signature should reorder
    these by matching declared shapes to that signature rather than trusting this order.
    """
    tensors = cb.get("tensors") or {}
    commands = cb.get("commands") or []
    if not tensors or not commands:
        return None

    produced: list[str] = []
    consumed: list[str] = []
    for cmd in commands:
        ops = cmd.get("operands") or {}
        for key, name in ops.items():
            if not isinstance(name, str):
                continue
            (produced if _produces(key, name, tensors) else consumed).append(name)

    produced_set = set(produced)
    # leaves, in first-consumption order, de-duplicated
    seen: set[str] = set()
    leaves: list[str] = []
    for name in consumed:
        if name in produced_set or name in seen or name not in tensors:
            continue
        seen.add(name)
        leaves.append(name)

    # The output is the LAST produced tensor that is DECLARED. An internal accumulator is produced but
    # not declared (measured: `acc_Y0` is a dst and absent from `tensors`), so walking backwards over the
    # produced list and taking the first declared hit lands on the committed result rather than on it.
    out = next((n for n in reversed(produced) if n in tensors), None)
    if out is None or not leaves:
        return None
    return leaves, out


def _flatten(nested) -> list[int]:
    """Flatten a nested list of ANY rank to its scalars, row-major.

    This peeled exactly one level, which is right for a rank-2 operand and silently wrong for a
    rank-3 one: a batched activation came back as a list of LISTS, and ``Tensor`` then rejected it for
    having 4 elements where its shape needs 12. That reads as a malformed input rather than as an
    unsupported rank, which is how a rank the contract admits stays unreachable -- the target contract
    declares rank-3 contractions legal and nothing downstream could materialize one.

    Recursing costs nothing here and is rank-agnostic, so the next rank does not need another edit.
    """
    out: list[int] = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


# ---- metrics -----------------------------------------------------------------------------------
# Common runtime metrics (matches src/merlin/resources/schemas/metrics.schema.yaml).
#
# A plain accumulator the simulator updates as it executes a command buffer. ``as_dict``
# emits the common metric vocabulary so results are comparable across targets/backends.

COMMON_METRIC_NAMES = (
    "cycles",
    "bytes_moved",
    "bytes_read",
    "bytes_written",
    "command_count",
    "dispatch_count",
    "pack_count",
    "resident_hits",
    "resident_misses",
    "evictions",
    "accumulator_commits",
    "intermediate_write_bytes",
)


@dataclass
class Metrics:
    cycles: int = 0
    bytes_moved: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    command_count: int = 0
    dispatch_count: int = 0
    pack_count: int = 0
    resident_hits: int = 0
    resident_misses: int = 0
    evictions: int = 0
    accumulator_commits: int = 0
    intermediate_write_bytes: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
