"""Exact definitions of the affine-map and data-movement command-buffer opcodes.

``AFFINE_MAP``, the movement forms of ``MOVEMENT``, ``CONCAT``, ``INSERT_SLICE`` and ``FILL`` are defined
ONCE, here, and the reference engine, the command-buffer simulator and the capsule golden all call these
functions. Host requantization uses one exact rational RNE boundary shared by every engine.

Every attribute an opcode needs is read and validated here, and an absent required attribute or an
unknown one is refused by name (:class:`IntegerOpError`) rather than defaulted: an engine that applied a
default the buffer never stated would agree with another engine that applied the same default, and the
capsule would pass having proved nothing about the attribute it omitted.

The numeric form of ``AFFINE_MAP`` is :class:`merlin.targetgen.recipe_numerics.IntegerRecipe`: integer
weights and an int32 constant in the accumulator, ReLU on the accumulator, then ONE readout.
"""

from __future__ import annotations

from fractions import Fraction
from math import lcm as integer_lcm
from typing import Any

import numpy as np

from merlin.runtime.exact_readout import _integer, requantize_int32
from merlin.targetgen.exact_arithmetic import rational

from .tensor import Tensor


class IntegerOpError(ValueError):
    """A command's operands or attributes do not state a well-defined integer operation."""


#: Integer container ranges an output may saturate into.
_RANGES = {"i8": (-128, 127), "i16": (-(2**15), 2**15 - 1), "i32": (-(2**31), 2**31 - 1)}
_EXACT_ATTRS = {"scale_numerator", "scale_denominator", "zero_point", "rounding"}
_AFFINE_ATTRS = frozenset(
    {"weights", "constant", "activation", "readout_scale", "output_dtype", "quant_min", "quant_max"} | _EXACT_ATTRS
)
_MOVE_FORMS = {
    "identity": frozenset(),
    "permute": frozenset({"perm"}),
    "slice": frozenset({"offsets", "sizes", "strides"}),
    "pad": frozenset({"low", "high", "mode"}),
    "broadcast": frozenset(),
    "repeat": frozenset({"repeats"}),
}
_MOVE_COMMON = frozenset(
    {"form", "output_dtype", "semantic", "scale", "quant_min", "quant_max", "value_code"} | _EXACT_ATTRS
)


def _array(t: Tensor) -> np.ndarray:
    return np.asarray(t.data, dtype=np.int64).reshape(t.shape)


def _tensor(values: np.ndarray, dtype: str) -> Tensor:
    return Tensor(tuple(int(d) for d in values.shape), [int(v) for v in values.reshape(-1)], dtype)


def _int_list(attrs: dict[str, Any], key: str, *, op: str, length: int | None = None) -> list[int]:
    value = attrs.get(key)
    if not isinstance(value, list) or any(type(v) is not int for v in value):
        raise IntegerOpError(f"{op}: {key} must be a list of integers, got {value!r}")
    if length is not None and len(value) != length:
        raise IntegerOpError(f"{op}: {key} needs {length} entries (one per dimension), got {len(value)}")
    return value


def _range(attrs: dict[str, Any], dtype: str, *, op: str) -> tuple[int, int]:
    if dtype not in _RANGES:
        raise IntegerOpError(f"{op}: output_dtype must be one of {sorted(_RANGES)}, got {dtype!r}")
    low, high = _RANGES[dtype]
    qmin, qmax = attrs.get("quant_min", low), attrs.get("quant_max", high)
    if type(qmin) is not int or type(qmax) is not int or not low <= qmin <= qmax <= high:
        raise IntegerOpError(f"{op}: quant_min/quant_max must be integers within {dtype}'s range")
    return qmin, qmax


def _scale(attrs, legacy_key):
    if "scale_numerator" in attrs or "scale_denominator" in attrs:
        try:
            n, d = attrs["scale_numerator"], attrs["scale_denominator"]
            if isinstance(n, list):
                (n,) = n
            if isinstance(d, list):
                (d,) = d
            if _integer(d) <= 0:
                raise ValueError("nonpositive scale denominator")
            return Fraction(_integer(n), _integer(d))
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegerOpError("paired scalar integer scale numerator/positive denominator required") from exc
    return rational(attrs[legacy_key]) if legacy_key in attrs else None


def _requantize(values: np.ndarray, attrs: dict[str, Any], dtype: str, *, op: str, scale: Any = None) -> np.ndarray:
    """Exact host RNE, zero-point addition and clamp; no scaled MVIN semantics."""
    coefficient = _scale(attrs, "scale") if scale is None else rational(scale)
    if coefficient is None:
        return values
    qmin, qmax = _range(attrs, dtype, op=op)
    try:
        return requantize_int32(
            values,
            scale_numerator=coefficient.numerator,
            scale_denominator=coefficient.denominator,
            zero_point=attrs.get("zero_point", 0),
            quant_min=qmin,
            quant_max=qmax,
            rounding=attrs.get("rounding", "half_to_even"),
        )
    except ValueError as exc:
        raise IntegerOpError(f"{op}: {exc}") from exc


def affine_map(x: Tensor, y: Tensor | None, attrs: dict[str, Any], *, op: str = "AFFINE_MAP") -> Tensor:
    """Exact integer affine map followed by one rational host quantization."""
    from merlin.targetgen.recipe_numerics import IntegerRecipe, RecipeNumericsError

    unknown = sorted(set(attrs) - _AFFINE_ATTRS - {"name"})
    if unknown:
        raise IntegerOpError(f"{op}: unknown attribute(s) {unknown}")
    missing = sorted({"weights", "constant", "activation", "output_dtype"} - set(attrs))
    if missing:
        raise IntegerOpError(f"{op}: required attribute(s) {missing} absent; no default is applied")
    weights = _int_list(attrs, "weights", op=op)
    if len(weights) != (1 if y is None else 2):
        raise IntegerOpError(f"{op}: {len(weights)} weight(s) for {1 if y is None else 2} operand(s)")
    constant, scale, dtype = attrs["constant"], _scale(attrs, "readout_scale"), str(attrs["output_dtype"])
    if type(constant) is not int or scale is None:
        raise IntegerOpError(f"{op}: constant must be an integer and readout_scale a number")
    qmin, qmax = _range(attrs, dtype, op=op)
    a = _array(x)
    operands = [a]
    if y is not None:
        b = _array(y)
        if b.ndim > a.ndim or any(bd not in (1, ad) for bd, ad in zip(b.shape[::-1], a.shape[::-1], strict=False)):
            raise IntegerOpError(f"{op}: y{list(b.shape)} does not broadcast to x{list(a.shape)}")
        if a.ndim and (b.ndim == 0 or b.shape[-1] != a.shape[-1]):
            raise IntegerOpError(f"{op}: y may broadcast along leading dims only, not the innermost one")
        operands.append(np.broadcast_to(b, a.shape))
    try:
        if dtype == "i32":
            if scale != 1 or attrs.get("zero_point", 0) != 0:
                raise IntegerOpError(f"{op}: an i32 result has no readout; readout_scale must be 1.0")
            accumulator = sum(w * v for w, v in zip(weights, operands, strict=True)) + constant
            if attrs["activation"] == "relu":
                accumulator = np.maximum(accumulator, 0)
            elif attrs["activation"] != "none":
                raise IntegerOpError(f"{op}: activation must be none or relu")
            if np.any(accumulator < qmin) or np.any(accumulator > qmax):
                raise IntegerOpError(f"{op}: the int32 accumulator overflowed its declared range")
            return _tensor(np.asarray(accumulator), dtype)
        recipe = IntegerRecipe(
            weights=tuple(weights),
            constant=constant,
            readout_scale=scale,
            activation=str(attrs["activation"]),
            qmin=qmin,
            qmax=qmax,
        )
        # Preserve the unrounded accumulator for the one host quantize boundary.
        accumulator = sum(w * v for w, v in zip(weights, operands, strict=True)) + constant
        if recipe.activation == "relu":
            accumulator = np.maximum(accumulator, 0)
        return _tensor(
            _requantize(
                accumulator,
                {**attrs, "scale_numerator": str(scale.numerator), "scale_denominator": str(scale.denominator)},
                dtype,
                op=op,
            ),
            dtype,
        )
    except RecipeNumericsError as exc:
        raise IntegerOpError(f"{op}: {exc}") from exc


def move(src: Tensor, attrs: dict[str, Any], dst_shape: list[int] | None, *, op: str = "MOVEMENT") -> Tensor:
    """A single-source ``MOVEMENT``: identity, permute, slice, pad, broadcast or repeat (+ optional scale)."""
    form = str(attrs.get("form", "identity"))
    if form not in _MOVE_FORMS:
        raise IntegerOpError(f"{op}: form must be one of {sorted(_MOVE_FORMS)}, got {form!r}")
    unknown = sorted(set(attrs) - _MOVE_COMMON - _MOVE_FORMS[form] - {"name"})
    if unknown:
        raise IntegerOpError(f"{op}: attribute(s) {unknown} do not belong to form {form!r}")
    missing = sorted(_MOVE_FORMS[form] - set(attrs))
    if missing:
        raise IntegerOpError(f"{op}: form {form!r} requires {missing}; no default is applied")
    if "value_code" in attrs and not (form == "pad" and attrs.get("mode") == "constant"):
        raise IntegerOpError(f"{op}: value_code belongs to a constant pad only")
    dtype = str(attrs.get("output_dtype", src.dtype))
    a = _array(src)
    rank = a.ndim
    if form == "identity":
        out = a
    elif form == "permute":
        perm = _int_list(attrs, "perm", op=op, length=rank)
        if sorted(perm) != list(range(rank)):
            raise IntegerOpError(f"{op}: perm {perm} is not a permutation of {rank} dims")
        out = np.transpose(a, perm)
    elif form == "slice":
        offsets = _int_list(attrs, "offsets", op=op, length=rank)
        sizes = _int_list(attrs, "sizes", op=op, length=rank)
        strides = _int_list(attrs, "strides", op=op, length=rank)
        if any(s < 1 for s in strides) or any(
            o < 0 or n < 0 or o + (n - 1) * s >= d
            for o, n, s, d in zip(offsets, sizes, strides, a.shape, strict=True)
            if n
        ):
            raise IntegerOpError(f"{op}: slice is out of bounds or has a nonpositive stride")
        out = a[tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, sizes, strides, strict=True))]
    elif form == "pad":
        low = _int_list(attrs, "low", op=op, length=rank)
        high = _int_list(attrs, "high", op=op, length=rank)
        if any(v < 0 for v in low + high):
            raise IntegerOpError(f"{op}: pad widths must be nonnegative")
        mode = attrs["mode"]
        widths = list(zip(low, high, strict=True))
        if mode == "constant":
            code = attrs.get("value_code")
            if type(code) is not int:
                raise IntegerOpError(f"{op}: a constant pad requires an integer value_code (the quantized pad value)")
            out = np.pad(a, widths, mode="constant", constant_values=code)
        elif mode == "reflect":
            if any(lo >= d or hi >= d for (lo, hi), d in zip(widths, a.shape, strict=True)):
                raise IntegerOpError(f"{op}: a reflect pad must be narrower than the dimension it reflects")
            out = np.pad(a, widths, mode="reflect")
        elif mode == "replicate":
            out = np.pad(a, widths, mode="edge")
        else:
            raise IntegerOpError(f"{op}: pad mode must be constant, reflect or replicate, got {mode!r}")
    elif form == "broadcast":
        if dst_shape is None:
            raise IntegerOpError(f"{op}: a broadcast needs its declared destination shape")
        target = tuple(int(d) for d in dst_shape)
        if rank and (len(target) < rank or target[-1] != a.shape[-1]):
            raise IntegerOpError(f"{op}: a broadcast replays rows along leading dims only, not the innermost one")
        try:
            out = np.broadcast_to(a, target)
        except ValueError as exc:
            raise IntegerOpError(f"{op}: {list(a.shape)} does not broadcast to {list(target)}") from exc
    else:  # repeat
        repeats = attrs["repeats"]
        if not isinstance(repeats, list) or len(repeats) < rank or any(type(r) is not int or r < 1 for r in repeats):
            raise IntegerOpError(f"{op}: repeats must list a positive count per dim (at least the source rank)")
        out = np.tile(a, repeats)
    out = _requantize(np.asarray(out), attrs, dtype, op=op)
    if dst_shape is not None and tuple(out.shape) != tuple(int(d) for d in dst_shape):
        raise IntegerOpError(f"{op}: the {form} result {list(out.shape)} differs from the declared {list(dst_shape)}")
    return _tensor(np.ascontiguousarray(out), dtype)


def concat(srcs: list[Tensor], attrs: dict[str, Any], *, op: str = "CONCAT") -> Tensor:
    """``dst = cat([requant(src_i, scales[i])], axis)``; ``scales`` absent means shared qparams."""
    unknown = sorted(set(attrs) - {"axis", "output_dtype", "scales", "quant_min", "quant_max", "name"})
    if unknown:
        raise IntegerOpError(f"{op}: unknown attribute(s) {unknown}")
    if len(srcs) < 2:
        raise IntegerOpError(f"{op}: needs at least two sources")
    axis = attrs.get("axis")
    if type(axis) is not int:
        raise IntegerOpError(f"{op}: axis is required (an integer); no default is applied")
    dtype = str(attrs.get("output_dtype", srcs[0].dtype))
    scales = attrs.get("scales")
    if scales is not None and (not isinstance(scales, list) or len(scales) != len(srcs)):
        raise IntegerOpError(f"{op}: scales must list one scale per source")
    parts = [
        _requantize(_array(s), attrs, dtype, op=op, scale=None if scales is None else float(scales[i]))
        for i, s in enumerate(srcs)
    ]
    try:
        return _tensor(np.concatenate(parts, axis=axis), dtype)
    except (ValueError, np.exceptions.AxisError) as exc:
        raise IntegerOpError(f"{op}: {exc}") from exc


def insert_slice(dest: Tensor, src: Tensor, attrs: dict[str, Any], *, op: str = "INSERT_SLICE") -> Tensor:
    """``dst = dest`` with ``dst[offsets + i*strides] = src`` (slice_scatter / select_scatter)."""
    unknown = sorted(set(attrs) - {"offsets", "strides", "output_dtype", "name"})
    if unknown:
        raise IntegerOpError(f"{op}: unknown attribute(s) {unknown}")
    base, part = _array(dest), _array(src)
    if base.ndim != part.ndim:
        raise IntegerOpError(f"{op}: src rank {part.ndim} differs from dest rank {base.ndim}; keep the size-1 dim")
    offsets = _int_list(attrs, "offsets", op=op, length=base.ndim)
    strides = _int_list(attrs, "strides", op=op, length=base.ndim)
    if any(s < 1 for s in strides) or any(
        o < 0 or (n and o + (n - 1) * s >= d)
        for o, n, s, d in zip(offsets, part.shape, strides, base.shape, strict=True)
    ):
        raise IntegerOpError(f"{op}: the slice lies outside dest or has a nonpositive stride")
    out = base.copy()
    out[tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, part.shape, strides, strict=True))] = part
    return _tensor(out, str(attrs.get("output_dtype", dest.dtype)))


def fill(shape: list[int], attrs: dict[str, Any], *, op: str = "FILL") -> Tensor:
    """Every element of the declared destination is ``value_code`` (an already-quantized code)."""
    unknown = sorted(set(attrs) - {"value_code", "output_dtype", "name"})
    if unknown:
        raise IntegerOpError(f"{op}: unknown attribute(s) {unknown}")
    code, dtype = attrs.get("value_code"), attrs.get("output_dtype")
    if type(code) is not int or not isinstance(dtype, str):
        raise IntegerOpError(f"{op}: value_code (an integer code) and output_dtype are required")
    qmin, qmax = _range({}, dtype, op=op)
    if not qmin <= code <= qmax:
        raise IntegerOpError(f"{op}: value_code {code} does not fit {dtype}")
    return _tensor(np.full(tuple(int(d) for d in shape), code, dtype=np.int64), dtype)


#: The opcodes :func:`execute` defines (``MOVEMENT`` forms go through :func:`move`).
OPCODES = ("AFFINE_MAP", "CONCAT", "INSERT_SLICE", "FILL", "WINDOW_SUM")


def declared_shape(cb: dict[str, Any], name: str) -> list[int] | None:
    """A tensor's declared shape in a command buffer, or None when it is not declared."""
    spec = (cb.get("tensors") or {}).get(name)
    return list(spec["shape"]) if isinstance(spec, dict) and isinstance(spec.get("shape"), list) else None


def source_keys(ops: dict[str, Any]) -> list[str]:
    """A CONCAT's ``src0, src1, ...`` operand keys in source order."""
    keys = [key for key in ops if key.startswith("src") and key[3:].isdigit()]
    return sorted(keys, key=lambda key: int(key[3:]))


def execute(op: str, ops: dict[str, Any], attrs: dict[str, Any], env: dict[str, Tensor], shape) -> Tensor:
    """Evaluate one AFFINE_MAP / CONCAT / INSERT_SLICE / FILL command against ``env``."""
    where = f"{op} {ops.get('dst')!r}"
    try:
        if op == "AFFINE_MAP":
            return affine_map(env[ops["x"]], env[ops["y"]] if "y" in ops else None, attrs, op=where)
        if op == "CONCAT":
            return concat([env[ops[key]] for key in source_keys(ops)], attrs, op=where)
        if op == "INSERT_SLICE":
            return insert_slice(env[ops["dest"]], env[ops["src"]], attrs, op=where)
    except KeyError as exc:
        raise IntegerOpError(f"{where}: operand {exc} is missing or undefined") from exc
    if op == "WINDOW_SUM":
        if "src" not in ops or ops["src"] not in env:
            raise IntegerOpError(f"{where}: operand src is missing or undefined")
        return window_sum(env[ops["src"]], attrs, shape, op=where)
    if op != "FILL":
        raise IntegerOpError(f"{op} is not an integer-op opcode")
    if shape is None:
        raise IntegerOpError(f"{where}: a fill needs its declared destination shape")
    return fill(shape, attrs, op=where)


# --- WINDOW_SUM -----------------------------------------------------------------------------------
_WINDOW_MODES = {
    "window": frozenset({"kernel", "stride", "padding", "count_include_pad"}),
    "adaptive": frozenset(),
    "reduce": frozenset({"dims", "average"}),
    "window_transpose": frozenset({"kernel", "stride", "padding", "count_include_pad"}),
    "adaptive_transpose": frozenset(),
    "fold": frozenset({"kernel", "stride", "padding", "dilation"}),
}
_WINDOW_OPTIONAL = frozenset(
    {"ceil_mode", "divisor_override", "layout", "quant_min", "quant_max", "name", "readout_scale"} | _EXACT_ATTRS
)
_WINDOW_COMMON = frozenset({"mode", "output_dtype"})


def _spatial_axes(rank: int, spatial: int, layout: str, *, op: str) -> list[int]:
    if layout == "nhwc":
        axes = list(range(rank - 1 - spatial, rank - 1))
    elif layout == "nchw":
        axes = list(range(rank - spatial, rank))
    else:
        raise IntegerOpError(f"{op}: layout must be nhwc (channels-last) or nchw, got {layout!r}")
    if not axes or axes[0] < 1:
        raise IntegerOpError(f"{op}: a rank-{rank} tensor has no {spatial} spatial dims under {layout}")
    return axes


def _pool_windows(size: int, kernel: int, stride: int, pad: int, ceil_mode: bool, include_pad: bool):
    """``[(start, end, count)]`` per output position of one pooled dim, in torch's avg-pool semantics."""
    span = size + 2 * pad - kernel
    if span < 0 or stride < 1 or kernel < 1 or pad < 0 or 2 * pad > kernel:
        raise IntegerOpError("pool geometry is invalid (kernel, stride or padding)")
    out = (-(-span // stride) if ceil_mode else span // stride) + 1
    if ceil_mode and (out - 1) * stride >= size + pad:
        out -= 1
    windows = []
    for index in range(out):
        start = index * stride - pad
        end = min(start + kernel, size + pad)
        padded = end - start
        lo, hi = max(start, 0), min(end, size)
        windows.append((lo, hi, padded if include_pad else hi - lo))
    return windows


def _adaptive_windows(size: int, out: int):
    if out < 1:
        raise IntegerOpError("adaptive output size must be positive")
    return [
        ((i * size) // out, -(-((i + 1) * size) // out), -(-((i + 1) * size) // out) - (i * size) // out)
        for i in range(out)
    ]


def window_sum(src: Tensor, attrs: dict[str, Any], dst_shape: list[int] | None, *, op: str = "WINDOW_SUM") -> Tensor:
    """Exact integer window sums with one readout per output; see the ABI's WINDOW_SUM entry."""
    mode = attrs.get("mode")
    if mode not in _WINDOW_MODES:
        raise IntegerOpError(f"{op}: mode must be one of {sorted(_WINDOW_MODES)}, got {mode!r}")
    allowed = _WINDOW_MODES[mode] | _WINDOW_COMMON | _WINDOW_OPTIONAL
    unknown = sorted(set(attrs) - allowed)
    if unknown:
        raise IntegerOpError(f"{op}: attribute(s) {unknown} do not belong to mode {mode!r}")
    missing = sorted((_WINDOW_MODES[mode] | _WINDOW_COMMON) - set(attrs))
    if missing:
        raise IntegerOpError(f"{op}: mode {mode!r} requires {missing}; no default is applied")
    dtype = str(attrs["output_dtype"])
    qmin, qmax = _range(attrs, dtype, op=op)
    scale = _scale(attrs, "readout_scale")
    if scale is None or scale <= 0:
        raise IntegerOpError(f"{op}: readout_scale must be a finite positive number")
    if dst_shape is None:
        raise IntegerOpError(f"{op}: the destination shape must be declared")
    a = _array(src)
    target = tuple(int(d) for d in dst_shape)
    layout = str(attrs.get("layout", "nhwc"))
    divisor = attrs.get("divisor_override")
    if divisor is not None and (type(divisor) is not int or divisor < 1):
        raise IntegerOpError(f"{op}: divisor_override must be a positive integer")

    if mode == "reduce":
        dims = _int_list(attrs, "dims", op=op)
        if type(attrs["average"]) is not bool:
            raise IntegerOpError(f"{op}: average must be a boolean")
        axes = sorted({d % a.ndim for d in dims})
        acc = a.sum(axis=tuple(axes), keepdims=True)
        count = int(np.prod([a.shape[d] for d in axes])) if attrs["average"] else 1
        acc = acc.reshape(target) if acc.size == int(np.prod(target)) else None
        if acc is None:
            raise IntegerOpError(f"{op}: the reduced shape does not match the declared {list(target)}")
        counts = np.full(acc.shape, divisor or count, dtype=np.int64)
        return _finish(acc, counts, scale, dtype, qmin, qmax, op=op, attrs=attrs)

    if mode == "fold":
        kernel = _int_list(attrs, "kernel", op=op)
        spatial = len(kernel)
        stride = _int_list(attrs, "stride", op=op, length=spatial)
        padding = _int_list(attrs, "padding", op=op, length=spatial)
        dilation = _int_list(attrs, "dilation", op=op, length=spatial)
        if len(target) != spatial + 2 or a.ndim != 3:
            raise IntegerOpError(f"{op}: fold reads [N, C*prod(kernel), L] and writes [N, C, *spatial]")
        n, c = target[0], target[1]
        out_sizes = target[2:]
        blocks = [
            (size + 2 * p - d * (k - 1) - 1) // s + 1
            for size, p, d, k, s in zip(out_sizes, padding, dilation, kernel, stride, strict=True)
        ]
        if a.shape != (n, c * int(np.prod(kernel)), int(np.prod(blocks))):
            raise IntegerOpError(f"{op}: fold input {list(a.shape)} does not match kernel/blocks of {list(target)}")
        acc = np.zeros(target, dtype=np.int64)
        columns = a.reshape((n, c, *kernel, *blocks))
        for tap in np.ndindex(*kernel):
            for block in np.ndindex(*blocks):
                position = [
                    b * s - p + t * d for b, s, p, t, d in zip(block, stride, padding, tap, dilation, strict=True)
                ]
                if all(0 <= q < size for q, size in zip(position, out_sizes, strict=True)):
                    acc[(slice(None), slice(None), *position)] += columns[(slice(None), slice(None), *tap, *block)]
        return _finish(acc, np.ones(acc.shape, dtype=np.int64), scale, dtype, qmin, qmax, op=op, attrs=attrs)

    transpose = mode.endswith("_transpose")
    source_shape = target if transpose else a.shape
    window_shape = a.shape if transpose else target
    if mode.startswith("adaptive"):
        spatial = a.ndim - 2
        axes = _spatial_axes(a.ndim, spatial, layout, op=op)
        per_dim = [_adaptive_windows(source_shape[ax], window_shape[ax]) for ax in axes]
    else:
        kernel = _int_list(attrs, "kernel", op=op)
        spatial = len(kernel)
        axes = _spatial_axes(a.ndim, spatial, layout, op=op)
        stride = _int_list(attrs, "stride", op=op, length=spatial)
        padding = _int_list(attrs, "padding", op=op, length=spatial)
        if type(attrs["count_include_pad"]) is not bool:
            raise IntegerOpError(f"{op}: count_include_pad must be a boolean")
        per_dim = [
            _pool_windows(source_shape[ax], k, s, p, bool(attrs.get("ceil_mode", False)), attrs["count_include_pad"])
            for ax, k, s, p in zip(axes, kernel, stride, padding, strict=True)
        ]
    if tuple(len(windows) for windows in per_dim) != tuple(window_shape[ax] for ax in axes):
        raise IntegerOpError(f"{op}: the pooled extent differs from the declared one")
    if any(d != t for i, (d, t) in enumerate(zip(source_shape, window_shape, strict=True)) if i not in axes):
        raise IntegerOpError(f"{op}: batch and channel extents must be preserved")

    def region(window_index):
        index = [slice(None)] * a.ndim
        for ax, windows, w in zip(axes, per_dim, window_index, strict=True):
            index[ax] = slice(windows[w][0], windows[w][1])
        return tuple(index)

    def window_count(window_index):
        if divisor is not None:
            return divisor
        return int(np.prod([windows[w][2] for windows, w in zip(per_dim, window_index, strict=True)]))

    grid = [range(len(windows)) for windows in per_dim]
    if not transpose:
        acc = np.zeros(target, dtype=np.int64)
        counts = np.ones(target, dtype=np.int64)
        for window_index in np.ndindex(*[len(g) for g in grid]):
            out_index = [slice(None)] * a.ndim
            for ax, w in zip(axes, window_index, strict=True):
                out_index[ax] = slice(w, w + 1)
            total = a[region(window_index)].sum(axis=tuple(axes), keepdims=True)
            acc[tuple(out_index)] = total
            counts[tuple(out_index)] = window_count(window_index)
        return _finish(acc, counts, scale, dtype, qmin, qmax, op=op, attrs=attrs)
    # Transposed (a pool backward): each window's value spreads over the positions it covers, divided by
    # that window's count. Stated with integer weights L/count_w (L the least common multiple of every
    # count) into one int32 accumulator and the single readout readout_scale/L.
    windows = list(np.ndindex(*[len(g) for g in grid]))
    lcm = integer_lcm(*(window_count(w) for w in windows)) if windows else 1
    acc = np.zeros(target, dtype=np.int64)
    for window_index in windows:
        source_index = [slice(None)] * a.ndim
        for ax, w in zip(axes, window_index, strict=True):
            source_index[ax] = slice(w, w + 1)
        acc[region(window_index)] += a[tuple(source_index)] * (lcm // window_count(window_index))
    return _finish(acc, np.full(target, lcm, dtype=np.int64), scale, dtype, qmin, qmax, op=op, attrs=attrs)


def _finish(acc, counts, scale, dtype, qmin, qmax, *, op, attrs) -> Tensor:
    if np.any(acc < -(2**31)) or np.any(acc > 2**31 - 1):
        raise IntegerOpError(f"{op}: int32 accumulation overflow")
    if dtype == "i32":
        if scale != 1 or np.any(counts != 1) or attrs.get("zero_point", 0) != 0:
            raise IntegerOpError(f"{op}: an i32 result has no readout; it needs readout_scale 1.0 and no divisor")
        return _tensor(acc, dtype)
    ratios = [rational(scale) / int(count) for count in counts.flat]
    try:
        result = requantize_int32(
            acc.reshape(-1),
            scale_numerator=[r.numerator for r in ratios],
            scale_denominator=[r.denominator for r in ratios],
            quant_min=qmin,
            quant_max=qmax,
            zero_point=attrs.get("zero_point", 0),
            rounding=attrs.get("rounding", "half_to_even"),
        )
    except ValueError as exc:
        raise IntegerOpError(f"{op}: {exc}") from exc
    return _tensor(result.reshape(acc.shape), dtype)
