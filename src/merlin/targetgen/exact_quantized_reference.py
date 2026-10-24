"""Independent exact-rational oracle for source quantized FX graphs.

Unknown operations refuse qualification. Eager floating divergence is diagnostic.
"""

from __future__ import annotations

import math
import operator
from fractions import Fraction
from itertools import product
from types import SimpleNamespace

import numpy as np

from merlin.targetgen.exact_arithmetic import (
    ExactTensor,
    exact_array as _finite_exact_array,
    materialize as _finite_materialize,
)
from merlin.targetgen.exact_arithmetic import (
    dequantize as dequantize,
)
from merlin.targetgen.exact_arithmetic import (
    quantize as quantize,
)
from merlin.targetgen.exact_arithmetic import (
    rational as _finite_rational,
)
from merlin.targetgen.exact_arithmetic import (
    round_binary as round_binary,
)


from contextvars import ContextVar
from merlin.targetgen.exact_boundary_special import BoundarySpecial
_boundary_state=ContextVar('exact_reference_boundary_state',default=None)

def rational(value):
    state=_boundary_state.get()
    if isinstance(value,BoundarySpecial):return value
    if state is not None and not isinstance(value,Fraction):
        number=float(value)
        if not math.isfinite(number):
            state['special_seen']=True
            import struct
            word=(struct.unpack('<I',struct.pack('<f',number))[0] | (1<<22)) if math.isnan(number) else None
            return BoundarySpecial(0 if math.isnan(number) else 1 if number>0 else -1,word)
    return _finite_rational(value)

def exact_array(value):
    if _boundary_state.get() is None:return _finite_exact_array(value)
    if isinstance(value,ExactTensor):return value.values
    if hasattr(value,'detach'):value=value.detach().cpu().tolist()
    return np.vectorize(rational,otypes=[object])(np.asarray(value,dtype=object))


def materialize(value):
    if isinstance(value,tuple):return tuple(materialize(x) for x in value)
    if isinstance(value,list):return [materialize(x) for x in value]
    if isinstance(value,dict):return {k:materialize(x) for k,x in value.items()}
    state=_boundary_state.get()
    explicit_overflow=state is not None and state.get('f32_overflow')=='ieee_infinity'
    special=isinstance(value,ExactTensor) and any(isinstance(x,BoundarySpecial) for x in value.values.flat)
    if isinstance(value,ExactTensor) and (special or explicit_overflow and str(value.dtype)=='torch.float32'):
        import struct
        import torch
        state=_boundary_state.get()
        if special and (not state or state.get('floating_specials')!='single_residual_quiet_word_f32' or value.dtype!=torch.float32):
            raise ValueError('special rational arithmetic requires a final byte quantizer')
        words=[]
        for x in value.values.flat:
            if isinstance(x,BoundarySpecial):
                if not x.sign and x.word is None:raise ValueError('competing or invalid NaNs lack a deterministic input word')
                words.append(x.word if not x.sign else 0x7f800000 | (0x80000000 if x.sign<0 else 0))
            else:words.append(struct.unpack('<I',struct.pack('<f',round_binary(x,32,overflow='infinity' if explicit_overflow else 'refuse')))[0])
        return torch.tensor(words,dtype=torch.uint32).view(torch.float32).reshape(value.values.shape)
    return _finite_materialize(value)


def _clamp_array(values,bound,*,lower):
    return np.vectorize(lambda value: value if isinstance(value,BoundarySpecial) and not value.sign else max(value,bound) if lower else min(value,bound),otypes=[object])(values)


def mac_observation(lhs, rhs, accumulator, scales):
    """Integer diagnostics only; rounded scale products never feed the oracle."""
    maximum = max((abs(int(x)) for x in accumulator.flat), default=0)
    factors = None
    if all(scale is not None for scale in scales):
        factors = exact_array(scales[0]) * exact_array(scales[1])

        def diagnostic_float(value):
            try:
                result = float(value)
            except OverflowError:
                return None
            return result if math.isfinite(result) else None

        factors = np.vectorize(diagnostic_float, otypes=[object])(factors).tolist()
    return dict(
        lhs=scales[0],
        rhs=scales[1],
        product=factors,
        operands=[dict(quantized_max_abs=max((abs(int(x)) for x in value.flat), default=0)) for value in (lhs, rhs)],
        accumulator=accumulator.tolist(),
        accumulator_max_abs=maximum,
    )


def _convolution(x, w, bias, stride, padding, dilation, groups):
    dims = x.ndim - 2

    def expand(value):
        return [value] * dims if isinstance(value, int) else list(value)

    stride, padding, dilation = map(expand, (stride, padding, dilation))
    if isinstance(padding, str):
        raise ValueError("exact convolution requires explicit numeric padding")
    kernel = w.shape[2:]
    shape = tuple(
        (x.shape[2 + i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1 for i in range(dims)
    )
    if any(size < 0 for size in shape):
        raise ValueError("invalid convolution geometry")
    output = np.full((x.shape[0], w.shape[0], *shape), Fraction(0), dtype=object)
    per_group = w.shape[0] // groups
    for n, co, *point in np.ndindex(output.shape):
        total = Fraction(0) if bias is None else bias[co]
        for ci in range(w.shape[1]):
            for tap in np.ndindex(kernel):
                source = tuple(point[d] * stride[d] - padding[d] + tap[d] * dilation[d] for d in range(dims))
                if all(0 <= source[d] < x.shape[2 + d] for d in range(dims)):
                    total += x[(n, co // per_group * w.shape[1] + ci, *source)] * w[(co, ci, *tap)]
        output[(n, co, *point)] = total
    return output


def _pool(x, name, args, kwargs, *, gradient=None):
    dims = int(name.split("pool", 1)[1][0])
    prefix = x.ndim - dims
    if "adaptive" in name:
        output_size = args[1]
        shape = (output_size,) * dims if isinstance(output_size, int) else tuple(output_size)

        def windows(point):
            return [
                range(
                    point[d] * x.shape[prefix + d] // shape[d],
                    ((point[d] + 1) * x.shape[prefix + d] + shape[d] - 1) // shape[d],
                )
                for d in range(dims)
            ]

        divisor_override = None
    else:

        def seq(value, default):
            if value is None or value == []:
                value = default
            return [value] * dims if isinstance(value, int) else list(value)

        kernel = seq(args[1], 1)
        stride = seq(args[2] if len(args) > 2 else kwargs.get("stride"), kernel)
        pad = seq(args[3] if len(args) > 3 else kwargs.get("padding", 0), 0)
        ceil = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        include = args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)
        divisor_override = args[6] if len(args) > 6 else kwargs.get("divisor_override")
        shape = []
        for size, k, s, p in zip(x.shape[prefix:], kernel, stride, pad, strict=True):
            out = (size + 2 * p - k + (s - 1 if ceil else 0)) // s + 1
            if ceil and (out - 1) * s >= size + p:
                out -= 1
            shape.append(out)

        def windows(point):
            return [
                range(
                    point[d] * stride[d] - pad[d],
                    min(point[d] * stride[d] - pad[d] + kernel[d], x.shape[prefix + d] + pad[d]),
                )
                for d in range(dims)
            ]

    out = np.empty((*x.shape[:prefix], *shape), dtype=object)
    adjoint = np.full(x.shape, Fraction(0), dtype=object) if gradient is not None else None
    if gradient is not None and gradient.shape != out.shape:
        raise ValueError("pool gradient shape does not match forward geometry")
    for index in np.ndindex(out.shape):
        ranges = windows(index[prefix:])
        valid = [p for p in product(*ranges) if all(0 <= p[d] < x.shape[prefix + d] for d in range(dims))]
        count = divisor_override or (
            math.prod(len(r) for r in ranges) if "adaptive" not in name and include else len(valid)
        )
        if not count:
            raise ValueError("exact average pool has an empty divisor")
        if gradient is None:
            out[index] = sum((x[(*index[:prefix], *p)] for p in valid), Fraction(0)) / count
        else:
            for point in valid:
                adjoint[(*index[:prefix], *point)] += gradient[index] / count
    return out if gradient is None else adjoint


def _pool_backward(gradient, shape, name, args, kwargs):
    source = np.empty(shape, dtype=object)
    dims = int(name.split("pool", 1)[1][0])
    forward_args = (source, gradient.shape[-dims:]) if "adaptive" in name else (source, *args[2:])
    return _pool(source, name, forward_args, kwargs, gradient=gradient)


def _fold(columns, output_size, kernel_size, dilation, padding, stride):
    """Adjoint of spatial patch extraction, accumulating rational entries unchanged."""
    unbatched = columns.ndim == 2
    if unbatched:
        columns = columns[None]
    taps = math.prod(kernel_size)
    channels, remainder = divmod(columns.shape[1], taps)
    blocks = tuple(
        (s + 2 * p - d * (k - 1) - 1) // t + 1
        for s, p, d, k, t in zip(output_size, padding, dilation, kernel_size, stride, strict=True)
    )
    if remainder or math.prod(blocks) != columns.shape[2] or any(b <= 0 for b in blocks):
        raise ValueError("fold columns do not match spatial geometry")
    out = np.full((columns.shape[0], channels, *output_size), Fraction(0), dtype=object)
    for batch, channel in np.ndindex(out.shape[:2]):
        for tap_index, tap in enumerate(np.ndindex(tuple(kernel_size))):
            for block_index, block in enumerate(np.ndindex(blocks)):
                point = tuple(
                    b * s - p + k * d for b, s, p, k, d in zip(block, stride, padding, tap, dilation, strict=True)
                )
                if all(0 <= i < s for i, s in zip(point, output_size, strict=True)):
                    out[(batch, channel, *point)] += columns[batch, channel * taps + tap_index, block_index]
    return out[0] if unbatched else out


def convolution_backward(operands, args, node_name, sites):
    """Exact spatial adjoints, plus independent code-domain GEMM observations."""
    stride, padding, dilation = map(list, (args[4], args[5], args[6]))
    transposed, output_padding, groups, mask = args[7:11]
    def geometry(value):
        return value.values.shape if isinstance(value, ExactTensor) else value.shape
    grad = exact_array(operands[0]) if any(mask) else np.empty(geometry(operands[0]), dtype=object)
    source = exact_array(operands[1]) if mask[1] else np.empty(geometry(operands[1]), dtype=object)
    weight = exact_array(operands[2]) if mask[0] else np.empty(geometry(operands[2]), dtype=object)
    if transposed or any(output_padding) or source.ndim != 4:
        raise ValueError("exact reference requires non-transposed 2-D convolution_backward")
    dx, dw = np.full(source.shape, Fraction(0), dtype=object), np.full(weight.shape, Fraction(0), dtype=object)
    db = np.sum(grad, axis=(0, 2, 3)) if mask[2] else None
    n, cin, h, width = source.shape
    cout, cg, kh, kw = weight.shape
    fg, oh, ow = cout // groups, grad.shape[2], grad.shape[3]
    for batch, co, oy, ox in np.ndindex(grad.shape):
        for ci, ky, kx in np.ndindex((cg, kh, kw)):
            iy, ix = oy * stride[0] - padding[0] + ky * dilation[0], ox * stride[1] - padding[1] + kx * dilation[1]
            if 0 <= iy < h and 0 <= ix < width:
                channel = co // fg * cg + ci
                if mask[0]:
                    dx[batch, channel, iy, ix] += grad[batch, co, oy, ox] * weight[co, ci, ky, kx]
                if mask[1]:
                    dw[co, ci, ky, kx] += grad[batch, co, oy, ox] * source[batch, channel, iy, ix]
    if all(isinstance(x, ExactTensor) and x.codes is not None for x in operands):
        g, x, w = [value.codes for value in operands]
        positions = n * oh * ow
        rows = [g[:, i * fg : (i + 1) * fg].transpose(1, 0, 2, 3).reshape(fg, positions) for i in range(groups)]
        for which in (0, 1):
            if not mask[which]:
                continue
            for i in range(groups):
                if which == 0:
                    left, right = w[i * fg : (i + 1) * fg].reshape(fg, cg * kh * kw).T, rows[i]
                else:
                    padded = np.pad(
                        x[:, i * cg : (i + 1) * cg],
                        ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                    )
                    patches = [
                        padded[
                            :,
                            :,
                            ky * dilation[0] : ky * dilation[0] + oh * stride[0] : stride[0],
                            kx * dilation[1] : kx * dilation[1] + ow * stride[1] : stride[1],
                        ]
                        .transpose(0, 2, 3, 1)
                        .reshape(positions, cg)
                        for ky, kx in np.ndindex((kh, kw))
                    ]
                    left, right = rows[i], np.concatenate(patches, axis=1)
                scale_order = (2, 0) if which == 0 else (0, 1)
                scales = [operands[j].scale for j in scale_order]
                scales = [value.detach().cpu().tolist() if hasattr(value, "detach") else value for value in scales]
                sites.append(
                    dict(
                        node=f"{node_name}.{'grad_input' if which == 0 else 'grad_weight'}{i}",
                        kind="conv2d",
                        **mac_observation(left, right, np.matmul(left, right), scales),
                        readout=None,
                        readout_order="exact rational; one boundary RNE",
                        arithmetic="exact_rational",
                    )
                )
    dtype = operands[0].dtype
    return tuple(
        ExactTensor(value, dtype) if wanted else None for value, wanted in zip((dx, dw, db), mask, strict=True)
    )


def evaluate_operation(target, args, kwargs):
    """Closed rational operations. Unknown semantics cannot acquire an exactness claim."""
    import torch

    name = str(target)
    flat = torch.utils._pytree.tree_leaves((args, kwargs))
    exact = next((x for x in flat if isinstance(x, ExactTensor)), None)
    source = exact if exact is not None else next((x for x in flat if isinstance(x, torch.Tensor)), None)
    if "out" in kwargs or name.endswith(".out"):
        raise ValueError("exact reference refuses output-buffer mutation")
    dtype = kwargs.get("dtype") or getattr(source, "dtype", torch.get_default_dtype())
    if dtype not in {None, torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        raise ValueError("exact reference refuses an implicit integer cast")
    op = name.split(".")[1] if name.startswith("aten.") else name
    if op in {"fill", "full", "full_like", "zeros", "zeros_like", "ones", "ones_like"}:
        value = 0 if op.startswith("zeros") else 1 if op.startswith("ones") else args[1]
        shape = args[0] if op in {"full", "zeros", "ones"} else (args[0].values.shape if isinstance(args[0], ExactTensor) else args[0].shape)
        return ExactTensor(np.full(shape, rational(value), dtype=object), dtype)
    if op.startswith(("avg_pool", "adaptive_avg_pool", "_adaptive_avg_pool")) and op.endswith("backward"):
        # The primal tensor supplies geometry only. Its storage values have no
        # dependence path to the adjoint, including nonfinite unused payloads.
        shape = args[1].values.shape if isinstance(args[1], ExactTensor) else args[1].shape
        return ExactTensor(_pool_backward(exact_array(args[0]), shape, name, args, kwargs), dtype)
    a = [exact_array(x) if isinstance(x, (ExactTensor, torch.Tensor)) else x for x in args]
    if op in {"add", "sub", "mul", "div", "true_divide"}:
        alpha = rational(args[2] if len(args) > 2 and op in {"add", "sub"} else kwargs.get("alpha", 1))
        left, right = exact_array(a[0]), exact_array(a[1])
        if op in {"add", "sub"}:
            right = right * alpha
        if kwargs.get("rounding_mode") is not None:
            raise ValueError("exact reference refuses rounded division")
        out = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.truediv,
            "true_divide": operator.truediv,
        }[op](left, right)
    elif op in {"neg", "relu", "abs"}:
        out = -a[0] if op == "neg" else _clamp_array(a[0], 0, lower=True) if op == "relu" else abs(a[0])
    elif op == "elu":
        alpha = args[1] if len(args) > 1 else kwargs.get("alpha", 1)
        if alpha != 0:
            raise ValueError("exact rational reference refuses a nonzero exponential branch")
        scale = args[2] if len(args) > 2 else kwargs.get("scale", 1)
        out = _clamp_array(a[0], 0, lower=True) * rational(scale)
    elif op in {"clamp", "clamp_min", "clamp_max", "hardtanh"}:
        low = args[1] if len(args) > 1 else kwargs.get("min_val", -1) if op == "hardtanh" else kwargs.get("min")
        high = args[2] if len(args) > 2 else kwargs.get("max_val", 1) if op == "hardtanh" else kwargs.get("max")
        if op == "clamp_max":
            low, high = None, low
        out = a[0]
        if low is not None:
            out = _clamp_array(out, rational(low), lower=True)
        if high is not None:
            out = _clamp_array(out, rational(high), lower=False)
    elif op in {"mm", "bmm", "matmul", "linear", "addmm", "baddbmm"}:
        if op in {"addmm", "baddbmm"}:
            out = (rational(kwargs.get("beta", 1)) * a[0] if kwargs.get("beta",1) else Fraction(0)) + rational(kwargs.get("alpha", 1)) * np.matmul(a[1], a[2])
        elif op == "linear":
            out = np.matmul(a[0], a[1].swapaxes(-1, -2))
            if len(a) > 2 and a[2] is not None:
                out += a[2]
        else:
            out = np.matmul(a[0], a[1])
    elif op in {"fmod", "remainder"}:
        def remainder_value(x, y):
            if not y:
                raise ValueError("exact remainder refuses a literal zero divisor")
            quotient = x / y
            whole = math.trunc(quotient) if op == "fmod" else math.floor(quotient)
            return x - whole * y
        other = a[1] if len(a) > 1 else kwargs["other"]
        divisor = exact_array(other) if isinstance(other, (ExactTensor, torch.Tensor)) else rational(other)
        out = np.vectorize(remainder_value, otypes=[object])(a[0], divisor)
    elif op == "cumsum":
        axis = args[1] if len(args) > 1 else kwargs["dim"]
        out = np.cumsum(a[0], axis=axis, dtype=object)
    elif op == "native_dropout":
        probability = args[1] if len(args) > 1 else kwargs["p"]
        training = args[2] if len(args) > 2 else kwargs.get("train", True)
        if training and probability not in (0, 1):
            raise ValueError("exact reference refuses a stochastic mask")
        kept = not training or probability == 0
        values = a[0] if kept else np.full(a[0].shape, Fraction(0), dtype=object)
        return ExactTensor(values, dtype), ExactTensor(np.full(a[0].shape, kept, dtype=object), torch.bool)
    elif op in {"sum", "mean", "amax", "amin", "prod"}:
        axis = args[1] if len(args) > 1 else kwargs.get("dim")
        axis = None if axis is None or axis == [] else tuple(axis) if isinstance(axis, (list, tuple)) else axis
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        out = {"sum": np.sum, "mean": np.mean, "amax": np.max, "amin": np.min, "prod": np.prod}[op](a[0], axis=axis, keepdims=keep)
    elif op in {"conv1d", "conv2d", "conv3d", "convolution"}:
        if op == "convolution" and (args[6] or any(args[7])):
            raise ValueError("exact reference refuses transposed convolution")
        stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
        pad = args[4] if len(args) > 4 else kwargs.get("padding", 0)
        dilation = args[5] if len(args) > 5 else kwargs.get("dilation", 1)
        groups = args[8] if op == "convolution" else args[6] if len(args) > 6 else kwargs.get("groups", 1)
        out = _convolution(a[0], a[1], a[2] if len(a) > 2 else None, stride, pad, dilation, groups)
    elif op.startswith(("avg_pool", "adaptive_avg_pool", "_adaptive_avg_pool")) and "backward" not in op:
        out = _pool(a[0], name, args, kwargs)
    elif op.startswith(("avg_pool", "adaptive_avg_pool", "_adaptive_avg_pool")) and op.endswith("backward"):
        out = _pool_backward(a[0], a[1].shape, name, args, kwargs)
    elif op == "col2im":
        out = _fold(a[0], *args[1:])
    elif op in {"view", "reshape", "_unsafe_view", "flatten"}:
        if op == "flatten":
            start = (args[1] if len(args) > 1 else 0) % a[0].ndim
            end = (args[2] if len(args) > 2 else -1) % a[0].ndim
            if end < start:
                raise ValueError("invalid flatten dimensions")
            shape = (*a[0].shape[:start], math.prod(a[0].shape[start : end + 1]), *a[0].shape[end + 1 :])
        else:
            shape = args[1]
        out = a[0].reshape(shape)
    elif op in {"clone", "detach", "contiguous", "alias"}:
        out = a[0].copy()
    elif op in {"_to_copy", "to"}:
        dtype = kwargs.get("dtype", dtype)
        if dtype not in {torch.float16, torch.float32, torch.float64}:
            raise ValueError("exact reference refuses an intermediate integer cast")
        out = a[0].copy()
    elif op in {"permute", "transpose", "t"}:
        out = (
            a[0].transpose(args[1])
            if op == "permute"
            else a[0].swapaxes(args[1], args[2])
            if op == "transpose"
            else a[0].T
        )
    elif op in {"squeeze", "unsqueeze"}:
        out = (
            np.expand_dims(a[0], args[1])
            if op == "unsqueeze"
            else np.squeeze(a[0], axis=args[1] if len(args) > 1 else None)
        )
    elif op in {"expand", "broadcast_to"}:
        shape = tuple(
            old if new == -1 else new
            for old, new in zip((1,) * (len(args[1]) - a[0].ndim) + a[0].shape, args[1], strict=True)
        )
        out = np.broadcast_to(a[0], shape)
    elif op == "repeat":
        out = np.tile(a[0], args[1])
    elif op in {"slice", "select"}:
        indices = [slice(None)] * a[0].ndim
        indices[args[1]] = slice(args[2], args[3], args[4] if len(args) > 4 else 1) if op == "slice" else args[2]
        out = a[0][tuple(indices)]
    elif op in {"cat", "stack"}:
        out = (np.concatenate if op == "cat" else np.stack)(
            [exact_array(x) for x in args[0]], axis=args[1] if len(args) > 1 else 0
        )
    elif op == "constant_pad_nd" or op.startswith(("reflection_pad", "replication_pad")):
        widths = [(0, 0)] * a[0].ndim
        for i in range(len(args[1]) // 2):
            widths[-1 - i] = tuple(args[1][2 * i : 2 * i + 2])
        if op == "constant_pad_nd":
            cropped = a[0][
                tuple(slice(max(-lo, 0), size - max(-hi, 0)) for size, (lo, hi) in zip(a[0].shape, widths, strict=True))
            ]
            widths = [(max(lo, 0), max(hi, 0)) for lo, hi in widths]
            out = np.pad(cropped, widths, constant_values=rational(args[2] if len(args) > 2 else 0))
        else:
            # Reflect about the ORIGINAL endpoints: cropping before reflection
            # would change the mirror when one side crops and the other pads.
            out = a[0]
            for axis, (low, high) in enumerate(widths):
                size = a[0].shape[axis]
                if size <= 0 or size + low + high <= 0:
                    raise ValueError("exact padding requires nonempty dimensions")
                positions = np.arange(size + low + high) - low
                if op.startswith("reflection"):
                    if low >= size or high >= size:
                        raise ValueError("reflection padding must be smaller than the input dimension")
                    positions = np.where(positions < 0, -positions, positions)
                    positions = np.where(positions >= size, 2 * size - 2 - positions, positions)
                else:
                    positions = np.clip(positions, 0, size - 1)
                out = np.take(out, positions, axis=axis)
    elif op in {"slice_scatter", "select_scatter"}:
        out = a[0].copy()
        indices = [slice(None)] * out.ndim
        indices[args[2]] = (
            slice(args[3], args[4], args[5] if len(args) > 5 else 1) if op == "slice_scatter" else args[3]
        )
        out[tuple(indices)] = a[1]
    elif op in {"fill", "full", "full_like", "zeros", "zeros_like", "ones", "ones_like"}:
        value = 0 if op.startswith("zeros") else 1 if op.startswith("ones") else args[1]
        shape = args[0] if op in {"full", "zeros", "ones"} else a[0].shape
        out = np.full(shape, rational(value), dtype=object)
    else:
        raise ValueError(f"exact reference refuses rational operation {name}")
    return ExactTensor(np.asarray(out, dtype=object), dtype)


def run(model, inputs):
    """Evaluate source Q/DQ graph exactly; eager differences are informational only."""
    import copy
    import logging

    import torch

    sites = []
    counts = {"linear": 0, "conv2d": 0, "matmul": 0}
    boundary_policy = getattr(model, "_recipe_quantization_stats", {}).get("framework_capture_policy", {}).get("quantization_nonfinite")

    class Reference(torch.fx.Interpreter):
        def run_node(self, node):
            if node.op != "call_function":
                return super().run_node(node)
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            name = str(node.target)
            if name.startswith("quantized_decomposed."):
                channel = "per_channel" in name
                if "quantize_per_" not in name:
                    raise ValueError(f"exact reference refuses quantizer {name}")
                value, scale, zp = args[:3]
                axis, low, high, dtype = args[3:7] if channel else (None, *args[3:6])
                if dtype not in {torch.int8, torch.uint8}:
                    raise ValueError("exact quantization requires int8 or uint8 storage")
                storage = torch.iinfo(dtype)
                if not storage.min <= low <= high <= storage.max:
                    raise ValueError("quantization range exceeds storage")
                if ".dequantize_" in name:
                    position = 7 if channel else 6
                    out_dtype = kwargs.get("out_dtype", args[position] if len(args) > position else torch.float32)
                    return dequantize(value, scale, zp, low, high, axis=axis, dtype=out_dtype)
                codes = quantize(value, scale, zp, low, high, axis=axis, nonfinite=boundary_policy)
                return torch.tensor(codes.tolist(), dtype=dtype).reshape(codes.shape)
            leaves = torch.utils._pytree.tree_leaves((args, kwargs))
            if name == "aten.isnan.default" and isinstance(args[0], ExactTensor):
                values = np.vectorize(lambda x: not isinstance(x, Fraction) and math.isnan(float(x)), otypes=[bool])(args[0].values)
                return torch.tensor(values, dtype=torch.bool)
            if name == "aten.where.self" and any(isinstance(x, ExactTensor) for x in leaves):
                def storage(value):
                    return value.values if isinstance(value, ExactTensor) else value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value

                selected = np.where(storage(args[0]), storage(args[1]), storage(args[2]))
                # A selector observes only selected elements. An unselected NaN
                # does not enter rational arithmetic or invalidate that branch.
                dtype = next(x.dtype for x in args[1:] if isinstance(x, (ExactTensor, torch.Tensor)))
                if boundary_policy == "byte_storage_zero_nan_saturating_infinity":
                    values = np.vectorize(lambda x: x if not isinstance(x, Fraction) and not math.isfinite(float(x)) else rational(x), otypes=[object])(selected)
                    return ExactTensor(values, dtype)
                return ExactTensor(exact_array(selected), dtype)
            constant = name.split(".")[1] if name.startswith("aten.") else None
            if name == "aten._assert_tensor_metadata.default":
                # A metadata assertion has no value semantics: check the declared dtype
                # against the exact tensor's carried dtype and produce nothing.
                value, declared = args[0], kwargs.get("dtype", args[3] if len(args) > 3 else None)
                if isinstance(value, ExactTensor):
                    if declared is not None and declared != value.dtype:
                        raise ValueError("exact reference observed a tensor metadata mismatch")
                    return None
                return super().run_node(node)
            # A non-floating constant (a boolean mask, an integer index) carries no
            # rational value to track; it is evaluated as the framework defines it.
            requested = kwargs.get("dtype")
            if (
                not any(isinstance(x, ExactTensor) for x in leaves)
                and constant in {"full", "zeros", "ones"}
                and requested is not None
                and not requested.is_floating_point
            ):
                return super().run_node(node)
            if not any(isinstance(x, ExactTensor) for x in leaves) and constant not in {
                "fill",
                "full",
                "full_like",
                "zeros",
                "zeros_like",
                "ones",
                "ones_like",
            }:
                return super().run_node(node)
            if node.target == operator.getitem or getattr(node.target, "_m2m_tuple_selection", False):
                value = args[0]
                return (
                    ExactTensor(value.values[args[1]], value.dtype)
                    if isinstance(value, ExactTensor)
                    else value[args[1]]
                )
            if name == "aten.convolution_backward.default":
                counts["conv2d"] += 1
                return convolution_backward(args[:3], args, node.name, sites)
            result = evaluate_operation(node.target, args, kwargs)
            op = name.split(".")[1] if name.startswith("aten.") else name
            if op in {"mm", "bmm", "matmul", "linear", "addmm", "baddbmm", "conv1d", "conv2d", "conv3d", "convolution"}:
                kind = "linear" if op == "linear" else "conv2d" if "conv" in op else "matmul"
                counts[kind] += 1
                operands = args[1:3] if op in {"addmm", "baddbmm"} else args[:2]
                scales = [x.scale if isinstance(x, ExactTensor) else None for x in operands]
                scales = [scale.detach().cpu().tolist() if hasattr(scale, "detach") else scale for scale in scales]
                acc = None
                if all(isinstance(x, ExactTensor) and x.codes is not None for x in operands) and op in {
                    "mm",
                    "bmm",
                    "matmul",
                    "linear",
                    "addmm",
                    "baddbmm",
                }:
                    lhs, rhs = [x.codes for x in operands]
                    if op == "linear":
                        rhs = rhs.swapaxes(-1, -2)
                    acc = np.matmul(lhs, rhs)
                elif all(isinstance(x, ExactTensor) and x.codes is not None for x in operands) and "conv" in op:
                    lhs, rhs = [x.codes for x in operands]
                    stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
                    padding = args[4] if len(args) > 4 else kwargs.get("padding", 0)
                    dilation = args[5] if len(args) > 5 else kwargs.get("dilation", 1)
                    groups = args[8] if op == "convolution" else args[6] if len(args) > 6 else kwargs.get("groups", 1)
                    acc = _convolution(lhs, rhs, None, stride, padding, dilation, groups)
                    acc = np.moveaxis(acc, 1, -1).reshape(-1, rhs.shape[0])
                    acc = np.vectorize(int, otypes=[object])(acc)
                sites.append(
                    dict(
                        node=node.name,
                        kind=kind,
                        **(
                            mac_observation(operands[0].codes, operands[1].codes, acc, scales)
                            if acc is not None
                            else dict(lhs=scales[0], rhs=scales[1], accumulator=None)
                        ),
                        readout=None,
                        readout_order="exact rational; one boundary RNE",
                        arithmetic="exact_rational",
                    )
                )
            return result

    saved = copy.deepcopy(tuple(inputs))
    reference = Reference(model, garbage_collect_values=False)
    state={'special_seen':False,'floating_specials':getattr(model, '_recipe_quantization_stats', {}).get('framework_capture_policy', {}).get('readout_specials'),'f32_overflow':getattr(model, '_recipe_quantization_stats', {}).get('framework_capture_policy', {}).get('f32_boundary_overflow')} if boundary_policy=='byte_storage_zero_nan_saturating_infinity' else None
    token=_boundary_state.set(state)
    try:
        exact = reference.run(*inputs)
        if state is not None and state['special_seen'] and state['floating_specials']!='single_residual_quiet_word_f32':
            leaves=torch.utils._pytree.tree_leaves(exact)
            if any(isinstance(x,ExactTensor) and x.codes is None for x in leaves):
                raise ValueError('special rational arithmetic requires a final byte quantizer')
        output = materialize(exact)
    finally:
        _boundary_state.reset(token)
    divergence = {"status": "not_checked", "informational": True}
    try:
        eager = model(*saved)
        eager_leaves, eager_tree = torch.utils._pytree.tree_flatten(eager)
        actual_leaves, actual_tree = torch.utils._pytree.tree_flatten(output)
        different = 0
        if eager_tree != actual_tree:
            raise ValueError("eager and exact output structures differ")
        for left, right in zip(eager_leaves, actual_leaves, strict=True):
            different += int((left != right).sum()) if isinstance(left, torch.Tensor) else int(left != right)
        divergence = {
            "status": "diverged" if different else "equal",
            "different_elements": different,
            "informational": True,
        }
        if different:
            logging.getLogger(__name__).info("eager-float divergence: %s", divergence)
    except Exception as exc:
        divergence["reason"] = str(exc)
    return SimpleNamespace(
        output=output,
        post_inputs=inputs,
        sites=sites,
        contraction_count=sum(counts.values()),
        linear_count=counts["linear"],
        conv2d_count=counts["conv2d"],
        matmul_count=counts["matmul"],
        arithmetic="exact_rational",
        eager_float_divergence=divergence,
    )
