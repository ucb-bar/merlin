"""Independent rational evaluation of original source cases and byte boundaries.

Never imports lowered components, provider schedules or simulator outputs.
"""

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path


def canonical_scalar(dtype, operation, value):
    """ATen scalar promotion precedes the exact rational operation.

    Add/sub materialize their scalar in storage; mul/div use opmath. This
    depends only on the declared dtype and literal, never tensor payloads.
    """
    import math

    from merlin.targetgen.exact_arithmetic import round_binary

    dtype = {"complex<f32>": "f32", "complex<f64>": "f64"}.get(dtype, dtype)
    if isinstance(value, complex):
        component = dtype
        return complex(
            canonical_scalar(component, operation, value.real), canonical_scalar(component, operation, value.imag)
        )
    if dtype not in {"f16", "bf16", "f32", "f64"} or not math.isfinite(value):
        return value
    target = dtype if operation in {"aten.add.Scalar", "aten.sub.Scalar"} else "f64" if dtype == "f64" else "f32"
    try:
        return round_binary(
            Fraction(value),
            16 if target in {"f16", "bf16"} else 32 if target == "f32" else 64,
            bfloat16=target == "bf16",
        )
    except OverflowError:
        return math.copysign(float("inf"), value)


def main():
    import numpy as np
    import torch

    from merlin.targetgen.core_aten_capture import loader_source
    from merlin.targetgen.exact_arithmetic import ExactTensor, quantize, round_binary
    from merlin.targetgen.exact_quantized_reference import evaluate_operation

    directory = Path(sys.argv[1])
    request = json.loads((directory / "quantized-native-request.json").read_text())
    p = request["contract"]
    scale = Fraction(p["scale_numerator"], p["scale_denominator"])
    import merlin.targetgen.exact_quantized_reference as oracle

    original_rational = oracle.rational
    oracle.rational = lambda x: Gaussian.of(x) if isinstance(x, (Gaussian, complex)) else original_rational(x)

    def exact_array(value):
        if isinstance(value, ExactTensor):
            return value.values
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        return np.vectorize(oracle.rational, otypes=[object])(np.asarray(value, dtype=object))

    oracle.exact_array = exact_array
    outputs = {}
    for record in request["records"]:
        case = record["case"]
        loader = directory / "quantized-reference-source.py"
        loader.write_text(loader_source(case))
        spec = importlib.util.spec_from_file_location("quantized_reference_source", loader)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        _, inputs = source.get_model_and_inputs()
        exact = []
        for value in inputs:
            if value.is_complex():
                real = quantize(value.real, scale, nonfinite=p["nonfinite"]) * scale
                imag = quantize(value.imag, scale, nonfinite=p["nonfinite"]) * scale
                values = np.vectorize(Gaussian, otypes=[object])(real, imag)
            elif not value.is_floating_point():
                values = np.asarray(value.detach().cpu().tolist(), dtype=object)
            else:
                values = quantize(value, scale, nonfinite=p["nonfinite"]) * scale
            exact.append(ExactTensor(np.asarray(values, dtype=object), torch.float32))
        arguments = source._decode(source._DOCUMENT["arguments"], exact)
        # Dtypes govern only the final required dequantization, not accumulation.
        args, kwargs = arguments["args"], arguments["kwargs"]
        kwargs = dict(kwargs)
        if case["overload"] in {"aten.add.Scalar", "aten.sub.Scalar", "aten.mul.Scalar", "aten.div.Scalar"}:
            dtype = record["input_abi"][0]["dtype"]
            if len(args) > 1:
                args = list(args)
                args[1] = canonical_scalar(dtype, case["overload"], args[1])
            elif "other" in kwargs:
                kwargs["other"] = canonical_scalar(dtype, case["overload"], kwargs["other"])

        if "dtype" in kwargs and kwargs["dtype"] is not None:
            kwargs["dtype"] = torch.float32
        op = source._op(case["overload"])
        if (
            str(op).split(".")[1] in {"sum", "mean", "amax", "amin"}
            and args
            and isinstance(args[0], ExactTensor)
            and args[0].values.ndim == 0
        ):
            if len(args) > 1:
                args = list(args)
                args[1] = []
            elif "dim" in kwargs:
                kwargs["dim"] = []

        if str(op) == "aten._fft_r2c.default":
            result = rational_real_transform(args, oracle.exact_array, ExactTensor)
        elif str(op) == "aten._fft_c2r.default":
            result = rational_inverse_real_transform(args, oracle.exact_array, ExactTensor)
        elif str(op) == "aten.convolution_backward.default":
            result = convolution_adjoint(args, oracle.exact_array, ExactTensor)
        elif str(op) == "aten.convolution.default" and args[6]:
            result = transposed_convolution(args, oracle.exact_array, ExactTensor)
        else:
            result = evaluate_operation(op, args, kwargs)
        leaves = [value for value in torch.utils._pytree.tree_leaves(result) if value is not None]
        if len(leaves) != len(record["output_indices"]):
            raise ValueError("quantized oracle output count differs")
        for leaf, index, abi in zip(leaves, record["output_indices"], record["output_abi"], strict=True):
            if not isinstance(leaf, ExactTensor):
                raise ValueError("non-rational source output")
            dtype = abi["dtype"]
            if dtype == "i1":
                # A literal endpoint mask is a Boolean storage result, not a
                # calibrated arithmetic output. Keep every bit independently.
                if any(v not in (False, True, 0, 1) for v in leaf.values.flat):
                    raise ValueError("non-Boolean rational mask")
                outputs[str(index)] = np.asarray(leaf.values, dtype=np.bool_).tobytes().hex()
                continue
            if dtype == "complex<f32>":
                real = np.vectorize(lambda v: Gaussian.of(v).real, otypes=[object])(leaf.values)
                imag = np.vectorize(lambda v: Gaussian.of(v).imag, otypes=[object])(leaf.values)
                values = np.stack([quantize(real, scale), quantize(imag, scale)], axis=-1) * scale
                dtype = "f32"
            else:
                values = quantize(leaf, scale) * scale
            if dtype not in {"f16", "bf16", "f32", "f64"}:
                raise ValueError("non-floating quantized output")
            rounded = [
                round_binary(
                    v, 16 if dtype in {"f16", "bf16"} else 64 if dtype == "f64" else 32, bfloat16=dtype == "bf16"
                )
                for v in np.asarray(values, dtype=object).flat
            ]
            tensor = torch.tensor(
                rounded,
                dtype={"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}[dtype],
            )
            outputs[str(index)] = bytes(tensor.contiguous().view(torch.uint8).tolist()).hex()
        # Nonmutating source arguments are also exported by the batch ABI.
        for index, original in zip(record.get("native_eager_indices", {}).get("posts", ()), inputs):
            outputs[str(index)] = bytes(original.detach().contiguous().reshape(-1).view(torch.uint8).tolist()).hex()
    (directory / "quantized-native-reference.json").write_text(
        json.dumps(dict(expected_output_bytes=outputs, engine="independent_source_exact_rational")) + "\n"
    )


class Gaussian:
    """Two rational components; no binary floating complex arithmetic."""

    def __init__(self, real=0, imag=0):
        self.real, self.imag = Fraction(real), Fraction(imag)

    @staticmethod
    def of(value):
        return (
            value
            if isinstance(value, Gaussian)
            else Gaussian(value.real, value.imag)
            if isinstance(value, complex)
            else Gaussian(value)
        )

    def __add__(self, other):
        other = self.of(other)
        return Gaussian(self.real + other.real, self.imag + other.imag)

    __radd__ = __add__

    def __neg__(self):
        return Gaussian(-self.real, -self.imag)

    def __sub__(self, other):
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) + -self

    def __mul__(self, other):
        other = self.of(other)
        return Gaussian(
            self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.of(other)
        d = other.real**2 + other.imag**2
        return Gaussian(
            (self.real * other.real + self.imag * other.imag) / d, (self.imag * other.real - self.real * other.imag) / d
        )


def rational_real_transform(args, array, Tensor):
    """Independent DFT where every required root of unity is rational.

    No trigonometric approximation or compiler coefficient table is consulted.
    The fourth roots are exact; other roots require an algebraic oracle.
    """
    import math

    import numpy as np

    values = array(args[0])
    axes, normalization, onesided = args[1:4]
    total = math.prod(values.shape[axis] for axis in axes)
    roots = (Gaussian(1), Gaussian(0, -1), Gaussian(-1), Gaussian(0, 1))
    for axis in axes:
        n = values.shape[axis]
        if n not in {1, 2, 4}:
            raise ValueError("transform needs a non-rational root of unity")
        result = np.empty(values.shape, dtype=object)
        for point in np.ndindex(values.shape):
            acc = Gaussian()
            for j in range(n):
                source = list(point)
                source[axis] = j
                acc += Gaussian.of(values[tuple(source)]) * roots[(4 * point[axis] * j // n) % 4]
            result[point] = acc
        values = result
    divisor = 1 if normalization == 0 else total if normalization == 2 else math.isqrt(total)
    if normalization not in {0, 1, 2} or normalization == 1 and divisor * divisor != total:
        raise ValueError("transform normalization is not rational")
    if divisor != 1:
        values = np.vectorize(lambda v: v / divisor, otypes=[object])(values)
    if onesided:
        selection = [slice(None)] * values.ndim
        selection[axes[-1]] = slice(0, values.shape[axes[-1]] // 2 + 1)
        values = values[tuple(selection)]
    return Tensor(values, args[0].dtype)


def rational_inverse_real_transform(args, array, Tensor):
    """Hermitian completion followed by an exact rational inverse DFT."""
    import numpy as np

    half = array(args[0])
    axes, normalization, size = args[1:4]
    shape = list(half.shape)
    shape[axes[-1]] = size
    complete = np.empty(shape, dtype=object)
    for point in np.ndindex(tuple(shape)):
        source = list(point)
        mirrored = point[axes[-1]] > size // 2
        if mirrored:
            for axis in axes:
                source[axis] = -source[axis] % shape[axis]
        value = Gaussian.of(half[tuple(source)])
        # conjugate(DFT(conjugate(x))) is the unnormalized inverse DFT.
        complete[point] = Gaussian(value.real, value.imag if mirrored else -value.imag)
    result = rational_real_transform(
        [Tensor(complete, args[0].dtype), axes, normalization, False], lambda v: v.values, Tensor
    ).values
    return Tensor(np.vectorize(lambda v: v.real, otypes=[object])(result), args[0].dtype)


def transposed_convolution(args, array, Tensor):
    import numpy as np
    import torch

    x, w, bias = args[:3]
    stride, pad, dil, _, opad, groups = args[3:9]
    x, w = array(x), array(w)
    spatial = tuple(
        (n - 1) * s - 2 * p + d * (k - 1) + o + 1
        for n, s, p, d, k, o in zip(x.shape[2:], stride, pad, dil, w.shape[2:], opad, strict=True)
    )
    out = np.full((x.shape[0], w.shape[1] * groups, *spatial), Fraction(0), dtype=object)
    if bias is not None:
        out += array(bias).reshape((1, -1) + (1,) * len(spatial))
    cg = x.shape[1] // groups
    for point in np.ndindex(x.shape):
        n, ci, *position = point
        for co in range(w.shape[1]):
            for kernel in np.ndindex(w.shape[2:]):
                dest = tuple(
                    i * s - p + k * d for i, s, p, k, d in zip(position, stride, pad, kernel, dil, strict=True)
                )
                if all(0 <= i < size for i, size in zip(dest, spatial, strict=True)):
                    out[(n, ci // cg * w.shape[1] + co, *dest)] += x[point] * w[(ci, co, *kernel)]
    return Tensor(out, torch.float32)


def convolution_adjoint(args, array, Tensor):
    import numpy as np
    import torch

    gradient, source, weight = args[:3]
    stride, pad, dil = args[4:7]
    transposed, opad, groups, mask = args[7:11]
    g, x, w = map(array, (gradient, source, weight))
    dx = np.full(x.shape, Fraction(0), dtype=object)
    dw = np.full(w.shape, Fraction(0), dtype=object)
    db = np.sum(g, axis=(0, *range(2, g.ndim)))
    cg = x.shape[1] // groups
    fg = g.shape[1] // groups
    # Enumerate the primal's exact source/destination relation. No float sums.
    if transposed:
        for point in np.ndindex(x.shape):
            n, ci, *position = point
            for local_co in range(fg):
                co = ci // cg * fg + local_co
                for kernel in np.ndindex(w.shape[2:]):
                    dest = tuple(
                        i * s - p + k * d for i, s, p, k, d in zip(position, stride, pad, kernel, dil, strict=True)
                    )
                    if not all(0 <= i < size for i, size in zip(dest, g.shape[2:], strict=True)):
                        continue
                    wi = (ci, local_co, *kernel)
                    gi = (n, co, *dest)
                    if mask[0]:
                        dx[point] += g[gi] * w[wi]
                    if mask[1]:
                        dw[wi] += g[gi] * x[point]
    else:
        for gi in np.ndindex(g.shape):
            n, co, *position = gi
            for local_ci in range(cg):
                ci = co // fg * cg + local_ci
                for kernel in np.ndindex(w.shape[2:]):
                    dest = tuple(
                        i * s - p + k * d for i, s, p, k, d in zip(position, stride, pad, kernel, dil, strict=True)
                    )
                    if not all(0 <= i < size for i, size in zip(dest, x.shape[2:], strict=True)):
                        continue
                    xi = (n, ci, *dest)
                    wi = (co, local_ci, *kernel)
                    if mask[0]:
                        dx[xi] += g[gi] * w[wi]
                    if mask[1]:
                        dw[wi] += g[gi] * x[xi]
    return tuple(Tensor(v, torch.float32) if wanted else None for v, wanted in zip((dx, dw, db), mask, strict=True))
