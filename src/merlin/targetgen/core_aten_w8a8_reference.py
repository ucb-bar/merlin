"""Independent PyTorch W8A8 oracle for captured source layouts and host epilogues.

This walks the ORIGINAL source, never the prepared integer body or a device catalog.
Unsupported source operations refuse qualification rather than using compiled answers.
"""

from __future__ import annotations

import itertools


def quantize_codes(x, scale, *, quant_min, quant_max):
    """Exact rational divide and RNE/clamp, independent of reciprocal rounding."""
    import torch

    from merlin.targetgen.exact_arithmetic import quantize

    codes = quantize(x, scale, quant_min=quant_min, quant_max=quant_max)
    return torch.tensor(codes.tolist(), dtype=torch.int32).reshape(codes.shape)


def quantize_operand(original, *, quant_min, quant_max, eps):
    """Observe the runtime f32 calibration scale, then quantize by exact rational RNE."""
    import torch

    x = original.float()
    finite = torch.isfinite(x)
    valid = bool(finite.all())
    safe = torch.where(finite, x, torch.zeros_like(x))
    amax = safe.abs().amax()
    unclamped = amax / ((quant_max - quant_min) / 2.0)
    scale = torch.clamp(unclamped, min=eps) if valid else torch.tensor(eps, dtype=torch.float32)
    rounded = torch.round(safe * (1.0 / scale))
    codes = quantize_codes(safe, scale, quant_min=quant_min, quant_max=quant_max)
    metrics = dict(
        scale=scale.item(),
        amax=amax.item(),
        eps_clamp_hit=bool(unclamped <= eps),
        nonfinite_count=int((~finite).sum()),
        saturation_rate=float(((rounded < quant_min) | (rounded > quant_max)).float().mean()),
        round_to_zero_fraction=float((codes == 0).float().mean()),
        code_histogram={str(k): int((codes == k).sum()) for k in range(quant_min, quant_max + 1)},
    )
    return codes, scale, valid, metrics


def validate_dispatch(dispatch, *, full_k, quant_min, quant_max):
    """Recompute the per-primitive obligation independently of compiler helpers."""
    if not dispatch or dispatch.get("full_k") != full_k:
        raise ValueError("W8A8 oracle requires actual source-bound K dispatch")
    internal = dispatch["numerical_semantics"]["internal_arithmetic"]
    bits, storage_bits = internal["mac_result_bits"], internal["signed_operand_bits"]
    if (
        internal["mac_result_overflow"] != f"wrap_to_{bits}_bits"
        or internal["full_operation_overflow_policy"] != "bounded_exact_requires_each_partial_sum"
        or not -(1 << (storage_bits - 1)) <= quant_min <= quant_max < (1 << (storage_bits - 1))
    ):
        raise ValueError("W8A8 oracle refuses unsupported internal arithmetic")
    product_bound = max(abs(quant_min), abs(quant_max)) ** 2
    limit = ((1 << (bits - 1)) - 1) // product_bound
    ks = dispatch["primitive_ks"]
    if not ks or any(type(k) is not int or not 0 < k <= limit for k in ks):
        raise ValueError("W8A8 oracle refuses primitive K exceeding internal MAC bound")
    if full_k * product_bound > (1 << 31) - 1:
        raise ValueError("W8A8 oracle refuses inexact i32 aggregation")
    return limit


def evaluate(module, arrays, selected, *, quant_min, quant_max, eps, dispatch=None):
    import torch
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType
    from xdsl.ir.affine import AffineDimExpr

    selected = list(selected)
    dtypes = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "i1": torch.bool,
        "index": torch.int64,
        "f32": torch.float32,
        "f64": torch.float64,
        "i8": torch.int8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
    }
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    env = {
        arg: torch.from_numpy(array.copy()).to(dtypes[str(arg.type.element_type)]).reshape(arg.type.get_shape())
        for arg, array in zip(function.body.block.args, arrays, strict=True)
    }
    scales = []
    selected = set(selected)
    selected_order = {op: i for i, op in enumerate(op for op in module.walk() if op in selected)}

    def dtype(t):
        return dtypes[str(t.element_type if isinstance(t, TensorType) else t)]

    def attr(op, key):
        return op.properties[key] if key in op.properties else op.attributes[key]

    def constant(op):
        value = attr(op, "value")
        if isinstance(value, DenseIntOrFPElementsAttr):
            return torch.tensor(list(value.get_values()), dtype=dtype(op.results[0].type)).reshape(
                op.results[0].type.get_shape()
            )
        return torch.tensor(value.value.data, dtype=dtype(op.results[0].type))

    def scalar_body(block, operands, point=(), captured=None, single_operation=False):
        from merlin.targetgen.source_scalar import (
            convert_to_float,
            fused_multiply_add,
            logical_shift,
            unsigned_to_float,
        )

        local = dict(captured or {})
        local.update(zip(block.args, operands, strict=True))
        for inner in block.ops:
            name = inner.name
            xs = [local[v] if v in local else env[v] for v in inner.operands]
            if name == "linalg.yield":
                return xs[0]
            if name == "scf.yield":
                return tuple(xs)
            if name == "arith.constant":
                y = constant(inner)
            elif name == "arith.bitcast":
                target = dtype(inner.results[0].type)
                source_width = 1 if xs[0].dtype == torch.bool else xs[0].element_size() * 8
                target_width = 1 if target == torch.bool else torch.empty((), dtype=target).element_size() * 8
                if source_width != target_width:
                    raise ValueError("source bitcast must preserve storage width")
                y = xs[0].contiguous().view(target)
            elif name in {"arith.uitofp", "arith.extui"}:
                width = xs[0].element_size() * 8 if xs[0].dtype != torch.bool else 1
                target = dtype(inner.results[0].type)
                if name == "arith.uitofp":
                    y = unsigned_to_float(xs[0], width, target)
                else:
                    if width >= 64:
                        raise ValueError("source unsigned extension has no wider supported integer storage")
                    y = (xs[0].to(torch.int64) & ((1 << width) - 1)).to(target)
            elif name in {"arith.extf", "arith.truncf", "arith.extsi", "arith.trunci", "arith.sitofp"}:
                target = dtype(inner.results[0].type)
                if name in {"arith.extsi", "arith.sitofp"} and xs[0].dtype == torch.bool:
                    y = -xs[0].to(target)
                elif name == "arith.trunci" and target == torch.bool:
                    y = (xs[0] & 1).to(target)
                elif name in {"arith.extf", "arith.truncf", "arith.sitofp"}:
                    y = convert_to_float(xs[0], target)
                else:
                    y = xs[0].to(target)
            elif name in {"arith.addf", "arith.addi"}:
                y = xs[0] ^ xs[1] if name == "arith.addi" and xs[0].dtype == torch.bool else xs[0] + xs[1]
            elif name in {"arith.mulf", "arith.muli"}:
                y = xs[0] * xs[1]
            elif name in {"arith.subf", "arith.subi"}:
                y = xs[0] ^ xs[1] if name == "arith.subi" and xs[0].dtype == torch.bool else xs[0] - xs[1]
            elif name == "arith.divf":
                y = xs[0] / xs[1]
            elif name == "arith.negf":
                y = -xs[0]
            elif name == "math.fma":
                y = fused_multiply_add(*xs)
            elif name in {"arith.andi", "arith.ori", "arith.xori"}:
                y = {"arith.andi": torch.bitwise_and, "arith.ori": torch.bitwise_or, "arith.xori": torch.bitwise_xor}[
                    name
                ](*xs)
            elif name in {"arith.shrui", "arith.shli"}:
                width = xs[0].element_size() * 8 if xs[0].dtype != torch.bool else 1
                y = logical_shift(*xs, width, left=name == "arith.shli")
            elif name == "math.absf":
                y = torch.abs(xs[0])
            elif name == "linalg.index":
                y = torch.tensor(point[inner.dim.value.data], dtype=torch.int64)
            elif name == "tensor.extract":
                index = tuple(int(v) for v in xs[1:])
                y = xs[0][index]
            elif name == "scf.for":
                lower, upper, step = map(int, xs[:3])
                if step <= 0 or len(range(lower, upper, step)) > 1_000_000:
                    raise ValueError("source IEEE loop exceeds the closed static work budget")
                carried = tuple(xs[3:])
                for index in range(lower, upper, step):
                    carried = scalar_body(inner.body.block, (torch.tensor(index), *carried), point, local)
                local.update(zip(inner.results, carried, strict=True))
                continue
            elif name == "scf.if":
                region = inner.regions[0 if bool(xs[0]) else 1]
                local.update(zip(inner.results, scalar_body(region.block, (), point, local), strict=True))
                continue
            elif name == "arith.cmpi":
                predicate = inner.predicate.value.data
                left, right = xs
                if predicate in (2, 3, 4, 5) and left.dtype == torch.bool:
                    left, right = -left.to(torch.int8), -right.to(torch.int8)
                if predicate in (6, 7, 8, 9):
                    # Flipping the sign bit maps unsigned order to signed order,
                    # without a narrowing cast or a binary64 bridge for i64.
                    if left.dtype != torch.bool:
                        sign = -(1 << (left.element_size() * 8 - 1))
                        left, right = left ^ sign, right ^ sign
                    predicate -= 4
                if predicate not in range(6):
                    raise ValueError("unsupported source integer predicate")
                y = (left == right, left != right, left < right, left <= right, left > right, left >= right)[predicate]
            elif name in {"arith.divui", "arith.remui"}:
                if any(bool((v < 0).any()) for v in xs):
                    raise ValueError("source index division requires nonnegative operands")
                y = torch.div(xs[0], xs[1], rounding_mode="floor") if name == "arith.divui" else xs[0] % xs[1]
                y = y.to(dtype(inner.results[0].type))
            elif name == "arith.cmpf":
                predicate = inner.predicate.value.data
                left, right = xs
                unordered = torch.isnan(left) | torch.isnan(right)
                comparisons = (left == right, left > right, left >= right, left < right, left <= right, left != right)
                if predicate in range(1, 7):
                    y = comparisons[predicate - 1] & ~unordered
                elif predicate in range(8, 14):
                    y = comparisons[predicate - 8] | unordered
                elif predicate in (0, 7, 14, 15):
                    y = {0: torch.zeros_like(unordered), 7: ~unordered, 14: unordered, 15: torch.ones_like(unordered)}[
                        predicate
                    ]
                else:
                    raise ValueError("unsupported source floating predicate")
            elif name == "arith.select":
                y = torch.where(xs[0], xs[1], xs[2])
            else:
                raise ValueError(f"W8A8 reference refuses scalar op {name}")
            local[inner.results[0]] = y
        if single_operation:
            return y
        raise ValueError("source region lacks yield")

    def generic(op, xs):
        maps = [a.data for a in op.get_indexing_maps()]
        if op in selected:
            q, ss, valid, operands = [], [], [], []
            for original in xs[:2]:
                codes, scale, finite, metrics = quantize_operand(
                    original, quant_min=quant_min, quant_max=quant_max, eps=eps
                )
                q.append(codes.to(torch.int64))
                ss.append(scale)
                valid.append(finite)
                operands.append(metrics)
            labels = []
            for amap in maps:
                if any(not isinstance(expr, AffineDimExpr) for expr in amap.results):
                    raise ValueError("integer oracle requires projected contraction maps")
                labels.append("".join(chr(ord("a") + expr.position) for expr in amap.results))
            # PyTorch integer einsum uses integer mm/bmm; independently of emitted MAC loops.
            acc = torch.einsum(labels[0] + "," + labels[1] + "->" + labels[2], *q)
            reduction = set(labels[0] + labels[1]) - set(labels[2])
            if len(reduction) != 1:
                raise ValueError("oracle requires a single reduction dimension")
            reduction_label = next(iter(reduction))
            full_k = q[0].shape[labels[0].index(reduction_label)]
            schedule = dispatch[selected_order[op]] if dispatch is not None else None
            limit = (
                validate_dispatch(schedule, full_k=full_k, quant_min=quant_min, quant_max=quant_max)
                if schedule
                else None
            )
            if dispatch is not None and schedule is None:
                raise ValueError("W8A8 oracle lacks dispatch for selected contraction")
            partial_max = 0
            if schedule:
                # Each emitted primitive has a bounded K; independently measure
                # reference partial sums with that same maximum tile extent.
                tile = max(schedule["primitive_ks"])
                for offset in range(0, full_k, tile):
                    sliced = []
                    for value, label in zip(q, labels[:2], strict=True):
                        axes = [slice(None)] * value.ndim
                        axes[label.index(reduction_label)] = slice(offset, offset + tile)
                        sliced.append(value[tuple(axes)])
                    for prefix in range(1, sliced[0].shape[labels[0].index(reduction_label)] + 1):
                        prefixes = []
                        for value, label in zip(sliced, labels[:2], strict=True):
                            axes = [slice(None)] * value.ndim
                            axes[label.index(reduction_label)] = slice(0, prefix)
                            prefixes.append(value[tuple(axes)])
                        partial = torch.einsum(labels[0] + "," + labels[1] + "->" + labels[2], *prefixes)
                        partial_max = max(partial_max, int(partial.to(torch.int64).abs().max()))
            maximum = int(acc.to(torch.int64).abs().max())
            combined = ss[0].double() * ss[1].double()
            import numpy as np

            from merlin.targetgen.exact_arithmetic import exact_array, round_binary

            exact = exact_array(acc) * exact_array(ss[0]) * exact_array(ss[1])
            rounded = np.vectorize(round_binary, otypes=[float])(exact)
            result = torch.tensor(rounded, dtype=torch.float32)
            restored = result.to(xs[-1].dtype)
            if all(valid):
                if not bool(torch.isfinite(result).all() and torch.isfinite(restored).all()):
                    raise ValueError("int8_w8a8 dequantized result is not representable")
            else:
                restored = torch.full_like(restored, float("nan"))
            bits = schedule["numerical_semantics"]["internal_arithmetic"]["mac_result_bits"] if schedule else None
            scales.append(
                dict(
                    lhs=ss[0].item(),
                    rhs=ss[1].item(),
                    granularity="per_tensor",
                    output_lsb=combined.item(),
                    operands=operands,
                    accumulator=acc.tolist(),
                    accumulator_metrics=dict(
                        max_abs=maximum,
                        i32_headroom=(1 << 31) - 1 - maximum,
                        primitive_max_abs=partial_max if schedule else None,
                        internal_headroom=((1 << (bits - 1)) - 1 - partial_max) if bits else None,
                        internal_bound=((1 << (bits - 1)) - 1) if bits else None,
                        primitive_worst_case_bound=max(schedule["primitive_ks"])
                        * max(abs(quant_min), abs(quant_max)) ** 2
                        if schedule
                        else None,
                    ),
                    k_split=dict(
                        full_k=full_k,
                        primitive_ks=schedule["primitive_ks"] if schedule else None,
                        max_legal_primitive_k=limit,
                        split_required=full_k > limit if limit else None,
                    ),
                )
            )
            return restored
        extents = [None] * maps[0].num_dims
        for amap, x in zip(maps, xs, strict=True):
            for expr, size in zip(amap.results, x.shape, strict=True):
                if isinstance(expr, AffineDimExpr):
                    old = extents[expr.position]
                    if old is not None and old != size:
                        raise ValueError("inconsistent source loop extent")
                    extents[expr.position] = size
        if any(x is None for x in extents):
            raise ValueError("source loop extent is underivable")
        result = xs[-1].clone()
        for point in itertools.product(*(range(n) for n in extents)):
            indices = [amap.eval(point, ()) for amap in maps]
            values = [x[index] for x, index in zip(xs[:-1], indices[:-1], strict=True)]
            values.append(result[indices[-1]])
            result[indices[-1]] = scalar_body(op.body.block, values, point)
        return result

    for op in function.body.block.ops:
        xs = [env[v] for v in op.operands]
        name = op.name
        if name == "func.return":
            return xs, scales
        if name == "arith.constant":
            y = constant(op)
        elif name.startswith(("arith.", "math.")):
            from types import SimpleNamespace

            y = scalar_body(SimpleNamespace(args=(), ops=(op,)), [], single_operation=True)
        elif name == "tensor.empty":
            y = torch.empty(op.results[0].type.get_shape(), dtype=dtype(op.results[0].type))
        elif name in {"tensor.expand_shape", "tensor.collapse_shape", "tensor.reshape"}:
            y = xs[0].reshape(op.results[0].type.get_shape())
        elif name == "tensor.splat":
            y = xs[0].expand(op.results[0].type.get_shape()).clone()
        elif name == "linalg.fill":
            y = xs[0].expand(xs[1].shape).clone()
        elif name == "linalg.transpose":
            y = xs[0].permute(tuple(attr(op, "permutation").get_values())).contiguous()
        elif name in {"tensor.extract_slice", "tensor.insert_slice"}:
            offsets, sizes, strides = [
                list(attr(op, k).get_values()) for k in ("static_offsets", "static_sizes", "static_strides")
            ]
            if any(n < 0 for ns in (offsets, sizes, strides) for n in ns):
                raise ValueError("dynamic slice not supported by reference")
            indices = tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, sizes, strides, strict=True))
            if name == "tensor.extract_slice":
                y = xs[0][indices].reshape(op.results[0].type.get_shape())
            else:
                y = xs[1].clone()
                y[indices] = xs[0].reshape(y[indices].shape)
        elif name in {"linalg.generic", "linalg.matmul", "linalg.batch_matmul"}:
            y = generic(op, xs)
        elif name == "linalg.reduce":
            # Keep source reduction order, instead of Torch's tree-sum order.
            dimensions = tuple(attr(op, "dimensions").get_values())
            y = xs[1].clone()
            for point in itertools.product(*(range(n) for n in xs[0].shape)):
                index = tuple(v for i, v in enumerate(point) if i not in dimensions)
                y[index] = scalar_body(op.regions[0].block, [xs[0][point], y[index]])
        else:
            raise ValueError(f"W8A8 reference refuses source op {name}")
        if len(op.results) != 1:
            raise ValueError("reference requires single-result source operations")
        env[op.results[0]] = y
    raise ValueError("source function lacks return")


def run_pt2e_reference(model, inputs):
    """Exact rational source Q/DQ oracle; eager float divergence is informational."""
    from merlin.targetgen.exact_quantized_reference import run

    return run(model, inputs)


def selected_pt2e_reference(model, inputs, *, declared_readout=False, numerics="integer_readout"):
    """Bind the exact-arithmetic oracle independently of the chosen lowering numerics."""
    import copy
    import hashlib
    from pathlib import Path

    from merlin.runtime import exact_readout
    from merlin.targetgen import exact_arithmetic, exact_quantized_reference

    if numerics not in {"float_readout", "integer_readout"}:
        raise ValueError(f"unknown readout numerics {numerics!r}")
    source = Path(exact_quantized_reference.__file__).resolve()
    identity = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "arithmetic": "exact_rational",
        "numerics": numerics,
        "dependencies": [
            {
                "path": str(Path(exact_arithmetic.__file__).resolve()),
                "sha256": hashlib.sha256(Path(exact_arithmetic.__file__).read_bytes()).hexdigest(),
            },
            {
                "path": str(Path(exact_readout.__file__).resolve()),
                "sha256": hashlib.sha256(Path(exact_readout.__file__).read_bytes()).hexdigest(),
            },
        ],
    }
    return run_pt2e_reference(model, copy.deepcopy(inputs)), identity


def write_pt2e_reference(path, reference, leaves, source):
    """Retain independent outputs and contraction observations beside capture."""
    import json

    import torch

    from merlin.targetgen._capture_tensor_values import to_native

    def raw(value):
        return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes().hex()

    path.write_text(
        json.dumps(
            {
                "engine": "independent_pt2e_exact_rational",
                "arithmetic": "exact_rational",
                "eager_float_divergence": reference.eager_float_divergence,
                "outputs": [to_native(value) for value in leaves],
                "output_bytes": [raw(value) for value in leaves],
                "scales": reference.sites,
                "post_inputs": [to_native(value) for value in reference.post_inputs],
                "post_input_bytes": [raw(value) for value in reference.post_inputs],
                "source": source,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    import sys

    if "--quantized-native" in sys.argv:
        from merlin.targetgen.quantized_native_reference import main as quantized_main

        return quantized_main()

    import json
    import sys
    from pathlib import Path

    import numpy as np

    from merlin.frontends.linalg_mlir import parse_mlir_file

    directory = Path(sys.argv[1])
    prefix = "native-int8" if "--native" in sys.argv[2:] else "w8a8"
    request = json.loads((directory / f"{prefix}-request.json").read_text())
    module = parse_mlir_file(
        directory / ("model.native_reference.mlir" if prefix == "native-int8" else "model.before_w8a8.mlir")
    )
    operations = list(module.walk())
    with np.load(directory / "inputs.npz") as archive:
        arrays = [archive[f"in{i}"] for i in range(len(archive.files))]
    values, scales = evaluate(
        module,
        arrays,
        [operations[i] for i in request["selected_ordinals"]],
        quant_min=request["quant_min"],
        quant_max=request["quant_max"],
        eps=request["eps"],
        dispatch=request["dispatch"],
    )
    document = {
        "outputs": [v.tolist() for v in values],
        "scales": scales,
        "accumulators": [s["accumulator"] for s in scales],
        "output_bytes": [v.contiguous().numpy().tobytes().hex() for v in values],
        "engine": "independent_pytorch_integer_einsum",
        "calibration": "dynamic",
        "calibration_graph": None,
    }
    (directory / f"{prefix}-reference.json").write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
