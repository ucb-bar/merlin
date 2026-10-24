"""Exact integer recipe extraction from converted Q/DQ source graphs."""

from __future__ import annotations

import math
from typing import Any


def data_arguments(node: Any) -> list[Any]:
    """Recipe operands whose values participate, excluding shape-only tensors.

    Annotation and exact recipe extraction share this semantic operand selection.
    In particular a pool adjoint reads its gradient, while its source tensor
    supplies only the destination geometry.
    """
    name = str(node.target)
    if name.startswith(("aten.fill.", "aten.full.", "aten.full_like.", "aten.zeros_like.", "aten.ones_like.")):
        return []
    if name.startswith("aten.cat."):
        return list(node.args[0]) if node.args else []
    if name.startswith(("aten.add.Tensor", "aten.sub.Tensor", "aten.slice_scatter.", "aten.select_scatter.")):
        return list(node.args[:2])
    return list(node.args[:1])


def _per_tensor_scale(node: Any, operator: str) -> float | None:
    """The literal per-tensor scale of a folded quantize/dequantize node, or None."""
    if getattr(node, "op", None) != "call_function" or not str(node.target).startswith(operator):
        return None
    import torch

    if len(node.args) < 6:
        return None
    _, _, zero, low, high, dtype = node.args[:6]
    limits = {torch.int8: (-128, 127), torch.uint8: (0, 255)}.get(dtype)
    if (
        limits is None
        or any(type(v) is not int for v in (zero, low, high))
        or not limits[0] <= low < high <= limits[1]
        or not low <= zero <= high
    ):
        return None
    scale = node.args[1] if len(node.args) > 1 else None
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        return None
    from merlin.targetgen.exact_arithmetic import round_binary

    return float(scale) if round_binary(scale) == scale else None


def _window_divisor(node: Any) -> int | None:
    """The additive reduction divisor, or None when source geometry must be retained."""
    argument = node.args[0] if node.args and hasattr(node.args[0], "meta") else None
    source = argument.meta.get("val") if argument is not None else None
    # Converted quantize/dequantize nodes often carry no value metadata; walk the Q/DQ chain to the
    # first value that does (they all have the same shape).
    while source is None and argument is not None and argument.args and hasattr(argument.args[0], "meta"):
        argument = argument.args[0]
        source = argument.meta.get("val")
    result = node.meta.get("val")
    if source is None or result is None or not hasattr(source, "shape"):
        return None
    name = str(node.target)
    if "backward" in name or name.startswith("aten.col2im"):
        return None  # a transpose spreads each window over many outputs: no single divisor per output
    shape = [int(extent) for extent in source.shape]
    if name.startswith("aten.sum."):
        return 1
    if name.startswith("aten.mean."):
        if not shape:
            return 1  # a scalar reduction has one element, including explicit axis 0/-1
        dims = node.args[1] if len(node.args) > 1 else None
        dims = [dims] if isinstance(dims, int) else list(dims or range(len(shape)))
        return math.prod(shape[dim % len(shape)] for dim in dims) or None
    if "adaptive_avg_pool" in name:
        out = [int(extent) for extent in result.shape]
        rank = len(node.args[1]) if len(node.args) > 1 else 0
        if not rank or any(i % o for i, o in zip(shape[-rank:], out[-rank:], strict=True)):
            return None  # windows of different sizes
        return math.prod(i // o for i, o in zip(shape[-rank:], out[-rank:], strict=True))
    if "avg_pool" in name:
        kernel = node.args[1] if len(node.args) > 1 else None
        padding = node.args[3] if len(node.args) > 3 else 0
        override = node.args[6] if len(node.args) > 6 else None
        if len(node.args) > 4 and node.args[4]:
            return None  # ceil_mode: a partial last window divides by fewer elements
        if isinstance(override, int) and override > 0:
            return override
        if any(int(p) for p in (padding if isinstance(padding, (list, tuple)) else [padding])):
            return None  # padded border windows divide by a different count
        return math.prod(int(k) for k in (kernel if isinstance(kernel, (list, tuple)) else [kernel]))
    return None


def integer_recipes(graph_module: Any, *, declared_domain=None) -> list[dict[str, Any]]:
    """Integer recipe parameters for each converted affine map and movement, keyed by node name.

    Reads the scales PT2E calibrated on the node's own dequantize inputs and quantize output, and
    states the operation in the form :mod:`merlin.targetgen.recipe_numerics` defines: integer weights
    and an int32 constant into the accumulator, then one readout. A node whose scales are not literal
    per-tensor constants, or whose form the numerics refuse, is recorded with the reason.
    """
    from merlin.targetgen import recipe_numerics as RN
    from merlin.targetgen.exact_arithmetic import rational

    rows = []
    for node in graph_module.graph.nodes:
        family = node.meta.get("merlin_recipe_family")
        if family is None:
            continue
        name = str(node.target)
        inputs = [
            _per_tensor_scale(arg, "quantized_decomposed.dequantize_per_tensor")
            for arg in data_arguments(node)
            if hasattr(arg, "op")
        ]
        outputs = [_per_tensor_scale(user, "quantized_decomposed.quantize_per_tensor") for user in node.users]
        s_out = next((scale for scale in outputs if scale is not None), None)
        row: dict[str, Any] = {"node": node.name, "operator": name, "family": family, "input_scales": inputs}
        try:
            if s_out is None or None in inputs:
                raise RN.RecipeNumericsError(
                    "the node's quantization scales/ranges require literal valid byte per-tensor qparams"
                )
            output_nodes = [
                user
                for user in node.users
                if _per_tensor_scale(user, "quantized_decomposed.quantize_per_tensor") is not None
            ]
            if any(tuple(user.args[1:6]) != tuple(output_nodes[0].args[1:6]) for user in output_nodes):
                raise RN.RecipeNumericsError("multiple output quantizers require different exact recipes")
            bounds = {"qmin": output_nodes[0].args[3], "qmax": output_nodes[0].args[4]}
            row["output_scale"] = s_out
            input_nodes = [arg for arg in data_arguments(node) if hasattr(arg, "op")]
            parameter_nodes = [*input_nodes, *output_nodes]
            if any(
                arg.args[2] != 0
                or str(arg.args[5]) != "torch.int8"
                or arg.args[3] not in (-128, -127)
                or arg.args[4] != 127
                for arg in parameter_nodes
            ):
                if declared_domain is not None and family in {"affine_map", "operand_sum"}:
                    row["declared_affine_domain"] = dict(
                        status="refused", reason="centered affine byte domain lacks a floating intermediate proof"
                    )
                    raise RN.RecipeNumericsError(row["declared_affine_domain"]["reason"])
                # The source Q/DQ expression retains all centering, boundary
                # predicates and one final RNE. These parameters do not grant
                # a device schedule: integerization, independent exact oracle
                # and selected provider coverage still qualify the full source.
                row.update(
                    readout="full_i32",
                    requantization="retained_source_exact_integer",
                    input_qparams=[
                        dict(
                            scale=arg.args[1],
                            zero_point=arg.args[2],
                            quant_min=arg.args[3],
                            quant_max=arg.args[4],
                            dtype=str(arg.args[5]),
                        )
                        for arg in input_nodes
                    ],
                    output_qparams=dict(
                        scale=s_out, zero_point=output_nodes[0].args[2], **bounds, dtype=str(output_nodes[0].args[5])
                    ),
                    integer_recipe=dict(
                        form="retained_affine_qdq",
                        centering="retained_source_integer",
                        readout="full_i32",
                        rounding="half_to_even",
                    ),
                )
                if family not in {"affine_map", "movement", "window_mean", "operand_sum"}:
                    raise RN.RecipeNumericsError("unproved byte Q/DQ family")
                rows.append(row)
                continue
            scalar = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], (int, float)) else None
            alpha = rational(
                node.args[2]
                if name.startswith(("aten.add.Scalar", "aten.sub.Scalar")) and len(node.args) > 2
                else node.kwargs.get("alpha", 1)
            )
            coefficients, offset = None, rational(0)
            if name.startswith(("aten.sub.Tensor", "aten.add.Tensor")) and scalar is None and len(inputs) == 2:
                sign = -1 if name.startswith("aten.sub") else 1
                coefficients = (rational(inputs[0]), sign * alpha * rational(inputs[1]))
            elif name.startswith(("aten.add.", "aten.sub.")) and scalar is not None:
                sign = -1 if name.startswith("aten.sub") else 1
                coefficients, offset = (rational(inputs[0]),), sign * alpha * rational(scalar)
            elif name.startswith("aten.mul.") and scalar is not None:
                coefficients = (rational(scalar) * rational(inputs[0]),)
            elif name.startswith("aten.div.") and scalar is not None:
                coefficients = (rational(inputs[0]) / rational(scalar),)
            elif name.startswith("aten.neg"):
                coefficients = (-rational(inputs[0]),)
            elif name.startswith("aten.relu"):
                coefficients = (rational(inputs[0]),)
            recipe = None
            if coefficients is not None:
                if declared_domain is not None:
                    from merlin.targetgen.affine_float_domain import prove_affine_domain

                    try:
                        row["declared_affine_domain"] = prove_affine_domain(coefficients, offset)
                    except ValueError as exc:
                        row["declared_affine_domain"] = dict(status="refused", reason=str(exc))
                        raise RN.RecipeNumericsError(str(exc)) from exc
                activation = "relu" if name.startswith("aten.relu") else "none"
                try:
                    if len(coefficients) == 1:
                        recipe = RN.affine(
                            inputs[0],
                            s_out,
                            gain=coefficients[0] / rational(inputs[0]),
                            offset=offset,
                            activation=activation,
                            **bounds,
                        )
                    else:
                        recipe = RN.linear_combination(
                            inputs[0], abs(coefficients[1]), s_out, sign_b=-1 if coefficients[1] < 0 else 1, **bounds
                        )
                except RN.RecipeNumericsError:
                    # A bounded common integer seed/weight is one possible schedule,
                    # not a requirement on exact source semantics. Retain the real
                    # coefficients and bias on host; provider source coverage and
                    # the independent integerization oracle still gate placement.
                    row["integer_recipe"] = RN.retained_affine_readout(
                        coefficients, offset, s_out, activation=activation, **bounds
                    )
            if family == "window_mean":
                divisor = _window_divisor(node)
                # Variable windows and adjoints retain their exact source
                # geometry/coefficient expression. A single constant divisor is
                # only a simplification, not a prerequisite for exact lowering.
                ratio = RN.window_mean_readout(inputs[0], divisor or 1, s_out, **bounds)
                row.update(
                    readout="full_i32",
                    requantization="retained_source_exact_integer" if divisor is None else "exact_rational_host",
                    scale_numerator=str(ratio.numerator),
                    scale_denominator=str(ratio.denominator),
                )
                row["divisor"] = "retained_source_geometry" if divisor is None else divisor
            if recipe is not None:
                row["integer_recipe"] = recipe.to_dict()
            elif family == "movement":
                value = (
                    node.args[1]
                    if name.startswith(("aten.fill", "aten.full"))
                    else node.args[2]
                    if name.startswith("aten.constant_pad_nd") and len(node.args) > 2
                    else node.args[3]
                    if name.startswith("aten.pad") and len(node.args) > 3
                    else 0.0
                    if name.startswith(("aten.constant_pad_nd", "aten.pad"))
                    else None
                )
                if isinstance(value, (int, float)):
                    row["constant_code"] = RN.constant_code(value, s_out, **bounds)
                if name.startswith("aten.zeros_like."):
                    row["constant_code"] = RN.constant_code(0, s_out, **bounds)
                elif name.startswith("aten.ones_like."):
                    row["constant_code"] = RN.constant_code(1, s_out, **bounds)
                row["inputs"] = [RN.requantize(scale, s_out, **bounds) for scale in inputs]
            elif family != "window_mean" and "integer_recipe" not in row:
                raise RN.RecipeNumericsError("no exact integer recipe for the annotated operation")
        except (RN.RecipeNumericsError, TypeError, ValueError, ZeroDivisionError) as exc:
            row["refused"] = str(exc)
        rows.append(row)
    return rows
