#!/usr/bin/env python3
"""Static activation planes: a wide per-tensor activation grid as int8 digit planes -- capture venv.

A recipe's ``static_planes`` block (:func:`merlin.targetgen.quant_recipe.with_static_planes`) gives the
activation operand of selected contractions a grid of ``k`` int8 digits with ONE static scale per
tensor, calibrated by PT2E like any activation edge. With the recipe's int8 activation range
``[low, high]`` and ``base = high + 1``, the top digit spans ``[low, high]`` and every lower digit
``[0, base - 1]``, so the grid is ``[low * base**(k-1), high * base**(k-1) + base**(k-1) - 1]``: with
an int8 range and ``k = 2``, 15 bits where one plane carries 8.

After ``convert_pt2e`` each such contraction is rewritten into ``k`` int8 contractions against the
SAME int8 weight:

    q   = clamp(round(x * (1 / s)), grid)          (the PT2E quantize, on the wide grid)
    d_j = the base-``base`` digits of q, top first  (exact: floor and subtract on integers)
    y   = (sum_j base**j * float(d_j @ W)) * s * s_w + bias

The device runs each ``d_j @ W`` (int8 x int8 -> i32, the weight unchanged). The host combines the
accumulators inside the rescale it already applies to one (``* s * s_w`` per output column, then the
bias), so the rescale reads ``k`` accumulators and no pass is added. The quantize pass over ``x`` also
emits the digits, and one set of digits serves every contraction that reads the same quantized ``x``.

Only a Linear whose weight PT2E quantized to int8 is rewritten; the planes annotation selects nothing
else (``_recipe_quantizer``), so a planes-grid operand this module cannot rewrite is an error, never
a silent wide fake-quantization. Nothing here imports merlin, names a model or a target.
"""

from __future__ import annotations

import math
from typing import Any

#: Accumulators are i32: every digit-times-weight sum must fit, whatever the observed values.
_I32_MAX = (1 << 31) - 1
#: Created operations that move no element (their cost is not host arithmetic).
_LAYOUT = ("reshape", "transpose", "_int_mm", "view", "t")


def grid(low: int, high: int, planes: int) -> tuple[int, int, int]:
    """``(grid_min, grid_max, base)`` of ``planes`` digits whose top digit spans ``[low, high]``."""
    if not (isinstance(low, int) and isinstance(high, int) and low < 0 < high):
        raise ValueError("static planes need a signed integer activation range")
    if type(planes) is not int or planes < 2:
        raise ValueError("static planes need at least two digit planes")
    base = high + 1
    span = base ** (planes - 1)
    return low * span, high * span + span - 1, base


def storage_dtype(grid_min: int, grid_max: int) -> str:
    """The narrowest signed integer dtype name that holds the grid (PT2E's quantized storage)."""
    for bits in (16, 32):
        if -(1 << (bits - 1)) <= grid_min and grid_max < (1 << (bits - 1)):
            return f"int{bits}"
    raise ValueError(f"a grid of [{grid_min}, {grid_max}] has no integer storage dtype")


def in_scope(owner: str | None, owners: Any) -> bool:
    """Whether a contraction owned by module path ``owner`` is in a block's ``owners`` (``None``: all)."""
    if owners is None:
        return True
    if owner is None:
        return False
    return any(owner == prefix or owner.startswith(prefix + ".") for prefix in owners)


def _owner(node: Any) -> str | None:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, dict) or not stack:
        return None
    paths = [entry[0] for entry in stack.values() if isinstance(entry, (tuple, list)) and entry]
    return str(paths[-1]) if paths else None


def _fake_mode(graph_module: Any) -> Any:
    import torch
    from torch._guards import detect_fake_mode

    vals = [n.meta.get("val") for n in graph_module.graph.nodes if isinstance(n.meta.get("val"), torch.Tensor)]
    mode = detect_fake_mode(vals)
    if mode is None:
        raise ValueError("static planes need the exported graph's observed values")
    return mode


def _observe_attr(graph_module: Any, node: Any, mode: Any) -> None:
    """Give a stored-tensor node (a folded int8 weight, its scales) the observed value emission needs."""
    if node.op != "get_attr" or "val" in node.meta:
        return
    value = graph_module
    for part in str(node.target).split("."):
        value = getattr(value, part)
    node.meta["val"] = mode.from_tensor(value)


def _cost(created: list[Any]) -> int:
    """Host elementwise work of the created operations: one per output element of each."""
    total = 0
    for node in created:
        name = getattr(node.target, "__name__", str(node.target)).split(".")[0]
        if name not in _LAYOUT:
            total += math.prod(node.meta["val"].shape)
    return total


def rewrite_static_planes(graph_module: Any, *, planes: int, low: int, high: int) -> list[dict[str, Any]]:
    """Rewrite every Linear whose activation is quantized on the planes grid; return one row per site.

    ``low``/``high`` are the recipe's int8 activation range. Raises ``ValueError`` for a planes-grid
    operand whose contraction is not that form, so a capture never keeps a wide fake-quantization.
    """
    import torch
    from _dynamic_contractions import Emitter  # noqa: PLC0415 -- a sibling, by bare name

    aten, qd = torch.ops.aten, torch.ops.quantized_decomposed
    grid_min, grid_max, base = grid(low, high, planes)
    graph = graph_module.graph

    def on_grid(dq: Any) -> bool:
        return bool(
            isinstance(dq, torch.fx.Node)
            and dq.op == "call_function"
            and dq.target == qd.dequantize_per_tensor.default
            and tuple(dq.args[3:5]) == (grid_min, grid_max)
        )

    sites = [n for n in graph.nodes if n.op == "call_function" and any(on_grid(a) for a in n.args[:2])]
    if not sites:
        return []
    mode = _fake_mode(graph_module)
    digits_of: dict[Any, list[Any]] = {}
    rows = []
    for node in sites:
        if node.target != aten.linear.default or not on_grid(node.args[0]) or on_grid(node.args[1]):
            raise ValueError(f"{node.name}: a planes-grid operand outside a Linear's activation")
        dq_x, dq_w = node.args[0], node.args[1]
        q_x = dq_x.args[0]
        if not (
            isinstance(q_x, torch.fx.Node)
            and q_x.target == qd.quantize_per_tensor.default
            and tuple(q_x.args[1:5]) == tuple(dq_x.args[1:5])
        ):
            raise ValueError(f"{node.name}: the planes-grid dequantize does not read its own quantize")
        x, scale, zero = q_x.args[0], q_x.args[1], q_x.args[2]
        if not isinstance(scale, float) or zero != 0:
            raise ValueError(f"{node.name}: a planes grid needs one symmetric float scale")
        per_channel = dq_w.target == qd.dequantize_per_channel.default
        if per_channel:
            w, w_scale, w_zero, axis, w_low, w_high, w_dtype = dq_w.args[:7]
            if axis != 0:
                raise ValueError(f"{node.name}: weight scales must be per output channel")
        elif dq_w.target == qd.dequantize_per_tensor.default:
            w, w_scale, w_zero, w_low, w_high, w_dtype = dq_w.args[:6]
        else:
            raise ValueError(f"{node.name}: the weight is not a PT2E-quantized stored tensor")
        if w_dtype != torch.int8 or getattr(w, "op", None) != "get_attr":
            raise ValueError(f"{node.name}: the weight is not a stored int8 tensor")
        for stored in (w, w_scale, w_zero):
            if isinstance(stored, torch.fx.Node):
                _observe_attr(graph_module, stored, mode)
        if isinstance(w_zero, torch.fx.Node):
            if bool(getattr(graph_module, str(w_zero.target)).ne(0).any()):
                raise ValueError(f"{node.name}: the weight has a zero point")
        elif w_zero != 0:
            raise ValueError(f"{node.name}: the weight has a zero point")
        shape = list(x.meta["val"].shape)
        reduction, outputs = shape[-1], int(w.meta["val"].shape[0])
        if reduction * max(-low, high) * max(-w_low, w_high) > _I32_MAX:
            raise ValueError(f"{node.name}: an i32 accumulator may overflow for reduction K={reduction}")
        site = Emitter(graph_module, node, mode=mode, transform="static_planes")
        cost = 0
        if q_x not in digits_of:
            # The digits replace the quantize: emitted where it stands, shared by all its readers.
            emit = Emitter(graph_module, q_x, mode=mode, transform="static_planes")
            emit.lineage = sorted({*emit.lineage, *site.lineage})
            value = x if x.meta["val"].dtype == torch.float32 else emit(aten.to.dtype, x, torch.float32)
            rows2d = emit(aten.reshape.default, value, [-1, reduction])
            rest = emit(
                aten.clamp.default,
                emit(aten.round.default, emit(aten.mul.Tensor, rows2d, 1.0 / scale)),
                grid_min,
                grid_max,
            )
            digits = []
            for power in range(planes - 1, -1, -1):
                digit = rest
                if power:
                    unit = float(base**power)
                    digit = emit(aten.floor.default, emit(aten.div.Tensor, rest, unit))
                    rest = emit(aten.sub.Tensor, rest, emit(aten.mul.Tensor, digit, unit))
                digits.append(emit(aten._to_copy.default, digit, dtype=torch.int8))
            digits_of[q_x] = digits
            cost += _cost(emit.created)
        weight_t = site(aten.transpose.int, w, 0, 1)
        accumulators = [site(aten._int_mm.default, digit, weight_t) for digit in digits_of[q_x]]
        value = site(aten._to_copy.default, accumulators[0], dtype=torch.float32)
        for accumulator in accumulators[1:]:
            lower = site(aten._to_copy.default, accumulator, dtype=torch.float32)
            value = site(aten.add.Tensor, site(aten.mul.Tensor, value, float(base)), lower)
        value = site(aten.mul.Tensor, value, scale)
        value = site(aten.mul.Tensor, value, w_scale)
        if per_channel:
            value = site(aten._to_copy.default, value, dtype=torch.float32)
        value = site(aten.reshape.default, value, [*shape[:-1], outputs])
        bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
        if bias is not None:
            if isinstance(bias, torch.fx.Node):
                _observe_attr(graph_module, bias, mode)
            value = site(aten.add.Tensor, value, bias)
        expected = node.meta.get("val")
        if expected is None or tuple(value.meta["val"].shape) != tuple(expected.shape):
            raise ValueError(f"static planes changed {node.name}'s result shape")
        if value.meta["val"].dtype != expected.dtype:
            value = site(aten.to.dtype, value, expected.dtype)
        cost += _cost(site.created)
        rows.append(
            {
                "node": node.name,
                "owner": _owner(node),
                "shape": [math.prod(shape[:-1]), reduction, outputs],
                "planes": planes,
                "scale": scale,
                "int8_macs": planes * math.prod(shape[:-1]) * reduction * outputs,
                "device_calls": planes,
                "host_elements": cost,
            }
        )
        node.replace_all_uses_with(value)
        graph.erase_node(node)
        for dead in (dq_x, dq_w):
            if not dead.users:
                graph.erase_node(dead)
    for q_x in digits_of:
        if not q_x.users:
            graph.erase_node(q_x)
    for node in list(graph.nodes):
        if node.op == "get_attr" and not node.users:
            graph.erase_node(node)
    graph.lint()
    graph_module.recompile()
    return rows
