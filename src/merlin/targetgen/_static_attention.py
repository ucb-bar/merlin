#!/usr/bin/env python3
"""Activation contractions with STATIC per-head scales -- runs INSIDE the capture venv.

The ``static_heads`` mode of a recipe's ``dynamic`` block (:func:`merlin.targetgen.quant_recipe.
with_dynamic_contractions`) uses the block's mechanisms with scales fixed at capture time, so each device
call (one per head) reads ONE scale per operand, the form a readout applies itself:

* per-head scales: an operand ``X[..., H, M, K]`` of a selected contraction takes one symmetric scale per
  index of its head axis (the axis before the matrix), the minmax of that head over the calibration pass,
  ``scale = absmax / ((quant_max - quant_min) / 2)``, clamped to the recipe's observer epsilon;
* smooth shift: where the contraction reaches a row softmax only through shift-preserving steps
  (:func:`_dynamic_contractions.feeds_row_softmax`), its second operand ``B[..., H, K, N]`` has the
  calibrated mean ``c[h, k]`` over the softmax axis subtracted, which moves every score row by a constant
  the softmax ignores. When ``B`` is a Linear's output seen only through layout steps (its sole reader),
  the shift is folded into that Linear's bias (``b - c`` on the matching features); otherwise it is a host
  subtraction on the operand;
* fixed planes: a softmax-produced operand ``P`` is replaced by ``e = exp(s - rowmax(s))`` of the
  softmax's own input (row maximum 1) split into k int8 planes with FIXED scales ``1 / (q * (q + 1)**j)``,
  ``q`` the recipe's symmetric int8 bound; the host divides the result rows by ``sum(e)`` after the
  contraction.

Statistics come from one pass over the calibration inputs on the float graph, before PT2E: per head the
absmax of each scaled operand, and per (head, row) of a smoothed operand its min, max, sum and count over
the other axes. The minmax of ``B - c`` per head is then ``max_k max(|min - c|, |max - c|)``, exact.
Nothing here imports merlin, names a model or a target.
"""

from __future__ import annotations

import math
from typing import Any

try:  # the capture venv imports the siblings by bare name; merlin's own tooling by package
    import _dynamic_contractions as DC
except ImportError:
    from merlin.targetgen import _dynamic_contractions as DC

#: Operations that move elements without changing them (a fold through them is exact).
_LAYOUT_NAMES = ("transpose", "permute", "view", "reshape", "_unsafe_view", "contiguous", "clone", "expand", "t")
#: Created operations that move no element (their cost is not host arithmetic).
_NO_COST = ("reshape", "transpose", "_int_mm", "view", "select", "stack", "t")
_RECORDS: dict[int, dict] = {}


def _record(run: int, key: str, value: Any, smooth: bool) -> None:
    """Calibration hook inserted into the graph: per-head (and per head-row) statistics of ``value``."""
    import torch

    stats = _RECORDS[run].setdefault(key, {})
    x = value.detach().to(torch.float64)
    head = x.dim() - 3 if x.dim() >= 3 else None
    others = [d for d in range(x.dim()) if d != head]
    absmax = x.abs().amax(dim=others) if head is not None else x.abs().amax().reshape(1)
    stats["absmax"] = absmax if "absmax" not in stats else torch.maximum(stats["absmax"], absmax)
    if smooth:
        rows = [d for d in range(x.dim()) if d not in (head, x.dim() - 2)]
        low, high = x.amin(dim=rows), x.amax(dim=rows)
        if head is None:
            low, high = low.reshape(1, -1), high.reshape(1, -1)
        total = x.sum(dim=rows).reshape(low.shape)
        count = math.prod(x.shape[d] for d in rows)
        if "low" in stats:
            low, high = torch.minimum(stats["low"], low), torch.maximum(stats["high"], high)
            total, count = stats["sum"] + total, stats["count"] + count
        stats.update(low=low, high=high, sum=total, count=count)


def plan(node: Any, block: dict) -> dict:
    """What a site needs: which operands take static scales, the smooth shift, the probability planes."""
    lhs, rhs = node.args[:2]
    if DC.softmax_produced(rhs):
        raise ValueError("a softmax-produced second operand has no static form")
    context = DC.softmax_produced(lhs)
    smooth = not context and block.get("smooth") == DC.SOFTMAX_AXIS and DC.feeds_row_softmax(node)
    planes = int((block.get("planes") or {}).get(DC.SOFTMAX_OUTPUT, 1)) if context else 1
    return {"context": context, "smooth": smooth, "planes": planes}


def calibrate(graph_module: Any, plans: dict, samples: list) -> dict:
    """Run ``samples`` through the float graph with statistics hooks on every planned operand."""
    import torch

    run = len(_RECORDS) + 1
    _RECORDS[run] = {}
    graph = graph_module.graph
    hooks = []
    for node, site in plans.items():
        operands = [("rhs", node.args[1], site["smooth"])]
        if not site["context"]:
            operands.insert(0, ("lhs", node.args[0], False))
        for role, value, smooth in operands:
            with graph.inserting_before(node):
                hooks.append(graph.call_function(_record, (run, f"{node.name}:{role}", value, smooth)))
    graph_module.recompile()
    try:
        with torch.no_grad():
            for sample in samples:
                graph_module(*(sample if isinstance(sample, (tuple, list)) else (sample,)))
    finally:
        for hook in hooks:
            graph.erase_node(hook)
        graph_module.recompile()
    return _RECORDS.pop(run)


def _scale(absmax: Any, low: int, high: int, eps: float) -> Any:
    return (absmax / ((high - low) / 2.0)).clamp(min=eps).to(absmax.dtype)


def _shaped(vector: Any, rank: int, trailing: int) -> Any:
    """A per-head vector ``[H]`` (or ``[H, R]``) laid out to broadcast over an operand of ``rank``."""
    import torch

    tensor = vector.to(torch.float32)
    if rank < 3:  # no head axis: one scale, or one shift per row
        body = list(vector.shape[1:]) if vector.dim() == 2 else []
        return tensor.reshape(body + [1] * trailing) if body else tensor.reshape([1] * rank)
    return tensor.reshape([1] * (rank - 3) + list(vector.shape) + [1] * trailing)


def _fold_chain(value: Any) -> tuple[Any, list] | None:
    """The Linear whose output ``value`` is, through single-reader layout steps only; else None."""
    import torch

    chain = []
    while (
        value.op == "call_function"
        and getattr(value.target, "__name__", "").split(".")[0] in _LAYOUT_NAMES
        and len(value.users) == 1
    ):
        if any(isinstance(a, torch.fx.Node) for a in value.args[1:]) or value.kwargs:
            return None
        chain.append(value)
        value = value.args[0]
    if value.op != "call_function" or value.target != torch.ops.aten.linear.default or len(value.users) != 1:
        return None
    bias = value.args[2] if len(value.args) > 2 else value.kwargs.get("bias")
    if getattr(value.args[1], "op", None) != "get_attr" or (bias is not None and bias.op != "get_attr"):
        return None
    return value, list(reversed(chain))


def fold_shift(graph_module: Any, operand: Any, shift: Any) -> bool:
    """Subtract ``shift[h, k]`` from ``operand`` by folding it into the bias of the Linear producing it.

    The feature each ``(h, k)`` reads is found by pushing the Linear's feature index through the same
    layout steps; every other position of that ``(h, k)`` must read the same feature, and features read by
    several ``(h, k)`` (grouped heads) must take one shift. Returns False, changing nothing, otherwise.
    """
    import torch

    found = _fold_chain(operand)
    if found is None:
        return False
    linear, chain = found
    shape = list(linear.meta["val"].shape)
    index = torch.arange(shape[-1], dtype=torch.float64).expand(shape)
    for step in chain:
        index = step.target(index, *step.args[1:])
    rank = index.dim()
    if rank < 2 or list(index.shape) != list(operand.meta["val"].shape):
        return False
    head = rank - 3 if rank >= 3 else None
    keep = [d for d in (head, rank - 2) if d is not None]
    order = keep + [d for d in range(rank) if d not in keep]
    moved = index.permute(*order).reshape([index.shape[d] for d in keep] + [-1])
    if not bool((moved == moved[..., :1]).all()):
        return False
    features = moved[..., 0].reshape(shift.shape).to(torch.int64)
    bias_node = linear.args[2] if len(linear.args) > 2 else linear.kwargs.get("bias")
    weight = graph_module
    for part in str(linear.args[1].target).split("."):
        weight = getattr(weight, part)
    if bias_node is None:
        bias = torch.zeros(shape[-1], dtype=weight.dtype)
    else:
        bias = graph_module
        for part in str(bias_node.target).split("."):
            bias = getattr(bias, part)
    delta = torch.full((shape[-1],), float("nan"), dtype=torch.float64)
    for feature, value in zip(features.flatten().tolist(), shift.flatten().tolist(), strict=True):
        if not math.isnan(delta[feature]) and abs(delta[feature] - value) > 1e-6 * max(1.0, abs(value)):
            return False
        delta[feature] = value
    delta = torch.nan_to_num(delta, nan=0.0)
    emit = DC.Emitter(graph_module, linear, transform="static_attention_shift")
    folded = emit.constant(f"_static_shift_{linear.name}", (bias.detach().to(torch.float64) - delta).to(bias.dtype))
    args = list(linear.args)
    if len(args) > 2:
        args[2] = folded
        linear.args = tuple(args)
    else:
        linear.kwargs = {**linear.kwargs, "bias": folded}
    return True


def _cost(created: list) -> int:
    return sum(
        math.prod(node.meta["val"].shape)
        for node in created
        if getattr(node.target, "__name__", str(node.target)).split(".")[0] not in _NO_COST
    )


def _quantize(emit: Any, value: Any, scale: Any, qmax: int) -> Any:
    torch, aten = emit.torch, DC._aten()
    grid = emit(aten.clamp.default, emit(aten.round.default, emit(aten.div.Tensor, value, scale)), -qmax, qmax)
    return emit(aten._to_copy.default, grid, dtype=torch.int8)


def rewrite(graph_module: Any, node: Any, site: dict, stats: dict, *, low: int, high: int, eps: float) -> dict:
    """Replace one activation contraction by its static per-head integer form; return its row."""
    import torch

    aten = DC._aten()
    qmax = min(-low, high)
    lhs, rhs = node.args[:2]
    record_rhs = stats[f"{node.name}:rhs"]
    folded = False
    shift = None
    if site["smooth"]:
        shift = record_rhs["sum"] / record_rhs["count"]
        record_rhs = dict(
            record_rhs,
            absmax=torch.maximum((record_rhs["low"] - shift).abs(), (record_rhs["high"] - shift).abs()).amax(-1),
        )
        folded = fold_shift(graph_module, rhs, shift)
    emit = DC.Emitter(graph_module, node, transform="static_attention")
    census = DC._census()
    b = DC._f32(emit, rhs)
    rank_b = b.meta["val"].dim()
    if shift is not None and not folded:
        b = emit(aten.sub.Tensor, b, emit.constant(f"_static_mean_{node.name}", _shaped(shift, rank_b, 1)))
    rhs_scale = _scale(record_rhs["absmax"], low, high, eps)
    b_scale = emit.constant(f"_static_scale_{node.name}_rhs", _shaped(rhs_scale, rank_b, 2))
    qb = _quantize(emit, b, b_scale, qmax)
    if not site["context"]:
        a = DC._f32(emit, lhs)
        lhs_scale = _scale(stats[f"{node.name}:lhs"]["absmax"], low, high, eps)
        rank_a = a.meta["val"].dim()
        a_scale = emit.constant(f"_static_scale_{node.name}_lhs", _shaped(lhs_scale, rank_a, 2))
        planes = [(_quantize(emit, a, a_scale, qmax), _shaped(lhs_scale * rhs_scale, max(rank_a, rank_b), 2))]
        total = None
    else:
        softmax = lhs
        while softmax.target not in (aten._softmax.default, aten.softmax.int):
            softmax = softmax.args[2] if softmax.target == aten.where.ScalarSelf else softmax.args[0]
        scores = DC._f32(emit, softmax.args[0])
        if not DC._last_axis(softmax.args[1], scores.meta["val"].dim()):
            raise ValueError("the softmax is not over the last axis")
        peak = emit(aten.amax.default, scores, [-1], True)
        peak = emit(aten.where.ScalarSelf, emit(aten.eq.Scalar, peak, float("-inf")), 0.0, peak)
        e = emit(aten.exp.default, emit(aten.sub.Tensor, scores, peak))
        total = emit(aten.clamp.default, emit(aten.sum.dim_IntList, e, [-1], True), DC.SCALE_FLOOR)
        planes, residual = [], e
        for index in range(site["planes"]):
            steps = float(qmax * (qmax + 1) ** index)  # plane j has the fixed scale 1 / steps
            grid = emit(
                aten.clamp.default, emit(aten.round.default, emit(aten.mul.Tensor, residual, steps)), -qmax, qmax
            )
            planes.append((emit(aten._to_copy.default, grid, dtype=torch.int8), _shaped(rhs_scale / steps, rank_b, 2)))
            if index + 1 < site["planes"]:
                residual = emit(aten.sub.Tensor, residual, emit(aten.div.Tensor, grid, steps))
    result = None
    for plane, scale in planes:
        acc = DC._int_matmul(emit, plane, qb)
        out = acc.meta["val"]
        census["int8_macs"] += math.prod(out.shape) * int(plane.meta["val"].shape[-1])
        census["device_calls"] += math.prod(out.shape[:-2]) if out.dim() > 2 else 1
        term = emit(
            aten.mul.Tensor,
            emit(aten.to.dtype, acc, torch.float32),
            emit.constant(f"_static_scale_{node.name}_out", scale),
        )
        result = term if result is None else emit(aten.add.Tensor, result, term)
    if total is not None:
        result = emit(aten.div.Tensor, result, total)
    census["host_elements"] = _cost(emit.created)
    row = DC._finish(emit, node, result, census, smooth=bool(site["smooth"]), planes=(site["planes"], 1))
    return {**row, "scales": "static_heads", "shift_folded": folded}


def rewrite_sites(
    graph_module: Any, sites: list, block: dict, samples: Any, *, low: int, high: int, eps: float, owner: Any
) -> list[dict]:
    """Plan, calibrate (one pass over ``samples``) and rewrite every site; a site with no static form is kept."""
    rows, plans = [], {}
    for node in sites:
        try:
            plans[node] = plan(node, block)
        except ValueError as why:
            rows.append({"node": node.name, "owner": owner(node), "status": "kept", "why": str(why)})
    stats = calibrate(graph_module, plans, list(samples)) if plans else {}
    for node, site in plans.items():
        name = owner(node)
        row = rewrite(graph_module, node, site, stats, low=low, high=high, eps=eps)
        rows.append({"family": "activation_contraction", "owner": name, **row, "status": "rewritten"})
    return rows
