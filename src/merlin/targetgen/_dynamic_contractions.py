#!/usr/bin/env python3
"""Integer contractions with run-time scales on non-reduction axes -- runs INSIDE the capture venv.

A recipe's ``dynamic`` block (:func:`merlin.targetgen.quant_recipe.with_dynamic_contractions`) selects
contraction families whose operands are quantized at RUN TIME rather than with a calibrated static scale.
This module rewrites each selected contraction of an exported graph into its integer form, written in
ordinary ATen operations the capture already lowers. Three general mechanisms, none named after a layer:

(A) dynamic scales: for ``C = A[..., M, K] @ B[..., K, N]`` the host takes one absmax per row of ``A``
    and one per column of ``B`` (both over the reduced ``K``), quantizes each operand to symmetric
    int8, the device runs a plain ``int8 x int8 -> i32`` matmul per batch slice, and the host rescales
    the exact accumulator by the outer product of the two scale vectors. A stored weight (a Linear's)
    is quantized once, here, with one scale per output channel. Every scale sits on an axis the
    contraction does not reduce, so no scale is applied inside a sum.
(B) planes: an operand is split into k int8 planes ``X ~ sum_p P_p * s_p``, each the quantized residual
    of the ones before with its own scales; that is k device matmuls against the same other operand,
    summed on the host. Used where one plane loses the operand's dynamic range (softmax probabilities).
(C) smooth shift: when ``C`` reaches a softmax over its column axis only through scalar factors, masks
    and casts, ``B``'s mean over that axis is subtracted first. Each score row moves by a constant,
    which the softmax ignores, so the function is unchanged while ``B``'s range shrinks.

The integer form is the PROGRAM: the capture's golden and the compiler's INT8 reference both evaluate
these operations, so the device contraction is exact by construction and only the operand rounding is
an approximation. Nothing here imports merlin, names a model or a target; torch is imported lazily.
"""

from __future__ import annotations

import math
from typing import Any

#: The recipe's spelling of the plane selectors and the smooth rule (``quant_recipe``).
SOFTMAX_OUTPUT = "softmax_output"
WEIGHT = "weight"
SOFTMAX_AXIS = "softmax_axis"
#: A scale is never zero: a row of zeros quantizes to zeros under the smallest normal f32 scale.
SCALE_FLOOR = 1.1754943508222875e-38
#: Metadata the emitted operations inherit from the contraction they replace.
_INHERITED_META = ("nn_module_stack", "source_fn_stack", "stack_trace", "from_node")
#: A rewritten site's independent check (``sites[i]["verification"]``): its emitted int8 result against
#: the float contraction of its own operands, on the recipe's agreement samples. Recorded, never judged.
VERIFICATION_SCHEMA = "m2m.dynamic_site_verification.v1"
#: The in-process row key that carries a site's verification spec (FX nodes) until the check runs.
VERIFY_KEY = "_verify"
_CHECKS: dict[int, dict] = {}


def qmax_of(recipe: dict) -> int:
    """The symmetric int8 bound the recipe's activation range allows on both signs."""
    activation = recipe.get("activation") or {}
    low, high = activation.get("quant_min"), activation.get("quant_max")
    if not isinstance(low, int) or not isinstance(high, int) or low >= 0 or high <= 0:
        raise ValueError("a dynamic contraction needs the recipe's signed integer activation range")
    return min(-low, high)


class Emitter:
    """Insert ATen calls before ``anchor`` with observed values and the anchor's ownership and lineage."""

    def __init__(
        self, graph_module: Any, anchor: Any, *, mode: Any = None, transform: str = "dynamic_contraction"
    ) -> None:
        import torch
        from torch._guards import detect_fake_mode

        self.torch = torch
        self.gm, self.graph, self.anchor = graph_module, graph_module.graph, anchor
        vals = [a.meta.get("val") for a in anchor.all_input_nodes if isinstance(a.meta.get("val"), torch.Tensor)]
        self.mode = mode if mode is not None else detect_fake_mode(vals)
        self.transform = transform
        custom = dict(anchor.meta.get("custom") or {})
        node_id = [custom["m2m_node_id"]] if "m2m_node_id" in custom else []
        self.lineage = sorted({*(custom.get("m2m_lineage") or ()), *node_id})
        self.created: list[Any] = []

    def __call__(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        torch = self.torch
        with self.graph.inserting_before(self.anchor):
            node = self.graph.call_function(target, args, kwargs)
        node.meta.update({k: self.anchor.meta[k] for k in _INHERITED_META if k in self.anchor.meta})
        if self.lineage:
            node.meta["custom"] = {"m2m_lineage": list(self.lineage), "m2m_transform": self.transform}

        def observed(value: Any) -> Any:
            if isinstance(value, torch.fx.Node):
                return value.meta["val"]
            if isinstance(value, (list, tuple)):
                return type(value)(observed(v) for v in value)
            return value

        with self.mode:
            node.meta["val"] = target(*observed(args), **{k: observed(v) for k, v in kwargs.items()})
        self.created.append(node)
        return node

    def constant(self, name: str, tensor: Any) -> Any:
        """A stored tensor computed here (a quantized weight), registered on the module."""
        attr = name
        index = 0
        while hasattr(self.gm, attr):
            index += 1
            attr = f"{name}_{index}"
        self.gm.register_buffer(attr, tensor)
        with self.graph.inserting_before(self.anchor):
            node = self.graph.get_attr(attr)
        with self.mode:
            node.meta["val"] = self.mode.from_tensor(tensor)
        node.meta.update({k: self.anchor.meta[k] for k in _INHERITED_META if k in self.anchor.meta})
        return node


# --------------------------------------------------------------------------------- graph patterns


def _aten():
    import torch

    return torch.ops.aten


def _last_axis(dim: Any, rank: int) -> bool:
    dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
    return len(dims) == 1 and isinstance(dims[0], int) and dims[0] % rank == rank - 1


def _depends_on(value: Any, root: Any, limit: int = 64) -> bool:
    """Whether ``value`` is computed from ``root`` (a bounded backward walk)."""
    import torch

    pending, seen = [value], set()
    while pending and len(seen) < limit:
        node = pending.pop()
        if node is root:
            return True
        if not isinstance(node, torch.fx.Node) or node in seen:
            continue
        seen.add(node)
        pending.extend(node.all_input_nodes)
    return len(seen) >= limit  # undecided counts as dependent: never claim a shift is harmless


def feeds_row_softmax(node: Any) -> bool:
    """Whether every use of ``node`` reaches a softmax over its LAST axis through shift-preserving steps.

    Allowed steps: a multiply or divide by a number, a dtype cast, an added tensor (a mask) that does not
    depend on ``node``, a fill or a select by a mask, and the empty-row test ``amax(.) == -inf``. Under
    those steps a per-row constant added to ``node`` reaches the softmax as a per-row constant.
    """
    import torch

    aten = _aten()
    rank = node.meta["val"].dim()
    casts = {aten.to.dtype, aten._to_copy.default, aten.clone.default, aten.contiguous.default}
    scalar_steps = {aten.mul.Tensor, aten.div.Tensor, aten.mul_.Tensor, aten.div_.Tensor}
    softmaxes = {aten._softmax.default, aten.softmax.int}
    checks = {aten._assert_tensor_metadata.default}  # a metadata assertion reads no value
    pending, reached = [node], False
    while pending:
        value = pending.pop()
        users = list(value.users)
        if not users:
            return False
        for user in users:
            args = user.args
            if user.target in softmaxes and args[0] is value:
                if not _last_axis(args[1], rank):
                    return False
                reached = True
            elif user.target in checks:
                continue
            elif user.target in casts and args[0] is value:
                pending.append(user)
            elif user.target in scalar_steps and args[0] is value:
                if isinstance(args[1], torch.fx.Node):
                    return False
                pending.append(user)
            elif user.target == aten.mul.Tensor and args[1] is value and not isinstance(args[0], torch.fx.Node):
                pending.append(user)
            elif user.target == aten.add.Tensor and value in args[:2]:
                other = args[1] if args[0] is value else args[0]
                if isinstance(other, torch.fx.Node) and _depends_on(other, node):
                    return False
                pending.append(user)
            elif user.target == aten.masked_fill.Scalar and args[0] is value:
                pending.append(user)
            elif user.target in (aten.where.self, aten.where.ScalarOther) and args[1] is value:
                if len(args) > 2 and isinstance(args[2], torch.fx.Node) and _depends_on(args[2], node):
                    return False
                pending.append(user)
            elif user.target == aten.amax.default and args[0] is value and _last_axis(args[1], rank):
                tests = list(user.users)
                if not tests or any(
                    t.target != aten.eq.Scalar or not (isinstance(t.args[1], float) and t.args[1] == -math.inf)
                    for t in tests
                ):
                    return False
            else:
                return False
    return reached


def softmax_produced(value: Any) -> bool:
    """Whether ``value`` is a softmax's output, through casts and the empty-row guard."""
    import torch

    aten = _aten()
    for _ in range(8):
        if not isinstance(value, torch.fx.Node) or value.op != "call_function":
            return False
        if value.target in (aten._softmax.default, aten.softmax.int):
            return True
        if value.target in (aten.to.dtype, aten._to_copy.default, aten.clone.default, aten.contiguous.default):
            value = value.args[0]
        elif value.target == aten.where.ScalarSelf:
            value = value.args[2]
        else:
            return False
    return False


# ------------------------------------------------------------------------------------ the rewrite


def _f32(emit: Emitter, value: Any) -> Any:
    torch = emit.torch
    return value if value.meta["val"].dtype == torch.float32 else emit(_aten().to.dtype, value, torch.float32)


def _runtime_planes(emit: Emitter, value: Any, *, axis: int, planes: int, qmax: int, census: dict) -> list:
    """``[(int8 plane, f32 scale)]`` of a run-time f32 operand, one scale per slice along ``axis``'s
    complement: absmax over ``axis`` (the reduced one), each plane the quantized residual of the last."""
    torch, aten = emit.torch, _aten()
    out, residual = [], value
    for index in range(planes):
        scale = emit(aten.amax.default, emit(aten.abs.default, residual), [axis], True)
        scale = emit(aten.clamp.default, emit(aten.div.Tensor, scale, float(qmax)), SCALE_FLOOR)
        grid = emit(aten.clamp.default, emit(aten.round.default, emit(aten.div.Tensor, residual, scale)), -qmax, qmax)
        plane = emit(aten._to_copy.default, grid, dtype=torch.int8)
        out.append((plane, scale))
        census["host_elements"] += 4 * residual.meta["val"].numel()  # |x|, max, divide-round-clamp, convert
        if index + 1 < planes:
            residual = emit(aten.sub.Tensor, residual, emit(aten.mul.Tensor, grid, scale))
            census["host_elements"] += 2 * residual.meta["val"].numel()
    return out


def static_planes(tensor: Any, *, axis: int, planes: int, qmax: int) -> list:
    """The same planes of a STORED tensor, computed once here (eager torch, f32)."""
    import torch

    out, residual = [], tensor.detach().to(torch.float32)
    for _ in range(planes):
        scale = (residual.abs().amax(dim=axis, keepdim=True) / float(qmax)).clamp(min=SCALE_FLOOR)
        grid = torch.clamp(torch.round(residual / scale), -qmax, qmax)
        out.append((grid.to(torch.int8), scale))
        residual = residual - grid * scale
    return out


def _int_matmul(emit: Emitter, lhs: Any, rhs: Any) -> Any:
    """``lhs[..., M, K] @ rhs[..., K, N]`` in int8 -> i32, one ``_int_mm`` per batch slice."""
    aten = _aten()
    a, b = lhs.meta["val"], rhs.meta["val"]
    batch = list(a.shape[:-2])
    if list(b.shape[:-2]) != batch:
        raise ValueError("a dynamic contraction needs equal batch extents on both operands")
    if not batch:
        return emit(aten._int_mm.default, lhs, rhs)
    count = math.prod(batch)
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    lhs3 = emit(aten.reshape.default, lhs, [count, m, k])
    rhs3 = emit(aten.reshape.default, rhs, [count, k, n])
    slices = [
        emit(aten._int_mm.default, emit(aten.select.int, lhs3, 0, i), emit(aten.select.int, rhs3, 0, i))
        for i in range(count)
    ]
    return emit(aten.reshape.default, emit(aten.stack.default, slices, 0), [*batch, m, n])


def _combine(emit: Emitter, lhs_planes: list, rhs_planes: list, census: dict) -> Any:
    """``sum_{p,q} float(P_p @ Q_q) * s_p * t_q``: device matmuls, host outer-product rescale and sum."""
    torch, aten = emit.torch, _aten()
    total = None
    for plane, scale in lhs_planes:
        for other, other_scale in rhs_planes:
            acc = _int_matmul(emit, plane, other)
            out = acc.meta["val"]
            census["int8_macs"] += math.prod(out.shape) * int(plane.meta["val"].shape[-1])
            census["device_calls"] += math.prod(out.shape[:-2]) if out.dim() > 2 else 1
            term = emit(
                aten.mul.Tensor, emit(aten.mul.Tensor, emit(aten.to.dtype, acc, torch.float32), scale), other_scale
            )
            census["host_elements"] += 3 * out.numel()  # convert, row scale, column scale
            if total is not None:
                census["host_elements"] += out.numel()
            total = term if total is None else emit(aten.add.Tensor, total, term)
    return total


def _census() -> dict:
    return {"int8_macs": 0, "device_calls": 0, "host_elements": 0}


def rewrite_activation_contraction(graph_module: Any, node: Any, block: dict, qmax: int) -> dict:
    """Replace ``matmul(A, B)`` (two run-time operands) by its dynamic integer form; return its row."""
    aten = _aten()
    emit = Emitter(graph_module, node)
    census = _census()
    lhs, rhs = node.args[:2]
    smooth = block.get("smooth") == SOFTMAX_AXIS and feeds_row_softmax(node)
    planes = dict(block.get("planes") or {})
    lhs_planes = planes.get(SOFTMAX_OUTPUT, 1) if softmax_produced(lhs) else 1
    rhs_planes = planes.get(SOFTMAX_OUTPUT, 1) if softmax_produced(rhs) else 1
    a, b = _f32(emit, lhs), _f32(emit, rhs)
    if smooth:
        b = emit(aten.sub.Tensor, b, emit(aten.mean.dim, b, [-1], True))
        census["host_elements"] += 2 * b.meta["val"].numel()
    result = _combine(
        emit,
        _runtime_planes(emit, a, axis=-1, planes=lhs_planes, qmax=qmax, census=census),
        _runtime_planes(emit, b, axis=-2, planes=rhs_planes, qmax=qmax, census=census),
        census,
    )
    row = _finish(emit, node, result, census, smooth=smooth, planes=(lhs_planes, rhs_planes))
    row[VERIFY_KEY] = verification_spec(emit, (lhs, rhs), smooth=smooth)
    return row


def rewrite_weight_contraction(graph_module: Any, node: Any, weight: Any, block: dict, qmax: int) -> dict:
    """Replace ``linear(x, W[, b])`` by run-time row scales on ``x`` and a stored per-channel ``W``."""
    aten = _aten()
    emit = Emitter(graph_module, node)
    census = _census()
    x = node.args[0]
    bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
    planes = dict(block.get("planes") or {})
    shape = list(x.meta["val"].shape)
    rows = emit(aten.reshape.default, _f32(emit, x), [-1, shape[-1]])
    # One scale per output channel of W (its rows), the planes of W^T laid out [K, N] for the device.
    stored = []
    for index, (plane, scale) in enumerate(static_planes(weight, axis=1, planes=planes.get(WEIGHT, 1), qmax=qmax)):
        stored.append(
            (
                emit.constant(f"_dyn_w_{node.name}_{index}", plane.t().contiguous()),
                emit.constant(f"_dyn_s_{node.name}_{index}", scale.reshape(1, -1).contiguous()),
            )
        )
    result = _combine(emit, _runtime_planes(emit, rows, axis=-1, planes=1, qmax=qmax, census=census), stored, census)
    if bias is not None:
        result = emit(aten.add.Tensor, result, _f32(emit, bias))
    result = emit(aten.reshape.default, result, [*shape[:-1], int(weight.shape[0])])
    row = _finish(emit, node, result, census, smooth=False, planes=(1, len(stored)))
    reference = weight.detach().to(emit.torch.float32).clone()
    row[VERIFY_KEY] = verification_spec(emit, (x,) if bias is None else (x, bias), kind="linear", weight=reference)
    return row


def _finish(emit: Emitter, node: Any, result: Any, census: dict, *, smooth: bool, planes: tuple) -> dict:
    expected = node.meta["val"]
    if result.meta["val"].dtype != expected.dtype:
        result = emit(_aten().to.dtype, result, expected.dtype)
    if tuple(result.meta["val"].shape) != tuple(expected.shape):
        raise ValueError(f"dynamic contraction {node.name} changed its result shape")
    row = {
        "node": node.name,
        "shapes": [list(a.meta["val"].shape) for a in node.args[:2] if hasattr(a, "meta")],
        "planes": list(planes),
        "smooth": smooth,
        # The replaced contraction's trace lineage: the issuer ties this site's device calls to it.
        "lineage": list(emit.lineage),
        **census,
    }
    node.replace_all_uses_with(result)
    emit.graph.erase_node(node)
    emit.replacement = result
    return row


def verification_spec(emit: Emitter, operands: tuple, **reference: Any) -> dict:
    """What the site check needs: the emitted result and the replaced contraction's float operands."""
    spec = {"out": emit.replacement, "operands": tuple(operands), "kind": "matmul", "shift": None, "smooth": False}
    spec.update(reference)
    return spec


def _site_metrics(got: Any, ref: Any) -> dict:
    """One sample's agreement, defined as the recipe agreement's (cosine of the flattened tensors, max |e|)."""
    import torch

    g, r = got.reshape(-1), ref.reshape(-1)
    if g.numel() != r.numel() or not g.numel():
        return {"finite": False}
    if not (bool(torch.isfinite(g).all()) and bool(torch.isfinite(r).all())):
        return {"finite": False}
    error = g - r
    norms = float(g.norm()) * float(r.norm())
    cosine = float((g * r).sum()) / norms if norms else (1.0 if not float(error.abs().max()) else 0.0)
    rms_ref = float(r.square().mean().sqrt())
    rms_err = float(error.square().mean().sqrt())
    if not rms_ref and rms_err:
        return {"finite": False}
    return {
        "finite": True,
        "cosine": cosine,
        "max_abs": float(error.abs().max()),
        "rel_rmse": rms_err / rms_ref if rms_ref else 0.0,
    }


def _check(run: int, key: str, out: Any, *operands: Any) -> None:
    """Site-check hook: the float reference of the replaced contraction, from the same operands."""
    import torch

    spec = _CHECKS[run]["specs"][key]
    values = [value.detach().to(torch.float64) for value in operands]
    if spec["kind"] == "linear":
        bias = values[1] if len(values) > 1 else None
        reference = torch.nn.functional.linear(values[0], spec["weight"].to(torch.float64), bias)
    else:
        a, b = values
        if spec["shift"] is not None:
            b = b - spec["shift"].to(torch.float64)
        if spec["smooth"]:
            b = b - b.mean(dim=-1, keepdim=True)
        reference = torch.matmul(a, b)
    reference = reference.to(torch.float32).to(torch.float64)
    _CHECKS[run]["rows"].setdefault(key, []).append(_site_metrics(out.detach().to(torch.float64), reference))


def _summary(samples: list, spec: dict) -> dict:
    finite = bool(samples) and all(row["finite"] for row in samples)
    shifted = spec["shift"] is not None or spec["smooth"]
    record = {
        "schema": VERIFICATION_SCHEMA,
        "samples": len(samples),
        "finite": finite,
        "reference": f"float64_{spec['kind']}_of_site_operands_rounded_f32" + ("_shift_adjusted" if shifted else ""),
        "cosine_min": None,
        "cosine_mean": None,
        "max_abs_error": None,
        "rel_rmse_max": None,
    }
    if finite:
        record.update(
            cosine_min=min(row["cosine"] for row in samples),
            cosine_mean=sum(row["cosine"] for row in samples) / len(samples),
            max_abs_error=max(row["max_abs"] for row in samples),
            rel_rmse_max=max(row["rel_rmse"] for row in samples),
        )
    return record


def verify_sites(graph_module: Any, rows: list, samples: list) -> None:
    """Check every rewritten site of ``rows`` on ``samples`` in one pass; write ``verification`` per row.

    Each site's emitted int8 result is compared with the float contraction (f64, rounded to f32) of the
    very operands the rewritten graph gives it, so the record measures the site's own integer form, not
    any upstream difference. A smooth-shifted site is compared with the shifted contraction it is meant
    to compute (the shift moves score rows by constants the softmax ignores). The pass leaves the graph
    as it found it; the specs (FX nodes) are removed from the rows.
    """
    import torch

    try:
        from _capture_integer_mm import exact_integer_mm
    except ImportError:
        from merlin.targetgen._capture_integer_mm import exact_integer_mm

    specs = {index: row.pop(VERIFY_KEY) for index, row in enumerate(rows) if VERIFY_KEY in row}
    if not specs:
        return
    run = max(_CHECKS, default=0) + 1
    _CHECKS[run] = {"specs": {}, "rows": {}}
    graph, hooks = graph_module.graph, []
    for index, spec in specs.items():
        key = f"site{index}"
        _CHECKS[run]["specs"][key] = spec
        with graph.inserting_after(spec["out"]):
            hooks.append(graph.call_function(_check, (run, key, spec["out"], *spec["operands"])))
    graph_module.recompile()
    try:
        with torch.no_grad(), exact_integer_mm(torch):
            for sample in samples:
                graph_module(*(sample if isinstance(sample, (tuple, list)) else (sample,)))
    finally:
        for hook in hooks:
            graph.erase_node(hook)
        graph_module.recompile()
        checked = _CHECKS.pop(run)
    for index, spec in specs.items():
        rows[index]["verification"] = _summary(checked["rows"].get(f"site{index}", []), spec)
