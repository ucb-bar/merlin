"""Existing graph arithmetic, attention spelling and static-plane preparation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .portable import helper

_specs = helper("_recipe_specs")
RecipeError = _specs.RecipeError
_spec = _specs._spec
_recipe_operators = _specs._recipe_operators

_SDPA_FIELDS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
_INHERITED_META = ("nn_module_stack", "source_fn_stack", "stack_trace", "from_node")


def _bias_onto_accumulator_grid(graph_module: Any) -> dict[str, Any]:
    """Round the quantized graph's float biases onto its accumulator grid without mutating source tensors.

    For ``y = conv(dq(x; s_x), dq(W; s_w)) + b`` the accumulator step of output channel ``c`` is
    ``S_c = s_x * s_w[c]`` (f32). The bias becomes ``k_c * S_c`` with ``k_c = round(b_c / S_c)``
    (half to even), an i32 the output path can preload. A bias whose ``k`` leaves i32 is refused.
    Returns the census: sites, channels and the largest change in units of ``S``.
    """
    import torch

    def attr(node: Any) -> Any:
        value = graph_module
        for part in str(node.target).split("."):
            value = getattr(value, part)
        return value

    def named(node: Any, name: str) -> bool:
        return getattr(node, "op", None) == "call_function" and name in str(node.target)

    sites = channels = 0
    worst = 0.0
    contractions = (torch.ops.aten.conv2d.default, torch.ops.aten.linear.default)
    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target not in contractions or len(node.args) < 3:
            continue
        x, w, bias = node.args[:3]
        if getattr(bias, "op", None) != "get_attr" or not named(x, "dequantize_per_tensor"):
            continue
        if not named(w, "dequantize_per_channel") and not named(w, "dequantize_per_tensor"):
            continue
        s_x = x.args[1]
        if isinstance(s_x, torch.fx.Node):
            s_x = attr(s_x)
        s_x = torch.as_tensor(s_x, dtype=torch.float32).reshape(())
        s_w = w.args[1]
        s_w = torch.as_tensor(attr(s_w) if isinstance(s_w, torch.fx.Node) else s_w, dtype=torch.float32).reshape(-1)
        stored = attr(bias)
        step = (s_x * s_w).expand(stored.shape[0]) if s_w.numel() == 1 else s_x * s_w
        if step.shape != stored.shape or not bool(torch.all(step > 0)):
            raise RecipeError(f"bias of {node.name} has no positive per-channel accumulator step")
        codes = torch.round(stored.to(torch.float64) / step.to(torch.float64))
        if bool(torch.any(codes.abs() > 2**31 - 1)):
            raise RecipeError(f"bias of {node.name} leaves the i32 accumulator")
        # The bias is the rescale of the integer code itself, in the integer rewrite's own order
        # (f32(k) * s_x, then * s_w), so it is exactly the value a preloaded k would contribute.
        per_channel = s_w.expand(stored.shape[0]) if s_w.numel() == 1 else s_w
        snapped = (codes.to(torch.float32) * s_x) * per_channel
        worst = max(worst, float(((snapped.double() - stored.double()) / step.double()).abs().max()))
        owned = snapped.to(dtype=stored.dtype, device=stored.device).detach()
        owned.requires_grad_(stored.requires_grad)
        if isinstance(stored, torch.nn.Parameter):
            owned = torch.nn.Parameter(owned, requires_grad=stored.requires_grad)
        parent = graph_module
        *owners, leaf = str(bias.target).split(".")
        for name in owners:
            parent = getattr(parent, name)
        setattr(parent, leaf, owned)
        sites += 1
        channels += int(stored.numel())
    return {
        "sites": sites,
        "channels": channels,
        "max_change_in_steps": worst,
        "rule": "k = round_half_even(b / (s_x * s_w)); b := (f32(k) * s_x) * s_w, the integer rewrite's rescale order",
    }


def spell_attention(graph_module: Any) -> list[dict[str, Any]]:
    """Rewrite each fused attention call as its math definition, two matmuls around a softmax, in place.

    ``scaled_dot_product_attention`` hides both of its contractions from an operator annotation, so
    a recipe that quantizes activation x activation contractions would see none. The rewrite is the
    math reference the capture itself lowers the fused call to: half-precision operands are computed
    in f32 and the result cast back; ``Q @ K^T`` times the scale; a floating mask is added and a
    boolean mask fills the excluded scores with ``-inf``; a softmax over the key axis, where a row
    with no admitted key is zero (SDPA's rule; a plain softmax would give NaN); and ``P @ V``. A call
    outside that definition is LEFT FUSED and reported, never approximated: dropout, a causal flag
    and a grouped-query broadcast. Returns one row per call: ``{"node", "status", "why"}`` with
    status ``spelled`` or ``kept``.
    """
    import math

    import torch
    from torch._guards import detect_fake_mode

    aten = torch.ops.aten
    half = (torch.float16, torch.bfloat16)
    graph = graph_module.graph
    rows: list[dict[str, Any]] = []
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != aten.scaled_dot_product_attention.default:
            continue
        bound = dict(zip(_SDPA_FIELDS, node.args))
        bound.update(node.kwargs)
        query, key, value, mask = bound["query"], bound["key"], bound["value"], bound.get("attn_mask")
        vals = [x.meta.get("val") if isinstance(x, torch.fx.Node) else None for x in (query, key, value, mask)]
        why = None
        if float(bound.get("dropout_p") or 0.0) != 0.0:
            why = "dropout"
        elif bound.get("is_causal"):
            why = "causal flag"
        elif bound.get("enable_gqa"):
            why = "grouped-query broadcast"
        elif not all(isinstance(v, torch.Tensor) and v.is_floating_point() and v.dim() >= 2 for v in vals[:3]):
            why = "operands are not observed floating tensors of rank >= 2"
        elif mask is not None and not isinstance(vals[3], torch.Tensor):
            why = "mask is not an observed tensor"
        if why is not None:
            rows.append({"node": node.name, "status": "kept", "why": why})
            continue
        scale = bound.get("scale")
        scale = float(scale) if scale is not None else 1.0 / math.sqrt(int(vals[0].shape[-1]))
        mode = detect_fake_mode([v for v in vals if v is not None])
        custom = dict(node.meta.get("custom") or {})
        lineage = sorted(
            {*(custom.get("m2m_lineage") or ()), *([custom["m2m_node_id"]] if "m2m_node_id" in custom else [])}
        )

        def emit(target, *args, _node=node, _mode=mode, _lineage=lineage):
            created = graph.call_function(target, args)
            created.meta.update({k: _node.meta[k] for k in _INHERITED_META if k in _node.meta})
            if _lineage:
                created.meta["custom"] = {
                    "m2m_lineage": list(_lineage),
                    "m2m_transform": "attention_spelled_as_matmuls",
                }
            observed = [a.meta["val"] if isinstance(a, torch.fx.Node) else a for a in args]
            with _mode:
                created.meta["val"] = target(*observed)
            return created

        out_dtype = vals[0].dtype
        boolean = mask is not None and vals[3].dtype == torch.bool
        with graph.inserting_before(node):
            if out_dtype in half:
                query, key, value = (emit(aten.to.dtype, x, torch.float32) for x in (query, key, value))
                if mask is not None and not boolean and vals[3].dtype != torch.float32:
                    mask = emit(aten.to.dtype, mask, torch.float32)
            scores = emit(aten.matmul.default, query, emit(aten.transpose.int, key, -2, -1))
            scores = emit(aten.mul.Tensor, scores, scale)
            if boolean:
                scores = emit(aten.masked_fill.Scalar, scores, emit(aten.logical_not.default, mask), float("-inf"))
            elif mask is not None:
                scores = emit(aten.add.Tensor, scores, mask)
            probs = emit(aten._softmax.default, scores, -1, False)
            if boolean:
                empty = emit(aten.eq.Scalar, emit(aten.amax.default, scores, [-1], True), float("-inf"))
                probs = emit(aten.where.ScalarSelf, empty, 0.0, probs)
            result = emit(aten.matmul.default, probs, value)
            if out_dtype in half:
                result = emit(aten.to.dtype, result, out_dtype)
        observed, expected = result.meta["val"], node.meta["val"]
        if tuple(observed.shape) != tuple(expected.shape) or observed.dtype != expected.dtype:
            raise RecipeError(f"spelled attention {node.name} changed its result type")
        node.replace_all_uses_with(result)
        graph.erase_node(node)
        rows.append({"node": node.name, "status": "spelled", "why": ""})
    if any(row["status"] == "spelled" for row in rows):
        graph.lint()
        graph_module.recompile()
    return rows


def _has_floating_recipe_work(exported: Any, recipe: Mapping[str, Any]) -> bool:
    """Whether an integral-input graph still computes a floating recipe operation inside."""
    import torch

    operators = _recipe_operators(torch)
    targets = {op for family in recipe.get("families") or () for op in operators.get(family, ())}
    return any(
        node.op == "call_function"
        and node.target in targets
        and isinstance((value := node.meta.get("val")), torch.Tensor)
        and value.is_floating_point()
        for node in exported.graph.nodes
    )


def _source_weight_geometry_matches(
    source_layers: list[dict[str, Any]], owner: str, observed_shape: list[int] | None
) -> bool:
    """A staged weight can inherit only its exact, unique source module's placement."""
    matching = [row for row in source_layers if row.get("fqn") == owner]
    return bool(
        len(matching) == 1 and isinstance(observed_shape, list) and matching[0].get("weight_shape") == observed_shape
    )


def _planes_grid(recipe: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """The ``static_planes`` block's digit count and the recipe's activation range it builds on."""
    block, activation = recipe.get("static_planes") or {}, recipe["activation"]
    return block.get("planes"), activation.get("quant_min"), activation.get("quant_max")


def _planes_spec(recipe: Mapping[str, Any], *, eps: float):
    """The activation spec of a planes site: the recipe's, on the wide grid, with the block's observer."""
    SP = helper("_static_planes")  # noqa: PLC0415 -- a sibling, by bare name

    planes, low, high = _planes_grid(recipe)
    try:
        grid_min, grid_max, _ = SP.grid(low, high, planes)
        dtype = SP.storage_dtype(grid_min, grid_max)
    except ValueError as error:
        raise RecipeError(f"static_planes: {error}") from error
    wide = {**recipe["activation"], "dtype": dtype, "quant_min": grid_min, "quant_max": grid_max}
    # The epsilon floors the STEP; on the wide grid it floors the same RANGE as on the int8 one.
    floor = eps * (high - low) / (grid_max - grid_min)
    return _spec({**wide, "observer": recipe["static_planes"].get("observer") or "minmax"}, is_weight=False, eps=floor)


def _planes_site(block: Mapping[str, Any], node: Any, owner: str | None) -> bool:
    """A Linear or ungrouped Conv2d of a planes family, owned inside the block's scope, takes the planes grid."""
    SP = helper("_static_planes")  # noqa: PLC0415 -- a sibling, by bare name
    import torch

    aten = torch.ops.aten
    groups = node.args[6] if len(node.args) > 6 else node.kwargs.get("groups", 1)
    return bool(
        "contraction" in (block.get("families") or ())
        and (node.target == aten.linear.default or (node.target == aten.conv2d.default and groups == 1))
        and SP.in_scope(owner, block.get("owners"))
    )


def _rewrite_planes(quantized: Any, recipe: Mapping[str, Any], annotated: int) -> list[dict[str, Any]]:
    """The converted graph's planes sites in their digit form: every annotated one, or an error."""
    if not recipe.get("static_planes"):
        return []
    SP = helper("_static_planes")  # noqa: PLC0415 -- a sibling, by bare name

    planes, low, high = _planes_grid(recipe)
    try:
        rows = SP.rewrite_static_planes(quantized, planes=planes, low=low, high=high)
    except ValueError as error:
        raise RecipeError(f"static_planes: {error}") from error
    if len(rows) != annotated:
        raise RecipeError(f"static_planes: {annotated} Linear(s) annotated on the planes grid, {len(rows)} rewritten")
    return [{**row, "status": "rewritten"} for row in rows]


def _epilogue_of(recipe: Mapping[str, Any], activation_spec: Any) -> Any:
    if not recipe.get("epilogue"):
        return None
    IE = helper("_integer_epilogue")  # noqa: PLC0415 -- a sibling, by bare name

    try:
        return IE.Epilogue(recipe["epilogue"], activation_spec=activation_spec, recovery=recipe.get("recovery"))
    except IE.EpilogueError as error:
        raise RecipeError(f"epilogue: {error}") from error
