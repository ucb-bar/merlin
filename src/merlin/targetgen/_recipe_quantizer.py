#!/usr/bin/env python3
"""Build TorchAO's own quantization objects from a recipe — runs INSIDE the capture venv.

The capture venv is the only interpreter with torch; the recipe is derived on the other side, from
a hardware target, and crosses as JSON (:mod:`merlin.targetgen.quant_recipe`). This module is the
generic half that turns that data into the framework's extension points:

* a static recipe becomes a PT2E :class:`Quantizer` whose ``QuantizationSpec`` s (dtype, range,
  symmetry, per-tensor or per-channel, observer) are read from the recipe, annotating the aten
  contractions;
* a dynamic recipe becomes a ``quantize_`` configuration with the recipe's granularity — built
  PER LAYER, through TorchAO's own ``AOBaseConfig`` subclass for that (:func:`build_fqn_config`),
  so a layer the target cannot run in its numeric form is mapped to ``None`` rather than being
  quantized because it happened to be a Linear.

There is one quantizer for every target. A target changes the recipe, never this file.
Scoped recipes reuse Merlin's shared SW admission screen; no target-name dispatch
or framework source patch is involved.

WHOSE LIMIT REFUSED A LAYER. :mod:`quant_layer_plan` decides what the TARGET absorbs; this file
adds what this FRAMEWORK BUILD can express of that decision, and the two are recorded under
different codes. A Conv2d the hardware absorbs perfectly well is still not something ``quantize_``
transforms, and filing that as a hardware gap would send someone to read RTL about a Python
dispatch table.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

RECIPE_SCHEMA = "quant_recipe_v1"
DEFAULT_OBSERVER_EPSILON = 2**-12

#: The recipe family of a contraction between two RUN-TIME activations (attention's ``Q @ K^T`` and
#: ``P @ V``). Spelled as :data:`merlin.targetgen.quant_recipe.ACTIVATION_CONTRACTION`; repeated
#: because the capture venv imports this file without merlin. Both operands take the recipe's
#: ACTIVATION spec: neither is a stored weight, so the weight observer and range would describe an
#: operand the model never supplied.
ACTIVATION_CONTRACTION = "activation_contraction"
#: At or below this reduction extent an activation x activation product is an outer product (a
#: rotary frequency table is one), not a contraction worth an int8 operand. TorchAO declines a Linear
#: on the same rule; mirrored from ``_activation_contractions.MIN_REDUCTION``.
MIN_ACTIVATION_REDUCTION = 16

#: Recipe dtype -> the torch dtype a spec is built with. A dtype absent here is refused, not mapped
#: to a neighbour: the recipe states the hardware's element format and a near miss is a different
#: datapath.
_TORCH_DTYPE = {
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "fp8_e4m3": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2",
}


class RecipeError(ValueError):
    """The recipe cannot be realised with this framework build; says which field and why."""


def _require(recipe: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise RecipeError(f"unsupported recipe schema {recipe.get('schema')!r}")
    if recipe.get("status") != "derived":
        raise RecipeError(f"the recipe is {recipe.get('status')!r}, not derived: {recipe.get('underivable')}")
    weight, activation = recipe.get("weight"), recipe.get("activation")
    if not isinstance(weight, Mapping) or not isinstance(activation, Mapping):
        raise RecipeError("a derived recipe carries a weight and an activation spec")
    return weight, activation


def _torch_dtype(name: str):
    import torch

    spelled = _TORCH_DTYPE.get(str(name))
    dtype = getattr(torch, spelled, None) if spelled else None
    if dtype is None:
        raise RecipeError(f"recipe dtype {name!r} has no torch dtype in this build")
    return dtype


def _observer_name(tensor: Mapping[str, Any], *, is_weight: bool) -> str:
    floating_format = str(tensor.get("dtype", "")).startswith("fp8_")
    default = "minmax" if is_weight or floating_format else "histogram"
    return str(tensor.get("observer") or default)


def _spec(tensor: Mapping[str, Any], *, is_weight: bool, eps: float):
    """One ``QuantizationSpec`` from one recipe tensor spec."""
    import torch
    from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
    from torchao.quantization.pt2e.quantizer import QuantizationSpec

    granularity, symmetric = tensor.get("granularity"), bool(tensor.get("symmetric"))
    dtype = _torch_dtype(tensor.get("dtype"))
    floating_format = str(tensor.get("dtype", "")).startswith("fp8_")
    arguments: dict[str, Any] = {"eps": eps}
    if granularity == "tensor":
        qscheme = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
        # A weight is observed once, so its range is its min and max; an activation is observed
        # over a calibration stream, where a histogram is robust to a rare outlier. TorchAO's
        # HistogramObserver uses torch.iinfo and cannot observe an FP8 dtype; its MinMaxObserver
        # explicitly supports float8. Never silently replace an explicitly requested observer.
        observer_name = _observer_name(tensor, is_weight=is_weight)
        if floating_format and observer_name == "histogram":
            raise RecipeError("FP8 histogram observation is unavailable in TorchAO PT2E; use minmax")
        observer = {"minmax": MinMaxObserver, "histogram": HistogramObserver}.get(observer_name)
        import _recipe_observers as RO  # noqa: PLC0415 -- a sibling, by bare name

        percentile = RO.percentile_of(observer_name)
        if percentile is not None and not floating_format:
            observer, arguments["percentile"] = RO.percentile_observer(), percentile
        if observer is None:
            raise RecipeError(f"recipe observer {tensor.get('observer')!r} is not one this quantizer builds")
        axis = None
    elif granularity == "channel" and is_weight:
        qscheme = torch.per_channel_symmetric if symmetric else torch.per_channel_affine
        observer, axis = PerChannelMinMaxObserver, 0
    else:
        raise RecipeError(
            f"a static {'weight' if is_weight else 'activation'} scale per "
            f"{granularity!r} has no PT2E spec; the recipe asks for something this "
            f"path cannot express"
        )
    return QuantizationSpec(
        dtype=dtype,
        observer_or_fake_quant_ctr=observer.with_args(**arguments),
        quant_min=tensor.get("quant_min"),
        quant_max=tensor.get("quant_max"),
        qscheme=qscheme,
        ch_axis=axis,
    )


def _recipe_operators(torch: Any) -> dict[str, tuple[Any, ...]]:
    """ATen spellings covered by each recipe family, shared by the skip check and PT2E."""
    #: The aten operators that ARE the recipe's families. A family the recipe does not list is not
    #: annotated and stays in floating point, which is what "the target cannot absorb it" means.
    operators = {
        # A MATMUL IS A CONTRACTION. This listed `conv2d` and `linear` only, which are the spellings a
        # module built from `nn.Conv2d`/`nn.Linear` exports to -- and a model that calls
        # `torch.matmul` directly exports to `aten.matmul.default` and matched nothing, so the
        # quantizer annotated zero contractions and `validate` rejected the model as containing no
        # operator the recipe covers. Measured on `M3_host_island_seam_gemmini`, whose exported graph
        # is exactly `2x aten.matmul.default`: it is the accelerator/host-island/accelerator seam
        # capsule, so the one capsule built to prove that composition could not be captured at all.
        #
        # `mm` and `bmm` are listed beside it because which of the three survives export is a property
        # of the input ranks and of whatever decomposition ran, not of the model's intent -- matching
        # one spelling and not its siblings is the same bug with a different rank.
        "contraction": (
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
        ),
        # The same contraction spellings, between two run-time activations: annotated only when the
        # recipe lists this family, with the activation spec on BOTH operands.
        ACTIVATION_CONTRACTION: (
            torch.ops.aten.matmul.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
        ),
        "operand_sum": (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor),
        "window_mean": (
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.ops.aten.avg_pool2d.default,
            torch.ops.aten.mean.dim,
        ),
    }
    return operators


#: ``aten.scaled_dot_product_attention`` argument names, in schema order.
_SDPA_FIELDS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
#: Metadata a spelled-out attention step inherits from the fused call it replaces: ownership (the
#: quantizer's placement rule reads it) and source lineage (the capture trace reads it).
_INHERITED_META = ("nn_module_stack", "source_fn_stack", "stack_trace", "from_node")


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


def _dynamic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals and sites of a dynamic rewrite, for the capture's stats."""
    done = [r for r in rows if r.get("status") == "rewritten"]
    return {
        "rewritten": len(done),
        "kept_in_float": [r for r in rows if r.get("status") != "rewritten"],
        **{key: sum(int(r.get(key, 0)) for r in done) for key in ("int8_macs", "device_calls", "host_elements")},
        "sites": done,
    }


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
    import _static_planes as SP  # noqa: PLC0415 -- a sibling, by bare name

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
    """A Linear of a planes family, owned inside the block's scope, takes the planes grid."""
    import _static_planes as SP  # noqa: PLC0415 -- a sibling, by bare name
    import torch

    return bool(
        "contraction" in (block.get("families") or ())
        and node.target == torch.ops.aten.linear.default
        and SP.in_scope(owner, block.get("owners"))
    )


def _rewrite_planes(quantized: Any, recipe: Mapping[str, Any], annotated: int) -> list[dict[str, Any]]:
    """The converted graph's planes sites in their digit form: every annotated one, or an error."""
    if not recipe.get("static_planes"):
        return []
    import _static_planes as SP  # noqa: PLC0415 -- a sibling, by bare name

    planes, low, high = _planes_grid(recipe)
    try:
        rows = SP.rewrite_static_planes(quantized, planes=planes, low=low, high=high)
    except ValueError as error:
        raise RecipeError(f"static_planes: {error}") from error
    if len(rows) != annotated:
        raise RecipeError(f"static_planes: {annotated} Linear(s) annotated on the planes grid, {len(rows)} rewritten")
    return [{**row, "status": "rewritten"} for row in rows]


def build_quantizer(
    recipe: Mapping[str, Any],
    *,
    layer_plan: Mapping[str, Any],
    source_layer_inventory: list[dict[str, Any]] | None = None,
    eps: float = DEFAULT_OBSERVER_EPSILON,
    fold_candidates: list | None = None,
):
    """The PT2E quantizer a static recipe describes, limited to placed operations.

    ATen operator names alone cannot license quantization: an unsupported module may call
    ``aten.linear`` internally. Export's ``nn_module_stack`` identifies the owning module.
    Leaf operations use their module plan; container/root arithmetic uses its own
    functional operation plan. Missing ownership or unknown leaves are refused.
    """
    import torch
    from torchao.quantization.pt2e.quantizer import QuantizationAnnotation, Quantizer

    weight, activation = _require(recipe)
    if activation.get("mode") != "static":
        raise RecipeError("build_quantizer realises static recipes; a dynamic one goes through quantize_config")
    operators = _recipe_operators(torch)
    listed = tuple(recipe.get("families") or ())
    targets = {op for family in listed if family == "contraction" for op in operators.get(family, ())}
    pairs = set(operators[ACTIVATION_CONTRACTION]) if ACTIVATION_CONTRACTION in listed else set()
    sums = set(operators["operand_sum"]) if "operand_sum" in listed else set()
    means = set(operators["window_mean"]) if "window_mean" in listed else set()
    # Families the recipe's ``dynamic`` block takes out of static annotation (_dynamic_contractions).
    dynamic_block = dict(recipe.get("dynamic") or {})
    dynamic_families = set(dynamic_block.get("families") or ())
    if "contraction" in dynamic_families and _weight_rounding(recipe) is not None:
        raise RecipeError("the dynamic block rounds stored weights to nearest; GPTQ codes are set on the static path")
    # Linears whose activation takes the recipe's ``static_planes`` grid (_static_planes).
    planes_block = dict(recipe.get("static_planes") or {})
    if not targets and not pairs:
        raise RecipeError(f"the recipe quantizes {recipe.get('families')}, none of which this quantizer can annotate")
    #: What a value may pass through on its way to a reader without ceasing to be the same integers:
    #: an activation a readout applies, and shape changes that touch no element.
    transparent = {
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.dropout.default,
    }
    # Shape-only views preserve a stored parameter's identity. An arbitrary producer does not:
    # quantizing the second input of an activation x activation matmul as a weight would use the
    # wrong observer and imply a constant operand that the model never supplied.
    weight_views = {
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.contiguous.default,
    }

    decisions = {str(d.get("fqn")): d for d in layer_plan.get("layers") or ()}
    software_decisions = []

    def software_allows(node: Any, family: str) -> bool:
        selected = recipe.get("software_admission")
        if selected is None:
            return True  # Compatibility recipes retain their recorded policy.
        from merlin.targetgen.software_spec import admit_operation

        value = node.args[0].meta.get("val") if node.args and isinstance(node.args[0], torch.fx.Node) else None
        prospective = {
            "family": family,
            "operand_dtype": activation["dtype"],
            "accum_dtype": recipe["accumulator_dtype"],
            "rank": len(value.shape) if isinstance(value, torch.Tensor) else None,
            "layout": "row_major_contiguous" if isinstance(value, torch.Tensor) and value.is_contiguous() else None,
            "scale_granularity": activation["granularity"],
        }
        decision = admit_operation(selected, str(node.target), prospective, "accelerator")
        software_decisions.append(
            {
                "node": node.name,
                **decision,
                "scope": "prospective quantization screen; not realized precision or target admission",
            }
        )
        return decision["status"] != "unsupported"

    def owner(node: Any) -> str | None:
        stack = node.meta.get("nn_module_stack")
        if not isinstance(stack, Mapping) or not stack:
            return None
        paths = [entry[0] for entry in stack.values() if isinstance(entry, (tuple, list)) and entry]
        return str(paths[-1]) if paths else None

    def stored_floating_weight(value: Any) -> bool:
        if not isinstance(value, torch.fx.Node):
            return False
        if value.op == "get_attr":
            tensor = value.meta.get("val")
            return isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        return bool(
            value.op == "call_function"
            and value.target in weight_views
            and value.args
            and stored_floating_weight(value.args[0])
        )

    def stored_weight_shape(value: Any) -> list[int] | None:
        while isinstance(value, torch.fx.Node) and value.op == "call_function" and value.target in weight_views:
            value = value.args[0] if value.args else None
        tensor = value.meta.get("val") if isinstance(value, torch.fx.Node) and value.op == "get_attr" else None
        return list(tensor.shape) if isinstance(tensor, torch.Tensor) else None

    def runtime_pair(node: Any) -> bool:
        """Both operands are run-time floating tensors (rank >= 2), neither a stored parameter."""
        operands = node.args[:2]
        if len(operands) < 2 or not all(isinstance(a, torch.fx.Node) for a in operands):
            return False
        if any(stored_floating_weight(a) for a in operands):
            return False
        values = [a.meta.get("val") for a in operands]
        return all(isinstance(v, torch.Tensor) and v.is_floating_point() and v.dim() >= 2 for v in values)

    def eligible(node: Any, family: str) -> bool:
        # An activation contraction is a contraction to the SW admission screen; the family only
        # decides which spec its second operand takes.
        if not software_allows(node, "contraction" if family == ACTIVATION_CONTRACTION else family):
            return False
        if family == ACTIVATION_CONTRACTION and int(node.args[0].meta["val"].shape[-1]) <= MIN_ACTIVATION_REDUCTION:
            return False
        if family == "contraction":
            if len(node.args) < 2 or not stored_floating_weight(node.args[1]):
                return False
            operand = node.args[0]
            value = operand.meta.get("val") if isinstance(operand, torch.fx.Node) else None
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                return False
        path = owner(node)
        if path is None:
            return False
        if path:
            decision = decisions.get(path)
            if source_layer_inventory is not None and family == "contraction":
                if not _source_weight_geometry_matches(source_layer_inventory, path, stored_weight_shape(node.args[1])):
                    return False
            if decision is not None:
                return bool(decision.get("placement") == "device" and decision.get("family") == family)
            # The inventory contains leaves only. A container still owns functional
            # arithmetic between its children (for example, a residual add). Prove
            # it is a container in this plan before using the root operation rule.
            if not any(fqn.startswith(path + ".") for fqn in decisions):
                return False
        # Root/container arithmetic has no leaf decision. Check the actual stored
        # operand and its shape with the same rule as a module.
        if family == "contraction":
            kernel_node = node.args[1]
            while kernel_node.op == "call_function":
                kernel_node = kernel_node.args[0]
            kernel = kernel_node.meta.get("val")
            shape = list(kernel.shape) if kernel is not None and hasattr(kernel, "shape") else None
        else:
            shape = None
        import quant_layer_plan as QLP

        direct = QLP.plan(
            recipe, [{"fqn": "<root>", "kind": "functional", "family": family, "weight_shape": shape}]
        ).decisions[0]
        return direct.placement == QLP.DEVICE

    def family_of(node: Any) -> str | None:
        if node.target in pairs and runtime_pair(node):
            return ACTIVATION_CONTRACTION
        if node.target in targets:
            return "contraction"
        if node.target in sums:
            return "operand_sum"
        if node.target in means:
            return "window_mean"
        return None

    def closable(nodes: Any) -> set[Any]:
        """The sums and means whose result only quantized readers ever see.

        Putting a sum on the integer grid is worth its accuracy cost only when the result is
        requantized, because that is what closes it into one integer region a unit can take. A sum
        read by a normalization, or returned, stays a float add whose operands were quantized for
        nothing (measured on a two-layer decoder: 8 sums and 5 means annotated, none placeable,
        17 more host regions). A reader may itself be a candidate, so this is a fixpoint: drop
        every candidate with a reader that is neither a contraction nor a surviving candidate.
        """
        candidates = set()
        for node in nodes:
            if node.target in sums and eligible(node, "operand_sum"):
                operands = [a for a in node.args[:2] if isinstance(a, torch.fx.Node) and a.op != "get_attr"]
                if len(operands) == 2:
                    candidates.add(node)
            elif (
                node.target in means
                and eligible(node, "window_mean")
                and node.args
                and isinstance(node.args[0], torch.fx.Node)
            ):
                candidates.add(node)

        def readers(node: Any) -> list[Any]:
            found, pending = [], list(node.users)
            while pending:
                user = pending.pop()
                if user.op == "call_function" and user.target in transparent:
                    pending.extend(user.users)
                else:
                    found.append(user)
            return found

        while True:
            dropped = {
                node
                for node in candidates
                if not (seen := readers(node))
                or any(
                    (r.target not in targets or not eligible(r, "contraction")) and r not in candidates for r in seen
                )
            }
            if not dropped:
                return candidates
            candidates -= dropped

    class CapabilityQuantizer(Quantizer):
        def __init__(self) -> None:
            self.activation = _spec(activation, is_weight=False, eps=eps)
            self.weight = _spec(weight, is_weight=True, eps=eps)
            self.annotated = 0
            self.annotated_pairs = 0
            self.pair_sites: list[dict[str, Any]] = []
            self.annotated_sums = 0
            self.annotated_means = 0
            self.left_in_float = 0
            self.refused_by_placement: list[dict[str, str | None]] = []
            self.software_decisions = software_decisions
            self.fold_candidates = fold_candidates or []
            self.dynamic_rows: list[dict[str, Any]] = []
            self.planes = _planes_spec(recipe, eps=eps) if planes_block else None
            self.annotated_planes = 0

        def rewrite_dynamic(self, graph_module: Any) -> list[dict[str, Any]]:
            """Rewrite the dynamic families' placed contractions into their integer form, in place.

            Runs before PT2E sees the graph, so a rewritten contraction is never also annotated. One the
            rewrite cannot express is left in floating point and listed, never approximated.
            """
            if not dynamic_families:
                return []
            import _dynamic_contractions as DC  # noqa: PLC0415 -- a sibling, by bare name

            qmax = DC.qmax_of(recipe)
            for node in list(graph_module.graph.nodes):
                family = family_of(node) if node.op == "call_function" else None
                if family not in dynamic_families or not eligible(node, family):
                    continue
                row = {"family": family, "owner": owner(node)}
                weight = None
                if family == "contraction":
                    weight_node = node.args[1] if len(node.args) > 1 else None
                    if node.target == torch.ops.aten.linear.default and getattr(weight_node, "op", None) == "get_attr":
                        weight = graph_module
                        for part in str(weight_node.target).split("."):
                            weight = getattr(weight, part)
                    if weight is None:
                        self.dynamic_rows.append(
                            {**row, "node": node.name, "status": "kept", "why": "no stored weight"}
                        )
                        continue
                try:
                    if weight is None:
                        done = DC.rewrite_activation_contraction(graph_module, node, dynamic_block, qmax)
                    else:
                        done = DC.rewrite_weight_contraction(graph_module, node, weight, dynamic_block, qmax)
                except ValueError as why:
                    self.dynamic_rows.append({**row, "node": node.name, "status": "kept", "why": str(why)})
                    continue
                self.dynamic_rows.append({**row, **done, "status": "rewritten"})
            if any(r["status"] == "rewritten" for r in self.dynamic_rows):
                graph_module.graph.lint()
                graph_module.recompile()
            return self.dynamic_rows

        def transform_for_annotation(self, graph_module: Any) -> Any:
            # TorchAO calls this after folding Conv+BN and before inserting
            # observers, while the exact FX nodes and their rewired users are
            # still available to prove each folded source identity.
            if self.fold_candidates:
                from m2m.capture.trace import attach_pt2e_conv_bn_folds

                attach_pt2e_conv_bn_folds(graph_module, self.fold_candidates)
            return graph_module

        def annotate(self, graph_module: Any) -> Any:
            nodes = list(graph_module.graph.nodes)
            kept = closable(nodes)
            self.left_in_float = sum(1 for n in nodes if (n.target in sums or n.target in means) and n not in kept)
            for node in nodes:
                family = family_of(node)
                if family in dynamic_families:
                    continue  # rewritten by rewrite_dynamic, or listed there as kept in floating point
                if family is not None and not eligible(node, family):
                    self.refused_by_placement.append(
                        {"node": node.name, "operator": str(node.target), "owner": owner(node), "family": family}
                    )
                    continue
                if (node.target in sums or node.target in means) and node not in kept:
                    continue
                if node.target in sums:
                    # A sum of two TENSORS, both operands on the recipe's activation grid. A scalar
                    # or a parameter operand is a different operation (a bias, a shift) and is left
                    # alone. The sum's result stays in floating point like a contraction's: whoever
                    # reads it quantizes its own input edge, and group formation closes the region.
                    operands = [a for a in node.args[:2] if isinstance(a, torch.fx.Node) and a.op != "get_attr"]
                    if len(operands) == 2:
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={operand: self.activation for operand in operands}, _annotated=True
                        )
                        self.annotated_sums += 1
                    continue
                if node.target in means:
                    # Only the input: a mean has no stored operand, and its reader quantizes its own edge.
                    if node.args and isinstance(node.args[0], torch.fx.Node):
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={node.args[0]: self.activation}, _annotated=True
                        )
                        self.annotated_means += 1
                    continue
                if family == ACTIVATION_CONTRACTION:
                    # Two run-time operands, each on the activation grid with its own calibrated
                    # static scale; the result stays in floating point like any contraction's.
                    operand, other = node.args[:2]
                    node.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map={operand: self.activation, other: self.activation}, _annotated=True
                    )
                    self.annotated_pairs += 1
                    self.pair_sites.append(
                        {
                            "node": node.name,
                            "owner": owner(node),
                            "shapes": [list(a.meta["val"].shape) for a in (operand, other)],
                        }
                    )
                    continue
                if node.target not in targets or len(node.args) < 2:
                    continue
                operand, kernel = node.args[:2]
                planes = self.planes is not None and _planes_site(planes_block, node, owner(node))
                # Both inputs of the contraction; its result stays in floating point and the next
                # contraction quantizes its own input edge. What follows a contraction is absorbed
                # by the compiler's group formation, not assumed here.
                # A weight's channel is its OUTPUT axis: dim 0 of a Linear's [N, K] or a conv's
                # [Cout, ...], but the LAST dim of a matmul's stored [..., K, N] operand. Dim 0 there is
                # the reduction, and a scale along a reduction is one no readout can apply.
                stored = self.weight
                if self.weight.ch_axis is not None and node.target not in (
                    torch.ops.aten.linear.default,
                    torch.ops.aten.conv2d.default,
                ):
                    stored = dataclasses.replace(self.weight, ch_axis=kernel.meta["val"].dim() - 1)
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={operand: self.planes if planes else self.activation, kernel: stored},
                    _annotated=True,
                )
                self.annotated_planes += int(planes)
                self.annotated += int(not planes)
            # Closed sum/mean candidates form a set. Its fixpoint can revisit a
            # reader in any object-hash order, so keep every decision (including
            # repeated screens) but expose the complete report rows canonically.
            # Otherwise equivalent captures disagree only in metadata and fail
            # byte-exact independent replay.
            self.software_decisions.sort(
                key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            return graph_module

        def validate(self, graph_module: Any) -> None:
            del graph_module
            # COUNT EVERY FAMILY THE RECIPE LISTS, not just contractions. Three counters are kept and
            # this read one, so a model carrying only the operand sums or window means a recipe also
            # covers was refused with a message saying its families were not covered -- which was
            # false, and pointed a reader at the model rather than at this table.
            rewritten = sum(r["status"] == "rewritten" for r in self.dynamic_rows)
            annotated = self.annotated + self.annotated_planes + self.annotated_pairs
            if annotated + self.annotated_sums + self.annotated_means + rewritten == 0:
                raise RecipeError(
                    f"the model has no operator the recipe's families cover "
                    f"(families={list(listed)}; annotated contractions=0, activation contractions=0, "
                    f"sums=0, means=0)"
                )

    return CapabilityQuantizer()


def quantize_config(recipe: Mapping[str, Any]):
    """The ``quantize_`` configuration a dynamic recipe describes."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
        PerRow,
        PerTensor,
    )

    weight, activation = _require(recipe)
    if activation.get("mode") != "dynamic":
        raise RecipeError("quantize_config realises dynamic recipes")
    if _weight_rounding(recipe) is not None:
        raise RecipeError("GPTQ weight rounding is realised on the static PT2E path; quantize_ rounds to nearest")
    granularity = {"tensor": PerTensor, "channel": PerRow}.get(str(weight.get("granularity")))
    if granularity is None:
        raise RecipeError(
            f"a dynamic recipe with weights per {weight.get('granularity')!r} has no configuration in this build"
        )
    dtype = str(weight.get("dtype"))
    if dtype == "int8":
        return Int8DynamicActivationInt8WeightConfig(granularity=granularity())
    if dtype in ("fp8_e4m3", "fp8_e5m2"):
        return Float8DynamicActivationFloat8WeightConfig(granularity=granularity())
    raise RecipeError(f"no dynamic configuration for recipe dtype {dtype!r}")


#: The module kinds ``quantize_`` actually transforms under the configs :func:`quantize_config`
#: builds. This is a property of the FRAMEWORK BUILD, not of any target: TorchAO's handlers for the
#: dynamic-activation configs replace a Linear's weight parameter, and nothing else's. A device
#: layer of any other kind is mapped to ``None`` with that named as the reason, so the plan's
#: hardware verdict is never quietly rewritten into a framework one.
_QUANTIZE_TRANSFORMS = ("Linear", "LazyLinear")
FRAMEWORK_CANNOT_EXPRESS = "framework_cannot_express"


def layer_inventory(model: Any) -> list[dict[str, Any]]:
    """The layers of ``model`` as plain mappings :mod:`quant_layer_plan` can decide on.

    Leaf modules only, and no decision is taken here: this reports the model's own shape (name,
    class, stored-operand shape) and nothing else, so that the placement rule stays in the one
    module that is readable from both sides of the capture boundary and testable without torch.
    """
    layers: list[dict[str, Any]] = []
    for fqn, module in model.named_modules():
        if not fqn or any(True for _ in module.children()):
            continue  # container-owned functional arithmetic is planned per operation
        weight = getattr(module, "weight", None)
        layers.append(
            {
                "fqn": fqn,
                "kind": type(module).__name__,
                "weight_shape": (list(weight.shape) if weight is not None and hasattr(weight, "shape") else None),
                "stores_operand": any(True for _ in module.named_parameters(recurse=False))
                or any(True for _ in module.named_buffers(recurse=False)),
            }
        )
    return layers


def build_fqn_config(recipe: Mapping[str, Any], layer_plan: Mapping[str, Any]):
    """TorchAO's per-layer configuration for ``layer_plan``, as its own ``AOBaseConfig`` subclass.

    Returns ``(config, notes)``. ``config`` maps each layer's fully qualified name to the
    configuration the recipe describes, or to ``None`` where the layer is not quantized — which is
    TorchAO's own spelling for "leave this module alone", so nothing here forks its behaviour.
    ``notes`` is one entry per host layer saying which limit refused it and in whose words.

    Every layer the plan saw appears in the mapping, including the refused ones. An absent key and
    a ``None`` key behave the same way today; recording ``None`` explicitly is what makes the
    refusal reviewable instead of inferable from a gap in a dict.
    """
    from collections import OrderedDict

    try:  # its current name, with the one it shipped under before as the fallback
        from torchao.quantization import FqnToConfig as _FqnToConfig
    except ImportError:  # pragma: no cover - older builds
        from torchao.quantization import ModuleFqnToConfig as _FqnToConfig

    shared = quantize_config(recipe)
    mapping: OrderedDict[str, Any] = OrderedDict()
    notes: list[dict[str, Any]] = []
    for layer in layer_plan.get("layers") or ():
        fqn, kind = str(layer.get("fqn") or ""), str(layer.get("kind") or "")
        if layer.get("placement") != "device":
            mapping[fqn] = None
            notes.append(
                {"fqn": fqn, "kind": kind, "refusal": layer.get("refusal"), "why": layer.get("why"), "by": "target"}
            )
            continue
        if kind not in _QUANTIZE_TRANSFORMS:
            mapping[fqn] = None
            notes.append(
                {
                    "fqn": fqn,
                    "kind": kind,
                    "refusal": FRAMEWORK_CANNOT_EXPRESS,
                    "why": (
                        f"the target absorbs this layer ({layer.get('why')}), but quantize_ transforms "
                        f"{list(_QUANTIZE_TRANSFORMS)} under this configuration and not a {kind!r}"
                    ),
                    "by": "framework",
                }
            )
            continue
        mapping[fqn] = shared
    if not any(value is not None for value in mapping.values()):
        raise RecipeError(
            f"no layer of this model is both absorbed by the target and expressible by this framework build: {notes}"
        )
    return _FqnToConfig(mapping), notes


def _plan_layers(
    recipe: Mapping[str, Any], model: Any, *, source_layer_inventory: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The per-layer plan for ``model`` under ``recipe``, decided by the rule both sides share.

    The rule lives in :mod:`quant_layer_plan`, which imports neither merlin nor a framework, so the
    capture venv reads the same module the derivation side tests. It sits beside this file and is
    imported as a sibling by bare name, the way this file is itself imported.
    """
    import sys
    from pathlib import Path

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import quant_layer_plan as QLP

    return QLP.plan(
        recipe, layer_inventory(model) if source_layer_inventory is None else source_layer_inventory
    ).to_dict()


def _weight_rounding(recipe: Mapping[str, Any], prepared: Any = None) -> Any:
    """How ``recipe`` rounds its weights: ``None`` for round-to-nearest, else GPTQ (``_gptq.py``).

    Without ``prepared`` this returns the recipe's validated GPTQ parameters. With a prepared PT2E
    module it returns the hook that accumulates Hessians during calibration (``track``) and sets
    GPTQ's codes into the stored weights before ``convert_pt2e`` folds them (``apply``). The
    algorithm, and why the hook sits at that point, are documented in ``_gptq.py``, a sibling
    imported by bare name like ``quant_layer_plan``.
    """
    import sys
    from pathlib import Path

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import _gptq

    try:
        parameters = _gptq.rounding_of(recipe["weight"])
    except _gptq.GPTQError as error:
        raise RecipeError(str(error)) from error
    if parameters is None or prepared is None:
        return parameters
    import torch

    return _gptq.PT2EWeightRounding(prepared, parameters, contractions=_recipe_operators(torch)["contraction"])


def _as_tuple(sample: Any) -> tuple:
    return tuple(sample) if isinstance(sample, (tuple, list)) else (sample,)


def calibration_limit(recipe: Mapping[str, Any], default: int) -> int:
    """How many calibration samples a static recipe observes: its policy's count, else ``default``.

    The count is part of the recipe (``framework_capture_policy.calibration_count``) because it
    changes the scales, so two captures of one recipe observe the same number of samples and a
    receipt keyed on the recipe digest names it. Which samples is the loader's calibration stream.
    """
    policy = recipe.get("framework_capture_policy") or {}
    value = policy.get("calibration_count", default)
    if type(value) is not int or value < 1:
        raise RecipeError("framework_capture_policy.calibration_count must be a positive integer")
    return value


def _no_batch_norm_fold_needed(original_snapshot: Any, exported: Any) -> bool:
    """Prove the selected original and PT2E-input graphs contain no BN to fold.

    A missing upstream fold-lineage API is harmless only for complete, plain
    ATen graphs with no BatchNorm call. Unknown/custom calls cannot establish
    absence, even when their names do not mention BatchNorm.
    """
    if not isinstance(original_snapshot, Mapping):
        return False
    if original_snapshot.get("schema") != "m2m.frontend_graph.v1" or original_snapshot.get("status") != "complete":
        return False
    nodes = original_snapshot.get("nodes")
    if not isinstance(nodes, list):
        return False
    for node in nodes:
        if not isinstance(node, Mapping):
            return False
        if node.get("op") not in {"call_function", "call_method", "call_module"}:
            continue
        target = node.get("target")
        if node.get("classification") != "aten" or not isinstance(target, str):
            return False
        if "batch_norm" in target.lower() or "batchnorm" in target.lower():
            return False
    graph = getattr(exported, "graph", None)
    if graph is None:
        return False
    for node in graph.nodes:
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        if node.op == "call_module" and node.target == "_guards_fn":
            # torch.export.module() injects a shape-guard submodule after the
            # original ATen snapshot. The selected framework's GuardsFn does
            # not perform model computation and cannot hide a BN call.
            get_submodule = getattr(exported, "get_submodule", None)
            guard = get_submodule("_guards_fn") if callable(get_submodule) else None
            guard_type = type(guard)
            if guard_type.__module__ == "torch.export._unlift" and guard_type.__qualname__ == "GuardsFn":
                continue
            return False
        target = str(node.target)
        if node.op != "call_function" or not target.startswith("aten."):
            return False
        if "batch_norm" in target.lower() or "batchnorm" in target.lower():
            return False
    return True


def apply_recipe(
    model: Any,
    recipe: Mapping[str, Any],
    *,
    example_inputs: tuple,
    calibration_inputs: Iterable[Any] | None = None,
    calibration_samples: int = 100,
    original_frontend_snapshot: dict[str, Any] | None = None,
    source_layer_inventory: list[dict[str, Any]] | None = None,
) -> Any:
    """Quantize ``model`` under ``recipe`` and return the module the capture should lower."""
    import torch

    _weight, activation = _require(recipe)
    # Integral EXTERNAL inputs are not proof that a model is already quantized: a language model
    # accepts integer token IDs, then performs floating contractions after an embedding. Inspect
    # the operations covered by this recipe before skipping calibration. A graph with only integer
    # contractions (such as an authored integer seam) still needs no observer.
    tensors = [t for t in (example_inputs or ()) if isinstance(t, torch.Tensor)]
    integral_inputs = bool(tensors) and not any(torch.is_floating_point(t) for t in tensors)

    def skip_already_integer() -> Any:
        model._recipe_quantization_stats = {  # type: ignore[attr-defined]
            "applied": False,
            "why": (
                f"example inputs are integral ({sorted({str(t.dtype) for t in tensors})}) and no "
                "floating-point operation covered by the recipe needs quantization"
            ),
        }
        return model

    if activation.get("mode") == "dynamic":
        if recipe.get("software_admission") is not None:
            raise RecipeError("scoped SW admission is currently supported by static PT2E recipes only")
        # Dynamic TorchAO transforms floating Linear modules rather than observing the external
        # input. Integer token IDs must not suppress those internal projections either.
        if integral_inputs and not any(
            isinstance(module, (torch.nn.Linear, torch.nn.LazyLinear))
            and (weight := getattr(module, "weight", None)) is not None
            and torch.is_floating_point(weight)
            for module in model.modules()
        ):
            return skip_already_integer()
        from torchao.quantization import quantize_

        layer_plan = _plan_layers(recipe, model)
        config, notes = build_fqn_config(recipe, layer_plan)
        # filter_fn=None is REQUIRED, not tidiness: quantize_ defaults it to "is this a Linear",
        # and TorchAO refuses to hold both that and a per-layer mapping. Which layers are
        # quantized is the plan's verdict, derived from the target; being a Linear is not a reason.
        quantize_(model, config, filter_fn=None)
        placed = sum(1 for value in config.fqn_to_config.values() if value is not None)
        model._recipe_quantization_stats = {  # type: ignore[attr-defined]
            "api": "quantize_",
            "recipe_sha256": recipe.get("recipe_sha256"),
            "plan_sha256": layer_plan.get("plan_sha256"),
            "layers_seen": len(config.fqn_to_config),
            "layers_quantized": placed,
            # Every layer that is NOT in the target's numeric form, and whose limit put it there.
            "layers_on_host": notes,
            "refusals": layer_plan.get("refusals"),
            "weight_granularity": recipe["weight"]["granularity"],
            "activation_granularity": recipe["activation"]["granularity"],
        }
        return model

    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    if not example_inputs:
        raise RecipeError("a static recipe needs example inputs to export the model")
    # An ExportedProgram.module() GraphModule may return None from eval(); the
    # owned module, not that method's return value, remains the export input.
    model.eval()
    exported_program = torch.export.export(model, tuple(example_inputs))
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import (
            attach_original_identity,
            snapshot_exported_program,
            tuple_selection_trace_program,
        )

        actual = snapshot_exported_program(exported_program, stage="quantization_input")
        attach_original_identity(exported_program, original_frontend_snapshot, actual)
        # PT2E replays this exact ExportedProgram. Its proxy interpreter may
        # alias tuple selections onto the producer's result proxies; preserve
        # the selecting call-site lineage at that replay, not by guessing from
        # node order or tensor shape after quantization.
        exported_program = tuple_selection_trace_program(exported_program)
    exported = exported_program.module()
    # The selected frontend owns the semantic rewrite from string-padding
    # Conv2d to the numeric overload TorchAO observes. Apply it to the exact
    # exported graph before either the integral-input check or PT2E folding:
    # the recipe's operator table must never misread an unnormalized floating
    # Conv as absent. This does not rewrite the source model or its recipe.
    from m2m.capture.pt2e_padding import normalize_conv2d_padding

    padding_normalizations = normalize_conv2d_padding(exported)
    if original_frontend_snapshot is not None:
        source_nodes = {row["id"]: row for row in original_frontend_snapshot.get("nodes", ())}
        source_padding_schema = torch.ops.aten.conv2d.padding._schema.arguments
        padding_index = next(
            index for index, argument in enumerate(source_padding_schema) if argument.name == "padding"
        )
        padding_argument = source_padding_schema[padding_index]
        for row in padding_normalizations:
            origins = set(row["source_ids"]).intersection(source_nodes)
            # Precision staging may add intermediate identities. Exactly one
            # original string-padding Conv must own this rewritten node. The
            # selected source spelling must also agree on the literal padding
            # mode: an unrelated same-shaped Conv is not semantic ancestry.
            source = source_nodes[next(iter(origins))] if len(origins) == 1 else {}
            args, kwargs = source.get("args"), source.get("kwargs")
            if isinstance(args, list) and isinstance(kwargs, Mapping) and "padding" not in kwargs:
                source_padding = (
                    args[padding_index]
                    if len(args) > padding_index
                    else padding_argument.default_value
                    if padding_argument.has_default_value()
                    else None
                )
            elif isinstance(args, list) and isinstance(kwargs, Mapping) and len(args) <= padding_index:
                source_padding = kwargs.get("padding")
            else:
                source_padding = None
            if source.get("target") != "aten.conv2d.padding" or source_padding != row["padding"]:
                raise RecipeError("Conv2d string-padding normalization lost original source ancestry")
    # Only a recipe that quantizes activation x activation contractions needs a fused attention's
    # two matmuls in the graph; every other recipe sees the graph exactly as exported.
    attention = spell_attention(exported) if ACTIVATION_CONTRACTION in (recipe.get("families") or ()) else []
    if integral_inputs and not _has_floating_recipe_work(exported, recipe):
        return skip_already_integer()
    layer_plan = _plan_layers(recipe, model, source_layer_inventory=source_layer_inventory)
    policy = recipe.get("framework_capture_policy") or {}
    epsilon = policy.get("observer_epsilon", DEFAULT_OBSERVER_EPSILON)
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise RecipeError("observer epsilon must be finite and positive")
    fold_candidates = []
    fold_provenance_api = "not_applicable"
    if original_frontend_snapshot is not None:
        from m2m.capture import trace as capture_trace

        fold_api = getattr(capture_trace, "pt2e_conv_bn_fold_candidates", None)
        if callable(fold_api):
            fold_candidates = fold_api(exported, original_frontend_snapshot)
            fold_provenance_api = "selected_model2mlir"
        elif _no_batch_norm_fold_needed(original_frontend_snapshot, exported):
            # There is no BatchNorm source identity for a PT2E fold to carry.
            # This is an exact graph absence proof, not a family-level alias.
            fold_provenance_api = "not_needed_no_batch_norm"
        else:
            raise RecipeError(
                "selected model2MLIR lacks pt2e_conv_bn_fold_candidates; "
                "cannot prove PT2E Conv+BatchNorm source identity for this graph"
            )
    quantizer = build_quantizer(
        recipe,
        layer_plan=layer_plan,
        source_layer_inventory=source_layer_inventory,
        eps=epsilon,
        fold_candidates=fold_candidates,
    )
    dynamic = quantizer.rewrite_dynamic(exported)
    prepared = prepare_pt2e(exported, quantizer)
    samples = calibration_inputs if calibration_inputs is not None else (tuple(example_inputs),)
    sample_limit = calibration_limit(recipe, calibration_samples)
    # GPTQ weight rounding, when the recipe asks for it: Hessians accumulate during this same
    # calibration pass, and the codes are set into the stored weights before convert folds them.
    weight_rounding = _weight_rounding(recipe, prepared)
    if weight_rounding is not None:
        samples = weight_rounding.track(samples)
    calibrated = 0
    with torch.no_grad():
        for sample in samples:
            if calibrated >= sample_limit:
                break
            prepared(*_as_tuple(sample))
            calibrated += 1
    if calibrated == 0:
        raise RecipeError("static calibration saw no input sample")
    weight_rounding_report = weight_rounding.apply() if weight_rounding is not None else None
    quantized = convert_pt2e(prepared, use_reference_representation=False, fold_quantize=True)
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import attach_quantization_boundaries

        operators = _recipe_operators(torch)
        selected = {op for family in recipe.get("families") or () for op in operators.get(family, ())}
        attach_quantization_boundaries(quantized, selected)
    planes = _rewrite_planes(quantized, recipe, quantizer.annotated_planes)
    pruned = None
    try:  # the capture library's own dead-state sweep, when present
        from m2m.capture.torchao_pipeline import _drop_unused_graph_state

        pruned = _drop_unused_graph_state(quantized)
    except Exception:  # noqa: BLE001 -- an optional tidy-up; the graph is correct without it
        pass
    try:
        from torch.ao.quantization import allow_exported_model_train_eval

        allow_exported_model_train_eval(quantized)
    except (ImportError, AttributeError):
        pass
    quantized._recipe_quantization_stats = {  # type: ignore[attr-defined]
        "api": "pt2e",
        "recipe_sha256": recipe.get("recipe_sha256"),
        "plan_sha256": layer_plan.get("plan_sha256"),
        "layers_on_host": [d for d in layer_plan.get("layers") or () if d.get("placement") == "host"],
        "operators_refused_by_placement": quantizer.refused_by_placement,
        # Every annotated contraction, whatever its operands: the capture's census compares this with
        # the contractions the integer rewrite saw. The activation x activation share is broken out.
        "annotated_contractions": quantizer.annotated + quantizer.annotated_pairs,
        **(
            {
                "annotated_activation_contractions": quantizer.annotated_pairs,
                "activation_contraction_sites": quantizer.pair_sites,
                "attention_spelled": attention,
            }
            if ACTIVATION_CONTRACTION in (recipe.get("families") or ())
            else {}
        ),
        # Contractions the recipe's dynamic block rewrote into integer form (never PT2E-annotated), with
        # the Phase 2 cost they carry: device int8 MACs and calls, and the host's elementwise work
        # (absmax and mean passes, quantize, the outer-product rescale per output element and plane).
        **({"dynamic_contractions": _dynamic_summary(dynamic)} if recipe.get("dynamic") else {}),
        # Linears whose activation the static_planes block split into int8 digit planes (not counted
        # in annotated_contractions: integerization never sees them), with the same cost columns.
        **({"static_planes": _dynamic_summary(planes)} if recipe.get("static_planes") else {}),
        "annotated_sums": quantizer.annotated_sums,
        "annotated_means": quantizer.annotated_means,
        # Sums and means the recipe's families cover and the graph does not requantize: left alone.
        "sums_and_means_left_in_float": quantizer.left_in_float,
        "calibration_samples": calibrated,
        **({"weight_rounding": weight_rounding_report} if weight_rounding_report is not None else {}),
        "pruned_dead_state_tensors": pruned,
        "weight_granularity": recipe["weight"]["granularity"],
        "activation_granularity": recipe["activation"]["granularity"],
        "software_admission": quantizer.software_decisions,
        "fold_provenance_api": fold_provenance_api,
        "conv2d_padding_normalizations": padding_normalizations,
        "framework_capture_policy": {
            "activation_observer": _observer_name(recipe["activation"], is_weight=False),
            "weight_observer": _observer_name(recipe["weight"], is_weight=True),
            "observer_epsilon": epsilon,
            "calibration_source": "explicit_calibration_stream" if calibration_inputs is not None else "example_inputs",
            "calibration_count": calibrated,
            "calibration_sample_limit": sample_limit,
        },
    }
    return quantized


def agreement(reference: Any, candidate: Any, samples: Iterable[Any], *, limit: int = 64) -> dict[str, Any]:
    """How a quantized model's outputs compare with the floating-point model's, on real samples.

    A contract change is never an optimisation: moving a weight scale from per-channel to
    per-tensor changes the numbers, and this is the receipt that says by how much. Reported per
    stream, not judged here; the consumer holds it to the workload's own bar.
    """
    import torch

    count = agree = 0
    cosines: list[float] = []
    worst = 0.0
    with torch.no_grad():
        for sample in samples:
            if count >= limit:
                break
            inputs = _as_tuple(sample)
            expected = torch.utils._pytree.tree_flatten(reference(*inputs))[0][0].float().flatten()
            actual = torch.utils._pytree.tree_flatten(candidate(*inputs))[0][0].float().flatten()
            cosines.append(float(torch.nn.functional.cosine_similarity(expected, actual, dim=0)))
            worst = max(worst, float((expected - actual).abs().max()))
            agree += int(int(expected.argmax()) == int(actual.argmax()))
            count += 1
    if not count:
        return {"samples": 0}
    return {
        "samples": count,
        "argmax_agreement": agree / count,
        "cosine_min": min(cosines),
        "cosine_mean": sum(cosines) / count,
        "max_abs_error": worst,
    }
