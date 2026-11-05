"""Validate recipe tensor specifications and construct their TorchAO quantization specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


RECIPE_SCHEMA = "quant_recipe_v1"

DEFAULT_OBSERVER_EPSILON = 2**-12

ACTIVATION_CONTRACTION = "activation_contraction"

MIN_ACTIVATION_REDUCTION = 16

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
    from _activation_contractions import activation_reduction_minimum

    try:
        activation_reduction_minimum(recipe)
    except ValueError as error:
        raise RecipeError(str(error)) from error
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


def _spec(tensor: Mapping[str, Any], *, is_weight: bool, eps: float, channel_axis: int = 0):
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
        observer, axis = PerChannelMinMaxObserver, channel_axis
        # PT2E forwards ch_axis only to observers named "PerChannel".
        # Bind custom observers' axis in their constructor as well.
        arguments["ch_axis"] = axis
        group = tensor.get("channels_per_scale")
        if group is not None:
            if type(group) is not int or group < 1 or not symmetric:
                raise RecipeError("channels_per_scale needs a positive integer and symmetric channel scales")
            import _recipe_observers as RO  # noqa: PLC0415 -- a sibling, by bare name

            observer, arguments["channels_per_scale"] = RO.grouped_channel_observer(), group
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
