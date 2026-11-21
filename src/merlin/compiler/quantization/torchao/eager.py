"""TorchAO module configurations and exact layer inventory/placement adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .calibration import _weight_rounding
from .portable import helper

_specs = helper("_recipe_specs")
RecipeError = _specs.RecipeError
_require = _specs._require


_QUANTIZE_TRANSFORMS = ("Linear", "LazyLinear")
FRAMEWORK_CANNOT_EXPRESS = "framework_cannot_express"


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
    capture venv reads the same module the derivation side tests. The portable
    loader binds its exact selected source file without extending sys.path.
    """
    QLP = helper("quant_layer_plan")

    return QLP.plan(
        recipe, layer_inventory(model) if source_layer_inventory is None else source_layer_inventory
    ).to_dict()
