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

There is one quantizer for every target. A target changes the recipe, never this file, which
carries no target name, no scheme name and no merlin import.

WHOSE LIMIT REFUSED A LAYER. :mod:`quant_layer_plan` decides what the TARGET absorbs; this file
adds what this FRAMEWORK BUILD can express of that decision, and the two are recorded under
different codes. A Conv2d the hardware absorbs perfectly well is still not something ``quantize_``
transforms, and filing that as a hardware gap would send someone to read RTL about a Python
dispatch table.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

RECIPE_SCHEMA = "quant_recipe_v1"

#: Recipe dtype -> the torch dtype a spec is built with. A dtype absent here is refused, not mapped
#: to a neighbour: the recipe states the hardware's element format and a near miss is a different
#: datapath.
_TORCH_DTYPE = {"int8": "int8", "fp8_e4m3": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2"}


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


def _spec(tensor: Mapping[str, Any], *, is_weight: bool, eps: float):
    """One ``QuantizationSpec`` from one recipe tensor spec."""
    import torch
    from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
    from torchao.quantization.pt2e.quantizer import QuantizationSpec

    granularity, symmetric = tensor.get("granularity"), bool(tensor.get("symmetric"))
    if granularity == "tensor":
        qscheme = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
        # A weight is observed once, so its range is its min and max; an activation is observed
        # over a calibration stream, where a histogram is robust to a rare outlier.
        observer = {"minmax": MinMaxObserver, "histogram": HistogramObserver}.get(
            str(tensor.get("observer") or ("minmax" if is_weight else "histogram"))
        )
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
        dtype=_torch_dtype(tensor.get("dtype")),
        observer_or_fake_quant_ctr=observer.with_args(eps=eps),
        quant_min=tensor.get("quant_min"),
        quant_max=tensor.get("quant_max"),
        qscheme=qscheme,
        ch_axis=axis,
    )


def build_quantizer(recipe: Mapping[str, Any], *, eps: float = 2**-12):
    """The PT2E quantizer a static recipe describes."""
    import torch
    from torchao.quantization.pt2e.quantizer import QuantizationAnnotation, Quantizer

    weight, activation = _require(recipe)
    if activation.get("mode") != "static":
        raise RecipeError("build_quantizer realises static recipes; a dynamic one goes through quantize_config")
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
        "operand_sum": (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor),
        "window_mean": (
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.ops.aten.avg_pool2d.default,
            torch.ops.aten.mean.dim,
        ),
    }
    listed = tuple(recipe.get("families") or ())
    targets = {op for family in listed if family == "contraction" for op in operators.get(family, ())}
    sums = set(operators["operand_sum"]) if "operand_sum" in listed else set()
    means = set(operators["window_mean"]) if "window_mean" in listed else set()
    if not targets:
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
            if node.target in sums:
                operands = [a for a in node.args[:2] if isinstance(a, torch.fx.Node) and a.op != "get_attr"]
                if len(operands) == 2:
                    candidates.add(node)
            elif node.target in means and node.args and isinstance(node.args[0], torch.fx.Node):
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
                if not (seen := readers(node)) or any(r.target not in targets and r not in candidates for r in seen)
            }
            if not dropped:
                return candidates
            candidates -= dropped

    class CapabilityQuantizer(Quantizer):
        def __init__(self) -> None:
            self.activation = _spec(activation, is_weight=False, eps=eps)
            self.weight = _spec(weight, is_weight=True, eps=eps)
            self.annotated = 0
            self.annotated_sums = 0
            self.annotated_means = 0
            self.left_in_float = 0

        def annotate(self, graph_module: Any) -> Any:
            nodes = list(graph_module.graph.nodes)
            kept = closable(nodes)
            self.left_in_float = sum(1 for n in nodes if (n.target in sums or n.target in means) and n not in kept)
            for node in nodes:
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
                if node.target not in targets or len(node.args) < 2:
                    continue
                operand, kernel = node.args[:2]
                # Both inputs of the contraction; its result stays in floating point and the next
                # contraction quantizes its own input edge. What follows a contraction is absorbed
                # by the compiler's group formation, not assumed here.
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={operand: self.activation, kernel: self.weight}, _annotated=True
                )
                self.annotated += 1
            return graph_module

        def validate(self, graph_module: Any) -> None:
            del graph_module
            # COUNT EVERY FAMILY THE RECIPE LISTS, not just contractions. Three counters are kept and
            # this read one, so a model carrying only the operand sums or window means a recipe also
            # covers was refused with a message saying its families were not covered -- which was
            # false, and pointed a reader at the model rather than at this table.
            if self.annotated + self.annotated_sums + self.annotated_means == 0:
                raise RecipeError(
                    f"the model has no operator the recipe's families cover "
                    f"(families={list(listed)}; annotated contractions=0, sums=0, means=0)"
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
            continue  # a container's arithmetic is its children's
        weight = getattr(module, "weight", None)
        layers.append(
            {
                "fqn": fqn,
                "kind": type(module).__name__,
                "weight_shape": (list(weight.shape) if weight is not None and hasattr(weight, "shape") else None),
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
    mapping: "OrderedDict[str, Any]" = OrderedDict()
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


def _plan_layers(recipe: Mapping[str, Any], model: Any) -> dict[str, Any]:
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

    return QLP.plan(recipe, layer_inventory(model)).to_dict()


def _as_tuple(sample: Any) -> tuple:
    return tuple(sample) if isinstance(sample, (tuple, list)) else (sample,)


def apply_recipe(
    model: Any,
    recipe: Mapping[str, Any],
    *,
    example_inputs: tuple,
    calibration_inputs: Iterable[Any] | None = None,
    calibration_samples: int = 100,
) -> Any:
    """Quantize ``model`` under ``recipe`` and return the module the capture should lower."""
    import torch

    _weight, activation = _require(recipe)
    # A MODEL ALREADY IN THE RECIPE'S INTEGERS HAS NOTHING TO CALIBRATE. An observer watches a
    # floating-point activation to decide the scale that maps it onto a grid; handed a tensor already
    # on that grid there is no range to learn, and the question the recipe asks does not apply.
    #
    # This crashed rather than saying so. `HistogramObserver` calls `torch.histc`, which has no int8
    # kernel, so the capture died with `NotImplementedError: "histogram_cpu" not implemented for
    # 'Char'` -- an error about a missing torch kernel, several layers below the actual mistake.
    # Measured on `M3_host_island_seam_gemmini`, whose loader hands out a single `torch.int8` input
    # because the capsule's arithmetic IS integer: it is hand-written to be the integer seam, and the
    # recipe path regressed a capsule that had generated before recipes existed.
    #
    # Skipped rather than refused, and recorded: the model is already the thing the recipe would have
    # produced, so the capture should proceed and say that it did nothing, which is a fact a reader of
    # the capsule's provenance needs. A refusal here would be correct about the category error and
    # wrong about what to do next.
    tensors = [t for t in (example_inputs or ()) if isinstance(t, torch.Tensor)]
    if tensors and not any(torch.is_floating_point(t) for t in tensors):
        model._recipe_quantization_stats = {  # type: ignore[attr-defined]
            "applied": False,
            "why": (
                "every example input is already integral "
                f"({sorted({str(t.dtype) for t in tensors})}), so there is no floating-point range for "
                "an observer to learn; the model is already expressed on the recipe's grid"
            ),
        }
        return model
    if activation.get("mode") == "dynamic":
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
    exported = torch.export.export(model.eval(), tuple(example_inputs)).module()
    quantizer = build_quantizer(recipe)
    prepared = prepare_pt2e(exported, quantizer)
    samples = calibration_inputs if calibration_inputs is not None else (tuple(example_inputs),)
    calibrated = 0
    with torch.no_grad():
        for sample in samples:
            if calibrated >= calibration_samples:
                break
            prepared(*_as_tuple(sample))
            calibrated += 1
    if calibrated == 0:
        raise RecipeError("static calibration saw no input sample")
    quantized = convert_pt2e(prepared, use_reference_representation=False, fold_quantize=True)
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
        "annotated_contractions": quantizer.annotated,
        "annotated_sums": quantizer.annotated_sums,
        "annotated_means": quantizer.annotated_means,
        # Sums and means the recipe's families cover and the graph does not requantize: left alone.
        "sums_and_means_left_in_float": quantizer.left_in_float,
        "calibration_samples": calibrated,
        "pruned_dead_state_tensors": pruned,
        "weight_granularity": recipe["weight"]["granularity"],
        "activation_granularity": recipe["activation"]["granularity"],
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
