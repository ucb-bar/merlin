"""Build the placed PT2E quantizer and preserve exact source ownership eligibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .annotation import capability_quantizer_type
from .calibration import _weight_rounding
from .graphs import _planes_site, _source_weight_geometry_matches
from .portable import helper

_specs = helper("_recipe_specs")
DEFAULT_OBSERVER_EPSILON = _specs.DEFAULT_OBSERVER_EPSILON
ACTIVATION_CONTRACTION = _specs.ACTIVATION_CONTRACTION
RecipeError = _specs.RecipeError
_require = _specs._require
_recipe_operators = _specs._recipe_operators


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
    shared_sums = None
    if "operand_sum_grid" in recipe:
        IS = helper("_integer_operand_sum")  # noqa: PLC0415 -- capture interpreter sibling

        shared_sums = IS.SharedGridSums(recipe)
    # Families the recipe's ``dynamic`` block takes out of static annotation (_dynamic_contractions).
    activation_reduction_minimum = helper("_activation_contractions").activation_reduction_minimum

    minimum_reduction = activation_reduction_minimum(recipe)
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
        from merlin.development.targetgen.software_spec import admit_operation

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
        if family == ACTIVATION_CONTRACTION and int(node.args[0].meta["val"].shape[-1]) < minimum_reduction:
            # A small reduction is a historical profitability heuristic, not an
            # unsupported instruction. An explicit normalized P×V recipe must
            # also cover short prefill/decode contexts. Keep unrelated outer
            # products (e.g. rotary frequencies) on their existing path.
            DC = helper("_dynamic_contractions")

            if not (dynamic_block.get("context_output") == "int8_value_grid" and DC.softmax_produced(node.args[0])):
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
        """Find closed sums/means, plus explicitly authorized V2 float exits.

        Putting a sum on the integer grid is worth its accuracy cost only when the result is
        requantized, because that is what closes it into one integer region a unit can take. A sum
        read by a normalization, or returned, stays a float add whose operands were quantized for
        nothing (measured on a two-layer decoder: 8 sums and 5 means annotated, none placeable,
        17 more host regions). A reader may itself be a candidate, so this is a fixpoint: drop
        every candidate with a reader that is neither a contraction nor a surviving candidate.
        Explicit shared-grid V2 instead supplies a dequantized boundary for checked sums;
        it does not change means or float consumers.
        """
        candidates = set()
        for node in nodes:
            # Calibrated attention plane recombination belongs to its numerical
            # policy, not to the source residual family. PT2E preserves FX meta.
            if shared_sums is not None and node.meta.get("merlin_static_attention_owned"):
                continue
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

        # Only the explicit V2 shared-grid policy authorizes a quantized exit
        # followed by a float boundary. Preview the SAME strict source selector
        # (dtype, equal shape, alpha, mutation/alias rules) before exempting sums;
        # means and unsupported adds retain ordinary closure requirements.
        float_exits = set()
        if shared_sums is not None and shared_sums.allow_float_boundary:
            shared_sums.select(candidates)
            float_exits = {node for node in shared_sums.selected if readers(node)}
        while True:
            dropped = {
                node
                for node in candidates
                if node not in float_exits
                and (
                    not (seen := readers(node))
                    or any(
                        (r.target not in targets or not eligible(r, "contraction")) and r not in candidates
                        for r in seen
                    )
                )
            }
            if not dropped:
                return candidates
            candidates -= dropped

    def quantized_reader(reader: Any, source: Any) -> bool:
        """Whether ``reader`` quantizes ``source`` on the activation grid at its input (an epilogue's exit)."""
        if shared_sums is not None and shared_sums.quantizes(reader, source):
            return True
        family = family_of(reader) if reader.op == "call_function" else None
        if family == "contraction" and family not in dynamic_families and reader.args and reader.args[0] is source:
            return eligible(reader, family) and not (planes_block and _planes_site(planes_block, reader, owner(reader)))
        return family == ACTIVATION_CONTRACTION and family not in dynamic_families and source in reader.args[:2]

    return capability_quantizer_type(
        Quantizer,
        QuantizationAnnotation=QuantizationAnnotation,
        activation=activation,
        closable=closable,
        dynamic_block=dynamic_block,
        dynamic_families=dynamic_families,
        eligible=eligible,
        eps=eps,
        family_of=family_of,
        fold_candidates=fold_candidates,
        listed=listed,
        means=means,
        owner=owner,
        planes_block=planes_block,
        quantized_reader=quantized_reader,
        recipe=recipe,
        shared_sums=shared_sums,
        software_decisions=software_decisions,
        sums=sums,
        targets=targets,
        torch=torch,
        weight=weight,
    )()
