"""PT2E quantizer annotation and dynamic-family realization methods."""

from __future__ import annotations

import json
from typing import Any

from .graphs import _epilogue_of, _planes_site, _planes_spec
from .portable import helper

_specs = helper("_recipe_specs")
ACTIVATION_CONTRACTION = _specs.ACTIVATION_CONTRACTION
RecipeError = _specs.RecipeError
_spec = _specs._spec


def capability_quantizer_type(
    Quantizer,
    *,
    QuantizationAnnotation,
    activation,
    closable,
    dynamic_block,
    dynamic_families,
    eligible,
    eps,
    family_of,
    fold_candidates,
    listed,
    means,
    owner,
    planes_block,
    quantized_reader,
    recipe,
    shared_sums,
    software_decisions,
    sums,
    targets,
    torch,
    weight,
):
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
            # The recipe's output-path epilogue (_integer_epilogue): contraction + absorbed stages as one pattern.
            self.epilogue = _epilogue_of(recipe, self.activation)
            self.shared_sums = shared_sums
            extensions = helper("_recipe_extensions")

            self.extensions = extensions.Extensions(recipe, self.epilogue)

        def rewrite_dynamic(
            self, graph_module: Any, samples: Any = None, limit: int | None = None
        ) -> list[dict[str, Any]]:
            """Rewrite the dynamic families' placed contractions into their integer form, in place.

            Runs before PT2E sees the graph, so a rewritten contraction is never also annotated. One the
            rewrite cannot express is left in floating point and listed, never approximated. Under the
            block's ``static_heads`` scales the activation contractions are calibrated on ``samples``.
            """
            if not dynamic_families:
                return []
            DC = helper("_dynamic_contractions")  # noqa: PLC0415 -- a sibling, by bare name

            if dynamic_block.get("scales") == "static_heads":
                original_nodes = set(graph_module.graph.nodes)
                rows = self._rewrite_static_heads(graph_module, samples, limit)
                if shared_sums is not None:
                    for node in graph_module.graph.nodes:
                        if node not in original_nodes:
                            node.meta["merlin_static_attention_owned"] = True
                return rows
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

        def _rewrite_static_heads(self, graph_module: Any, samples: Any, limit: int | None) -> list[dict[str, Any]]:
            SA = helper("_static_attention")  # noqa: PLC0415 -- a sibling, by bare name

            if dynamic_families != {ACTIVATION_CONTRACTION} or samples is None:
                raise RecipeError("static per-head scales cover activation contractions, calibrated on samples")
            if iter(samples) is samples:
                raise RecipeError("static per-head scales read the calibration stream before PT2E does; pass a list")
            samples = list(samples)[:limit]
            sites = [
                n
                for n in graph_module.graph.nodes
                if n.op == "call_function"
                and family_of(n) == ACTIVATION_CONTRACTION
                and eligible(n, ACTIVATION_CONTRACTION)
            ]
            policy = recipe.get("framework_capture_policy") or {}
            try:
                self.dynamic_rows = SA.rewrite_sites(
                    graph_module,
                    sites,
                    dynamic_block,
                    samples,
                    low=activation["quant_min"],
                    high=activation["quant_max"],
                    eps=policy.get("observer_epsilon", eps),
                    owner=owner,
                )
            except ValueError as error:
                raise RecipeError(f"static per-head scales: {error}") from error
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
            if self.epilogue is not None:  # after Conv+BN folding, before any observer sees a weight
                self.epilogue.equalize(
                    graph_module, lambda n: family_of(n) == "contraction" and eligible(n, "contraction")
                )
            return graph_module

        def annotate(self, graph_module: Any) -> Any:
            nodes = list(graph_module.graph.nodes)
            kept = closable(nodes)
            self.extensions.select(
                nodes,
                lambda n: "contraction" not in dynamic_families and eligible(n, "contraction"),
                quantized_reader,
            )
            if shared_sums is not None:
                shared_sums.select(kept)
            self.left_in_float = sum(
                1
                for n in nodes
                if not self.extensions.owns(n)
                and (n.target in sums or n.target in means)
                and (n not in kept or (n.target in sums and shared_sums is not None and n not in shared_sums.selected))
            )
            for node in nodes:
                if self.extensions.owns(node):
                    continue  # the owning contraction annotates the residual and its skip edge
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
                    if shared_sums is not None:
                        self.annotated_sums += int(shared_sums.annotate(node, self.activation))
                        continue
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
                    axis = kernel.meta["val"].dim() - 1
                    # PT2E forwards ch_axis only to observers whose class name
                    # contains "PerChannel". Custom grouped observers also need
                    # the axis bound in their constructor; the spec alone leaves
                    # their observer on its default reduction axis.
                    stored = _spec(weight, is_weight=True, eps=eps, channel_axis=axis)
                if self.epilogue is not None and not planes and node.target in self.epilogue.contractions:
                    self.epilogue.annotate(node, stored, quantized_reader, owner(node))
                    self.annotated += 1
                    continue
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

    return CapabilityQuantizer
