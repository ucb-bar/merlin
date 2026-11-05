"""Opt-in residuals on one observed activation grid, inside the capture interpreter.

Both inputs and the output share one PT2E observer. After conversion the add
is expressed on its unit grid, just like an integer contraction epilogue:
``clamp(a_i8 + b_i8)`` (optionally ReLU), followed by the original output
dequantization. Integer additions are exact in the f32 capture carrier. A
compiler can therefore use unit load/readout multipliers without introducing
the separately rounded, arbitrary-ratio load approximation.

This changes the quantized model. It establishes no quality, placement or
target execution verdict; ordinary independent checks remain necessary.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from typing import Any

SCHEMA = "shared_grid_operand_sum_v1"
FLOAT_BOUNDARY_SCHEMA = "shared_grid_operand_sum_v2"
KEY = "operand_sum_grid"
DISTINCT_FRONTIERS = "distinct_contraction_frontiers"


def _float_boundary(policy: Any) -> bool:
    return policy in (
        {"schema": FLOAT_BOUNDARY_SCHEMA, "output_boundary": "dequantize"},
        {"schema": FLOAT_BOUNDARY_SCHEMA, "output_boundary": "dequantize", "selection": DISTINCT_FRONTIERS},
    )


class OperandSumError(ValueError):
    """The explicitly selected shared-grid arithmetic cannot be represented."""


def _disjoint_static_attention(recipe: Mapping[str, Any]) -> bool:
    """Validate a separate calibrated attention family without rewriting its policy."""
    from merlin.targetgen import quant_recipe as QR

    block = recipe.get("dynamic")
    if (
        not _float_boundary(recipe.get(KEY))
        or not isinstance(block, Mapping)
        or set(block) - {"schema", "families", "scales", "planes", "smooth", "site_agreement"}
        or block.get("schema") != QR.DYNAMIC_SCHEMA
        or block.get("families") != [QR.ACTIVATION_CONTRACTION]
        or block.get("scales") != QR.STATIC_HEADS
        or block.get("smooth") not in (None, QR.SOFTMAX_AXIS)
        or not isinstance(block.get("planes", {}), Mapping)
    ):
        return False
    try:
        # Reuse the existing attention policy validator, including allowed plane
        # selectors/counts, rather than inventing a second numerical contract.
        QR.with_dynamic_contractions(
            recipe, families=block["families"], scales=block["scales"],
            planes=block.get("planes"), smooth=block.get("smooth") == QR.SOFTMAX_AXIS,
        )
        QR.site_agreement_of(recipe)
    except (TypeError, ValueError):
        return False
    return True


def validate(recipe: Mapping[str, Any]) -> None:
    """Check the closed opt-in policy without importing a framework."""
    if recipe.get(KEY) != {"schema": SCHEMA} and not _float_boundary(recipe.get(KEY)):
        raise OperandSumError("operand_sum_grid needs an exact V1 closed policy or V2 dequantize boundary")
    activation = recipe.get("activation") or {}
    if (
        recipe.get("schema") != "quant_recipe_v1"
        or recipe.get("status") != "derived"
        or "operand_sum" not in (recipe.get("families") or ())
        or recipe.get("accumulator_dtype") != "int32"
        or activation.get("dtype") != "int8"
        or activation.get("mode") != "static"
        or activation.get("granularity") != "tensor"
        or activation.get("symmetric") is not True
        or type(activation.get("quant_min")) is not int
        or type(activation.get("quant_max")) is not int
        or [activation["quant_min"], activation["quant_max"]] != [-128, 127]
        or (recipe.get("dynamic") and not _disjoint_static_attention(recipe))
        or recipe.get("static_planes")
    ):
        raise OperandSumError("shared-grid sums require static symmetric per-tensor signed i8/i32 arithmetic")


class SharedGridSums:
    """One quantizer's selected sites; no state is shared between captures."""

    def __init__(self, recipe: Mapping[str, Any]):
        validate(recipe)
        self.allow_float_boundary = recipe[KEY].get("output_boundary") == "dequantize"
        self.residual_selection = recipe[KEY].get("selection")
        self.selected: dict[Any, Any] = {}
        self.sites: list[dict[str, Any]] = []
        self.rewritten: list[dict[str, Any]] = []

    @staticmethod
    def _fresh(value: Any) -> bool:
        """A schema-proved newly allocated Tensor, never a placeholder or view."""
        import torch

        schema = getattr(getattr(value, "target", None), "_schema", None)
        returns = getattr(schema, "returns", ())
        return (
            getattr(value, "op", None) == "call_function"
            and isinstance(value.target, torch._ops.OpOverload)
            and value.target.namespace == "aten"
            and not schema.is_mutable
            and len(returns) == 1
            and returns[0].alias_info is None
            and returns[0].type.kind() == "TensorType"
        )

    def select(self, candidates: Any) -> None:
        """Select placed, equal-shape f32 sums; closure policy is decided by the caller."""
        import torch

        # This is a recipe-selection filter, not a proof of float equivalence:
        # layout/elementwise branches of one contraction should not be treated
        # as separate residual producers. Stop at explicit contraction nodes or
        # graph inputs, ignore stored constants, and propagate all other dataflow
        # edges. An unrecognized contraction is conservatively traversed instead.
        contractions = {
            "aten.linear.default", "aten.mm.default", "aten.bmm.default", "aten.matmul.default",
            "aten.addmm.default", "aten.convolution.default", "aten._convolution.default",
            "aten.conv1d.default", "aten.conv2d.default", "aten.conv3d.default",
        }
        frontiers: dict[Any, frozenset[Any]] = {}

        def frontier(value: Any) -> frozenset[Any]:
            pending = [value]
            while pending:
                current = pending[-1]
                if current in frontiers:
                    pending.pop()
                elif current.op == "get_attr":
                    frontiers[current] = frozenset()
                elif current.op == "placeholder" or str(current.target) in contractions:
                    frontiers[current] = frozenset((current,))
                else:
                    missing = [n for n in current.all_input_nodes if n not in frontiers]
                    if missing:
                        pending.extend(missing)
                    else:
                        frontiers[current] = frozenset().union(*(frontiers[n] for n in current.all_input_nodes))
            return frontiers[value]

        self.selected.clear()
        self.sites.clear()
        self.rewritten.clear()
        for node in candidates:
            if node.target not in (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor) or len(node.args) != 2:
                continue
            if node.target == torch.ops.aten.add_.Tensor and (
                not self._fresh(node.args[0]) or list(node.args[0].users) != [node]
            ):
                continue
            alpha = node.kwargs.get("alpha", 1)
            if type(alpha) not in (int, float) or alpha != 1 or set(node.kwargs) - {"alpha"}:
                continue
            values = [arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None for arg in node.args]
            if any(arg.op == "get_attr" for arg in node.args if isinstance(arg, torch.fx.Node)):
                continue
            if not all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 for value in values):
                continue
            if not values[0].shape or values[0].shape != values[1].shape:
                continue
            if self.residual_selection == DISTINCT_FRONTIERS:
                left, right = (frontier(arg) for arg in node.args)
                if not left or not right or not left.isdisjoint(right):
                    continue
                if not any(str(n.target) in contractions for n in left | right):
                    continue
            users = list(node.users)
            last = (
                users[0]
                if len(users) == 1 and users[0].target in (torch.ops.aten.relu.default, torch.ops.aten.relu_.default)
                else node
            )
            self.selected[node] = last

    def quantizes(self, reader: Any, source: Any) -> bool:
        return reader in self.selected and source in reader.args

    def annotate(self, node: Any, activation: Any) -> bool:
        import torch
        from torchao.quantization.pt2e.quantizer import QuantizationAnnotation, SharedQuantizationSpec

        if node not in self.selected:
            return False
        left, right = node.args
        shared = SharedQuantizationSpec((left, node))
        inputs = {left: activation}
        if right is not left:
            inputs[right] = shared
        last = self.selected[node]
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=inputs, output_qspec=shared if last is node else None, _annotated=True
        )
        if last is not node:
            last.meta["quantization_annotation"] = QuantizationAnnotation(output_qspec=shared, _annotated=True)
        self.sites.append(
            {
                "node": node.name,
                "last": last.name,
                "relu": last is not node,
                "mutable": node.target == torch.ops.aten.add_.Tensor,
                "inputs": [left.name, right.name],
                "shape": list(left.meta["val"].shape),
            }
        )
        return True

    @staticmethod
    def _scale(node: Any, *, quantize: bool) -> bytes:
        import torch

        args = node.args
        if (
            len(args) != 6
            or type(args[1]) not in (float, int)
            or type(args[2]) is not int
            or args[2] != 0
            or any(type(n) is not int for n in args[3:5])
            or list(args[3:5]) != [-128, 127]
            or args[5] != torch.int8
            or set(node.kwargs) - ({"out_dtype"} if not quantize else set())
            or (not quantize and node.kwargs.get("out_dtype") not in (None, torch.float32))
        ):
            raise OperandSumError("a selected sum lost its literal symmetric i8/f32 quantization ABI")
        try:
            value = float(args[1])
            bits = struct.pack("<f", value)
        except (ValueError, OverflowError, struct.error) as error:
            raise OperandSumError("the shared scale must be a finite positive f32") from error
        if not math.isfinite(value) or value <= 0 or struct.unpack("<f", bits)[0] != value:
            raise OperandSumError("the shared scale must be a finite positive f32")
        return bits

    def rewrite(self, gm: Any) -> list[dict[str, Any]]:
        """Make every selected converted site unit-grid arithmetic, or refuse."""
        import torch

        dq = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        q = torch.ops.quantized_decomposed.quantize_per_tensor.default
        nodes = {node.name: node for node in gm.graph.nodes}
        rows = []
        plans = []
        for site in self.sites:
            node, last = nodes[site["node"]], nodes[site["last"]]
            if (
                node.target not in (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor)
                or (node.target == torch.ops.aten.add_.Tensor) != site["mutable"]
                or len(node.args) != 2
                or type(node.kwargs.get("alpha", 1)) not in (int, float)
                or node.kwargs.get("alpha", 1) != 1
                or set(node.kwargs) - {"alpha"}
                or (
                    last is not node
                    and (
                        last.target not in (torch.ops.aten.relu.default, torch.ops.aten.relu_.default)
                        or last.args != (node,)
                        or last.kwargs
                        or list(node.users) != [last]
                    )
                )
            ):
                raise OperandSumError("a selected sum changed its add or optional ReLU arithmetic")
            inputs = list(node.args[:2])
            users = list(last.users)
            if not all(isinstance(value, torch.fx.Node) and value.target == dq for value in inputs):
                raise OperandSumError("a selected sum lost its two per-tensor dequantized inputs")
            # PT2E creates DQ nodes without tensor metadata. Reopen each actual
            # Q->source edge instead of trusting invented DQ shapes or executing
            # the full model again merely to populate metadata.
            sources = []
            for value in inputs:
                quantize = value.args[0]
                if not isinstance(quantize, torch.fx.Node) or quantize.target != q:
                    raise OperandSumError("a selected sum lost its observed input quantization edge")
                sources.append(quantize.args[0])
            if [source.name for source in sources] != site["inputs"] or any(
                not isinstance(source.meta.get("val"), torch.Tensor)
                or source.meta["val"].dtype != torch.float32
                or list(source.meta["val"].shape) != site["shape"]
                for source in (*sources, node)
            ):
                raise OperandSumError("a selected sum changed its equal-shape f32 source metadata")
            if node.target == torch.ops.aten.add_.Tensor and (
                not self._fresh(sources[0]) or list(inputs[0].users) != [node]
            ):
                raise OperandSumError("a mutable sum lost its private fresh input")
            if len(users) != 1 or users[0].target != q:
                raise OperandSumError("a selected sum must leave through one quantize")
            output = users[0]
            scale = self._scale(inputs[0], quantize=False)
            if self._scale(inputs[1], quantize=False) != scale or self._scale(output, quantize=True) != scale:
                raise OperandSumError("a selected sum's input and output scales differ")
            if any(self._scale(value.args[0], quantize=True) != scale for value in inputs):
                raise OperandSumError("a selected sum changed an observed input quantization grid")
            if not output.users or any(
                user.target != dq or self._scale(user, quantize=False) != scale for user in output.users
            ):
                raise OperandSumError("a selected sum must restore its original output grid")
            plans.append((site, node, inputs, output, scale))
        # A prior residual's output Q may also be the next residual's input Q.
        # Validate the entire original graph before changing any shared grid.
        for site, node, inputs, output, scale in plans:
            for index, value in enumerate(inputs):
                with gm.graph.inserting_before(node):
                    unit = gm.graph.call_function(dq, (value.args[0], 1.0, *value.args[2:]), value.kwargs)
                unit.meta = dict(value.meta)
                custom = dict(unit.meta.get("custom") or node.meta.get("custom") or {})
                if custom.get("m2m_lineage"):
                    custom["m2m_transform"] = "shared_grid_operand_sum"
                    unit.meta["custom"] = custom
                args = list(node.args)
                args[index] = unit
                node.args = tuple(args)
            output.args = (output.args[0], 1.0, *output.args[2:])
            rows.append({**site, "status": "rewritten", "shared_scale_f32_hex": scale.hex()})
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        self.rewritten = rows
        return rows

    def census(self) -> dict[str, Any]:
        return {
            "schema": FLOAT_BOUNDARY_SCHEMA if self.allow_float_boundary else SCHEMA,
            "annotated": len(self.sites), "rewritten": len(self.rewritten),
            **({"output_boundary": "dequantize", "target_execution_qualified": False}
               if self.allow_float_boundary else {}),
            **({"selection": self.residual_selection} if self.residual_selection else {}),
        }
