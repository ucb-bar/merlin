"""Build a physical-layout graph from provenance-annotated linalg-on-tensors IR.

The extractor is policy driven: semantic operation names and layout capabilities are supplied by
the caller.  It intentionally keeps only activation tensors of the configured rank.  Constant
weight transforms and compiler-created ``empty``/``splat`` initializers are not activation edges.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Block, Operation, SSAValue

from ..lowering.physical_layout import LayoutGraph, LayoutOp, LayoutValue


@dataclass(frozen=True)
class RegionLayoutPolicy:
    """Caller-provided semantic and physical-layout capabilities."""

    layouts: tuple[str, ...]
    canonical_layout: str
    preferred_accelerator_layout: str
    accelerator_ops: frozenset[str]
    preserving_ops: frozenset[str]
    residual_ops: frozenset[str]
    view_ops: frozenset[str]
    fixed_layout_ops: frozenset[str]
    weight_ops: frozenset[str] = frozenset()
    activation_rank: int = 4


def _text_attr(op: Operation, name: str) -> str:
    return str(getattr(op.attributes.get(name), "data", "") or "")


def _shape(value: SSAValue) -> tuple[int, ...]:
    ty = value.type
    if not isinstance(ty, TensorType):
        return ()
    return tuple(int(dim) for dim in ty.get_shape())


def _dtype(value: SSAValue) -> str:
    ty = value.type
    return str(ty.get_element_type()) if isinstance(ty, TensorType) else str(ty)


def _entry_block(module) -> Block:
    for op in module.walk():
        if op.name == "func.func":
            return op.regions[0].blocks[0]
    raise ValueError("module has no func.func entry")


def _semantic(ops: list[Operation]) -> str:
    names = [_text_attr(op, "prov.op") for op in ops if _text_attr(op, "prov.op")]
    if not names:
        return ""
    counts = Counter(names)
    # Most frequent is the enclosing semantic op.  First occurrence resolves only exact ties.
    return max(dict.fromkeys(names), key=lambda name: (counts[name], -names.index(name)))


def _is_helper_producer(value: SSAValue) -> bool:
    owner = value.owner
    return isinstance(owner, Operation) and owner.name in {
        "arith.constant", "tensor.empty", "tensor.splat",
    }


def _safe_view(inputs: list[SSAValue], outputs: list[SSAValue]) -> bool:
    """Prove a view leaves logical activation axes unchanged.

    Equal rank alone is insufficient: flatten/unflatten or axis reassociation changes what a
    contiguous NHWC buffer means.  The bounded proof accepts only one-to-one, shape-identical views.
    A richer affine-axis proof can extend this without weakening the fail-closed behavior.
    """
    return (len(inputs) == 1 and len(outputs) == 1
            and _shape(inputs[0]) == _shape(outputs[0]))


def extract_region_layout_graph(module, policy: RegionLayoutPolicy) -> tuple[LayoutGraph, dict[str, Any]]:
    """Extract a target-neutral graph and an auditable structural census."""
    block = _entry_block(module)
    top_ops = [op for op in block.ops if op.name != "func.return"]
    regions: OrderedDict[str, list[Operation]] = OrderedDict()
    region_of: dict[Operation, str] = {}
    for op in top_ops:
        region = _text_attr(op, "prov.region_id")
        if region:
            regions.setdefault(region, []).append(op)
            region_of[op] = region
    semantic_of = {name: _semantic(ops) for name, ops in regions.items()}

    names: dict[SSAValue, str] = {}
    for index, arg in enumerate(block.args):
        names[arg] = f"arg{index}"
    for op_index, op in enumerate(block.ops):
        for result_index, result in enumerate(op.results):
            names[result] = f"v{op_index}_{result_index}"

    # The first non-weight rank-4 entry argument supplies the capture's batch extent.  This is
    # not a baked batch=1 assumption and prevents rank-4 OIHW weights entering the activation graph.
    batch_extent: int | None = None
    for arg in block.args:
        shape = _shape(arg)
        if len(shape) != policy.activation_rank:
            continue
        consumers = []
        for use in arg.uses:
            region = region_of.get(use.operation, "")
            consumers.append(semantic_of.get(region, ""))
        if consumers and all(item in policy.weight_ops for item in consumers):
            continue
        batch_extent = shape[0]
        break
    if batch_extent is None:
        raise ValueError("could not derive an activation batch extent from the entry signature")

    def is_activation(value: SSAValue) -> bool:
        shape = _shape(value)
        if len(shape) != policy.activation_rank or shape[0] != batch_extent:
            return False
        owner = value.owner
        if isinstance(owner, Operation):
            region = region_of.get(owner, "")
            if semantic_of.get(region, "") in policy.weight_ops:
                return False
        return True

    graph_values: dict[str, LayoutValue] = {}

    def add_value(value: SSAValue) -> str:
        name = names[value]
        graph_values.setdefault(name, LayoutValue(name, _shape(value), _dtype(value)))
        return name

    layout_ops: list[LayoutOp] = []
    included_by_kind: Counter[str] = Counter()
    omitted_regions: Counter[str] = Counter()

    for region, ops in regions.items():
        owned = set(ops)
        inputs: list[SSAValue] = []
        outputs: list[SSAValue] = []
        for op in ops:
            for value in op.operands:
                if ((isinstance(value.owner, Block) or value.owner not in owned)
                        and value not in inputs and is_activation(value)
                        and not _is_helper_producer(value)):
                    inputs.append(value)
            for value in op.results:
                if (value not in outputs and is_activation(value)
                        and any(use.operation not in owned for use in value.uses)):
                    outputs.append(value)
        semantic = semantic_of[region]
        if semantic in policy.weight_ops or not (inputs or outputs):
            omitted_regions[semantic or "unclassified"] += 1
            continue

        input_names = tuple(add_value(value) for value in inputs)
        output_names = tuple(add_value(value) for value in outputs)
        supported = policy.layouts
        preferred: str | None = None
        couple = False
        kind = "fixed_boundary"
        reason = "semantic operator has no declared layout-polymorphic lowering"

        if semantic in policy.accelerator_ops:
            kind = "accelerator"
            couple = bool(input_names and output_names)
            preferred = policy.preferred_accelerator_layout
            reason = "accelerator activation ports share one selectable physical layout"
        elif semantic in policy.residual_ops:
            kind = "residual"
            couple = len(input_names) >= 2 and bool(output_names)
            reason = "residual inputs and output must agree physically"
        elif semantic in policy.preserving_ops:
            kind = "layout_preserving"
            couple = bool(input_names and output_names)
            reason = "element indexing is independent of physical activation axis order"
        elif semantic in policy.view_ops and _safe_view(inputs, outputs):
            kind = "layout_preserving_view"
            couple = True
            reason = "shape-identical one-to-one view preserves logical axes"
        else:
            supported = (policy.canonical_layout,)
            if semantic in policy.view_ops:
                kind = "unsafe_view_boundary"
                reason = "view changes rank or shape and has no proved physical-axis mapping"
            elif semantic in policy.fixed_layout_ops:
                kind = "fixed_boundary"
                reason = "current lowering is fixed to the canonical activation layout"
            else:
                kind = "unknown_boundary"
                reason = "operation is not classified by the supplied layout policy"

        layout_ops.append(LayoutOp(
            name=region,
            kind=kind,
            inputs=input_names,
            outputs=output_names,
            couple=couple,
            supported_layouts=supported,
            preferred_layout=preferred,
            reason=reason,
        ))
        included_by_kind[kind] += 1

    # An unattributed tensor op is never silently treated as transparent.  This catches missing
    # provenance on a reduction (or a newly introduced operation) as an explicit conversion boundary.
    unattributed: list[str] = []
    for op_index, op in enumerate(block.ops):
        if op in region_of or op.name in {
            "func.return", "arith.constant", "tensor.empty", "tensor.splat",
        }:
            continue
        inputs = [value for value in op.operands if is_activation(value)
                  and not _is_helper_producer(value)]
        outputs = [value for value in op.results if is_activation(value)]
        if not (inputs or outputs):
            continue
        name = f"unattributed_{op_index}_{op.name}"
        layout_ops.append(LayoutOp(
            name=name,
            kind="unattributed_boundary",
            inputs=tuple(add_value(value) for value in inputs),
            outputs=tuple(add_value(value) for value in outputs),
            supported_layouts=(policy.canonical_layout,),
            reason="activation operation has no provenance/layout contract",
        ))
        included_by_kind["unattributed_boundary"] += 1
        unattributed.append(name)

    # ABI entry and exit storage are explicit boundaries, not implicit hard constraints on every
    # internal value.  This lets one conversion feed an entire accelerator-preferred chain.
    for index, arg in enumerate(block.args):
        if arg not in names or names[arg] not in graph_values:
            continue
        layout_ops.append(LayoutOp(
            name=f"graph_input_{index}",
            kind="graph_input",
            outputs=(names[arg],),
            supported_layouts=(policy.canonical_layout,),
            reason="entry ABI declares canonical physical layout",
        ))
        included_by_kind["graph_input"] += 1
    returns = [op for op in block.ops if op.name == "func.return"]
    if returns:
        for index, value in enumerate(returns[0].operands):
            if names.get(value) not in graph_values:
                continue
            layout_ops.append(LayoutOp(
                name=f"graph_output_{index}",
                kind="graph_output",
                inputs=(names[value],),
                supported_layouts=(policy.canonical_layout,),
                reason="result ABI declares canonical physical layout",
            ))
            included_by_kind["graph_output"] += 1

    semantic_counts = Counter(semantic_of.values())
    graph = LayoutGraph(policy.layouts, policy.canonical_layout, graph_values, tuple(layout_ops))
    census = {
        "schema": "provenance_region_layout_census_v1",
        "batch_extent": batch_extent,
        "activation_rank": policy.activation_rank,
        "top_level_operations": len(list(block.ops)),
        "provenance_regions": len(regions),
        "semantic_region_counts": dict(sorted(semantic_counts.items())),
        "layout_graph_values": len(graph_values),
        "layout_graph_ops": len(layout_ops),
        "included_op_counts": dict(sorted(included_by_kind.items())),
        "omitted_region_counts": dict(sorted(omitted_regions.items())),
        "unattributed_activation_ops": unattributed,
    }
    return graph, census
