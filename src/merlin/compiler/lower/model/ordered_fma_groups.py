"""Source-owned closure analysis for guarded ordered floating contractions.

Groups contain live source operations, including every intermediate cast and
arithmetic boundary. Analysis grants no numerical relaxation, certificate,
physical alias fact, scheduling change or external-call purity. An emitter must
prove the complete endpoint and validate the retained source witness again.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class _Witness:
    operation: object
    parent: object
    operands: tuple
    operand_types: tuple
    result_types: tuple
    result_uses: tuple
    attributes: tuple
    properties: tuple
    region_contents: tuple


@dataclass(frozen=True)
class _ContextWitness:
    operation: object
    parent_block: object
    parent_region: object
    parent_operation: object
    attributes: tuple
    properties: tuple


def _context_snapshot(operation) -> _ContextWitness:
    block = operation.parent
    region = block.parent if block is not None else None
    owner = region.parent if region is not None else None
    return _ContextWitness(
        operation,
        block,
        region,
        owner,
        tuple(sorted(operation.attributes.items())),
        tuple(sorted(operation.properties.items())),
    )


@dataclass(frozen=True)
class OrderedFMAGroup:
    """A source DAG and its exact typed frontier; no duplicate IR is created."""

    contractions: tuple
    operations: tuple
    inputs: tuple
    bf16_outputs: tuple
    live_f32_outputs: tuple
    unsupported_uses: tuple
    _witness: tuple[_Witness, ...] = field(repr=False)
    _contexts: tuple[_ContextWitness, ...] = field(repr=False)

    @property
    def closed_bf16_endpoints(self) -> bool:
        return bool(self.bf16_outputs) and not self.live_f32_outputs and not self.unsupported_uses


def _source_pure_body(operation) -> bool:
    from xdsl.traits import Pure

    for region in operation.regions:
        if len(region.blocks) != 1:
            return False
        for scalar in region.block.ops:
            if scalar.name == "linalg.yield":
                continue
            if scalar.regions or not scalar.has_trait(Pure):
                return False
            # Fast math and strict FP need separate numeric/effect contracts.
            if any(
                "strictfp" in name or "strictfp" in str(value)
                for name, value in (*scalar.attributes.items(), *scalar.properties.items())
            ):
                return False
            for name, value in (*scalar.attributes.items(), *scalar.properties.items()):
                if name == "fastmath" and str(value) != "#arith.fastmath<none>":
                    return False
    return True


def _supported(operation) -> bool:
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

    if any(
        "strictfp" in name or "strictfp" in str(value)
        for name, value in (*operation.attributes.items(), *operation.properties.items())
    ):
        return False
    if any(not isinstance(v.type, TensorType) or any(n <= 0 for n in v.type.get_shape()) for v in operation.results):
        return False
    if isinstance(operation, linalg.GenericOp):
        if (
            len(operation.results) != 1
            or len(operation.outputs) != 1
            or any(x.data != linalg.IteratorType.PARALLEL for x in operation.iterator_types)
            or not _source_pure_body(operation)
        ):
            return False
        maps = [x.data for x in operation.indexing_maps]
        if len(maps) != len(operation.operands):
            return False
        dimensions = len(operation.iterator_types)
        output_map = maps[-1]
        if (
            output_map.num_dims != dimensions
            or output_map.num_symbols
            or len(output_map.results) != dimensions
            or sorted(expr.position for expr in output_map.results if isinstance(expr, AffineDimExpr))
            != list(range(dimensions))
        ):
            return False
        extents = dict(
            zip((expr.position for expr in output_map.results), operation.outputs[0].type.get_shape(), strict=True)
        )
        for affine, value in zip(maps, operation.operands, strict=True):
            if not isinstance(value.type, TensorType) or affine.num_dims != dimensions or affine.num_symbols:
                return False
            if len(affine.results) != len(value.type.get_shape()):
                return False
            for expr, extent in zip(affine.results, value.type.get_shape(), strict=True):
                if isinstance(expr, AffineDimExpr):
                    if extents.get(expr.position) != extent:
                        return False
                elif isinstance(expr, AffineConstantExpr):
                    if not 0 <= expr.value < extent:
                        return False
                else:
                    return False
        return True
    if isinstance(operation, linalg.ReduceOp):
        return _source_pure_body(operation)
    return isinstance(operation, (tensor.ExtractSliceOp, tensor.ExpandShapeOp, tensor.CollapseShapeOp, tensor.CastOp))


def _bf16(value) -> bool:
    from xdsl.dialects.builtin import TensorType, bf16

    return isinstance(value.type, TensorType) and value.type.element_type == bf16


def _f32(value) -> bool:
    from xdsl.dialects.builtin import TensorType, f32

    return isinstance(value.type, TensorType) and value.type.element_type == f32


def _snapshot(operation) -> _Witness:
    return _Witness(
        operation,
        operation.parent,
        tuple(operation.operands),
        tuple(value.type for value in operation.operands),
        tuple(value.type for value in operation.results),
        tuple(frozenset((use.operation, use.index) for use in value.uses) for value in operation.results),
        tuple(sorted(operation.attributes.items())),
        tuple(sorted(operation.properties.items())),
        tuple(
            tuple(
                (block, tuple(block.args), tuple(value.type for value in block.args), tuple(block.ops))
                for block in region.blocks
            )
            for region in operation.regions
        ),
    )


def validate_group_source(group: OrderedFMAGroup) -> None:
    """Refuse changes to retained source operations, uses, types or numerics."""
    from merlin.compiler.lower.model.ordered_fma_rewrite import _match

    if any(_match(operation) is None for operation in group.contractions):
        raise ValueError("ordered FMA group source changed after analysis")
    for context in group._contexts:
        if _context_snapshot(context.operation) != context:
            raise ValueError("ordered FMA group enclosing source context changed after analysis")
    for witness in group._witness:
        if _snapshot(witness.operation) != witness:
            raise ValueError("ordered FMA group source changed after analysis")


def analyze_ordered_fma_groups(module) -> tuple[OrderedFMAGroup, ...]:
    """Close zero-seeded ordered contractions through their source consumers.

    Pure static pointwise generics, registered reductions and tensor views can
    remain inside a group. A BF16 result is an endpoint unless a view-only path
    feeds another matched contraction. Live f32 uses and unsupported operations
    are recorded explicitly. Closure is only a necessary source condition: an
    accelerator provider still owes a complete numerical/effect/ownership proof.
    """
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import TensorType

    from merlin.compiler.lower.model.ordered_fma_rewrite import _match

    module.verify()
    ordered = list(module.walk())
    positions = {op: i for i, op in enumerate(ordered)}
    roots = tuple(op for op in ordered if _match(op) is not None)
    root_set = set(roots)
    views = (tensor.ExtractSliceOp, tensor.ExpandShapeOp, tensor.CollapseShapeOp, tensor.CastOp)
    internal_bf16 = set()
    for root in roots:
        for value in root.inputs:
            while _bf16(value):
                internal_bf16.add(value)
                owner = value.owner
                if not isinstance(owner, views) or not _supported(owner):
                    break
                value = owner.operands[0]
    closures = []
    for root in roots:
        reached = {root}
        pending = list(root.results)
        endpoints = set()
        escapes = set()
        unsupported = set()
        ancestor = root
        strict = False
        while ancestor is not None:
            strict |= any(
                "strictfp" in name or "strictfp" in str(value)
                for name, value in (*ancestor.attributes.items(), *ancestor.properties.items())
            )
            ancestor = ancestor.parent_op()
        if strict or not _source_pure_body(root):
            closures.append((reached, endpoints, set(root.results), {(value, root, -1) for value in root.results}))
            continue
        while pending:
            value = pending.pop()
            if _bf16(value) and value not in internal_bf16:
                endpoints.add(value)
                continue
            for use in value.uses:
                user = use.operation
                if user in reached:
                    continue
                if user not in root_set and not _supported(user):
                    unsupported.add((value, user, use.index))
                    if _f32(value):
                        escapes.add(value)
                    continue
                reached.add(user)
                pending.extend(user.results)
        closures.append((reached, endpoints, escapes, unsupported))
    # Consumers shared by multiple roots define one compound source obligation.
    groups = []
    for closure in closures:
        merged = list(closure)
        index = 0
        while index < len(groups):
            if merged[0] & groups[index][0]:
                other = groups.pop(index)
                merged = [left | right for left, right in zip(merged, other, strict=True)]
                index = 0
            else:
                index += 1
        groups.append(merged)
    result = []
    for reached, endpoints, escapes, unsupported in groups:
        ops = tuple(sorted(reached, key=positions.__getitem__))
        inputs = []
        for operation in ops:
            for value in operation.operands:
                if value.owner not in reached and isinstance(value.type, TensorType) and value not in inputs:
                    inputs.append(value)
        witnesses = tuple(_snapshot(child) for operation in ops for child in operation.walk())
        contexts = {}
        for operation in ops:
            ancestor = operation
            while ancestor is not None:
                contexts[ancestor] = _context_snapshot(ancestor)
                ancestor = ancestor.parent_op()
        output_order = lambda value: (positions[value.owner], value.index)
        result.append(
            OrderedFMAGroup(
                tuple(op for op in ops if op in root_set),
                ops,
                tuple(inputs),
                tuple(sorted(endpoints, key=output_order)),
                tuple(sorted(escapes, key=output_order)),
                tuple(sorted(unsupported, key=lambda item: (positions.get(item[1], -1), item[2]))),
                witnesses,
                tuple(contexts.values()),
            )
        )
    return tuple(sorted(result, key=lambda group: positions[group.operations[0]]))
