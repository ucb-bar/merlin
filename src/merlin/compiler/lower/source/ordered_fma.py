"""Source-owned closure analysis for guarded ordered floating contractions.

Groups contain live source operations, including every intermediate cast and
arithmetic boundary. Analysis grants no numerical relaxation, certificate,
physical alias fact, scheduling change or external-call purity. An emitter must
prove the complete endpoint and validate the retained source witness again.
"""

from __future__ import annotations

import math as scalar_math
from dataclasses import dataclass, field

# ---- ordered_fma_groups ------------------------------------------------------------------------


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


# ---- ordered_fma_matmul ------------------------------------------------------------------------
# Independent-output scheduling for explicitly ordered floating contractions.
#
# This builder changes only the interleaving of independent output elements. Each
# output starts at positive zero and uses increasing-K f32 FMA. BF16 operands are
# widened exactly before multiplication. It grants no reassociation permission.


def build_ordered_fma_matmul(
    lhs,
    rhs,
    *,
    output_tile: int,
    accumulation_order: str,
    rhs_transposed: bool = False,
    pre_widen_operands: str = "none",
    assume_rne: bool = False,
    assume_finite_intermediates: bool = False,
):
    """Return detached ops and f32 output for static, equally batched tensors.

    Callers must prove source zero initialization, increasing-K fused f32
    accumulation, RNE and finite intermediates. No broadcasting or transpose is
    inferred; ``rhs_transposed`` explicitly selects [..., N, K] storage. Both
    inputs have the same f32 or BF16 element type. Full output
    tiles and the statically known tail use distinct loops, so no out-of-bounds
    load is masked or speculated. Explicit BF16 pre-widening selects ``none``,
    ``lhs``, ``rhs`` or ``both`` and materializes exact f32 copies once before
    the contraction; its allocation and conversion costs
    belong to the caller's performance accounting. This is opt-in; no default
    pipeline calls it.
    """
    from xdsl.dialects import arith, math, scf, tensor
    from xdsl.dialects.builtin import FloatAttr, IndexType, TensorType, bf16, f32
    from xdsl.ir import Block, Region

    if accumulation_order != "zero_seeded_increasing_k_fma" or not assume_rne or not assume_finite_intermediates:
        raise ValueError("explicit zero-seeded ordered-FMA, finite-intermediate and RNE contracts required")
    if isinstance(output_tile, bool) or not isinstance(output_tile, int) or output_tile < 1:
        raise ValueError("positive integer output tile required")
    if any(not isinstance(v.type, TensorType) for v in (lhs, rhs)):
        raise TypeError("tensor operands required")
    dtype = lhs.type.element_type
    if dtype not in (f32, bf16) or rhs.type.element_type != dtype:
        raise TypeError("matching f32 or BF16 operand types required")
    if pre_widen_operands not in ("none", "lhs", "rhs", "both"):
        raise ValueError("pre_widen_operands must be none, lhs, rhs or both")
    if pre_widen_operands != "none" and dtype != bf16:
        raise ValueError("pre-widening requires BF16 operands")
    ls, rs = lhs.type.get_shape(), rhs.type.get_shape()
    if len(ls) < 2 or len(ls) != len(rs) or any(n <= 0 for n in (*ls, *rs)):
        raise ValueError("equal static positive operand ranks at least two required")
    if not isinstance(rhs_transposed, bool):
        raise TypeError("rhs_transposed must be a bool")
    rhs_k, rhs_n = (rs[-1], rs[-2]) if rhs_transposed else (rs[-2], rs[-1])
    if ls[:-2] != rs[:-2] or ls[-1] != rhs_k:
        raise ValueError("matching batch and contraction dimensions required")
    shape = (*ls[:-2], ls[-2], rhs_n)
    output_type = TensorType(f32, shape)
    empty = tensor.EmptyOp((), output_type)
    zero = arith.ConstantOp(FloatAttr(0.0, f32))
    constants = {}
    for n in (
        0,
        1,
        ls[-1],
        output_tile,
        *shape,
        (shape[-1] // output_tile) * output_tile,
        *range(min(output_tile, shape[-1])),
    ):
        if n not in constants:
            constants[n] = arith.ConstantOp.from_int_and_width(n, IndexType())

    def c(n):
        return constants[n].result

    packed_ops = []
    if pre_widen_operands != "none":

        def widen_operand(source):
            dimensions = source.type.get_shape()
            packed_type = TensorType(f32, dimensions)
            packed = tensor.EmptyOp((), packed_type)

            def copy(parent, depth, indices, destination):
                if depth == len(dimensions):
                    element = tensor.ExtractOp(source, indices, bf16)
                    wide = arith.ExtFOp(element.result, f32)
                    insert = tensor.InsertOp(wide.result, destination, indices)
                    parent.add_ops([element, wide, insert])
                    return insert.result
                body = Block(arg_types=[IndexType(), packed_type])
                value = copy(body, depth + 1, [*indices, body.args[0]], body.args[1])
                body.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(dimensions[depth]), c(1), [destination], Region(body))
                parent.add_op(loop)
                return loop.results[0]

            root = Block()
            value = copy(root, 0, [], packed.tensor)
            return [packed, *[root.detach_op(op) for op in list(root.ops)]], value

        if pre_widen_operands in ("lhs", "both"):
            lhs_ops, lhs = widen_operand(lhs)
            packed_ops.extend(lhs_ops)
        if pre_widen_operands in ("rhs", "both"):
            rhs_ops, rhs = widen_operand(rhs)
            packed_ops.extend(rhs_ops)

    def tile(parent, indices, column, width, destination):
        inner = Block(arg_types=[IndexType(), *([f32] * width)])
        a = tensor.ExtractOp(lhs, [*indices, inner.args[0]], lhs.type.element_type)
        inner.add_op(a)
        av = a.result
        if lhs.type.element_type == bf16:
            wide = arith.ExtFOp(av, f32)
            inner.add_op(wide)
            av = wide.result
        columns = []
        for lane in range(width):
            if lane:
                offset = arith.AddiOp(column, c(lane))
                parent.add_op(offset)
                columns.append(offset.result)
            else:
                columns.append(column)
        totals = []
        for lane, col in enumerate(columns):
            rhs_indices = [col, inner.args[0]] if rhs_transposed else [inner.args[0], col]
            b = tensor.ExtractOp(rhs, [*indices[:-1], *rhs_indices], rhs.type.element_type)
            inner.add_op(b)
            bv = b.result
            if rhs.type.element_type == bf16:
                wide = arith.ExtFOp(bv, f32)
                inner.add_op(wide)
                bv = wide.result
            fused = math.FmaOp(av, bv, inner.args[lane + 1])
            inner.add_op(fused)
            totals.append(fused.result)
        inner.add_op(scf.YieldOp(*totals))
        reduction = scf.ForOp(c(0), c(ls[-1]), c(1), [zero.result] * width, Region(inner))
        parent.add_op(reduction)
        for col, value in zip(columns, reduction.results):
            insert = tensor.InsertOp(value, destination, [*indices, col])
            parent.add_op(insert)
            destination = insert.result
        return destination

    def nest(parent, depth, indices, destination):
        if depth == len(shape) - 1:
            end = (shape[-1] // output_tile) * output_tile
            if end:
                body = Block(arg_types=[IndexType(), output_type])
                value = tile(body, indices, body.args[0], output_tile, body.args[1])
                body.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(end), c(output_tile), [destination], Region(body))
                parent.add_op(loop)
                destination = loop.results[0]
            remainder = shape[-1] - end
            if remainder:
                destination = tile(parent, indices, c(end), remainder, destination)
            return destination
        body = Block(arg_types=[IndexType(), output_type])
        value = nest(body, depth + 1, [*indices, body.args[0]], body.args[1])
        body.add_op(scf.YieldOp(value))
        loop = scf.ForOp(c(0), c(shape[depth]), c(1), [destination], Region(body))
        parent.add_op(loop)
        return loop.results[0]

    root = Block()
    result = nest(root, 0, [], empty.tensor)
    ops = [root.detach_op(op) for op in list(root.ops)]
    return [empty, zero, *constants.values(), *packed_ops, *ops], result


# ---- ordered_fma_rewrite -----------------------------------------------------------------------
# Explicit scheduling of structurally proved, zero-seeded FMA contractions.
#
# Only canonical equally batched matrix indexing and an increasing innermost K
# reduction are accepted. Source identity attributes do not select operations.
# The caller supplies finite-intermediate and RNE contracts for matched operations.


def _match(op):
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import FloatAttr, TensorType, bf16, f32
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if not isinstance(op, linalg.GenericOp) or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        return None
    a, b = op.inputs
    out = op.outputs[0]
    if any(not isinstance(v.type, TensorType) for v in (a, b, out)):
        return None
    dtype = a.type.element_type
    if dtype not in (bf16, f32) or b.type.element_type != dtype or out.type.element_type != f32:
        return None
    ash, bsh, osh = (v.type.get_shape() for v in (a, b, out))
    rank = len(osh)
    if rank < 2 or len(ash) != rank or len(bsh) != rank or any(n <= 0 for n in (*ash, *bsh, *osh)):
        return None
    if op.results[0].type != out.type or ash[:-2] != bsh[:-2] or osh[:-2] != ash[:-2] or osh[-2] != ash[-2]:
        return None
    if tuple(x.data for x in op.iterator_types) != (
        *([linalg.IteratorType.PARALLEL] * rank),
        linalg.IteratorType.REDUCTION,
    ):
        return None
    prefix = tuple(AffineDimExpr(i) for i in range(rank - 2))
    m, n, k = (AffineDimExpr(i) for i in (rank - 2, rank - 1, rank))
    maps = tuple(x.data for x in op.indexing_maps)
    if (
        len(maps) != 3
        or maps[0] != AffineMap(rank + 1, 0, (*prefix, m, k))
        or maps[2] != AffineMap(rank + 1, 0, (*prefix, m, n))
    ):
        return None
    if maps[1] == AffineMap(rank + 1, 0, (*prefix, k, n)):
        transposed = False
        expected_rhs = (*ash[:-2], ash[-1], osh[-1])
    elif maps[1] == AffineMap(rank + 1, 0, (*prefix, n, k)):
        transposed = True
        expected_rhs = (*ash[:-2], osh[-1], ash[-1])
    else:
        return None
    if bsh != expected_rhs:
        return None
    seed = out.owner
    if not isinstance(seed, (tensor.SplatOp, linalg.FillOp)):
        return None
    constant = seed.operands[0].owner
    if (
        not isinstance(constant, arith.ConstantOp)
        or not isinstance(constant.value, FloatAttr)
        or constant.result.type != f32
    ):
        return None
    zero = constant.value.value.data
    if zero != 0.0 or scalar_math.copysign(1.0, zero) != 1.0:
        return None
    if len(op.body.blocks) != 1:
        return None
    body = op.body.block
    ops = list(body.ops)
    if len(body.args) != 3 or not isinstance(ops[-1], linalg.YieldOp):
        return None
    if dtype == bf16:
        if len(ops) != 4 or not all(isinstance(x, arith.ExtFOp) for x in ops[:2]):
            return None
        if (
            ops[0].input is not body.args[0]
            or ops[1].input is not body.args[1]
            or any(x.result.type != f32 for x in ops[:2])
        ):
            return None
        lhs, rhs = ops[0].result, ops[1].result
    else:
        if len(ops) != 2:
            return None
        lhs, rhs = body.args[:2]
    fused = ops[-2]
    if not isinstance(fused, math.FmaOp) or tuple(fused.operands) != (lhs, rhs, body.args[2]):
        return None
    if tuple(ops[-1].arguments) != (fused.result,):
        return None
    return transposed


def rewrite_ordered_fma_contractions(
    module,
    *,
    output_tile: int,
    pre_widen_operands: str = "none",
    assume_rne: bool = False,
    assume_finite_intermediates: bool = False,
):
    """Rewrite proved source FMA generics; no default pipeline invokes this pass.

    No broadcasts, nonzero/negative-zero seeds, changed reduction order or
    additional arithmetic are accepted. BF16 widening selection is explicit;
    f32 operands need no widening. Allocation and conversion remain in emitted
    IR and must be included in measured costs. Source attributes are preserved
    on the result-producing operation without controlling applicability.
    """
    from xdsl.dialects.builtin import bf16
    from xdsl.rewriter import Rewriter

    if type(output_tile) is not int or output_tile <= 0:
        raise ValueError("positive integer output tile required")
    if pre_widen_operands not in ("none", "lhs", "rhs", "both"):
        raise ValueError("pre_widen_operands must be none, lhs, rhs or both")
    if not assume_rne or not assume_finite_intermediates:
        raise ValueError("explicit finite-intermediate and RNE contracts required")
    module.verify()
    matched = [(op, result) for op in module.walk() if (result := _match(op)) is not None]
    counts = {"rewritten": 0, "bf16": 0, "f32": 0, "source_macs": 0}
    for op, transposed in matched:
        dtype = op.inputs[0].type.element_type
        packing = pre_widen_operands if dtype == bf16 else "none"
        ops, result = build_ordered_fma_matmul(
            *op.inputs,
            output_tile=output_tile,
            rhs_transposed=transposed,
            pre_widen_operands=packing,
            accumulation_order="zero_seeded_increasing_k_fma",
            assume_rne=assume_rne,
            assume_finite_intermediates=assume_finite_intermediates,
        )
        result.owner.attributes.update(op.attributes)
        counts["rewritten"] += 1
        counts[str(dtype)] += 1
        counts["source_macs"] += scalar_math.prod(op.results[0].type.get_shape()) * op.inputs[0].type.get_shape()[-1]
        Rewriter.replace_op(op, ops, [result])
    module.verify()
    return counts
