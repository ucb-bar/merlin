"""Recognize actual generated structure before issuing abstract proof obligations.

This is a structural guard, never an equivalence proof receipt. In particular,
the loop recognizer checks actual IV substitution, carried state, tail bounds,
body preservation and unrelated surrounding IR independently of code emission.
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Block, Operation, SSAValue

from .certificates import TRANSFORM_SCOPES, LoopWitness, TransformationChange, ir_digest, ir_text, parse_ir
from .normalization import constant_integer, normalize_shapes, simplify_integers
from .planning import apply_order, plan_storage, reuse_storage


class _LoopRecognizer:
    def __init__(self, before: builtin.ModuleOp, witnesses: Sequence[LoopWitness]):
        operations = tuple(before.walk())
        self.selected = {}
        for witness in witnesses:
            if not 0 <= witness.operation_index < len(operations):
                raise ValueError("loop witness operation index is outside the actual IR")
            op = operations[witness.operation_index]
            if not isinstance(op, scf.ForOp) or ir_digest(op) != witness.body_sha256:
                raise ValueError("loop witness does not identify the actual source loop")
            if op in self.selected:
                raise ValueError("duplicate loop witness")
            self.selected[op] = witness
        self.context: dict[SSAValue, SSAValue] = {}
        self.seen: set[Operation] = set()

    def values(self, before: Sequence[SSAValue], after: Sequence[SSAValue]) -> bool:
        return len(before) == len(after) and all(self.context.get(a) is b for a, b in zip(before, after, strict=True))

    def block(self, before: Block, after: Block) -> bool:
        if tuple(a.type for a in before.args) != tuple(a.type for a in after.args):
            return False
        self.context.update(zip(before.args, after.args, strict=True))
        left, right = tuple(before.ops), tuple(after.ops)
        cursor = 0
        for op in left:
            if op in self.selected:
                if cursor + 1 >= len(right) or not self.segment(op, right[cursor], right[cursor + 1]):
                    return False
                cursor += 2
            else:
                if cursor >= len(right) or not self.operation(op, right[cursor]):
                    return False
                cursor += 1
        return cursor == len(right)

    def operation(self, before: Operation, after: Operation) -> bool:
        if (
            before.name != after.name
            or before.attributes != after.attributes
            or before.properties != after.properties
            or tuple(r.type for r in before.results) != tuple(r.type for r in after.results)
            or not self.values(before.operands, after.operands)
            or before.successors
            or after.successors
            or len(before.regions) != len(after.regions)
        ):
            return False
        self.context.update(zip(before.results, after.results, strict=True))
        for a, b in zip(before.regions, after.regions, strict=True):
            if len(a.blocks) != len(b.blocks):
                return False
            if not all(self.block(x, y) for x, y in zip(a.blocks, b.blocks, strict=True)):
                return False
        return True

    def segment(self, source: Operation, stride: Operation, target: Operation) -> bool:
        if (
            not isinstance(source, scf.ForOp)
            or not isinstance(stride, arith.ConstantOp)
            or not isinstance(target, scf.ForOp)
        ):
            return False
        witness = self.selected[source]
        actual_bounds = tuple(constant_integer(v) for v in (source.lb, source.ub, source.step))
        if actual_bounds != (witness.lower, witness.upper, witness.step):
            return False
        if witness.tile_iterations <= 1 or witness.step <= 0 or witness.lower < 0 or witness.upper < witness.lower:
            return False
        if (
            witness.index_bits <= 1
            or witness.upper + witness.step * witness.tile_iterations > (1 << (witness.index_bits - 1)) - 1
        ):
            return False
        if tuple(str(v.type) for v in source.iter_args) != witness.carried_types:
            return False
        if (
            not isinstance(stride.result.type, builtin.IndexType)
            or stride.attributes
            or constant_integer(stride.result) != witness.step * witness.tile_iterations
            or target.lb is not self.context.get(source.lb)
            or target.ub is not self.context.get(source.ub)
            or target.step is not stride.result
            or not self.values(source.iter_args, target.iter_args)
            or target.attributes
            or target.properties != source.properties
            or tuple(r.type for r in target.results) != tuple(r.type for r in source.results)
        ):
            return False
        block = target.body.block
        if tuple(v.type for v in block.args) != tuple(v.type for v in source.body.block.args):
            return False
        body = tuple(block.ops)
        if len(body) != 4:
            return False
        add, minimum, inner, outer_yield = body
        if not isinstance(add, arith.AddiOp) or not isinstance(minimum, arith.MinUIOp):
            return False
        if add.attributes or add.overflow_flags != arith.IntegerOverflowAttr("none"):
            return False
        if add.lhs is not block.args[0] or add.rhs is not stride.result:
            return False
        if minimum.attributes or minimum.lhs is not add.result or minimum.rhs is not self.context.get(source.ub):
            return False
        if not isinstance(inner, scf.ForOp) or not isinstance(outer_yield, scf.YieldOp):
            return False
        if (
            inner.lb is not block.args[0]
            or inner.ub is not minimum.result
            or inner.step is not self.context.get(source.step)
            or tuple(inner.iter_args) != tuple(block.args[1:])
            or inner.attributes != source.attributes
            or inner.properties != source.properties
            or tuple(v.type for v in inner.results) != tuple(v.type for v in source.results)
            or tuple(outer_yield.operands) != tuple(inner.results)
            or outer_yield.attributes
        ):
            return False
        self.context.update(zip(source.results, target.results, strict=True))
        self.seen.add(source)
        return self.block(source.body.block, inner.body.block)


def validate_change(change: TransformationChange) -> None:
    """Reject stale identities, forged witnesses or different actual rewrites."""
    change.validate_identity()
    if change.kind not in TRANSFORM_SCOPES or change.certificate.scope != TRANSFORM_SCOPES[change.kind]:
        raise ValueError("transformation coverage cannot be changed by candidate declarations")
    expected_parameters = {
        "ordered_loop_segmentation": {"index_bits", "tile_iterations"},
        "sequential_dependency_schedule": {"block_index", "order"},
    }.get(change.kind, set())
    if set(change.parameters) != expected_parameters:
        raise ValueError("unexpected transformation parameters")
    before, after = parse_ir(change.before_ir), parse_ir(change.after_ir)
    if change.kind == "ordered_loop_segmentation":
        if any(not isinstance(w, LoopWitness) for w in change.certificate.witnesses):
            raise ValueError("unexpected loop witness type")
        recognizer = _LoopRecognizer(before, change.certificate.witnesses)
        if not recognizer.operation(before, after) or recognizer.seen != set(recognizer.selected):
            raise ValueError("actual IR is not the witnessed order-preserving loop segmentation")
        for witness in change.certificate.witnesses:
            if witness.index_bits != change.parameters.get("index_bits"):
                raise ValueError("loop index width does not match requested compilation parameters")
            loop = tuple(before.walk())[witness.operation_index]
            depth = 0
            parent = loop.parent_op()
            while parent is not None:
                depth += isinstance(parent, scf.ForOp)
                parent = parent.parent_op()
            tiles = change.parameters.get("tile_iterations", ())
            if depth >= len(tiles) or witness.tile_iterations != tiles[depth]:
                raise ValueError("loop tile does not match requested compilation parameters")
        return
    # These guards regenerate a structural result from current IR facts. Their
    # corresponding abstract semantics remain separate host proof obligations.
    if change.kind == "shape_view_normalization":
        witnesses = normalize_shapes(before)
    elif change.kind == "exact_integer_simplification":
        witnesses = simplify_integers(before)
    elif change.kind == "buffer_reuse":
        witnesses = reuse_storage(before, plan_storage(before))
    elif change.kind == "sequential_dependency_schedule":
        witnesses = (apply_order(before, int(change.parameters["block_index"]), change.parameters["order"]),)
    else:
        raise ValueError("unknown shared transformation kind")
    before.verify()
    if ir_text(before) != ir_text(after) or witnesses != change.certificate.witnesses:
        raise ValueError("actual IR or witnesses differ from the current structural transformation")
