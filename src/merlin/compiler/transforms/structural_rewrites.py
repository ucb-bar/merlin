"""Conservative scalar-ring and statically known view normalization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, memref, scf
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.traits import get_effects, is_speculatable
from xdsl.transforms.common_subexpression_elimination import CSEDriver
from xdsl.transforms.dead_code_elimination import LiveSet, is_trivially_dead
from xdsl.transforms.loop_invariant_code_motion import can_be_hoisted

from merlin.compiler.transforms.certificates import IntegerWitness, ShapeWitness, ir_digest, ir_text, parse_ir

# ---- normalization -----------------------------------------------------------------------------


def constant_integer(value: SSAValue) -> int | None:
    owner = value.owner
    if isinstance(owner, arith.ConstantOp) and isinstance(owner.value, builtin.IntegerAttr):
        return owner.value.value.data
    return None


def signed_wrap(value: int, width: int) -> int:
    if width <= 0:
        raise ValueError("integer width must be positive")
    modulus = 1 << width
    value %= modulus
    return value - modulus if value >= modulus // 2 else value


def normalize_shapes(module: builtin.ModuleOp) -> tuple[ShapeWitness, ...]:
    """Eliminate identical memref casts and materialize static dimensions.

    No reinterpretation, guessed strides, dynamic-to-static assumptions or
    element enumeration. The identity cast also applies to contiguous views
    presented through an already identical memref descriptor.
    """
    witnesses = []
    for index, op in enumerate(tuple(module.walk())):
        if isinstance(op, memref.CastOp) and op.source.type == op.dest.type:
            shape = op.source.type.get_shape() if isinstance(op.source.type, builtin.MemRefType) else ()
            witnesses.append(ShapeWitness(index, "identity_memref_cast", tuple(shape)))
            Rewriter.replace_op(op, [], [op.source])
        elif isinstance(op, memref.DimOp) and isinstance(op.source.type, builtin.MemRefType):
            axis = constant_integer(op.index)
            shape = op.source.type.get_shape()
            if axis is not None and 0 <= axis < len(shape) and shape[axis] >= 0:
                replacement = arith.ConstantOp.from_int_and_width(shape[axis], builtin.IndexType())
                witnesses.append(ShapeWitness(index, "static_memref_dimension", tuple(shape), axis))
                Rewriter.replace_op(op, replacement)
    return tuple(witnesses)


def simplify_integers(module: builtin.ModuleOp) -> tuple[IntegerWitness, ...]:
    """Fold scalar wrapping integers and neutral operands, without poison flags.

    Index arithmetic, tensors, floating arithmetic and nsw/nuw operations are
    deliberately excluded. Constants are canonicalized modulo the declared
    bit width; reductions are never reassociated.
    """
    witnesses = []
    for index, op in enumerate(tuple(module.walk())):
        if not isinstance(op, arith.AddiOp | arith.SubiOp | arith.MuliOp):
            continue
        dtype = op.result.type
        if not isinstance(dtype, builtin.IntegerType) or dtype.signedness.data != builtin.Signedness.SIGNLESS:
            continue
        if op.overflow_flags != arith.IntegerOverflowAttr("none"):
            continue
        lhs, rhs = constant_integer(op.lhs), constant_integer(op.rhs)
        width = dtype.width.data
        if lhs is not None and rhs is not None:
            if isinstance(op, arith.AddiOp):
                value = lhs + rhs
            elif isinstance(op, arith.SubiOp):
                value = lhs - rhs
            else:
                value = lhs * rhs
            result = signed_wrap(value, width)
            replacement = arith.ConstantOp.from_int_and_width(result, dtype)
            witnesses.append(IntegerWitness(index, op.name, "constant_ring", width, lhs, rhs, result))
            Rewriter.replace_op(op, replacement)
            continue
        retained: int | None = None
        if isinstance(op, arith.AddiOp):
            retained = 1 if lhs == 0 else 0 if rhs == 0 else None
        elif isinstance(op, arith.SubiOp):
            retained = 0 if rhs == 0 else None
        elif isinstance(op, arith.MuliOp):
            retained = 1 if lhs == 1 else 0 if rhs == 1 else None
        if retained is not None:
            witnesses.append(IntegerWitness(index, op.name, "neutral_operand", width, lhs, rhs, None, retained))
            Rewriter.replace_op(op, [], [op.operands[retained]])
    return tuple(witnesses)


# ---- unrolling ---------------------------------------------------------------------------------
# Bounded ordered unrolling; code growth follows an explicit compiler budget.
#
# This mechanism selects no profitable factor. It preserves each original body
# evaluation, including its effects and loop-carried state, in its original order.


@dataclass(frozen=True)
class UnrollWitness:
    operation_index: int
    lower: int
    upper: int
    step: int
    factor: int
    index_bits: int
    iterations: int
    full_groups: int
    remainder: int
    source_sha256: str
    carried_types: tuple[str, ...]


def _depth(loop: scf.ForOp) -> int:
    depth = 0
    parent = loop.parent_op()
    while parent is not None:
        depth += isinstance(parent, scf.ForOp)
        parent = parent.parent_op()
    return depth


def _repeat_body(source: Block, target: Block, iv: SSAValue, state: Sequence[SSAValue]) -> tuple[SSAValue, ...]:
    mapper = dict(zip(source.args, (iv, *state), strict=True))
    operations = tuple(source.ops)
    terminal = operations[-1]
    if not isinstance(terminal, scf.YieldOp) or terminal.attributes or terminal.properties:
        raise ValueError("ordered unrolling requires the exact standard yield")
    for operation in operations[:-1]:
        target.add_op(operation.clone(mapper))
    return tuple(mapper.get(value, value) for value in terminal.operands)


def _unroll_one(loop: scf.ForOp, witness: UnrollWitness) -> None:
    source = loop.body.block
    if witness.iterations <= witness.factor:
        emitted = Block()
        state = tuple(loop.iter_args)
        for lane in range(witness.iterations):
            iv = arith.ConstantOp.from_int_and_width(witness.lower + lane * witness.step, builtin.IndexType())
            emitted.add_op(iv)
            state = _repeat_body(source, emitted, iv.result, state)
        replacements = list(emitted.ops)
        for operation in replacements:
            emitted.detach_op(operation)
        Rewriter.replace_op(loop, replacements, list(state))
        return
    bulk_upper = witness.lower + witness.full_groups * witness.factor * witness.step
    stride = arith.ConstantOp.from_int_and_width(witness.factor * witness.step, builtin.IndexType())
    bound = arith.ConstantOp.from_int_and_width(bulk_upper, builtin.IndexType())
    bulk_body = Block(arg_types=[argument.type for argument in source.args])
    state = tuple(bulk_body.args[1:])
    for lane in range(witness.factor):
        iv = bulk_body.args[0]
        if lane:
            offset = arith.ConstantOp.from_int_and_width(lane * witness.step, builtin.IndexType())
            addition = arith.AddiOp(iv, offset.result)
            bulk_body.add_ops([offset, addition])
            iv = addition.result
        state = _repeat_body(source, bulk_body, iv, state)
    bulk_body.add_op(scf.YieldOp(*state))
    bulk = scf.ForOp(loop.lb, bound.result, stride.result, loop.iter_args, bulk_body)
    replacements: list[Operation] = [stride, bound, bulk]
    result = bulk.results
    if witness.remainder:
        tail_body = Block(arg_types=[argument.type for argument in source.args])
        mapper = dict(zip(source.args, tail_body.args, strict=True))
        tail_body.add_ops(operation.clone(mapper) for operation in source.ops)
        tail = scf.ForOp(bound.result, loop.ub, loop.step, bulk.results, tail_body)
        replacements.append(tail)
        result = tail.results
    Rewriter.replace_op(loop, replacements, list(result))


def unroll_ordered_loops(
    module: builtin.ModuleOp, factors: Sequence[int], *, index_bits: int, max_added_operations: int = 4096
) -> tuple[UnrollWitness, ...]:
    """Fully inline small loops or partially unroll a bounded bulk plus exact tail.

    Factors index original nesting depth. Dynamic/negative bounds and unsupported
    metadata remain unchanged. Requested arithmetic overflow or code growth
    refuses the transformation. No tensor elements or trip-count-sized structures
    are allocated. Callers should transform a private clone for transactional use.
    """
    if (
        type(index_bits) is not int
        or not 2 <= index_bits <= 128
        or type(max_added_operations) is not int
        or max_added_operations < 1
        or any(type(factor) is not int or factor < 1 for factor in factors)
    ):
        raise ValueError("unrolling requires explicit positive factors, index width and growth budget")
    module.verify()
    original_count = sum(1 for _ in module.walk())
    candidates = [
        (index, operation, _depth(operation), ir_digest(operation))
        for index, operation in enumerate(module.walk())
        if isinstance(operation, scf.ForOp)
    ]
    witnesses = []
    for index, loop, depth, source_hash in reversed(candidates):
        if depth >= len(factors) or factors[depth] == 1 or loop.attributes or loop.properties:
            continue
        lower, upper, step = (constant_integer(value) for value in (loop.lb, loop.ub, loop.step))
        if lower is None or upper is None or step is None or lower < 0 or upper < lower or step < 1:
            continue
        if not all(isinstance(value.type, builtin.IndexType) for value in (loop.lb, loop.ub, loop.step)):
            continue
        factor = factors[depth]
        if upper + step * factor > (1 << (index_bits - 1)) - 1:
            raise ValueError("unrolled loop arithmetic exceeds the selected signed index width")
        count = (upper - lower + step - 1) // step
        groups, remainder = divmod(count, factor)
        # A conservative bound is checked before cloning any repeated body.
        body_size = sum(1 for _ in loop.walk())
        growth = min(count, factor) * (body_size + 3) + body_size + 8
        if sum(1 for _ in module.walk()) - original_count + growth > max_added_operations:
            raise ValueError("unrolling exceeds the selected operation growth budget")
        witness = UnrollWitness(
            index,
            lower,
            upper,
            step,
            factor,
            index_bits,
            count,
            groups,
            remainder,
            source_hash,
            tuple(str(value.type) for value in loop.iter_args),
        )
        _unroll_one(loop, witness)
        witnesses.append(witness)
    module.verify()
    return tuple(reversed(witnesses))


def validate_unrolled_change(change) -> None:
    """Bind actual emitted instructions, IV substitutions and state to the source.

    Trusted structural replay is explicit; arithmetic/fold obligations are checked
    by the independent host Lean route, not by this correspondence validator.
    """
    expected = {"factors", "index_bits", "max_added_operations"}
    if set(change.parameters) != expected:
        raise ValueError("unrolling parameters differ from the explicit source selection")
    change.validate_identity()
    source = parse_ir(change.before_ir)
    witnesses = unroll_ordered_loops(
        source,
        change.parameters["factors"],
        index_bits=change.parameters["index_bits"],
        max_added_operations=change.parameters["max_added_operations"],
    )
    if witnesses != change.certificate.witnesses or ir_text(source) != ir_text(parse_ir(change.after_ir)):
        raise ValueError("emitted unrolling differs from actual ordered source bodies and yields")


# ---- scalar_witness ----------------------------------------------------------------------------
# Actual-operation witnesses for conservative scalar reuse and invariant motion.
#
# These records carry no proof authority. Their replay validator is part of the
# explicit structural-checker trusted computing base, separately from Lean lemmas.


@dataclass(frozen=True)
class ScalarRewriteWitness:
    operation_index: int
    operation: str
    rule: str
    width: int
    operation_sha256: str
    representative_index: int | None = None
    loop_index: int | None = None


SCALAR_SCOPES = {
    "scalar_redundancy_elimination": (
        "total_scalar_integer_operations",
        "preserved_unknown_effects",
        "ssa_use_def_and_dominance",
    ),
    "scalar_loop_invariant_motion": (
        "total_scalar_integer_operations",
        "total_speculation",
        "loop_external_operands",
        "unchanged_effect_order",
    ),
}


def validate_scalar_change(change) -> None:
    """Reopen and replay the admitted mechanism against the actual complete IR."""
    from merlin.compiler.transforms.certificates import ir_text, parse_ir

    change.validate_identity()
    if change.kind not in SCALAR_SCOPES or change.certificate.scope != SCALAR_SCOPES[change.kind]:
        raise ValueError("scalar transformation coverage differs from the fixed semantic domain")
    if change.parameters or any(not isinstance(w, ScalarRewriteWitness) for w in change.certificate.witnesses):
        raise ValueError("scalar transformation has unsupported parameters or witnesses")
    before, after = parse_ir(change.before_ir), parse_ir(change.after_ir)
    transform = {
        "scalar_redundancy_elimination": eliminate_scalar_redundancy,
        "scalar_loop_invariant_motion": hoist_scalar_invariants,
    }[change.kind]
    witnesses = transform(before)
    if not witnesses or witnesses != change.certificate.witnesses or ir_text(before) != ir_text(after):
        raise ValueError("actual scalar IR or witnesses differ from the admitted rewrite")


# ---- pure_scalar -------------------------------------------------------------------------------
# Guarded upstream xDSL scalar CSE/DCE and loop-invariant motion.
#
# All transformed operations are total, effect-free, fixed-width scalar integer
# operations. Unknown effects, memory accesses, regions, index arithmetic, poison
# flags and unmodeled metadata remain unchanged. Hardware policies are absent.

_BINARY = (arith.AddiOp, arith.SubiOp, arith.MuliOp)
_EXTENSIONS = (arith.ExtSIOp, arith.ExtUIOp, arith.TruncIOp)


def eligible_scalar_operation(op: Operation) -> bool:
    """Closed total arithmetic domain; operation traits alone do not admit OOT code."""
    if type(op) not in (arith.ConstantOp, *_BINARY, *_EXTENSIONS):
        return False
    if op.regions or op.successors or op.attributes or len(op.results) != 1 or get_effects(op) != set():
        return False
    values = (*op.operands, *op.results)
    if any(
        not isinstance(value.type, builtin.IntegerType)
        or value.type.signedness.data != builtin.Signedness.SIGNLESS
        or not 1 <= value.type.width.data <= 256
        for value in values
    ):
        return False
    allowed = {"value"} if type(op) is arith.ConstantOp else {"overflowFlags"} if type(op) in _BINARY else set()
    if set(op.properties) - allowed:
        return False
    if type(op) in _BINARY and op.overflow_flags != arith.IntegerOverflowAttr("none"):
        return False
    if type(op) is arith.ConstantOp and not isinstance(op.value, builtin.IntegerAttr):
        return False
    return True


class _ScalarCSE(CSEDriver):
    """Use upstream equivalence, dominance scopes and rewriting with a narrower domain."""

    def __init__(self, indices, fingerprints, witnesses):
        super().__init__()
        self.indices, self.fingerprints, self.witnesses = indices, fingerprints, witnesses

    def _simplify_operation(self, op: Operation):
        if not eligible_scalar_operation(op):
            return
        dead = is_trivially_dead(op)
        existing = None if dead else self._known_ops.get(op)
        uses = tuple(tuple(result.uses) for result in op.results)
        super()._simplify_operation(op)
        if op in self._to_erase or uses != tuple(tuple(result.uses) for result in op.results):
            self.witnesses.append(
                ScalarRewriteWitness(
                    self.indices[op],
                    op.name,
                    "dce" if dead else "cse",
                    op.results[0].type.width.data,
                    self.fingerprints[op],
                    None if existing is None else self.indices[existing],
                )
            )


def _snapshot(module):
    operations = tuple(module.walk())
    # Only admitted leaf instructions need fingerprints. Hashing every enclosing
    # region would repeatedly serialize whole functions and nested loop bodies.
    return (
        {op: i for i, op in enumerate(operations)},
        {op: ir_digest(op) for op in operations if eligible_scalar_operation(op)},
    )


def eliminate_scalar_redundancy(module: builtin.ModuleOp) -> tuple[ScalarRewriteWitness, ...]:
    """Mutate admitted scalar operations, retaining opaque consumers and all effects.

    Upstream CSE includes leaf DCE. A guarded upstream liveness fixpoint then
    deletes complete dead scalar chains. Every other operation is a liveness
    root, including unreachable-block effects and terminators. No CFG is changed.
    Space depends on IR size, not tensor dimensions or iteration counts.
    """
    module.verify()
    indices, fingerprints = _snapshot(module)
    witnesses = []
    _ScalarCSE(indices, fingerprints, witnesses).simplify(module)
    live = LiveSet()
    for op in module.walk():
        if not eligible_scalar_operation(op):
            live.set_live(op)
    while live.changed:
        live.changed = False
        # Include unreachable blocks too: opaque operations remain liveness
        # roots, and their captured scalar inputs must never be deleted.
        for op in reversed(tuple(module.walk())):
            live.propagate_op_liveness(op)
    for op in module.walk():
        if not live.is_live(op):
            if not eligible_scalar_operation(op):
                raise ValueError("upstream liveness would remove an operation outside the scalar domain")
            witnesses.append(
                ScalarRewriteWitness(
                    indices[op],
                    op.name,
                    "dce",
                    op.results[0].type.width.data,
                    fingerprints[op],
                )
            )
    for region in module.regions:
        live.delete_dead(region, None)
    module.verify()
    return tuple(witnesses)


def hoist_scalar_invariants(module: builtin.ModuleOp) -> tuple[ScalarRewriteWitness, ...]:
    """Hoist total scalar expressions with operands outside the immediate loop.

    Upstream ownership and speculation analysis provide the movement preconditions.
    Only direct-body admitted operations are moved; unknown region semantics are
    not crossed. Zero-trip and dynamic loops are safe because this domain has no
    trapping, poison-producing or observable operations. Loop-carried values and
    induction variables remain variant. Chained invariants move in SSA order.
    """
    module.verify()
    indices, fingerprints = _snapshot(module)
    witnesses = []
    loops = tuple(op for op in module.walk() if type(op) is scf.ForOp)
    for loop in reversed(loops):
        builder = Builder(InsertPoint.before(loop))
        for op in tuple(loop.body.block.ops):
            if not eligible_scalar_operation(op) or not is_speculatable(op) or not can_be_hoisted(op, loop.body):
                continue
            witnesses.append(
                ScalarRewriteWitness(
                    indices[op],
                    op.name,
                    "licm",
                    op.results[0].type.width.data,
                    fingerprints[op],
                    loop_index=indices[loop],
                )
            )
            op.detach()
            builder.insert(op)
    module.verify()
    return tuple(witnesses)
