"""Guarded upstream xDSL scalar CSE/DCE and loop-invariant motion.

All transformed operations are total, effect-free, fixed-width scalar integer
operations. Unknown effects, memory accesses, regions, index arithmetic, poison
flags and unmodeled metadata remain unchanged. Hardware policies are absent.
"""

from __future__ import annotations

from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, scf
from xdsl.ir import Operation
from xdsl.rewriter import InsertPoint
from xdsl.traits import get_effects, is_speculatable
from xdsl.transforms.common_subexpression_elimination import CSEDriver
from xdsl.transforms.dead_code_elimination import LiveSet, is_trivially_dead
from xdsl.transforms.loop_invariant_code_motion import can_be_hoisted

from .certificates import ir_digest
from .scalar_witness import ScalarRewriteWitness

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
