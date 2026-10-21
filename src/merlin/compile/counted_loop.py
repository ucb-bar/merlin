"""Form a counted CFG loop around an explicitly repeated state transition.

Callers supply the repetition semantics and check any memory-address mapping.
This helper only threads SSA state and integer inductions through existing CFG
blocks. It does not infer independence or make an instruction schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects import arith, cf, func
from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Block, BlockArgument, OpResult


@dataclass(frozen=True)
class LoopInductions:
    counter: BlockArgument
    offsets: tuple[BlockArgument, ...]


def simplify_state_cfg(function: func.FuncOp) -> None:
    """Apply xDSL's pure-expression CSE separately within each CFG block.

    Unknown effects remain untouched. Keeping the scopes local avoids extending
    scalar lifetimes across branches or loop backedges. No arithmetic rewrite
    or instruction scheduling is performed here.
    """
    from xdsl.transforms.common_subexpression_elimination import cse

    function.verify()
    for block in function.body.blocks:
        cse(block)
    function.verify()


def repeat_state_cfg(
    function: func.FuncOp,
    *,
    iterations: int,
    steps: tuple[int, ...],
    index_type: IntegerType,
    initial_state: OpResult | None = None,
) -> LoopInductions:
    """Repeat a closed, single-state CFG and expose dominating offset values.

    The entry block initializes state and branches to the body. Its operations
    remain in the preheader; every existing return becomes a loop backedge.
    Offsets start at zero and increase by the caller's positive steps. The
    final offsets and counter must fit the supplied unsigned integer type.
    For a single straight-line block, explicitly supply the first operation's
    state result. That initialization stays outside the repeated body.
    All eligibility checks precede mutation.
    """
    if not isinstance(index_type, IntegerType):
        raise ValueError("counted CFG needs an explicit integer induction type")
    limit = 1 << index_type.width.data
    if type(iterations) is not int or not 0 < iterations < limit:
        raise ValueError("iteration count must be positive and fit the induction type")
    if type(steps) is not tuple or any(
        type(step) is not int or step <= 0 or step * iterations >= limit for step in steps
    ):
        raise ValueError("positive induction steps must fit through the final backedge")
    blocks = tuple(function.body.blocks)
    if not blocks or function.args or len(function.function_type.outputs) != 1:
        raise ValueError("counted CFG needs a closed function with one state result")
    entry = blocks[0]
    branch = entry.last_op
    if initial_state is None:
        if not isinstance(branch, cf.BranchOp) or len(branch.arguments) != 1 or branch.successor is entry:
            raise ValueError("counted CFG entry must branch with one initial state")
        state, body = branch.arguments[0], branch.successor
        if len(body.args) != 1 or body.args[0].type != state.type:
            raise ValueError("counted CFG body must accept the initial state")
    else:
        if (
            not isinstance(initial_state, OpResult)
            or len(blocks) != 1
            or initial_state.owner is not entry.first_op
            or initial_state.owner.successors
            or len(initial_state.owner.results) != 1
            or initial_state.owner is branch
        ):
            raise ValueError("straight-line initialization must be the first operation's sole result")
        state = initial_state
        body = None
    if state.type != function.function_type.outputs.data[0]:
        raise ValueError("counted CFG entry and return state types differ")
    operations = tuple(op for block in blocks for op in block.ops)
    if any(op.regions or entry in op.successors for op in operations):
        raise ValueError("counted CFG body must be flat and cannot branch to its preheader")
    returns = tuple(op for op in operations if isinstance(op, func.ReturnOp))
    if not returns or any(len(op.arguments) != 1 or op.arguments[0].type != state.type for op in returns):
        raise ValueError("counted CFG returns must carry exactly the state value")
    function.verify()

    if body is None:
        body = Block(arg_types=[state.type])
        for op in tuple(entry.ops)[1:]:
            body.add_op(entry.detach_op(op))
        state.replace_all_uses_with(body.args[0])
        branch = cf.BranchOp(body, state)
        entry.add_op(branch)
        function.body.add_block(body)
    header = Block(arg_types=[state.type, index_type, *([index_type] * len(steps))])
    done = Block(arg_types=[state.type])
    function.body.insert_block_before(header, body)
    function.body.add_block(done)
    zero = arith.ConstantOp.from_int_and_width(0, index_type)
    one = arith.ConstantOp.from_int_and_width(1, index_type)
    count = arith.ConstantOp.from_int_and_width(iterations, index_type, truncate_bits=True)
    increments = tuple(arith.ConstantOp.from_int_and_width(step, index_type, truncate_bits=True) for step in steps)
    entry.erase_op(branch)
    entry.add_ops([zero, one, count, *increments, cf.BranchOp(header, state, zero, *([zero] * len(steps)))])
    more = arith.CmpiOp(header.args[1], count, "ult")
    header.add_ops([more, cf.ConditionalBranchOp(more, body, [header.args[0]], done, [header.args[0]])])
    for old_return in returns:
        block = old_return.parent_block()
        assert block is not None
        next_counter = arith.AddiOp(header.args[1], one)
        next_offsets = tuple(
            arith.AddiOp(offset, step) for offset, step in zip(header.args[2:], increments, strict=True)
        )
        block.insert_ops_before(
            [next_counter, *next_offsets, cf.BranchOp(header, old_return.arguments[0], next_counter, *next_offsets)],
            old_return,
        )
        block.erase_op(old_return)
    done.add_op(func.ReturnOp(done.args[0]))
    return LoopInductions(header.args[1], tuple(header.args[2:]))
