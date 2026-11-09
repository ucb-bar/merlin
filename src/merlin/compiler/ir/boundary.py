"""Observe SSA boundaries without changing the program or granting movement legality.

Identity and source order determine inputs, captures and live results. Nested region
arguments are internal; values captured from enclosing scopes remain inputs. These
analyses do not classify effects, aliasing, purity, control flow, or target legality.
Their callers retain the existing source/custody and transformation admission checks.
"""

from __future__ import annotations


def free_values(kernel_ops):
    """Ordered, de-duplicated values the kernel must receive as arguments.

    A value is *free* (a parameter) when it is used anywhere in the kernel — including
    deep inside an op's regions — but is defined outside it. This must recurse into
    regions: model2MLIR ``linalg.generic`` bodies *capture* outer tensors (e.g. a
    ``tensor.extract`` gather for RoPE/embeddings), and those captures are not visible
    as top-level operands. Missing them yields an ``IsolatedFromAbove`` violation when
    the op is cloned into the kernel func.

    Values defined *inside* the kernel — results of any kernel op (or nested op) and the
    block arguments of the ops' own regions (e.g. a generic's ``^bb0`` element args) —
    are not parameters.
    """
    internal: set[int] = set()
    all_ops: list = []
    for op in kernel_ops:
        for sub in op.walk():
            all_ops.append(sub)
            for res in sub.results:
                internal.add(id(res))
            for region in sub.regions:
                for block in region.blocks:
                    for arg in block.args:
                        internal.add(id(arg))

    params: list = []
    seen: set[int] = set()
    for sub in all_ops:
        for operand in sub.operands:
            if id(operand) in internal or id(operand) in seen:
                continue
            seen.add(id(operand))
            params.append(operand)
    return params


def region_captures(op) -> list:
    """Values ``op``'s REGIONS read from the enclosing scope -- the operands ``op.operands`` omits.

    ``op.operands`` is the op's own signature and nothing more: for ``scf.for`` that is
    ``lb``/``ub``/``step``/``iter_args``. A tensor the loop BODY reads out of the enclosing scope is
    not an operand of the loop at all, so a node built from ``op.operands`` alone understates what the
    node reads -- and a memory planner fed that node computes a live range that ends before the last
    read. It then hands those bytes to another buffer while the loop is still reading them: wrong
    numbers, no diagnostic, and only on the input sizes that happen to collide.

    A value is captured when it is used anywhere inside the regions (at any nesting depth) and is NOT
    defined inside them -- i.e. it is neither the result of an inner op nor a block argument of any
    inner block (a loop induction variable and an ``iter_args`` body argument are both block args, so
    both are correctly internal). Same rule the outliner already applies in ``free_values``; stated
    once more here because this is a different question about the same structure.
    """
    internal: set[int] = set()
    inner_ops: list = []
    for region in op.regions:
        for block in region.blocks:
            for arg in block.args:
                internal.add(id(arg))
            for top in block.ops:
                for sub in top.walk():
                    inner_ops.append(sub)
                    for res in sub.results:
                        internal.add(id(res))
                    for sub_region in sub.regions:
                        for sub_block in sub_region.blocks:
                            for sub_arg in sub_block.args:
                                internal.add(id(sub_arg))
    captured: list = []
    seen: set[int] = set()
    for sub in inner_ops:
        for value in sub.operands:
            key = id(value)
            if key in internal or key in seen:
                continue
            seen.add(key)
            captured.append(value)
    return captured


def live_values(operations):
    """Return ordered external inputs and externally observed results of owned IR trees.

    This observes uses only; it neither proves effect freedom nor authorizes motion.
    """
    # Regions may capture external SSA values without listing them as parent
    # operands. Include the entire owned tree, excluding its own block args.
    nested = tuple(dict.fromkeys(child for root in operations for child in root.walk()))
    members = set(nested)
    inputs, outputs = {}, {}
    for op in nested:
        for value in op.operands:
            owner = value.owner
            local_block_arg = hasattr(owner, "parent_op") and owner.parent_op() in members
            if owner not in members and not local_block_arg:
                inputs[value] = None
        for value in op.results:
            if any(use.operation not in members for use in value.uses):
                outputs[value] = None

    return list(inputs), list(outputs)
