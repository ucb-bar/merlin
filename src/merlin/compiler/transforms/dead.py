"""Erase dead tensor cones only after complete typed purity and validity checks.

Preserve source ordering, numerical/effect obligations and explicit selection.
"""

from __future__ import annotations


def _dead_tensor_operation_is_pure(op) -> bool:
    """Require typed value semantics, including every structured scalar body.

    Dialect names and unused results confer no execution-effect information.
    In particular unregistered quantization ops survive until their separate
    explicit lowering provides typed semantics. Physical allocation effects
    remain outside this tensor cleanup.
    """
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import MemRefType, NoneAttr, TensorType, UnrankedMemRefType, UnregisteredOp
    from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.irdl import IRDLOperation
    from xdsl.traits import get_effects, is_speculatable

    if not isinstance(op, IRDLOperation) or isinstance(op, UnregisteredOp) or op.successors:
        return False
    definition = op.get_irdl_definition()
    if any(key not in definition.attributes and not key.startswith("prov.") for key in op.attributes) or any(
        key not in definition.properties for key in op.properties
    ):
        return False
    if any(
        isinstance(value.type, MemRefType | UnrankedMemRefType)
        or isinstance(value.type, TensorType) and value.type.encoding != NoneAttr()
        for value in (*op.operands, *op.results)
    ):
        return False
    if isinstance(op, LinalgStructuredOperation):
        if (
            not op.results
            or any(not isinstance(value.type, TensorType) for value in op.results)
            or not op.regions
            or any(len(region.blocks) != 1 for region in op.regions)
            or op.properties.get("library_call") is not None
        ):
            return False
        # The structured tensor operation is value-only only when its actual
        # scalar body is known pure. Its generic container has no such trait.
        return all(
            _dead_tensor_operation_is_pure(child)
            for region in op.regions
            for child in region.block.ops
        )
    if op.regions:
        return False
    if isinstance(op, YieldOp):
        return True
    if isinstance(op, tensor.EmptyOp):
        return get_effects(op) == set()
    return get_effects(op) == set() and is_speculatable(op)

def prune_dead_pure_tensor_ops(module) -> int:
    """Erase dead top-level pure tensor-expression cones; return the number of erased ops.

    The pass is deliberately conservative: it works within each ``func.func`` entry block and
    only removes result-producing operations with known value-only effects and nontrapping
    semantics. Calls, stores, terminators, unregistered ops and unknown scalar bodies survive.
    It therefore provides a common cleanup seam for scalar/RVV and generated target dialects
    without teaching the frontend anything about a particular accelerator.
    """
    from xdsl.utils.exceptions import VerifyException

    erased = 0
    for func in [op for op in module.walk() if op.name == "func.func"]:
        if not func.regions or not func.regions[0].blocks:
            continue
        block = func.regions[0].blocks[0]
        # Actual SSA uses include captures inside live/unknown scalar regions
        # and uses in other blocks; a local dialect-name liveness approximation
        # must never hide those edges. Reverse erasure releases dead consumers
        # before considering their producers.
        for op in reversed(list(block.ops)):
            if op.results and all(not result.uses for result in op.results) and _dead_tensor_operation_is_pure(op):
                # DCE must not turn malformed/missing source semantics into an
                # apparently valid program by erasing the invalid operation.
                try:
                    op.verify()
                except VerifyException:
                    continue
                block.erase_op(op)
                erased += 1
    return erased
