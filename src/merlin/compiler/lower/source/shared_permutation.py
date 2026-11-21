"""Structural common-permutation proof for caller-proven uniform elementwise work.

This helper proves only layout compatibility. The caller must separately prove
that its arithmetic is uniform and commutes with a common permutation. It does
not mutate IR or erase transposes, so other uses of either operand stay live.
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import NoneAttr, TensorType
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.ir import SSAValue


def prove_shared_transpose(inputs: Sequence[SSAValue], logical_shape: Sequence[int]) -> dict:
    """Return a static, unencoded tensor layout proof, or refuse without edits.

    No divisibility or hardware tile constraints apply here. Tails and arbitrary
    ranks are legal; downstream implementations own their resource restrictions.
    """
    if not inputs:
        raise ValueError("shared permutation requires operands")
    proofs = []
    element_type = None
    for value in inputs:
        op = value.owner
        if not isinstance(op, TransposeOp):
            raise ValueError("operand lacks an explicit transpose")
        if len(op.operands) != 2 or len(op.results) != 1:
            raise ValueError("unexpected transpose arity")
        before, after = op.operands[0].type, value.type
        destination = op.operands[1].type
        if any(
            not isinstance(ty, TensorType) or not isinstance(ty.encoding, NoneAttr)
            for ty in (before, after, destination)
        ):
            raise ValueError("shared permutation requires unencoded tensors")
        if before.get_element_type() != after.get_element_type() or destination != after:
            raise ValueError("transpose element or destination type differs")
        if element_type is not None and after.get_element_type() != element_type:
            raise ValueError("operand element types disagree")
        element_type = after.get_element_type()
        a, b = tuple(before.get_shape()), tuple(after.get_shape())
        permutation = tuple(op.permutation.get_values())
        if any(x <= 0 for x in (*a, *b)) or sorted(permutation) != list(range(len(a))):
            raise ValueError("layout requires positive static axes and a permutation")
        if b != tuple(logical_shape) or tuple(a[i] for i in permutation) != b:
            raise ValueError("transpose shape proof differs")
        proofs.append(
            dict(
                kind="shared_explicit_transpose",
                permutation=list(permutation),
                physical_shape=list(a),
                logical_shape=list(b),
            )
        )
    if any(proof != proofs[0] for proof in proofs[1:]):
        raise ValueError("operand permutations or shapes disagree")
    return proofs[0]
