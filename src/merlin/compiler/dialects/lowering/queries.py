"""Read linalg operands shared by lowering analyses and stages."""

def find_matmuls(module):
    """All linalg matmul-family ops in the module (quantized + plain)."""
    from xdsl.dialects.linalg import ops as linalg_ops

    found = []
    for op in module.walk():
        if isinstance(op, (linalg_ops.QuantizedMatmulOp, linalg_ops.MatmulOp)):
            found.append(op)
    return found


def matmul_lhs_rhs(mm):
    """(lhs, rhs) SSA values of a linalg matmul-family op (inputs[0], inputs[1])."""
    return mm.inputs[0], mm.inputs[1]


# linalg named ops that map onto a runtime VECTOR_MAP combine. `linalg.sub` is deliberately absent:
# the runtime implements add and mul only, and lowering a subtract as an add would be a miscompile.
ELEMENTWISE_COMBINES = {"linalg.add": "add", "linalg.mul": "mul"}


def find_elementwise(module):
    """All linalg elementwise ops the interface layer can materialize, as (op, combine) pairs."""
    return [(op, ELEMENTWISE_COMBINES[op.name]) for op in module.walk() if op.name in ELEMENTWISE_COMBINES]


def elementwise_operands(op):
    """(lhs, rhs) SSA values of a linalg elementwise op."""
    return op.inputs[0], op.inputs[1]
