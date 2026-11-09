"""Recognize a closed source x * sigmoid(x) region without changing arithmetic."""

from .model import KernelRequest, pointwise_identity_maps


def recognize_silu(source: KernelRequest, output: str) -> tuple[str, str]:
    """Return the two computations; target eligibility remains a separate check.

    Keep the original graph and its source floating-point semantics. A backend
    may fuse this region only under its selected numerical and storage policy.
    The sigmoid cannot have another user or escape as a separate output.
    """
    result = source.node(output)
    if (result.op != "f32_mul" or result.effect != "pure" or len(result.inputs) != 2
            or result.type.dtype != "f32" or result.type.numerical_policy != "raw-bit-pattern"
            or dict(result.attrs) != {"numeric_semantics": "ieee754-f32-mul-rne-v1"}
            or result.index_maps != pointwise_identity_maps(len(result.type.shape), 2)):
        raise ValueError("SiLU source needs an exact pointwise f32 product")
    for i, key in enumerate(result.inputs):
        sigmoid = source.node(key)
        operand = source.node(result.inputs[1 - i])
        if (sigmoid.op == "source_f32_composite_unqualified" and sigmoid.effect == "pure"
                and dict(sigmoid.attrs) == {
                    "literal_bits": "0x3f800000", "numeric_semantics": "source-f32-composite-unqualified-v1",
                    "operation": "sigmoid_f32",
                }
                and sigmoid.inputs == (operand.id,) and sigmoid.type == operand.type == result.type
                and sigmoid.index_maps == pointwise_identity_maps(len(result.type.shape), 1)
                and sigmoid.id not in source.outputs
                and [node.id for node in source.nodes if sigmoid.id in node.inputs] == [result.id]):
            return sigmoid.id, result.id
    raise ValueError("SiLU source needs one private sigmoid of the other product operand")
