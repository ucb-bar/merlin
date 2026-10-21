"""Join routing demands to complete, independently numbered source operations."""

from merlin.common import mlir_query as mq


def indexed_demand_operations(operations):
    """Retain full source indices while omitting only recognized fill initializers.

    The demand reader omits carriers tagged ``fill``. Those are not a license to
    omit arbitrary arithmetic: only a literal constant, its tensor splat, or an
    actual linalg fill can be outside the routing census. Callers must still
    account for *all* operations in source bindings and task ownership.

    Untagged operations remain in this inventory. If the demand reader missed
    one, exact operation identity/count checks must refuse the incomplete join.
    """
    indexed = []
    for index, operation in enumerate(operations):
        name = mq.op_name(operation)
        tag = mq.attr_str(operation, "prov.op")
        if tag == "fill" and name not in ("arith.constant", "tensor.splat", "linalg.fill"):
            raise ValueError(f"source operation {index} ({name}) has an unsupported omitted fill carrier")
        if name == "linalg.fill" or tag == "fill":
            continue
        indexed.append((index, operation))
    return indexed
