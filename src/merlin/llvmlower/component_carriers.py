"""Carry exact integer components through element-preserving tensor index maps."""


def data_operand_indices(op):
    if op.name in {
        "tensor.reshape",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.cast",
        "tensor.extract_slice",
        "linalg.copy",
        "linalg.broadcast",
        "linalg.transpose",
    }:
        return [0]
    if op.name == "tensor.insert_slice":
        return [0, 1]
    if op.name == "tensor.concat":
        return list(range(len(op.operands)))
    if op.name == "linalg.generic" and len(op.inputs) == 1 and len(op.results) == 1:
        body = list(op.body.block.ops)
        if len(body) == 1 and body[0].name == "linalg.yield" and body[0].operands[0] is op.body.block.args[0]:
            return [0]
    return []


def carry(value, inputs, element, operations):
    """Clone a checked carrier at integer width, preserving every source index."""
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import TensorType
    from xdsl.rewriter import Rewriter

    op = value.owner
    indices = data_operand_indices(op)
    if not indices or len(op.results) != 1:
        raise ValueError("exact component carrier lacks one complete element-preserving map")
    mapper = dict(zip((op.operands[i] for i in indices), inputs, strict=True))
    if op.name.startswith("linalg."):
        for output in op.outputs:
            empty = tensor.EmptyOp([], TensorType(element, output.type.get_shape()))
            operations.append(empty)
            mapper[output] = empty.tensor
    cloned = op.clone(value_mapper=mapper)
    if cloned.name.startswith("linalg."):
        for argument in list(cloned.body.block.args):
            Rewriter.replace_value_with_new_type(argument, element)
    result = Rewriter.replace_value_with_new_type(cloned.results[0], TensorType(element, value.type.get_shape()))
    operations.append(cloned)
    return result
