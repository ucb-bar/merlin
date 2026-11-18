"""Explicit provider source translators preserve ordered SSA boundary types."""
from .linalg_recognition import LinalgBridgeError, _type


def operation_name(op):
    return op.attributes["op_name__"].data if op.name == "builtin.unregistered" else op.name


def translate_registered_source(op, *, translator, ordinal, env, nodes):
    if op.regions or op.successors:
        raise LinalgBridgeError("registered source operation must be regionless and successor-free")
    operand_ids = tuple(env.get(value) for value in op.operands)
    if any(value is None for value in operand_ids):
        raise LinalgBridgeError("registered source operation reads an undefined tensor")
    result_ids = tuple(f"op{ordinal}r{index}" for index in range(len(op.results)))
    translated = tuple(translator(op, operand_ids, result_ids))
    if tuple(node.id for node in translated) != result_ids:
        raise LinalgBridgeError("registered source translator must preserve every ordered result")
    if any(
        node.inputs != operand_ids or node.type != _type(value.type)
        or node.effect != "pure" or node.index_maps
        for node, value in zip(translated, op.results, strict=True)
    ):
        raise LinalgBridgeError("registered source translator changed operands, result types, or effects")
    nodes.extend(translated)
    env.update(zip(op.results, result_ids, strict=True))
