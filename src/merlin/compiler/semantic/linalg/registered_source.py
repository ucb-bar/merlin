"""Explicit provider source translators preserve ordered SSA boundary types."""

from merlin.compiler.semantic.linalg.linalg_recognition import LinalgBridgeError, _type
from merlin.compiler.semantic.model import SemanticNode


def operation_name(op):
    from xdsl.dialects.builtin import StringAttr

    if op.name != "builtin.unregistered":
        return op.name
    name = op.attributes.get("op_name__")
    if not isinstance(name, StringAttr) or not name.data:
        raise LinalgBridgeError("unregistered source operation has no name")
    return name.data


def translate_registered_source(op, *, translator, ordinal, env, nodes):
    if op.regions or op.successors:
        raise LinalgBridgeError("registered source operation must be regionless and successor-free")
    operand_ids = tuple(env.get(value) for value in op.operands)
    if any(value is None for value in operand_ids):
        raise LinalgBridgeError("registered source operation reads an undefined tensor")
    result_ids = tuple(f"op{ordinal}r{index}" for index in range(len(op.results)))
    translated = tuple(translator(op, operand_ids, result_ids))
    if any(type(node) is not SemanticNode for node in translated):
        raise LinalgBridgeError("registered source translator must return typed semantic nodes")
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
