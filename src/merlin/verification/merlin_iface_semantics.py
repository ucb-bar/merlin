"""A deliberately narrow value semantics for the frozen ``merlin_iface`` text grammar.

The OOT compiler consumes/produces this contract text, while the in-tree proof interpreter reads
``interface.*`` xDSL ops. Convert the custom assembly to MLIR's generic form, parse the resulting
SSA graph, then interpret only tensor leaves, resident pack, matmul, commit, and evict. The generic
form conversion is spelling only; every parsed operation and attribute is checked here. A new
mnemonic, an unknown attribute, or an arbitrary leaf alias abstains.

This proves integer VALUES at a concrete shape. ``resident_pack`` is value preserving in this
algebra; physical layout, DMA, and device execution require separate checks.
"""

from __future__ import annotations

from typing import Any

from .smt_semantics import Encoded, Encoder, Tensor, UnsupportedSemantics, _elem_width, _shape


def parse_merlin_iface(text: str):
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.parser import Parser

    from merlin.targetgen.contract.interface_emit import to_generic_form

    try:
        generic = to_generic_form(text)
        ctx = Context(allow_unregistered=True)
        ctx.load_dialect(Builtin)
        return Parser(ctx, generic).parse_module()
    except Exception as exc:
        # xDSL parser errors are checker abstentions. The original text is still hashed in the
        # receipt, so a failed conversion cannot be mistaken for the successfully parsed artifact.
        raise UnsupportedSemantics(f"merlin_iface text cannot be parsed completely: {exc}") from exc


def _string_attr(attrs: dict, key: str) -> str:
    from xdsl.dialects.builtin import StringAttr

    value = attrs.get(key)
    if not isinstance(value, StringAttr):
        raise UnsupportedSemantics(f"merlin_iface {key!r} must be a string attribute")
    return value.data


def _kind(op) -> str:
    return _string_attr(op.attributes, "op_name__")


def _only_attrs(op, allowed: set[str]) -> None:
    unknown = set(op.attributes) - allowed - {"op_name__"}
    if unknown or op.properties or op.regions:
        raise UnsupportedSemantics(f"{_kind(op)} has unencoded attributes/properties/regions: {sorted(unknown)}")


def merlin_iface_signature(text: str) -> dict[str, Any]:
    """Typed declaration summary for receipts; the semantic walk remains the authority."""
    module = parse_merlin_iface(text)
    leaves = []
    outputs = []
    for op in module.body.block.ops:
        name = _kind(op)
        if name == "merlin_iface.tensor" and len(op.results) == 1:
            leaves.append({"name": _string_attr(op.attributes, "name"), "type": str(op.results[0].type)})
        elif name == "merlin_iface.commit" and len(op.results) == 1:
            outputs.append({"name": _string_attr(op.attributes, "name"), "type": str(op.results[0].type)})
    return {"grammar": "merlin_iface", "inputs": leaves, "outputs": outputs}


def encode_merlin_iface(enc: Encoder, module, *, shared: list[Tensor], acc_width: int = 32) -> Encoded:
    """Interpret actual ``merlin_iface`` SSA, binding ``name=argN`` to source argument N."""
    from xdsl.dialects.builtin import ArrayAttr

    if set(module.attributes) != {"merlin_iface.version", "merlin_iface.target", "merlin_iface.abi_version"}:
        raise UnsupportedSemantics("merlin_iface module metadata differs from the frozen grammar")
    if _string_attr(module.attributes, "merlin_iface.version") != "0.1":
        raise UnsupportedSemantics("unsupported merlin_iface grammar version")
    _string_attr(module.attributes, "merlin_iface.target")
    _string_attr(module.attributes, "merlin_iface.abi_version")
    if len(module.body.blocks) != 1:
        raise UnsupportedSemantics("merlin_iface module must have one block")
    if module.properties or module.body.block.args:
        raise UnsupportedSemantics("merlin_iface module has unencoded properties or block arguments")

    env: dict[Any, Tensor] = {}
    resident: set[Any] = set()
    declared: set[int] = set()
    outputs: dict[str, Tensor] = {}
    for op in module.body.block.ops:
        name = _kind(op)
        if name == "merlin_iface.tensor":
            _only_attrs(op, {"name", "role"})
            if op.operands or len(op.results) != 1:
                raise UnsupportedSemantics("merlin_iface.tensor must declare one leaf value")
            label = _string_attr(op.attributes, "name")
            if _string_attr(op.attributes, "role") not in ("input", "weight"):
                raise UnsupportedSemantics("merlin_iface.tensor role is not an input or weight")
            if not label.startswith("arg") or not label[3:].isdigit() or label != f"arg{int(label[3:])}":
                raise UnsupportedSemantics(f"merlin_iface leaf {label!r} has no source argument binding")
            index = int(label[3:])
            if index >= len(shared) or index in declared:
                raise UnsupportedSemantics(f"merlin_iface leaf {label!r} is duplicate or outside source arguments")
            shape = _shape(op.results[0].type)
            width = _elem_width(op.results[0].type)
            leaf = shared[index]
            if (leaf.rows, leaf.cols, leaf.width) != (*shape, width):
                raise UnsupportedSemantics(f"merlin_iface leaf {label!r} type disagrees with source argument")
            env[op.results[0]] = leaf
            declared.add(index)
        elif name == "merlin_iface.resident_pack":
            _only_attrs(op, {"layout"})
            if len(op.operands) != 1 or len(op.results) != 1:
                raise UnsupportedSemantics("merlin_iface.resident_pack expects one tensor and one handle")
            _string_attr(op.attributes, "layout")
            if (
                str(op.results[0].type) != "!merlin_iface.resident"
                or op.operands[0] not in env
                or not str(op.operands[0].type).startswith("tensor<")
            ):
                raise UnsupportedSemantics("merlin_iface.resident_pack has an unbound source or wrong result type")
            env[op.results[0]] = env[op.operands[0]]
            resident.add(op.results[0])
        elif name == "merlin_iface.matmul":
            _only_attrs(op, set())
            if len(op.operands) != 2 or len(op.results) != 1:
                raise UnsupportedSemantics("merlin_iface.matmul expects two operands and one accumulator")
            lhs, rhs = op.operands
            if (
                lhs not in env
                or rhs not in resident
                or not str(lhs.type).startswith("tensor<")
                or str(rhs.type) != "!merlin_iface.resident"
                or str(op.results[0].type) != f"!merlin_iface.acc<i{acc_width}>"
            ):
                raise UnsupportedSemantics("merlin_iface.matmul has an unbound operand or wrong accumulator type")
            env[op.results[0]] = enc.matmul(env[lhs], env[rhs], acc_width=acc_width)
        elif name == "merlin_iface.commit":
            _only_attrs(op, {"name", "epilogue", "output_dtype"})
            if (
                len(op.operands) != 1
                or len(op.results) != 1
                or op.operands[0] not in env
                or str(op.operands[0].type) != f"!merlin_iface.acc<i{acc_width}>"
            ):
                raise UnsupportedSemantics("merlin_iface.commit has an unbound accumulator or wrong arity")
            epilogue = op.attributes.get("epilogue")
            if not isinstance(epilogue, ArrayAttr) or epilogue.data:
                raise UnsupportedSemantics("merlin_iface.commit epilogue is not an empty list")
            if _string_attr(op.attributes, "output_dtype") != f"i{acc_width}":
                raise UnsupportedSemantics("merlin_iface.commit output dtype is not the accumulator dtype")
            value = env[op.operands[0]]
            if (*_shape(op.results[0].type), _elem_width(op.results[0].type)) != (value.rows, value.cols, value.width):
                raise UnsupportedSemantics("merlin_iface.commit result type disagrees with the accumulator")
            label = _string_attr(op.attributes, "name")
            if label in outputs:
                raise UnsupportedSemantics(f"merlin_iface.commit output {label!r} appears twice")
            outputs[label] = value
            env[op.results[0]] = value
        elif name == "merlin_iface.evict":
            _only_attrs(op, set())
            if len(op.operands) != 1 or op.results or op.operands[0] not in resident:
                raise UnsupportedSemantics("merlin_iface.evict references no live resident handle")
            resident.remove(op.operands[0])
        else:
            raise UnsupportedSemantics(f"no value semantics for {name!r} in merlin_iface")
    if declared != set(range(len(shared))):
        raise UnsupportedSemantics("merlin_iface does not declare every source argument exactly once")
    if len(outputs) != 1:
        raise UnsupportedSemantics(
            f"merlin_iface value proof requires exactly one committed output; found {len(outputs)}"
        )
    return Encoded(outputs=outputs, inputs=shared)


# Preserve historical object lookup; implementation and completion state live here.
for _compat_definition in (
    parse_merlin_iface,
    _string_attr,
    _kind,
    _only_attrs,
    merlin_iface_signature,
    encode_merlin_iface,
):
    _compat_definition.__module__ = "merlin.verify.merlin_iface_semantics"
del _compat_definition
