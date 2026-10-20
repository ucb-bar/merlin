"""Expose an original generated descriptor-return identity to bufferization.

The original emitter must supply a separate, checked, void descriptor forwarder.
It must call the unchanged entry once and reject every returned descriptor field
that differs from the declared argument. This module establishes neither that
law nor full writes, noescape, release, completion, numerical or target effects.
Only fresh, unobserved destination initializers are admitted. Other calls stay
on their original route, including calls to arbitrary external functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

from xdsl.dialects import bufferization, func, tensor
from xdsl.dialects.builtin import (
    IntegerType,
    MemRefType,
    NoneAttr,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    UnitAttr,
)
from xdsl.dialects.linalg.ops import FillOp
from xdsl.ir import Block, Region
from xdsl.traits import SymbolTable

FEATURE = "descriptor_result_identity"


@dataclass(frozen=True)
class DescriptorResultIdentity:
    """Original emitter selection, not a certificate reconstructed from data.

    ``borrowed_symbol`` names its checked void ranked-C forwarder; the original
    result-bearing symbol and C guards remain available and unchanged. Exact
    argument types/access bind the current declaration. The host must join this
    selection to the actual emitted forwarder and its original implementation.
    """

    symbol: str
    borrowed_symbol: str
    wrapper_symbol: str
    result_argument: int
    argument_types: tuple[TensorType, ...]
    argument_access: tuple[str, ...]


def _metadata(operation, allowed=()):
    return all(name in allowed or name.startswith("prov.") for name in operation.attributes)


def _fresh_root(value):
    if sum(1 for _ in value.uses) != 1:
        return None
    owner = value.owner
    if isinstance(owner, FillOp):
        if (
            len(owner.inputs) != 1
            or len(owner.outputs) != 1
            or not _metadata(owner)
            or len(owner.body.blocks) != 1
            or len(owner.body.block.ops) != 1
            or owner.body.block.last_op.name != "linalg.yield"
            or tuple(owner.body.block.last_op.operands) != (owner.body.block.args[0],)
        ):
            return None
        value = owner.outputs[0]
        if sum(1 for _ in value.uses) != 1:
            return None
        owner = value.owner
    if not isinstance(owner, tensor.EmptyOp) or owner.operands or not _metadata(owner):
        return None
    return owner


def rewrite_descriptor_result_identities(module, identities, *, max_operations=100000):
    """Rewrite eligible calls only; retain initialization and opaque observers.

    Protect each selected empty root with standard ``alloc_tensor`` semantics.
    Otherwise pre-bufferization CSE can merge it with an unrelated live output
    initializer across the newly exposed alias. No scalar arithmetic is added or
    removed. The companion pipeline edit exposes derived equivalent results.
    """
    if type(max_operations) is not int or max_operations <= 0:
        raise ValueError("descriptor identity requires a positive operation bound")
    operations = tuple(islice(module.walk(), max_operations + 1))
    if len(operations) > max_operations:
        raise ValueError("descriptor identity source operation budget exceeded")
    identities = tuple(islice(identities, max_operations + 1))
    if len(identities) > max_operations:
        raise ValueError("descriptor identity selection operation budget exceeded")
    names = set()
    for operation in module.body.block.ops:
        name = operation.properties.get("sym_name", operation.attributes.get("sym_name"))
        if isinstance(name, StringAttr):
            if name.data in names:
                raise ValueError("ambiguous top-level descriptor symbol")
            names.add(name.data)
    selected = set()
    plans = []
    for identity in identities:
        if type(identity) is not DescriptorResultIdentity:
            raise TypeError("original typed descriptor identity required")
        symbols = (identity.symbol, identity.borrowed_symbol, identity.wrapper_symbol)
        types, access, result = identity.argument_types, identity.argument_access, identity.result_argument
        if (
            any(type(name) is not str or not name or len(name) > 256 for name in symbols)
            or len(set(symbols)) != 3
            or identity.symbol in selected
            or type(types) is not tuple
            or type(access) is not tuple
            or len(types) != 3
            or len(access) != 3
            or access[:2] != ("read", "read")
            or access[2] not in ("write", "read-write")
            or type(result) is not int
            or result != 2
            or any(
                not isinstance(typ, TensorType)
                or typ.encoding != NoneAttr()
                or len(typ.get_shape()) != 2
                or any(d <= 0 for d in typ.get_shape())
                or not isinstance(typ.get_element_type(), IntegerType)
                for typ in types
            )
        ):
            raise ValueError("unsupported original descriptor identity domain")
        selected.add(identity.symbol)
        if any(name in names for name in symbols[1:]):
            raise ValueError("descriptor identity wrapper or forwarder symbol collision")
        names.update(symbols[1:])
        declarations = [
            op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == identity.symbol
        ]
        if len(declarations) != 1:
            raise ValueError("descriptor identity requires one top-level declaration")
        declaration = declarations[0]
        attrs = declaration.arg_attrs
        if (
            declaration.body.blocks
            or set(declaration.properties) - {"sym_name", "sym_visibility", "function_type", "arg_attrs"}
            or tuple(declaration.function_type.inputs) != types
            or tuple(declaration.function_type.outputs) != (types[result],)
            or not _metadata(declaration, ("llvm.emit_c_interface",))
            or attrs is None
            or len(attrs) != len(types)
            or any(
                set(attr.data) != {"bufferization.access"} or attr.data["bufferization.access"] != StringAttr(expected)
                for attr, expected in zip(attrs, access, strict=True)
            )
        ):
            raise ValueError("descriptor identity differs from the original declaration")
        calls = []
        for operation in operations:
            if (
                not isinstance(operation, func.CallOp)
                or SymbolTable.lookup_symbol(operation, operation.callee) is not declaration
            ):
                continue
            if tuple(value.type for value in operation.arguments) != types or tuple(operation.result_types) != (
                types[result],
            ):
                raise ValueError("descriptor identity call differs from the original ABI")
            root = _fresh_root(operation.arguments[result])
            if root is not None and _metadata(operation) and set(operation.properties) == {"callee"}:
                calls.append((operation, root))
        if calls:
            plans.append((identity, declaration, calls))
    added = sum(8 + len(calls) for _, _, calls in plans)
    if len(operations) + added > max_operations:
        raise ValueError("descriptor identity construction operation budget exceeded")
    report = []
    for identity, declaration, calls in plans:
        types = identity.argument_types
        buffers = tuple(MemRefType(typ.get_element_type(), typ.get_shape()) for typ in types)
        block = Block(arg_types=types)
        arguments = []
        for index, (argument, typ) in enumerate(zip(block.args, buffers, strict=True)):
            properties = {} if index == identity.result_argument else {"read_only": UnitAttr()}
            operation = bufferization.ToBufferOp.build(operands=[argument], result_types=[typ], properties=properties)
            block.add_op(operation)
            arguments.append(operation.results[0])
        block.add_op(func.CallOp(identity.borrowed_symbol, arguments, []))
        result = bufferization.ToTensorOp(arguments[identity.result_argument], restrict=True, writable=True)
        block.add_ops([result, func.ReturnOp(result.tensor)])
        wrapper = func.FuncOp(
            identity.wrapper_symbol, (types, (types[identity.result_argument],)), Region(block), visibility="private"
        )
        external = func.FuncOp(identity.borrowed_symbol, (buffers, ()), Region(), visibility="private")
        external.attributes["llvm.emit_c_interface"] = UnitAttr()
        declaration.parent.insert_ops_before([external, wrapper], declaration)
        for call, root in calls:
            allocated = bufferization.AllocTensorOp(root.results[0].type)
            allocated.attributes.update(root.attributes)
            root.parent.insert_op_before(allocated, root)
            root.results[0].replace_all_uses_with(allocated.results[0])
            root.parent.erase_op(root)
            call.properties["callee"] = SymbolRefAttr(identity.wrapper_symbol)
        report.append(
            {
                "symbol": identity.symbol,
                "borrowed_symbol": identity.borrowed_symbol,
                "wrapper_symbol": identity.wrapper_symbol,
                "result_argument": identity.result_argument,
                "calls": len(calls),
                "protected_empty_roots": len(calls),
            }
        )
    return report


def edit_pipeline(passes):
    """Use upstream's actual body-derived identity before result conversion."""
    output = list(passes)
    bufferize = [i for i, value in enumerate(output) if value.partition("{")[0] == "one-shot-bufferize"]
    conversion = [i for i, value in enumerate(output) if value.partition("{")[0] == "buffer-results-to-out-params"]
    if len(bufferize) != 1 or len(conversion) != 1 or bufferize[0] >= conversion[0]:
        raise ValueError("descriptor identity needs one ordered bufferization/result conversion pair")
    if "drop-equivalent-buffer-results" not in output[bufferize[0] + 1 : conversion[0]]:
        output.insert(conversion[0], "drop-equivalent-buffer-results")
    return output


def ensure_registered():
    """Register the existing default-off pipeline hook; issue no selection."""
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                FEATURE,
                "PASS",
                "Expose checked descriptor-result body identity before output conversion.",
                edit_pipeline=edit_pipeline,
            )
        )
    return FEATURE


def implementation_paths():
    """Direct implementation dependencies; not an arbitrary closure/grant API."""
    import inspect
    from pathlib import Path

    from xdsl.ir import Operation

    from . import impr_features

    owners = (
        Operation,
        Block,
        Region,
        SymbolTable,
        func.FuncOp,
        bufferization.AllocTensorOp,
        bufferization.ToBufferOp.build,
        tensor.EmptyOp,
        TensorType,
        FillOp,
    )
    return tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                Path(impr_features.__file__).resolve(),
                *(Path(inspect.getsourcefile(owner)).resolve() for owner in owners),
            )
        )
    )
