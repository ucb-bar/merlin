"""Structural reader: a parsed interface module -> a plain workload model.

Everything here walks real xDSL IR (ops, operands, results, typed attributes).  No text
scanning, no regular expressions: the only string handling is reading the *value* of a string
attribute the parser already produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    Builtin,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.parser import Parser

from . import iface_dialect as IF


class InterfaceError(Exception):
    """The interface module is not something this backend can accept."""


# --------------------------------------------------------------------------------------------
# attribute readers


def _py(attr: Any) -> Any:
    """A typed xDSL attribute as a plain Python value."""
    if isinstance(attr, StringAttr):
        return attr.data
    if isinstance(attr, IntegerAttr):
        return int(attr.value.data)
    if isinstance(attr, FloatAttr):
        return float(attr.value.data)
    if isinstance(attr, ArrayAttr):
        return [_py(a) for a in attr.data]
    return attr


def _attrs(op) -> dict[str, Any]:
    return {k: _py(v) for k, v in op.attributes.items()}


def _dtype_of(elem) -> str:
    if isinstance(elem, IntegerType):
        return f"i{elem.width.data}"
    return str(elem)


def _tensor_shape_dtype(typ) -> tuple[list[int], str]:
    if not isinstance(typ, TensorType):
        raise InterfaceError(f"expected a builtin tensor type, got {typ}")
    return [int(d) for d in typ.get_shape()], _dtype_of(typ.get_element_type())


# --------------------------------------------------------------------------------------------
# workload model


@dataclass
class TensorDecl:
    name: str
    shape: list[int]
    dtype: str
    role: str


@dataclass
class Node:
    """One interface operation, normalised."""

    kind: str                       # the mnemonic, e.g. "matmul"
    name: str                       # result name for producing ops ("" otherwise)
    ins: list[str] = field(default_factory=list)     # ssa-value keys of the operands
    out_shape: list[int] = field(default_factory=list)
    out_dtype: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workload:
    version: str
    target: str
    abi_version: str
    tensors: dict[str, TensorDecl]
    nodes: list[Node]
    #: ssa key -> the logical name it carries (leaf name, resident handle, accumulator, output)
    values: dict[str, str]

    def leaf(self, name: str) -> TensorDecl:
        return self.tensors[name]


def parse_module(text: str) -> ModuleOp:
    """Parse + verify interface MLIR against the typed `merlin_iface` dialect."""
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(IF.MERLIN_IFACE)
    module = Parser(ctx, text).parse_module()
    module.verify()
    return module


def is_merlin_iface(module: ModuleOp) -> bool:
    return "merlin_iface.version" in module.attributes


def read(module: ModuleOp) -> Workload:
    """Walk a verified `merlin_iface` module into the plain workload model."""
    mattrs = {k: _py(v) for k, v in module.attributes.items()}
    version = str(mattrs.get("merlin_iface.version", ""))
    if version != IF.GRAMMAR_VERSION:
        raise InterfaceError(
            f"merlin_iface version {version!r} is not implemented (this backend implements "
            f"{IF.GRAMMAR_VERSION!r})"
        )

    tensors: dict[str, TensorDecl] = {}
    nodes: list[Node] = []
    values: dict[str, str] = {}
    counter = {"acc": 0, "res": 0}

    def key(v) -> str:
        return values[_ssa(v)]

    def _ssa(v) -> str:
        return str(id(v))

    for op in module.body.ops:
        attrs = _attrs(op)
        if isinstance(op, IF.TensorOp):
            shape, dtype = _tensor_shape_dtype(op.res.type)
            name = attrs["name"]
            tensors[name] = TensorDecl(name, shape, dtype, attrs["role"])
            values[_ssa(op.res)] = name
            continue

        if isinstance(op, IF.ResidentPackOp):
            src = key(op.operands_[0])
            handle = f"{src}_res"
            values[_ssa(op.res)] = handle
            nodes.append(Node("resident_pack", handle, [src], attrs=attrs))
            counter["res"] += 1
            continue

        if isinstance(op, IF.EvictOp):
            nodes.append(Node("evict", "", [key(op.operands_[0])]))
            continue

        if isinstance(op, IF.MatmulOp):
            handle = f"acc{counter['acc']}"
            counter["acc"] += 1
            values[_ssa(op.res)] = handle
            nodes.append(Node("matmul", handle,
                              [key(v) for v in op.operands_], attrs=attrs))
            continue

        if isinstance(op, IF.CommitOp):
            shape, dtype = _tensor_shape_dtype(op.res.type)
            name = attrs["name"]
            values[_ssa(op.res)] = name
            nodes.append(Node("commit", name, [key(op.operands_[0])], shape, dtype, attrs))
            continue

        # every remaining op produces a named tensor from tensor operands
        if op.res is None:
            raise InterfaceError(f"{op.name}: unexpected op with no result")
        shape, dtype = _tensor_shape_dtype(op.res.type)
        name = attrs.get("name")
        if name is None:
            raise InterfaceError(f"{op.name}: missing `name`")
        values[_ssa(op.res)] = name
        kind = op.name.split(".", 1)[1]
        nodes.append(Node(kind, name, [key(v) for v in op.operands_], shape, dtype, attrs))

    return Workload(version, str(mattrs.get("merlin_iface.target", "")),
                    str(mattrs.get("merlin_iface.abi_version", "")),
                    tensors, nodes, values)
