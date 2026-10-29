"""Closed structural data support; no evaluation, simplification or text fallback."""

from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineMap,
    AffineSymExpr,
)
from xdsl.ir.affine.affine_set import AffineConstraintExpr, AffineConstraintKind, AffineSet


def affine_payload(value):
    """Return an injective class-labelled closed record, or None if unsupported.

    Child expressions remain real objects and recurse through the same exact
    whitelist at the caller. Unknown subclasses never inherit permission.
    """
    kind = type(value)
    name = kind.__module__ + "." + kind.__qualname__
    if kind is AffineMap:
        return [name, value.num_dims, value.num_symbols, value.results]
    if kind is AffineSet:
        return [name, value.num_dims, value.num_symbols, value.constraints]
    if kind in (AffineDimExpr, AffineSymExpr):
        return [name, value.position]
    if kind is AffineConstantExpr:
        return [name, value.value]
    if kind is AffineBinaryOpExpr and type(value.kind) is AffineBinaryOpKind:
        return [name, value.kind, value.lhs, value.rhs]
    if kind is AffineConstraintExpr and type(value.kind) is AffineConstraintKind:
        return [name, value.kind, value.lhs, value.rhs]
    return None
