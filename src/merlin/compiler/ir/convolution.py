"""Typed original convolution witnesses and shared pure-IR predicates.

These records describe source and emitted tensor regions; they grant no allocation,
target, execution or timing evidence. Compiler transforms and independent replay
consume the same types without importing one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoundedIm2ColSource:
    """Original typed source witness, frozen before the old island is erased."""

    geometry: tuple[int, ...]
    source_ir: str
    input: Any
    weight: Any
    initializer: Any
    input_type: str
    gather_type: str

    @property
    def f(self):
        return self.geometry[10]

    @property
    def k(self):
        return self.geometry[11]

    @property
    def m(self):
        return self.geometry[12]

    @property
    def identity(self):
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(
                (self.geometry, self.source_ir, self.input_type, self.gather_type), separators=(",", ":")
            ).encode()
        ).hexdigest()


@dataclass(frozen=True)
class BoundedIm2ColArm:
    width: int
    first_ow: int
    panels_per_row: int
    loop: Any
    packing_empty: Any
    patch: Any
    gather: Any
    contraction: Any
    output_empty: Any
    output_fill: Any
    insert: Any


@dataclass(frozen=True)
class BoundedIm2ColRewrite:
    source: BoundedIm2ColSource
    source_identity: str
    width: int
    operations: tuple[Any, ...]
    arms: tuple[BoundedIm2ColArm, ...]
    result: Any


def _bounded_plain(op, properties=()):
    return (
        all(key.startswith("prov.") for key in op.attributes)
        and set(op.properties) <= set(properties)
        and not op.successors
    )


def _is_copy_body(op) -> bool:
    """The gather's body is a pure element copy: one ``linalg.yield`` of the input block argument."""
    from xdsl.dialects.linalg.ops import YieldOp

    block = op.body.block
    ops = list(block.ops)
    if len(ops) != 1 or not isinstance(ops[0], YieldOp):
        return False
    return len(ops[0].operands) == 1 and ops[0].operands[0] is block.args[0]


def _reassoc(groups: "list[list[int]]"):  # noqa: UP037 -- Preserve the original annotation.
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, i64

    return ArrayAttr([ArrayAttr([IntegerAttr(j, i64) for j in g]) for g in groups])


BoundedIm2ColSource.__module__ = "merlin.compiler.lower.model.im2col_pack"
BoundedIm2ColArm.__module__ = "merlin.compiler.lower.model.im2col_pack"
BoundedIm2ColRewrite.__module__ = "merlin.compiler.lower.model.im2col_pack"
