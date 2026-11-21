"""Preserve caller-certified integer value bounds through verified tensor views.

Facts are keyed by actual SSA results, never symbols or attributes alone. The
caller must prove every seeded producer's arithmetic/effects and bind its source
certificate. Unknown signed-byte producers conservatively retain their type
range. This helper performs no target dispatch or graph mutation.
"""

from __future__ import annotations

import math

from xdsl.dialects.builtin import NoneAttr, TensorType, i8
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.dialects.tensor import CollapseShapeOp, ExpandShapeOp
from xdsl.ir import SSAValue


class SignedByteDomain:
    def __init__(self, value: SSAValue, minimum: int, maximum: int, certificate: dict):
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.certificate = certificate
        self.require()

    def require(self):
        if not isinstance(self.value.type, TensorType) or self.value.type.get_element_type() != i8:
            raise ValueError("signed-byte tensor producer required")
        if not isinstance(self.value.type.encoding, NoneAttr) or any(dim <= 0 for dim in self.value.type.get_shape()):
            raise ValueError("positive static unencoded tensor producer required")
        if (
            any(type(bound) is not int for bound in (self.minimum, self.maximum))
            or not -128 <= self.minimum <= self.maximum <= 127
        ):
            raise ValueError("source interval must be inside the complete signed-byte type")
        if not isinstance(self.certificate, dict) or not self.certificate:
            raise ValueError("explicit caller-proven producer certificate required")
        if self.certificate.get("minimum") != self.minimum or self.certificate.get("maximum") != self.maximum:
            raise ValueError("producer certificate contradicts interval endpoints")


def trace_signed_byte_domain(value: SSAValue, facts=()):
    """Return a bound on an actual value, refusing invalid view witnesses."""
    mapping = {}
    for fact in facts:
        if not isinstance(fact, SignedByteDomain):
            raise ValueError("typed caller-certified domain facts required")
        fact.require()
        if fact.value in mapping:
            raise ValueError("duplicate or contradictory producer domain facts")
        mapping[fact.value] = fact

    def visit(current, active):
        if current in active:
            raise ValueError("cyclic tensor-domain source chain")
        ty = current.type
        if not isinstance(ty, TensorType) or ty.get_element_type() != i8 or not isinstance(ty.encoding, NoneAttr):
            raise ValueError("unencoded signed-byte tensor use required")
        if any(dim <= 0 for dim in ty.get_shape()):
            raise ValueError("static positive tensor use required")
        if current in mapping:
            fact = mapping[current]
            return dict(
                minimum=fact.minimum,
                maximum=fact.maximum,
                reason="caller-proven typed producer",
                certificate=fact.certificate,
                producer_shape=list(ty.get_shape()),
                path=[],
            )
        owner = current.owner
        if isinstance(owner, (CollapseShapeOp, ExpandShapeOp, TransposeOp)):
            owner.verify()
            if len(owner.results) != 1 or current is not owner.results[0]:
                raise ValueError("view result identity differs")
            source = owner.operands[0]
            before = source.type
            if not isinstance(before, TensorType) or before.get_element_type() != ty.get_element_type():
                raise ValueError("view changes integer element type")
            if not isinstance(before.encoding, NoneAttr) or any(dim <= 0 for dim in before.get_shape()):
                raise ValueError("view source is not static and unencoded")
            if math.prod(before.get_shape()) != math.prod(ty.get_shape()):
                raise ValueError("view does not preserve all source elements")
            result = visit(source, active | {current})
            witness = dict(
                operation=owner.name, source_shape=list(before.get_shape()), result_shape=list(ty.get_shape())
            )
            if isinstance(owner, TransposeOp):
                permutation = list(owner.permutation.get_values())
                if (
                    sorted(permutation) != list(range(len(before.get_shape())))
                    or tuple(before.get_shape()[i] for i in permutation) != ty.get_shape()
                ):
                    raise ValueError("transpose permutation and shape differ")
                witness["permutation"] = permutation
            return dict(result, path=result["path"] + [witness])
        return dict(minimum=-128, maximum=127, reason="complete signed-byte type; producer unknown", path=[])

    return visit(value, set())
