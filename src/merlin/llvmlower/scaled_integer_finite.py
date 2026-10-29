"""Source-derived finite domains for broadcast scaled integer producers.

This analysis supplies a conservative integer-bit scale guard. It changes no
source arithmetic, installs no route, and grants no alias or floating effect
permission. A provider must bind the runtime scan, source layout/lifetime, and
RNE mode before omitting a corresponding per-element finite check.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from fractions import Fraction

from .ordered_fma_groups import _context_snapshot, _snapshot, _supported
from .source_expression_interval import IntervalEffectContract, _numeric_context, _scalar_attributes

_MAX_WORD = 0x7F7FFFFF


def _exact_positive(word: int) -> Fraction:
    if type(word) is not int or not 0 <= word <= _MAX_WORD:
        raise ValueError("finite positive binary32 word required")
    exponent, fraction = word >> 23, word & 0x7FFFFF
    significand = fraction if exponent == 0 else fraction | 0x800000
    shift = -149 if exponent == 0 else exponent - 150
    return Fraction(significand) * Fraction(2) ** shift


def _floor_word(value: Fraction) -> int:
    if value < 0:
        raise ValueError("nonnegative finite-domain bound required")
    low, high = 0, _MAX_WORD
    while low < high:
        middle = (low + high + 1) // 2
        if _exact_positive(middle) <= value:
            low = middle
        else:
            high = middle - 1
    return low


@dataclass(frozen=True)
class ScaledIntegerFiniteDomain:
    """Original signed conversion and ordered constant multiplies, then scale."""

    integer_bits: int
    constant_words: tuple[int, ...]

    def scale_limit_word(self) -> int:
        if type(self.integer_bits) is not int or not 2 <= self.integer_bits <= 64:
            raise ValueError("signed integer width in [2,64] required")
        if not self.constant_words:
            raise ValueError("at least one original constant multiply required")
        # RN conversion of the entire signed domain is enclosed by +/-2^(w-1),
        # including the rounded positive maximum. Every subsequent rounded
        # intermediate is bounded above, without reassociating source products.
        bound = Fraction(2) ** (self.integer_bits - 1)
        maximum = _exact_positive(_MAX_WORD)
        for word in self.constant_words:
            if type(word) is not int or not 0 <= word < 1 << 32:
                raise ValueError("literal binary32 word required")
            exact = bound * _exact_positive(word & 0x7FFFFFFF)
            if exact > maximum:
                raise ValueError("constant chain may overflow before the variable scale")
            floor = _floor_word(exact)
            bound = _exact_positive(floor if _exact_positive(floor) == exact else floor + 1)
        return _MAX_WORD if bound == 0 else _floor_word(maximum / bound)

    def admits_scale_word(self, word: int) -> bool:
        if type(word) is not int or not 0 <= word < 1 << 32:
            raise ValueError("literal scale binary32 word required")
        return word & 0x7FFFFFFF <= self.scale_limit_word()


@dataclass(frozen=True)
class BroadcastScaledIntegerFiniteProof:
    value: object
    operation: object
    integer_operand: int
    scale_operand: int
    scale_axes: tuple[int, ...]
    domain_extents: tuple[int, ...]
    repetition: int
    domain: ScaledIntegerFiniteDomain
    _witness: tuple = field(repr=False)
    _contexts: tuple = field(repr=False)
    _effects: IntervalEffectContract = field(repr=False)


def prove_broadcast_scaled_integer_finite(value, *, effects: IntervalEffectContract):
    """Retain a typed cast/ordered-constant-multiply/broadcast-scale witness.

    Exact static parallel indexing maps establish the smaller immutable scale
    domain. Extra uses of original values remain legal: no value is replaced or
    moved. Empty/reduction/unknown-effect/strict contexts refuse preparation.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerType, Signedness, TensorType, f32
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import BlockArgument, OpResult
    from xdsl.ir.affine import AffineDimExpr
    from merlin.common.mlir_query import op_name

    from .bounded_rne_maps import _roundeven_intrinsic

    effects.validate()
    if not isinstance(value, OpResult) or value.type != f32:
        raise ValueError("typed binary32 producer required")
    block = value.owner.parent_block()
    parent = block.parent_region().parent_op() if block is not None else None
    if not isinstance(parent, linalg.GenericOp) or not _supported(parent):
        raise ValueError("nonempty static pure parallel tensor map required")
    if any(op_name(op).startswith(("func.", "llvm.")) and not _roundeven_intrinsic(op) for op in block.ops):
        raise ValueError("unknown call or effects in preparation consumer")
    if value.owner.name != "arith.mulf":
        raise ValueError("original final scale multiply required")
    scales = [v for v in value.owner.operands if isinstance(v, BlockArgument) and v.type == f32 and v.owner is block]
    if len(scales) != 1:
        raise ValueError("one tensor block-argument scale required")
    scale = scales[0]
    current = next(v for v in value.owner.operands if v is not scale)
    chain, words, constants = [], [], []
    while isinstance(current, OpResult) and current.owner.name == "arith.mulf":
        operation = current.owner
        _scalar_attributes(operation)
        matches = [
            v
            for v in operation.operands
            if isinstance(v, OpResult)
            and isinstance(v.owner, arith.ConstantOp)
            and isinstance(v.owner.value, FloatAttr)
            and v.type == f32
        ]
        if len(matches) != 1:
            raise ValueError("ordered single-literal multiply chain required")
        constant = matches[0].owner
        _scalar_attributes(constant)
        words.append(struct.unpack("<I", struct.pack("<f", constant.value.value.data))[0])
        constants.append(constant)
        chain.append(operation)
        current = next(v for v in operation.operands if v is not matches[0])
    if not isinstance(current, OpResult) or current.owner.name != "arith.sitofp":
        raise ValueError("original signed integer conversion required")
    conversion = current.owner
    _scalar_attributes(conversion)
    integer = conversion.operands[0]
    if (
        current.type != f32
        or not isinstance(integer, BlockArgument)
        or integer.owner is not block
        or not isinstance(integer.type, IntegerType)
        or integer.type.signedness.data == Signedness.UNSIGNED
    ):
        raise ValueError("source tensor signed integer argument required")
    if integer.index >= len(parent.inputs) or scale.index >= len(parent.inputs):
        raise ValueError("output initializer is not an immutable producer input")
    maps = [a.data for a in parent.indexing_maps]
    integer_map, scale_map, output_map = maps[integer.index], maps[scale.index], maps[-1]
    if integer_map != output_map or any(not isinstance(e, AffineDimExpr) for e in scale_map.results):
        raise ValueError("complete integer domain and exact scale projection required")
    axes = tuple(e.position for e in scale_map.results)
    if len(set(axes)) != len(axes):
        raise ValueError("scale projection duplicates a domain axis")
    output_type = parent.outputs[0].type
    assert isinstance(output_type, TensorType)
    extents = [0] * output_map.num_dims
    for expr, extent in zip(output_map.results, output_type.get_shape(), strict=True):
        extents[expr.position] = extent
    repetition = math.prod(extents[i] for i in range(len(extents)) if i not in axes)
    if repetition <= 1:
        raise ValueError("scale domain has no repeated source consumers")
    _scalar_attributes(value.owner)
    domain = ScaledIntegerFiniteDomain(integer.type.width.data, tuple(reversed(words)))
    domain.scale_limit_word()
    operations = tuple(dict.fromkeys([parent, *block.ops, conversion, *chain, *constants]))
    contexts = []
    for operation in operations:
        current_op = operation
        while current_op is not None:
            _numeric_context(current_op)
            if current_op not in contexts:
                contexts.append(current_op)
            current_op = current_op.parent_op()
    return BroadcastScaledIntegerFiniteProof(
        value,
        parent,
        integer.index,
        scale.index,
        axes,
        tuple(extents),
        repetition,
        domain,
        tuple(_snapshot(op) for op in operations),
        tuple(_context_snapshot(op) for op in contexts),
        effects,
    )


def validate_broadcast_scaled_integer_finite(proof):
    if not isinstance(proof, BroadcastScaledIntegerFiniteProof):
        raise ValueError("typed source finite-domain witness required")
    if any(_snapshot(w.operation) != w for w in proof._witness) or any(
        _context_snapshot(w.operation) != w for w in proof._contexts
    ):
        raise ValueError("source finite-domain operations/context changed")
    current = prove_broadcast_scaled_integer_finite(proof.value, effects=proof._effects)
    fields = ("operation", "integer_operand", "scale_operand", "scale_axes", "domain_extents", "repetition", "domain")
    if any(getattr(current, name) != getattr(proof, name) for name in fields):
        raise ValueError("source finite-domain semantic fields changed")


def emit_finite_scale_scan(domain: ScaledIntegerFiniteDomain, *, symbol: str):
    """Portable read-only integer-bit scan; caller binds exact spans/lifetime.

    No FP instruction, rounding-mode change, or pointer cache is introduced.
    Negative/overflowing layouts refuse. Empty extent performs no input read.
    A true result is valid only with the separately bound RNE source contract.
    """
    if (
        not isinstance(domain, ScaledIntegerFiniteDomain)
        or not symbol
        or not symbol.isascii()
        or not symbol.isidentifier()
    ):
        raise ValueError("typed finite domain and explicit C identifier required")
    limit = domain.scale_limit_word()
    return f"""#include <stdint.h>
#include <stddef.h>
int {symbol}(const float *scale,size_t count,ptrdiff_t stride){{
 if(!count)return 1;
 if(!scale||stride<0||(stride&&(count-1)>(size_t)PTRDIFF_MAX/(size_t)stride))return 0;
 for(size_t i=0;i<count;i++){{uint32_t word;__builtin_memcpy(&word,scale+(ptrdiff_t)i*stride,4);
  if((word&0x7fffffffu)>0x{limit:08x}u)return 0;}}
 return 1;
}}
"""
