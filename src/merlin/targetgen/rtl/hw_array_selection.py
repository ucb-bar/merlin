"""Bounded typed array-create/get dependencies and local known-bit selection.

CIRCT creation operands are MSB first: runtime index zero selects the last
operand. A dependency roster establishes no state, memory-history or role fact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin, IntegerType, UnregisteredAttr
from xdsl.dialects.hw import HW, ArrayType
from xdsl.ir import OpResult, SSAValue
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError, VerifyException

from .hw_observations import _name


@dataclass(frozen=True)
class ArraySelectionLimits:
    operations: int
    elements: int
    aggregate_bits: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("array selection requires explicit positive whole-roster budgets")


def _array_type(typ):
    if isinstance(typ, UnregisteredAttr) and typ.attr_name.data == "hw.array" and typ.is_type.data:
        context = Context()
        context.load_dialect(Builtin)
        context.load_dialect(HW)
        try:
            typ = Parser(context, str(typ)).parse_type()
        except (ParseError, VerifyException) as error:
            raise ValueError("array selection has an unsupported original array type") from error
    if not isinstance(typ, ArrayType) or len(typ) <= 0:
        raise ValueError("array selection requires an original positive fixed array type")
    return typ


def _scalar(typ, scalar_bits):
    if not isinstance(typ, IntegerType) or typ != IntegerType(typ.width.data) or not 0 < typ.width.data <= scalar_bits:
        raise ValueError("array selection refuses nested, opaque or unbounded scalar types")
    return typ.width.data


def _fields(op):
    if op.regions or len(op.results) != 1:
        raise ValueError("array selection requires one exact original result and no regions")
    if set(op.attributes) & set(op.properties):
        raise ValueError("array selection has ambiguous attribute/property ownership")
    if (set(op.attributes) | set(op.properties)) - {"op_name__", "sv.namehint"}:
        raise ValueError("array selection has unsupported original attributes")


def preflight_array_selections(definitions, occurrence_modules, *, limits, scalar_bits):
    """Charge every original rooted creation/get before allocating scalar rosters.

    Definitions are already bounded original module bodies; occurrence identities
    remain distinct. Unsupported aggregate types are retained by the caller.
    """
    if type(limits) is not ArraySelectionLimits or type(scalar_bits) is not int or scalar_bits <= 0:
        raise ValueError("array preflight requires explicit original type and aggregate budgets")
    totals = {"operations": 0, "elements": 0, "aggregate_bits": 0}
    for module, occurrences in Counter(occurrence_modules).items():
        for op in definitions[module]:
            kind = _name(op)
            if kind not in {"hw.array_create", "hw.array_get"}:
                continue
            totals["operations"] += occurrences
            # Count actual creation slots even when the declared type is opaque.
            count = len(op.operands) if kind == "hw.array_create" else 0
            bits = 0
            typ = op.results[0].type if kind == "hw.array_create" and op.results else None
            if kind == "hw.array_get" and op.operands:
                typ = op.operands[0].type
            try:
                array = _array_type(typ)
                count = max(count, len(array))
                bits = len(array) * _scalar(array.get_element_type(), scalar_bits)
            except ValueError:
                pass
            totals["elements"] += occurrences * count
            totals["aggregate_bits"] += occurrences * bits
            if any(totals[field] > getattr(limits, field) for field in totals):
                raise ValueError("complete original array roster exceeds its pre-expansion budget")
    return totals


@dataclass(frozen=True)
class TypedArraySelection:
    index: SSAValue
    elements: tuple[SSAValue, ...]
    original_creation_operands: tuple[SSAValue, ...]
    element_width: int
    index_width: int

    @property
    def full_index_domain_defined(self):
        return (1 << self.index_width) == len(self.elements)

    def evaluate(self, original_creation_values, index):
        """Select only an explicit in-domain known-bit input; never fill a hole."""
        if type(original_creation_values) is not tuple or len(original_creation_values) != len(self.elements):
            raise ValueError("array values require every original creation operand in original order")
        if any(type(v) is not int or not 0 <= v < 1 << self.element_width for v in original_creation_values):
            raise ValueError("array values must preserve the exact original scalar bitvector type")
        if type(index) is not int or not 0 <= index < 1 << self.index_width or index >= len(self.elements):
            raise ValueError("array index is outside the defined original selection domain")
        return original_creation_values[len(self.elements) - 1 - index]


def typed_array_selection(op, *, scalar_bits, limits):
    """Retain exact single-level creation/get typing and runtime index order.

    Callers preflight the whole selected original roster first. The local bounds
    are checked again before tuple expansion; non-power-of-two index domains
    remain conditional, and evaluate refuses every out-of-range input.
    """
    if type(limits) is not ArraySelectionLimits or type(scalar_bits) is not int or scalar_bits <= 0:
        raise ValueError("typed array selection requires explicit type and expansion budgets")
    if limits.operations < 2:
        raise ValueError("array selection exceeds its creation/get operation budget")
    _fields(op)
    if _name(op) != "hw.array_get" or len(op.operands) != 2:
        raise ValueError("array selection requires the complete original array/index operand roster")
    value, index = op.operands
    array = _array_type(value.type)
    width = _scalar(array.get_element_type(), scalar_bits)
    index_width = _scalar(index.type, scalar_bits)
    if index_width != (len(array) - 1).bit_length() or op.results[0].type != array.get_element_type():
        raise ValueError("array selection index/result differs from its exact original declared type")
    if len(array) > limits.elements or len(array) * width > limits.aggregate_bits:
        raise ValueError("array selection exceeds its pre-expansion aggregate budget")
    if not isinstance(value, OpResult) or value.index != 0 or _name(value.owner) != "hw.array_create":
        raise ValueError("array selection retains opaque or unsupported original aggregate producers")
    create = value.owner
    _fields(create)
    if create.parent is not op.parent or len(create.operands) != len(array):
        raise ValueError("array creation differs from its exact original local element roster")
    if any(element.type != array.get_element_type() for element in create.operands):
        raise ValueError("array creation element types differ from the original array element type")
    # Public CIRCT HW/Comb rationale: lexical creation operands are MSB to LSB.
    original = tuple(create.operands)
    return TypedArraySelection(index, tuple(reversed(original)), original, width, index_width)
