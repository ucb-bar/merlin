"""Conservative bounds for actual counted LLVM index emission.

These bounds specialize host control only. They say nothing about payload values,
physical pointer aliases, hardware effects, or completion. Unknown arithmetic and
signed overflow discard the bound instead of granting a scheduling permission.
"""

from __future__ import annotations

from .llvm_program import FnBuilderControlAdapter

_MIN = -(1 << 63)
_MAX = (1 << 63) - 1


def _bounded(low, high):
    return (low, high) if _MIN <= low <= high <= _MAX else None


class FnBuilderBoundedControlAdapter(FnBuilderControlAdapter):
    """Track inclusive i64 intervals through the ordinary real LLVM builder.

    Only constants and actual ascending counted-loop indices introduce facts.
    There is deliberately no API for a caller to assert a range for an unknown
    argument. Branch predicates do not refine ranges. The emitted arithmetic and
    loops are exactly the ordinary control adapter's operations.
    """

    def __init__(self, builder):
        super().__init__(builder)
        self._bounds = {}

    def bounds(self, value):
        if type(value) is int:
            return _bounded(value, value)
        return self._bounds.get(value)

    def _remember(self, value, bounds):
        if bounds is not None:
            self._bounds[value] = bounds
        return value

    def value(self, value):
        return self._remember(super().value(value), self.bounds(value))

    def op(self, opcode, lhs, rhs, dtype="i64"):
        result = super().op(opcode, lhs, rhs, dtype)
        left, right = self.bounds(lhs), self.bounds(rhs)
        bounds = None
        if dtype == "i64" and left is not None and right is not None:
            a, b = left
            c, d = right
            if opcode == "add":
                bounds = _bounded(a + c, b + d)
            elif opcode == "sub":
                bounds = _bounded(a - d, b - c)
            elif opcode == "mul":
                products = (a * c, a * d, b * c, b * d)
                bounds = _bounded(min(products), max(products))
        return self._remember(result, bounds)

    def select(self, condition, lhs, rhs):
        result = super().select(condition, lhs, rhs)
        left, right = self.bounds(lhs), self.bounds(rhs)
        if left is not None and right is not None:
            self._remember(result, (min(left[0], right[0]), max(left[1], right[1])))
        return result

    def loop(self, start, stop, step, callback):
        def body(index):
            if start < stop:
                last = start + ((stop - start - 1) // step) * step
                self._remember(index, _bounded(start, last))
            callback(index)

        super().loop(start, stop, step, body)
