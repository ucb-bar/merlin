"""Conservative integer sum-of-products domains supplied by a typed producer.

The caller must bind these facts to the actual producer, output storage and
consumer. This arithmetic certificate does not establish that source binding.
It admits exact signed-i32 accumulation only, with every prefix in range.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegerSumProductsRange:
    terms: int
    lhs_min: int
    lhs_max: int
    rhs_min: int
    rhs_max: int
    initial_min: int = 0
    initial_max: int = 0

    def interval(self):
        values = (
            self.terms,
            self.lhs_min,
            self.lhs_max,
            self.rhs_min,
            self.rhs_max,
            self.initial_min,
            self.initial_max,
        )
        if any(type(value) is not int for value in values) or self.terms < 0:
            raise ValueError("integer bounds and nonnegative term count required")
        for low, high in (
            (self.lhs_min, self.lhs_max),
            (self.rhs_min, self.rhs_max),
            (self.initial_min, self.initial_max),
        ):
            if not -(1 << 31) <= low <= high < (1 << 31):
                raise ValueError("ordered signed-i32 source bounds required")
        products = [a * b for a in (self.lhs_min, self.lhs_max) for b in (self.rhs_min, self.rhs_max)]
        low = self.initial_min + self.terms * min(products)
        high = self.initial_max + self.terms * max(products)
        prefix_low = min(self.initial_min, low)
        prefix_high = max(self.initial_max, high)
        if prefix_low < -(1 << 31) or prefix_high >= (1 << 31):
            raise ValueError("possible signed-i32 accumulation overflow")
        if self.terms and (min(products) < -(1 << 31) or max(products) >= (1 << 31)):
            raise ValueError("possible signed-i32 product overflow")
        return low, high

    def require_contained(self, low, high):
        actual_low, actual_high = self.interval()
        if actual_low < low or actual_high > high:
            raise ValueError("readout domain does not cover producer interval")
        return actual_low, actual_high


@dataclass(frozen=True)
class ConstantIntegerSumProductsRange:
    """Ordered integer contraction with an immutable constant weight vector.

    Every activation may independently take any integer in ``[lhs_min,lhs_max]``.
    The interval includes source products and every increasing-K prefix. A
    caller still binds the constants, activation type/range, seed, source order,
    and consumer to its actual typed source; this certificate changes no IR.
    """

    weights: tuple[int, ...]
    lhs_min: int
    lhs_max: int
    initial_min: int = 0
    initial_max: int = 0

    def _intervals(self):
        if type(self.weights) is not tuple:
            raise ValueError("immutable integer weight tuple required")
        bounds = (self.lhs_min, self.lhs_max, self.initial_min, self.initial_max)
        if any(type(value) is not int for value in (*bounds, *self.weights)):
            raise ValueError("literal integer weights and bounds required")
        for low, high in ((self.lhs_min, self.lhs_max), (self.initial_min, self.initial_max)):
            if not -(1 << 31) <= low <= high < (1 << 31):
                raise ValueError("ordered signed-i32 source bounds required")
        if any(not -(1 << 31) <= weight < (1 << 31) for weight in self.weights):
            raise ValueError("signed-i32 constant weights required")
        low, high = self.initial_min, self.initial_max
        prefix_low, prefix_high = low, high
        for weight in self.weights:
            products = (self.lhs_min * weight, self.lhs_max * weight)
            product_low, product_high = min(products), max(products)
            if product_low < -(1 << 31) or product_high >= (1 << 31):
                raise ValueError("possible signed-i32 product overflow")
            low += product_low
            high += product_high
            prefix_low, prefix_high = min(prefix_low, low), max(prefix_high, high)
            if prefix_low < -(1 << 31) or prefix_high >= (1 << 31):
                raise ValueError("possible signed-i32 accumulation prefix overflow")
        return (low, high), (prefix_low, prefix_high)

    def interval(self):
        """Final integer enclosure, after checking all source prefixes."""
        return self._intervals()[0]

    def prefix_interval(self):
        """Integer enclosure of the seed and every source prefix."""
        return self._intervals()[1]

    def require_exact_binary_conversion(self, *, significand_bits: int):
        """Sufficient exact-cast proof for a binary floating significand.

        All integers through +/-2**p are exactly representable with p bits.
        This check is conservative above that range. It grants no permission
        to move or reassociate any subsequent rounded floating arithmetic.
        """
        if type(significand_bits) is not int or not 2 <= significand_bits <= 64:
            raise ValueError("binary significand width in [2,64] required")
        low, high = self.interval()
        limit = 1 << significand_bits
        if low < -limit or high > limit:
            raise ValueError("integer domain may require an inexact floating conversion")
        return low, high
