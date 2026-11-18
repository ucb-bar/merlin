"""Exact grouping of equal-weight integer radix product planes.

The caller guarantees signed digit magnitudes no greater than base-1, K terms,
integer accumulation without overflow and exact power-of-two scaling afterward.
No operand values, model identities or measured outputs select this plan.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RadixProductGroup:
    exponent: int
    pairs: tuple[tuple[int, int], ...]
    accumulator_bound: int


@dataclass(frozen=True)
class RadixProductPlan:
    radix_bits: int
    digits: int
    reduction_length: int
    weighted_absolute_bound: int
    groups: tuple[RadixProductGroup, ...]


def plan_radix_product_groups(
    *,
    radix_bits: int,
    digits: int,
    reduction_length: int,
    accumulator_bits: int = 32,
    reconstruction_precision_bits: int = 53,
) -> RadixProductPlan:
    """Refuse unless every original and grouped intermediate is exact.

    Sum of absolute weighted terms is K*(base**digits-1)**2. Bounding this by
    2**p guarantees exact integer products, weighted terms and every partial
    addition in a binary format with p significand bits, in either order.
    Group accumulation must separately fit the signed integer accumulator.
    Scaling of the reconstructed integer by external factors is outside scope.
    """
    values = (radix_bits, digits, reduction_length, accumulator_bits, reconstruction_precision_bits)
    if any(type(x) is not int or x <= 0 for x in values):
        raise ValueError("positive integer domains required")
    if radix_bits * digits > reconstruction_precision_bits:
        raise ValueError("digit width exceeds reconstruction precision")
    base = 1 << radix_bits
    absolute = reduction_length * (base**digits - 1) ** 2
    if absolute > 1 << reconstruction_precision_bits:
        raise ValueError("weighted partial sums may round")
    groups = []
    for total in range(2 * digits - 1):
        pairs = tuple((i, total - i) for i in range(digits) if 0 <= total - i < digits)
        bound = len(pairs) * reduction_length * (base - 1) ** 2
        if bound > (1 << (accumulator_bits - 1)) - 1:
            raise ValueError("group may overflow signed accumulator")
        groups.append(RadixProductGroup(total * radix_bits, pairs, bound))
    return RadixProductPlan(radix_bits, digits, reduction_length, absolute, tuple(groups))
