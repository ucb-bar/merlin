"""Conservative known-bit, pointer and bounded affine value domains for RV64."""

from __future__ import annotations

from dataclasses import dataclass

from .riscv_decode import MASK as MASK
from .riscv_decode import signed


@dataclass(frozen=True)
class Bits:
    mask: int = 0
    value: int = 0
    base: str | None = None
    offset: int = 0
    # Post-fixpoint loop annotation (never part of the fixed point itself):
    # (constant, pointer_base, ((header_pc, step, count), ...)) meaning
    # value = [pointer_base +] constant + sum(step * k_header), 0 <= k < count.
    affine: tuple | None = None
    # Symbolic bit window ``(symbol, offset, at, width)``: bits ``[at, at + width)`` equal
    # ``(symbol + offset) mod 2**width`` for one unknown value ``symbol``; ``mask``/``value``
    # describe only bits outside the window. Two windows of one symbol are comparable.
    window: tuple | None = None

    @classmethod
    def const(cls, value):
        return cls(MASK, value & MASK)

    @property
    def exact(self):
        return self.value if self.mask == MASK else None

    def join(self, other):
        mask = self.mask & other.mask & ~(self.value ^ other.value) & MASK
        pointer = self.base is not None and self.base == other.base and self.offset == other.offset
        window = self.window if self.window == other.window else None
        return Bits(
            mask, self.value & mask, self.base if pointer else None, self.offset if pointer else 0, window=window
        )

    def record(self):
        return dict(
            kind="const" if self.exact is not None else "unknown",
            raw=self.exact,
            **(
                dict(known_bits=dict(mask=self.mask, value=self.value, width=64))
                if self.exact is None and self.mask
                else {}
            ),
            **(dict(pointer_base=self.base, byte_offset=self.offset) if self.base else {}),
            **(_affine_record(self.affine) if self.affine else {}),
            **(_window_record(self.window) if self.window else {}),
        )


_VALUE_SET_LIMIT = 4096


def _affine_record(affine):
    if isinstance(affine, _Table):
        return dict(
            affine=dict(
                terms=[dict(header_pc=h, count=n) for h, n in affine.dims],
                basis="proved_trip_counts_pointwise_cpu_semantics",
            ),
            value_set=sorted(set(affine.values)),
        )
    constant, pointer, terms = affine
    out = dict(
        affine=dict(
            base=constant & MASK if pointer is None else None,
            **(dict(pointer_base=pointer, byte_offset=constant) if pointer else {}),
            terms=[dict(header_pc=h, step=step, count=count) for h, step, count in terms],
            basis="proved_induction_and_trip_counts",
        )
    )
    total = 1
    for _, _, count in terms:
        total *= count
    if pointer is None and total <= _VALUE_SET_LIMIT:
        values = {constant}
        for _, step, count in terms:
            values = {v + step * k for v in values for k in range(count)}
        out["value_set"] = sorted(v & MASK for v in values)
    return out


UNKNOWN = Bits()

ZERO = Bits.const(0)


def _window_record(window):
    symbol, offset, at, width = window
    return dict(window=dict(symbol=symbol, offset=offset, at=at, width=width))


def _cut_window(window, at, width):
    """The part of a window that lies in bits ``[at, at + width)`` of its own position, or None."""
    if window is None or width <= 0:
        return None
    return (window[0], window[1] % (1 << width), at, width)


def _sext(value, bits):
    low = (1 << bits) - 1
    mask, result = value.mask & low, value.value & low
    if mask & (1 << (bits - 1)):
        mask |= MASK ^ low
        if result & (1 << (bits - 1)):
            result |= MASK ^ low
    window = value.window
    if window is not None:
        window = _cut_window(window, window[2], min(window[3], bits - window[2]))
    return Bits(mask, result, window=window)


def _window_arithmetic(kind, a, b):
    """Integer operations on a symbolic window and an exact operand.

    Low bits of a sum depend only on low operand bits, so an add keeps a window at bit 0.
    An OR or AND with an exact value keeps the window's bits below the first bit the value
    changes; constant shifts move it. Anything else leaves the window's bits unknown.
    """
    if b.window is not None and a.window is None and kind in ("add", "or", "and"):
        a, b = b, a
    if kind in ("slt", "sltu"):
        return Bits(MASK ^ 1, 0)
    constant = b.exact
    if b.window is not None or constant is None:
        return UNKNOWN
    symbol, offset, at, width = a.window
    span = ((1 << width) - 1) << at
    outside = Bits(a.mask & ~span & MASK, a.value & ~span & MASK)
    if kind in ("add", "sub"):
        if at:
            return UNKNOWN
        delta = constant if kind == "add" else -constant
        return Bits(window=(symbol, (offset + delta) % (1 << width), 0, width))
    if kind in ("or", "and"):
        inside = (constant & span) >> at
        changed = inside if kind == "or" else ~inside & ((1 << width) - 1)
        kept = min(width, (changed & -changed).bit_length() - 1) if changed else width
        rest = arithmetic(kind, outside, b)
        keep = ((1 << kept) - 1) << at
        return Bits(rest.mask & ~keep & MASK, rest.value & ~keep & MASK, window=_cut_window(a.window, at, kept))
    if kind in ("sll", "srl", "sra"):
        amount = constant & 63
        if not amount:
            return a
        rest = arithmetic(kind, outside, b)
        if kind == "sll":
            moved = _cut_window(a.window, at + amount, min(width, 64 - at - amount))
        else:
            moved = _cut_window(a.window, at - amount, width) if at >= amount else None
        if moved is None:
            return rest
        keep = ((1 << moved[3]) - 1) << moved[2]
        return Bits(rest.mask & ~keep & MASK, rest.value & ~keep & MASK, window=moved)
    return UNKNOWN


def arithmetic(kind, a, b):
    if a.window is not None or b.window is not None:
        return _window_arithmetic(kind, a, b)
    av, bv = a.exact, b.exact
    if kind in ("add", "sub"):
        if bv is not None and a.base:
            delta = signed(bv, 64) * (1 if kind == "add" else -1)
            return Bits(base=a.base, offset=a.offset + delta)
        if av is not None and b.base and kind == "add":
            return Bits(base=b.base, offset=b.offset + signed(av, 64))
        if av is not None and bv is not None:
            return Bits.const(av + bv if kind == "add" else av - bv)
        # Standard known-bits addition with an explicit carry (a - b == a + ~b + 1).
        # A bit is known when both operand bits and its incoming carry are known.
        a_zero, a_one = a.mask & ~a.value & MASK, a.mask & a.value
        b_zero, b_one = b.mask & ~b.value & MASK, b.mask & b.value
        carry = 0
        if kind == "sub":
            b_zero, b_one, carry = b_one, b_zero, 1
        largest = ((~a_zero & MASK) + (~b_zero & MASK) + carry) & MASK
        smallest = (a_one + b_one + carry) & MASK
        carry_zero = ~(largest ^ a_zero ^ b_zero) & MASK
        carry_one = (smallest ^ a_one ^ b_one) & MASK
        known = (a_zero | a_one) & (b_zero | b_one) & (carry_zero | carry_one) & MASK
        return Bits(known, smallest & known)
    if kind == "and":
        zero = (a.mask & ~a.value) | (b.mask & ~b.value)
        ones = a.mask & a.value & b.mask & b.value
        return Bits((zero | ones) & MASK, ones)
    if kind == "or":
        ones = (a.mask & a.value) | (b.mask & b.value)
        zero = a.mask & ~a.value & b.mask & ~b.value
        return Bits((zero | ones) & MASK, ones)
    if kind == "xor":
        return Bits(a.mask & b.mask, (a.value ^ b.value) & a.mask & b.mask)
    if kind in ("sll", "srl", "sra") and bv is not None:
        amount = bv & 63
        if not amount:
            return a
        if kind == "sll":
            return Bits(((a.mask << amount) | ((1 << amount) - 1)) & MASK, (a.value << amount) & MASK)
        high = MASK ^ ((1 << (64 - amount)) - 1)
        sign_known = bool(a.mask & (1 << 63))
        fill_mask = high if kind == "srl" or sign_known else 0
        fill_value = high if kind == "sra" and sign_known and a.value >> 63 else 0
        return Bits((a.mask >> amount) | fill_mask, (a.value >> amount) | fill_value)
    if av is not None and bv is not None:
        if kind == "mul":
            return Bits.const(av * bv)
        if kind in ("slt", "sltu"):
            return Bits.const(int((signed(av, 64) < signed(bv, 64)) if kind == "slt" else av < bv))
    if kind in ("slt", "sltu"):
        return Bits(MASK ^ 1, 0)
    return UNKNOWN


@dataclass(frozen=True)
class _Table:
    """A register value at every point of its loops' joint iteration space.

    ``dims`` is ((header_pc, count), ...); ``values`` enumerates the points in
    row-major order (last header fastest), each reduced modulo 2**64.
    """

    dims: tuple
    values: tuple
