"""Byte bounds for a case's native products, derived from the case's declared operands.

A completed result embeds the case's canonical input values (JSON text) and a self-contained ELF embeds
the same stimulus (binary). Fixed bounds refused the largest declared cases (an LM-head weight of 65.5M
elements made a 164 MB result), so their performance was never observed. Each bound here is the fixed
floor plus a proportional allowance for exactly what the case declares; nothing is target-specific.
"""

from __future__ import annotations

#: Proportional margin over the declared payload (layout/indentation and the other embedded copies).
MARGIN = 2


def _bits(dtype) -> int | None:
    """Element width from a dtype spelling such as ``i8``, ``u16``, ``f32``, ``bf16`` (else None)."""
    if not isinstance(dtype, str):
        return None
    digits = "".join(ch for ch in dtype if ch.isdigit())
    prefix = dtype[: len(dtype) - len(digits)] if dtype.endswith(digits) else ""
    if not digits or prefix not in {"i", "u", "si", "ui", "f", "bf", "int", "uint", "float"}:
        return None
    return int(digits)


def _elements(shape) -> int | None:
    if not isinstance(shape, list | tuple) or not all(type(d) is int and d >= 0 for d in shape):
        return None
    count = 1
    for dim in shape:
        count *= dim
    return count


def _operands(capsule) -> list[tuple[int, int, bool]]:
    """``(elements, bits, integer)`` of each declared input whose shape and dtype parse."""
    out = []
    for operand in (capsule or {}).get("inputs") or ():
        if not isinstance(operand, dict):
            continue
        count, bits = _elements(operand.get("shape")), _bits(operand.get("dtype"))
        if count is not None and bits is not None:
            out.append((count, bits, operand["dtype"][0] in "iu"))
    return out


def _json_chars(bits: int, integer: bool) -> int:
    """Widest JSON text of one element plus its separator."""
    if integer:
        return len(str(-(1 << (bits - 1)))) + 2
    return len(repr(-2.2250738585072014e-308)) + 2  # widest float spelling


def result_bytes(capsule, floor: int) -> int:
    """Bound for a completed result that embeds the case's canonical inputs as JSON."""
    payload = sum(count * _json_chars(bits, integer) for count, bits, integer in _operands(capsule))
    return max(int(floor), int(floor) + MARGIN * payload)


def binary_bytes(capsule, floor: int) -> int:
    """Bound for a self-contained executable that embeds the case's inputs in binary."""
    payload = sum(count * ((bits + 7) // 8) for count, bits, _ in _operands(capsule))
    return max(int(floor), int(floor) + MARGIN * payload)
