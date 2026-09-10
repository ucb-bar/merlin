"""Structural integrity of a device readback, independent of any numeric tolerance.

A transport that reconstructs a buffer out of device state can lose a whole
residue class of words -- every second 4-byte word, say -- while leaving every
other word bit-exact.  Compared value-by-value against a tolerance scaled to the
golden's magnitude, such a buffer can still PASS: measured on one 7232-element
capsule, 5172 zeroed elements produced only 988 reported mismatches, because the
remaining 4184 goldens were smaller than the absolute tolerance.  A mandatory
cert tier can therefore be granted to a buffer half of which was never read
back at all.

This module refuses that shape STRUCTURALLY.  It needs no golden, no tolerance
and no knowledge of the device: it only observes that a periodic subset of the
word indices is exactly zero while the rest is not.  No arithmetic kernel
produces that; a half-width transport produces exactly that.

The rule is deliberately fail-closed.  A buffer whose every word is zero does
NOT trip it (the complement carries no evidence, and a wholly zero buffer is
already visible to any value comparison); a buffer with one periodic hole and
live data elsewhere does.  A legitimately zero column that happens to align
with a stride is refused rather than certified -- a false refusal is a question
to answer, a false certification is a result nobody can retract.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Word strides worth testing.  These are the transport granularities a
#: reconstruction can halve or quarter (an 8-byte beat, a 16-byte quarter-line,
#: a 32-byte line, a 64-byte line over 4-byte words), not a property of any
#: particular device.
STRIDES: tuple[int, ...] = (2, 4, 8, 16)

#: Below these sizes the pattern is not evidence: a short buffer can hold an
#: all-zero residue class by arithmetic accident.
MIN_WORDS = 16
MIN_CLASS_WORDS = 8


class ReadbackIntegrityError(Exception):
    """A readback whose STRUCTURE disqualifies it, whatever its values say."""


def residue_class_defect(
    words: Sequence[int],
    *,
    strides: Sequence[int] = STRIDES,
    min_words: int = MIN_WORDS,
    min_class_words: int = MIN_CLASS_WORDS,
) -> str | None:
    """Return a named diagnostic if a whole residue class of words is exactly zero.

    ``words`` are the raw readback words as unsigned integers (the bit patterns,
    never floats -- ``-0.0`` and ``0.0`` are different bit patterns and only an
    exactly zero pattern counts).  Returns ``None`` when nothing is wrong.
    """
    total = len(words)
    if total < min_words:
        return None
    nonzero_total = sum(1 for w in words if w)
    if nonzero_total == 0:
        # Nothing survives anywhere: there is no complement to contrast with,
        # and a wholly empty buffer is not this defect.
        return None
    for stride in strides:
        if stride < 2 or total // stride < min_class_words:
            continue
        for residue in range(stride):
            klass = words[residue::stride]
            if len(klass) < min_class_words:
                continue
            if any(klass):
                continue
            complement_nonzero = nonzero_total  # klass is all zero, so all of it is outside
            if complement_nonzero == 0:
                continue
            return (
                "readback_residue_class_zeroed: every 4-byte word at index % "
                f"{stride} == {residue} is exactly zero ({len(klass)} of {len(klass)}), "
                f"while {complement_nonzero} of the remaining {total - len(klass)} words "
                "are non-zero. A periodic hole of this shape is a transport defect, "
                "not an arithmetic result; the readback is refused before any value "
                "is compared."
            )
    return None


def words_from_bytes(raw: bytes) -> list[int]:
    """Split a readback into little-endian 4-byte words, without interpreting them."""
    if len(raw) % 4:
        raise ReadbackIntegrityError(
            f"readback is {len(raw)} bytes, not a whole number of 4-byte words")
    return [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]


def require_intact(raw: bytes, *, transport: str) -> None:
    """Raise :class:`ReadbackIntegrityError` if ``raw`` carries a structural defect."""
    defect = residue_class_defect(words_from_bytes(raw))
    if defect is not None:
        raise ReadbackIntegrityError(f"{transport}: {defect}")
