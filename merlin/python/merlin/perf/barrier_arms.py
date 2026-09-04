"""Two arms of one kernel that differ ONLY in how many completion barriers the compiler inserted.

The sibling scheduling family reorders a command stream (:mod:`merlin.perf.command_stream_gen`) on the
theory that issue order is the lever a hardware-interlocked, command-driven accelerator gives a
compiler. MEASURED on this repo's command-driven target, it is not: a legal permutation of the same
multiset cost **exactly zero** cycles at two depths, with the reorder verified to reach the ELF and the
counter verified to move with work. Both results have the same cause -- the dispatch queue tracks the
dependence itself -- and read together they say where the lever actually is: not in what order the
commands issue, but in **how many barriers the compiler inserted that the hardware did not need**.

This module builds that A/B. It asks the TARGET'S OWN emitter for the same kernel twice, at two
settings of the emitter's retire knob, and then PROVES the two sources are the same program plus
barriers rather than asserting it:

* the low-barrier arm's line sequence must be a SUBSEQUENCE of the high-barrier arm's -- so every
  instruction of one appears, in order, in the other, and nothing was added, dropped or reworded;
* every line the high-barrier arm adds must be the SAME line -- one repeated statement, which is what
  makes "the lever is barrier count" true by construction rather than by reading.

⚠️ NOTHING HERE KNOWS HOW A BARRIER IS SPELLED. The barrier statement is *discovered* as the line the
two arms differ by, so a target whose emitter spells its retire differently is served without an edit,
and a target whose two settings differ by something OTHER than a repeated inserted line is REFUSED with
its reason rather than measured as if the difference were a barrier. Refusing is the point: a pair that
differs by more than the lever prices the difference and calls it the lever, which is how a
neighbouring family came to compare an ~82-cycle lever against an ~280-cycle uncancelled term.

⚠️ A ZERO-BARRIER PAIR IS A RESULT, NOT A FAILURE. A kernel with one job has no redundant barrier to
remove, so both settings emit the identical program and ``removed`` is 0. That member is this family's
NEGATIVE CONTROL: its measured differential must be exactly zero, and it is the only member for which
that is true by construction.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = [
    "BarrierPair", "REFUSED_HETEROGENEOUS_INSERTION", "REFUSED_KNOB_UNSUPPORTED",
    "REFUSED_NOT_A_PURE_INSERTION", "RetireArmsError", "pair_from_emitter",
]

#: The emitter did not accept the retire knob at all, so this target cannot express the lever.
REFUSED_KNOB_UNSUPPORTED = "knob_unsupported"
#: The high-barrier arm is not the low-barrier arm plus inserted lines: something was reworded.
REFUSED_NOT_A_PURE_INSERTION = "not_a_pure_insertion"
#: The inserted lines are not all the same statement, so "the difference" is not one repeated barrier.
REFUSED_HETEROGENEOUS_INSERTION = "heterogeneous_insertion"


class RetireArmsError(RuntimeError):
    """A pair that would not be the same work with a different barrier count. Carries its ``reason``."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class BarrierPair:
    """One kernel emitted at two retire settings, with the difference established rather than assumed."""

    #: Source at the setting that emits the fewest barriers.
    minimal: str
    #: Source at the setting that emits one barrier per unit of work.
    maximal: str
    #: The two settings, in the order (minimal, maximal) they were requested.
    settings: tuple[str, str]
    #: The line the two arms differ by, taken from the diff rather than from any target vocabulary.
    #: ``None`` when the two arms are the identical program (the negative-control member).
    barrier_statement: str | None
    #: How many barriers the lever removes. Zero is a legitimate result, not an error.
    removed: int

    def to_dict(self) -> dict:
        return {"settings": list(self.settings), "barrier_statement": self.barrier_statement,
                "removed": self.removed,
                "identical_programs": self.minimal == self.maximal}


def _inserted_lines(low: Sequence[str], high: Sequence[str]) -> list[str]:
    """The lines ``high`` has and ``low`` does not, or raise when ``low`` is not a subsequence.

    A two-pointer walk, not a diff library and not a pattern: the question is exactly "does every line
    of the short source appear, in order, in the long one", and answering it any other way would admit
    a pair whose shared lines had been reordered.
    """
    extra: list[str] = []
    i = 0
    for line in high:
        if i < len(low) and low[i] == line:
            i += 1
        else:
            extra.append(line)
    if i != len(low):
        raise RetireArmsError(
            REFUSED_NOT_A_PURE_INSERTION,
            f"the low-barrier arm's line {i + 1} ({low[i]!r}) has no match in emission order in the "
            "high-barrier arm; the two settings changed the kernel, not only its barrier count")
    return extra


def pair_from_emitter(emit: Callable[..., str], command_buffer: dict, *,
                      settings: tuple[str, str], knob: str = "retire") -> BarrierPair:
    """Emit ``command_buffer`` twice through ``emit`` and return the established barrier pair.

    ``emit`` is the target's OWN driver emitter, called as ``emit(command_buffer, **{knob: setting})``.
    The knob name is a parameter because it belongs to the emitter's signature, not to this module; a
    target whose emitter does not accept it is refused with :data:`REFUSED_KNOB_UNSUPPORTED` rather than
    silently measured at one setting twice.
    """
    if len(settings) != 2 or settings[0] == settings[1]:
        raise ValueError(f"settings must be two DISTINCT emitter settings, got {settings!r}")
    sources = []
    for setting in settings:
        try:
            sources.append(emit(command_buffer, **{knob: setting}))
        except TypeError as exc:
            raise RetireArmsError(
                REFUSED_KNOB_UNSUPPORTED,
                f"this target's emitter does not accept {knob}={setting!r} ({exc}); it cannot express "
                "the barrier lever, so no pair exists to measure") from exc
    low_src, high_src = sources
    extra = _inserted_lines(low_src.split("\n"), high_src.split("\n"))
    distinct = set(extra)
    if len(distinct) > 1:
        raise RetireArmsError(
            REFUSED_HETEROGENEOUS_INSERTION,
            f"the high-barrier arm adds {len(distinct)} distinct statements {sorted(distinct)!r}; the "
            "difference between the arms is then not one repeated barrier and pricing it as one would "
            "attribute the rest of the difference to the lever")
    return BarrierPair(minimal=low_src, maximal=high_src, settings=(settings[0], settings[1]),
                       barrier_statement=(extra[0] if extra else None), removed=len(extra))
