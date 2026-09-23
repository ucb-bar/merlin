"""What a capsule may put in its one slot for a cycle ceiling, and what that slot means when it holds
a word instead of a number.

``performance.cost.projected_cycles`` is the field a cost plane reads to find the ceiling a capsule
declares, and such a reader needs an integer: a word resolves to no ceiling, so the assessment comes
back incomplete and the plane can reject a physically impossible count but never a merely slow one.

Measured across the corpus before this module existed: 113 of 595 capsules carried the field, all 113
wrote ``derived_at_preflight``, none wrote an integer, and no code anywhere read that string -- no
preflight pass resolved it and no branch matched on it. It was a promissory note nothing honoured.
The field also had no schema type at all (the ``performance`` block reached disk only through
``additionalProperties: true``), so a typo in it was indistinguishable from a declaration.

This module does not give any capsule a bound. It makes the ABSENCE of one a declaration rather than
an accident, which is the precondition for a capsule ever owing a number -- and it names, in
:data:`DERIVED_AT_ASSESSMENT`, the one word that means "a number is owed and it is derived, not
typed", so the resolver in :mod:`merlin.perf.cost_plane` can honour it without spelling it a second
time.

Target-agnostic by construction: a cycle count is a fact about one target and none is named here. The
vocabulary below is about what KIND of thing the field holds, never about how many cycles anything
takes.
"""

from __future__ import annotations

from typing import Any

#: The words the field may carry INSTEAD of a number, each saying why the capsule states no literal
#: count and what -- if anything -- stands in its place.
#:
#: Closed on purpose. Left open, an unrecognised word resolves to "no ceiling" downstream exactly as a
#: recognised one does, so a typo produces a capsule that quietly demands nothing -- which is the
#: failure this whole module is about, reintroduced one spelling at a time.
NO_CYCLE_BOUND: dict[str, str] = {
    "derived_at_preflight": (
        "the member states no literal count: its ceiling is DERIVED at assessment time from the "
        "array the target's own facts declare and the work its own command buffer counted, times "
        "the slack the gate declaration states -- see merlin.perf.cost_plane.resolve_ceiling"
    ),
    "unbounded": (
        "this member exists to be measured, not to be held to a ceiling -- a law-fitting point has no "
        "budget it could exceed"
    ),
}

#: The subset of :data:`NO_CYCLE_BOUND` whose members are a PROMISE OF A DERIVATION rather than a
#: refusal of one. A capsule spelling one of these owes a cycle count; the number simply does not live
#: in the capsule, because it is a fact about the machine the capsule is being run on and a capsule is
#: run on more than one.
#:
#: Named here rather than matched on in the reader, so the word exists in exactly one place. It was a
#: promissory note nothing honoured for 113 capsules; a second spelling of it in the resolver is how
#: it would become one again.
DERIVED_AT_ASSESSMENT: frozenset[str] = frozenset({"derived_at_preflight"})

#: The word for a member that owes NO count at all. The complement of the set above, computed rather
#: than typed so the two cannot disagree about a word added to the vocabulary.
NO_BOUND_AT_ALL: frozenset[str] = frozenset(NO_CYCLE_BOUND) - DERIVED_AT_ASSESSMENT


class CycleBoundError(ValueError):
    """The declared cycle ceiling is neither a count nor a declared reason for having none."""


def validate(value: Any, *, owner: str) -> None:
    """Raise unless ``value`` is a positive int or a declared no-bound word.

    Fail closed in both directions.

    * A NON-POSITIVE int is refused. Zero is not a ceiling any correct run could meet, and it would
      have passed the generic non-emptiness check the performance block already applies -- ``0`` is
      not ``None``, not an empty string and not an empty container -- so it is the one malformed
      value that reaches a reader looking exactly like a bound.
    * A count written as a STRING (``"4096"``) is refused for the opposite reason: it reads as a
      bound to a human reviewing the profile and resolves to no ceiling in the reader, so the capsule
      looks stricter than it is.
    * An UNRECOGNISED word is refused rather than treated as "no bound", because silently meaning the
      permissive thing is how the field got into this state.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise CycleBoundError(
            f"{owner}: performance.cost.projected_cycles must be a positive integer (a cycle "
            f"ceiling) or one of {sorted(NO_CYCLE_BOUND)}; got {value!r}"
        )
    if isinstance(value, int):
        if value <= 0:
            raise CycleBoundError(
                f"{owner}: performance.cost.projected_cycles is {value!r}. A ceiling of {value} is "
                f"not a bound any run can meet -- state a positive count, or one of "
                f"{sorted(NO_CYCLE_BOUND)} to declare that this member owes none"
            )
        return
    if value not in NO_CYCLE_BOUND:
        raise CycleBoundError(
            f"{owner}: performance.cost.projected_cycles is {value!r}, which is neither a cycle "
            f"count nor a declared reason for having none. Use a positive integer, or one of "
            f"{sorted(NO_CYCLE_BOUND)} -- a word this vocabulary does not know resolves to no "
            f"ceiling downstream, so the member would be graded as though it had claimed nothing"
        )


def declared_cycles(capsule: Any) -> int | None:
    """The integer ceiling ``capsule`` declares, or ``None`` when it declares none.

    ``None`` is returned for a declared no-bound word AND for a capsule carrying no performance block
    at all: both are "this capsule owes no cycle count", and a caller that needs to tell them apart
    should read the field itself. A malformed value RAISES rather than reading as ``None`` -- a
    capsule whose declaration nobody could parse has not declared the absence of a bound, and
    treating it as though it had is the permissive default this module exists to remove.
    """
    node: Any = capsule
    for key in ("performance", "cost", "projected_cycles"):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if node is None:
        return None
    validate(node, owner=str((capsule or {}).get("name", "<unnamed capsule>")))
    return int(node) if isinstance(node, int) else None
