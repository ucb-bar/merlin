"""Which PROGRAM a cycle count is about, resolved from stated facts rather than assumed.

WHY THIS EXISTS. :mod:`merlin.perf.target_reference` already refuses to score a candidate against a
reference measured on another DESIGN -- "cycles are not comparable across designs" -- and that
refusal is right. It has an exact twin that the ledger did not make: cycles are not comparable
across PROGRAMS either, and three different programs in this project carry whole-model numbers for
the same network on the same device.

Measured, and this is the whole reason the module exists:

* one entry point takes the network's parameters as pointer arguments, its weights arrive already
  quantized, and each contraction's activation scale arrives as a calibrated constant argument;
* the same network's own capsule interface is a FLOAT program that a compiler must quantize
  itself -- not the first one repackaged, a different arithmetic with a different op census;
* a third program takes NO arguments at all -- parameters are linked in as constant blobs, the
  entry is a bare ``main``, its input quantizer runs at BUILD time and is in no measured number,
  and the figure it prints is the SUM OF PER-CALL CYCLE BRACKETS rather than one window.

A ratio between the first and the third was quoted widely as a compiler result. It is not one: the
numerator and the denominator are different programs, on different devices, measured over
different windows. And no compiler edit can move a candidate from one identity to another, because
the identity is a property of the lowering and of the CAPTURE it came from, not of the compiler's
quality. A gap that cannot be closed by the loop being scored is not a gap, it is a category error
with a number attached.

THREE STATES, NEVER TWO -- the rule :mod:`merlin.perf.design_identity` established for devices, for
the same reason. A program is NAMED, or it is ``UNKNOWN(reason)``. "We could not tell which program
this is" is not a softer "the programs differ": a caller that collapsed the two would discard
comparisons that are fine, and a caller that collapsed UNKNOWN into "same" would publish the ratio
this module exists to stop.

TWO AGREEING FACTS, NEVER ONE. An argument count alone does not name a program -- two lowerings can
land on the same arity by coincidence, and a reader who keys on it has keyed on a number, not on a
program. So an identity is resolved only when at least :data:`MIN_AGREEING_FACTS` of its declared
facts are STATED by the candidate and every stated fact agrees. Facts the candidate leaves out are
not held against it; they are recorded in ``confirmed_by`` so a reader can see how much evidence the
resolution actually had.

THE ROSTER IS DATA. The identities themselves are declared in the reference ledger
(``program_identities:``), not in this file: adding a fourth program identity must be a reviewed
edit to a contract file, and nothing here may know what any particular program is called. This
module knows only how to resolve facts against a declared roster and how to refuse.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "MIN_AGREEING_FACTS",
    "REFERENCE_KEY",
    "ROSTER_KEY",
    "UNKNOWN_IDENTITY",
    "ProgramIdentity",
    "ProgramIdentityError",
    "Comparability",
    "comparability",
    "declared_identities",
    "identity_of_reference",
    "require_comparable",
    "resolve",
]

#: How many of an identity's declared facts a candidate must STATE, and agree on, before the
#: identity is named. One fact is a coincidence waiting to happen: an argument count is a number two
#: unrelated lowerings can share, and the mistake this module exists to prevent was made by a reader
#: who had exactly one fact (a count) and treated it as a program.
MIN_AGREEING_FACTS = 2

#: The key a reference entry declares its program identity under.
REFERENCE_KEY = "program_identity"

#: The key the ledger declares the roster under.
ROSTER_KEY = "program_identities"

#: The name an entry states when the program behind its number is UNRECORDED. It is a DECLARATION,
#: not a resolution: a roster entry under this name carries no ``facts``, so :func:`resolve` can
#: never land on it, and an entry naming it is readable and unscoreable -- which is the third state.
UNKNOWN_IDENTITY = "UNKNOWN"


class ProgramIdentityError(ValueError):
    """A comparison was attempted between programs that are not the same program.

    Derives from ``ValueError`` and NOT from ``ReferenceError``-by-inheritance-only: callers that
    already catch ``ValueError`` around reference resolution keep working, and callers that want to
    distinguish this refusal from a missing reference can.
    """


class ProgramIdentity(dict):
    """The resolution, as data: ``name``, ``facts``, ``confirmed_by``, and ``reason`` when unresolved."""

    @property
    def resolved(self) -> bool:
        return bool(self.get("name"))


class Comparability(dict):
    """Whether two programs may be scored against each other: ``comparable`` plus a ``reason``."""

    @property
    def ok(self) -> bool:
        return bool(self.get("comparable"))


def _unknown(reason: str, **extra: Any) -> ProgramIdentity:
    return ProgramIdentity(name=None, facts={}, confirmed_by=(), reason=reason, **extra)


def declared_identities(ledger: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The roster the ledger declares, as ``name -> entry``.

    An empty roster is returned as empty rather than raised on: a ledger that declares no identities
    is a ledger whose entries cannot be checked, and saying so is the caller's job (the shipped
    ledger is gated by a test), not this function's.
    """
    roster = ledger.get(ROSTER_KEY) if isinstance(ledger, Mapping) else None
    if not isinstance(roster, Mapping):
        return {}
    return {str(name): dict(entry) for name, entry in roster.items() if isinstance(entry, Mapping)}


def _declared_facts(entry: Mapping[str, Any]) -> dict[str, Any]:
    facts = entry.get("facts")
    return dict(facts) if isinstance(facts, Mapping) else {}


def _stated(facts: Mapping[str, Any] | None) -> dict[str, Any]:
    """The facts a candidate actually states -- ``None`` values are NOT statements.

    A field present and null is the honest way to say "I do not know this about myself", and folding
    it into a statement would let a program resolve on evidence it never gave.
    """
    if not isinstance(facts, Mapping):
        return {}
    return {str(key): value for key, value in facts.items() if value is not None}


def resolve(facts: Mapping[str, Any] | None, roster: Mapping[str, Any]) -> ProgramIdentity:
    """Name the program whose declared facts these are, or ``UNKNOWN(reason)``.

    Resolution, and every step can refuse:

    1. the candidate states at least :data:`MIN_AGREEING_FACTS` facts;
    2. the roster declares at least one identity;
    3. for exactly one identity, every fact the candidate states that the identity also declares
       AGREES, and at least :data:`MIN_AGREEING_FACTS` of them do so;
    4. no second identity clears step 3 -- two matches is a ROSTER DEFECT and is reported rather
       than resolved by picking one, because the same facts would then name two programs.

    A fact the candidate states that NO identity declares is not an error: the roster describes what
    distinguishes the identities, not everything a program can say about itself.
    """
    stated = _stated(facts)
    if len(stated) < MIN_AGREEING_FACTS:
        return _unknown(
            f"the program states {len(stated)} fact(s) ({', '.join(sorted(stated)) or 'none'}) and "
            f"{MIN_AGREEING_FACTS} agreeing facts are required to name a program; one fact is a "
            "number two unrelated lowerings can share, which is how a program gets misidentified"
        )

    declared = {str(name): dict(entry) for name, entry in (roster or {}).items() if isinstance(entry, Mapping)}
    if not declared:
        return _unknown("the roster declares no program identities, so there is nothing to resolve against")

    matches: dict[str, tuple[str, ...]] = {}
    disagreements: dict[str, list[str]] = {}
    for name, entry in declared.items():
        expected = _declared_facts(entry)
        if not expected:
            continue
        agreeing: list[str] = []
        conflicts: list[str] = []
        for key, value in expected.items():
            if key not in stated:
                continue
            if stated[key] == value:
                agreeing.append(key)
            else:
                conflicts.append(f"{key}={stated[key]!r} (declared {value!r})")
        if conflicts:
            disagreements[name] = conflicts
        elif len(agreeing) >= MIN_AGREEING_FACTS:
            matches[name] = tuple(sorted(agreeing))

    if not matches:
        detail = "; ".join(f"{name}: {', '.join(rows)}" for name, rows in sorted(disagreements.items()))
        return _unknown(
            "no declared program identity matches these facts"
            + (f" -- {detail}" if detail else f"; the roster declares {sorted(declared)}"),
            stated_facts=dict(stated),
        )
    if len(matches) > 1:
        return _unknown(
            f"these facts match more than one declared program identity ({', '.join(sorted(matches))}); "
            "one set of facts cannot name two programs -- the roster does not distinguish them",
            stated_facts=dict(stated),
        )

    name, agreed = next(iter(matches.items()))
    return ProgramIdentity(
        name=name,
        facts=dict(stated),
        confirmed_by=agreed,
        unconfirmed=tuple(sorted(set(_declared_facts(declared[name])) - set(agreed))),
    )


def identity_of_reference(reference: Mapping[str, Any]) -> ProgramIdentity:
    """The identity a reference entry DECLARES, or ``UNKNOWN(reason)`` when it declares none.

    A reference states its identity by name -- it was measured, so the program that produced it is
    known and does not have to be inferred from a fact bundle. An entry that names none is unknown,
    never assumed to share the identity of whatever it is about to be compared with.
    """
    name = str((reference or {}).get(REFERENCE_KEY) or "").strip()
    if not name:
        return _unknown(
            "the reference states no `program_identity`, so which program produced its cycle count "
            "is unrecorded; a number whose program is unknown cannot be scored against, because the "
            "candidate may not be able to produce that program at all"
        )
    if name == UNKNOWN_IDENTITY:
        # NOT a name that can match another one. Two entries both declaring UNKNOWN are not two
        # entries of the same program -- they are two entries nobody identified, and treating the
        # label as an identity would make the honest declaration the most permissive thing in the
        # file. This is the difference between three states and two wearing three labels.
        return _unknown(
            "the reference declares its program identity as UNKNOWN: the program behind this number "
            "is unrecorded, which is a statement that it cannot be scored against, not a program"
        )
    return ProgramIdentity(name=name, facts={}, confirmed_by=("declared_by_the_reference",))


def comparability(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Comparability:
    """Whether a candidate program and a reference program may be scored against each other.

    Takes two :class:`ProgramIdentity` results (or anything with ``name``/``reason``). Comparable
    only when BOTH are named and the names are equal; unknown on either side is refused with the
    reason that side gave, so the refusal says what is missing rather than that something is.
    """
    candidate_name = str((candidate or {}).get("name") or "")
    reference_name = str((reference or {}).get("name") or "")
    if not reference_name:
        return Comparability(
            comparable=False,
            candidate=candidate_name or None,
            reference=None,
            reason="the reference's program identity is unknown: "
            + str((reference or {}).get("reason") or "no reason recorded"),
        )
    if not candidate_name:
        return Comparability(
            comparable=False,
            candidate=None,
            reference=reference_name,
            reason=f"the reference was measured on the {reference_name!r} program and the candidate's "
            "program identity is unknown: " + str((candidate or {}).get("reason") or "no reason recorded"),
        )
    if candidate_name != reference_name:
        return Comparability(
            comparable=False,
            candidate=candidate_name,
            reference=reference_name,
            reason=f"the candidate is a {candidate_name!r} program and the reference was measured on a "
            f"{reference_name!r} program; cycles are not comparable across program identities any more "
            "than across designs, and no compiler edit moves a candidate between them -- the identity "
            "is a property of the lowering, not of its quality",
        )
    return Comparability(comparable=True, candidate=candidate_name, reference=reference_name, reason="")


def require_comparable(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Comparability:
    """:func:`comparability`, raising :class:`ProgramIdentityError` instead of returning a refusal.

    The raising form is for call sites that RETURN A SCORE -- a per-field gap, a ratio -- where a
    refusal record could be read past. Call sites that return a record the loop reads should use
    :func:`comparability` and embed the refusal, because a raise into a path wrapped in
    ``try/except`` is indistinguishable from nothing happening.
    """
    verdict = comparability(candidate, reference)
    if not verdict.ok:
        raise ProgramIdentityError(verdict["reason"])
    return verdict
