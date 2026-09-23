"""Hold a target's DECLARED on-chip geometry to the machine DERIVED from its RTL facts.

Every target that schedules against on-chip storage learns its geometry twice. Once from something it
DECLARES -- a C header its programs compile against, a params file, a contract field -- and once from
what an extractor DERIVES from the RTL. The two describe the same silicon by different routes, and
nothing compares them, so a declaration regenerated against a different elaboration schedules against a
machine that is not the one being measured. Every cycle count still looks plausible, because the program
still runs.

WHY THIS LIVES IN THE CORE AND NOT IN A TARGET'S BACKEND. The comparison names no target and needs
none: it takes a mapping of declared numbers, derives the machine for whichever target it was given, and
reports. Written inside one target's binding it would be correct and then copied, and the second copy is
where the two drift -- which is the shape of overfit the repo's cardinal rule exists to prevent. Here it
is covered by ``check_no_target_name.py`` for free, so this file can never grow a target literal without
a gate saying so.

A DISAGREEMENT RAISES, and names both numbers rather than preferring one. Which of the two is stale is
not a question this function can answer, and picking silently is how the wrong number becomes the agreed
number. An UNDERIVABLE machine does NOT raise: inside a sandbox the RTL facts are not mounted, and
refusing there would make "we could not look" indistinguishable from "the two disagree". It is recorded
instead, so a reader can tell agreement from silence.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["GeometryDisagreement", "crosscheck_declared_geometry", "AXES"]

#: Declared-geometry axis -> the attribute of the projected geometry it must equal. The declaring side
#: supplies its OWN spelling for each; this is the shared vocabulary the comparison is made in.
AXES: tuple[str, ...] = ("block", "operand_rows", "accumulator_rows")


class GeometryDisagreement(ValueError):
    """A target's declared geometry and its derived machine describe different silicon."""


def crosscheck_declared_geometry(declared: Mapping[str, int], target: str | None) -> str:
    """Compare ``declared`` against the machine derived for ``target``; return what was established.

    ``declared`` maps :data:`AXES` names to the numbers the target's own declaration states. A caller
    whose declaration spells them differently translates at the call site -- the translation is the one
    genuinely target-specific part, and it stays with the target.

    Returns a sentence a receipt can carry: which machine agreed, or why none could be consulted.
    Raises :class:`GeometryDisagreement` when both spoke and they differ.
    """
    if not target:
        return "not checked: no target named"
    unknown = sorted(set(declared) - set(AXES))
    if unknown:
        raise GeometryDisagreement(
            f"{target}: declared geometry names {unknown}, which is not one of {list(AXES)}. The "
            "comparison is made in a shared vocabulary; a caller translates its own spelling at the "
            "call site rather than widening this one."
        )
    try:
        from merlin.compile.scheduling.derive import geometry_from_machine

        from .derive import derive

        geometry = geometry_from_machine(derive(target))
    except Exception as exc:  # noqa: BLE001 - an unavailable derivation is recorded, never fatal
        return f"unavailable: {type(exc).__name__}: {exc}"

    disagree = {
        axis: (value, getattr(geometry, axis)) for axis, value in declared.items() if value != getattr(geometry, axis)
    }
    if disagree:
        raise GeometryDisagreement(
            f"{target}: what this target DECLARES and the machine derived from its RTL facts describe "
            "different silicon: "
            + "; ".join(f"{k} declared={d} derived={v}" for k, (d, v) in sorted(disagree.items()))
            + ". One of the two is stale. This refuses rather than picking, because a schedule built on "
            "the wrong one still runs and still reports a number."
        )
    return f"agrees with merlin.sched.mach ({geometry.sources.get('machine', 'no digest')})"
