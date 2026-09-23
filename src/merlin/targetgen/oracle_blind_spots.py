"""What each oracle tier of a target cannot see, as data, and the tier a program therefore needs.

A tier that cannot observe a behaviour certifies every program in that behaviour's class, right or
wrong, and nothing in the result says so. The measured case: one executable passed the functional
tier with no mismatching output and failed on the FPGA with every output wrong, because the
functional model pins an accumulator base the RTL rotates. That was known, and it lived in notes.

This module makes it a declaration the grader can act on. A target's contract lists its blind
spots under ``oracle_blind_spots``; each names the tiers that are blind, and the CLASS of program
it concerns in the vocabulary the corpus already states about itself (``tier_policy.capsule_axes``:
an instruction class, a mode, an epilogue stage, an op). From that:

* :func:`blind_tiers` -- which tiers cannot judge this capsule, and why;
* :func:`required_tier` -- the shallowest declared tier that can; ``None`` when no declared tier can,
  which is a finding about the ladder and never a pass;
* :func:`audit` -- capsules whose deepest reachable tier sits inside a blind spot for their own
  axes. Such a capsule can be screened and can never be certified, whatever its result says.

Nothing here names a target, a tier or an instruction: the entries are the target's data, and a
target with no entries declares no blind spot and changes nothing.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import tier_policy

CONTRACT_KEY = "oracle_blind_spots"
SCHEMA = "oracle_blind_spot_audit_v1"
#: A capsule whose every reachable tier is blind to one of its own axes.
UNCERTIFIABLE = "certifiable_only_inside_blind_spot"
#: No declared tier can see the class at all: the ladder itself is short.
NO_SEEING_TIER = "no_declared_tier_can_see"


class BlindSpotError(ValueError):
    """A registry entry is malformed; says which entry and which field."""


@dataclass(frozen=True)
class BlindSpot:
    """One class of behaviour a set of tiers cannot observe."""

    id: str
    tiers: tuple[str, ...]
    #: ``(axis, value)`` pairs in the capsule-axis vocabulary. A program exercising ANY of them is
    #: in the class: an entry lists the several ways one behaviour is reached.
    axes: tuple[tuple[str, str], ...]
    what: str
    evidence: str
    reproducer: str | None = None
    engine: str | None = None

    def concerns(self, axes: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        held = set(axes)
        return tuple(pair for pair in self.axes if pair in held)


def parse(entries: Any) -> tuple[BlindSpot, ...]:
    """Registry entries as declared in a contract. Fails closed on a malformed one: an entry that
    silently matched nothing would read as "no blind spot", which is the defect this exists for."""
    if entries is None:
        return ()
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise BlindSpotError(f"`{CONTRACT_KEY}` is a list of entries")
    out: list[BlindSpot] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise BlindSpotError(f"entry {index} is not a mapping")
        ident = str(entry.get("id") or "")
        if not ident or ident in seen:
            raise BlindSpotError(f"entry {index} needs a unique `id`")
        seen.add(ident)
        tiers = tuple(str(t) for t in entry.get("tiers") or ())
        axes = tuple(
            (str(pair[0]), str(pair[1]))
            for pair in entry.get("axes") or ()
            if isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2
        )
        if not tiers:
            raise BlindSpotError(f"{ident}: names no blind tier")
        if not axes or len(axes) != len(entry.get("axes") or ()):
            raise BlindSpotError(f"{ident}: `axes` is a non-empty list of [axis, value] pairs")
        for key in ("what", "evidence"):
            if not str(entry.get(key) or "").strip():
                raise BlindSpotError(f"{ident}: an entry without `{key}` is a rumour, not a record")
        out.append(
            BlindSpot(
                id=ident,
                tiers=tiers,
                axes=axes,
                what=str(entry["what"]).strip(),
                evidence=str(entry["evidence"]).strip(),
                reproducer=(str(entry["reproducer"]).strip() if entry.get("reproducer") else None),
                engine=(str(entry["engine"]) if entry.get("engine") else None),
            )
        )
    return tuple(out)


def for_target(target: str, *, contract: Mapping[str, Any] | None = None) -> tuple[BlindSpot, ...]:
    """The blind spots ``target``'s own contract declares."""
    if contract is None:
        return _declared(str(target))
    return parse((contract or {}).get(CONTRACT_KEY))


@functools.lru_cache(maxsize=None)
def _declared(target: str) -> tuple[BlindSpot, ...]:
    # Asked once per capsule tier by the grader; a contract does not change under a running grade.
    from .target_registry import load_contract

    return parse((load_contract(target) or {}).get(CONTRACT_KEY))


def program_axes(trace: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    """The axis values an EMITTED program exercises, read off its decoded instruction trace.

    A capsule states what it asks for; which instructions answer it is the backend's choice, and a
    blind spot is about the instructions. A backend that reaches for a loop command on a capsule
    whose declaration never mentions one has put that program in the loop command's class.
    """
    found: set[tuple[str, str]] = set()
    for instruction in (trace or {}).get("instructions") or ():
        if isinstance(instruction, Mapping) and instruction.get("class"):
            found.add(("instruction_class", str(instruction["class"])))
    return found


def blind_tiers(
    spots: Iterable[BlindSpot], capsule: Mapping[str, Any], *, program: Iterable[tuple[str, str]] = ()
) -> dict[str, list[dict[str, Any]]]:
    """``{tier: [{id, via}]}`` for every tier that cannot judge ``capsule``.

    ``program`` adds what the emitted program was observed to use (:func:`program_axes`).
    """
    axes = tier_policy.capsule_axes(capsule) | set(program)
    out: dict[str, list[dict[str, Any]]] = {}
    for spot in spots:
        via = spot.concerns(axes)
        if not via:
            continue
        for tier in spot.tiers:
            out.setdefault(tier, []).append({"id": spot.id, "via": [list(pair) for pair in via]})
    return out


def required_tier(spots: Iterable[BlindSpot], capsule: Mapping[str, Any], declared_tiers: Iterable[str]) -> str | None:
    """The shallowest of ``declared_tiers`` that is blind to none of the capsule's axes.

    Every tier shallower than the deepest blind one is refused too, even if no entry names it: a
    ladder is ordered by what a tier observes, and a blind spot is a statement about a depth.
    """
    ladder = tier_policy.tier_depth_order(declared_tiers)
    blind = blind_tiers(spots, capsule)
    if not blind:
        return ladder[0] if ladder else None
    universe = tier_policy.tier_depth_order([*ladder, *blind])
    deepest_blind = max(universe.index(tier) for tier in blind)
    for tier in ladder:
        if universe.index(tier) > deepest_blind:
            return tier
    return None


@dataclass
class Audit:
    target: str
    capsules: int = 0
    concerned: int = 0
    findings: list[dict[str, Any]] = field(default_factory=list)
    unexercised: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "target": self.target,
            "capsules": self.capsules,
            "capsules_in_a_blind_class": self.concerned,
            "findings": self.findings,
            "blind_spots_no_capsule_exercises": self.unexercised,
        }


def audit(target: str, capsules: Iterable[Mapping[str, Any]], *, spots: Sequence[BlindSpot] | None = None) -> Audit:
    """Capsules that can only ever be judged by a tier blind to what they exercise.

    A capsule's reach is the deepest of its required tiers, lowered by a declared correctness
    ceiling. When that reach does not get past the blind tiers for its own axes, a pass is a screen
    and the corpus is claiming a certification no tier it runs can give.
    """
    spots = tuple(spots) if spots is not None else for_target(target)
    report = Audit(target=target)
    exercised: set[str] = set()
    for capsule in capsules:
        report.capsules += 1
        blind = blind_tiers(spots, capsule)
        if not blind:
            continue
        report.concerned += 1
        exercised.update(row["id"] for rows in blind.values() for row in rows)
        declared = [str(t) for t in capsule.get("required_oracle_tiers") or ()]
        cap, _source = tier_policy.declared_ceiling(capsule)
        ladder = tier_policy.tier_depth_order([*declared, *([cap] if cap else [])])
        reach = [t for t in ladder if not cap or ladder.index(t) <= ladder.index(cap)]
        needed = required_tier(spots, capsule, reach)
        if needed is not None:
            continue
        seeing = required_tier(spots, capsule, declared)
        report.findings.append(
            {
                "capsule": str(capsule.get("name") or ""),
                "finding": UNCERTIFIABLE if seeing is not None else NO_SEEING_TIER,
                "reaches": reach[-1] if reach else None,
                "declared_ceiling": cap,
                "needs": seeing,
                "blind": blind,
            }
        )
    report.unexercised = sorted(spot.id for spot in spots if spot.id not in exercised)
    return report


def judge_result(
    spots: Iterable[BlindSpot],
    capsule: Mapping[str, Any],
    tiers: Mapping[str, Any],
    trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Whether the tiers a graded program PASSED can see what that program did.

    ``tiers`` is a result's per-tier record. The deepest passing tier has to lie past every tier
    blind to the capsule's axes or to the instructions the program was observed to use; otherwise
    the pass is a screen. Reported beside a grade, and never a substitute for one.
    """
    passed = [str(t) for t, row in tiers.items() if isinstance(row, Mapping) and row.get("status") == "pass"]
    blind = blind_tiers(spots, capsule, program=program_axes(trace))
    record: dict[str, Any] = {"blind": blind, "passed_tiers": tier_policy.tier_depth_order(passed)}
    if not blind or not passed:
        record["seen"] = not blind
        return record
    universe = tier_policy.tier_depth_order([*passed, *blind])
    deepest_blind = max(universe.index(tier) for tier in blind)
    record["seen"] = any(universe.index(tier) > deepest_blind for tier in passed)
    if not record["seen"]:
        record["finding"] = "passed_only_where_the_oracle_cannot_see"
    return record
