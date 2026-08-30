"""What a cycle count may be COMPARED TO -- and who produced it.

A cycle count on its own is not a result. ``score_capsule.json``'s ``cycles_diagnostic`` block holds
``{capsule: {tier: cycles}}`` and nothing else, so the same capsule legitimately reads 1090, 3078 and
8889 across three submissions and there is no field in the record that says which of those numbers is
about the thing you care about. Two separate defects follow from that, and this module closes both.

**1. A number has to say which PROGRAM produced it.** ``TierResult.toolchain`` exists precisely to
record that -- a block-scaled capsule is graded on the harness's own reference kernel rather than on the
submission, so a pass there measures the fixture -- and it reaches ``cycles_diagnostic`` nowhere. That
is the exact shape of the recorded failure where a run scored 40/40 and nine of the forty were the
fixture. Worse, the field is populated by no adapter today, so the honest state of every count now on
disk is ``UNATTRIBUTED`` -- which is emphatically not "the submission". :func:`attribute` returns that
third state rather than defaulting to the flattering one.

**2. A number has to have something to be compared TO.** The corpus already declares one: four capsules
carry a ``comparison_group`` field, written when they were and consumed by nothing since. The comparand
it enables needs no performance model at all -- ``cycles(fused)`` against ``cycles(A) + cycles(B)`` at
the identical shape is arithmetic -- which is what makes fusion the cheapest level in the corpus to
measure and the reason those groups were declared in the first place. What was missing was the PARTS
(each declared group had exactly one member) and a reader that does the sum.

Three rules keep the arithmetic honest:

* **One tier, or no comparison.** Cycles from different tiers are different measurements; a fused count
  at an elaborated-RTL tier minus a part count at a functional one is not a saving. The group resolves
  at each tier where every member reported, and at no other.
* **Same submission, or not citable.** A group whose members were not all produced by the same program
  still reports its arithmetic -- it is useful diagnostically -- but ``citable`` is False and the
  reason names the programs involved.
* **A missing member is named, never dropped.** A group with no parts, or with a part that never
  reported at this tier, reports ``incomplete`` and says which member is missing. Silently summing the
  parts that did report would understate the parts and manufacture a fusion win.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FUSED", "PART", "SUBMISSION", "OTHER_PROGRAM", "UNATTRIBUTED",
    "Member", "GroupComparand", "attribute", "declared_groups", "toolchain_by_tier",
    "cycles_provenance", "fusion_comparands", "render",
]

#: The two roles a member of a comparison group may take.
FUSED, PART = "fused", "part"

#: What produced a cycle count.
SUBMISSION = "submission"          # the program under grade
OTHER_PROGRAM = "other_program"    # a named program that is NOT the submission (a harness fixture)
UNATTRIBUTED = "UNATTRIBUTED"      # the adapter reported no program at all -- NOT the submission


def attribute(toolchain: Any, *, submission: str | None) -> str:
    """Which program a tier's cycle count is about.

    Fails closed: an adapter that reported nothing yields ``UNATTRIBUTED``, never ``SUBMISSION``.
    Defaulting the other way is what let a fixture's pass travel as a submission's.
    """
    name = str(toolchain).strip() if toolchain else ""
    if not name:
        return UNATTRIBUTED
    if submission and name == str(submission).strip():
        return SUBMISSION
    return OTHER_PROGRAM


def toolchain_by_tier(tiers: Mapping[str, Any] | None, *,
                      ladder: Sequence[str] = ()) -> dict[str, str | None]:
    """The program each tier says it graded, keyed by tier -- the companion of ``cycles_by_tier``.

    Same shape rules as that function: a tier record in the bare-string form (``"pass"``) carries no
    fields and contributes nothing, and the result is ordered by the caller's ladder with unknown tiers
    appended. A tier that reported cycles but no toolchain appears here with ``None``, because "this
    count exists and nobody said what made it" is the fact that has to be visible.
    """
    found: dict[str, str | None] = {}
    for name, record in (tiers or {}).items():
        if not isinstance(record, Mapping):
            continue
        if record.get("cycles") is None:
            continue
        tc = record.get("toolchain")
        found[name] = str(tc) if tc else None
    ordered = {t: found[t] for t in ladder if t in found}
    ordered.update({t: v for t, v in found.items() if t not in ordered})
    return ordered


def cycles_provenance(tiers: Mapping[str, Any] | None, *, submission: str | None,
                      ladder: Sequence[str] = ()) -> dict[str, dict]:
    """``{tier: {cycles, toolchain, graded_program}}`` for one capsule.

    This is the block that sits ALONGSIDE ``cycles_diagnostic``: the counts keep their existing shape
    (readers parse them as integers) and the attribution rides beside them, so a count can no longer
    reach a summary without a statement of what produced it.
    """
    out: dict[str, dict] = {}
    for tier, tc in toolchain_by_tier(tiers, ladder=ladder).items():
        record = (tiers or {}).get(tier)
        out[tier] = {"cycles": record.get("cycles") if isinstance(record, Mapping) else None,
                     "toolchain": tc,
                     "graded_program": attribute(tc, submission=submission)}
    return out


@dataclass(frozen=True)
class Member:
    """One capsule's place in a comparison group, and the number it contributed."""

    capsule: str
    role: str
    cycles: int | None = None
    toolchain: str | None = None
    graded_program: str = UNATTRIBUTED

    def as_dict(self) -> dict:
        return {"capsule": self.capsule, "role": self.role, "cycles": self.cycles,
                "toolchain": self.toolchain, "graded_program": self.graded_program}


@dataclass(frozen=True)
class GroupComparand:
    """``cycles(fused)`` against ``cycles(part) + cycles(part)``, at ONE tier."""

    group: str
    tier: str | None
    fused: Member | None
    parts: tuple[Member, ...] = ()
    status: str = "incomplete"          # resolved | incomplete
    reason: str = ""
    #: True only when every contributing count was produced by the SAME program AND that program is
    #: the submission under grade. A False here does not invalidate the arithmetic; it forbids quoting.
    citable: bool = False

    @property
    def fused_cycles(self) -> int | None:
        return self.fused.cycles if self.fused else None

    @property
    def sum_of_parts(self) -> int | None:
        if not self.parts or any(p.cycles is None for p in self.parts):
            return None
        return sum(int(p.cycles) for p in self.parts)

    @property
    def saving_cycles(self) -> int | None:
        f, s = self.fused_cycles, self.sum_of_parts
        return None if f is None or s is None else s - f

    @property
    def saving_fraction(self) -> float | None:
        s, d = self.sum_of_parts, self.saving_cycles
        return None if not s or d is None else d / s

    def as_dict(self) -> dict:
        return {"group": self.group, "tier": self.tier, "status": self.status, "reason": self.reason,
                "citable": self.citable,
                "fused": self.fused.as_dict() if self.fused else None,
                "parts": [p.as_dict() for p in self.parts],
                "fused_cycles": self.fused_cycles, "sum_of_parts": self.sum_of_parts,
                "saving_cycles": self.saving_cycles, "saving_fraction": self.saving_fraction}


def declared_groups(capsules: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """``{group: {role: [capsule names]}}`` from the corpus's own ``comparison_group`` declarations.

    Accepts both spellings the corpus contains: the mapping form ``{name, role}``, and the bare string
    that four capsules have carried since they were written. A bare string names the group and states no
    role, so the member lands under an explicit ``"unspecified"`` role rather than being guessed into the
    fused slot -- guessing is how a part would be compared against itself.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for cap in capsules:
        decl = cap.get("comparison_group")
        if decl is None:
            continue
        if isinstance(decl, Mapping):
            group = str(decl.get("name") or "").strip()
            role = str(decl.get("role") or "unspecified").strip().lower()
        else:
            group, role = str(decl).strip(), "unspecified"
        if not group:
            continue
        out.setdefault(group, {}).setdefault(role, []).append(str(cap.get("name") or ""))
    for roles in out.values():
        for names in roles.values():
            names.sort()
    return out


def fusion_comparands(capsules: Sequence[Mapping[str, Any]],
                      cycles_diagnostic: Mapping[str, Mapping[str, Any]], *,
                      provenance: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
                      submission: str | None = None,
                      ladder: Sequence[str] = ()) -> dict[str, dict]:
    """The comparand for every declared group, keyed by group name.

    ``cycles_diagnostic`` is the grade's own block: ``{capsule: {tier: cycles}}``. ``provenance``, when
    given, is the companion attribution block (:func:`cycles_provenance` per capsule); without it every
    count is ``UNATTRIBUTED`` and no group is citable -- which is the true state of every graded run on
    disk today, since no adapter populates ``TierResult.toolchain``.
    """
    groups = declared_groups(capsules)
    out: dict[str, dict] = {}
    for group, roles in sorted(groups.items()):
        fused_names = roles.get(FUSED) or []
        part_names = roles.get(PART) or []
        unspecified = roles.get("unspecified") or []
        if not fused_names:
            out[group] = GroupComparand(
                group, None, None, (), "incomplete",
                (f"no member declares role {FUSED!r}"
                 + (f"; {len(unspecified)} member(s) declare the group as a bare name with no role, so "
                    f"which one is the fused implementation is not stated" if unspecified else ""))
            ).as_dict()
            continue
        if len(fused_names) > 1:
            out[group] = GroupComparand(
                group, None, None, (), "incomplete",
                f"{len(fused_names)} members declare role {FUSED!r} ({', '.join(fused_names)}); a group "
                f"compares ONE fused implementation against the parts it replaces").as_dict()
            continue
        if not part_names:
            out[group] = GroupComparand(
                group, None, None, (), "incomplete",
                f"{fused_names[0]} declares role {FUSED!r} but the group has no member with role "
                f"{PART!r}, so there is nothing to compare it against").as_dict()
            continue

        fused_name = fused_names[0]
        # ONE tier, or no comparison. Take the tiers where EVERY member reported, in ladder order.
        per_capsule = {n: dict(cycles_diagnostic.get(n) or {}) for n in [fused_name] + part_names}
        common = set(per_capsule[fused_name])
        for n in part_names:
            common &= set(per_capsule[n])
        common = {t for t in common if all(per_capsule[n].get(t) is not None
                                           for n in per_capsule)}
        if not common:
            missing = [n for n in per_capsule if not per_capsule[n]]
            out[group] = GroupComparand(
                group, None, None, (), "incomplete",
                (f"no tier reported a cycle count for every member"
                 + (f"; nothing was reported at all for {', '.join(sorted(missing))}" if missing else
                    f"; members reported at "
                    + "; ".join(f"{n}@{sorted(per_capsule[n])}" for n in sorted(per_capsule))))
            ).as_dict()
            continue
        ordered = [t for t in ladder if t in common] or sorted(common)
        tier = ordered[-1]      # the deepest tier both reported at: the most authoritative comparison

        def member(nm: str, role: str) -> Member:
            prov = ((provenance or {}).get(nm) or {}).get(tier) or {}
            tc = prov.get("toolchain")
            return Member(nm, role, int(per_capsule[nm][tier]), str(tc) if tc else None,
                          prov.get("graded_program") or attribute(tc, submission=submission))

        fused = member(fused_name, FUSED)
        parts = tuple(member(n, PART) for n in part_names)
        programs = {m.graded_program for m in (fused, *parts)}
        citable = programs == {SUBMISSION}
        reason = f"every member reported at {tier}; comparison is cycles(fused) vs sum(cycles(parts))"
        if not citable:
            reason += ("; NOT citable: " + ", ".join(
                f"{m.capsule} was graded by {m.toolchain or 'an unnamed program'} ({m.graded_program})"
                for m in (fused, *parts) if m.graded_program != SUBMISSION))
        out[group] = GroupComparand(group, tier, fused, parts, "resolved", reason, citable).as_dict()
    return out


def render(comparands: Mapping[str, Mapping[str, Any]]) -> str:
    """One line per group. A resolved-but-not-citable group prints its arithmetic AND its refusal."""
    lines = []
    for group, c in sorted(comparands.items()):
        if c.get("status") != "resolved":
            lines.append(f"{group}: INCOMPLETE — {c.get('reason')}")
            continue
        frac = c.get("saving_fraction")
        lines.append(
            f"{group}@{c['tier']}: fused {c['fused_cycles']} vs parts {c['sum_of_parts']} "
            f"(saving {c['saving_cycles']}"
            + (f", {frac * 100:.1f}%" if frac is not None else "") + ")"
            + ("" if c.get("citable") else "  [NOT CITABLE]"))
        if not c.get("citable"):
            lines.append(f"    {c.get('reason')}")
    return "\n".join(lines)
