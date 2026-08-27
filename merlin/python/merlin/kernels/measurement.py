"""Which substrate is ALLOWED to produce which number, per target — declared, not hardcoded.

The beam picks its cycle count by looking for a measurement stamped ``spike`` and its wall time by
looking for one stamped ``k1``. The comment above that code is honest about what it is: the choice is
"INHERENT to this path, not a derivable per-target fact". That is fine for one target and wrong for
five — a target with no spike model silently yields no cycles, and a number read off the wrong
substrate is worse than no number because it gets cited.

Three separable things get conflated when there is no declaration, and each has burned a result:

* **Cycle authority vs wall authority.** Two substrates can both emit a ``cycles`` field while only one
  of them is authoritative; the other is an estimate derived from a timer. Picking by field name gets
  the estimate.
* **The TIER a number was reached at.** A functional simulator and an RTL simulation both produce
  cycles, and only one of them is a hardware claim. A headline quoted without its tier is the recorded
  failure where a completion certificate was read as an output-equality result.
* **The denominator.** Attainment needs a speed-of-light model, and which one is legitimate is a
  per-target fact, not a constant.

Declared per target in its ``target_experiment.yaml`` under ``measurement``; absent, this returns an
authority that answers UNKNOWN for everything rather than falling back to a default substrate. An
undeclared authority must not silently become somebody else's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["MeasurementAuthority", "authority_for", "pick", "citable"]

#: Tiers in ascending order of what they can claim, cheapest first. A number from a tier BELOW the one a
#: claim requires is not a weaker version of that claim, it is a different claim.
TIER_ORDER = ("static", "functional", "cycle_model", "rtl", "fpga", "silicon")


@dataclass(frozen=True)
class MeasurementAuthority:
    """Which substrate may produce which quantity for one target, and at what tier."""

    target: str
    cycles_from: str | None = None        # substrate label authoritative for cycle counts
    wall_from: str | None = None          # substrate label authoritative for wall time
    cycles_tier: str = "functional"
    wall_tier: str = "silicon"
    #: The attainment DENOMINATOR: the model a measured rate is compared against.
    speed_of_light: str | None = None
    #: Tier a number must reach before it may be quoted as a hardware result.
    citable_tier: str = "rtl"
    notes: tuple[str, ...] = ()
    declared: bool = False                # False = nothing was declared; every answer is UNKNOWN

    def gaps(self) -> tuple[str, ...]:
        """What this target cannot measure, stated. An empty authority is a gap, not a zero."""
        if not self.declared:
            return (f"{self.target}: no measurement authority declared — cycles, wall time and "
                    f"attainment are all UNKNOWN, which is NOT the same as zero",)
        out = []
        if not self.cycles_from:
            out.append(f"{self.target}: no cycle authority declared")
        if not self.wall_from:
            out.append(f"{self.target}: no wall-time authority declared")
        if not self.speed_of_light:
            out.append(f"{self.target}: no speed-of-light model, so attainment has no denominator")
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "cycles_from": self.cycles_from, "wall_from": self.wall_from,
                "cycles_tier": self.cycles_tier, "wall_tier": self.wall_tier,
                "speed_of_light": self.speed_of_light, "citable_tier": self.citable_tier,
                "declared": self.declared, "gaps": list(self.gaps()), "notes": list(self.notes)}


def authority_for(target: str, descriptor: dict | None = None) -> MeasurementAuthority:
    """The declared authority for ``target``. Never guesses a substrate."""
    block: dict = {}
    if descriptor is not None:
        block = dict((descriptor.get("measurement") or {}))
    else:
        try:
            from merlin.targetgen.target_experiment import load_capability_manifest
            contract = load_capability_manifest(target).contract
            block = dict((contract.get("measurement") or {}))
        except Exception:  # noqa: BLE001 — no descriptor: undeclared, and say so
            block = {}
    if not block:
        return MeasurementAuthority(target=target, declared=False)
    return MeasurementAuthority(
        target=target,
        cycles_from=block.get("cycles_from"),
        wall_from=block.get("wall_from"),
        cycles_tier=str(block.get("cycles_tier") or "functional"),
        wall_tier=str(block.get("wall_tier") or "silicon"),
        speed_of_light=block.get("speed_of_light"),
        citable_tier=str(block.get("citable_tier") or "rtl"),
        notes=tuple(block.get("notes") or ()),
        declared=True,
    )


def pick(measurements, authority: MeasurementAuthority, quantity: str = "cycles"):
    """The value of ``quantity`` from the substrate this target says is authoritative for it.

    Returns ``(value, substrate)``, or ``(None, None)`` when the authority is undeclared or the
    authoritative substrate did not report. Deliberately does NOT fall back to any other substrate that
    happens to carry the field: two substrates can both emit ``cycles`` while only one is authoritative,
    and picking by field name gets the timer-derived estimate.
    """
    want = authority.cycles_from if quantity == "cycles" else authority.wall_from
    if not want:
        return None, None
    field_name = {"cycles": "cycles", "wall": "wall_ns"}.get(quantity, quantity)
    for m in measurements or ():
        if str(m.get("target")) == want and m.get(field_name) is not None:
            return m.get(field_name), want
    return None, None


def citable(authority: MeasurementAuthority, tier_reached: str) -> bool:
    """May a number reached at ``tier_reached`` be quoted as a hardware result for this target?

    The recorded failure this prevents: a completion certificate quoted as an output-equality result,
    and a headline score quoted without the tier it was reached at. An unknown tier is NOT citable --
    fail closed, because the cost of a wrongly-cited number is that it gets repeated.
    """
    if not authority.declared or tier_reached not in TIER_ORDER:
        return False
    return TIER_ORDER.index(tier_reached) >= TIER_ORDER.index(authority.citable_tier)


def whole_model_objective(claimed_mac_fraction: float | None, attainment: float | None, *,
                          numerics_ok: bool) -> float | None:
    """The whole-model objective: claimed MAC fraction x measured attainment, gated on numerics.

    Never a bare pass count. A kernel-scoped score can look excellent while the model runs at a few
    percent of peak, because most of the arithmetic was never claimed by the schedule at all --
    measured across several models, 86-89% of linalg ops never reached the vectorized path.

    Fail-closed on numerics: a fast wrong answer scores None, not a number with a caveat attached.
    Returns None when either input is unknown, because an unknown factor is not 1.0.
    """
    if not numerics_ok:
        return None
    if claimed_mac_fraction is None or attainment is None:
        return None
    return float(claimed_mac_fraction) * float(attainment)
