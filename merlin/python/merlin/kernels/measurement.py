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
    #: Where the declaration was read from, so a reader never has to guess which file answered.
    source: str | None = None
    #: Why the lookup failed, when it did. An undeclared authority and an UNREADABLE one are
    #: different facts: the first is a policy statement about the target, the second is a broken
    #: lookup wearing that statement's clothes. Reporting the second as the first sends whoever reads
    #: the run record to argue about the target's contract instead of fixing the path.
    lookup_error: str | None = None

    def gaps(self) -> tuple[str, ...]:
        """What this target cannot measure, stated. An empty authority is a gap, not a zero."""
        if not self.declared:
            if self.lookup_error:
                return (f"{self.target}: measurement authority could NOT BE READ ({self.lookup_error})"
                        f" — this is a failed lookup, not a target that declares nothing; cycles, "
                        f"wall time and attainment are UNKNOWN until it resolves",)
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
                "declared": self.declared, "gaps": list(self.gaps()), "notes": list(self.notes),
                "source": self.source, "lookup_error": self.lookup_error}


def _tracked_measurement_block(target: str) -> tuple[dict, str | None]:
    """The ``measurement:`` block from the target's REPO-ANCHORED contract, ignoring MERLIN_OUT_ROOT.

    Returns ``(block, error)``. Reads the reviewed contract and, failing that, the residual that the
    contract is generated from — the residual is where the declaration is authored, so it answers
    even for a target whose contract has not been regenerated. Never raises: a failure comes back as
    a reason string so the caller can report a broken lookup as a broken lookup.
    """
    from merlin.common.paths import targets_dir, tracked_out_dir
    from merlin.common.yaml import load_yaml

    homes = (targets_dir() / target, tracked_out_dir() / "artifacts" / "targets" / target)
    errors: list[str] = []
    for home in homes:
        for name in ("target_contract.yaml", "residual.yaml"):
            path = home / "contracts" / name
            if not path.is_file():
                continue
            try:
                doc = load_yaml(path) or {}
            except Exception as exc:  # noqa: BLE001 — a malformed tracked file is a real answer
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            block = dict((doc.get("measurement") or {}))
            if block:
                return block, None
    return {}, ("; ".join(errors) if errors else None)


def authority_for(target: str, descriptor: dict | None = None) -> MeasurementAuthority:
    """The declared authority for ``target``. Never guesses a substrate."""
    block: dict = {}
    source: str | None = None
    lookup_error: str | None = None
    if descriptor is not None:
        block = dict((descriptor.get("measurement") or {}))
        source = "descriptor"
    else:
        try:
            from merlin.targetgen.target_experiment import load_capability_manifest
            contract = load_capability_manifest(target).contract
            block = dict((contract.get("measurement") or {}))
            source = "capability_manifest"
        except Exception as exc:  # noqa: BLE001
            lookup_error = f"{type(exc).__name__}: {exc}"
        if not block:
            # The generated targets' contracts are TRACKED files that happen to live under `out/`, and
            # the registry resolves them through the REDIRECTABLE out root. So a run with
            # MERLIN_OUT_ROOT pointed elsewhere (a test, a worktree, a relocated checkout) finds no
            # contract and concludes the target declares no authority -- at which point every cycle,
            # wall time and attainment silently becomes UNKNOWN and the search loses its objective.
            # A committed declaration must not depend on where output is being written, so fall back
            # to the repo-anchored copy and record that that is what answered.
            tracked, err = _tracked_measurement_block(target)
            if tracked:
                block, source, lookup_error = tracked, "tracked_contract", None
            elif err and not lookup_error:
                lookup_error = err
    if not block:
        return MeasurementAuthority(target=target, declared=False, lookup_error=lookup_error)
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
        source=source,
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
