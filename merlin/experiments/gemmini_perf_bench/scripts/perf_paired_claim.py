"""Decide a two-arm performance claim from MEASURED cycles.

A differential family says two programs doing the same work should not cost the same. Deciding that
was routed through ``merlin.perf.differential.compare``, which compares two composed demand
envelopes -- and on a target whose resource times neither add nor max (a partial composition) that
function refuses by construction, correctly: a cycle delta is not derivable from demands when the
composition is partial. Refusing is sound, but it left every differential family undecided, because
nothing else decided them either.

The evidence that DOES settle a paired claim is the pair of measurements. This module reads them.

Two things the declarations lacked and now carry, because without either the claim is unfalsifiable:

* WHICH ARM SHOULD WIN. ``comparison_roles`` were bare names -- ``resident``/``spilling``,
  ``minimum_barriers``/``barrier_after_every_job`` -- with nothing saying which is predicted to be
  cheaper. A pair with no predicted direction cannot be wrong, so it cannot be right either. A
  family may instead declare that the two merely DIFFER, which is a weaker but still refutable
  claim, and is the honest shape for a symmetric comparison such as two operand encodings.
* WHAT COUNTS AS A DIFFERENCE. The band is the measured dispersion of each pair's own replicates,
  never a constant. On a deterministic cycle-accurate oracle that dispersion is zero, so a single
  cycle is a real difference and nothing is averaged away.

The NEGATIVE CONTROL is checked, not just declared: a family names a member whose two arms are the
same program, and if that member shows a saving the instrument is measuring itself and the whole
cohort is REFUSED rather than scored.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from merlin.perf import barrier_arms as BA

ANALYZER = "perf_paired_claim.analyze_paired_claim/v1"

ESTABLISHED = "ESTABLISHED"
REFUTED = "REFUTED"
REFUSED = "REFUSED"

#: A family that predicts a difference without predicting its sign declares this instead of a role.
EITHER = "either"


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"verdict": REFUSED, "reason": reason, **extra}


def _dispersion(values: Sequence[float]) -> float:
    return (max(values) - min(values)) if len(values) > 1 else 0.0


def analyze_paired_claim(descriptors: object, results: object,
                         per_unit: object = None) -> dict[str, Any]:
    """Decide one differential family over already-measured arms.

    ``results`` rows carry ``capsule``, ``arm`` (a declared comparison role), ``replicate`` and
    ``cycles``. Every member must report both of its declared arms.

    ``per_unit`` maps capsule -> how many units the cheap arm REMOVED (for the synchronization
    family, completion points, from :func:`merlin.perf.barrier_arms.paired_removal`). It is required
    exactly when the family's own falsifier declares a per-unit GROWTH observation, and is ignored
    otherwise. A direction test cannot decide a growth claim: "this arm is cheaper on every member"
    is true of a saving that is CONSTANT across members, which is precisely what a per-unit claim
    denies. Deciding such a family on direction alone reports a verdict on an assertion nothing
    evaluated -- so when the counts are absent the cohort is REFUSED, never quietly passed.
    """
    if not isinstance(descriptors, Sequence) or not descriptors:
        return _fail("no capsule descriptors were supplied")
    if not isinstance(results, Sequence) or not results:
        return _fail("no measured rows were supplied")

    contracts, families, falsifiers = [], set(), []
    for d in descriptors:
        if not isinstance(d, Mapping):
            return _fail("a descriptor is not a mapping")
        perf = d.get("performance")
        if not isinstance(perf, Mapping):
            return _fail("a descriptor carries no performance block")
        if perf.get("claim") != "DIFFERENTIAL":
            return _fail("this procedure decides DIFFERENTIAL claims only",
                         observed_claim=perf.get("claim"))
        families.add(str(perf.get("family")))
        contracts.append(perf.get("acceptance"))
        falsifiers.append(perf.get("falsifier"))
    if len(families) != 1:
        return _fail("descriptors span more than one family", families=sorted(families))
    if any(not isinstance(c, Mapping) for c in contracts):
        return _fail("a member carries no frozen acceptance contract")
    contract = contracts[0]
    if any(c != contract for c in contracts):
        return _fail("members disagree about the frozen acceptance contract")
    if contract.get("analyzer") != ANALYZER:
        return _fail("the contract names a different analyzer",
                     declared=contract.get("analyzer"), this=ANALYZER)

    # The falsifier decides WHICH question this cohort answers, so it must be one question. Members
    # disagreeing here would mean half a family tested a direction and half tested a growth.
    if any(not isinstance(f, Mapping) for f in falsifiers):
        return _fail("a member carries no falsifier")
    falsifier = falsifiers[0]
    if any(f != falsifier for f in falsifiers):
        return _fail("members disagree about the falsifier")

    roles = contract.get("roles")
    if (not isinstance(roles, Sequence) or isinstance(roles, str) or len(roles) != 2
            or any(not isinstance(r, str) or not r for r in roles)):
        return _fail("the contract does not name exactly two comparison roles")
    roles = [str(r) for r in roles]
    predicted = contract.get("expected_faster")
    if predicted is None or (predicted != EITHER and predicted not in roles):
        return _fail("the contract does not predict which role is cheaper, nor that they merely "
                     "differ, so no measurement could contradict it", roles=roles)
    control = contract.get("negative_control_capsule")

    # gather cycles per (capsule, arm)
    per: dict[tuple[str, str], list[float]] = {}
    for row in results:
        if not isinstance(row, Mapping):
            return _fail("a measured row is not a mapping")
        capsule, arm = str(row.get("capsule") or ""), str(row.get("arm") or "")
        cycles = row.get("cycles")
        if arm not in roles:
            return _fail("a row names an arm the contract does not declare", arm=arm, roles=roles)
        if not isinstance(cycles, (int, float)) or isinstance(cycles, bool) or cycles <= 0:
            return _fail("a row carries no positive cycle count", capsule=capsule, arm=arm)
        per.setdefault((capsule, arm), []).append(float(cycles))

    capsules = sorted({c for c, _ in per})
    if not capsules:
        return _fail("no member reported a measured arm")
    missing = [f"{c}/{a}" for c in capsules for a in roles if (c, a) not in per]
    if missing:
        return _fail("a member did not report both declared arms", missing=missing[:8])

    rows, breaches = [], []
    for capsule in capsules:
        a_vals, b_vals = per[(capsule, roles[0])], per[(capsule, roles[1])]
        band = max(_dispersion(a_vals), _dispersion(b_vals))
        a, b = min(a_vals), min(b_vals)
        delta = b - a                       # positive => roles[0] is cheaper
        rows.append({"capsule": capsule, roles[0]: a, roles[1]: b,
                     "delta_cycles": delta, "replicate_band": band})
        if control is not None and capsule == str(control):
            if abs(delta) > band:
                breaches.append({"capsule": capsule, "delta_cycles": delta, "band": band})

    if breaches:
        return {"verdict": REFUSED, "rows": rows, "control_breaches": breaches,
                "reason": ("the negative control, whose arms are the same program, shows a cycle "
                           "difference beyond its own replicate band -- the instrument is measuring "
                           "itself, so no member of this cohort can be scored")}

    judged = [r for r in rows if str(control) != r["capsule"]] or rows
    if predicted == EITHER:
        indistinguishable = [r["capsule"] for r in judged
                             if abs(r["delta_cycles"]) <= r["replicate_band"]]
        if indistinguishable:
            outcome = {"verdict": REFUTED, "rows": rows, "reason": (
                f"{len(indistinguishable)} member(s) cost the same on both arms within their own "
                f"replicate band, so the two are not distinguishable"),
                "indistinguishable": indistinguishable[:8]}
        else:
            outcome = {"verdict": ESTABLISHED, "rows": rows,
                       "reason": f"all {len(judged)} member(s) separate the two arms beyond their band"}
    else:
        sign = 1.0 if predicted == roles[0] else -1.0
        losers = [r["capsule"] for r in judged if sign * r["delta_cycles"] <= r["replicate_band"]]
        if losers:
            outcome = {"verdict": REFUTED, "rows": rows, "reason": (
                f"{predicted!r} was predicted cheaper and is not, on {len(losers)} member(s), beyond "
                f"their measured replicate band"), "members": losers[:8]}
        else:
            outcome = {"verdict": ESTABLISHED, "rows": rows,
                       "reason": (f"{predicted!r} is cheaper on all {len(judged)} member(s), beyond "
                                  f"each member's own replicate band")}
    return _apply_growth_falsifier(outcome, falsifier, judged, roles, per_unit)


def _apply_growth_falsifier(outcome: dict[str, Any], falsifier: Mapping[str, Any],
                            judged: Sequence[Mapping[str, Any]], roles: Sequence[str],
                            per_unit: object) -> dict[str, Any]:
    """Fold the declared per-unit growth test into a direction verdict, or refuse for want of counts.

    Only ever downgrades. A cohort that fails the direction test is already decided, and a cohort
    that passes it has shown only that one arm is cheaper -- not that the saving scales with the
    count removed, which is the separate thing this family's falsifier asserts.
    """
    if str(falsifier.get("observation") or "") != BA.PER_UNIT_GROWTH_OBSERVATION:
        return outcome
    if outcome["verdict"] == REFUSED:
        return outcome
    if not isinstance(per_unit, Mapping) or not per_unit:
        return _fail(
            "this family's falsifier tests whether the saving GROWS with the count removed, and no "
            "per-unit counts were supplied -- the direction result alone does not decide it",
            declared_observation=falsifier.get("observation"), direction_verdict=outcome["verdict"],
            rows=outcome.get("rows"))
    points, missing = [], []
    for row in judged:
        capsule = str(row["capsule"])
        removed = per_unit.get(capsule)
        if not isinstance(removed, int) or isinstance(removed, bool):
            missing.append(capsule)
            continue
        # delta_cycles is positive when roles[0] -- the arm that removed the units -- is cheaper, so
        # it IS the saving those removals bought.
        points.append({"removed": removed, "cycles_saved": float(row["delta_cycles"])})
    if missing:
        return _fail("a member has no per-unit count, so its saving cannot be attributed",
                     members=missing[:8], rows=outcome.get("rows"))
    growth = BA.analyze_barrier_claim(points)
    merged = dict(outcome)
    merged["growth"] = growth
    if growth["verdict"] == BA.ESTABLISHED:
        merged["reason"] = f"{outcome['reason']}; {growth['reason']}"
        return merged
    merged["verdict"] = REFUTED if growth["verdict"] == BA.REFUTED else REFUSED
    merged["reason"] = (
        f"the arms separate ({outcome['reason']}), but the declared falsifier is about GROWTH and "
        f"it does not hold: {growth.get('reason')}")
    return merged


# ---------------------------------------------------------------------------------------------
# PREFLIGHT: may this cohort be measured at all?
#
# ``analyze_paired_claim`` refuses evidence that cannot decide the claim. That is the last line, not
# the first: by the time it speaks, every arm of every member has been simulated. The questions it
# asks about the DECLARATION -- are there two roles, is one of them predicted cheaper, is the band
# measured rather than assumed -- are answerable before any cycle is spent, and asking them at
# launch is what stops a campaign producing an undecidable cohort at full price.
#
# The replicate schedule is a PARAMETER here, exactly as it is for the residency family: these
# contracts state a FLOOR (``noise_band.minimum_replicate_count``, or ``acceptance.replicates``) and
# the run authors the identities. A floor the declaration does not state is not defaulted to two --
# it is refused, because a band this family calls MEASURED, measured over a schedule nobody
# declared, is a band the analyzer would compute from whatever the run happened to do.
# ---------------------------------------------------------------------------------------------

SCHEMA_VERSION = 1

#: The one band kind this procedure implements. A contract declaring a CONSTANT band is declaring a
#: different test -- the analyzer never reads a constant -- so it is refused rather than scored
#: against a bound nothing evaluates.
BAND_MEASURED_DISPERSION = "measured_replicate_dispersion"


class _Refusal(ValueError):
    """One named reason a declaration cannot be admitted."""


def _mapping_or_refuse(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _Refusal(f"{label} must be a mapping")
    return value


def _simple_name(value: object) -> bool:
    """A replicate identity: a non-empty name with no separators of its own."""
    return (isinstance(value, str) and bool(value) and value.strip() == value
            and not any(character.isspace() or character in "/\\" for character in value))


def _declared_family(descriptors: object) -> str | None:
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str):
        return None
    names = set()
    for descriptor in descriptors:
        performance = descriptor.get("performance") if isinstance(descriptor, Mapping) else None
        if isinstance(performance, Mapping):
            names.add(str(performance.get("family")))
    return names.pop() if len(names) == 1 else None


def _validated_declaration(descriptors: object,
                           replicates: Sequence[str]) -> tuple[list[Mapping[str, Any]],
                                                               dict[str, Any]]:
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        raise _Refusal("no capsule descriptors were supplied")
    members: list[Mapping[str, Any]] = []
    names: list[str] = []
    contracts: list[Any] = []
    falsifiers: list[Any] = []
    families: set[str] = set()
    for index, raw in enumerate(descriptors):
        descriptor = _mapping_or_refuse(raw, f"descriptor {index}")
        name = descriptor.get("name")
        if not isinstance(name, str) or not name:
            raise _Refusal(f"descriptor {index} has no capsule name")
        performance = _mapping_or_refuse(descriptor.get("performance"),
                                         f"descriptor {name!r} performance")
        if performance.get("claim") != "DIFFERENTIAL":
            raise _Refusal(
                f"descriptor {name!r} declares {performance.get('claim')!r}; this procedure "
                "decides DIFFERENTIAL claims only")
        families.add(str(performance.get("family")))
        contracts.append(performance.get("acceptance"))
        falsifiers.append(performance.get("falsifier"))
        names.append(name)
        members.append(descriptor)
    if len(set(names)) != len(names):
        raise _Refusal("the cohort repeats a capsule name")
    if len(families) != 1:
        raise _Refusal(f"descriptors span {len(families)} families {sorted(families)}")
    contract = _mapping_or_refuse(contracts[0], "the frozen acceptance contract")
    if any(entry != contract for entry in contracts):
        raise _Refusal("members disagree about the frozen acceptance contract")
    if contract.get("analyzer") != ANALYZER:
        raise _Refusal(
            f"the contract names analyzer {contract.get('analyzer')!r}, not {ANALYZER!r}")
    falsifier = _mapping_or_refuse(falsifiers[0], "the family's falsifier")
    if any(entry != falsifier for entry in falsifiers):
        raise _Refusal("members disagree about the falsifier, so the cohort asks two questions")

    roles = contract.get("roles")
    if (not isinstance(roles, Sequence) or isinstance(roles, str) or len(roles) != 2
            or any(not isinstance(role, str) or not role for role in roles)
            or roles[0] == roles[1]):
        raise _Refusal("the contract does not name exactly two distinct comparison roles")
    roles = [str(role) for role in roles]
    predicted = contract.get("expected_faster")
    if predicted is None or (predicted != EITHER and predicted not in roles):
        raise _Refusal(
            f"the contract predicts neither which role is cheaper nor that they merely differ "
            f"({EITHER!r}), so no measurement could contradict it")

    band = _mapping_or_refuse(contract.get("band"), "the contract's band")
    if band.get("kind") != BAND_MEASURED_DISPERSION:
        raise _Refusal(
            f"the contract declares a {band.get('kind')!r} band; this procedure decides against a "
            f"{BAND_MEASURED_DISPERSION!r} one and would otherwise ignore the declared bound")
    if band.get("declared_constant") is not None:
        raise _Refusal(
            "the contract declares a constant band beside a measured one; the analyzer reads the "
            "measured dispersion, so the constant would be declared and never applied")

    from merlin.perf import claim_reach
    schedules = []
    for descriptor in members:
        try:
            schedules.append(claim_reach.replicate_contract(descriptor["performance"]))
        except ValueError as exc:
            raise _Refusal(f"the replicate contract is malformed: {exc}") from exc
    # The floor lives beside the band rather than inside the acceptance block for these families, so
    # it is NOT covered by the acceptance-equality check above. A cohort whose members disagree
    # about it would have its schedule decided by whichever member happened to be read first.
    if any(entry != schedules[0] for entry in schedules):
        raise _Refusal("members disagree about the replicate floor their band is measured over")
    schedule = schedules[0]
    if schedule is None:
        raise _Refusal(
            "this family's band is the MEASURED replicate dispersion and its declaration states no "
            "replicate count; a schedule chosen by the run would make the band whatever the run "
            "happened to do")
    if schedule.minimum_count < 2:
        raise _Refusal(
            f"the declaration schedules {schedule.minimum_count} replicate(s); one leaves the "
            "dispersion UNDETERMINABLE rather than zero, and a zero band makes every single-cycle "
            "difference a result and the negative control unable to fire")
    identities = tuple(replicates or ())
    if not identities or any(not _simple_name(entry) for entry in identities):
        raise _Refusal("every replicate identity must be a simple non-empty name")
    if len(set(identities)) != len(identities):
        raise _Refusal("the replicate schedule repeats an identity")
    if schedule.identities and tuple(schedule.identities) != identities:
        raise _Refusal(
            f"the contract freezes replicates {list(schedule.identities)} and the run offers "
            f"{list(identities)}")
    if len(identities) < schedule.minimum_count:
        raise _Refusal(
            f"the run offers {len(identities)} replicate(s) against a declared floor of "
            f"{schedule.minimum_count} ({schedule.source})")

    control_declared = falsifier.get("negative_control")
    if not isinstance(control_declared, str) or not control_declared:
        raise _Refusal(
            "the family declares no negative control; a paired instrument that is never compared "
            "against itself cannot show that it is measuring the lever rather than itself")
    control_capsule = contract.get("negative_control_capsule")
    if control_capsule is not None and str(control_capsule) not in names:
        raise _Refusal(
            f"the contract names {str(control_capsule)!r} as its negative control and that capsule "
            "is not in this cohort, so the control would never be measured")

    unresolved: list[dict[str, Any]] = []
    if control_capsule is None:
        unresolved.append({
            "fact": "negative_control_capsule",
            "declared": control_declared,
            "detail": ("the family names its negative control in prose and the contract binds it to "
                       "no member, so the analyzer's control check has nothing to check; the "
                       "cohort is measurable and its control is NOT verified"),
        })

    growth = str(falsifier.get("observation") or "") == BA.PER_UNIT_GROWTH_OBSERVATION
    if growth:
        if len(members) < 2:
            raise _Refusal(
                "the falsifier asserts the saving GROWS with the count removed, and one paired "
                "point cannot show growth")
        unresolved.append({
            "fact": "per_unit_counts",
            "declared": falsifier.get("observation"),
            "detail": ("the decision needs how many units the cheap arm removed on each member; "
                       "that is measured with the arms and is not derivable from a descriptor, so "
                       "the run must seal it or the cohort is REFUSED at decision time"),
        })

    evidence = _mapping_or_refuse(contract.get("evidence"), "the contract's evidence block")
    lanes = []
    for simulator_key, tier_key in (("correctness_simulator", "correctness_tier"),
                                    ("timing_simulator", "timing_tier")):
        simulator, tier = evidence.get(simulator_key), evidence.get(tier_key)
        if not isinstance(simulator, str) or not simulator or not isinstance(tier, str) or not tier:
            raise _Refusal(f"the contract's evidence omits its {simulator_key}/{tier_key}")
        lanes.append((simulator, tier))

    resolved = {
        "contract": contract,
        "roles": roles,
        "lanes": lanes,
        "identities": identities,
        "cohort": {
            "negative_control": control_declared,
            "negative_control_capsule": (str(control_capsule) if control_capsule is not None
                                         else None),
            "roles": roles,
            "expected_faster": predicted,
            "capsules": list(names),
            "band": {"kind": band.get("kind"), "source": "measured replicate dispersion per pair"},
            "replicates": list(identities),
            "replicate_floor": schedule.minimum_count,
            "replicate_source": schedule.source,
            "per_unit_growth_falsifier": growth,
            "evidence_lanes": [{"simulator": simulator, "tier": tier} for simulator, tier in lanes],
        },
        "unresolved": unresolved,
    }
    return members, resolved


def preflight_paired_claim(descriptors: object, *,
                           replicates: Sequence[str]) -> dict[str, Any]:
    """Validate a frozen differential declaration and the run's replicate schedule.

    ``replicates`` is REQUIRED and has no default. These contracts declare a replicate FLOOR and let
    the run author the identities, so the schedule is a run fact; inventing one here would make the
    band this family calls measured depend on an analyzer assumption instead.
    """
    family = _declared_family(descriptors)
    try:
        members, resolved = _validated_declaration(descriptors, replicates)
    except (_Refusal, KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "family": family,
            "claim": "DIFFERENTIAL",
            "status": REFUSED,
            "declaration": None,
            "cohort": None,
            "replicates": [],
            "expected_identities": [],
            "unresolved_facts": [],
            "refusal_reasons": [str(exc)],
        }
    expected = [{"family": family, "capsule": str(descriptor.get("name")), "arm": arm,
                 "simulator": simulator, "replicate": replicate, "tier": tier}
                for descriptor in members
                for arm in resolved["roles"]
                for replicate in resolved["identities"]
                for simulator, tier in resolved["lanes"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "claim": "DIFFERENTIAL",
        "status": "READY",
        "declaration": copy.deepcopy(dict(resolved["contract"])),
        "cohort": resolved["cohort"],
        "replicates": list(resolved["identities"]),
        "expected_identities": expected,
        "unresolved_facts": resolved["unresolved"],
        "refusal_reasons": [],
    }
