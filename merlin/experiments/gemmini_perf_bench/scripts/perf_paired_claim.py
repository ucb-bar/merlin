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

from collections.abc import Mapping, Sequence
from typing import Any

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


def analyze_paired_claim(descriptors: object, results: object) -> dict[str, Any]:
    """Decide one differential family over already-measured arms.

    ``results`` rows carry ``capsule``, ``arm`` (a declared comparison role), ``replicate`` and
    ``cycles``. Every member must report both of its declared arms.
    """
    if not isinstance(descriptors, Sequence) or not descriptors:
        return _fail("no capsule descriptors were supplied")
    if not isinstance(results, Sequence) or not results:
        return _fail("no measured rows were supplied")

    contracts, families = [], set()
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
            return {"verdict": REFUTED, "rows": rows, "reason": (
                f"{len(indistinguishable)} member(s) cost the same on both arms within their own "
                f"replicate band, so the two are not distinguishable"),
                "indistinguishable": indistinguishable[:8]}
        return {"verdict": ESTABLISHED, "rows": rows,
                "reason": f"all {len(judged)} member(s) separate the two arms beyond their band"}

    sign = 1.0 if predicted == roles[0] else -1.0
    losers = [r["capsule"] for r in judged if sign * r["delta_cycles"] <= r["replicate_band"]]
    if losers:
        return {"verdict": REFUTED, "rows": rows, "reason": (
            f"{predicted!r} was predicted cheaper and is not, on {len(losers)} member(s), beyond "
            f"their measured replicate band"), "members": losers[:8]}
    return {"verdict": ESTABLISHED, "rows": rows,
            "reason": (f"{predicted!r} is cheaper on all {len(judged)} member(s), beyond each "
                       f"member's own replicate band")}
