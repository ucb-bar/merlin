"""Decide a move-in placement family from the candidate's OWN emitted traces, PER REGIME.

The third ``EMITS`` family, and the first whose demand is not made of every member. That is not a
softening: it is the measured fact that shapes it. On pinned elaborated RTL, moving the staging off
the front of the nest and into it measured -22.3% and -12.7% on two shapes and -3.2% and -0.6% on two
others. The two it paid on are LOAD-critical and the two it did nothing on are MESH-critical, and on
a mesh-critical shape the order that keeps the array's operand resident stages a whole run of moving
windows together -- so a demand for just-in-time placement there would FIGHT the demand that actually
pays (:mod:`merlin.analysis.perf.stationary_claim`). A corpus that asked for it everywhere would be asking a
compiler to be slower on two thirds of its work.

So this family sorts its members into three buckets and demands the property of one of them:

* ``lever_bearing`` -- a load-critical member with more than one compute position. Hoisting is
  expressible and just-in-time placement is distinguishable from it, so the property is refutable.
* ``control`` -- a single-position member. The whole movement set IS what the first compute reads, so
  the exposed prefix is compulsory and ANY correct program passes. This is what makes a
  lever-bearing failure attributable to the placement rather than to something broader.
* ``exempt`` -- a mesh-critical member. The demand is WAIVED and the waiver is recorded.

THE EXEMPT BUCKET IS REQUIRED, not merely tolerated, and that is the honest part. A family that
demanded the property everywhere and a family that conditioned it on the regime are the same family
until a member actually falls in the exempt bucket. Requiring one makes the conditioning visible in
every report, and makes deleting it a test failure rather than a silent strengthening.

THE REGIME BOUNDARY IS NOT GUESSED, AND IT IS NOT NEEDED. The discriminant is the reload multiplicity
(see :func:`merlin.analysis.perf.schedule_proxy._regime`), and the four measured points bound the boundary to
the open interval (4, 7] and no further -- one of them carries a ~7% per-arm noise floor. Rather than
pick a number inside that interval, this classifies only OUTSIDE it: at or below the lower edge a
member is load-critical under every boundary the evidence admits, at or above the upper edge it is
mesh-critical under every one, and a member landing strictly between is UNCLASSIFIABLE and refuses
the cohort. The two edges are read from the family's own frozen acceptance -- device data the selected
target's recipe supplies and the profile merges in, reviewed with the declaration -- never from a literal
here, and a contract declaring neither gets a refusal.

THE CLASSIFICATION IS NOT A BOUND. It decides only which bucket a member lands in; the pass/fail
arithmetic comes entirely from the trace. A mis-classification can weaken the discrimination report,
which is visible, but it cannot manufacture a failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.analysis.perf import movein_placement as MP
from merlin.analysis.perf import stationary_residency as SR

ANALYZER = "merlin.analysis.perf.movein_claim.analyze_movein_claim/v1"

ESTABLISHED = "ESTABLISHED"
REFUTED = "REFUTED"
REFUSED = "REFUSED"

LEVER_BEARING = "lever_bearing"
CONTROL = "control"
EXEMPT = "exempt"

SCHEMA_VERSION = 1
READY = "READY"


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"verdict": REFUSED, "reason": reason, **extra}


def _refused(family: str | None, *reasons: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "claim": "EMITS",
        "status": REFUSED,
        "declaration": None,
        "cohort": None,
        "replicates": [],
        "expected_identities": [],
        "unresolved_facts": [],
        "refusal_reasons": [str(reason) for reason in reasons],
    }


def regime_bracket(acceptance: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """``(load_critical_at_or_below, mesh_critical_at_or_above)`` from the frozen contract, or ``None``.

    Read from the declaration because the interval is MEASURED evidence about a device, not a
    property of this code, and because a number a reviewer never saw is a knob. Both edges must be
    present and the lower must be strictly below the upper, or there is no band to be undecided in
    and the classification would be a coin toss dressed as a derivation.
    """
    if not isinstance(acceptance, Mapping):
        return None
    block = acceptance.get("regime")
    if not isinstance(block, Mapping):
        return None
    low, high = block.get("load_critical_at_or_below"), block.get("mesh_critical_at_or_above")
    for value in (low, high):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return None
    return (int(low), int(high)) if int(low) < int(high) else None


def _undeclared_bracket(acceptance: Mapping[str, Any]) -> str:
    """Why no bracket was read, in the contract's own words when it recorded any.

    The edges are DEVICE data the selected target supplies; a contract minted for a target that
    supplied none records that in ``regime.bracket``, and the refusal repeats it so the reason names
    where the missing measurement belongs rather than only that it is missing.
    """
    regime = acceptance.get("regime") if isinstance(acceptance, Mapping) else None
    note = regime.get("bracket") if isinstance(regime, Mapping) else None
    return f" (the contract records the bracket as {note!r})" if isinstance(note, str) and note else ""


def admitted_positions(acceptance: Mapping[str, Any] | None) -> int | None:
    """How many compute positions' worth of staging may stand in front of the nest, or ``None``."""
    if not isinstance(acceptance, Mapping):
        return None
    value = (acceptance.get("property") or {}).get("admitted_staging_positions")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def _geometry(capsule: Mapping[str, Any]) -> Mapping[str, Any] | None:
    performance = capsule.get("performance") if isinstance(capsule, Mapping) else None
    geometry = performance.get("shape_geometry") if isinstance(performance, Mapping) else None
    return geometry if isinstance(geometry, Mapping) else None


def _extents(capsule: Mapping[str, Any]) -> tuple[int, int, int] | None:
    geometry = _geometry(capsule)
    if geometry is None:
        return None
    values = []
    for axis in ("M", "K", "N"):
        value = geometry.get(axis)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return None
        values.append(int(value))
    return (values[0], values[1], values[2])


def member_row_block(capsule: Mapping[str, Any], supplied: object = None) -> int | None:
    """The array row block a member's extents are counted in, or ``None``.

    Prefers the observer's, which was read from the target about to be run on; falls back to the one
    the capsule records, which is what its extents were minted against -- so a caller holding only
    the declaration can classify a member without resolving a target.
    """
    geometry = _geometry(capsule)
    for value in (supplied, (geometry or {}).get("row_block")):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return int(value)
    return None


def declared_positions(capsule: Mapping[str, Any], *, row_block: object = None) -> int | None:
    """How many compute positions the declared shape tiles into, or ``None``."""
    block = member_row_block(capsule, row_block)
    extents = _extents(capsule) if block else None
    if extents is None or block is None:
        return None
    m, k, n = extents
    return (-(-m // block)) * (-(-k // block)) * (-(-n // block))


def declared_reload_multiplicity(capsule: Mapping[str, Any], *, row_block: object = None) -> int | None:
    """The discriminant: how many output row blocks the declared shape spans, or ``None``."""
    block = member_row_block(capsule, row_block)
    extents = _extents(capsule) if block else None
    return None if extents is None or block is None else -(-extents[0] // block)


def classify(capsule: Mapping[str, Any], *, bracket: tuple[int, int] | None, row_block: object = None) -> str | None:
    """``LEVER_BEARING`` / ``CONTROL`` / ``EXEMPT`` / ``None`` when the evidence does not settle it.

    CONTROL is decided first and on its own terms: a single-position member is one where the whole
    movement set is compulsory before the first compute, so any correct program passes it whatever
    regime it is in. Only then does the regime decide, and only outside the undecided band.
    """
    positions = declared_positions(capsule, row_block=row_block)
    if positions is None or bracket is None:
        return None
    if positions <= 1:
        return CONTROL
    multiplicity = declared_reload_multiplicity(capsule, row_block=row_block)
    if multiplicity is None:
        return None
    low, high = bracket
    if multiplicity <= low:
        return LEVER_BEARING
    if multiplicity >= high:
        return EXEMPT
    return None


def _buckets(
    descriptors: Sequence[Any], *, row_block: object, bracket: tuple[int, int] | None
) -> tuple[dict[str, list[str]], list[str]]:
    buckets: dict[str, list[str]] = {LEVER_BEARING: [], CONTROL: [], EXEMPT: []}
    unclassified: list[str] = []
    for descriptor in descriptors:
        name = str(descriptor.get("name") or "<unnamed>") if isinstance(descriptor, Mapping) else "<unnamed>"
        bucket = classify(descriptor, row_block=row_block, bracket=bracket) if isinstance(descriptor, Mapping) else None
        if bucket is None:
            unclassified.append(name)
        else:
            buckets[bucket].append(name)
    return buckets, unclassified


def _lanes(evidence: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs = [
        (evidence.get("structural_instrument"), evidence.get("structural_tier")),
        (evidence.get("correctness_simulator"), evidence.get("correctness_tier")),
    ]
    return [(str(a), str(b)) for a, b in pairs if isinstance(a, str) and a and isinstance(b, str) and b]


def preflight_movein_evidence(
    descriptors: object, *, replicates: Sequence[str], target: object = None
) -> dict[str, Any]:
    """The single launch-time precondition: can this cohort decide anything, and only where it should?

    Five things must hold: one frozen contract naming its lanes; a declared staging depth and a
    declared regime bracket; a derivable array row block; all three buckets non-empty -- including
    ``exempt``, without which the regime conditioning is inert and this family is indistinguishable
    from one demanding just-in-time placement of shapes it is measured not to help.
    """
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        return _refused(None, "no capsule descriptors were supplied")
    if not isinstance(replicates, Sequence) or isinstance(replicates, str) or not replicates:
        return _refused(None, "the run authored no replicate schedule")
    performances = [d.get("performance") if isinstance(d, Mapping) else None for d in descriptors]
    if any(not isinstance(p, Mapping) for p in performances):
        return _refused(None, "a descriptor carries no performance block")
    families = {str(p.get("family")) for p in performances}
    family = next(iter(families)) if len(families) == 1 else None
    if family is None:
        return _refused(None, f"descriptors span more than one family: {sorted(families)}")
    if any(p.get("claim") != "EMITS" for p in performances):
        return _refused(family, "this procedure decides EMITS claims only")
    contracts = [p.get("acceptance") for p in performances]
    if any(not isinstance(c, Mapping) for c in contracts):
        return _refused(family, "a member carries no frozen acceptance contract")
    if any(c != contracts[0] for c in contracts):
        return _refused(family, "the members do not agree on one frozen acceptance contract")
    contract = contracts[0]
    lanes = _lanes(contract.get("evidence") if isinstance(contract.get("evidence"), Mapping) else {})
    if not lanes:
        return _refused(family, "the frozen acceptance names no instrument and tier to observe at")
    if admitted_positions(contract) is None:
        return _refused(
            family,
            "the frozen acceptance declares no admitted staging depth, so how much movement may "
            "stand in front of the nest is UNDECLARED and no placement could be decided",
        )
    bracket = regime_bracket(contract)
    if bracket is None:
        return _refused(
            family,
            "the frozen acceptance declares no regime bracket, so which members this demand applies "
            "to is UNDECLARED -- and applying it to all of them is measured to ask for nothing on "
            "mesh-critical shapes and to fight the reuse order that does pay there" + _undeclared_bracket(contract),
        )

    row_block = SR.array_row_block(str(target)) if target is not None else None
    if target is not None and row_block is None:
        return _refused(
            family,
            f"target {str(target)!r} does not derive an array row count from its own extracted "
            "geometry, so no member's regime or position count can be read",
        )
    buckets, unclassified = _buckets(descriptors, row_block=row_block, bracket=bracket)
    if unclassified:
        return _refused(
            family,
            "these members land inside the band the four measured points do not separate "
            f"({bracket[0]} < reload multiplicity < {bracket[1]}), or declare extents this family "
            f"cannot read, so which lever pays on them is UNKNOWN: {sorted(unclassified)}",
        )
    if not buckets[LEVER_BEARING]:
        return _refused(
            family,
            "no member of this cohort is a load-critical multi-position shape, so every member would "
            "pass without the lever; a demand that cannot fail is not a demand",
        )
    if not buckets[CONTROL]:
        return _refused(
            family,
            "this cohort carries no control -- no single-position member whose whole movement set is "
            "compulsory before the first compute -- so a failure could not be attributed to the "
            "placement this family names",
        )
    if not buckets[EXEMPT]:
        return _refused(
            family,
            "this cohort carries no exempt member -- no mesh-critical shape on which the demand is "
            "waived -- so its regime conditioning is never exercised and it is indistinguishable "
            "from a family demanding just-in-time placement of shapes it is measured not to help",
        )
    expected = [
        {
            "family": family,
            "capsule": str(descriptor.get("name")),
            "role": classify(descriptor, row_block=row_block, bracket=bracket),
            "simulator": instrument,
            "replicate": replicate,
            "tier": tier,
        }
        for descriptor in descriptors
        for replicate in replicates
        for instrument, tier in lanes
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "claim": "EMITS",
        "status": READY,
        "declaration": dict(contract),
        "cohort": {key: sorted(value) for key, value in buckets.items()},
        "replicates": [str(value) for value in replicates],
        "expected_identities": expected,
        "unresolved_facts": [] if row_block else ["array_row_block"],
        "refusal_reasons": [],
        "row_block": row_block,
        "regime_bracket": list(bracket),
        "admitted_staging_positions": admitted_positions(contract),
    }


def analyze_movein_claim(descriptors: object, results: object, per_unit: object = None) -> dict[str, Any]:
    """Decide the family over already-observed traces, demanding the property of one bucket only.

    ``results`` rows carry ``capsule``, the ``row_block`` the observation was classified with, and
    either a ``verdict`` produced by :func:`merlin.analysis.perf.movein_placement.placement_verdict` or the
    ``trace`` (and the ``target`` it was emitted for) to run it over. Every declared member must
    report, exempt ones included -- their verdict is RECORDED and not required, which is what a
    waiver is.
    """
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        return _fail("no capsule descriptors were supplied")
    if not isinstance(results, Sequence) or isinstance(results, str) or not results:
        return _fail("no observed rows were supplied")

    families: set[str] = set()
    contracts: list[Any] = []
    declared: dict[str, Mapping[str, Any]] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            return _fail("a descriptor is not a mapping")
        performance = descriptor.get("performance")
        if not isinstance(performance, Mapping):
            return _fail("a descriptor carries no performance block")
        if performance.get("claim") != "EMITS":
            return _fail("this procedure decides EMITS claims only", observed_claim=performance.get("claim"))
        families.add(str(performance.get("family")))
        contracts.append(performance.get("acceptance"))
        declared[str(descriptor.get("name") or "<unnamed>")] = descriptor
    if len(families) != 1:
        return _fail("descriptors span more than one family", families=sorted(families))
    if any(not isinstance(contract, Mapping) for contract in contracts):
        return _fail("a member carries no frozen acceptance contract")
    if any(contract != contracts[0] for contract in contracts):
        return _fail("the members do not agree on one frozen acceptance contract")
    contract = contracts[0]
    bracket = regime_bracket(contract)
    positions = admitted_positions(contract)
    if bracket is None or positions is None:
        return _fail(
            "the frozen acceptance declares no regime bracket or no admitted staging depth, so "
            "neither which members this demand applies to nor what it admits of them is stated"
            + _undeclared_bracket(contract)
        )

    # Supplied by the observer when it had a target; otherwise each member falls back to the
    # granularity it was minted against, recorded in its own declaration.
    blocks = {row.get("row_block") for row in results if isinstance(row, Mapping)}
    blocks.discard(None)
    if len(blocks) > 1:
        return _fail(
            "the observed rows disagree about the array row block, so they were not made against "
            "one machine and no member's regime can be read",
            row_blocks=sorted(str(b) for b in blocks),
        )
    row_block = next(iter(blocks)) if blocks else None

    buckets: dict[str, str] = {}
    for name, descriptor in declared.items():
        bucket = classify(descriptor, row_block=row_block, bracket=bracket)
        if bucket is None:
            return _fail(
                f"member {name!r} lands inside the band the measured points do not separate, or "
                "declares extents this family cannot read, so which lever pays on it is UNKNOWN"
            )
        buckets[name] = bucket
    for bucket, why in (
        (LEVER_BEARING, "no member of this cohort could refute the claim; it would pass without the lever"),
        (CONTROL, "this cohort carries no control, so a failure cannot be attributed to the placement"),
        (
            EXEMPT,
            "this cohort carries no exempt member, so its regime conditioning is never exercised and "
            "it is indistinguishable from a family demanding the property where it is measured not to pay",
        ),
    ):
        if bucket not in buckets.values():
            return _fail(why, cohort={n: b for n, b in sorted(buckets.items())})

    observed: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, Mapping):
            return _fail("an observed row is not a mapping")
        name = str(row.get("capsule") or "")
        if name not in buckets:
            return _fail(f"observed row names {name!r}, which is not a declared member of this family")
        verdict = row.get("verdict")
        if not isinstance(verdict, Mapping):
            trace = row.get("trace")
            if trace is None:
                return _fail(f"member {name!r} reports neither a verdict nor a trace to decide one from")
            try:
                verdict = MP.placement_verdict(trace, admitted_positions=positions, target=row.get("target"))
            except TypeError as exc:
                return _fail(f"member {name!r} reports an unusable trace: {exc}")
        observed[name] = dict(verdict)

    missing = sorted(set(buckets) - set(observed))
    if missing:
        return _fail("these declared members did not report", missing=missing)

    controls_failed = sorted(n for n, b in buckets.items() if b == CONTROL and observed[n].get("verdict") == MP.FAIL)
    if controls_failed:
        return _fail(
            "a control member failed -- on these the whole movement set is what the first compute "
            "reads, so the exposed prefix is compulsory and the defect this family names cannot "
            "show. A program that hoists HERE hoists what it had to, which is a different and "
            "broader reading, and no lever-bearing failure beside it can be attributed to the "
            "placement. The cohort is refused rather than scored",
            controls_failed=controls_failed,
            members=observed,
        )
    undecided = sorted(n for n, b in buckets.items() if b != EXEMPT and observed[n].get("verdict") == MP.REFUSED)
    if undecided:
        return _fail(
            "these members could not be decided, so the family is undecided rather than passed",
            undecidable=undecided,
            reasons={n: observed[n].get("reason") for n in undecided},
            members=observed,
        )
    waived = sorted(n for n, b in buckets.items() if b == EXEMPT)
    failed = sorted(n for n, b in buckets.items() if b == LEVER_BEARING and observed[n].get("verdict") == MP.FAIL)
    if failed:
        return {
            "verdict": REFUTED,
            "reason": (
                "these load-critical members stage operands in front of the nest that no compute is "
                "outstanding to overlap, so the runtime waits through movement it could have placed "
                "beside the positions that read it"
            ),
            "failed": failed,
            "exposed_prefix": {n: observed[n].get("exposed_prefix") for n in failed},
            "excess": {n: observed[n].get("excess") for n in failed},
            "waived": waived,
            "waiver": (
                "the demand was NOT made of these mesh-critical members: just-in-time placement is "
                "measured to do nothing on them, and the order that does pay there stages a run of "
                "moving windows together"
            ),
            "waived_verdicts": {n: observed[n].get("verdict") for n in waived},
            "members": observed,
            "cohort": {name: bucket for name, bucket in sorted(buckets.items())},
        }
    return {
        "verdict": ESTABLISHED,
        "reason": (
            "every load-critical member stages no more in front of the nest than its first compute "
            "position reads, and every control does too. The exposed staging is absent where it is "
            "measured to cost; this is not a claim about movement volume or mid-stream placement"
        ),
        "waived": waived,
        "waiver": (
            "the demand was NOT made of these mesh-critical members, by declaration rather than by "
            "omission; their observed verdicts are recorded beside this one"
        ),
        "waived_verdicts": {n: observed[n].get("verdict") for n in waived},
        "members": observed,
        "cohort": {name: bucket for name, bucket in sorted(buckets.items())},
    }
