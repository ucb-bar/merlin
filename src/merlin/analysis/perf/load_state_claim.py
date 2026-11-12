"""Decide a load-state residency family from the candidate's OWN emitted traces.

A family in this corpus has so far been decided from measured cycles -- a fitted law, or a paired
delta between two arms. Neither shape fits a demand on the COMPILER: a law measures the machine, and
a pair measures two emitter settings. Both are silent about whether today's candidate used the
mechanism, which is exactly the hole this family exists to close.

So this analyzer decides an ``EMITS`` claim: the evidence is the candidate's own instruction stream,
the property is
:func:`merlin.analysis.perf.load_state_residency.residency_verdict`'s, and the verdict is about the compiler
rather than about the target.

THE COHORT MUST BE ABLE TO DISCRIMINATE, and that is CHECKED rather than declared -- the discipline
``perf_paired_claim`` applies to its negative control, applied to the whole cohort. A demand whose
every member could be satisfied without the lever is a demand that cannot fail, which this repo has
shipped before (an epilogue the readout never applies; a relu whose clamp was unreachable). So:

* a member whose declared operands need MORE THAN ONE movement configuration is LEVER-BEARING: a
  program holding one configuration must re-establish them in turn, so the property is refutable on
  it;
* a member whose operands all share one configuration is a CONTROL: ONE held configuration suffices,
  so the member is passable without addressing a second state at all.

The control is not a member on which nothing can fail -- a program that reconfigures before every
transfer fails it too -- and claiming otherwise would be the overclaim this file is trying to avoid.
What it IS, exactly, is the member that the DEFECT THIS FAMILY NAMES passes: a nest holding a single
configuration re-establishes nothing here, because its one held configuration is the only one the
shape needs. That is what makes a lever-bearing failure attributable. If a control fails, the program
is redundant even where one configuration would do -- a different and broader defect -- and a
lever-bearing failure beside it cannot be attributed to the pitch alternation, so the cohort is
REFUSED rather than scored.

A cohort missing either kind is REFUSED. Not "passed with a note" -- refused, because the number it
would report is a number about nothing.

THE CLASSIFICATION IS NOT A BOUND. Which members are lever-bearing is read from the declared operand
shapes (the trailing extent of each operand the operation reads is its row pitch under this corpus's
row-major operand layout). It decides only which bucket a member lands in; the pass/fail arithmetic
comes entirely from the trace. So a mis-classification can weaken the cohort's discrimination
report -- which is visible -- but it can never manufacture a failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.analysis.perf import load_state_residency as LSR

ANALYZER = "merlin.analysis.perf.load_state_claim.analyze_load_state_claim/v1"

ESTABLISHED = "ESTABLISHED"
REFUTED = "REFUTED"
REFUSED = "REFUSED"

#: What a member's shapes make refutable. Declared here so a report can name the bucket.
LEVER_BEARING = "lever_bearing"
CONTROL = "control"


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"verdict": REFUSED, "reason": reason, **extra}


def declared_movement_configurations(capsule: Mapping[str, Any]) -> int | None:
    """How many DISTINCT movement configurations this capsule's declared operands require.

    The row pitch of a row-major operand is its trailing extent, so two operands whose trailing
    extents differ cannot be moved under one configuration. Returns ``None`` when the declaration
    does not say -- an operand with no shape, or a capsule that declares no inputs -- so the caller
    refuses to classify rather than assuming one.

    Only operands the operation READS are counted: a committed output leaves through the store path,
    which carries its own configuration and is not what this family is about.
    """
    if not isinstance(capsule, Mapping):
        return None
    inputs = capsule.get("inputs")
    if not isinstance(inputs, Sequence) or not inputs:
        return None
    pitches: set[int] = set()
    for operand in inputs:
        if not isinstance(operand, Mapping):
            return None
        shape = operand.get("shape")
        if not isinstance(shape, Sequence) or not shape:
            return None
        trailing = shape[-1]
        if not isinstance(trailing, int) or isinstance(trailing, bool) or trailing < 1:
            return None
        pitches.add(trailing)
    return len(pitches)


def classify(capsule: Mapping[str, Any]) -> str | None:
    """``LEVER_BEARING`` / ``CONTROL`` / ``None`` when the declaration does not settle it."""
    distinct = declared_movement_configurations(capsule)
    if distinct is None:
        return None
    return LEVER_BEARING if distinct > 1 else CONTROL


SCHEMA_VERSION = 1
READY = "READY"


def _refused(family: str | None, *reasons: str) -> dict[str, Any]:
    """A preflight refusal in the shape the launching stage reads.

    The stage prints ``refusal_reasons``; a refusal that carried the reason under any other key read
    to it as ``unknown``, which is the same as saying nothing.
    """
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


def _buckets(descriptors: Sequence[Any]) -> tuple[dict[str, list[str]], list[str]]:
    buckets: dict[str, list[str]] = {LEVER_BEARING: [], CONTROL: []}
    unclassified: list[str] = []
    for descriptor in descriptors:
        name = str(descriptor.get("name") or "<unnamed>") if isinstance(descriptor, Mapping) else "<unnamed>"
        bucket = classify(descriptor) if isinstance(descriptor, Mapping) else None
        if bucket is None:
            unclassified.append(name)
        else:
            buckets[bucket].append(name)
    return buckets, unclassified


def _lanes(evidence: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The (instrument, tier) pairs this family's own acceptance block declares.

    Read from the declaration rather than named here, so an EMITS family on a target whose structural
    read has a different instrument does not need this module edited. A family declaring no lane has
    nothing to measure and is refused by the caller.
    """
    pairs = [
        (evidence.get("structural_instrument"), evidence.get("structural_tier")),
        (evidence.get("correctness_simulator"), evidence.get("correctness_tier")),
    ]
    return [(str(a), str(b)) for a, b in pairs if isinstance(a, str) and a and isinstance(b, str) and b]


def preflight_load_state_evidence(
    descriptors: object, *, replicates: Sequence[str], target: object = None
) -> dict[str, Any]:
    """The single launch-time precondition: can this cohort decide anything?

    Run BEFORE any observation, so a cohort that could never refute the claim is refused at launch
    rather than after every cell has been paid for. Three things must hold: the members must agree on
    one frozen contract that names its lanes, the cohort must carry both a lever-bearing member and a
    control, and -- when a target is supplied -- that target must publish a load-state selector,
    without which the property is underivable and no verdict is reachable.
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

    buckets, unclassified = _buckets(descriptors)
    if unclassified:
        return _refused(
            family,
            "these members declare operands whose movement configuration count cannot be read, so "
            f"nothing says whether the lever is exercisable on them: {sorted(unclassified)}",
        )
    if not buckets[LEVER_BEARING]:
        return _refused(
            family,
            "no member of this cohort needs more than one movement configuration, so every member "
            "would pass without the lever; a demand that cannot fail is not a demand",
        )
    if not buckets[CONTROL]:
        return _refused(
            family,
            "this cohort carries no control -- no member on which one held configuration suffices -- "
            "so a failure could not be attributed to the alternation this family names",
        )
    selector = None
    if target is not None:
        selector = LSR.load_state_selector(str(target))
        if selector is None:
            return _refused(
                family,
                f"target {str(target)!r} publishes no load-state selector in its extracted "
                "load-configuration layout, so this family's property is underivable on it",
            )
    expected = [
        {
            "family": family,
            "capsule": str(descriptor.get("name")),
            "role": classify(descriptor),
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
        "unresolved_facts": [] if selector else ["load_state_selector"],
        "refusal_reasons": [],
        "selector": dict(selector) if selector else None,
    }


def analyze_load_state_claim(descriptors: object, results: object, per_unit: object = None) -> dict[str, Any]:
    """Decide the family over already-observed traces.

    ``results`` rows carry ``capsule`` and either a ``verdict`` produced by
    :func:`merlin.analysis.perf.load_state_residency.residency_verdict` or the ``trace`` to run it over, plus
    the ``selector`` the observation was made with. Every declared member must report.
    """
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        return _fail("no capsule descriptors were supplied")
    if not isinstance(results, Sequence) or isinstance(results, str) or not results:
        return _fail("no observed rows were supplied")

    families: set[str] = set()
    contracts: list[Any] = []
    buckets: dict[str, str] = {}
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
        name = str(descriptor.get("name") or "<unnamed>")
        bucket = classify(descriptor)
        if bucket is None:
            return _fail(f"member {name!r} declares operands this family cannot classify")
        buckets[name] = bucket
    if len(families) != 1:
        return _fail("descriptors span more than one family", families=sorted(families))
    if any(not isinstance(contract, Mapping) for contract in contracts):
        return _fail("a member carries no frozen acceptance contract")
    if any(contract != contracts[0] for contract in contracts):
        return _fail("the members do not agree on one frozen acceptance contract")
    if LEVER_BEARING not in buckets.values():
        return _fail("no member of this cohort could refute the claim; it would pass without the lever")
    if CONTROL not in buckets.values():
        return _fail("this cohort carries no control, so a failure cannot be attributed to the lever")

    observed: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, Mapping):
            return _fail("an observed row is not a mapping")
        name = str(row.get("capsule") or "")
        if name not in buckets:
            return _fail(f"observed row names {name!r}, which is not a declared member of this family")
        verdict = row.get("verdict")
        if not isinstance(verdict, Mapping):
            trace, selector = row.get("trace"), row.get("selector")
            if trace is None:
                return _fail(f"member {name!r} reports neither a verdict nor a trace to decide one from")
            try:
                verdict = LSR.residency_verdict(trace, selector=selector, capacity=row.get("capacity"))
            except TypeError as exc:
                return _fail(f"member {name!r} reports an unusable trace: {exc}")
        observed[name] = dict(verdict)

    missing = sorted(set(buckets) - set(observed))
    if missing:
        return _fail("these declared members did not report", missing=missing)

    controls_failed = sorted(n for n, b in buckets.items() if b == CONTROL and observed[n].get("verdict") == LSR.FAIL)
    if controls_failed:
        return _fail(
            "a control member failed -- on these the operands share one movement configuration, so "
            "a single held configuration suffices and the defect this family names cannot show. A "
            "program redundant HERE is redundant everywhere, which is a broader defect, and no "
            "lever-bearing failure beside it can be attributed to the pitch alternation. The cohort "
            "is refused rather than scored",
            controls_failed=controls_failed,
            members=observed,
        )
    refused = sorted(n for n in buckets if observed[n].get("verdict") == LSR.REFUSED)
    if refused:
        return _fail(
            "these members could not be decided, so the family is undecided rather than passed",
            undecidable=refused,
            reasons={n: observed[n].get("reason") for n in refused},
            members=observed,
        )
    failed = sorted(n for n, b in buckets.items() if b == LEVER_BEARING and observed[n].get("verdict") == LSR.FAIL)
    if failed:
        return {
            "verdict": REFUTED,
            "reason": (
                "these members re-establish load configurations the load path already holds, so the "
                "compiler is paying configuration work on every block instead of keeping its "
                "movement configurations resident"
            ),
            "failed": failed,
            "excess_commands": {n: observed[n].get("excess") for n in failed},
            "members": observed,
            "cohort": {name: bucket for name, bucket in sorted(buckets.items())},
        }
    return {
        "verdict": ESTABLISHED,
        "reason": (
            "every lever-bearing member issues exactly as many load configurations as it uses "
            "distinct ones, and every control does too. The redundant configuration work is absent; "
            "this is not a claim about the rest of the movement schedule"
        ),
        "members": observed,
        "cohort": {name: bucket for name, bucket in sorted(buckets.items())},
    }
