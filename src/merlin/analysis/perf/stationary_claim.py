"""Decide a stationary-operand residency family from the candidate's OWN emitted traces.

This is the second ``EMITS`` family in the corpus and it follows
:mod:`merlin.analysis.perf.load_state_claim` deliberately: the evidence is the candidate's own instruction
stream, the property is :func:`merlin.analysis.perf.stationary_residency.residency_verdict`'s, and the verdict
is about the COMPILER rather than about the machine. What it adds is a second bucket rule and a
second refusal, both of which exist because this lever's floor is conditional in a way the load
path's was not.

THE COHORT MUST BE ABLE TO DISCRIMINATE, and that is CHECKED rather than declared:

* a member whose declared output extent spans MORE THAN ONE array row block is LEVER-BEARING: the
  nest has positions that differ only in the accumulator tile and the moving operand's window, so a
  compiler naming the block at each of them re-shifts identical bytes, and the property is refutable;
* a member whose output extent fits one row block is a CONTROL: every position presents a different
  block, the floor equals the count at either floor kind, and ANY correct program passes it.

The control is not a member on which nothing can fail -- it would fail a program that named a block
twice for one position. What it IS, exactly, is the member that the DEFECT THIS FAMILY NAMES passes,
which is what makes a lever-bearing failure attributable to the reuse depth rather than to something
broader. If a control fails, the cohort is REFUSED rather than scored.

THE SECOND REFUSAL, and why it is not optional. The property has two floors (see
:mod:`merlin.analysis.perf.stationary_residency`): a run-length floor that needs no reordering, and a
distinct-tile floor that a loop order cannot dodge but that only applies where the target's
accumulator admits the reuse-ordered nest. A lever-bearing member decided at the WEAKER floor was not
asked the question it was minted to ask -- a compiler can pass it by scattering each block's
positions -- so such a member is reported UNDECIDED, not passed. Reading a weak-floor pass as a pass
is precisely how a demand that cannot fail gets shipped, and here it would have been invisible
because the verdict still says PASS.

THE CLASSIFICATION IS NOT A BOUND. Which members are lever-bearing is read from the declared output
extent against the target's own array row count. It decides only which bucket a member lands in; the
pass/fail arithmetic comes entirely from the trace. So a mis-classification can weaken the cohort's
discrimination report -- which is visible -- but it can never manufacture a failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.analysis.perf import stationary_residency as SR

ANALYZER = "merlin.analysis.perf.stationary_claim.analyze_stationary_claim/v1"

ESTABLISHED = "ESTABLISHED"
REFUTED = "REFUTED"
REFUSED = "REFUSED"

#: What a member's shapes make refutable. Declared here so a report can name the bucket.
LEVER_BEARING = "lever_bearing"
CONTROL = "control"

SCHEMA_VERSION = 1
READY = "READY"


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"verdict": REFUSED, "reason": reason, **extra}


def _refused(family: str | None, *reasons: str) -> dict[str, Any]:
    """A preflight refusal in the shape the launching stage reads."""
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


def _geometry(capsule: Mapping[str, Any]) -> Mapping[str, Any] | None:
    performance = capsule.get("performance") if isinstance(capsule, Mapping) else None
    geometry = performance.get("shape_geometry") if isinstance(performance, Mapping) else None
    return geometry if isinstance(geometry, Mapping) else None


def member_row_block(capsule: Mapping[str, Any], supplied: object = None) -> int | None:
    """The array row block a member's extents are counted in, or ``None``.

    Prefers the one the OBSERVER supplied, because that caller read it from the target it is about to
    run on. Falls back to the one the capsule itself records, which is what its extents were minted
    against -- so a caller holding only the declaration (an audit, a launch-time reach check) can
    classify a member without resolving a target, exactly as the load-state family can.
    """
    for value in (supplied, (_geometry(capsule) or {}).get("row_block")):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return int(value)
    return None


def declared_reuse_depth(capsule: Mapping[str, Any], *, row_block: object = None) -> int | None:
    """How many nest positions can present one block in a row, from the declaration alone.

    The stationary block of a tiled contraction is indexed by the reduction and output-column
    extents; the output-ROW extent varies only the accumulator tile and the moving operand's window,
    so it is exactly the number of consecutive positions one block serves. In array row blocks,
    because that is the granularity one command issues at.

    Returns ``None`` when neither the declaration nor the caller says -- never a substituted one,
    because a depth of one is the CONTROL verdict and inventing it would silently move a member into
    the bucket where nothing is demanded.
    """
    block = member_row_block(capsule, row_block)
    rows = (_geometry(capsule) or {}).get("M")
    if block is None or not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        return None
    return -(-rows // block)


def classify(capsule: Mapping[str, Any], *, row_block: object = None) -> str | None:
    """``LEVER_BEARING`` / ``CONTROL`` / ``None`` when the declaration does not settle it."""
    depth = declared_reuse_depth(capsule, row_block=row_block)
    if depth is None:
        return None
    return LEVER_BEARING if depth > 1 else CONTROL


def array_row_block(target: object) -> int | None:
    """The target's own array row count, or ``None`` -- the granularity a reuse depth is counted in."""
    return SR.array_row_block(str(target)) if target is not None else None


def _buckets(descriptors: Sequence[Any], *, row_block: object) -> tuple[dict[str, list[str]], list[str]]:
    buckets: dict[str, list[str]] = {LEVER_BEARING: [], CONTROL: []}
    unclassified: list[str] = []
    for descriptor in descriptors:
        name = str(descriptor.get("name") or "<unnamed>") if isinstance(descriptor, Mapping) else "<unnamed>"
        bucket = classify(descriptor, row_block=row_block) if isinstance(descriptor, Mapping) else None
        if bucket is None:
            unclassified.append(name)
        else:
            buckets[bucket].append(name)
    return buckets, unclassified


def _lanes(evidence: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The (instrument, tier) pairs this family's own acceptance block declares."""
    pairs = [
        (evidence.get("structural_instrument"), evidence.get("structural_tier")),
        (evidence.get("correctness_simulator"), evidence.get("correctness_tier")),
    ]
    return [(str(a), str(b)) for a, b in pairs if isinstance(a, str) and a and isinstance(b, str) and b]


def preflight_stationary_evidence(
    descriptors: object, *, replicates: Sequence[str], target: object = None
) -> dict[str, Any]:
    """The single launch-time precondition: can this cohort decide anything?

    Run BEFORE any observation, so a cohort that could never refute the claim is refused at launch
    rather than after every cell has been paid for. Four things must hold: the members must agree on
    one frozen contract that names its lanes; the cohort must carry both a lever-bearing member and a
    control; the emitted ABI must declare a retain form, without which no re-presentation is
    avoidable; and -- when a target is supplied -- its array row count must be derivable, without
    which no member can be classified at all.
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

    vocabulary = SR.stationary_vocabulary(target)
    if vocabulary is None or not vocabulary.retain_form:
        return _refused(
            family,
            "the emitted ABI declares no retain form -- more than one compute form inheriting one "
            "staging command, one of which consumes resident state -- so every naming command is "
            "compulsory and this family's property is underivable here",
        )
    if target is not None and vocabulary.sentinel is None:
        return _refused(
            family,
            f"the selected support for {str(target)!r} publishes no retain sentinel, so a position "
            "that keeps the array's operand cannot be told from one that names a block",
        )

    row_block = array_row_block(target) if target is not None else None
    if target is not None and row_block is None:
        return _refused(
            family,
            f"target {str(target)!r} does not derive an array row count from its own extracted "
            "geometry, so the reuse depth a member's declared output extent offers is UNKNOWN and "
            "no member can be classified",
        )
    buckets, unclassified = _buckets(descriptors, row_block=row_block)
    if unclassified:
        return _refused(
            family,
            "these members declare an output extent this family cannot read against the array's row "
            f"block, so nothing says whether the lever is exercisable on them: {sorted(unclassified)}",
        )
    if not buckets[LEVER_BEARING]:
        return _refused(
            family,
            "no member of this cohort spans more than one array row block, so every member presents "
            "a different block at every position and would pass without the lever; a demand that "
            "cannot fail is not a demand",
        )
    if not buckets[CONTROL]:
        return _refused(
            family,
            "this cohort carries no control -- no member whose output fits one row block -- so a "
            "failure could not be attributed to the reuse depth this family names",
        )
    expected = [
        {
            "family": family,
            "capsule": str(descriptor.get("name")),
            "role": classify(descriptor, row_block=row_block),
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
        "vocabulary": vocabulary.to_dict(),
    }


def analyze_stationary_claim(descriptors: object, results: object, per_unit: object = None) -> dict[str, Any]:
    """Decide the family over already-observed traces.

    ``results`` rows carry ``capsule``, the ``row_block`` the observation was classified with, and
    either a ``verdict`` produced by :func:`merlin.analysis.perf.stationary_residency.residency_verdict` or
    the ``trace`` and ``target`` to run it over. Every declared member must report.
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

    # The row block travels with the OBSERVATION, as the load-state family's selector does: the
    # caller that made the reading is the one that knew the target. Rows disagreeing about it were
    # not made against one machine, and a cohort spanning two machines is not one cohort.
    # The row block travels with the OBSERVATION when the caller supplied one, as the load-state
    # family's selector does; rows disagreeing about it were not made against one machine, and a
    # cohort spanning two machines is not one cohort. Absent, each member falls back to the
    # granularity it was minted against, which is recorded in its own declaration.
    blocks = {row.get("row_block") for row in results if isinstance(row, Mapping)}
    blocks.discard(None)
    if len(blocks) > 1:
        return _fail(
            "the observed rows disagree about the array row block, so they were not made against "
            "one machine and the reuse depth each member offers cannot be read",
            row_blocks=sorted(str(b) for b in blocks),
        )
    row_block = next(iter(blocks)) if blocks else None

    buckets: dict[str, str] = {}
    for name, descriptor in declared.items():
        bucket = classify(descriptor, row_block=row_block)
        if bucket is None:
            return _fail(f"member {name!r} declares an output extent this family cannot classify")
        buckets[name] = bucket
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
            trace = row.get("trace")
            if trace is None:
                return _fail(f"member {name!r} reports neither a verdict nor a trace to decide one from")
            try:
                verdict = SR.residency_verdict(trace, target=row.get("target"))
            except TypeError as exc:
                return _fail(f"member {name!r} reports an unusable trace: {exc}")
        observed[name] = dict(verdict)

    missing = sorted(set(buckets) - set(observed))
    if missing:
        return _fail("these declared members did not report", missing=missing)

    controls_failed = sorted(n for n, b in buckets.items() if b == CONTROL and observed[n].get("verdict") == SR.FAIL)
    if controls_failed:
        return _fail(
            "a control member failed -- on these the output fits one array row block, so every "
            "position presents a different block and the defect this family names cannot show. A "
            "program redundant HERE is redundant everywhere, which is a broader defect, and no "
            "lever-bearing failure beside it can be attributed to the reuse depth. The cohort is "
            "refused rather than scored",
            controls_failed=controls_failed,
            members=observed,
        )
    refused = sorted(n for n in buckets if observed[n].get("verdict") == SR.REFUSED)
    if refused:
        return _fail(
            "these members could not be decided, so the family is undecided rather than passed",
            undecidable=refused,
            reasons={n: observed[n].get("reason") for n in refused},
            members=observed,
        )
    # A lever-bearing member decided at the RUN-LENGTH floor was not asked the question it was minted
    # to ask: at that floor a compiler passes by scattering each block's positions instead of making
    # them consecutive, which costs exactly as much. Its PASS is therefore not evidence, and reading
    # it as one is how this demand would have become unfailable without anything looking wrong.
    weak = sorted(
        n
        for n, b in buckets.items()
        if b == LEVER_BEARING
        and observed[n].get("verdict") == SR.PASS
        and observed[n].get("floor_kind") != SR.REUSE_ORDERED_FLOOR
    )
    if weak:
        return _fail(
            "these lever-bearing members passed only at the run-length floor, which a loop order "
            "that scatters each block's positions satisfies at identical cost. The machine did not "
            "admit the reuse-ordered floor for them (its accumulator could not hold their committed "
            "output, or its capacity was underivable), so their pass is not evidence that the lever "
            "was used and the family is undecided rather than established",
            weak_floor=weak,
            floors={n: observed[n].get("floor_kind") for n in weak},
            members=observed,
        )
    failed = sorted(n for n, b in buckets.items() if b == LEVER_BEARING and observed[n].get("verdict") == SR.FAIL)
    if failed:
        return {
            "verdict": REFUTED,
            "reason": (
                "these members shift a stationary block through the array that it has already been "
                "given, so the compiler is paying the array fill on positions that differ only in "
                "the accumulator tile and the moving operand's window"
            ),
            "failed": failed,
            "excess_commands": {n: observed[n].get("excess") for n in failed},
            "floors": {n: observed[n].get("floor_kind") for n in failed},
            "members": observed,
            "cohort": {name: bucket for name, bucket in sorted(buckets.items())},
        }
    return {
        "verdict": ESTABLISHED,
        "reason": (
            "every lever-bearing member names a stationary block no more often than the tiles it "
            "presents require, at the floor its machine admits, and every control does too. The "
            "redundant shift-in work is absent; this is not a claim about the rest of the schedule"
        ),
        "members": observed,
        "cohort": {name: bucket for name, bucket in sorted(buckets.items())},
    }
