"""Decide a form-perf member against its vendor-reference bar from measured cycles.

A form-perf member (Phase 0's model-shaped per-form capsules) is measured twice: the CANDIDATE arm
is the compiler under optimization, held to the experiment's declared ``prohibited_instruction_roles``;
the VENDOR_REFERENCE arm is the target support package's own implementation of the identical form,
which may use every instruction role the target has, because it is the bar. The claim is that the
candidate beats the bar on this form, and the evidence is the pair of measurements.

Target-neutral: no opcode, simulator or operation is known here. The frozen contract names the arms,
the replicate schedule and the evidence lanes; everything else is read off the measured rows. Guards,
each a way the ratio could otherwise be satisfied by accident:

* every timing row needs a passing correctness grade -- a fast wrong program is not a measurement;
* each arm is ONE program artifact (a digest per arm, not a mix);
* a candidate row must carry its whole-program instruction-policy scan, and it must be clean. A row
  with no scan is refused: an unscanned candidate could be the vendor's own instruction sequence;
* the band is the sum of both arms' measured replicate dispersions, and a win inside it is REFUTED.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

ANALYZER = "perf_vendor_reference_claim.analyze_vendor_reference_claim/v1"
ESTABLISHED, REFUTED, REFUSED = "ESTABLISHED", "REFUTED", "REFUSED"
CANDIDATE, VENDOR = "candidate", "vendor_reference"
_BAND = "measured_replicate_dispersion"


class _Refusal(ValueError):
    pass


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _Refusal(f"{label} must be a mapping")
    return value


def _validated(descriptors: object, offered_replicates: Sequence[str] | None = None) -> dict[str, Any]:
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        raise _Refusal("no capsule descriptors were supplied")
    members = [_mapping(row, f"descriptor {index}") for index, row in enumerate(descriptors)]
    names = [row.get("name") for row in members]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise _Refusal("capsule descriptor names must be non-empty and unique")
    performances = [_mapping(row.get("performance"), f"{row['name']}.performance") for row in members]
    if any(p.get("claim") != "DIFFERENTIAL" for p in performances):
        raise _Refusal("vendor-reference analysis accepts DIFFERENTIAL claims only")
    families = {str(p.get("family") or "") for p in performances}
    if len(families) != 1 or "" in families:
        raise _Refusal(f"descriptors do not agree on one performance family: {sorted(families)}")
    contracts = [_mapping(p.get("acceptance"), "the frozen acceptance contract") for p in performances]
    contract = contracts[0]
    if any(row != contract for row in contracts):
        raise _Refusal("members disagree about the frozen acceptance contract")
    if contract.get("analyzer") != ANALYZER or contract.get("schema_version") != 1:
        raise _Refusal(
            f"acceptance names {contract.get('analyzer')!r} v{contract.get('schema_version')}, not {ANALYZER!r}"
        )
    for performance, row in zip(performances, members):
        arms = _mapping(performance.get("arms"), f"{row['name']}.performance.arms")
        if set(arms) != {CANDIDATE, VENDOR}:
            raise _Refusal(f"{row['name']} must declare exactly a {CANDIDATE!r} and a {VENDOR!r} arm")
        if _mapping(arms[CANDIDATE], "candidate arm").get("instruction_policy") in (None, "", "unrestricted"):
            raise _Refusal(f"{row['name']}: the candidate arm must be held to a declared instruction policy")
    band = _mapping(contract.get("band"), "acceptance.band")
    if band.get("kind") != _BAND or band.get("declared_constant") is not None:
        raise _Refusal("vendor-reference claims require a measured replicate-dispersion band with no constant")
    replicates = _mapping(contract.get("replicates"), "acceptance.replicates")
    exact = replicates.get("exact_count")
    identities = tuple(str(v) for v in (replicates.get("identities") or ()))
    if isinstance(exact, bool) or not isinstance(exact, int) or exact < 2 or len(identities) != exact:
        raise _Refusal("acceptance.replicates must name at least two exact identities")
    if len(set(identities)) != len(identities):
        raise _Refusal("acceptance.replicates.identities must be unique")
    if offered_replicates is not None and tuple(offered_replicates) != identities:
        raise _Refusal(
            f"the run offers replicates {list(offered_replicates)}, the contract requires {list(identities)}"
        )
    evidence = _mapping(contract.get("evidence"), "acceptance.evidence")
    lanes = []
    for simulator_key, tier_key in (("correctness_simulator", "correctness_tier"), ("timing_simulator", "timing_tier")):
        simulator, tier = evidence.get(simulator_key), evidence.get(tier_key)
        if not isinstance(simulator, str) or not simulator or not isinstance(tier, str) or not tier:
            raise _Refusal(f"acceptance.evidence omits {simulator_key}/{tier_key}")
        lanes.append((simulator, tier))
    return {
        "family": next(iter(families)),
        "members": members,
        "contract": contract,
        "identities": identities,
        "lanes": lanes,
        "evidence": evidence,
    }


def preflight_vendor_reference_claim(descriptors: object, *, replicates: Sequence[str]) -> dict[str, Any]:
    """Validate the frozen members and author one measurement identity per member x arm x replicate x lane."""
    try:
        resolved = _validated(descriptors, replicates)
    except (_Refusal, KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": 1,
            "family": None,
            "claim": "DIFFERENTIAL",
            "status": REFUSED,
            "declaration": None,
            "cohort": None,
            "replicates": [],
            "expected_identities": [],
            "unresolved_facts": [],
            "refusal_reasons": [str(exc)],
        }
    expected = [
        {
            "family": resolved["family"],
            "capsule": str(row["name"]),
            "program_arm": arm,
            "simulator": simulator,
            "replicate": replicate,
            "tier": tier,
        }
        for row in resolved["members"]
        for arm in (CANDIDATE, VENDOR)
        for replicate in resolved["identities"]
        for simulator, tier in resolved["lanes"]
    ]
    return {
        "schema_version": 1,
        "family": resolved["family"],
        "claim": "DIFFERENTIAL",
        "status": "READY",
        "declaration": copy.deepcopy(dict(resolved["contract"])),
        "cohort": {
            "capsules": sorted(str(row["name"]) for row in resolved["members"]),
            "arms": [CANDIDATE, VENDOR],
            "replicates": list(resolved["identities"]),
            "evidence_lanes": [{"simulator": s, "tier": t} for s, t in resolved["lanes"]],
        },
        "replicates": list(resolved["identities"]),
        "expected_identities": expected,
        "unresolved_facts": [],
        "refusal_reasons": [],
    }


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"verdict": REFUSED, "reason": reason, **extra}


def analyze_vendor_reference_claim(descriptors: object, results: object) -> dict[str, Any]:
    """Per member: candidate cycles over vendor-reference cycles, decided against the summed band."""
    try:
        resolved = _validated(descriptors)
    except (_Refusal, KeyError, TypeError, ValueError) as exc:
        return _fail(str(exc))
    if not isinstance(results, Sequence) or isinstance(results, str) or not results:
        return _fail("no measured rows were supplied")
    simulator, tier = str(resolved["evidence"]["timing_simulator"]), str(resolved["evidence"]["timing_tier"])
    rows = [
        row
        for row in results
        if isinstance(row, Mapping) and row.get("simulator") in (None, simulator) and row.get("tier") in (None, tier)
    ]
    if not rows:
        return _fail(f"no results belong to timing lane {simulator}/{tier}")
    unqualified = [f"{row.get('capsule')}/{row.get('program_arm')}" for row in rows if row.get("correct") is not True]
    if unqualified:
        return _fail("timing results lack a passing correctness grade", unqualified=unqualified[:12])
    unscanned = [
        f"{row.get('capsule')}/{row.get('replicate')}"
        for row in rows
        if row.get("program_arm") == CANDIDATE and (row.get("instruction_policy") or {}).get("status") != "clean"
    ]
    if unscanned:
        return _fail("candidate rows lack a clean whole-program instruction-policy scan", candidates=unscanned[:12])
    cycles: dict[tuple[str, str], dict[str, float]] = {}
    artifacts: dict[str, set[str]] = {CANDIDATE: set(), VENDOR: set()}
    allowed = {str(row["name"]) for row in resolved["members"]}
    for row in rows:
        arm, capsule, replicate, value = (
            row.get("program_arm"),
            row.get("capsule"),
            row.get("replicate"),
            row.get("cycles"),
        )
        if arm not in (CANDIDATE, VENDOR):
            return _fail(f"result names undeclared arm {arm!r}")
        if capsule not in allowed or replicate not in resolved["identities"]:
            return _fail(f"result names undeclared capsule/replicate {capsule!r}/{replicate!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return _fail(f"{capsule}/{arm}/{replicate} carries no positive cycle count")
        digest = row.get("artifact_sha256")
        if not isinstance(digest, str) or not digest:
            return _fail(f"{capsule}/{arm}/{replicate} carries no artifact digest")
        artifacts[arm].add(digest)
        slot = cycles.setdefault((str(capsule), str(arm)), {})
        if str(replicate) in slot:
            return _fail(f"duplicate timing result for {capsule}/{arm}/{replicate}")
        slot[str(replicate)] = float(value)
    if len(artifacts[CANDIDATE]) != 1:
        return _fail("the candidate arm is not one program artifact", artifacts=sorted(artifacts[CANDIDATE]))
    missing = [
        f"{capsule}/{arm}/{replicate}"
        for capsule in sorted(allowed)
        for arm in (CANDIDATE, VENDOR)
        for replicate in resolved["identities"]
        if replicate not in cycles.get((capsule, arm), {})
    ]
    if missing:
        return _fail("the timing cohort is incomplete", missing=missing[:12])
    verdict_rows, above = [], []
    for capsule in sorted(allowed):
        ours, bar = list(cycles[(capsule, CANDIDATE)].values()), list(cycles[(capsule, VENDOR)].values())
        band = (max(ours) - min(ours)) + (max(bar) - min(bar))
        delta = min(bar) - min(ours)
        verdict_rows.append(
            {
                "capsule": capsule,
                "candidate_cycles": min(ours),
                "vendor_reference_cycles": min(bar),
                "ours_over_vendor": min(ours) / min(bar),
                "delta_cycles": delta,
                "replicate_band": band,
            }
        )
        if not delta > band:
            above.append(capsule)
    if above:
        return {
            "verdict": REFUTED,
            "rows": verdict_rows,
            "capsules": above,
            "reason": f"{len(above)} form(s) do not beat their vendor bar beyond the summed replicate band",
        }
    return {
        "verdict": ESTABLISHED,
        "rows": verdict_rows,
        "reason": f"all {len(verdict_rows)} form(s) beat their vendor bar",
    }
