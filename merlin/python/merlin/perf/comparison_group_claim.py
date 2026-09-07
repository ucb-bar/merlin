"""Decide a target-neutral two-member comparison group from measured whole-program cycles.

Unlike a compiler A/B, the two arms here are two generated workload members (for example, a program
with a host-required island and the structurally matched program without it).  Each member is compiled
and timed independently under the same backend protocol.  The analyzer knows no target, opcode,
simulator, or operation name: the frozen descriptor declares the group field, role names, permitted
semantic difference, evidence lanes, and replicate schedule.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

ANALYZER = "merlin.perf.comparison_group_claim.analyze_comparison_group_claim/v1"
ESTABLISHED, REFUTED, REFUSED = "ESTABLISHED", "REFUTED", "REFUSED"
EITHER = "either"
_BAND = "measured_replicate_dispersion"


class _Refusal(ValueError):
    pass


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _Refusal(f"{label} must be a mapping")
    return value


def _simple_names(value: object, *, count: int | None = None) -> tuple[str, ...]:
    if (not isinstance(value, Sequence) or isinstance(value, str)
            or any(not isinstance(x, str) or not x or x.strip() != x for x in value)):
        raise _Refusal("replicate identities must be a list of simple non-empty names")
    names = tuple(value)
    if len(set(names)) != len(names):
        raise _Refusal("replicate identities are not unique")
    if count is not None and len(names) != count:
        raise _Refusal(f"replicate identities have length {len(names)}, expected {count}")
    return names


def _validated(descriptors: object, offered_replicates: Sequence[str] | None = None) -> dict[str, Any]:
    if (not isinstance(descriptors, Sequence) or isinstance(descriptors, str)
            or not descriptors):
        raise _Refusal("no capsule descriptors were supplied")
    members = [_mapping(row, f"descriptor {i}") for i, row in enumerate(descriptors)]
    names = [row.get("name") for row in members]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise _Refusal("capsule descriptor names must be non-empty and unique")

    performances = [_mapping(row.get("performance"), f"descriptor {row['name']!r} performance")
                    for row in members]
    if any(perf.get("claim") != "DIFFERENTIAL" for perf in performances):
        raise _Refusal("comparison-group analysis accepts DIFFERENTIAL claims only")
    families = {str(perf.get("family") or "") for perf in performances}
    if len(families) != 1 or "" in families:
        raise _Refusal(f"descriptors do not agree on one performance family: {sorted(families)}")
    contracts = [_mapping(perf.get("acceptance"), "the frozen acceptance contract")
                 for perf in performances]
    contract = contracts[0]
    if any(row != contract for row in contracts):
        raise _Refusal("members disagree about the frozen acceptance contract")
    if contract.get("analyzer") != ANALYZER:
        raise _Refusal(f"acceptance names {contract.get('analyzer')!r}, not {ANALYZER!r}")
    if contract.get("schema_version") != 1:
        raise _Refusal("acceptance schema_version must be 1")
    program_arm = contract.get("program_arm")
    if program_arm not in ("baseline", "candidate"):
        raise _Refusal(
            "comparison-group acceptance must name program_arm as 'baseline' or 'candidate'")

    roles_value = contract.get("roles")
    if (not isinstance(roles_value, Sequence) or isinstance(roles_value, str)
            or len(roles_value) != 2 or any(not isinstance(x, str) or not x for x in roles_value)
            or roles_value[0] == roles_value[1]):
        raise _Refusal("acceptance must name exactly two distinct comparison roles")
    roles = tuple(roles_value)
    predicted = contract.get("expected_faster")
    if predicted != EITHER and predicted not in roles:
        raise _Refusal("acceptance predicts neither a declared role nor 'either'")
    group_field = contract.get("group_field")
    if not isinstance(group_field, str) or not group_field:
        raise _Refusal("acceptance.group_field must be a non-empty field name")
    allowed_value = contract.get("allowed_attribute_differences")
    if (not isinstance(allowed_value, Sequence) or isinstance(allowed_value, str)
            or any(not isinstance(x, str) or not x for x in allowed_value)
            or len(set(allowed_value)) != len(allowed_value)):
        raise _Refusal("allowed_attribute_differences must be a unique list of names")
    allowed = set(allowed_value)

    band = _mapping(contract.get("band"), "acceptance.band")
    if band.get("kind") != _BAND or band.get("declared_constant") is not None:
        raise _Refusal("comparison groups require a measured replicate-dispersion band with no constant")
    rep = _mapping(contract.get("replicates"), "acceptance.replicates")
    exact = rep.get("exact_count")
    if isinstance(exact, bool) or not isinstance(exact, int) or exact < 2:
        raise _Refusal("acceptance.replicates.exact_count must be at least two")
    identities = _simple_names(rep.get("identities"), count=exact)
    if offered_replicates is not None and tuple(offered_replicates) != identities:
        raise _Refusal(
            f"the run offers replicates {list(offered_replicates)}, but the frozen contract requires "
            f"{list(identities)}")

    evidence = _mapping(contract.get("evidence"), "acceptance.evidence")
    lanes: list[tuple[str, str]] = []
    for sim_key, tier_key in (("correctness_simulator", "correctness_tier"),
                              ("timing_simulator", "timing_tier")):
        simulator, tier = evidence.get(sim_key), evidence.get(tier_key)
        if not isinstance(simulator, str) or not simulator or not isinstance(tier, str) or not tier:
            raise _Refusal(f"acceptance.evidence omits {sim_key}/{tier_key}")
        lanes.append((simulator, tier))

    groups: dict[str, dict[str, Mapping[str, Any]]] = {}
    for descriptor in members:
        decl = _mapping(descriptor.get(group_field), f"descriptor {descriptor['name']!r}.{group_field}")
        group, role = decl.get("name"), decl.get("role")
        if not isinstance(group, str) or not group or role not in roles:
            raise _Refusal(f"descriptor {descriptor['name']!r} has an invalid comparison group/role")
        if role in groups.setdefault(group, {}):
            raise _Refusal(f"comparison group {group!r} repeats role {role!r}")
        groups[group][str(role)] = descriptor
    incomplete = {group: sorted(set(roles) - set(rows)) for group, rows in groups.items()
                  if set(rows) != set(roles)}
    if incomplete:
        raise _Refusal(f"comparison groups do not contain exactly both roles: {incomplete}")

    # The pair is permitted to differ only where the acceptance says. External tensors, operation,
    # and every unlisted operation attribute must be identical; otherwise the cycle delta prices more
    # than the declared lever.
    for group, by_role in groups.items():
        left, right = by_role[roles[0]], by_role[roles[1]]
        if left.get("inputs") != right.get("inputs"):
            raise _Refusal(f"comparison group {group!r} external inputs differ outside the lever")
        lop = _mapping(left.get("operation"), f"{left['name']}.operation")
        rop = _mapping(right.get("operation"), f"{right['name']}.operation")
        if lop.get("op") != rop.get("op"):
            raise _Refusal(f"comparison group {group!r} operations differ outside the lever")
        la = dict(_mapping(lop.get("attributes"), f"{left['name']}.operation.attributes"))
        ra = dict(_mapping(rop.get("attributes"), f"{right['name']}.operation.attributes"))
        for key in allowed:
            la.pop(key, None); ra.pop(key, None)
        if la != ra:
            raise _Refusal(
                f"comparison group {group!r} operation attributes differ outside the declared "
                f"allowed set {sorted(allowed)}")

    return {"family": next(iter(families)), "members": members, "contract": contract,
            "roles": roles, "predicted": predicted, "groups": groups,
            "identities": identities, "lanes": lanes, "evidence": evidence}


def preflight_comparison_group_claim(descriptors: object, *,
                                     replicates: Sequence[str]) -> dict[str, Any]:
    """Validate the frozen groups and author their exact L2/L3 measurement identities."""
    try:
        resolved = _validated(descriptors, replicates)
    except (_Refusal, KeyError, TypeError, ValueError) as exc:
        return {"schema_version": 1, "family": None, "claim": "DIFFERENTIAL",
                "status": REFUSED, "declaration": None, "cohort": None, "replicates": [],
                "expected_identities": [], "unresolved_facts": [],
                "refusal_reasons": [str(exc)]}
    role_of = {str(row["name"]): str(row[resolved["contract"]["group_field"]]["role"])
               for row in resolved["members"]}
    expected = [
        {"family": resolved["family"], "capsule": str(row["name"]),
         "comparison_role": role_of[str(row["name"])],
         "program_arm": str(resolved["contract"]["program_arm"]), "simulator": simulator,
         "replicate": replicate, "tier": tier}
        for row in resolved["members"]
        for replicate in resolved["identities"]
        for simulator, tier in resolved["lanes"]
    ]
    return {"schema_version": 1, "family": resolved["family"], "claim": "DIFFERENTIAL",
            "status": "READY", "declaration": copy.deepcopy(dict(resolved["contract"])),
            "cohort": {"groups": sorted(resolved["groups"]), "roles": list(resolved["roles"]),
                       "capsules": sorted(str(row["name"]) for row in resolved["members"]),
                       "replicates": list(resolved["identities"]),
                       "evidence_lanes": [{"simulator": s, "tier": t}
                                          for s, t in resolved["lanes"]]},
            "replicates": list(resolved["identities"]), "expected_identities": expected,
            "unresolved_facts": [], "refusal_reasons": []}


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"verdict": REFUSED, "reason": reason, **extra}


def analyze_comparison_group_claim(descriptors: object, results: object) -> dict[str, Any]:
    """Compare each group's two measured members on one program artifact and timing lane."""
    try:
        resolved = _validated(descriptors)
    except (_Refusal, KeyError, TypeError, ValueError) as exc:
        return _fail(str(exc))
    if not isinstance(results, Sequence) or isinstance(results, str) or not results:
        return _fail("no measured rows were supplied")
    timing_sim = str(resolved["evidence"]["timing_simulator"])
    timing_tier = str(resolved["evidence"]["timing_tier"])
    rows = [row for row in results if isinstance(row, Mapping)
            and (row.get("simulator") in (None, timing_sim))
            and (row.get("tier") in (None, timing_tier))]
    if not rows:
        return _fail(f"no results belong to timing lane {timing_sim}/{timing_tier}")

    # A cycle is evidence only for a program that passed its complete grade.  In particular, the
    # whole-program lane contract can fail even when both simulators return exact numerics and a
    # positive counter value (for example, a supposedly mesh-only control that still executes a
    # scalar epilogue).  Treating such a row as timing evidence turns a failed control into an
    # apparently established differential.  ``correct`` is the target-neutral bit published by the
    # execution stage after numeric, simulator, protocol, and lane checks; absence is not success.
    unqualified = [f"{row.get('capsule')}/{row.get('replicate')}"
                   for row in rows if row.get("correct") is not True]
    if unqualified:
        return _fail("timing results lack a passing correctness/contract grade",
                     unqualified=unqualified[:12])

    expected_arm = str(resolved["contract"]["program_arm"])
    program_arms = {str(row.get("program_arm", row.get("arm"))) for row in rows}
    if program_arms != {expected_arm}:
        return _fail(
            f"results use program arms {sorted(program_arms)}, expected exactly {expected_arm!r}")
    identity_key = next((key for key in ("artifact_sha256", "package_sha256", "submission_sha256")
                         if any(row.get(key) is not None for row in rows)), None)
    if identity_key is None:
        return _fail("results carry no artifact/package/submission digest, so one program is not proven")
    artifacts = {str(row.get(identity_key)) for row in rows if row.get(identity_key) is not None}
    if len(artifacts) != 1 or any(row.get(identity_key) is None for row in rows):
        return _fail(f"results mix or omit {identity_key} artifact identity", artifacts=sorted(artifacts))

    by_capsule: dict[str, dict[str, float]] = {}
    allowed_capsules = {str(row["name"]) for row in resolved["members"]}
    for row in rows:
        capsule, replicate, cycles = row.get("capsule"), row.get("replicate"), row.get("cycles")
        if capsule not in allowed_capsules:
            return _fail(f"result names undeclared capsule {capsule!r}")
        if replicate not in resolved["identities"]:
            return _fail(f"result names undeclared replicate {replicate!r}")
        if isinstance(cycles, bool) or not isinstance(cycles, (int, float)) or cycles <= 0:
            return _fail(f"{capsule}/{replicate} carries no positive cycle count")
        slot = by_capsule.setdefault(str(capsule), {})
        if str(replicate) in slot:
            return _fail(f"duplicate timing result for {capsule}/{replicate}")
        slot[str(replicate)] = float(cycles)
    missing = [f"{capsule}/{replicate}" for capsule in sorted(allowed_capsules)
               for replicate in resolved["identities"]
               if replicate not in by_capsule.get(capsule, {})]
    if missing:
        return _fail("the timing cohort is incomplete", missing=missing[:12])

    roles = resolved["roles"]
    predicted = resolved["predicted"]
    verdict_rows, losers = [], []
    for group, members in sorted(resolved["groups"].items()):
        names = {role: str(members[role]["name"]) for role in roles}
        values = {role: list(by_capsule[names[role]].values()) for role in roles}
        representative = {role: min(values[role]) for role in roles}
        band = max(max(v) - min(v) for v in values.values())
        if predicted == EITHER:
            delta = abs(representative[roles[0]] - representative[roles[1]])
            passed = delta > band
        else:
            other = roles[1] if predicted == roles[0] else roles[0]
            delta = representative[other] - representative[predicted]
            passed = delta > band
        verdict_rows.append({"group": group, "members": names, **representative,
                             "delta_cycles": delta, "replicate_band": band})
        if not passed:
            losers.append(group)
    if losers:
        return {"verdict": REFUTED, "rows": verdict_rows, "groups": losers,
                "reason": f"the predicted group direction failed beyond its band in {len(losers)} group(s)"}
    return {"verdict": ESTABLISHED, "rows": verdict_rows,
            "reason": f"all {len(verdict_rows)} comparison groups separate in the predicted direction"}
