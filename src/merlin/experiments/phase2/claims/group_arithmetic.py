"""Decide a whole-against-the-sum-of-its-parts claim from measured whole-program cycles.

The comparison-group analyzer next door decides exactly two members against each other. A fusion
claim is not that shape: one fused program is compared against the ARITHMETIC SUM of the several
programs it replaces, so the group holds one ``whole`` member and N ``part`` members and the
comparand is ``sum_of_parts``. There was no analyzer for that shape, so every capsule declaring it
shipped with no ``acceptance`` block at all -- and a cohort that declares no analyzer is refused by
``phase2.claims.dispatch``, which is why the whole family sat out of the campaign rather than producing
a wrong answer. This module is that missing procedure.

Target-neutral like its neighbours: no opcode, simulator, operation name or shape is known here. The
frozen contract names the group field, the two role names, how many parts a group must contain, the
descriptor paths every member of a group must agree on, the evidence lanes and the replicate
schedule. Everything else is read off the measured rows.

Three structural guards, because the arithmetic here is otherwise trivially satisfiable:

* a group must hold EXACTLY one whole and exactly the declared number of parts -- a part that failed
  to build would otherwise shrink the sum and hand the whole an unearned win;
* the whole and every part must name DISTINCT operations -- three copies of one program would
  "establish" the claim by counting the same cycles once against twice;
* every path in ``demand_equal`` must be PRESENT on every member and equal within the group. An
  absent path is a refusal, never an assumed match, because the paths are what hold the members to
  one problem size and one datatype; without them the sum prices a different computation.

The band is the sum of the members' own measured replicate dispersions, not the largest of them: the
compared quantity is a sum, so its uncertainty is too. A win inside that band is REFUTED.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

#: The frozen identity the reviewed PF contract names. Kept as the historical spelling so capsules
#: sealed under it are decided by this implementation; ``dispatch`` maps its module to this owner.
ANALYZER = "merlin.perf.group_arithmetic_claim.analyze_group_arithmetic_claim/v1"
ESTABLISHED, REFUTED, REFUSED = "ESTABLISHED", "REFUTED", "REFUSED"
WHOLE, PARTS_SUM, EITHER = "whole", "parts_sum", "either"
_BAND = "measured_replicate_dispersion"


class _Refusal(ValueError):
    pass


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _Refusal(f"{label} must be a mapping")
    return value


def _simple_names(value: object, *, count: int | None = None) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or any(not isinstance(item, str) or not item or item.strip() != item for item in value)
    ):
        raise _Refusal("replicate identities must be a list of simple non-empty names")
    names = tuple(value)
    if len(set(names)) != len(names):
        raise _Refusal("replicate identities are not unique")
    if count is not None and len(names) != count:
        raise _Refusal(f"replicate identities have length {len(names)}, expected {count}")
    return names


_ABSENT = object()


def _at_path(descriptor: Mapping[str, Any], path: str) -> Any:
    """Read one dotted descriptor path, structurally. Absence is a distinct value, never None.

    ``None`` is a legitimate declared value in these contracts, so "the path is missing" and "the
    path says null" must not collapse: the first is a refusal and the second is data.
    """
    cursor: Any = descriptor
    for segment in path.split("."):
        if not isinstance(cursor, Mapping) or segment not in cursor:
            return _ABSENT
        cursor = cursor[segment]
    return cursor


def _operand_index(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    rows = descriptor.get("inputs")
    if not isinstance(rows, Sequence) or isinstance(rows, str) or not rows:
        raise _Refusal(f"{descriptor['name']} declares no inputs")
    index: dict[str, Any] = {}
    for row in rows:
        operand = _mapping(row, f"{descriptor['name']}.inputs entry")
        name = operand.get("name")
        if not isinstance(name, str) or not name:
            raise _Refusal(f"{descriptor['name']} declares an unnamed input")
        if name in index:
            raise _Refusal(f"{descriptor['name']} declares input {name!r} twice")
        index[name] = {key: operand.get(key) for key in ("role", "shape", "dtype")}
    return index


def _require_shared_operands(group: str, whole: Mapping[str, Any], parts: Sequence[Mapping[str, Any]]) -> None:
    """Hold the parts to the whole's problem, using the operands they have in common.

    A decomposition's parts do not share ALL their operands with the whole -- the intermediate one
    part produces and the next consumes is internal to the whole and appears in neither -- so a
    subset rule would refuse a correct group. What must hold is weaker and still load-bearing: every
    part touches at least one of the whole's declared operands, and every operand two members share
    by name is declared identically (role, shape, dtype). That is what stops a group assembled from
    programs built at different sizes or datatypes, where the summed cycles would price a different
    computation than the whole and the comparison would mean nothing.
    """
    whole_operands = _operand_index(whole)
    for part in parts:
        part_operands = _operand_index(part)
        shared = sorted(set(whole_operands) & set(part_operands))
        if not shared:
            raise _Refusal(
                f"group {group!r} part {part['name']!r} shares no declared operand with "
                f"{whole['name']!r}, so nothing holds the two to one problem"
            )
        for name in shared:
            if whole_operands[name] != part_operands[name]:
                raise _Refusal(
                    f"group {group!r} declares operand {name!r} differently in {whole['name']!r} and "
                    f"{part['name']!r}: {whole_operands[name]} vs {part_operands[name]}"
                )


def _validated(descriptors: object, offered_replicates: Sequence[str] | None = None) -> dict[str, Any]:
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        raise _Refusal("no capsule descriptors were supplied")
    members = [_mapping(row, f"descriptor {index}") for index, row in enumerate(descriptors)]
    names = [row.get("name") for row in members]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise _Refusal("capsule descriptor names must be non-empty and unique")

    performances = [_mapping(row.get("performance"), f"descriptor {row['name']!r} performance") for row in members]
    if any(performance.get("claim") != "DIFFERENTIAL" for performance in performances):
        raise _Refusal("group-arithmetic analysis accepts DIFFERENTIAL claims only")
    families = {str(performance.get("family") or "") for performance in performances}
    if len(families) != 1 or "" in families:
        raise _Refusal(f"descriptors do not agree on one performance family: {sorted(families)}")

    contracts = [
        _mapping(performance.get("acceptance"), "the frozen acceptance contract") for performance in performances
    ]
    contract = contracts[0]
    if any(row != contract for row in contracts):
        raise _Refusal("members disagree about the frozen acceptance contract")
    if contract.get("analyzer") != ANALYZER:
        raise _Refusal(f"acceptance names {contract.get('analyzer')!r}, not {ANALYZER!r}")
    if contract.get("schema_version") != 1:
        raise _Refusal("acceptance schema_version must be 1")
    program_arm = contract.get("program_arm")
    if program_arm not in ("baseline", "candidate"):
        raise _Refusal("group-arithmetic acceptance must name program_arm as 'baseline' or 'candidate'")

    whole_role, part_role = contract.get("whole_role"), contract.get("part_role")
    if (
        not isinstance(whole_role, str)
        or not whole_role
        or not isinstance(part_role, str)
        or not part_role
        or whole_role == part_role
    ):
        raise _Refusal("acceptance must name two distinct roles as whole_role and part_role")
    parts_per_group = contract.get("parts_per_group")
    if isinstance(parts_per_group, bool) or not isinstance(parts_per_group, int) or parts_per_group < 2:
        raise _Refusal("acceptance.parts_per_group must be an integer of at least two")
    predicted = contract.get("expected_lower")
    if predicted not in (WHOLE, PARTS_SUM, EITHER):
        raise _Refusal(f"acceptance.expected_lower must be one of {WHOLE!r}, {PARTS_SUM!r}, {EITHER!r}")

    group_field = contract.get("group_field")
    if not isinstance(group_field, str) or not group_field:
        raise _Refusal("acceptance.group_field must be a non-empty field name")
    demanded_value = contract.get("demand_equal")
    if (
        not isinstance(demanded_value, Sequence)
        or isinstance(demanded_value, str)
        or not demanded_value
        or any(not isinstance(item, str) or not item for item in demanded_value)
        or len(set(demanded_value)) != len(demanded_value)
    ):
        raise _Refusal("acceptance.demand_equal must be a non-empty unique list of descriptor paths")
    demanded = tuple(demanded_value)

    band = _mapping(contract.get("band"), "acceptance.band")
    if band.get("kind") != _BAND or band.get("declared_constant") is not None:
        raise _Refusal("group-arithmetic claims require a measured replicate-dispersion band with no constant")
    replicates = _mapping(contract.get("replicates"), "acceptance.replicates")
    exact = replicates.get("exact_count")
    if isinstance(exact, bool) or not isinstance(exact, int) or exact < 2:
        raise _Refusal("acceptance.replicates.exact_count must be at least two")
    identities = _simple_names(replicates.get("identities"), count=exact)
    if offered_replicates is not None and tuple(offered_replicates) != identities:
        raise _Refusal(
            f"the run offers replicates {list(offered_replicates)}, but the frozen contract requires {list(identities)}"
        )

    evidence = _mapping(contract.get("evidence"), "acceptance.evidence")
    lanes: list[tuple[str, str]] = []
    for simulator_key, tier_key in (("correctness_simulator", "correctness_tier"), ("timing_simulator", "timing_tier")):
        simulator, tier = evidence.get(simulator_key), evidence.get(tier_key)
        if not isinstance(simulator, str) or not simulator or not isinstance(tier, str) or not tier:
            raise _Refusal(f"acceptance.evidence omits {simulator_key}/{tier_key}")
        lanes.append((simulator, tier))

    groups: dict[str, dict[str, Any]] = {}
    for descriptor in members:
        declaration = _mapping(descriptor.get(group_field), f"descriptor {descriptor['name']!r}.{group_field}")
        group, role = declaration.get("name"), declaration.get("role")
        if not isinstance(group, str) or not group or role not in (whole_role, part_role):
            raise _Refusal(f"descriptor {descriptor['name']!r} has an invalid group name/role")
        slot = groups.setdefault(group, {"whole": None, "parts": []})
        if role == whole_role:
            if slot["whole"] is not None:
                raise _Refusal(f"group {group!r} declares more than one {whole_role!r} member")
            slot["whole"] = descriptor
        else:
            slot["parts"].append(descriptor)

    for group, slot in sorted(groups.items()):
        if slot["whole"] is None:
            raise _Refusal(f"group {group!r} has no {whole_role!r} member, so there is nothing to compare")
        if len(slot["parts"]) != parts_per_group:
            raise _Refusal(
                f"group {group!r} holds {len(slot['parts'])} {part_role!r} member(s), and the contract "
                f"demands exactly {parts_per_group}; a missing part would shrink the sum"
            )
        slot["parts"] = sorted(slot["parts"], key=lambda row: str(row["name"]))
        cohort = [slot["whole"], *slot["parts"]]
        operations = []
        for descriptor in cohort:
            operation = _mapping(descriptor.get("operation"), f"{descriptor['name']}.operation")
            op = operation.get("op")
            if not isinstance(op, str) or not op:
                raise _Refusal(f"{descriptor['name']} declares no operation name")
            operations.append(op)
        if len(set(operations)) != len(operations):
            raise _Refusal(
                f"group {group!r} repeats an operation across its members {operations}; a whole "
                "compared against copies of itself would establish the claim by arithmetic alone"
            )
        for path in demanded:
            values = [_at_path(descriptor, path) for descriptor in cohort]
            absent = [str(descriptor["name"]) for descriptor, value in zip(cohort, values) if value is _ABSENT]
            if absent:
                raise _Refusal(f"group {group!r} members {absent} do not declare demanded path {path!r}")
            if any(value != values[0] for value in values[1:]):
                raise _Refusal(f"group {group!r} members disagree on demanded path {path!r}: {values}")
        _require_shared_operands(group, slot["whole"], slot["parts"])

    return {
        "family": next(iter(families)),
        "members": members,
        "contract": contract,
        "whole_role": whole_role,
        "part_role": part_role,
        "predicted": predicted,
        "groups": groups,
        "identities": identities,
        "lanes": lanes,
        "evidence": evidence,
        "group_field": group_field,
    }


def preflight_group_arithmetic_claim(descriptors: object, *, replicates: Sequence[str]) -> dict[str, Any]:
    """Validate the frozen groups and author their exact correctness/timing measurement identities."""
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
    group_field = resolved["group_field"]
    role_of = {str(row["name"]): str(row[group_field]["role"]) for row in resolved["members"]}
    expected = [
        {
            "family": resolved["family"],
            "capsule": str(row["name"]),
            "comparison_role": role_of[str(row["name"])],
            "program_arm": str(resolved["contract"]["program_arm"]),
            "simulator": simulator,
            "replicate": replicate,
            "tier": tier,
        }
        for row in resolved["members"]
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
            "groups": sorted(resolved["groups"]),
            "roles": [resolved["whole_role"], resolved["part_role"]],
            "parts_per_group": int(resolved["contract"]["parts_per_group"]),
            "capsules": sorted(str(row["name"]) for row in resolved["members"]),
            "replicates": list(resolved["identities"]),
            "evidence_lanes": [{"simulator": simulator, "tier": tier} for simulator, tier in resolved["lanes"]],
        },
        "replicates": list(resolved["identities"]),
        "expected_identities": expected,
        "unresolved_facts": [],
        "refusal_reasons": [],
    }


def _fail(reason: str, **extra: Any) -> dict[str, Any]:
    return {"verdict": REFUSED, "reason": reason, **extra}


def analyze_group_arithmetic_claim(descriptors: object, results: object) -> dict[str, Any]:
    """Compare each group's whole against the summed parts on one program artifact and timing lane."""
    try:
        resolved = _validated(descriptors)
    except (_Refusal, KeyError, TypeError, ValueError) as exc:
        return _fail(str(exc))
    if not isinstance(results, Sequence) or isinstance(results, str) or not results:
        return _fail("no measured rows were supplied")
    timing_simulator = str(resolved["evidence"]["timing_simulator"])
    timing_tier = str(resolved["evidence"]["timing_tier"])
    rows = [
        row
        for row in results
        if isinstance(row, Mapping)
        and (row.get("simulator") in (None, timing_simulator))
        and (row.get("tier") in (None, timing_tier))
    ]
    if not rows:
        return _fail(f"no results belong to timing lane {timing_simulator}/{timing_tier}")

    # A cycle is evidence only for a program that passed its complete grade. Absence of the bit is
    # not success -- the same rule the comparison-group analyzer applies, and for the same reason.
    unqualified = [f"{row.get('capsule')}/{row.get('replicate')}" for row in rows if row.get("correct") is not True]
    if unqualified:
        return _fail("timing results lack a passing correctness/contract grade", unqualified=unqualified[:12])

    expected_arm = str(resolved["contract"]["program_arm"])
    program_arms = {str(row.get("program_arm", row.get("arm"))) for row in rows}
    if program_arms != {expected_arm}:
        return _fail(f"results use program arms {sorted(program_arms)}, expected exactly {expected_arm!r}")
    identity_key = next(
        (
            key
            for key in ("artifact_sha256", "package_sha256", "submission_sha256")
            if any(row.get(key) is not None for row in rows)
        ),
        None,
    )
    if identity_key is None:
        return _fail("results carry no artifact/package/submission digest, so one program is not proven")
    artifacts = {str(row.get(identity_key)) for row in rows if row.get(identity_key) is not None}
    if len(artifacts) != 1 or any(row.get(identity_key) is None for row in rows):
        return _fail(f"results mix or omit {identity_key} artifact identity", artifacts=sorted(artifacts))

    by_capsule: dict[str, dict[str, float]] = {}
    allowed = {str(row["name"]) for row in resolved["members"]}
    for row in rows:
        capsule, replicate, cycles = row.get("capsule"), row.get("replicate"), row.get("cycles")
        if capsule not in allowed:
            return _fail(f"result names undeclared capsule {capsule!r}")
        if replicate not in resolved["identities"]:
            return _fail(f"result names undeclared replicate {replicate!r}")
        if isinstance(cycles, bool) or not isinstance(cycles, (int, float)) or cycles <= 0:
            return _fail(f"{capsule}/{replicate} carries no positive cycle count")
        slot = by_capsule.setdefault(str(capsule), {})
        if str(replicate) in slot:
            return _fail(f"duplicate timing result for {capsule}/{replicate}")
        slot[str(replicate)] = float(cycles)
    missing = [
        f"{capsule}/{replicate}"
        for capsule in sorted(allowed)
        for replicate in resolved["identities"]
        if replicate not in by_capsule.get(capsule, {})
    ]
    if missing:
        return _fail("the timing cohort is incomplete", missing=missing[:12])

    predicted = resolved["predicted"]
    verdict_rows, losers = [], []
    for group, slot in sorted(resolved["groups"].items()):
        whole_name = str(slot["whole"]["name"])
        part_names = [str(row["name"]) for row in slot["parts"]]
        whole_values = list(by_capsule[whole_name].values())
        part_values = [list(by_capsule[name].values()) for name in part_names]
        whole_cycles = min(whole_values)
        parts_cycles = sum(min(values) for values in part_values)
        # THE SUM'S UNCERTAINTY IS THE SUM OF THE UNCERTAINTIES. Taking the largest single member's
        # dispersion, as a pairwise comparison legitimately does, would understate the band of a
        # quantity built by adding several measurements together.
        band = (max(whole_values) - min(whole_values)) + sum(max(v) - min(v) for v in part_values)
        if predicted == EITHER:
            delta = abs(parts_cycles - whole_cycles)
        elif predicted == WHOLE:
            delta = parts_cycles - whole_cycles
        else:
            delta = whole_cycles - parts_cycles
        passed = delta > band
        verdict_rows.append(
            {
                "group": group,
                "whole": whole_name,
                "parts": part_names,
                "whole_cycles": whole_cycles,
                "parts_sum_cycles": parts_cycles,
                "delta_cycles": delta,
                "replicate_band": band,
                "expected_lower": predicted,
            }
        )
        if not passed:
            losers.append(group)
    if losers:
        return {
            "verdict": REFUTED,
            "rows": verdict_rows,
            "groups": losers,
            "reason": (f"the predicted direction failed beyond the summed replicate band in {len(losers)} group(s)"),
        }
    return {
        "verdict": ESTABLISHED,
        "rows": verdict_rows,
        "reason": f"all {len(verdict_rows)} group(s) separate from their summed parts in the predicted direction",
    }
