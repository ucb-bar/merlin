"""Decide a target-neutral two-member comparison group from measured whole-program cycles.

Unlike a compiler A/B, the two arms here are two generated workload members (for example, a program
with a host-required island and the structurally matched program without it).  Each member is compiled
and timed independently under the same backend protocol.  The analyzer knows no target, opcode,
simulator, or operation name: the frozen descriptor declares the group field, role names, permitted
semantic difference, evidence lanes, and replicate schedule.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# ---- comparison_group_claim --------------------------------------------------------------------

ANALYZER = "merlin.analysis.perf.claims.comparison_group.analyze_comparison_group_claim/v1"
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
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or any(not isinstance(x, str) or not x or x.strip() != x for x in value)
    ):
        raise _Refusal("replicate identities must be a list of simple non-empty names")
    names = tuple(value)
    if len(set(names)) != len(names):
        raise _Refusal("replicate identities are not unique")
    if count is not None and len(names) != count:
        raise _Refusal(f"replicate identities have length {len(names)}, expected {count}")
    return names


def _validated(descriptors: object, offered_replicates: Sequence[str] | None = None) -> dict[str, Any]:
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        raise _Refusal("no capsule descriptors were supplied")
    members = [_mapping(row, f"descriptor {i}") for i, row in enumerate(descriptors)]
    names = [row.get("name") for row in members]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise _Refusal("capsule descriptor names must be non-empty and unique")

    performances = [_mapping(row.get("performance"), f"descriptor {row['name']!r} performance") for row in members]
    if any(perf.get("claim") != "DIFFERENTIAL" for perf in performances):
        raise _Refusal("comparison-group analysis accepts DIFFERENTIAL claims only")
    families = {str(perf.get("family") or "") for perf in performances}
    if len(families) != 1 or "" in families:
        raise _Refusal(f"descriptors do not agree on one performance family: {sorted(families)}")
    contracts = [_mapping(perf.get("acceptance"), "the frozen acceptance contract") for perf in performances]
    contract = contracts[0]
    if any(row != contract for row in contracts):
        raise _Refusal("members disagree about the frozen acceptance contract")
    if contract.get("analyzer") != ANALYZER:
        raise _Refusal(f"acceptance names {contract.get('analyzer')!r}, not {ANALYZER!r}")
    if contract.get("schema_version") != 1:
        raise _Refusal("acceptance schema_version must be 1")
    program_arm = contract.get("program_arm")
    if program_arm not in ("baseline", "candidate"):
        raise _Refusal("comparison-group acceptance must name program_arm as 'baseline' or 'candidate'")

    roles_value = contract.get("roles")
    if (
        not isinstance(roles_value, Sequence)
        or isinstance(roles_value, str)
        or len(roles_value) != 2
        or any(not isinstance(x, str) or not x for x in roles_value)
        or roles_value[0] == roles_value[1]
    ):
        raise _Refusal("acceptance must name exactly two distinct comparison roles")
    roles = tuple(roles_value)
    predicted = contract.get("expected_faster")
    if predicted != EITHER and predicted not in roles:
        raise _Refusal("acceptance predicts neither a declared role nor 'either'")
    group_field = contract.get("group_field")
    if not isinstance(group_field, str) or not group_field:
        raise _Refusal("acceptance.group_field must be a non-empty field name")
    allowed_value = contract.get("allowed_attribute_differences")
    if (
        not isinstance(allowed_value, Sequence)
        or isinstance(allowed_value, str)
        or any(not isinstance(x, str) or not x for x in allowed_value)
        or len(set(allowed_value)) != len(allowed_value)
    ):
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
            f"the run offers replicates {list(offered_replicates)}, but the frozen contract requires {list(identities)}"
        )

    evidence = _mapping(contract.get("evidence"), "acceptance.evidence")
    lanes: list[tuple[str, str]] = []
    for sim_key, tier_key in (("correctness_simulator", "correctness_tier"), ("timing_simulator", "timing_tier")):
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
    incomplete = {group: sorted(set(roles) - set(rows)) for group, rows in groups.items() if set(rows) != set(roles)}
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
            la.pop(key, None)
            ra.pop(key, None)
        if la != ra:
            raise _Refusal(
                f"comparison group {group!r} operation attributes differ outside the declared "
                f"allowed set {sorted(allowed)}"
            )

    return {
        "family": next(iter(families)),
        "members": members,
        "contract": contract,
        "roles": roles,
        "predicted": predicted,
        "groups": groups,
        "identities": identities,
        "lanes": lanes,
        "evidence": evidence,
    }


def preflight_comparison_group_claim(descriptors: object, *, replicates: Sequence[str]) -> dict[str, Any]:
    """Validate the frozen groups and author their exact L2/L3 measurement identities."""
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
    role_of = {str(row["name"]): str(row[resolved["contract"]["group_field"]]["role"]) for row in resolved["members"]}
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
            "roles": list(resolved["roles"]),
            "capsules": sorted(str(row["name"]) for row in resolved["members"]),
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
    rows = [
        row
        for row in results
        if isinstance(row, Mapping)
        and (row.get("simulator") in (None, timing_sim))
        and (row.get("tier") in (None, timing_tier))
    ]
    if not rows:
        return _fail(f"no results belong to timing lane {timing_sim}/{timing_tier}")

    # A cycle is evidence only for a program that passed its complete grade.  In particular, the
    # whole-program lane contract can fail even when both simulators return exact numerics and a
    # positive counter value (for example, a supposedly mesh-only control that still executes a
    # scalar epilogue).  Treating such a row as timing evidence turns a failed control into an
    # apparently established differential.  ``correct`` is the target-neutral bit published by the
    # execution stage after numeric, simulator, protocol, and lane checks; absence is not success.
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
    missing = [
        f"{capsule}/{replicate}"
        for capsule in sorted(allowed_capsules)
        for replicate in resolved["identities"]
        if replicate not in by_capsule.get(capsule, {})
    ]
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
        verdict_rows.append(
            {"group": group, "members": names, **representative, "delta_cycles": delta, "replicate_band": band}
        )
        if not passed:
            losers.append(group)
    if losers:
        return {
            "verdict": REFUTED,
            "rows": verdict_rows,
            "groups": losers,
            "reason": f"the predicted group direction failed beyond its band in {len(losers)} group(s)",
        }
    return {
        "verdict": ESTABLISHED,
        "rows": verdict_rows,
        "reason": f"all {len(verdict_rows)} comparison groups separate in the predicted direction",
    }


# ---- barrier_arms ------------------------------------------------------------------------------
# Two arms of one kernel that differ ONLY in how many completion barriers the compiler inserted.
#
# The sibling scheduling family reorders a command stream (:mod:`merlin.analysis.perf.command_stream_gen`) on the
# theory that issue order is the lever a hardware-interlocked, command-driven accelerator gives a
# compiler.  MEASURED on this repo's command-driven target, it is not: a legal permutation of the same
# multiset cost **exactly zero** cycles at two depths, with the reorder verified to reach the ELF and the
# counter verified to move with work.  Both results have the same cause -- the dispatch queue tracks the
# dependence itself -- and read together they say where the lever actually is: not in what order the
# commands issue, but in **how many barriers the compiler inserted that the hardware did not need**.
#
# There are two ways to obtain the pair, and this module carries both because they answer different
# questions and neither subsumes the other.
#
# BUILD THE PAIR (:func:`pair_from_emitter`).  Ask the TARGET'S OWN emitter for the same kernel twice, at
# two settings of its retire knob, and then PROVE the two sources are the same program plus barriers
# rather than asserting it:
#
# * the low-barrier arm's line sequence must be a SUBSEQUENCE of the high-barrier arm's -- so every
#   instruction of one appears, in order, in the other, and nothing was added, dropped or reworded;
# * every line the high-barrier arm adds must be the SAME line -- one repeated statement, which is what
#   makes "the lever is barrier count" true by construction rather than by reading.
#
# This is the arm-construction path, and it is what the synchronization family's declared regime asks
# for: one shape per pair, and the pair is two emissions of that ONE capsule.  ⚠️ NOTHING HERE KNOWS HOW
# A BARRIER IS SPELLED.  The barrier statement is *discovered* as the line the two arms differ by, so a
# target whose emitter spells its retire differently is served without an edit, and a target whose two
# settings differ by something OTHER than a repeated inserted line is REFUSED with its reason rather
# than measured as if the difference were a barrier.  Refusing is the point: a pair that differs by more
# than the lever prices the difference and calls it the lever, which is how a neighbouring family came
# to compare an ~82-cycle lever against an ~280-cycle uncancelled term.
#
# READ A PAIR THAT ALREADY EXISTS (:func:`count_barriers`, :func:`paired_removal`).  A tuning campaign
# does not emit both arms itself: the expensive arm is what the baseline compiler already produced and
# the cheap arm is what the candidate produced instead.  Those two command buffers are already in hand,
# so the job is to count what each one synchronizes.  Counting is structural.  A completion point is an
# ABI opcode in the emitted command buffer, taken from the buffer's own declared vocabulary -- not a
# regex over a listing, and not a target ISA constant: the same abstract vocabulary appears for any
# backend that speaks this ABI.  A buffer that declares no completion opcode yields UNKNOWN with a
# reason rather than a count of zero, because "no barriers found" and "cannot see barriers" must never
# read the same.
#
# ⚠️ A ZERO-BARRIER PAIR IS A RESULT, NOT A FAILURE.  A kernel with one job has no redundant barrier to
# remove, so both settings emit the identical program and ``removed`` is 0.  That member is this family's
# NEGATIVE CONTROL: its measured differential must be exactly zero, and it is the only member for which
# that is true by construction.
#
# Either way the verdict is the family's own falsifier (:func:`analyze_barrier_claim`): the saving must
# GROW with the number of barriers removed.  A single paired point cannot show that, so a cohort of one
# is REFUSED rather than scored -- one subtraction is an anecdote, and the pathology this family exists
# for is precisely that the cost is per barrier.

# --- Building the pair: two emissions of one capsule through the target's own emitter ---

#: The emitter did not accept the retire knob at all, so this target cannot express the lever.
REFUSED_KNOB_UNSUPPORTED = "knob_unsupported"
#: The high-barrier arm is not the low-barrier arm plus inserted lines: something was reworded.
REFUSED_NOT_A_PURE_INSERTION = "not_a_pure_insertion"
#: The inserted lines are not all the same statement, so "the difference" is not one repeated barrier.
REFUSED_HETEROGENEOUS_INSERTION = "heterogeneous_insertion"


class RetireArmsError(RuntimeError):
    """A pair that would not be the same work with a different barrier count. Carries its ``reason``."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class BarrierPair:
    """One kernel emitted at two retire settings, with the difference established rather than assumed."""

    #: Source at the setting that emits the fewest barriers.
    minimal: str
    #: Source at the setting that emits one barrier per unit of work.
    maximal: str
    #: The two settings, in the order (minimal, maximal) they were requested.
    settings: tuple[str, str]
    #: The line the two arms differ by, taken from the diff rather than from any target vocabulary.
    #: ``None`` when the two arms are the identical program (the negative-control member).
    barrier_statement: str | None
    #: How many barriers the lever removes. Zero is a legitimate result, not an error.
    removed: int

    def to_dict(self) -> dict:
        return {
            "settings": list(self.settings),
            "barrier_statement": self.barrier_statement,
            "removed": self.removed,
            "identical_programs": self.minimal == self.maximal,
        }


def _inserted_lines(low: Sequence[str], high: Sequence[str]) -> list[str]:
    """The lines ``high`` has and ``low`` does not, or raise when ``low`` is not a subsequence.

    A two-pointer walk, not a diff library and not a pattern: the question is exactly "does every line
    of the short source appear, in order, in the long one", and answering it any other way would admit
    a pair whose shared lines had been reordered.
    """
    extra: list[str] = []
    i = 0
    for line in high:
        if i < len(low) and low[i] == line:
            i += 1
        else:
            extra.append(line)
    if i != len(low):
        raise RetireArmsError(
            REFUSED_NOT_A_PURE_INSERTION,
            f"the low-barrier arm's line {i + 1} ({low[i]!r}) has no match in emission order in the "
            "high-barrier arm; the two settings changed the kernel, not only its barrier count",
        )
    return extra


def pair_from_emitter(
    emit: Callable[..., str], command_buffer: dict, *, settings: tuple[str, str], knob: str = "retire"
) -> BarrierPair:
    """Emit ``command_buffer`` twice through ``emit`` and return the established barrier pair.

    ``emit`` is the target's OWN driver emitter, called as ``emit(command_buffer, **{knob: setting})``.
    The knob name is a parameter because it belongs to the emitter's signature, not to this module; a
    target whose emitter does not accept it is refused with :data:`REFUSED_KNOB_UNSUPPORTED` rather than
    silently measured at one setting twice.
    """
    if len(settings) != 2 or settings[0] == settings[1]:
        raise ValueError(f"settings must be two DISTINCT emitter settings, got {settings!r}")
    sources = []
    for setting in settings:
        try:
            sources.append(emit(command_buffer, **{knob: setting}))
        except TypeError as exc:
            raise RetireArmsError(
                REFUSED_KNOB_UNSUPPORTED,
                f"this target's emitter does not accept {knob}={setting!r} ({exc}); it cannot express "
                "the barrier lever, so no pair exists to measure",
            ) from exc
    low_src, high_src = sources
    extra = _inserted_lines(low_src.split("\n"), high_src.split("\n"))
    distinct = set(extra)
    if len(distinct) > 1:
        raise RetireArmsError(
            REFUSED_HETEROGENEOUS_INSERTION,
            f"the high-barrier arm adds {len(distinct)} distinct statements {sorted(distinct)!r}; the "
            "difference between the arms is then not one repeated barrier and pricing it as one would "
            "attribute the rest of the difference to the lever",
        )
    return BarrierPair(
        minimal=low_src,
        maximal=high_src,
        settings=(settings[0], settings[1]),
        barrier_statement=(extra[0] if extra else None),
        removed=len(extra),
    )


# --- Reading a pair the measurement path already built, and deciding the claim ---

#: Opcodes that force the issuing host to observe that prior work retired.  Named at the ABI level,
#: which is target-independent: a backend that speaks this ABI emits these regardless of its ISA.
COMPLETION_OPCODES = ("COMMIT", "FENCE", "BARRIER")

#: The falsifier ``observation`` a family declares when its claim is PER-UNIT GROWTH rather than a
#: direction.  A direction claim ("this arm is cheaper") and a growth claim ("the saving grows with
#: the count removed") are different assertions and only the second one indicts a PER-UNIT cost, so a
#: family declaring this must have :func:`analyze_barrier_claim` run over its cohort -- deciding it
#: with a direction test alone would report a verdict on a claim nobody evaluated.  Compared as data
#: against the declaration, never matched as a pattern.
PER_UNIT_GROWTH_OBSERVATION = "paired_cycle_saving_by_removed_barrier_count"

ESTABLISHED = "ESTABLISHED"
REFUTED = "REFUTED"
REFUSED = "REFUSED"
UNKNOWN = "UNKNOWN"


def _commands(buffer: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = buffer.get("commands")
    return [r for r in rows if isinstance(r, Mapping)] if isinstance(rows, Sequence) else []


def count_barriers(buffer: object) -> dict[str, Any]:
    """Count completion points in one emitted command buffer, or say why it cannot be counted."""
    if not isinstance(buffer, Mapping):
        return {"status": UNKNOWN, "reason": "the command buffer is not a mapping"}
    rows = _commands(buffer)
    if not rows:
        return {"status": UNKNOWN, "reason": "the command buffer declares no commands"}
    opcodes = [str(r.get("opcode") or "") for r in rows]
    if any(not code for code in opcodes):
        return {"status": UNKNOWN, "reason": "a command declares no opcode, so the stream is unreadable"}
    present = sorted({code for code in opcodes if code in COMPLETION_OPCODES})
    if not present:
        # Distinguishing this from zero is the whole point: a program with no recognised completion
        # opcode has not been shown to synchronize rarely, only to be unreadable by this counter.
        return {
            "status": UNKNOWN,
            "reason": (
                "no completion opcode from the ABI vocabulary appears in this program, "
                "so its synchronization cannot be counted"
            ),
            "vocabulary": list(COMPLETION_OPCODES),
            "observed_opcodes": sorted(set(opcodes)),
        }
    return {
        "status": "counted",
        "barriers": sum(1 for c in opcodes if c in COMPLETION_OPCODES),
        "commands": len(rows),
        "completion_opcodes": present,
    }


def paired_removal(baseline_buffer: object, candidate_buffer: object) -> dict[str, Any]:
    """How many completion points the candidate removed relative to the baseline."""
    base, cand = count_barriers(baseline_buffer), count_barriers(candidate_buffer)
    if base["status"] != "counted" or cand["status"] != "counted":
        return {
            "status": UNKNOWN,
            "reason": f"baseline: {base.get('reason', 'ok')}; candidate: {cand.get('reason', 'ok')}",
        }
    return {
        "status": "counted",
        "baseline_barriers": base["barriers"],
        "candidate_barriers": cand["barriers"],
        "removed": base["barriers"] - cand["barriers"],
    }


def analyze_barrier_claim(points: object) -> dict[str, Any]:
    """Decide whether cycles saved grow with barriers removed.

    ``points`` is a sequence of already-measured pairs, each carrying ``removed`` (from
    :func:`paired_removal`) and ``cycles_saved``.  The claim is directional and per-unit: more
    barriers removed must buy more cycles.  A cohort that never varies the removed count cannot
    show that and is REFUSED, not scored.
    """
    if not isinstance(points, Sequence) or not points:
        return {"verdict": REFUSED, "reason": "no measured pairs were supplied"}
    usable: list[tuple[float, float]] = []
    for row in points:
        if not isinstance(row, Mapping):
            return {"verdict": REFUSED, "reason": "a measured pair is not a mapping"}
        removed, saved = row.get("removed"), row.get("cycles_saved")
        if not isinstance(removed, int) or isinstance(removed, bool):
            return {"verdict": REFUSED, "reason": "a pair does not declare how many barriers it removed"}
        if not isinstance(saved, (int, float)) or isinstance(saved, bool):
            return {"verdict": REFUSED, "reason": "a pair carries no measured cycle saving"}
        usable.append((float(removed), float(saved)))
    if len({r for r, _ in usable}) < 2:
        return {
            "verdict": REFUSED,
            "reason": (
                "every pair removed the same number of barriers, so the cohort cannot "
                "show a saving that GROWS with the removed count -- which is the claim"
            ),
        }
    n = len(usable)
    sx = sum(r for r, _ in usable)
    sy = sum(s for _, s in usable)
    sxx = sum(r * r for r, _ in usable)
    sxy = sum(r * s for r, s in usable)
    denominator = n * sxx - sx * sx
    if denominator == 0:
        return {"verdict": REFUSED, "reason": "the removed-barrier count does not vary"}
    per_barrier = (n * sxy - sx * sy) / denominator
    measured = {
        "cycles_per_removed_barrier": per_barrier,
        "n_pairs": n,
        "distinct_removed_counts": sorted({r for r, _ in usable}),
    }
    if per_barrier <= 0:
        return {
            "verdict": REFUTED,
            "measured": measured,
            "reason": (
                f"removing barriers bought {per_barrier:.3g} cycles each, so the "
                f"synchronization this family blames is not costing what it claims"
            ),
        }
    return {
        "verdict": ESTABLISHED,
        "measured": measured,
        "reason": f"each removed barrier is worth {per_barrier:.1f} cycles across {n} pairs",
    }


__all__ = [
    "BarrierPair",
    "COMPLETION_OPCODES",
    "ESTABLISHED",
    "PER_UNIT_GROWTH_OBSERVATION",
    "REFUSED",
    "REFUSED_HETEROGENEOUS_INSERTION",
    "REFUSED_KNOB_UNSUPPORTED",
    "REFUSED_NOT_A_PURE_INSERTION",
    "REFUTED",
    "RetireArmsError",
    "UNKNOWN",
    "analyze_barrier_claim",
    "count_barriers",
    "pair_from_emitter",
    "paired_removal",
]
