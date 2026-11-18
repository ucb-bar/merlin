"""Explicit region routes, completeness and factual host-refusal admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.analysis.optimization.candidates import gate_phase

# ---- route_admission ---------------------------------------------------------------------------
# Judge explicit region routes and the factual support for host refusals.

SCHEMA = "route_quality_v1"

#: The two gates, as they are spelled in ``src/merlin/resources/contracts/gate_phases.yaml``.
GATE_DECLARATION = "route_declaration_quality"
GATE_HOST_COMPUTE = "route_host_compute"

STATUS_OK = "ok"
#: A DECIDED finding. What blocks once the gate's declared phase is ``fail``.
STATUS_REPORTED = "reported"
#: Ran and could not decide. Never a pass, at either phase.
STATUS_INCOMPLETE = gate_phase.STATUS_INCOMPLETE

#: The stated refusal is contradicted by the target's own derived capability.
VERDICT_CONTRADICTED = "contradicted"
#: The derived capability positively excludes the work; the refusal stands.
VERDICT_CONFIRMED = "confirmed"
#: No rung decides this family on this target. Reported as such, never as either of the above.
VERDICT_UNKNOWN = "UNKNOWN"

#: A host route with no reason at all. ``route_partition`` requires a reason of a route-``D``
#: DECLINE; nothing required one of a route-``H`` placement, and a placement without one is a
#: refusal nobody can act on.
HOST_ROUTE_WITHOUT_REASON = "host_route_without_reason"

#: Where the family-to-evidence map lives. Contract data, because which derived field adjudicates a
#: family is a statement about the ABI's family vocabulary that a reviewer must be able to diff.
_CONTRACT = ("contract", "mlir_oot_backend_contract.yaml")

#: The program plan's own spelling for a task that runs on the host. Every other ``kind`` the plan
#: emits names a device task, so the split is read from the plan rather than from a target's lane
#: vocabulary.
_HOST_TASK_KIND = "host"

#: LLVM-dialect operations whose result is a computed VALUE rather than an address or a control
#: decision. ``comparison`` is deliberately excluded: a predicate over tensor data is how a
#: legal relu-free clamp is spelled on the host, and including it would make a bound check a
#: finding. Categories come from :mod:`merlin.analysis.programs.cfg_activity`, which owns the family split.
_COMPUTE_CATEGORIES = frozenset({"integer_arithmetic", "floating_arithmetic", "conversion"})


# --------------------------------------------------------------------------------------------
# Tier 1 — the declaration check
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RefusalVerdict:
    """One region routed to a host lane, adjudicated against the target's derived capability."""

    region: str
    family: str
    op: str
    lane: str
    reason: str | None
    verdict: str
    #: What decided it, in words a reader can act on without reopening the facet.
    why: str
    #: Which evidence rung spoke, or ``None`` when none did.
    rung: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "family": self.family,
            "op": self.op,
            "lane": self.lane,
            "reason": self.reason,
            "verdict": self.verdict,
            "why": self.why,
            "rung": self.rung,
        }


@dataclass(frozen=True)
class DeclarationReport:
    gate: str
    status: str
    verdicts: tuple[RefusalVerdict, ...]
    #: Why the check could not decide, when anything could not be. Empty otherwise.
    causes: tuple[str, ...] = ()

    @property
    def contradicted(self) -> tuple[RefusalVerdict, ...]:
        return tuple(v for v in self.verdicts if v.verdict == VERDICT_CONTRADICTED)

    @property
    def confirmed(self) -> tuple[RefusalVerdict, ...]:
        return tuple(v for v in self.verdicts if v.verdict == VERDICT_CONFIRMED)

    @property
    def unknown(self) -> tuple[RefusalVerdict, ...]:
        return tuple(v for v in self.verdicts if v.verdict == VERDICT_UNKNOWN)

    def blocks(self, phase: str | None = None) -> bool:
        """Does this report block, at its declared phase (or one supplied for a what-if)?"""
        resolved = gate_phase.configured_phase(self.gate) if phase is None else phase
        return gate_phase.blocks(resolved, self.status, failing=(STATUS_REPORTED,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "gate": self.gate,
            "status": self.status,
            "causes": list(self.causes),
            "counts": {
                VERDICT_CONTRADICTED: len(self.contradicted),
                VERDICT_CONFIRMED: len(self.confirmed),
                VERDICT_UNKNOWN: len(self.unknown),
            },
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def host_refusal_evidence(doc: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], str | None]:
    """The contract's family-to-evidence map, or ``({}, reason)`` when it is not readable.

    An unreadable contract makes every family UNKNOWN rather than assumed: a family whose
    adjudicating rung is guessed would produce a verdict nobody declared.
    """
    if doc is None:
        from merlin.common.paths import data_path

        path = data_path(*_CONTRACT)
        if not path.is_file():
            return {}, f"the backend ABI contract is not readable at {path.name}"
        try:
            import yaml

            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 -- an unparseable contract is UNKNOWN, not a default
            return {}, f"the backend ABI contract could not be parsed ({type(exc).__name__})"
    rows = ((doc.get("admission") or {}).get("route_quality") or {}).get("host_refusal_evidence")
    if not isinstance(rows, Mapping) or not rows:
        return {}, "the backend ABI contract declares no admission.route_quality.host_refusal_evidence"
    return {str(k): v for k, v in rows.items() if isinstance(v, Mapping)}, None


def _host_routed(params: Mapping[str, Any]) -> tuple[Any, str | None]:
    """``(predicate(region, lane) -> bool, cause)``: is this region routed to a host lane?

    Read from the PROGRAM's own records in the order they are authoritative: its explicit host
    region list, else the complement of its accelerator region list. Nothing here spells a lane or a
    target; a lane vocabulary belongs to the compiler that emitted the buffer, and a buffer that
    declares neither list makes the question undecidable rather than answered by a guess.
    """

    def _names(key: str) -> set[str] | None:
        value = params.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return {str(n) for n in value}
        return None

    host = _names("host_lane_regions")
    if host is not None:
        return (lambda region, lane: region in host), None
    unit = _names("mesh_regions")
    if unit is not None:
        return (lambda region, lane: region not in unit), None
    return None, (
        "the buffer declares neither params.host_lane_regions nor params.mesh_regions, so which "
        "regions it routed to a host lane is not decidable from it"
    )


def _applied_stages(facets: Sequence[Any]) -> tuple[frozenset[str], tuple[str, ...]]:
    """Every epilogue stage any declared readout of any facet applies, plus its evidence lines."""
    stages: set[str] = set()
    evidence: list[str] = []
    for facet in facets:
        for readout in getattr(facet, "readouts", ()) or ():
            applies = [str(s) for s in (readout.get("applies") or ())]
            stages.update(applies)
            if applies:
                evidence.append(
                    f"readout {readout.get('selector')!r} applies {applies}"
                    + (f" ({readout.get('evidence')})" if readout.get("evidence") else "")
                )
    return frozenset(stages), tuple(evidence)


def _capability_stages(facets: Sequence[Any]) -> tuple[frozenset[str], tuple[str, ...]]:
    """Stages the fully derived epilogue capability licenses, and the gaps that stopped it.

    The strongest rung. Every facet that cannot license a capability contributes its gap message
    instead, so a reader sees WHY the strong rung was silent rather than only that it was.
    """
    from merlin.target.capabilities import readout_derivation as _merlin_readout_derivation_1
    from merlin.target.capabilities import readout_facet as rf

    stages: set[str] = set()
    gaps: list[str] = []
    for facet in facets:
        try:
            capability = _merlin_readout_derivation_1.epilogue_capability(facet)
        except Exception as exc:  # noqa: BLE001 -- an underivable capability is a silent rung
            gaps.append(str(exc))
            continue
        for template in capability.ordered_stage_templates:
            stages.update(str(s) for s in template)
    return frozenset(stages), tuple(gaps)


def _adjudicate(
    spec: Mapping[str, Any] | None,
    *,
    facets: Sequence[Any],
    applied: frozenset[str],
    applied_evidence: tuple[str, ...],
    licensed: frozenset[str],
    capability_gaps: tuple[str, ...],
    family: str,
) -> tuple[str, str, str | None]:
    """``(verdict, why, rung)`` for one host-routed region."""
    if spec is None:
        return (
            VERDICT_UNKNOWN,
            f"the backend ABI contract names no evidence source for family {family!r}, so no rung "
            f"adjudicates this refusal; declare one under admission.route_quality.host_refusal_evidence",
            None,
        )

    # A field the facet could not derive makes the verdict UNKNOWN before any rung is consulted:
    # a confirmation built on an underived field would report a hardware refusal nobody established.
    required = [str(f) for f in (spec.get("requires_derived") or ())]
    missing = []
    for name in required:
        why = None
        for facet in facets:
            if getattr(facet, name, None) is not None:
                why = None
                break
            why = (facet.unknown or {}).get(name, "not derived")
        if why is not None:
            missing.append(f"{name} ({why})")
    if missing:
        return (
            VERDICT_UNKNOWN,
            f"family {family!r} is adjudicated by {spec.get('facet_field') or spec.get('epilogue_stage')!r}, "
            f"which needs " + "; ".join(missing),
            None,
        )

    stage = spec.get("epilogue_stage")
    if stage is not None:
        stage = str(stage)
        if stage in licensed:
            return (
                VERDICT_CONTRADICTED,
                f"the derived epilogue capability licenses the {stage!r} stage, so the target has a "
                f"datapath for family {family!r} on its readout path",
                "epilogue_capability",
            )
        if stage in applied:
            return (
                VERDICT_CONTRADICTED,
                f"the target's own declared readouts apply the {stage!r} stage "
                f"({'; '.join(applied_evidence)}), so the stated refusal of family {family!r} is "
                f"contradicted by the target",
                "backend_declared",
            )
        if applied_evidence:
            return (
                VERDICT_CONFIRMED,
                f"every readout the target declares was censused and none applies the {stage!r} "
                f"stage ({'; '.join(applied_evidence)})",
                "backend_declared",
            )
        return (
            VERDICT_UNKNOWN,
            "no readout census is derived for this target"
            + (
                f"; the epilogue capability is unlicensed because {'; '.join(capability_gaps)}"
                if capability_gaps
                else ""
            )
            + f", so nothing decides whether family {family!r} has a {stage!r} datapath",
            None,
        )

    facet_field = spec.get("facet_field")
    if facet_field is not None:
        name = str(facet_field)
        absent_field = str(spec.get("facet_absent_field") or "")
        for facet in facets:
            if getattr(facet, name, None) is not None:
                return (
                    VERDICT_CONTRADICTED,
                    f"the target's readout facet derives {name!r} "
                    f"({getattr(facet, name)!r}), which licenses family {family!r} on the unit",
                    "readout_facet",
                )
        absent = [str(getattr(f, absent_field)) for f in facets if absent_field and getattr(f, absent_field, None)]
        if absent:
            return (
                VERDICT_CONFIRMED,
                f"every facet was derived and none holds {name!r}: " + "; ".join(absent),
                "readout_facet",
            )
        return (
            VERDICT_UNKNOWN,
            f"no facet derives {name!r} and none says why it is absent, so nothing decides family {family!r}",
            None,
        )

    return (
        VERDICT_UNKNOWN,
        f"the contract's evidence row for family {family!r} names neither an epilogue stage nor a "
        f"readout-facet field, so it adjudicates nothing",
        None,
    )


def declaration_quality(
    command_buffer: Mapping[str, Any],
    *,
    facets: Sequence[Any],
    evidence: Mapping[str, Any] | None = None,
    evidence_cause: str | None = None,
) -> DeclarationReport:
    """Adjudicate every host-routed region's stated reason against the derived capability.

    ``facets`` are the target's :class:`~merlin.target.capabilities.readout_facet.ReadoutFacet` objects. They
    are passed in rather than loaded so the caller owns the (slow, filesystem-touching) derivation
    and so a test can state the capability it means to check.
    """
    if evidence is None:
        evidence, evidence_cause = host_refusal_evidence()
    params = command_buffer.get("params")
    params = params if isinstance(params, Mapping) else {}
    placement = params.get("lane_placement")
    causes: list[str] = []
    if evidence_cause:
        causes.append(evidence_cause)
    if not isinstance(placement, Sequence) or isinstance(placement, (str, bytes)):
        causes.append(
            "params.lane_placement is absent or is not a sequence of mappings, so no region's route "
            "is readable and no refusal can be adjudicated"
        )
        return DeclarationReport(GATE_DECLARATION, STATUS_INCOMPLETE, (), tuple(causes))

    is_host, cause = _host_routed(params)
    if is_host is None:
        causes.append(str(cause))
        return DeclarationReport(GATE_DECLARATION, STATUS_INCOMPLETE, (), tuple(causes))

    applied, applied_evidence = _applied_stages(facets)
    licensed, capability_gaps = _capability_stages(facets)

    verdicts: list[RefusalVerdict] = []
    for row in placement:
        if not isinstance(row, Mapping):
            causes.append("a params.lane_placement entry is not a mapping and was not adjudicated")
            continue
        lane = str(row.get("lane") or "")
        family = str(row.get("family") or "")
        region = str(row.get("region") or "")
        if not is_host(region, lane):
            continue  # routed A; tier 2 is what looks inside it
        op = str(row.get("op") or "")
        raw_reason = row.get("reason")
        reason = str(raw_reason) if raw_reason else None
        if reason is None:
            verdicts.append(
                RefusalVerdict(
                    region,
                    family,
                    op,
                    lane,
                    None,
                    VERDICT_CONTRADICTED,
                    f"the region is placed on host lane {lane!r} and states no reason "
                    f"({HOST_ROUTE_WITHOUT_REASON}); a host route is a refusal, and a refusal "
                    f"nobody can read is one nobody can act on",
                    None,
                )
            )
            continue
        verdict, why, rung = _adjudicate(
            evidence.get(family),
            facets=facets,
            applied=applied,
            applied_evidence=applied_evidence,
            licensed=licensed,
            capability_gaps=capability_gaps,
            family=family,
        )
        verdicts.append(RefusalVerdict(region, family, op, lane, reason, verdict, why, rung))

    decided = [v for v in verdicts if v.verdict == VERDICT_CONTRADICTED]
    undecided = [v for v in verdicts if v.verdict == VERDICT_UNKNOWN]
    if decided:
        status = STATUS_REPORTED
    elif undecided or causes:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_OK
    if undecided:
        causes.append(f"{len(undecided)} host-routed region(s) have no rung that decides their family")
    return DeclarationReport(GATE_DECLARATION, status, tuple(verdicts), tuple(causes))


def declaration_quality_for_target(command_buffer: Mapping[str, Any], target: str) -> DeclarationReport:
    """:func:`declaration_quality` with the facets derived for ``target``.

    ``target`` is a PARAMETER. Nothing here knows which one; an undeivable facet set leaves the
    report ``incomplete`` with the derivation failure as its cause.
    """
    from merlin.target.capabilities import readout_derivation as _merlin_readout_derivation_2
    from merlin.target.capabilities import readout_facet as rf

    try:
        facets = _merlin_readout_derivation_2.for_target(target)
    except Exception as exc:  # noqa: BLE001 -- an underivable target is incomplete, never ok
        return DeclarationReport(
            GATE_DECLARATION,
            STATUS_INCOMPLETE,
            (),
            (f"no readout facet could be derived for target {target!r}: {type(exc).__name__}: {exc}",),
        )
    return declaration_quality(command_buffer, facets=facets)


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def format_declaration_report(report: DeclarationReport, label: str = "") -> str:
    out = [f"== route declaration quality {label}".rstrip(), f"status: {report.status}"]
    by_family: dict[tuple[str, str], int] = {}
    for verdict in report.verdicts:
        by_family[(verdict.family, verdict.verdict)] = by_family.get((verdict.family, verdict.verdict), 0) + 1
    out.append(f"{'family/verdict':40} {'regions':>8}")
    for (family, verdict), n in sorted(by_family.items(), key=lambda kv: -kv[1]):
        out.append(f"{family + '/' + verdict:40} {n:8d}")
    for verdict in report.contradicted[:3]:
        out.append(f"  {verdict.region}: stated {verdict.reason!r}")
        out.append(f"    -> {verdict.why}")
    for cause in report.causes:
        out.append(f"  incomplete: {cause}")
    return "\n".join(out)


# ---- route_partition ---------------------------------------------------------------------------
# The admission partition, checked on a produced command buffer.
#
# WHAT THIS IS FOR
# ----------------
# A compiler handed a program must end up in exactly one of three routes, and it must SAY which:
#
# * ``A`` — accepted: target work was emitted;
# * ``H`` — an explicit host or composed route, whose seam is checked elsewhere;
# * ``D`` — an explicit decline: no target command and no target-visible effect.
#
# The routes must be disjoint and they must cover every input. A checker's own
# "outside my model" disposition is not a fourth route: it has to land in ``H`` or ``D``, never in
# ``A``. Written as the partition obligation this module enforces::
#
#     A, H, D pairwise disjoint          every input lands in one of them
#     a decline is EXPLICIT              exit code and an empty list are not a decline
#     a decline has NO target effect     it may not also emit commands
#
# **An empty command list plus a zero exit code does not identify accepted work.** That sentence is the
# whole module. It is also the single most productive defect this repo has: measured 2026-09-07, 244
# command buffers under ``out/runs`` are empty AND carry no decline, so nothing in them distinguishes
# "this compiler declined, correctly" from "this compiler dropped the program on the floor and reported
# success". Three of those cases were reproduced from a spec in one command each — an
# ``rmsnorm``, a forbidden direct-scratchpad op, and a stateful random op — every one exiting 0 with an
# empty stderr, no decline, and the program's declared OUTPUT TENSOR silently absent from the buffer.
#
# The vocabulary already existed and was only ever read as an EXCUSE:
# ``commandbuffer.validate_command_buffer`` lets a ``declined`` buffer skip its
# operands-reference-a-tensor check. Nothing ever required the key. So a backend that emits nothing and
# says nothing has been contract-conformant this whole time, which is why no grader caught it.
#
# WHY THIS IS SEPARATE FROM ``validate_command_buffer``
# -----------------------------------------------------
# Because turning this on CHANGES WHAT THE GRADER ACCEPTS, and the grader is an instrument: comparing two
# runs whose validators disagree compares the instruments as well as the submissions. ``preflight``,
# ``oot_runner`` and ``capsule_common`` all call ``validate_command_buffer``, so wiring the partition into
# it retro-fails live and archived submissions at once.
#
# This module is therefore a PURE classifier with no callers in the grading path. The gate
# ``build_tools/checks/check_route_partition.py`` reports against a ratchet. Wiring it into
# ``validate_command_buffer`` is a deliberate, scheduled instrument change — not a side effect of
# landing the check.
#
# TARGET-AGNOSTIC
# ---------------
# Nothing here knows a target. The partition is a property of the command-buffer ABI, which every
# endpoint speaks — the backend contract requires ``emit_command_buffer`` of every package regardless of
# whether the endpoint is RoCC instructions, a SIMT kernel or a self-hosted ISA. A target that grew a
# fourth route would declare it in the ABI, and this module would then read it from there.

#: Target work was emitted. Route ``A`` of the partition (``H`` is not distinguishable from ``A`` by
#: the buffer alone — the seam that separates them is declared outside it — so both land here and the
#: host-lane axis is what tells them apart).
ROUTE_ACCEPT = "A"

#: An explicit decline carrying a reason, with no emitted command. Route ``D``.
ROUTE_DECLINE = "D"

#: Neither route is identified. This is NOT a route — it is the partition failing to close, and the
#: reason it gets its own name is that it used to be silently counted as ``D``.
ROUTE_VIOLATION = "!"

ROUTES = (ROUTE_ACCEPT, ROUTE_DECLINE, ROUTE_VIOLATION)

#: No commands and no decline: indistinguishable from a dropped program reported as success.
SILENT_NO_WORK = "silent_no_work"

#: A decline AND emitted commands. The two routes are meant to be disjoint, and a consumer reading
#: ``declined`` excuses the case while a consumer reading ``commands`` executes it — so the same
#: artifact grades as "declined, not a failure" and still runs. Measured 2026-09-07: 30 such buffers
#: under ``out/runs``, one of them declining ``RES_PACK`` while emitting
#: ``RES_PACK, MATMUL_RESIDENT, COMMIT, EVICT`` and declaring an output.
DECLINES_AND_EMITS = "declines_and_emits"

#: A decline with no reason, or one that is not a mapping. A decline has to be readable by whoever has
#: to act on it; "declined: true" tells a caller nothing about what to fix.
DECLINE_WITHOUT_REASON = "decline_without_reason"

VIOLATION_KINDS = (SILENT_NO_WORK, DECLINES_AND_EMITS, DECLINE_WITHOUT_REASON)


class RouteVerdict:
    """One buffer's route, plus every way it fails the partition.

    ``route`` is :data:`ROUTE_VIOLATION` exactly when ``violations`` is non-empty, so a caller cannot
    read a route of ``D`` off a buffer that only looked declined because it was empty.
    """

    __slots__ = ("route", "violations", "reason", "declined_op", "n_commands", "outputs")

    def __init__(
        self,
        route: str,
        violations: list[tuple[str, str]],
        *,
        reason: str | None,
        declined_op: str | None,
        n_commands: int,
        outputs: list[str],
    ):
        self.route = route
        self.violations = violations
        self.reason = reason
        self.declined_op = declined_op
        self.n_commands = n_commands
        self.outputs = outputs

    @property
    def ok(self) -> bool:
        """True when this buffer lands in exactly one route."""
        return not self.violations

    @property
    def kinds(self) -> list[str]:
        """The violation kinds only, for counting."""
        return [k for k, _ in self.violations]

    def __repr__(self) -> str:
        return f"RouteVerdict(route={self.route!r}, n_commands={self.n_commands}, violations={self.kinds!r})"


def _output_names(cb: dict[str, Any]) -> list[str]:
    """Names the buffer declares as outputs.

    ``tensors`` is a mapping in the ABI, but a buffer that declares none has been seen to spell it as
    an empty LIST, so this tolerates both rather than raising on the shape — the point of the module is
    to classify malformed output, not to reject it before classifying.
    """
    tensors = cb.get("tensors")
    if not isinstance(tensors, dict):
        return []
    out = []
    for name, spec in tensors.items():
        if isinstance(spec, dict) and str(spec.get("role")) == "output":
            out.append(str(name))
    return sorted(out)


def route_of(cb: dict[str, Any]) -> RouteVerdict:
    """Classify one command buffer into the admission partition.

    Pure: reads the buffer and nothing else — no target, no manifest, no filesystem. A buffer that
    satisfies the partition returns :data:`ROUTE_ACCEPT` or :data:`ROUTE_DECLINE` with an empty
    ``violations``; anything else returns :data:`ROUTE_VIOLATION` and says why, with enough context in
    the message to act on without reopening the file.
    """
    commands = cb.get("commands")
    commands = commands if isinstance(commands, list) else []
    declined = cb.get("declined")
    has_decline = declined is not None
    outputs = _output_names(cb)

    reason = None
    declined_op = None
    if isinstance(declined, dict):
        raw_reason = declined.get("reason")
        reason = str(raw_reason) if raw_reason else None
        raw_op = declined.get("op")
        declined_op = str(raw_op) if raw_op else None

    violations: list[tuple[str, str]] = []

    if has_decline and commands:
        violations.append(
            (
                DECLINES_AND_EMITS,
                (
                    f"the buffer declines ({'op=' + repr(declined_op) + ', ' if declined_op else ''}"
                    f"reason={reason!r}) and ALSO emits {len(commands)} command(s) "
                    f"{[str(c.get('opcode')) for c in commands if isinstance(c, dict)][:6]}"
                    f"{', declaring output(s) ' + repr(outputs) if outputs else ''}. Routes A and D are "
                    f"disjoint: a consumer reading 'declined' excuses this case while a consumer reading "
                    f"'commands' executes it, so the same artifact both grades as declined and runs"
                ),
            )
        )

    if has_decline and not reason:
        violations.append(
            (
                DECLINE_WITHOUT_REASON,
                (
                    f"the buffer declines but gives no readable reason (declined={declined!r}). A decline is "
                    f"an instruction to whoever has to act on it; it needs to say what was not lowered"
                ),
            )
        )

    if not commands and not has_decline:
        violations.append(
            (
                SILENT_NO_WORK,
                (
                    f"the buffer emits no command and declares no 'declined' block"
                    f"{', and declares output(s) ' + repr(outputs) + ' that nothing computes' if outputs else ''}"
                    f". An empty command list plus a zero exit code does not identify accepted work, so this "
                    f"is indistinguishable from a program dropped on the floor and reported as success"
                ),
            )
        )

    if violations:
        route = ROUTE_VIOLATION
    elif commands:
        route = ROUTE_ACCEPT
    else:
        route = ROUTE_DECLINE
    return RouteVerdict(
        route, violations, reason=reason, declined_op=declined_op, n_commands=len(commands), outputs=outputs
    )


__all__ = [
    "DECLINES_AND_EMITS",
    "DECLINE_WITHOUT_REASON",
    "ROUTES",
    "ROUTE_ACCEPT",
    "ROUTE_DECLINE",
    "ROUTE_VIOLATION",
    "RouteVerdict",
    "SILENT_NO_WORK",
    "VIOLATION_KINDS",
    "route_of",
]
