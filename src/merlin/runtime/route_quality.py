"""The QUALITY of a route claim, checked on a produced command buffer.

WHAT THIS ADDS TO ``route_partition``
-------------------------------------
:mod:`merlin.runtime.route_partition` closes the partition: every region lands in ``A`` (accepted),
``H`` (an explicit host or composed route) or ``D`` (an explicit decline), and a buffer that lands
nowhere is a violation. That is a check on the SHAPE of the claim. Nothing checked the claim itself,
and the two ways a conformant buffer can still be wrong are exactly the two this module names.

**Tier 1 — the declaration gap.** A region routed ``H`` states a reason. Nothing required the reason
to be TRUE. Measured on a deployable ResNet-50 package (``lane_placement``, 173 rows): 119 regions
are placed on a scalar host lane, each carrying a reason of the form *"family 'quantize' has no
datapath on a 16x16 i8 systolic mesh"*. With the target's facts extracted, 116 of those 119 refusals
are contradicted by the target's OWN derived capability — the derived epilogue capability licenses
``relu``, the declared readout census applies ``acc_scale`` and ``maxpool``, and the facet's
``operand_sum`` licenses the integer sum of two separately scaled tensors that a residual add is.
Three come back ``UNKNOWN``. A declaration gap was shipped as a hardware limit, and every downstream
reader (a planner, a cost model, a person deciding whether the accelerator earns its area) believed
it.

The oracle is never a literal here. It is the readout facet derived from the target's own RTL facts
and backend hooks (:mod:`merlin.targetgen.readout_facet`), and which derived field adjudicates which
op family is DATA, declared in ``mlir_oot_backend_contract.yaml`` under
``admission.route_quality.host_refusal_evidence``. A family the contract does not name, or a field
the facet could not derive, comes back ``UNKNOWN`` with the reason — never a pass, and never a
confirmation. A refusal is CONFIRMED only when the deciding rung produced a positive derivation that
excludes the work; "we could not tell" and "the target refuses" are different verdicts and this
module keeps them apart.

**Tier 2 — host compute inside an accepted region.** Tier 1 only catches a package that SAYS it
moved work to the host. A package can route a region ``A``, emit target commands for it, and still
run arithmetic on the host inside the same region while saying nothing at all. Tier 1 cannot see
that: the buffer's placement record is clean.

This tier is checkable WITHOUT knowing the target's dialect, because the host ISA is always known —
a RISC-V scalar core plus declared standard extensions — and the host IR is in the package. The
discriminator is **data dependence, not opcode family**. Address arithmetic is
``integer_arithmetic`` too: a rule that counted arithmetic by family would veto every legal
descriptor setup, and an unsatisfiable rule is one everybody routes around. So an arithmetic
operation is a finding only when its operand chain reaches a LOAD FROM A TENSOR BUFFER — a
``kernel_abi`` argument, a module-level buffer, or a stack slot some tensor-reaching value was
stored into. Arithmetic that reaches only induction variables, constants and pointer roots is
addressing, and addressing is fine.

Work found in an ``H`` region is COVERAGE, not a defect: an ``H`` region is where host work is
supposed to be, and counting it is how a reader sees that the instrument is looking at the right
program. Absent or unparseable host IR is ``incomplete`` with a stated cause, never ``ok``.

PHASED, BOTH TIERS
------------------
Each tier is a gate with a phase declared in ``merlin/contract/gate_phases.yaml`` and read through
:func:`merlin.perf.gate_phase.configured_phase`. ``incomplete`` is a STATUS, orthogonal to phase,
and never a pass at either.

TARGET-AGNOSTIC
---------------
No target name, opcode, lane spelling or stage vocabulary is written here. Which regions went to a
host lane comes from the buffer's own ``params.host_lane_regions`` / ``params.mesh_regions``; the
accepted/host split of a task comes from the program plan's own ``kind``; epilogue stage names are
the command-buffer ABI's; the capability comes from the target's derived facet; and the
family-to-evidence map is contract data.

**Measured on that same package**, tier 2 reports 20 of 54 accepted tasks running host arithmetic on
tensor data — 4,936 operations — beside 37,865 operations of addressing in the same tasks that it
correctly clears. That ratio is the point: a family-count rule would have vetoed all 42,801.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.perf import gate_phase

__all__ = [
    "SCHEMA",
    "GATE_DECLARATION",
    "GATE_HOST_COMPUTE",
    "STATUS_OK",
    "STATUS_REPORTED",
    "STATUS_INCOMPLETE",
    "VERDICT_CONTRADICTED",
    "VERDICT_CONFIRMED",
    "VERDICT_UNKNOWN",
    "RefusalVerdict",
    "DeclarationReport",
    "TaskCompute",
    "HostComputeReport",
    "host_refusal_evidence",
    "declaration_quality",
    "declaration_quality_for_target",
    "host_compute",
    "route_by_task",
    "format_declaration_report",
    "format_host_compute_report",
]

SCHEMA = "route_quality_v1"

#: The two gates, as they are spelled in ``merlin/contract/gate_phases.yaml``.
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
#: finding. Categories come from :mod:`merlin.perf.host_cfg_activity`, which owns the family split.
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
    from merlin.targetgen import readout_facet as rf

    stages: set[str] = set()
    gaps: list[str] = []
    for facet in facets:
        try:
            capability = rf.epilogue_capability(facet)
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

    ``facets`` are the target's :class:`~merlin.targetgen.readout_facet.ReadoutFacet` objects. They
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
    from merlin.targetgen import readout_facet as rf

    try:
        facets = rf.for_target(target)
    except Exception as exc:  # noqa: BLE001 -- an underivable target is incomplete, never ok
        return DeclarationReport(
            GATE_DECLARATION,
            STATUS_INCOMPLETE,
            (),
            (f"no readout facet could be derived for target {target!r}: {type(exc).__name__}: {exc}",),
        )
    return declaration_quality(command_buffer, facets=facets)


# --------------------------------------------------------------------------------------------
# Tier 2 — host compute inside an accepted region
# --------------------------------------------------------------------------------------------


@dataclass
class TaskCompute:
    """One program task's host arithmetic, split by what the arithmetic reaches."""

    task: str
    route: str
    #: Arithmetic whose operand chain reaches a load from a tensor buffer. The defect, in an ``A``.
    on_tensor: int = 0
    #: Arithmetic that reaches only induction variables, constants and pointer roots. Addressing.
    on_addressing: int = 0
    #: Arithmetic whose chain reaches something this walk cannot classify (a call, inline assembly).
    undecided: int = 0
    #: Operation indices of the first few tensor-reaching findings, so a reader can open the IR.
    witnesses: list[int] = field(default_factory=list)
    #: What made a chain undecided, deduplicated.
    undecided_causes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "route": self.route,
            "arithmetic_on_tensor_data": self.on_tensor,
            "arithmetic_on_addressing": self.on_addressing,
            "arithmetic_undecided": self.undecided,
            "witness_operation_indices": self.witnesses[:8],
            "undecided_causes": sorted(set(self.undecided_causes))[:8],
        }


@dataclass(frozen=True)
class HostComputeReport:
    gate: str
    status: str
    tasks: tuple[TaskCompute, ...]
    causes: tuple[str, ...] = ()

    @property
    def findings(self) -> tuple[TaskCompute, ...]:
        """Accepted tasks that compute on tensor data on the host. The defect."""
        return tuple(t for t in self.tasks if t.route == "A" and t.on_tensor)

    @property
    def coverage(self) -> int:
        """Host arithmetic on tensor data in ``H`` tasks. Where host work is SUPPOSED to be."""
        return sum(t.on_tensor for t in self.tasks if t.route == "H")

    def blocks(self, phase: str | None = None) -> bool:
        resolved = gate_phase.configured_phase(self.gate) if phase is None else phase
        return gate_phase.blocks(resolved, self.status, failing=(STATUS_REPORTED,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "gate": self.gate,
            "status": self.status,
            "causes": list(self.causes),
            "host_coverage_arithmetic_on_tensor_data": self.coverage,
            "accepted_tasks_computing_on_tensor_data": len(self.findings),
            "tasks": [t.to_dict() for t in self.tasks],
        }


def route_by_task(command_buffer: Mapping[str, Any]) -> tuple[dict[str, str], tuple[str, ...]]:
    """``({task id: "A"|"H"}, causes)`` from the program plan's own task kinds.

    A task the plan calls a host task is route ``H``; every other kind names device work the plan
    committed to emitting, which is route ``A``. The split is the PLAN's, not a lane vocabulary
    invented here, so a target whose device kinds are spelled differently needs no edit.
    """
    params = command_buffer.get("params")
    params = params if isinstance(params, Mapping) else {}
    plan = params.get("global_program_plan")
    if not isinstance(plan, Mapping):
        return {}, ("params.global_program_plan is absent, so no host IR operation can be attributed to a route",)
    tasks = plan.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        return {}, ("the program plan declares no task list, so no host IR operation can be attributed to a route",)
    out: dict[str, str] = {}
    causes: list[str] = []
    for entry in tasks:
        if not isinstance(entry, Mapping):
            causes.append("a program-plan task entry is not a mapping and was not attributed")
            continue
        index = entry.get("task_index")
        kind = entry.get("kind")
        if index is None or kind is None:
            causes.append(f"a program-plan task declares no {'task_index' if index is None else 'kind'}")
            continue
        out[str(index)] = "H" if str(kind) == _HOST_TASK_KIND else "A"
    return out, tuple(causes)


def _tensor_arguments(command_buffer: Mapping[str, Any]) -> frozenset[int]:
    """Entry-block argument positions that carry a tensor buffer, from the kernel ABI itself."""
    abi = command_buffer.get("kernel_abi")
    args = abi.get("args") if isinstance(abi, Mapping) else None
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        return frozenset()
    return frozenset(i for i, a in enumerate(args) if isinstance(a, Mapping) and a.get("tensor"))


def _pointer_root(value: Any, entry: Any, op_index: Mapping[Any, int]) -> str:
    """The SSA root of a pointer: an entry argument, a stack slot, a module buffer, or UNKNOWN.

    A SECOND pointer-root walk, deliberately. :mod:`merlin.perf.host_cfg_activity` has one, but it
    is a closure and it reports a module-level buffer as ``UNKNOWN``; teaching it to resolve one
    would change the numbers that instrument has already published. Widening an instrument is a
    scheduled change, not a side effect of landing a check, so this walk is local and says so.
    """
    from xdsl.ir import Block

    seen: set[Any] = set()
    while value not in seen:
        seen.add(value)
        owner = getattr(value, "owner", None)
        if isinstance(owner, Block):
            return f"arg:{value.index}" if owner is entry else "UNKNOWN"
        if owner is None:
            return "UNKNOWN"
        name = _real_op_name(owner)
        if name == "llvm.alloca":
            return f"alloca:{op_index.get(owner, -1)}"
        if name in {"llvm.mlir.addressof", "llvm.mlir.addrspacecast"}:
            symbol = owner.properties.get("global_name") or owner.attributes.get("global_name")
            return f"global:{getattr(symbol, 'root_reference', symbol)}"
        if name in {"llvm.getelementptr", "llvm.bitcast"} and owner.operands:
            value = owner.operands[0]
            continue
        return "UNKNOWN"
    return "UNKNOWN"


def _real_op_name(operation: Any) -> str:
    from merlin.perf.host_cfg_activity import _real_name

    return _real_name(operation)


def _category_of(operation: Any) -> str:
    from merlin.perf.host_cfg_activity import _category

    return _category(_real_op_name(operation))


def host_compute(
    command_buffer: Mapping[str, Any],
    *,
    function: Any | None = None,
    absent_cause: str | None = None,
) -> HostComputeReport:
    """Find arithmetic on tensor data inside tasks the program routed ``A``.

    ``function`` is the parsed ``llvm.func`` of the package's host IR. Absent, the report is
    ``incomplete`` carrying ``absent_cause`` — never ``ok``, because a package with no host IR is
    one nobody looked inside.
    """
    routes, causes = route_by_task(command_buffer)
    causes = list(causes)
    if function is None:
        causes.append(absent_cause or "the package carries no parsed host IR, so nothing was looked inside")
        return HostComputeReport(GATE_HOST_COMPUTE, STATUS_INCOMPLETE, (), tuple(causes))
    if not routes:
        return HostComputeReport(GATE_HOST_COMPUTE, STATUS_INCOMPLETE, (), tuple(causes))

    from merlin.perf.host_cfg_index import prepare_host_cfg

    cfg = prepare_host_cfg(function)
    if not cfg.blocks:
        causes.append("the host IR function has no block, so it carries no operation to classify")
        return HostComputeReport(GATE_HOST_COMPUTE, STATUS_INCOMPLETE, (), tuple(causes))
    entry = cfg.blocks[0]
    operations = list(function.walk())
    op_index = {op: i for i, op in enumerate(operations)}
    tensor_args = _tensor_arguments(command_buffer)
    if not tensor_args:
        causes.append(
            "the buffer's kernel_abi names no tensor argument, so a load's pointer root cannot be "
            "told from a tensor buffer"
        )

    # Values stored into each stack slot, so a tensor value that round-trips the stack is still
    # tensor data when it comes back. An over-approximation in the SAFE direction: it can only
    # widen what counts as tensor-reaching, never narrow it.
    stored_into: dict[str, list[Any]] = {}
    for op in operations:
        if _category_of(op) == "store" and len(op.operands) >= 2:
            stored_into.setdefault(_pointer_root(op.operands[1], entry, op_index), []).append(op.operands[0])

    # A load's verdict memoized by load operation, then a value-level memo for the chain walk.
    value_verdict: dict[Any, tuple[bool, str | None]] = {}

    def reaches_tensor(value: Any, seen: frozenset) -> tuple[bool, str | None]:
        """``(reaches a tensor load, cause when undecided)`` for one SSA value."""
        from xdsl.ir import Block

        if value in value_verdict:
            return value_verdict[value]
        if value in seen:
            return False, None  # an SSA cycle proves nothing; the other incoming edges decide
        owner = getattr(value, "owner", None)
        if isinstance(owner, Block) or owner is None:
            # A block argument: an entry argument is a pointer root or a scalar parameter, and a
            # loop argument is an induction variable. Neither is tensor DATA.
            return False, None
        name = _real_op_name(owner)
        category = _category_of(owner)
        if category == "load":
            root = _pointer_root(owner.operands[0], entry, op_index) if owner.operands else "UNKNOWN"
            if root.startswith("arg:"):
                # A kernel argument the ABI calls a tensor is tensor data; any other entry argument
                # is a scalar parameter or a pointer root, and reading through it is addressing.
                position = int(root.partition(":")[2])
                value_verdict[value] = (position in tensor_args, None)
                return value_verdict[value]
            if root.startswith("global:"):
                value_verdict[value] = (True, None)
                return True, None
            if root.startswith("alloca:"):
                out: tuple[bool, str | None] = (False, None)
                for stored in stored_into.get(root, ()):
                    hit, cause = reaches_tensor(stored, seen | {value})
                    if hit:
                        out = (True, None)
                        break
                    if cause is not None:
                        out = (False, cause)
                value_verdict[value] = out
                return out
            value_verdict[value] = (False, f"a load's pointer root is not derivable ({name})")
            return value_verdict[value]
        if category in {"opaque_inline_asm", "other"} or name.endswith("llvm.call"):
            return False, f"an operand chain reaches {name!r}, whose result this walk cannot classify"
        if category == "constant":
            return False, None
        worst: str | None = None
        for operand in owner.operands:
            hit, cause = reaches_tensor(operand, seen | {value})
            if hit:
                value_verdict[value] = (True, None)
                return True, None
            if cause is not None:
                worst = cause
        value_verdict[value] = (False, worst)
        return False, worst

    rows: dict[str, TaskCompute] = {}
    unattributed = 0
    for op in operations:
        if _category_of(op) not in _COMPUTE_CATEGORIES:
            continue
        owner = op.attributes.get("merlin.global_task")
        task = getattr(getattr(owner, "value", None), "data", None)
        if task is None:
            unattributed += 1
            continue
        key = str(task)
        route = routes.get(key)
        if route is None:
            unattributed += 1
            continue
        row = rows.setdefault(key, TaskCompute(task=key, route=route))
        hit, cause = reaches_tensor(op.results[0], frozenset()) if op.results else (False, None)
        if hit:
            row.on_tensor += 1
            if len(row.witnesses) < 8:
                row.witnesses.append(op_index[op])
        elif cause is not None:
            row.undecided += 1
            row.undecided_causes.append(cause)
        else:
            row.on_addressing += 1

    if unattributed:
        causes.append(
            f"{unattributed} host arithmetic operation(s) carry no program-plan task attribution, so "
            f"their route is unknown and they were neither credited nor reported"
        )
    ordered = tuple(sorted(rows.values(), key=lambda r: (r.route, int(r.task))))
    findings = [r for r in ordered if r.route == "A" and r.on_tensor]
    undecided_in_accepted = any(r.route == "A" and r.undecided for r in ordered)
    if findings:
        status = STATUS_REPORTED
    elif undecided_in_accepted or causes:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_OK
    return HostComputeReport(GATE_HOST_COMPUTE, status, ordered, tuple(causes))


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


def format_host_compute_report(report: HostComputeReport, label: str = "") -> str:
    out = [f"== route host compute {label}".rstrip(), f"status: {report.status}"]
    out.append(f"accepted tasks computing on tensor data: {len(report.findings)}")
    out.append(f"host-route coverage (arithmetic on tensor data in H): {report.coverage}")
    for row in report.findings[:8]:
        out.append(
            f"  task {row.task} (A): {row.on_tensor} arithmetic op(s) on tensor data, "
            f"{row.on_addressing} on addressing; witnesses {row.witnesses[:4]}"
        )
    for cause in report.causes:
        out.append(f"  incomplete: {cause}")
    return "\n".join(out)
