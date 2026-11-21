"""Judge execution lanes only from the independent evidence actually recorded."""

from __future__ import annotations

from merlin.development.targetgen import trace_check as TCK
from merlin.development.targetgen.elf_lanes import ACCELERATOR_LANE as _ACCELERATOR_LANE
from merlin.development.targetgen.elf_lanes import EXECUTED_LANE_EVIDENCE, WHOLE_PROGRAM_COMPLETION_EVIDENCE
from merlin.development.targetgen import tier_cache as _TC
from merlin.development.targetgen.elf_lanes import EXECUTED_LANE_EVIDENCE
from merlin.evaluation.capsule.evidence.certificates import run_executables
from merlin.evaluation.capsule.operations.numerics import _engine_token as _engine_token
from merlin.evaluation.capsule.operations.numerics import _oracle_kind as _oracle_kind


# ---- lanes_evidence ----------------------------------------------------------------------------


def whole_program_completion_lane_report(
    capsule: dict,
    command_buffer: dict,
    *,
    global_plan_proof: dict,
    numeric: dict,
    tiers: dict,
    decoded_trace: dict | None,
) -> dict | None:
    """Credit compulsory mixed-program tasks only after the complete ELF actually passes.

    A compiler plan alone remains intent and earns nothing.  This evidence rung requires all three
    independent pieces: the host's source/ABI/CFG verifier proved every task occurs on every returning
    path; a mandatory oracle observed that complete kernel return with the exact semantic output; and
    the decoded artifact contains accelerator work before an ``on_mesh`` task is credited.  The task
    list is bound into ``global_plan_proof.plan_digest``, so this function never accepts an unverified
    free-standing lane claim.
    """
    lanes = capsule.get("lanes") or {}
    required = [str(x) for x in (lanes.get("require") or [])]
    forbidden = [str(x) for x in (lanes.get("forbid") or [])]
    if not required and not forbidden:
        return None
    if (
        (command_buffer.get("kernel_abi") or {}).get("kind") != "whole_program"
        or numeric.get("status") != "pass"
        or global_plan_proof.get("status") != "verified"
        or (global_plan_proof.get("control_flow") or {}).get("status") != "verified"
    ):
        return None

    def _tier_passed(value) -> bool:
        status = value.get("status") if isinstance(value, dict) else getattr(value, "status", None)
        mandatory = value.get("mandatory") if isinstance(value, dict) else getattr(value, "mandatory", False)
        return bool(mandatory and status == "pass")

    if not any(_tier_passed(value) for value in tiers.values()):
        return None
    tasks = ((command_buffer.get("params") or {}).get("global_program_plan") or {}).get("tasks") or []
    emitted = global_plan_proof.get("emitted_operations_by_task") or {}
    if not isinstance(tasks, list) or not tasks or any(not isinstance(row, dict) for row in tasks):
        return None
    completed: list[int] = []
    lane_counts: dict[str, int] = {}
    for row in tasks:
        ident, kind = row.get("task_index"), row.get("kind")
        count = emitted.get(ident, emitted.get(str(ident), 0))
        if (
            isinstance(ident, bool)
            or not isinstance(ident, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or kind not in ("contraction", "host")
        ):
            return None
        completed.append(ident)
        lane = _ACCELERATOR_LANE if kind == "contraction" else "scalar_rvv_lane"
        lane_counts[lane] = lane_counts.get(lane, 0) + 1

        # A claimed contraction task is not enough: the actual lowered artifact must carry accelerator
        # work. Absence is unmeasured because the decoder does not recognize every possible spelling.
    if lane_counts.get(_ACCELERATOR_LANE):
        try:
            if decoded_trace is None or not TCK.drives_accelerator(decoded_trace):
                lane_counts.pop(_ACCELERATOR_LANE, None)
        except Exception:  # noqa: BLE001 -- decoder uncertainty earns no credit
            lane_counts.pop(_ACCELERATOR_LANE, None)

    observed = sorted(lane for lane in {*required, *forbidden} if lane_counts.get(lane, 0) > 0)
    known_task_lanes = {_ACCELERATOR_LANE, "scalar_rvv_lane"}
    evidence = {
        lane: (
            WHOLE_PROGRAM_COMPLETION_EVIDENCE
            if lane in observed or (lane in forbidden and lane in known_task_lanes)
            else "unmeasured"
        )
        for lane in (*required, *forbidden)
    }
    report = {
        "required": required,
        "observed": observed,
        "unexercised": [lane for lane in required if lane not in observed],
        "evidence": evidence,
        "completed_tasks": completed,
        "task_lane_counts": dict(sorted(lane_counts.items())),
        "global_plan_proof": {
            key: global_plan_proof.get(key)
            for key in ("schema", "source_sha256", "plan_digest", "candidate_lowered_sha256")
            if global_plan_proof.get(key) is not None
        },
        "judged_by": WHOLE_PROGRAM_COMPLETION_EVIDENCE,
        "host_contractions_ran": None,
    }
    if forbidden:
        report["forbidden"] = forbidden
        report["violated"] = [lane for lane in forbidden if lane in observed]
        unmeasured = [lane for lane in forbidden if evidence[lane] == "unmeasured"]
        if unmeasured:
            report["unmeasured_forbidden"] = unmeasured
            # A caller-supplied report tells the finalizer that this path OWNS the verdict. Hand it over only
            # when this rung judged every declaration; a partial report would suppress the finalizer's
            # LANE_CONTRACT_NOT_EVALUATED refusal.
    if report["unexercised"] or report.get("unmeasured_forbidden"):
        return None
    return report


def _lane_evidence_report(
    capsule: dict,
    routing_plan: dict | None,
    mesh_execution: dict | None = None,
    host_execution: dict | None = None,
    lane_execution: dict | None = None,
    *,
    lane_counters: dict,
) -> dict | None:
    """Verify an INTEROP capsule's declared lanes against what the compiler actually DID.

    Returns ``None`` when the capsule declares neither ``lanes.require`` nor ``lanes.forbid``.

    THE EVIDENCE LADDER, strongest first. Each lane is judged on the best evidence available FOR THAT
    LANE, and the rung it was judged on is reported per lane -- a single label for the whole report had
    to lie about at least one of them.

    1. ``dispatch_ledger`` -- the ordered, runner-owned record of COMPLETED calls, each naming its lane.
       This is the only evidence that can prove a negative: if the ledger exists, a lane it does not
       name carried nothing, full stop. It is also the only rung with an open vocabulary, because the
       lane names come from the runtime's own entries rather than from anything written here.
    2. aggregate per-lane counters -- ``lane_execution`` (supplied by whoever ran the model, and able to
       name lanes this module has never heard of) then :data:`lane_counters` over the two legacy
       records. Real execution, but a total rather than a sequence.
    3. the routing plan -- INTENT ONLY. ``plan["on_mesh"]`` lists the ops the router ASSIGNED to the
       mesh; whether they ran there is a different fact. Measured on one submission: 15 matmuls assigned
       to the mesh, 15 host fallbacks at run time. A plan can satisfy a *required* lane only with the
       caveat recorded, and can never satisfy a *forbidden* one.

    Two shapes this must not fall into, both of which produced a hollow pass on the first capsule to use
    it:

    * **Not every key of the plan is a lane.** The plan also carries ``n_mesh_ops``, ``n_scalar_ops``,
      ``mesh_matmul_extents`` and ``note``; scanning it for truthy keys reported those as lanes that
      "carried work". A lane entry is a MAPPING of op -> count, so only dict-valued keys are lanes.
    * **A FORBIDDEN LANE IS JUDGED ONLY ON EXECUTION.** "The router planned no work here" is not "the
      compiler put no work here", and a negative assertion is exactly where plan-only evidence is most
      dangerous: absence of a plan entry is the DEFAULT state, so believing it would make the assertion
      vacuously true for free. A forbidden lane with no execution evidence is neither violated nor
      satisfied -- it is unmeasured, and reported as such.

    ``in_contract_vector_scalar`` is the lane this ladder is built for. The dispatch runtime has one
    device branch and executes those ops on the host ``.so``, so no rung above the plan can speak for it
    yet, and it stays plan-evidenced rather than being quietly promoted alongside the lanes that can be
    measured. When the runtime learns to name it in the ledger, it acquires evidence with no edit here.
    """
    _lanes_decl = capsule.get("lanes") or {}
    req = [str(x) for x in (_lanes_decl.get("require") or [])]
    forbid = [str(x) for x in (_lanes_decl.get("forbid") or [])]
    both = sorted(set(req) & set(forbid))
    if both:
        raise ValueError(
            f"capsule lanes {both} are both required and forbidden; one of the two assertions can never hold"
        )
    if not req and not forbid:
        return None
    plan = routing_plan or {}
    _planned = {k for k, v in plan.items() if isinstance(v, dict) and v}
    me = mesh_execution or {}
    he = host_execution or {}

    def _measured(value) -> int | None:
        """A real count, or None for 'nobody could tell'. ``UNKNOWN`` is a sentinel, not a number."""
        if value is None or isinstance(value, str):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

            # rung 1 -- completed calls, by lane

    ledger = me.get("dispatch_ledger")
    has_ledger = isinstance(ledger, list)
    ledger_counts: dict[str, int] = {}
    if has_ledger:
        for e in ledger:
            if isinstance(e, dict) and e.get("status") == "pass" and e.get("lane"):
                ln = str(e["lane"])
                ledger_counts[ln] = ledger_counts.get(ln, 0) + 1
                # The ledger speaks for the lanes it names AND for the lanes the runtime is known to log; reading a
                # silent zero for any OTHER lane would assert a negative the ledger never measured.
    ledger_scope = (set(ledger_counts) | set(lane_counters)) if has_ledger else set()

    # rung 2 -- aggregate counters, caller-supplied first
    _records = {"mesh": me, "host": he}
    agg = {lane: _measured(_records.get(rec, {}).get(key)) for lane, (rec, key) in lane_counters.items()}
    for lane, value in (lane_execution or {}).items():
        agg[str(lane)] = _measured(value)

        # A PLAN NEVER CREDITS A LANE. Only the executed rungs can put a lane in `carried`: an assignment is
        # not a completed call, and a lane the plan merely names must not be able to authorize itself. What
        # the plan still buys is the distinction between "measured, and nothing ran" and "nobody measured" --
        # both leave the lane unexercised, but only the second is reported in `plan_only_lanes`, which the
        # grader turns into `incomplete` rather than `fail`.
    evidence: dict[str, str] = {}
    carried: set[str] = set()
    for ln in sorted({*req, *forbid}):
        if ln in ledger_scope:
            evidence[ln], n = "dynamic_dispatch_ledger", ledger_counts.get(ln, 0)
        elif agg.get(ln) is not None:
            evidence[ln], n = "execution", int(agg[ln])
        else:
            evidence[ln], n = "routing_plan", 0
        if n > 0:
            carried.add(ln)

    unexercised = [ln for ln in req if ln not in carried]
    violated = [ln for ln in forbid if evidence.get(ln) in EXECUTED_LANE_EVIDENCE and ln in carried]
    unmeasured_forbid = [ln for ln in forbid if evidence.get(ln) not in EXECUTED_LANE_EVIDENCE]
    out = {
        "required": req,
        "observed": sorted(carried),
        "unexercised": unexercised,
        # The lanes this report is ABOUT. `carried` is only ever filled from the declared set above,
        # so `observed` is silent about a lane the capsule never named -- it is not a claim that no
        # such lane ran. The grader cross-checks this report against the dispatch ledger, which sees
        # every lane; without the scope stated here it compared a declared-only set to a ledger-wide
        # one and any capsule that dispatched on more lanes than it declared could never agree.
        "scope": sorted({*req, *forbid}),
        "evidence": {ln: evidence[ln] for ln in (*req, *forbid)},
        # Contractions apart from kernels: the host lane is mostly norms and activations, so a bare
        # kernel count is satisfied by any model at all. Reported rather than gated -- a capsule may
        # legitimately put only elementwise work on the host.
        "host_contractions_ran": _measured(he.get("contractions_ran")),
    }
    if forbid:
        out["forbidden"] = forbid
        out["violated"] = violated
        if unmeasured_forbid:
            out["unmeasured_forbidden"] = unmeasured_forbid
    plan_only = [ln for ln in req if evidence[ln] == "routing_plan"]
    if plan_only:
        out["caveat"] = (
            f"no execution accounting was recorded for {plan_only}, so those lanes report "
            f"what the router PLANNED, not what ran; a lane can appear here and still have "
            f"executed nothing"
        )
        out["plan_only_lanes"] = plan_only
    return out


def dispatch_boundary_report(mesh_execution: dict | None) -> dict:
    """Classify the *executed* host/accelerator sequence from the runtime ledger."""
    from merlin.development.targetgen import boundary as _bd
    from merlin.development.targetgen.elf_lanes import native_host_execution

    ledger = (mesh_execution or {}).get("dispatch_ledger")
    if not isinstance(ledger, list):
        return {
            "status": "missing",
            "boundary": _bd.UNKNOWN,
            "contains": [],
            "detail": "runner-owned dynamic dispatch ledger is missing",
        }
    sequence: list[str] = []
    unresolved = 0
    for entry in ledger:
        lane = entry.get("lane") if isinstance(entry, dict) else None
        if lane == "on_mesh":
            sequence.append(_bd.ACCEL)
        elif lane in ("scalar_rvv_lane", "host_fallback", "mesh_unavailable") or native_host_execution(entry):
            sequence.append(_bd.HOST)
        else:
            unresolved += 1
    segs = _bd.segments(sequence)
    return {
        "status": "pass" if sequence and not unresolved else "fail",
        "boundary": _bd.classify_sequence(sequence),
        "contains": sorted(_bd.patterns_in_sequence(sequence)),
        "n_dynamic_calls": len(ledger),
        "n_accel_calls": sum(x == _bd.ACCEL for x in sequence),
        "n_host_calls": sum(x == _bd.HOST for x in sequence),
        "n_unresolved": unresolved,
        "accel_segments": sum(k == _bd.ACCEL for k, _ in segs),
        "host_segments": sum(k == _bd.HOST for k, _ in segs),
        "detail": "derived only from completed calls in the runner-owned dispatch ledger",
    }


def oracle_source_scope(capsule, command_buffer, lowered_mlir, *, target=None):
    """Bind unpublished scratch to the original capsule source before any oracle call."""
    from contextlib import nullcontext
    from pathlib import Path

    from merlin.development.targetgen.contract.source_ownership import source_mutable_scope

    if (command_buffer.get("kernel_abi") or {}).get("kind") != "whole_program" or not any(
        spec.get("role") == "intermediate" for spec in command_buffer["tensors"].values()
    ):
        return nullcontext()
    source_name = capsule.get("linalg_mlir") or capsule.get("interface_mlir")
    if not source_name or not capsule.get("__dir__"):
        raise ValueError("whole-program scratch requires the original capsule source")
    source_path = Path(capsule["__dir__"]) / source_name
    lowered_abi = None
    if target is not None:
        from merlin.development.targetgen.contract.harness_abi import lowered_plan_abi
        from merlin.development.targetgen.target_experiment import load_capability_manifest

        contract = load_capability_manifest(target).contract
        lowered_abi = lowered_plan_abi(contract, target=target)
        # The plan names the bytes the compiler was given. For an enriched public
        # source those are re-derived from the selected original here, so the plan
        # is compared like with like and still bound to the selected source.
    from merlin.development.targetgen.public_source_abi import compiler_source

    source_bytes = compiler_source(capsule, source_path).compiler_bytes
    # The live original-snapshot owner selects the compiler transport. Recreate
    # its bytes from the trusted capsule, never from submitted plan metadata or
    # a candidate-controlled path. Shared/historical transports remain exact.
    from merlin.development.backend.feedback.snapshot_execution import active_snapshot

    executor = active_snapshot()
    if executor is not None:
        disclosure = (executor.owner.bundle.get("author_profile") or {}).get("disclosure") or {}
        if disclosure.get("input_format") == "neutral_math_v1":
            from merlin.development.backend.neutral_program_inputs import encode_source

            source_bytes = encode_source(source_bytes)

    return source_mutable_scope(source_bytes, command_buffer, lowered_mlir=lowered_mlir, lowered_abi=lowered_abi)


# ---- lanes -------------------------------------------------------------------------------------
# Capsule evaluation: lanes.

_LANE_COUNTERS: dict[str, tuple[str, str]] = {
    "on_mesh": ("mesh", "matmul_layers_on_mesh"),
    "scalar_rvv_lane": ("host", "kernels_ran"),
}


def _sim_telemetry(oracle_meta, ev_name: str) -> dict:
    """The simulator-identity fields for a TierResult, from what the adapter reported.

    Kept as ONE extractor because the three tier paths that build a TierResult must not disagree about
    what a record says the engine was — and because a field only some paths populate is worse than one
    none of them do: a consumer reads the absence as "verilator", never as "unrecorded".
    """
    kind = _oracle_kind(oracle_meta) or ev_name
    declared = oracle_meta.get("engine") if isinstance(oracle_meta, dict) else None
    # The adapter's OWN word first. Deriving the engine from the kind string works and is the fallback,
    # but an adapter that states its engine outright is the authority on which one it ran -- the same
    # rule `derived_from_rtl` already follows, for the same reason.
    out: dict = {"engine": str(declared) if declared else (_engine_token(kind) or kind)}
    if isinstance(oracle_meta, dict):
        if isinstance(oracle_meta.get("selection"), dict):
            out["engine_selection"] = dict(oracle_meta["selection"])
        if isinstance(oracle_meta.get("provenance"), dict):
            out["sim_provenance"] = dict(oracle_meta["provenance"])
    return out


def lane_report(
    capsule: dict,
    routing_plan: dict | None,
    mesh_execution: dict | None = None,
    host_execution: dict | None = None,
    lane_execution: dict | None = None,
) -> dict | None:
    """Retain the original counter-map override at the runner boundary."""
    return _lane_evidence_report(
        capsule,
        routing_plan,
        mesh_execution,
        host_execution,
        lane_execution,
        lane_counters=_LANE_COUNTERS,
    )


def forbids_accelerator(capsule: dict) -> bool:
    """Does this capsule declare that work must NOT reach the accelerator?"""
    return _ACCELERATOR_LANE in [str(x) for x in ((capsule.get("lanes") or {}).get("forbid") or ())]


def accelerator_lane_violated(capsule: dict, trace) -> bool:
    """True only when a forbidding capsule's emitted stream DECODES accelerator instructions.

    The asymmetry here is the whole reason this is sound on a path that owns no execution record. A
    decoded accelerator instruction proves work went there. Its ABSENCE proves nothing: the decoder
    recognizes the ``.insn r`` form, so a ``.word``-encoded kernel that does drive the device decodes as
    silent, and reading that silence as "the host carried it" would hand a forbidding capsule a free
    pass. Violated, or unmeasured -- never satisfied.
    """
    from merlin.development.targetgen import trace_check as _TCK

    if not forbids_accelerator(capsule):
        return False
    try:
        return bool(_TCK.drives_accelerator(trace))
    except Exception:  # noqa: BLE001 -- an undecodable trace measures nothing
        return False


def _tier_field(record, key, default=None):
    """One field of a tier record, whichever form it is in -- a :class:`TierResult` or its ``to_dict``."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def carried_verdict_is_attributed(record, execution_identity) -> bool:
    """Does this CARRIED tier verdict belong to the bytes being graded right now?

    A carried verdict is not a guess and not stale by construction. :mod:`merlin.development.targetgen.tier_cache`
    keys a certificate on :func:`tier_cache.execution_identity` -- a content digest of the executables
    this run produced, the target, and the toolchain pins -- and ``lookup`` re-verifies the stored
    record's own copy of that key before returning it, so a hit is a statement about exactly these
    bytes on exactly this instrument. ``tier_promote.record_cert`` resolves against the same identity
    and never re-hashes, so a certificate is never re-attributed to bytes edited since.

    That is why this asks the question rather than refusing outright: per-capsule promotion is the
    designed behaviour in continuous mode, so most graded capsules CARRY their expensive tier, and
    refusing every carried verdict would make a lane contract flap on cache state rather than on
    compiler quality -- the same class of defect the lane evidence exists to remove.

    It is asked INDEPENDENTLY, though: ``execution_identity`` is recomputed here from the bytes on disk
    (see :func:`graded_execution_identity`) and compared to the identity the carried block records. A
    mismatch, a malformed digest, or no identity at all is refused -- fail closed, because an
    unattributable verdict is exactly the stale-certificate case the strict rule was protecting against.
    """
    from merlin.development.targetgen.oracle_schedule import valid_execution_digest

    carried = _tier_field(record, "carried")
    if not isinstance(carried, dict):
        return False
    earned = carried.get("execution_identity")
    if not valid_execution_digest(earned) or not valid_execution_digest(execution_identity):
        return False
    return str(earned) == str(execution_identity)


def graded_execution_identity(*, target, generated, shas) -> str | None:
    """The content identity of what this run would execute, or ``None`` when it cannot be formed.

    The same function the certificate cache keys on, called on the bytes present NOW, so comparing it
    with a carried record's identity is an independent check rather than a restatement of the cache's
    own claim. ``None`` on any refusal (no readable executable, an unidentifiable device), which callers
    must treat as "not attributable".
    """
    try:
        return _TC.execution_identity(target=target, executables=run_executables(generated), toolchain_shas=shas)
    except Exception:  # noqa: BLE001 -- an identity we cannot form credits nothing
        return None


def executed_program_tiers(tiers: dict, execution_tiers, *, execution_identity=None) -> list[str]:
    """The oracle tiers that certify THESE bytes on an instrument, and passed.

    ``execution_tiers`` is the runner's own set of tiers that execute the program on an instrument
    (the target's ``tier_sim`` ladder plus any injected adapter) -- the L0/L1 command-buffer
    interpretation is never in it, because interpreting a buffer we built is not running the program.
    A tier is counted only when its record says it passed and that verdict is about this program:

    * ``not_applicable`` / ``budget_deferred`` -- the tier never ran; the first is honestly N/A and the
      second was deliberately not bought. Either way nothing executed, for these bytes or any others.
    * ``carried`` -- executed in an EARLIER run. Counted only when
      :func:`carried_verdict_is_attributed` matches the certificate's identity against
      ``execution_identity``, the identity recomputed from the bytes under grade; refused otherwise, and
      refused outright when the caller supplies no identity to compare against.

    Accepts either a :class:`TierResult` or its ``to_dict`` form, because the model path stores the
    second and the operator path the first.
    """
    out: list[str] = []
    for tier in sorted(str(t) for t in (execution_tiers or ())):
        record = (tiers or {}).get(tier)
        if record is None or _tier_field(record, "status") != "pass":
            continue
        if _tier_field(record, "not_applicable", False) or _tier_field(record, "budget_deferred", False):
            continue
        if _tier_field(record, "carried") and not carried_verdict_is_attributed(record, execution_identity):
            continue
        out.append(tier)
    return out


def operator_path_lane_execution(
    capsule: dict,
    *,
    tiers: dict,
    execution_tiers,
    decoded_trace: dict | None,
    execution_identity=None,
) -> dict | None:
    """Per-lane execution counts the OPERATOR path can honestly hand :func:`lane_report`, or ``None``.

    The op tier ladder owns no routing plan and no dispatch ledger, which is why a capsule requiring a
    lane went ungraded there (``LANE_CONTRACT_NOT_EVALUATED``). It does own the two halves a positive
    claim actually needs, and this function credits the accelerator lane only when it holds BOTH:

    * **Something ran.** An oracle tier executed this capsule's emitted program on an instrument and
      passed -- see :func:`executed_program_tiers`. A skipped, deferred or unavailable tier is not an
      execution, and neither is our own L0/L1 interpretation of the command buffer. A verdict CARRIED
      from an earlier run counts only when its certificate is attributed to the same
      ``execution_identity`` as the bytes under grade.
    * **What ran drove the accelerator.** The decoded instruction stream carries instructions of THIS
      target's accelerator classes, counted by :func:`merlin.development.targetgen.trace_check.accelerator_instruction_count`
      -- membership is the target's own RTL decode table claiming the opcode, never a list written here.

    Neither half alone says anything. A passing oracle tier with no accelerator instruction is a program
    the host computed; accelerator instructions with no oracle tier are a stream nobody executed. Only
    together do they say that the accelerator lane carried work, which is exactly what the required-lane
    assertion claims. With either half missing this returns ``None`` and the capsule stays unmeasured --
    ``lane_report`` is never handed a fabricated count, and a routing plan is never consulted.
    """
    from merlin.development.targetgen import trace_check as _TCK

    lanes = capsule.get("lanes") or {}
    if _ACCELERATOR_LANE not in [str(x) for x in (lanes.get("require") or ())]:
        return None
    if not executed_program_tiers(tiers, execution_tiers, execution_identity=execution_identity):
        return None
    if not isinstance(decoded_trace, dict):
        return None
    try:
        n = _TCK.accelerator_instruction_count(decoded_trace)
    except Exception:  # noqa: BLE001 -- an undecodable trace measures nothing, and credits nothing
        return None
    return {_ACCELERATOR_LANE: n} if n > 0 else None


def credit_executed_required_lanes(report: dict | None, capsule: dict, lane_execution: dict | None) -> dict | None:
    """Fold executed-lane evidence into ``report`` for the REQUIRED lanes it settles, leaving the rest alone.

    ``report`` is whatever the negative half already produced -- a linked-ELF or declared-program scan,
    which can refute a ``forbid`` and can never credit a ``require``. That half is copied through
    untouched: its ``violated``/``unmeasured_forbidden``/scan blocks are the artifact-level evidence and
    nothing here may weaken them. What is added is the positive half, and only for lanes that
    :func:`lane_report` itself judged on an executed rung, so the evidence vocabulary has one owner.
    """
    if not lane_execution:
        return report
    positive = lane_report(capsule, None, lane_execution=lane_execution)
    if positive is None:
        return report
    credited = [
        ln
        for ln in positive["required"]
        if ln in positive["observed"] and positive["evidence"].get(ln) in EXECUTED_LANE_EVIDENCE
    ]
    if not credited:
        return report
        # Stated beside the report because `judged_by` is a single label for the whole report and this
        # report now has two halves judged on different rungs -- the per-lane `evidence` mapping is the
        # authority, and this says which lanes the new positive rung answered for, and on how much work.
    provenance = {
        "lane_counts": dict(lane_execution),
        "credited_lanes": sorted(credited),
        "source": "executed oracle tier + decoded instruction trace",
    }
    if report is None:
        # Nothing measured the negative half (no scannable artifact). The positive report stands on its
        # own; any forbidden lane in it is still reported at its own rung and stays unjudged.
        out = dict(positive)
        out["execution_evidence"] = provenance
        return out
    out = dict(report)
    out["evidence"] = {**(report.get("evidence") or {}), **{ln: positive["evidence"][ln] for ln in credited}}
    out["observed"] = sorted(set(report.get("observed") or []) | set(credited))
    out["unexercised"] = [ln for ln in (report.get("unexercised") or []) if ln not in credited]
    still_unmeasured = [ln for ln in (report.get("unmeasured_required") or []) if ln not in credited]
    if report.get("unmeasured_required") is not None:
        if still_unmeasured:
            out["unmeasured_required"] = still_unmeasured
        else:
            out.pop("unmeasured_required", None)
            out.pop("caveat", None)
    out["execution_evidence"] = provenance
    return out
