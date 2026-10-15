"""Judge execution lanes only from the independent evidence actually recorded."""
from __future__ import annotations

from . import trace_check as TCK
from .elf_lanes import ACCELERATOR_LANE as _ACCELERATOR_LANE
from .elf_lanes import EXECUTED_LANE_EVIDENCE, WHOLE_PROGRAM_COMPLETION_EVIDENCE


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


def lane_report(
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
    from . import boundary as _bd
    from .elf_lanes import native_host_execution

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
